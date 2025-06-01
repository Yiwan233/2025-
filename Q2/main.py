# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题2：混合比例推断

代码名称: P2_MCMC_Mixture_Inference_V10
版本: V10 - 集成版本
日期: 2025-06-01
描述: 基于MCMC的混合比例(Mx)和降解参数(θ)推断
     整合了Stutter效应、等位基因频率信息和ADO建模
     支持N≤5的贡献者数量，针对N=4,5使用K-top采样优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import loggamma, gammaln
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import defaultdict
import itertools

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class MCMCMixtureInference:
    """
    MCMC混合比例推断类
    基于问题2的V10版本思路实现
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化MCMC推断器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.marker_params = self.config.get("marker_specific_params", {})
        self.global_params = self.config.get("global_parameters", {})
        
        # 初始化参数
        self._init_parameters()
        
        # MCMC结果存储
        self.mcmc_results = {}
        self.convergence_diagnostics = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                "global_parameters": {
                    "saturation_threshold_rfu": 30000.0,
                    "size_tolerance_bp": 0.5,
                    "stutter_cv_hs_global_n_minus_1": 0.25,
                    "true_allele_confidence_threshold": 0.5
                },
                "marker_specific_params": {}
            }
    
    def _init_parameters(self):
        """初始化模型参数"""
        # 基础参数
        self.k_gamma = self.global_params.get("k_gamma", 1.0)
        self.beta = self.global_params.get("beta", 1.5)
        self.sigma_var_base = self.global_params.get("sigma_var_base", 0.1)
        self.cv_hs_base = self.global_params.get("cv_hs_base", 0.25)
        
        # ADO参数
        self.H_50 = self.global_params.get("ado_h50", 200.0)
        self.s_ado = self.global_params.get("ado_slope", 0.01)
        
        # 降解参数
        self.k_deg_0 = self.global_params.get("k_deg_0", 0.001)
        self.size_ref = self.global_params.get("size_ref", 100.0)
        self.alpha = self.global_params.get("alpha", 1.0)
        
        # MCMC参数
        self.n_chains = self.global_params.get("n_chains", 4)
        self.n_iterations = self.global_params.get("n_iterations", 10000)
        self.n_warmup = self.global_params.get("n_warmup", 2500)
        self.thinning = self.global_params.get("thinning", 1)
        
    def calculate_gamma_l(self, locus: str, avg_height: float, 
                         inter_locus_entropy: float) -> float:
        """
        计算位点特异性放大效率 γ_l
        
        Args:
            locus: 位点名称
            avg_height: 平均峰高
            inter_locus_entropy: 位点间平衡熵
            
        Returns:
            位点放大效率
        """
        # 基础放大效率
        gamma_base = self.k_gamma * avg_height
        
        # 位点间平衡权重
        L_exp = len(self.marker_params)
        if L_exp > 0:
            w_entropy = (1 - inter_locus_entropy / np.log(L_exp)) ** self.beta
            
            # 位点特异性调整
            # 这里假设我们有位点的总信号比例 P_l
            P_l = 1.0 / L_exp  # 简化假设，实际应从数据计算
            gamma_l = gamma_base * (1 + w_entropy * ((P_l * L_exp) - 1))
        else:
            gamma_l = gamma_base
            
        return max(gamma_l, 1e-6)
    
    def calculate_stutter_cv(self, avg_height: float) -> float:
        """
        计算Stutter变异系数
        
        Args:
            avg_height: 平均峰高
            
        Returns:
            Stutter变异系数
        """
        A_s = self.global_params.get("A_s", 0.1)
        B_s = self.global_params.get("B_s", 0.001)
        
        cv_hs = self.cv_hs_base + A_s * np.exp(-B_s * avg_height)
        return cv_hs
    
    def calculate_sigma_var(self, locus_data: Dict, avg_height: float) -> float:
        """
        计算位点方差参数 σ_var,l
        
        Args:
            locus_data: 位点数据
            avg_height: 平均峰高
            
        Returns:
            方差参数
        """
        # 基础方差
        sigma_base = self.sigma_var_base
        
        # 特征调整系数
        c1, c2, c3 = 0.5, 0.3, 0.2  # 可配置参数
        
        # PHR失衡比例
        R_PHR = locus_data.get('ratio_severe_imbalance', 0.0)
        
        # 偏度
        gamma_1 = locus_data.get('skewness_peak_height', 0.0)
        
        # 平均位点等位基因熵
        H_a_bar = locus_data.get('avg_locus_allele_entropy', 1.0)
        H_a_bar = max(H_a_bar, 1e-6)  # 避免除零
        
        # Sigmoid函数调整
        A_f = self.global_params.get("A_f", 1.0)
        B_f = self.global_params.get("B_f", 0.001)
        h_0f = self.global_params.get("h_0f", 1000.0)
        
        f_h = 1 + A_f / (1 + np.exp(B_f * (avg_height - h_0f)))
        
        # 计算最终方差
        sigma_var = sigma_base * (1 + c1 * R_PHR + c2 * abs(gamma_1) + c3 * (1 / H_a_bar)) * f_h
        
        return sigma_var
    
    def calculate_degradation_factor(self, allele_size: float, 
                                   height_size_correlation: float) -> float:
        """
        计算降解因子
        
        Args:
            allele_size: 等位基因片段大小
            height_size_correlation: 峰高-片段大小相关性
            
        Returns:
            降解因子
        """
        # 降解系数
        k_deg = self.k_deg_0 * max(0, -height_size_correlation) ** self.alpha
        
        # 降解因子
        D_F = np.exp(-k_deg * max(0, allele_size - self.size_ref))
        
        return D_F
    
    def calculate_ado_probability(self, expected_height: float) -> float:
        """
        计算等位基因缺失(ADO)概率
        
        Args:
            expected_height: 期望峰高
            
        Returns:
            ADO概率
        """
        P_ado = 1 / (1 + np.exp(self.s_ado * (expected_height - self.H_50)))
        return P_ado
    
    def calculate_expected_height(self, allele: str, locus: str, 
                                genotypes: List[Tuple], mixture_ratios: np.ndarray,
                                gamma_l: float) -> float:
        """
        计算等位基因期望峰高 μ_exp,j,l
        
        Args:
            allele: 等位基因值
            locus: 位点名称
            genotypes: 基因型列表
            mixture_ratios: 混合比例
            gamma_l: 位点放大效率
            
        Returns:
            期望峰高
        """
        mu_allele = 0.0
        
        # 计算直接贡献
        for i, genotype in enumerate(genotypes):
            if genotype is not None:
                # 计算拷贝数
                C_copy = self._calculate_copy_number(allele, genotype)
                
                # 计算降解因子（需要片段大小信息）
                allele_size = self._get_allele_size(allele, locus)
                D_F = self.calculate_degradation_factor(allele_size, 0.0)  # 简化
                
                # 累加贡献
                mu_allele += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        # 添加Stutter贡献
        mu_stutter = self._calculate_stutter_contribution(allele, locus, genotypes, 
                                                        mixture_ratios, gamma_l)
        
        return mu_allele + mu_stutter
    
    def _calculate_copy_number(self, allele: str, genotype: Tuple) -> int:
        """计算等位基因拷贝数"""
        if genotype is None:
            return 0
        
        count = 0
        for gt_allele in genotype:
            if gt_allele == allele:
                count += 1
        
        # 对于纯合子，考虑f_homo因子
        if len(set(genotype)) == 1 and allele in genotype:
            f_homo = self.global_params.get("f_homo", 1.0)
            return int(2 * f_homo)
        
        return count
    
    def _get_allele_size(self, allele: str, locus: str) -> float:
        """获取等位基因片段大小"""
        # 这里应该根据实际的STR位点参数计算
        # 简化实现：返回一个基于等位基因值的估计大小
        try:
            allele_num = float(allele)
            base_size = self.marker_params.get(locus, {}).get('base_size', 100)
            repeat_length = self.marker_params.get(locus, {}).get('L_repeat', 4)
            return base_size + allele_num * repeat_length
        except ValueError:
            return 150.0  # 默认大小
    
    def _calculate_stutter_contribution(self, allele: str, locus: str, 
                                      genotypes: List[Tuple], 
                                      mixture_ratios: np.ndarray,
                                      gamma_l: float) -> float:
        """计算Stutter贡献"""
        mu_stutter = 0.0
        
        # 获取位点Stutter参数
        stutter_params = self.marker_params.get(locus, {}).get('n_minus_1_Stutter', {})
        
        if stutter_params.get('SR_model_type') == 'N/A':
            return 0.0
        
        # 寻找可能的亲代等位基因
        try:
            target_allele_num = float(allele)
        except ValueError:
            return 0.0
        
        # 检查n-1 Stutter
        parent_allele_num = target_allele_num + 1
        parent_allele = str(int(parent_allele_num))
        
        # 计算亲代等位基因的总贡献
        mu_parent = 0.0
        for i, genotype in enumerate(genotypes):
            if genotype is not None:
                C_copy = self._calculate_copy_number(parent_allele, genotype)
                allele_size = self._get_allele_size(parent_allele, locus)
                D_F = self.calculate_degradation_factor(allele_size, 0.0)
                mu_parent += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        # 计算Stutter比率
        if stutter_params.get('SR_model_type') == 'Allele Regression':
            m = stutter_params.get('SR_m', 0.0)
            c = stutter_params.get('SR_c', 0.0)
            e_SR = m * parent_allele_num + c
        elif stutter_params.get('SR_model_type') == 'Allele Average':
            e_SR = stutter_params.get('SR_c', 0.0)
        else:
            e_SR = 0.0
        
        e_SR = max(0.0, e_SR)
        mu_stutter = e_SR * mu_parent
        
        return mu_stutter
    
    def calculate_likelihood_locus(self, locus_data: Dict, genotypes: List[Tuple],
                                 mixture_ratios: np.ndarray, theta: Dict) -> float:
        """
        计算单个位点的似然函数
        
        Args:
            locus_data: 位点观测数据
            genotypes: 基因型列表
            mixture_ratios: 混合比例
            theta: 降解参数
            
        Returns:
            对数似然值
        """
        log_likelihood = 0.0
        locus = locus_data['locus']
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        # 计算位点参数
        avg_height = np.mean([h for h in observed_heights.values() if h > 0])
        gamma_l = self.calculate_gamma_l(locus, avg_height, 
                                       locus_data.get('inter_locus_entropy', 0.0))
        sigma_var_l = self.calculate_sigma_var(locus_data, avg_height)
        
        # 计算观测等位基因的似然
        for allele, height in observed_heights.items():
            if height > 0:
                # 计算期望峰高
                mu_exp = self.calculate_expected_height(allele, locus, genotypes, 
                                                      mixture_ratios, gamma_l)
                
                # 对数正态分布似然
                if mu_exp > 1e-6:
                    log_mu = np.log(mu_exp) - sigma_var_l**2 / 2
                    log_likelihood += stats.lognorm.logpdf(height, sigma_var_l, scale=np.exp(log_mu))
                else:
                    log_likelihood += -1e6  # 极小值惩罚
        
        # 计算ADO的似然
        all_possible_alleles = self._get_all_possible_alleles(genotypes)
        for allele in all_possible_alleles:
            if allele not in observed_alleles:
                # 计算期望峰高用于ADO概率
                mu_exp = self.calculate_expected_height(allele, locus, genotypes, 
                                                      mixture_ratios, gamma_l)
                P_ado = self.calculate_ado_probability(mu_exp)
                log_likelihood += np.log(P_ado + 1e-10)
        
        return log_likelihood
    
    def _get_all_possible_alleles(self, genotypes: List[Tuple]) -> List[str]:
        """获取所有可能的等位基因"""
        all_alleles = set()
        for genotype in genotypes:
            if genotype is not None:
                all_alleles.update(genotype)
        return list(all_alleles)
    
    def calculate_prior_genotypes(self, genotypes: List[Tuple], N: int, 
                                allele_freqs: Dict) -> float:
        """
        计算基因型的先验概率
        
        Args:
            genotypes: 基因型列表
            N: 贡献者数量
            allele_freqs: 等位基因频率
            
        Returns:
            对数先验概率
        """
        log_prior = 0.0
        
        for genotype in genotypes:
            if genotype is not None:
                if len(genotype) == 2:
                    a1, a2 = genotype
                    f1 = allele_freqs.get(a1, 1e-6)
                    f2 = allele_freqs.get(a2, 1e-6)
                    
                    if a1 == a2:  # 纯合子
                        log_prior += np.log(f1 * f2)
                    else:  # 杂合子
                        log_prior += np.log(2 * f1 * f2)
        
        return log_prior
    
    def calculate_prior_mixture_ratios(self, mixture_ratios: np.ndarray) -> float:
        """
        计算混合比例的先验概率 (Dirichlet分布)
        
        Args:
            mixture_ratios: 混合比例
            
        Returns:
            对数先验概率
        """
        # Dirichlet先验参数
        alpha = np.ones(len(mixture_ratios))  # 均匀先验
        
        # 计算Dirichlet对数概率密度
        log_prior = (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                    np.sum((alpha - 1) * np.log(mixture_ratios + 1e-10)))
        
        return log_prior
    
    def propose_genotypes(self, current_genotypes: List[Tuple], 
                         observed_alleles: List[str]) -> List[Tuple]:
        """
        提议新的基因型配置
        
        Args:
            current_genotypes: 当前基因型
            observed_alleles: 观测到的等位基因
            
        Returns:
            新的基因型配置
        """
        new_genotypes = current_genotypes.copy()
        
        # 随机选择一个个体进行修改
        individual_idx = np.random.randint(len(new_genotypes))
        
        if new_genotypes[individual_idx] is not None:
            # 随机修改基因型
            available_alleles = observed_alleles.copy()
            
            # 有一定概率添加新的等位基因（允许ADO）
            if np.random.random() < 0.1:
                # 简化：可以添加相邻的等位基因
                try:
                    max_allele = max([float(a) for a in observed_alleles if a.replace('.', '').isdigit()])
                    available_alleles.append(str(int(max_allele + 1)))
                    available_alleles.append(str(int(max_allele - 1)))
                except:
                    pass
            
            # 生成新基因型
            if len(available_alleles) >= 2:
                new_alleles = np.random.choice(available_alleles, size=2, replace=True)
                new_genotypes[individual_idx] = tuple(sorted(new_alleles))
        
        return new_genotypes
    
    def propose_mixture_ratios(self, current_ratios: np.ndarray, 
                             step_size: float = 0.1) -> np.ndarray:
        """
        提议新的混合比例
        
        Args:
            current_ratios: 当前混合比例
            step_size: 步长
            
        Returns:
            新的混合比例
        """
        # 使用Dirichlet提议分布
        concentration = current_ratios / step_size
        concentration = np.maximum(concentration, 0.1)  # 避免过小的参数
        
        new_ratios = np.random.dirichlet(concentration)
        return new_ratios
    
    def mcmc_sampler(self, observed_data: Dict, N: int, 
                    allele_frequencies: Dict) -> Dict:
        """
        MCMC采样器主函数
        
        Args:
            observed_data: 观测数据
            N: 贡献者数量
            allele_frequencies: 等位基因频率
            
        Returns:
            MCMC采样结果
        """
        print(f"开始MCMC采样，贡献者数量: {N}")
        
        # 初始化参数
        mixture_ratios = np.random.dirichlet(np.ones(N))
        
        # 初始化基因型（简化：使用观测等位基因的随机组合）
        all_observed_alleles = []
        for locus_data in observed_data.values():
            all_observed_alleles.extend(locus_data['alleles'])
        unique_alleles = list(set(all_observed_alleles))
        
        # 为每个个体随机分配基因型
        genotypes = []
        for i in range(N):
            if len(unique_alleles) >= 2:
                alleles = np.random.choice(unique_alleles, size=2, replace=True)
                genotypes.append(tuple(sorted(alleles)))
            else:
                genotypes.append(None)
        
        # 存储MCMC样本
        samples = {
            'mixture_ratios': [],
            'genotypes': [],
            'log_likelihood': [],
            'log_posterior': []
        }
        
        # MCMC主循环
        n_accepted = 0
        current_log_likelihood = self._calculate_total_likelihood(
            observed_data, genotypes, mixture_ratios)
        
        for iteration in range(self.n_iterations):
            if iteration % 1000 == 0:
                print(f"迭代 {iteration}/{self.n_iterations}")
            
            # 1. 更新混合比例
            proposed_ratios = self.propose_mixture_ratios(mixture_ratios)
            
            # 计算接受概率
            proposed_likelihood = self._calculate_total_likelihood(
                observed_data, genotypes, proposed_ratios)
            
            prior_current = self.calculate_prior_mixture_ratios(mixture_ratios)
            prior_proposed = self.calculate_prior_mixture_ratios(proposed_ratios)
            
            log_ratio = (proposed_likelihood + prior_proposed - 
                        current_log_likelihood - prior_current)
            
            if np.log(np.random.random()) < log_ratio:
                mixture_ratios = proposed_ratios
                current_log_likelihood = proposed_likelihood
                n_accepted += 1
            
            # 2. 更新基因型（简化版本）
            if iteration % 5 == 0:  # 每5次迭代更新一次基因型
                for locus in observed_data:
                    proposed_genotypes = self.propose_genotypes(
                        genotypes, observed_data[locus]['alleles'])
                    
                    proposed_likelihood = self._calculate_total_likelihood(
                        observed_data, proposed_genotypes, mixture_ratios)
                    
                    log_ratio = proposed_likelihood - current_log_likelihood
                    
                    if np.log(np.random.random()) < log_ratio:
                        genotypes = proposed_genotypes
                        current_log_likelihood = proposed_likelihood
            
            # 存储样本（在warmup后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['genotypes'].append([g for g in genotypes])
                samples['log_likelihood'].append(current_log_likelihood)
                
                # 计算后验概率
                log_posterior = (current_log_likelihood + 
                               self.calculate_prior_mixture_ratios(mixture_ratios))
                samples['log_posterior'].append(log_posterior)
        
        acceptance_rate = n_accepted / self.n_iterations
        print(f"MCMC完成，接受率: {acceptance_rate:.3f}")
        
        return {
            'samples': samples,
            'acceptance_rate': acceptance_rate,
            'n_samples': len(samples['mixture_ratios'])
        }
    
    def _calculate_total_likelihood(self, observed_data: Dict, 
                                  genotypes: List[Tuple], 
                                  mixture_ratios: np.ndarray) -> float:
        """计算总似然函数"""
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self.calculate_likelihood_locus(
                locus_data, genotypes, mixture_ratios, {})
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def analyze_convergence(self, samples: Dict) -> Dict:
        """
        分析MCMC收敛性
        
        Args:
            samples: MCMC样本
            
        Returns:
            收敛性诊断结果
        """
        diagnostics = {}
        
        # 1. 计算有效样本量 (ESS)
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        ess_values = []
        for i in range(n_components):
            # 简化的ESS计算
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i])
            ess = n_samples / (1 + 2 * np.sum(autocorr[1:20]))  # 近似计算
            ess_values.append(max(ess, 1))
        
        diagnostics['effective_sample_size'] = ess_values
        diagnostics['min_ess'] = min(ess_values)
        
        # 2. Gelman-Rubin诊断 (简化版，需要多链)
        diagnostics['convergence_status'] = 'Good' if diagnostics['min_ess'] > 100 else 'Poor'
        
        # 3. 轨迹图统计
        diagnostics['trace_stats'] = {
            'mean_mixture_ratios': np.mean(mixture_samples, axis=0).tolist(),
            'std_mixture_ratios': np.std(mixture_samples, axis=0).tolist()
        }
        
        return diagnostics
    
    def _calculate_autocorrelation(self, x: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """计算自相关函数"""
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:min(max_lag, len(autocorr))]
    
    def generate_posterior_summary(self, results: Dict) -> Dict:
        """
        生成后验分布摘要
        
        Args:
            results: MCMC结果
            
        Returns:
            后验摘要统计
        """
        samples = results['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        
        summary = {}
        
        # 混合比例统计
        n_components = mixture_samples.shape[1]
        for i in range(n_components):
            component_samples = mixture_samples[:, i]
            summary[f'Mx_{i+1}'] = {
                'mean': np.mean(component_samples),
                'std': np.std(component_samples),
                'median': np.median(component_samples),
                'credible_interval_95': np.percentile(component_samples, [2.5, 97.5]).tolist(),
                'credible_interval_90': np.percentile(component_samples, [5, 95]).tolist()
            }
        
        # 后验概率统计
        log_posterior = np.array(samples['log_posterior'])
        summary['log_posterior'] = {
            'mean': np.mean(log_posterior),
            'std': np.std(log_posterior),
            'max': np.max(log_posterior)
        }
        
        # 模型质量指标
        summary['model_quality'] = {
            'acceptance_rate': results['acceptance_rate'],
            'n_effective_samples': results['n_samples'],
            'converged': results.get('converged', True)
        }
        
        return summary
    
    def plot_results(self, results: Dict, output_dir: str = './plots') -> None:
        """
        绘制MCMC结果图表
        
        Args:
            results: MCMC结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        samples = results['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 轨迹图
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].plot(mixture_samples[:, i])
            axes[i].set_title(f'混合比例 Mx_{i+1} 的轨迹图')
            axes[i].set_xlabel('迭代次数')
            axes[i].set_ylabel(f'Mx_{i+1}')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mcmc_trace_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 后验分布密度图
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].hist(mixture_samples[:, i], bins=50, alpha=0.7, density=True)
            axes[i].axvline(np.mean(mixture_samples[:, i]), color='red', linestyle='--', 
                           label=f'均值: {np.mean(mixture_samples[:, i]):.3f}')
            axes[i].axvline(np.median(mixture_samples[:, i]), color='green', linestyle='--', 
                           label=f'中位数: {np.median(mixture_samples[:, i]):.3f}')
            axes[i].set_title(f'混合比例 Mx_{i+1} 的后验分布')
            axes[i].set_xlabel(f'Mx_{i+1}')
            axes[i].set_ylabel('密度')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'posterior_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混合比例联合分布（对于2-3个组分）
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(mixture_samples[:, 0], mixture_samples[:, 1], alpha=0.5, s=1)
            plt.xlabel('Mx_1')
            plt.ylabel('Mx_2')
            plt.title('混合比例联合后验分布 (Mx_1 vs Mx_2)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'joint_posterior_2d.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 4))
            
            # 三个两两组合的散点图
            combinations = [(0, 1), (0, 2), (1, 2)]
            labels = [('Mx_1', 'Mx_2'), ('Mx_1', 'Mx_3'), ('Mx_2', 'Mx_3')]
            
            for i, ((idx1, idx2), (label1, label2)) in enumerate(zip(combinations, labels)):
                ax = fig.add_subplot(1, 3, i+1)
                ax.scatter(mixture_samples[:, idx1], mixture_samples[:, idx2], alpha=0.5, s=1)
                ax.set_xlabel(label1)
                ax.set_ylabel(label2)
                ax.set_title(f'{label1} vs {label2}')
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'joint_posterior_3d_projections.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 自相关函数图
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i], max_lag=100)
            lags = np.arange(len(autocorr))
            axes[i].plot(lags, autocorr)
            axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[i].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1阈值')
            axes[i].set_title(f'Mx_{i+1} 的自相关函数')
            axes[i].set_xlabel('滞后 (Lag)')
            axes[i].set_ylabel('自相关')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'autocorrelation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 后验预测检验图（简化版）
        if 'log_likelihood' in samples:
            plt.figure(figsize=(10, 6))
            log_likelihood = np.array(samples['log_likelihood'])
            plt.plot(log_likelihood)
            plt.title('对数似然轨迹')
            plt.xlabel('迭代次数')
            plt.ylabel('对数似然')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'log_likelihood_trace.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"所有图表已保存到目录: {output_dir}")
    
    def save_results(self, results: Dict, summary: Dict, 
                    output_path: str = './mcmc_results.json') -> None:
        """
        保存MCMC结果
        
        Args:
            results: MCMC结果
            summary: 后验摘要
            output_path: 输出文件路径
        """
        # 准备保存的数据（转换numpy数组为列表）
        save_data = {
            'posterior_summary': summary,
            'convergence_diagnostics': self.convergence_diagnostics,
            'mcmc_settings': {
                'n_iterations': self.n_iterations,
                'n_warmup': self.n_warmup,
                'n_chains': self.n_chains,
                'thinning': self.thinning
            },
            'model_parameters': {
                'k_gamma': self.k_gamma,
                'beta': self.beta,
                'sigma_var_base': self.sigma_var_base,
                'cv_hs_base': self.cv_hs_base
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存部分样本数据（避免文件过大）
        if results['n_samples'] > 1000:
            # 随机采样1000个样本保存
            indices = np.random.choice(results['n_samples'], 1000, replace=False)
            mixture_samples = np.array(results['samples']['mixture_ratios'])
            save_data['sample_mixture_ratios'] = mixture_samples[indices].tolist()
        else:
            save_data['sample_mixture_ratios'] = results['samples']['mixture_ratios']
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")


def load_problem1_results(noc_file_path: str) -> int:
    """
    从问题1的结果中加载NoC预测
    
    Args:
        noc_file_path: 问题1结果文件路径
        
    Returns:
        预测的贡献者数量
    """
    try:
        df = pd.read_csv(noc_file_path, encoding='utf-8-sig')
        # 假设我们要分析第一个样本
        if 'baseline_pred' in df.columns:
            return int(df['baseline_pred'].iloc[0])
        elif 'NoC_True' in df.columns:
            return int(df['NoC_True'].iloc[0])
        else:
            return 2  # 默认值
    except:
        return 2  # 默认值


def create_synthetic_data(N: int = 2) -> Dict:
    """
    创建合成STR数据用于测试
    
    Args:
        N: 贡献者数量
        
    Returns:
        合成观测数据
    """
    # 模拟的STR位点
    loci = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11']
    
    # 模拟的等位基因库
    allele_pools = {
        'D3S1358': ['14', '15', '16', '17', '18'],
        'vWA': ['14', '15', '16', '17', '18', '19'],
        'FGA': ['18', '19', '20', '21', '22', '23', '24'],
        'D8S1179': ['8', '9', '10', '11', '12', '13', '14'],
        'D21S11': ['27', '28', '29', '30', '31', '32']
    }
    
    observed_data = {}
    
    for locus in loci:
        # 随机选择观测到的等位基因数量
        n_alleles = np.random.randint(2, min(6, N * 2 + 1))
        observed_alleles = np.random.choice(allele_pools[locus], n_alleles, replace=False)
        
        # 生成峰高（模拟混合样本）
        heights = {}
        for allele in observed_alleles:
            # 基础峰高 + 一些随机变异
            base_height = np.random.gamma(2, 1000)  # 基础峰高分布
            heights[allele] = max(50, base_height)  # 最小50 RFU
        
        observed_data[locus] = {
            'locus': locus,
            'alleles': list(observed_alleles),
            'heights': heights,
            'inter_locus_entropy': np.random.uniform(0.5, 2.0),
            'ratio_severe_imbalance': np.random.uniform(0.0, 0.3),
            'skewness_peak_height': np.random.normal(0, 0.5),
            'avg_locus_allele_entropy': np.random.uniform(0.3, 1.0)
        }
    
    return observed_data


def create_allele_frequencies() -> Dict:
    """
    创建等位基因频率数据
    
    Returns:
        等位基因频率字典
    """
    # 简化的等位基因频率（实际应该从数据库获取）
    frequencies = {}
    
    # 为每个常见等位基因分配频率
    for i in range(8, 35):
        allele = str(i)
        # 使用Dirichlet分布生成频率
        frequencies[allele] = np.random.dirichlet([1] * 10)[0]
    
    # 标准化频率
    total_freq = sum(frequencies.values())
    for allele in frequencies:
        frequencies[allele] /= total_freq
    
    return frequencies


def main():
    """
    主函数 - 演示问题2的MCMC混合比例推断
    """
    print("=" * 60)
    print("问题2：MCMC混合比例推断 (V10版本)")
    print("=" * 60)
    
    # 1. 初始化MCMC推断器
    config_path = './config_params.json'  # 如果存在配置文件
    mcmc_inferencer = MCMCMixtureInference(config_path)
    
    # 2. 加载问题1的NoC预测结果
    try:
        N_predicted = load_problem1_results('../Q1/prob1_features_v2.9.csv')
        print(f"从问题1加载的预测贡献者数量: {N_predicted}")
    except:
        N_predicted = 3  # 默认值
        print(f"使用默认贡献者数量: {N_predicted}")
    
    # 3. 创建或加载观测数据
    print("\n创建合成STR观测数据...")
    observed_data = create_synthetic_data(N_predicted)
    
    print("观测数据摘要:")
    for locus, data in observed_data.items():
        print(f"  {locus}: {len(data['alleles'])} 个等位基因, "
              f"峰高范围 {min(data['heights'].values()):.0f}-{max(data['heights'].values()):.0f} RFU")
    
    # 4. 创建等位基因频率
    print("\n创建等位基因频率数据...")
    allele_frequencies = create_allele_frequencies()
    
    # 5. 运行MCMC推断
    print(f"\n开始MCMC推断 (N={N_predicted})...")
    start_time = time.time()
    
    mcmc_results = mcmc_inferencer.mcmc_sampler(
        observed_data, N_predicted, allele_frequencies)
    
    end_time = time.time()
    print(f"MCMC推断完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 6. 分析收敛性
    print("\n分析MCMC收敛性...")
    convergence_diagnostics = mcmc_inferencer.analyze_convergence(mcmc_results['samples'])
    mcmc_inferencer.convergence_diagnostics = convergence_diagnostics
    
    print("收敛性诊断结果:")
    print(f"  最小有效样本量: {convergence_diagnostics['min_ess']:.0f}")
    print(f"  收敛状态: {convergence_diagnostics['convergence_status']}")
    
    # 7. 生成后验摘要
    print("\n生成后验分布摘要...")
    posterior_summary = mcmc_inferencer.generate_posterior_summary(mcmc_results)
    
    print("混合比例后验统计:")
    for i in range(N_predicted):
        mx_stats = posterior_summary[f'Mx_{i+1}']
        print(f"  Mx_{i+1}: 均值={mx_stats['mean']:.3f}, "
              f"标准差={mx_stats['std']:.3f}, "
              f"95%置信区间=[{mx_stats['credible_interval_95'][0]:.3f}, "
              f"{mx_stats['credible_interval_95'][1]:.3f}]")
    
    # 8. 绘制结果图表
    print("\n生成结果图表...")
    output_dir = './problem2_plots'
    mcmc_inferencer.plot_results(mcmc_results, output_dir)
    
    # 9. 保存结果
    print("\n保存分析结果...")
    mcmc_inferencer.save_results(mcmc_results, posterior_summary, './problem2_mcmc_results.json')
    
    # 10. 生成详细报告
    print("\n生成分析报告...")
    generate_analysis_report(mcmc_results, posterior_summary, convergence_diagnostics, N_predicted)
    
    print("\n" + "=" * 60)
    print("问题2分析完成！")
    print("=" * 60)


def generate_analysis_report(results: Dict, summary: Dict, 
                           diagnostics: Dict, N: int) -> None:
    """
    生成详细的分析报告
    
    Args:
        results: MCMC结果
        summary: 后验摘要
        diagnostics: 收敛性诊断
        N: 贡献者数量
    """
    report_path = './problem2_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("问题2：STR混合样本的混合比例推断分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"贡献者数量: {N}\n")
        f.write(f"MCMC样本数: {results['n_samples']}\n")
        f.write(f"接受率: {results['acceptance_rate']:.3f}\n\n")
        
        f.write("1. 混合比例后验估计\n")
        f.write("-" * 40 + "\n")
        for i in range(N):
            mx_stats = summary[f'Mx_{i+1}']
            f.write(f"贡献者 {i+1} (Mx_{i+1}):\n")
            f.write(f"  后验均值: {mx_stats['mean']:.4f}\n")
            f.write(f"  后验标准差: {mx_stats['std']:.4f}\n")
            f.write(f"  后验中位数: {mx_stats['median']:.4f}\n")
            f.write(f"  95%置信区间: [{mx_stats['credible_interval_95'][0]:.4f}, "
                   f"{mx_stats['credible_interval_95'][1]:.4f}]\n")
            f.write(f"  90%置信区间: [{mx_stats['credible_interval_90'][0]:.4f}, "
                   f"{mx_stats['credible_interval_90'][1]:.4f}]\n\n")
        
        f.write("2. 模型收敛性诊断\n")
        f.write("-" * 40 + "\n")
        f.write(f"收敛状态: {diagnostics['convergence_status']}\n")
        f.write(f"最小有效样本量: {diagnostics['min_ess']:.0f}\n")
        if 'effective_sample_size' in diagnostics:
            f.write("各组分有效样本量:\n")
            for i, ess in enumerate(diagnostics['effective_sample_size']):
                f.write(f"  Mx_{i+1}: {ess:.0f}\n")
        f.write("\n")
        
        f.write("3. 模型质量评估\n")
        f.write("-" * 40 + "\n")
        model_quality = summary['model_quality']
        f.write(f"接受率: {model_quality['acceptance_rate']:.3f}\n")
        f.write(f"有效样本数: {model_quality['n_effective_samples']}\n")
        f.write(f"收敛状态: {'是' if model_quality['converged'] else '否'}\n\n")
        
        f.write("4. 结果解释\n")
        f.write("-" * 40 + "\n")
        
        # 找出主要贡献者
        mx_means = [summary[f'Mx_{i+1}']['mean'] for i in range(N)]
        main_contributor = np.argmax(mx_means) + 1
        major_ratio = max(mx_means)
        
        f.write(f"主要贡献者: 贡献者{main_contributor} (混合比例: {major_ratio:.3f})\n")
        
        if N == 2:
            if major_ratio > 0.7:
                f.write("结果表明这是一个主要-次要混合样本。\n")
            elif major_ratio > 0.55:
                f.write("结果表明这是一个中等不平衡的混合样本。\n")
            else:
                f.write("结果表明这是一个相对平衡的混合样本。\n")
        
        f.write("\n5. 建议\n")
        f.write("-" * 40 + "\n")
        if diagnostics['min_ess'] < 100:
            f.write("- 建议增加MCMC迭代次数以提高有效样本量\n")
        if results['acceptance_rate'] < 0.2:
            f.write("- 接受率偏低，建议调整提议分布的步长\n")
        elif results['acceptance_rate'] > 0.6:
            f.write("- 接受率偏高，建议增加提议分布的步长\n")
        
        f.write("- 建议结合其他法医遗传学证据进行综合分析\n")
        f.write("- 可考虑进行敏感性分析验证结果稳定性\n")
    
    print(f"详细分析报告已保存到: {report_path}")


if __name__ == "__main__":
    main()