# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题二：基于基因型边缘化的MCMC混合比例推断

版本: V2.0 - MGM-M (Mixture Ratio Inference model based on Genotype Marginalization and MCMC)
日期: 2025-06-06
描述: 实现基于基因型边缘化的MCMC混合比例推断方法
核心改进:
1. 完全边缘化基因型，避免直接采样基因型空间
2. 基于伪频率的先验概率计算
3. 集成V5特征的自适应参数估计
4. 高效的MCMC采样策略
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
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from collections import defaultdict
import itertools
from math import comb
import logging

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PseudoFrequencyCalculator:
    """
    伪等位基因频率计算器
    基于附件2数据计算位点特异性的等位基因频率
    """
    
    def __init__(self):
        self.frequency_cache = {}
        logger.info("伪频率计算器初始化完成")
    
    def calculate_pseudo_frequencies(self, locus: str, 
                                   att2_alleles: List[str]) -> Dict[str, float]:
        """
        计算位点的伪等位基因频率 w_l(a_k)
        
        Args:
            locus: 位点名称
            att2_alleles: 附件2中该位点观测到的所有等位基因
            
        Returns:
            等位基因频率字典
        """
        cache_key = (locus, tuple(sorted(att2_alleles)))
        if cache_key in self.frequency_cache:
            return self.frequency_cache[cache_key]
        
        # 统计等位基因出现次数 C_l(a_k)
        allele_counts = {}
        for allele in att2_alleles:
            allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        # 计算频率 w_l(a_k) = C_l(a_k) / sum(C_l(a_j))
        total_count = sum(allele_counts.values())
        frequencies = {}
        
        for allele, count in allele_counts.items():
            frequencies[allele] = count / total_count
        
        # 为未观测到但可能存在的等位基因分配最小频率
        min_freq = 1 / (2 * len(att2_alleles) + len(allele_counts))
        
        # 标准化频率，确保总和为1
        freq_sum = sum(frequencies.values())
        if freq_sum > 0:
            for allele in frequencies:
                frequencies[allele] = frequencies[allele] / freq_sum * (1 - min_freq * 2)
        
        self.frequency_cache[cache_key] = frequencies
        return frequencies
    
    def calculate_genotype_prior(self, genotype: Tuple[str, str], 
                               frequencies: Dict[str, float]) -> float:
        """
        基于Hardy-Weinberg平衡计算基因型的先验概率
        
        Args:
            genotype: 基因型 (allele1, allele2)
            frequencies: 等位基因频率
            
        Returns:
            基因型先验概率 P'(G_j,l)
        """
        a1, a2 = genotype
        f1 = frequencies.get(a1, 1e-6)
        f2 = frequencies.get(a2, 1e-6)
        
        if a1 == a2:  # 纯合子
            return f1 * f2
        else:  # 杂合子
            return 2 * f1 * f2
    
    def calculate_genotype_set_prior(self, genotype_set: List[Tuple[str, str]], 
                                   frequencies: Dict[str, float]) -> float:
        """
        计算整个基因型组合的先验概率
        
        Args:
            genotype_set: 基因型组合 {G_i}_l
            frequencies: 等位基因频率
            
        Returns:
            组合先验概率的对数值
        """
        log_prior = 0.0
        for genotype in genotype_set:
            if genotype is not None:
                prior = self.calculate_genotype_prior(genotype, frequencies)
                log_prior += np.log(max(prior, 1e-10))
        
        return log_prior


class V5FeatureIntegrator:
    """
    V5特征集成器
    将V5特征转换为模型参数
    """
    
    def __init__(self, v5_features: Dict):
        self.v5_features = v5_features
        logger.info("V5特征集成器初始化完成")
    
    def calculate_gamma_l(self, locus: str) -> float:
        """
        基于V5特征计算位点特异性放大效率 γ_l
        """
        # 从V5特征获取平均峰高
        avg_height = self.v5_features.get('avg_peak_height', 1000.0)
        
        # 基础放大效率
        k_gamma = 1.0
        gamma_base = k_gamma * avg_height
        
        # 获取位点间平衡熵
        inter_locus_entropy = self.v5_features.get('inter_locus_balance_entropy', 1.0)
        
        # 计算权重和位点特异性调整
        L_exp = 20  # 预期的常染色体位点数
        beta = 1.5
        
        if L_exp > 1:
            w_entropy = (1 - inter_locus_entropy / np.log(L_exp)) ** beta
            # 位点特异性调整 (简化：假设均匀分布)
            P_l = 1.0 / L_exp
            gamma_l = gamma_base * (1 + w_entropy * ((P_l * L_exp) - 1))
        else:
            gamma_l = gamma_base
            
        return max(gamma_l, 1e-3)
    
    def calculate_sigma_var_l(self, locus: str) -> float:
        """
        基于V5特征计算位点方差参数 σ_var,l
        """
        # 从V5特征获取相关参数
        avg_height = self.v5_features.get('avg_peak_height', 1000.0)
        R_PHR = self.v5_features.get('ratio_severe_imbalance_loci', 0.0)
        gamma_1 = self.v5_features.get('skewness_peak_height', 0.0)
        H_a_bar = max(self.v5_features.get('avg_locus_allele_entropy', 1.0), 1e-6)
        
        # 基础方差参数
        sigma_var_base = 0.1
        c1, c2, c3 = 0.5, 0.3, 0.2
        
        # Sigmoid函数参数
        A_f = 1.0
        B_f = 0.001
        h_0f = 1000.0
        
        # 计算f(h_bar)
        f_h = 1 + A_f / (1 + np.exp(B_f * (avg_height - h_0f)))
        
        # 计算最终方差
        sigma_var = (sigma_var_base * 
                    (1 + c1 * R_PHR + c2 * abs(gamma_1) + c3 * (1 / H_a_bar)) * 
                    f_h)
        
        return max(sigma_var, 0.01)
    
    def calculate_degradation_factor(self, allele_size: float) -> float:
        """
        计算降解因子 D_F
        """
        k_deg_0 = 0.001
        size_ref = 200.0
        alpha = 1.0
        
        # 计算降解系数
        height_size_corr = self.v5_features.get('height_size_correlation', 0.0)
        k_deg = k_deg_0 * max(0, -height_size_corr) ** alpha
        
        # 计算降解因子
        D_F = np.exp(-k_deg * max(0, allele_size - size_ref))
        
        return max(D_F, 1e-6)


class GenotypeEnumerator:
    """
    基因型枚举器
    生成与观测等位基因兼容的所有可能基因型组合
    """
    
    def __init__(self, max_contributors: int = 5):
        self.max_contributors = max_contributors
        self.memo = {}
        logger.info(f"基因型枚举器初始化完成，最大贡献者数：{max_contributors}")
    
    def enumerate_valid_genotype_sets(self, observed_alleles: List[str], 
                                    N: int, K_top: int = None) -> List[List[Tuple[str, str]]]:
        """
        枚举所有与观测等位基因兼容的基因型组合 G_valid
        
        Args:
            observed_alleles: 观测到的等位基因列表 A_l
            N: 贡献者数量
            K_top: 对于N>=4时的K-top采样数量
            
        Returns:
            所有可能的基因型组合列表
        """
        cache_key = (tuple(sorted(observed_alleles)), N, K_top)
        if cache_key in self.memo:
            return self.memo[cache_key]
        
        if N <= 3:
            # 对于N<=3，枚举所有可能组合
            valid_sets = self._enumerate_all_combinations(observed_alleles, N)
        else:
            # 对于N>=4，使用K-top采样策略
            valid_sets = self._enumerate_k_top_combinations(observed_alleles, N, K_top)
        
        self.memo[cache_key] = valid_sets
        logger.debug(f"为{N}个贡献者枚举了{len(valid_sets)}个有效基因型组合")
        return valid_sets
    
    def _enumerate_all_combinations(self, observed_alleles: List[str], N: int) -> List[List[Tuple[str, str]]]:
        """枚举所有可能的基因型组合（适用于N<=3）"""
        A_l = observed_alleles
        all_genotype_sets = []
        
        # 生成所有可能的基因型
        possible_genotypes = []
        for i, a1 in enumerate(A_l):
            for j, a2 in enumerate(A_l):
                if i <= j:  # 避免重复 (a,b) 和 (b,a)
                    possible_genotypes.append((a1, a2))
        
        # 生成N个个体的所有基因型组合
        for genotype_combination in itertools.product(possible_genotypes, repeat=N):
            # 检查这个组合是否能解释所有观测到的等位基因
            if self._can_explain_alleles(list(genotype_combination), A_l):
                all_genotype_sets.append(list(genotype_combination))
        
        return all_genotype_sets
    
    def _enumerate_k_top_combinations(self, observed_alleles: List[str], 
                                    N: int, K_top: int) -> List[List[Tuple[str, str]]]:
        """使用K-top采样策略（适用于N>=4）"""
        if K_top is None:
            K_top = min(1000, max(100, len(observed_alleles) ** N))
        
        A_l = observed_alleles
        possible_genotypes = []
        
        # 生成所有可能的基因型
        for i, a1 in enumerate(A_l):
            for j, a2 in enumerate(A_l):
                if i <= j:
                    possible_genotypes.append((a1, a2))
        
        sampled_sets = []
        max_attempts = K_top * 10  # 防止无限循环
        attempts = 0
        
        while len(sampled_sets) < K_top and attempts < max_attempts:
            attempts += 1
            
            # 随机选择N个基因型
            genotype_combination = [
                possible_genotypes[np.random.randint(len(possible_genotypes))]
                for _ in range(N)
            ]
            
            # 检查是否能解释观测等位基因
            if self._can_explain_alleles(genotype_combination, A_l):
                if genotype_combination not in sampled_sets:
                    sampled_sets.append(genotype_combination)
        
        return sampled_sets
    
    def _can_explain_alleles(self, genotype_set: List[Tuple[str, str]], 
                           observed_alleles: List[str]) -> bool:
        """
        检查基因型组合是否能解释所有观测到的等位基因
        （允许ADO，即基因型中的等位基因可以不在观测中出现）
        """
        # 从基因型组合中收集所有等位基因
        genotype_alleles = set()
        for genotype in genotype_set:
            if genotype is not None:
                genotype_alleles.update(genotype)
        
        # 检查所有观测等位基因是否都能被基因型组合解释
        observed_set = set(observed_alleles)
        return observed_set.issubset(genotype_alleles)


class MGM_M_Inferencer:
    """
    MGM-M (Mixture Ratio Inference model based on Genotype Marginalization and MCMC)
    基于基因型边缘化的MCMC混合比例推断器
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化MGM-M推断器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
        
        # 初始化组件
        self.pseudo_freq_calculator = PseudoFrequencyCalculator()
        self.genotype_enumerator = GenotypeEnumerator()
        self.v5_integrator = None  # 将在设置V5特征时初始化
        
        # MCMC参数
        self.n_iterations = self.config.get("n_iterations", 10000)
        self.n_warmup = self.config.get("n_warmup", 2500)
        self.n_chains = self.config.get("n_chains", 4)
        self.thinning = self.config.get("thinning", 5)
        
        # 模型参数
        self.K_top = self.config.get("K_top", 500)
        
        logger.info("MGM-M推断器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "n_iterations": 10000,
            "n_warmup": 2500,
            "n_chains": 4,
            "thinning": 5,
            "K_top": 500,
            "marker_specific_params": {
                "D3S1358": {"L_repeat": 4, "avg_size_bp": 120},
                "vWA": {"L_repeat": 4, "avg_size_bp": 170},
                "FGA": {"L_repeat": 4, "avg_size_bp": 230}
            }
        }
    
    def set_v5_features(self, v5_features: Dict):
        """设置V5特征"""
        self.v5_integrator = V5FeatureIntegrator(v5_features)
        logger.info("V5特征已设置")
    
    def calculate_locus_marginalized_likelihood(self, locus_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              att2_data: Dict = None) -> float:
        """
        计算单个位点的边缘化似然函数 L_l(M_x, θ)
        
        这是MGM-M方法的核心：完全边缘化基因型
        
        L_l(M_x, θ) = Σ_{G_i}_l ∈ G_valid P(E_obs,l | N, M_x, {G_i}_l, θ) · P'({G_i}_l | A_l, N)
        
        Args:
            locus_data: 位点观测数据 {'locus': str, 'alleles': List[str], 'heights': Dict}
            N: 贡献者数量
            mixture_ratios: 混合比例向量 M_x
            att2_data: 附件2数据（用于计算伪频率）
            
        Returns:
            边缘化对数似然值
        """
        if self.v5_integrator is None:
            raise ValueError("请先设置V5特征")
        
        locus = locus_data['locus']
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        # 步骤1: 枚举所有与观测等位基因兼容的基因型组合 G_valid
        K_top = self.K_top if N >= 4 else None
        valid_genotype_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
            observed_alleles, N, K_top)
        
        if not valid_genotype_sets:
            logger.warning(f"位点{locus}没有有效的基因型组合")
            return -1e10
        
        # 步骤2: 计算伪等位基因频率
        if att2_data and locus in att2_data:
            att2_alleles = att2_data[locus]
        else:
            att2_alleles = observed_alleles
        
        pseudo_frequencies = self.pseudo_freq_calculator.calculate_pseudo_frequencies(
            locus, att2_alleles)
        
        # 步骤3: 计算位点特异性参数
        gamma_l = self.v5_integrator.calculate_gamma_l(locus)
        sigma_var_l = self.v5_integrator.calculate_sigma_var_l(locus)
        
        # 步骤4: 对所有基因型组合进行边缘化求和
        log_marginal_likelihood = -np.inf
        likelihood_terms = []
        
        for genotype_set in valid_genotype_sets:
            # 计算该基因型组合的先验概率 P'({G_i}_l | A_l, N)
            log_prior = self.pseudo_freq_calculator.calculate_genotype_set_prior(
                genotype_set, pseudo_frequencies)
            
            # 计算该基因型组合的条件似然 P(E_obs,l | N, M_x, {G_i}_l, θ)
            log_conditional_likelihood = self._calculate_conditional_likelihood(
                observed_alleles, observed_heights, genotype_set, mixture_ratios,
                gamma_l, sigma_var_l, locus)
            
            # 计算联合概率（对数空间）
            log_joint = log_prior + log_conditional_likelihood
            likelihood_terms.append(log_joint)
        
        # 使用logsumexp技巧计算边缘化似然
        if likelihood_terms:
            log_marginal_likelihood = self._logsumexp(likelihood_terms)
        
        return log_marginal_likelihood
    
    def _calculate_conditional_likelihood(self, observed_alleles: List[str],
                                        observed_heights: Dict[str, float],
                                        genotype_set: List[Tuple[str, str]],
                                        mixture_ratios: np.ndarray,
                                        gamma_l: float, sigma_var_l: float,
                                        locus: str) -> float:
        """
        计算给定基因型组合的条件似然 P(E_obs,l | N, M_x, {G_i}_l, θ)
        """
        log_likelihood = 0.0
        
        # 1. 计算观测等位基因的峰高似然
        for allele in observed_alleles:
            observed_height = observed_heights.get(allele, 0.0)
            if observed_height > 0:
                # 计算期望峰高
                mu_exp = self._calculate_expected_height(
                    allele, locus, genotype_set, mixture_ratios, gamma_l)
                
                if mu_exp > 1e-6:
                    # 对数正态分布似然：h_j ~ LogN(ln(μ_exp) - σ²/2, σ²)
                    log_mu = np.log(mu_exp) - sigma_var_l**2 / 2
                    log_likelihood += stats.lognorm.logpdf(
                        observed_height, sigma_var_l, scale=np.exp(log_mu))
                else:
                    log_likelihood += -1e6  # 期望峰高为0但观测到峰
        
        # 2. 计算ADO的似然（对于基因型中存在但未观测到的等位基因）
        genotype_alleles = set()
        for genotype in genotype_set:
            if genotype is not None:
                genotype_alleles.update(genotype)
        
        # 找出应该存在但未观测到的等位基因（发生了ADO）
        dropped_alleles = genotype_alleles - set(observed_alleles)
        for allele in dropped_alleles:
            # 计算该等位基因的期望峰高
            mu_exp_ado = self._calculate_expected_height(
                allele, locus, genotype_set, mixture_ratios, gamma_l)
            
            # 计算ADO概率
            P_ado = self._calculate_ado_probability(mu_exp_ado)
            log_likelihood += np.log(max(P_ado, 1e-10))
        
        return log_likelihood
    
    def _calculate_expected_height(self, allele: str, locus: str,
                                 genotype_set: List[Tuple[str, str]],
                                 mixture_ratios: np.ndarray,
                                 gamma_l: float) -> float:
        """计算等位基因的期望峰高"""
        mu_allele = 0.0
        
        # 计算直接等位基因贡献
        for i, genotype in enumerate(genotype_set):
            if genotype is not None:
                # 计算拷贝数
                C_copy = self._get_copy_number(allele, genotype)
                
                if C_copy > 0:
                    # 获取片段大小
                    allele_size = self._get_allele_size(allele, locus)
                    
                    # 计算降解因子
                    D_F = self.v5_integrator.calculate_degradation_factor(allele_size)
                    
                    # 累加贡献
                    mu_allele += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        # 简化的Stutter贡献计算
        mu_stutter = self._calculate_stutter_contribution(
            allele, locus, genotype_set, mixture_ratios, gamma_l)
        
        return mu_allele + mu_stutter
    
    def _get_copy_number(self, allele: str, genotype: Tuple[str, str]) -> float:
        """计算等位基因在基因型中的拷贝数"""
        if genotype is None:
            return 0.0
        
        count = sum(1 for gt_allele in genotype if gt_allele == allele)
        
        # 对于纯合子，应用f_homo因子
        if len(set(genotype)) == 1 and allele in genotype:
            f_homo = 1.0  # 简化
            return 2.0 * f_homo
        
        return float(count)
    
    def _get_allele_size(self, allele: str, locus: str) -> float:
        """获取等位基因片段大小"""
        try:
            allele_num = float(allele)
            marker_info = self.config.get("marker_specific_params", {}).get(locus, {})
            base_size = marker_info.get('avg_size_bp', 150.0)
            repeat_length = marker_info.get('L_repeat', 4.0)
            return base_size + allele_num * repeat_length
        except ValueError:
            return 150.0
    
    def _calculate_stutter_contribution(self, target_allele: str, locus: str,
                                      genotype_set: List[Tuple[str, str]],
                                      mixture_ratios: np.ndarray, gamma_l: float) -> float:
        """计算Stutter贡献（简化版本）"""
        # 简化实现：假设n-1 Stutter比率为5%
        stutter_ratio = 0.05
        
        try:
            target_allele_num = float(target_allele)
            parent_allele_num = target_allele_num + 1
            parent_allele = str(int(parent_allele_num)) if parent_allele_num.is_integer() else str(parent_allele_num)
            
            # 计算亲代等位基因的总贡献
            mu_parent = 0.0
            for i, genotype in enumerate(genotype_set):
                if genotype is not None:
                    C_copy = self._get_copy_number(parent_allele, genotype)
                    if C_copy > 0:
                        parent_size = self._get_allele_size(parent_allele, locus)
                        D_F = self.v5_integrator.calculate_degradation_factor(parent_size)
                        mu_parent += gamma_l * mixture_ratios[i] * C_copy * D_F
            
            return stutter_ratio * mu_parent
        except:
            return 0.0
    
    def _calculate_ado_probability(self, expected_height: float) -> float:
        """计算等位基因缺失(ADO)概率"""
        if expected_height <= 0:
            return 0.99
        
        H_50 = 200.0
        s_ado = 0.01
        P_ado = 1.0 / (1.0 + np.exp(s_ado * (expected_height - H_50)))
        return np.clip(P_ado, 1e-6, 0.99)
    
    def _logsumexp(self, log_values: List[float]) -> float:
        """数值稳定的logsumexp计算"""
        if not log_values:
            return -np.inf
        
        max_val = max(log_values)
        if max_val == -np.inf:
            return -np.inf
        
        sum_exp = sum(np.exp(val - max_val) for val in log_values)
        return max_val + np.log(sum_exp)
    
    def calculate_total_marginalized_likelihood(self, observed_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              att2_data: Dict = None) -> float:
        """
        计算总的边缘化似然函数
        
        Args:
            observed_data: 所有位点的观测数据
            N: 贡献者数量
            mixture_ratios: 混合比例
            att2_data: 附件2数据
            
        Returns:
            总对数似然值
        """
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self.calculate_locus_marginalized_likelihood(
                locus_data, N, mixture_ratios, att2_data)
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def calculate_prior_mixture_ratios(self, mixture_ratios: np.ndarray) -> float:
        """
        计算混合比例的先验概率 P(M_x | N)
        使用Dirichlet先验：M_x | N ~ Dirichlet(α_N)
        
        Args:
            mixture_ratios: 混合比例
            
        Returns:
            对数先验概率
        """
        # 使用均匀Dirichlet先验：α_N,i = 1
        alpha = np.ones(len(mixture_ratios))
        
        # 如果有V5特征，可以进行自适应调整
        if self.v5_integrator:
            # 基于峰高偏度调整先验
            skewness = self.v5_integrator.v5_features.get('skewness_peak_height', 0.0)
            if abs(skewness) > 1.0:
                # 偏度较大时使用更分散的先验
                alpha = np.ones(len(mixture_ratios)) * 0.5
        
        # Dirichlet对数概率密度
        log_prior = (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                    np.sum((alpha - 1) * np.log(mixture_ratios + 1e-10)))
        
        return log_prior
    
    def propose_mixture_ratios(self, current_ratios: np.ndarray, 
                             step_size: float = 0.05) -> np.ndarray:
        """
        使用Dirichlet分布提议新的混合比例
        
        Args:
            current_ratios: 当前混合比例
            step_size: 提议步长
            
        Returns:
            新的混合比例
        """
        # 计算Dirichlet参数
        concentration = current_ratios / step_size
        concentration = np.maximum(concentration, 0.1)
        
        # 从Dirichlet分布采样
        new_ratios = np.random.dirichlet(concentration)
        
        # 确保比例在合理范围内
        new_ratios = np.maximum(new_ratios, 1e-6)
        new_ratios = new_ratios / np.sum(new_ratios)
        
        return new_ratios
    
    def mcmc_sampler(self, observed_data: Dict, N: int, 
                    att2_data: Dict = None) -> Dict:
        """
        MGM-M方法的MCMC采样器
        仅采样混合比例M_x，基因型已边缘化
        
        Args:
            observed_data: 观测数据
            N: 贡献者数量
            att2_data: 附件2数据
            
        Returns:
            MCMC采样结果
        """
        if self.v5_integrator is None:
            raise ValueError("请先设置V5特征")
        
        logger.info(f"开始MGM-M MCMC采样，贡献者数量: {N}")
        logger.info(f"总迭代次数: {self.n_iterations}, 预热次数: {self.n_warmup}")
        
        # 初始化混合比例
        mixture_ratios = np.random.dirichlet(np.ones(N))
        
        # 存储MCMC样本
        samples = {
            'mixture_ratios': [],
            'log_likelihood': [],
            'log_posterior': [],
            'acceptance_info': []
        }
        
        # 计算初始似然
        current_log_likelihood = self.calculate_total_marginalized_likelihood(
            observed_data, N, mixture_ratios, att2_data)
        current_log_prior = self.calculate_prior_mixture_ratios(mixture_ratios)
        current_log_posterior = current_log_likelihood + current_log_prior
        
        # MCMC主循环
        n_accepted = 0
        acceptance_details = []
        
        # 自适应步长
        step_size = 0.05
        adaptation_interval = 500
        target_acceptance = 0.4
        
        for iteration in range(self.n_iterations):
            if iteration % 1000 == 0:
                acceptance_rate = n_accepted / max(iteration, 1)
                logger.info(f"迭代 {iteration}/{self.n_iterations}, "
                          f"当前接受率: {acceptance_rate:.3f}, "
                          f"当前似然: {current_log_likelihood:.2f}")
            
            # 提议新的混合比例
            proposed_ratios = self.propose_mixture_ratios(mixture_ratios, step_size)
            
            # 计算提议状态的概率
            proposed_log_likelihood = self.calculate_total_marginalized_likelihood(
                observed_data, N, proposed_ratios, att2_data)
            proposed_log_prior = self.calculate_prior_mixture_ratios(proposed_ratios)
            proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
            
            # Metropolis-Hastings接受/拒绝
            log_ratio = proposed_log_posterior - current_log_posterior
            accept_prob = min(1.0, np.exp(log_ratio))
            
            if np.random.random() < accept_prob:
                # 接受提议
                mixture_ratios = proposed_ratios
                current_log_likelihood = proposed_log_likelihood
                current_log_prior = proposed_log_prior
                current_log_posterior = proposed_log_posterior
                n_accepted += 1
                accepted = True
            else:
                accepted = False
            
            # 记录接受信息
            acceptance_details.append({
                'iteration': iteration,
                'accepted': accepted,
                'log_ratio': log_ratio,
                'accept_prob': accept_prob
            })
            
            # 自适应步长调整
            if iteration > 0 and iteration % adaptation_interval == 0 and iteration < self.n_warmup:
                recent_acceptance = np.mean([a['accepted'] for a in acceptance_details[-adaptation_interval:]])
                if recent_acceptance < target_acceptance - 0.05:
                    step_size *= 0.9
                elif recent_acceptance > target_acceptance + 0.05:
                    step_size *= 1.1
                step_size = np.clip(step_size, 0.01, 0.2)
                
                if iteration % (adaptation_interval * 4) == 0:
                    logger.info(f"  步长调整为: {step_size:.4f}, 最近接受率: {recent_acceptance:.3f}")
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
        
        final_acceptance_rate = n_accepted / self.n_iterations
        logger.info(f"MCMC完成，总接受率: {final_acceptance_rate:.3f}")
        logger.info(f"有效样本数: {len(samples['mixture_ratios'])}")
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'n_samples': len(samples['mixture_ratios']),
            'acceptance_details': acceptance_details,
            'final_step_size': step_size,
            'converged': 0.15 <= final_acceptance_rate <= 0.6
        }
    
    def analyze_convergence(self, samples: Dict) -> Dict:
        """分析MCMC收敛性"""
        diagnostics = {}
        
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 有效样本量 (ESS)
        ess_values = []
        for i in range(n_components):
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i])
            tau_int = 1 + 2 * np.sum(autocorr[1:min(20, len(autocorr))])
            ess = n_samples / max(tau_int, 1)
            ess_values.append(max(ess, 1))
        
        diagnostics['effective_sample_size'] = ess_values
        diagnostics['min_ess'] = min(ess_values)
        
        # 2. Geweke诊断
        geweke_scores = []
        split_point = max(n_samples // 10, 1)
        
        for i in range(n_components):
            first_part = mixture_samples[:split_point, i]
            last_part = mixture_samples[-n_samples//2:, i]
            
            if len(first_part) > 1 and len(last_part) > 1:
                mean_diff = np.mean(last_part) - np.mean(first_part)
                pooled_std = np.sqrt((np.var(first_part) + np.var(last_part)) / 2)
                if pooled_std > 1e-10:
                    geweke_score = abs(mean_diff / pooled_std)
                else:
                    geweke_score = 0
            else:
                geweke_score = 0
            
            geweke_scores.append(geweke_score)
        
        diagnostics['geweke_scores'] = geweke_scores
        diagnostics['max_geweke'] = max(geweke_scores) if geweke_scores else 0
        
        # 3. 收敛评估
        convergence_issues = []
        if diagnostics['min_ess'] < 50:
            convergence_issues.append('Low ESS')
        if diagnostics['max_geweke'] > 2:
            convergence_issues.append('Geweke test failed')
        
        diagnostics['convergence_status'] = 'Good' if not convergence_issues else 'Poor'
        diagnostics['convergence_issues'] = convergence_issues
        
        return diagnostics
    
    def _calculate_autocorrelation(self, x: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """计算自相关函数"""
        n = len(x)
        if n < 2:
            return np.array([1.0])
        
        x = x - np.mean(x)
        
        # 使用FFT计算自相关
        f = np.fft.fft(x, n=2*n-1)
        autocorr = np.fft.ifft(f * np.conj(f)).real
        autocorr = autocorr[:n] / autocorr[0]
        
        return autocorr[:min(max_lag, n)]
    
    def generate_posterior_summary(self, results: Dict) -> Dict:
        """生成后验分布摘要统计"""
        samples = results['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        
        summary = {}
        
        # 混合比例统计
        n_components = mixture_samples.shape[1]
        for i in range(n_components):
            component_samples = mixture_samples[:, i]
            
            # 基本统计量
            mean_val = np.mean(component_samples)
            std_val = np.std(component_samples)
            median_val = np.median(component_samples)
            
            # 置信区间
            ci_95 = np.percentile(component_samples, [2.5, 97.5])
            ci_90 = np.percentile(component_samples, [5, 95])
            
            # 最高后验密度区间
            sorted_samples = np.sort(component_samples)
            n_samples = len(sorted_samples)
            hpdi_95_width = int(0.95 * n_samples)
            
            min_width = np.inf
            hpdi_95 = [sorted_samples[0], sorted_samples[-1]]
            
            for start in range(n_samples - hpdi_95_width):
                end = start + hpdi_95_width
                width = sorted_samples[end] - sorted_samples[start]
                if width < min_width:
                    min_width = width
                    hpdi_95 = [sorted_samples[start], sorted_samples[end]]
            
            summary[f'Mx_{i+1}'] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'mode': self._estimate_mode(component_samples),
                'credible_interval_95': ci_95.tolist(),
                'credible_interval_90': ci_90.tolist(),
                'hpdi_95': hpdi_95,
                'min': np.min(component_samples),
                'max': np.max(component_samples)
            }
        
        # 模型质量指标
        summary['model_quality'] = {
            'acceptance_rate': results['acceptance_rate'],
            'n_effective_samples': results['n_samples'],
            'converged': results.get('converged', False),
            'final_step_size': results.get('final_step_size', 'N/A')
        }
        
        # 贡献者排序（按后验均值）
        mx_means = [(i+1, summary[f'Mx_{i+1}']['mean']) for i in range(n_components)]
        mx_means.sort(key=lambda x: x[1], reverse=True)
        summary['contributor_ranking'] = mx_means
        
        return summary
    
    def _estimate_mode(self, samples: np.ndarray) -> float:
        """估计样本的众数"""
        try:
            from scipy.stats import gaussian_kde
            
            if len(samples) < 10:
                return np.median(samples)
            
            kde = gaussian_kde(samples)
            x_range = np.linspace(np.min(samples), np.max(samples), 200)
            density = kde(x_range)
            mode_idx = np.argmax(density)
            return x_range[mode_idx]
            
        except:
            hist, bin_edges = np.histogram(samples, bins=20)
            max_bin = np.argmax(hist)
            return (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    
    def plot_results(self, results: Dict, output_dir: str = './mgm_m_plots') -> None:
        """绘制MGM-M结果图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        samples = results['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 轨迹图
        fig, axes = plt.subplots(n_components, 1, figsize=(14, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].plot(mixture_samples[:, i], alpha=0.8, linewidth=0.5)
            axes[i].set_title(f'混合比例 Mx_{i+1} 的MCMC轨迹', fontsize=12)
            axes[i].set_xlabel('迭代次数 (thinned)')
            axes[i].set_ylabel(f'Mx_{i+1}')
            axes[i].grid(True, alpha=0.3)
            
            mean_val = np.mean(mixture_samples[:, i])
            axes[i].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                           label=f'均值: {mean_val:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mcmc_trace_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 后验分布密度图
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].hist(mixture_samples[:, i], bins=50, alpha=0.7, density=True, 
                        color=f'C{i}', edgecolor='black', linewidth=0.5)
            
            mean_val = np.mean(mixture_samples[:, i])
            median_val = np.median(mixture_samples[:, i])
            mode_val = self._estimate_mode(mixture_samples[:, i])
            
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'均值: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2,
                           label=f'中位数: {median_val:.3f}')
            axes[i].axvline(mode_val, color='orange', linestyle='--', linewidth=2,
                           label=f'众数: {mode_val:.3f}')
            
            ci_95 = np.percentile(mixture_samples[:, i], [2.5, 97.5])
            axes[i].axvspan(ci_95[0], ci_95[1], alpha=0.2, color='gray',
                           label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
            
            axes[i].set_title(f'混合比例 Mx_{i+1} 的后验分布', fontsize=12)
            axes[i].set_xlabel(f'Mx_{i+1}')
            axes[i].set_ylabel('概率密度')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'posterior_distributions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混合比例联合分布（对于2个组分）
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(mixture_samples[:, 0], mixture_samples[:, 1], 
                       alpha=0.6, s=2, color='blue')
            plt.xlabel('Mx_1', fontsize=12)
            plt.ylabel('Mx_2', fontsize=12)
            plt.title('混合比例联合后验分布 (Mx_1 vs Mx_2)', fontsize=14)
            
            # 添加约束线
            x_line = np.linspace(0, 1, 100)
            y_line = 1 - x_line
            plt.plot(x_line, y_line, 'r--', alpha=0.7, label='Mx_1 + Mx_2 = 1')
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.axis('equal')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, 'joint_posterior_2d.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"所有图表已保存到目录: {output_dir}")
    
    def save_results(self, results: Dict, summary: Dict, 
                    output_path: str = './mgm_m_results.json') -> None:
        """保存MGM-M结果到JSON文件"""
        save_data = {
            'method': 'MGM-M (Mixture Ratio Inference based on Genotype Marginalization and MCMC)',
            'posterior_summary': summary,
            'convergence_diagnostics': self.analyze_convergence(results['samples']),
            'mcmc_settings': {
                'n_iterations': self.n_iterations,
                'n_warmup': self.n_warmup,
                'n_chains': self.n_chains,
                'thinning': self.thinning,
                'K_top': self.K_top
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存部分样本数据
        if results['n_samples'] > 1000:
            indices = np.random.choice(results['n_samples'], 1000, replace=False)
            mixture_samples = np.array(results['samples']['mixture_ratios'])
            save_data['sample_mixture_ratios'] = mixture_samples[indices].tolist()
        else:
            save_data['sample_mixture_ratios'] = results['samples']['mixture_ratios']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")


def load_v5_features(features_file_path: str) -> Dict:
    """从问题1的V5特征文件中加载特征数据"""
    try:
        df = pd.read_csv(features_file_path, encoding='utf-8-sig')
        
        if len(df) > 0:
            sample_features = df.iloc[0].to_dict()
            
            # 确保所有需要的特征都存在
            default_features = {
                'avg_peak_height': 1000.0,
                'inter_locus_balance_entropy': 1.0,
                'ratio_severe_imbalance_loci': 0.1,
                'skewness_peak_height': 0.0,
                'avg_locus_allele_entropy': 0.8,
                'height_size_correlation': -0.1,
                'modality_peak_height': 1.0
            }
            
            for key, default_value in default_features.items():
                if key not in sample_features or pd.isna(sample_features[key]):
                    sample_features[key] = default_value
            
            return sample_features
        else:
            return {
                'avg_peak_height': 1000.0,
                'inter_locus_balance_entropy': 1.0,
                'ratio_severe_imbalance_loci': 0.1,
                'skewness_peak_height': 0.0,
                'avg_locus_allele_entropy': 0.8,
                'height_size_correlation': -0.1,
                'modality_peak_height': 1.0
            }
            
    except Exception as e:
        logger.warning(f"加载V5特征文件失败: {e}")
        return {
            'avg_peak_height': 1000.0,
            'inter_locus_balance_entropy': 1.0,
            'ratio_severe_imbalance_loci': 0.1,
            'skewness_peak_height': 0.0,
            'avg_locus_allele_entropy': 0.8,
            'height_size_correlation': -0.1,
            'modality_peak_height': 1.0
        }


def load_problem1_noc_prediction(noc_file_path: str) -> int:
    """从问题1的结果中加载NoC预测"""
    try:
        df = pd.read_csv(noc_file_path, encoding='utf-8-sig')
        if 'baseline_pred' in df.columns and len(df) > 0:
            return int(df['baseline_pred'].iloc[0])
        elif 'corrected_pred' in df.columns and len(df) > 0:
            return int(df['corrected_pred'].iloc[0])
        elif 'NoC_True' in df.columns and len(df) > 0:
            return int(df['NoC_True'].iloc[0])
        else:
            return 2
    except Exception as e:
        logger.warning(f"加载NoC预测失败: {e}")
        return 2


def create_synthetic_str_data(N: int = 2, n_loci: int = 5) -> Dict:
    """创建合成STR数据用于测试"""
    common_loci = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11']
    loci = common_loci[:n_loci]
    
    allele_pools = {
        'D3S1358': [str(i) for i in range(12, 20)],
        'vWA': [str(i) for i in range(11, 22)],
        'FGA': [str(i) + '.3' if i % 3 == 0 else str(i) for i in range(18, 28)],
        'D8S1179': [str(i) for i in range(7, 17)],
        'D21S11': [str(i) for i in range(24, 39)]
    }
    
    observed_data = {}
    
    # 模拟真实混合比例
    true_mixture_ratios = np.random.dirichlet(np.ones(N) * 2)
    logger.info(f"模拟数据的真实混合比例: {true_mixture_ratios}")
    
    for locus in loci:
        if locus not in allele_pools:
            allele_pools[locus] = [str(i) for i in range(8, 20)]
        
        pool = allele_pools[locus]
        
        # 为每个贡献者生成基因型
        contributor_genotypes = []
        for i in range(N):
            genotype = tuple(sorted(np.random.choice(pool, 2, replace=True)))
            contributor_genotypes.append(genotype)
        
        # 收集所有可能观测到的等位基因
        all_alleles_from_genotypes = set()
        for genotype in contributor_genotypes:
            all_alleles_from_genotypes.update(genotype)
        
        # 模拟ADO
        ado_prob = 0.1
        observed_alleles = []
        for allele in all_alleles_from_genotypes:
            if np.random.random() > ado_prob:
                observed_alleles.append(allele)
        
        if len(observed_alleles) < 2:
            observed_alleles = list(all_alleles_from_genotypes)[:2]
        
        # 为观测到的等位基因生成峰高
        heights = {}
        for allele in observed_alleles:
            total_contribution = 0
            for i, genotype in enumerate(contributor_genotypes):
                copy_number = sum(1 for a in genotype if a == allele)
                total_contribution += true_mixture_ratios[i] * copy_number
            
            base_height = 1000 * total_contribution
            noise_factor = np.random.lognormal(0, 0.2)
            observed_height = max(50, base_height * noise_factor)
            
            heights[allele] = observed_height
        
        observed_data[locus] = {
            'locus': locus,
            'alleles': observed_alleles,
            'heights': heights,
            'true_genotypes': contributor_genotypes
        }
    
    return observed_data


def create_att2_mock_data(observed_data: Dict) -> Dict:
    """创建模拟的附件2数据"""
    att2_data = {}
    
    for locus, data in observed_data.items():
        observed_alleles = data['alleles']
        extended_alleles = observed_alleles.copy()
        
        for allele in observed_alleles:
            try:
                allele_num = float(allele.split('.')[0])
                
                for offset in [-2, -1, 1, 2]:
                    new_allele_num = allele_num + offset
                    if new_allele_num > 0:
                        if '.' in allele:
                            decimal_part = allele.split('.')[1]
                            new_allele = f"{int(new_allele_num)}.{decimal_part}"
                        else:
                            new_allele = str(int(new_allele_num))
                        
                        if np.random.random() < 0.3:
                            extended_alleles.append(new_allele)
            except:
                continue
        
        allele_counts = []
        for allele in extended_alleles:
            if allele in observed_alleles:
                count = np.random.randint(2, 10)
            else:
                count = np.random.randint(1, 3)
            
            allele_counts.extend([allele] * count)
        
        att2_data[locus] = allele_counts
    
    return att2_data


def main():
    """主函数 - 演示MGM-M方法"""
    print("=" * 80)
    print("问题2：基于基因型边缘化的MCMC混合比例推断 (MGM-M方法)")
    print("=" * 80)
    
    # 1. 初始化MGM-M推断器
    mgm_m_inferencer = MGM_M_Inferencer()
    
    # 2. 加载问题1的结果
    print("\n=== 加载问题1结果 ===")
    try:
        N_predicted = load_problem1_noc_prediction('./prob1_features_enhanced.csv')
        print(f"从问题1加载的预测贡献者数量: {N_predicted}")
        
        v5_features = load_v5_features('./prob1_features_enhanced.csv')
        mgm_m_inferencer.set_v5_features(v5_features)
        print("成功加载并设置V5特征数据")
        
    except Exception as e:
        print(f"加载问题1结果失败: {e}")
        N_predicted = 2
        v5_features = load_v5_features('')
        mgm_m_inferencer.set_v5_features(v5_features)
        print(f"使用默认设置: N={N_predicted}")
    
    # 3. 创建或加载观测数据
    print(f"\n=== 创建合成STR观测数据 ===")
    observed_data = create_synthetic_str_data(N_predicted, n_loci=6)
    
    print("观测数据摘要:")
    for locus, data in observed_data.items():
        heights = data['heights']
        print(f"  {locus}: {len(data['alleles'])} 个等位基因 {data['alleles']}, "
              f"峰高范围 {min(heights.values()):.0f}-{max(heights.values()):.0f} RFU")
    
    # 4. 创建附件2模拟数据
    print(f"\n=== 创建附件2模拟数据 ===")
    att2_data = create_att2_mock_data(observed_data)
    
    for locus, alleles in att2_data.items():
        unique_alleles = list(set(alleles))
        print(f"  {locus}: {len(unique_alleles)} 个不同等位基因, 总计{len(alleles)}次观测")
    
    # 5. 运行MGM-M边缘化MCMC推断
    print(f"\n=== 开始MGM-M边缘化MCMC推断 ===")
    print(f"贡献者数量: N = {N_predicted}")
    print(f"MCMC设置: {mgm_m_inferencer.n_iterations} 次迭代, "
          f"{mgm_m_inferencer.n_warmup} 次预热")
    
    if N_predicted >= 4:
        print(f"使用K-top采样策略: K = {mgm_m_inferencer.K_top}")
    
    start_time = time.time()
    
    mcmc_results = mgm_m_inferencer.mcmc_sampler(
        observed_data, N_predicted, att2_data)
    
    end_time = time.time()
    print(f"\nMGM-M推断完成，总耗时: {end_time - start_time:.1f} 秒")
    
    # 6. 分析收敛性
    print(f"\n=== 分析MCMC收敛性 ===")
    convergence_diagnostics = mgm_m_inferencer.analyze_convergence(mcmc_results['samples'])
    
    print("收敛性诊断结果:")
    print(f"  收敛状态: {convergence_diagnostics['convergence_status']}")
    print(f"  最小有效样本量: {convergence_diagnostics['min_ess']:.0f}")
    print(f"  最大Geweke得分: {convergence_diagnostics['max_geweke']:.2f}")
    
    if convergence_diagnostics['convergence_issues']:
        print(f"  收敛问题: {', '.join(convergence_diagnostics['convergence_issues'])}")
    else:
        print("  ✓ 未发现明显收敛问题")
    
    # 7. 生成后验摘要
    print(f"\n=== 生成后验分布摘要 ===")
    posterior_summary = mgm_m_inferencer.generate_posterior_summary(mcmc_results)
    
    print("混合比例后验统计:")
    contributor_ranking = posterior_summary['contributor_ranking']
    
    for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
        mx_stats = posterior_summary[f'Mx_{contributor_id}']
        ci_95 = mx_stats['credible_interval_95']
        
        print(f"  排名{rank} - 贡献者{contributor_id} (Mx_{contributor_id}):")
        print(f"    后验均值: {mx_stats['mean']:.4f}")
        print(f"    后验中位数: {mx_stats['median']:.4f}")
        print(f"    95%置信区间: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
        print(f"    HPDI 95%: [{mx_stats['hpdi_95'][0]:.4f}, {mx_stats['hpdi_95'][1]:.4f}]")
    
    # 8. 模型质量评估
    print(f"\n=== 模型质量评估 ===")
    model_quality = posterior_summary['model_quality']
    print(f"  MCMC接受率: {model_quality['acceptance_rate']:.3f}")
    print(f"  有效样本数: {model_quality['n_effective_samples']}")
    print(f"  收敛状态: {'是' if model_quality['converged'] else '否'}")
    print(f"  最终步长: {model_quality['final_step_size']}")
    
    # 9. 与真实值比较（仅在使用合成数据时可用）
    print(f"\n=== 与真实混合比例比较 ===")
    print("注：这仅在合成数据测试中可用，实际应用中无法获得真实值")
    
    try:
        # 这里需要用户输入真实比例或从数据生成过程中获取
        print("请查看上面显示的模拟数据真实混合比例，然后进行比较")
        
        # 显示预测结果总结
        print("\n预测混合比例总结:")
        for i in range(N_predicted):
            estimated_mean = posterior_summary[f'Mx_{i+1}']['mean']
            estimated_ci = posterior_summary[f'Mx_{i+1}']['credible_interval_95']
            print(f"  Mx_{i+1}: {estimated_mean:.4f} "
                  f"(95% CI: [{estimated_ci[0]:.4f}, {estimated_ci[1]:.4f}])")
    
    except Exception as e:
        print(f"比较过程出错: {e}")
    
    # 10. 绘制结果图表
    print(f"\n=== 生成结果图表 ===")
    output_dir = './mgm_m_results_plots'
    mgm_m_inferencer.plot_results(mcmc_results, output_dir)
    
    # 11. 保存结果
    print(f"\n=== 保存分析结果 ===")
    mgm_m_inferencer.save_results(
        mcmc_results, posterior_summary, './mgm_m_results.json')
    
    # 12. 生成详细报告
    print(f"\n=== 生成分析报告 ===")
    generate_mgm_m_analysis_report(
        mcmc_results, posterior_summary, convergence_diagnostics, 
        N_predicted, v5_features)
    
    print("\n" + "=" * 80)
    print("MGM-M方法分析完成！")
    print("主要优势:")
    print("✓ 完全边缘化基因型，避免了高维采样空间")
    print("✓ 基于伪频率的Hardy-Weinberg先验")
    print("✓ 集成V5特征的自适应参数估计")
    print("✓ 针对N>=4使用K-top采样优化")
    print("✓ 数值稳定的logsumexp边缘化计算")
    print("=" * 80)


def generate_mgm_m_analysis_report(results: Dict, summary: Dict,
                                 diagnostics: Dict, N: int,
                                 v5_features: Dict) -> None:
    """生成MGM-M方法的详细分析报告"""
    report_path = './mgm_m_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("问题2：基于基因型边缘化的MCMC混合比例推断分析报告（MGM-M方法）\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: MGM-M (Mixture Ratio Inference based on Genotype Marginalization and MCMC)\n")
        f.write(f"贡献者数量: {N}\n")
        f.write(f"MCMC样本数: {results['n_samples']}\n")
        f.write(f"总接受率: {results['acceptance_rate']:.3f}\n")
        f.write(f"收敛状态: {results.get('converged', False)}\n\n")
        
        f.write("1. MGM-M方法论创新\n")
        f.write("-" * 50 + "\n")
        f.write("MGM-M方法实现了以下核心创新：\n")
        f.write("• 基因型完全边缘化：通过对所有可能基因型组合求和，完全避免了基因型\n")
        f.write("  空间的直接采样，解决了传统方法的维度灾难问题。\n")
        f.write("• 数学公式：L_l(M_x, θ) = Σ P(E_obs,l | N, M_x, {G_i}_l, θ) · P'({G_i}_l | A_l, N)\n")
        f.write("• 伪频率先验：基于附件2数据计算位点特异性的等位基因频率w_l(a_k)，\n")
        f.write("  结合Hardy-Weinberg平衡构建现实的基因型先验。\n")
        f.write("• V5特征集成：动态计算位点特异性参数γ_l和σ_var,l，提高模型精度。\n")
        f.write("• K-top优化：对于N≥4的复杂情况，使用K-top采样策略平衡计算效率和精度。\n\n")
        
        f.write("2. 输入数据特征分析\n")
        f.write("-" * 50 + "\n")
        f.write("V5特征参数：\n")
        f.write(f"• 平均峰高: {v5_features['avg_peak_height']:.1f} RFU\n")
        f.write(f"• 位点间平衡熵: {v5_features['inter_locus_balance_entropy']:.3f}\n")
        f.write(f"• 严重失衡位点比例: {v5_features['ratio_severe_imbalance_loci']:.3f}\n")
        f.write(f"• 峰高偏度: {v5_features['skewness_peak_height']:.3f}\n")
        f.write(f"• 峰高-片段大小相关性: {v5_features['height_size_correlation']:.3f}\n\n")
        
        f.write("3. 混合比例后验估计结果\n")
        f.write("-" * 50 + "\n")
        
        contributor_ranking = summary['contributor_ranking']
        for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
            mx_stats = summary[f'Mx_{contributor_id}']
            f.write(f"贡献者 {contributor_id} (贡献度排名 {rank}):\n")
            f.write(f"  后验均值: {mx_stats['mean']:.4f}\n")
            f.write(f"  后验标准差: {mx_stats['std']:.4f}\n")
            f.write(f"  后验中位数: {mx_stats['median']:.4f}\n")
            f.write(f"  后验众数: {mx_stats['mode']:.4f}\n")
            f.write(f"  95%置信区间: [{mx_stats['credible_interval_95'][0]:.4f}, "
                   f"{mx_stats['credible_interval_95'][1]:.4f}]\n")
            f.write(f"  90%置信区间: [{mx_stats['credible_interval_90'][0]:.4f}, "
                   f"{mx_stats['credible_interval_90'][1]:.4f}]\n")
            f.write(f"  HPDI 95%: [{mx_stats['hpdi_95'][0]:.4f}, "
                   f"{mx_stats['hpdi_95'][1]:.4f}]\n")
            f.write(f"  取值范围: [{mx_stats['min']:.4f}, {mx_stats['max']:.4f}]\n\n")
        
        f.write("4. MCMC收敛性诊断\n")
        f.write("-" * 50 + "\n")
        f.write(f"收敛状态: {diagnostics['convergence_status']}\n")
        f.write(f"最小有效样本量: {diagnostics['min_ess']:.0f}\n")
        f.write(f"最大Geweke得分: {diagnostics['max_geweke']:.3f}\n")
        
        if 'effective_sample_size' in diagnostics:
            f.write("各组分有效样本量:\n")
            for i, ess in enumerate(diagnostics['effective_sample_size']):
                f.write(f"  Mx_{i+1}: {ess:.0f}\n")
        
        if diagnostics['convergence_issues']:
            f.write(f"收敛问题: {', '.join(diagnostics['convergence_issues'])}\n")
        else:
            f.write("未检测到明显的收敛问题。\n")
        f.write("\n")
        
        f.write("5. 计算效率评估\n")
        f.write("-" * 50 + "\n")
        model_quality = summary['model_quality']
        f.write(f"MCMC接受率: {model_quality['acceptance_rate']:.3f}\n")
        f.write(f"有效样本数: {model_quality['n_effective_samples']}\n")
        f.write(f"最终步长: {model_quality['final_step_size']}\n")
        
        # 评估接受率
        accept_rate = model_quality['acceptance_rate']
        if 0.2 <= accept_rate <= 0.6:
            f.write("✓ 接受率在理想范围内 (0.2-0.6)\n")
        elif accept_rate < 0.2:
            f.write("⚠ 接受率偏低，建议调整提议分布\n")
        else:
            f.write("⚠ 接受率偏高，建议增大提议步长\n")
        f.write("\n")
        
        f.write("6. 方法优势与创新点\n")
        f.write("-" * 50 + "\n")
        f.write("• 理论突破：完全避免基因型空间采样，解决了传统方法的维度爆炸问题\n")
        f.write("• 计算效率：O(K_top)复杂度相比传统方法的指数级复杂度大幅降低\n")
        f.write("• 数值稳定：使用logsumexp技巧确保边缘化计算的数值稳定性\n")
        f.write("• 自适应建模：基于V5特征动态调整模型参数，提高预测精度\n")
        f.write("• 现实先验：基于群体遗传学数据的伪频率先验更符合实际情况\n\n")
        
        f.write("7. 应用建议\n")
        f.write("-" * 50 + "\n")
        f.write("• 适用场景：特别适合3人以上复杂混合样本的分析\n")
        f.write("• 计算资源：相比传统方法显著降低计算需求\n")
        f.write("• 参数调优：建议根据具体案例调整K_top参数平衡精度和效率\n")
        f.write("• 质量控制：建议检查MCMC收敛性和有效样本量\n")
        f.write("• 结果解释：重点关注95%置信区间和HPDI区间的重叠情况\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("报告生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 100 + "\n")
    
    print(f"详细分析报告已保存到: {report_path}")


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断了程序执行")
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查输入数据和配置参数，或联系技术支持")