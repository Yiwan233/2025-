# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题2：混合比例推断 (修正版)

代码名称: P2_MCMC_Corrected_V10
版本: V10.1 - 边缘化修正版本
日期: 2025-06-03
描述: 基于MCMC的混合比例(Mx)和降解参数(θ)推断
     正确实现了基因型的边缘化处理，严格对齐P2 V10方案
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

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class GenotypeEnumerator:
    """
    基因型枚举器 - 生成与观测等位基因兼容的所有可能基因型组合
    """
    
    def __init__(self, max_contributors: int = 5):
        self.max_contributors = max_contributors
        self.memo = {}  # 缓存已计算的结果
    
    def enumerate_valid_genotype_sets(self, observed_alleles: List[str], 
                                    N: int, K_top: int = None) -> List[List[Tuple[str, str]]]:
        """
        枚举所有与观测等位基因兼容的基因型组合
        
        Args:
            observed_alleles: 观测到的等位基因列表
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
            f.write("⚠ 接受率偏高，可能需要增大提议步长\n")
        f.write("\n")
        
        f.write("6. 结果解释与法医学意义\n")
        f.write("-" * 50 + "\n")
        
        # 分析混合模式
        main_contributor_ratio = max([summary[f'Mx_{i+1}']['mean'] for i in range(N)])
        
        if N == 2:
            if main_contributor_ratio > 0.8:
                mixture_type = "极不平衡混合样本"
                f.write("混合模式: 极不平衡混合 (主要贡献者占80%以上)\n")
                f.write("法医学意义: 次要贡献者信号较弱，可能存在等位基因缺失风险。\n")
            elif main_contributor_ratio > 0.65:
                mixture_type = "不平衡混合样本"
                f.write("混合模式: 不平衡混合 (主要贡献者占65-80%)\n")
                f.write("法医学意义: 需要谨慎评估次要贡献者的基因型。\n")
            else:
                mixture_type = "相对平衡混合样本"
                f.write("混合模式: 相对平衡混合 (主要贡献者占65%以下)\n")
                f.write("法医学意义: 两个贡献者信号相对均衡，基因型推断相对可靠。\n")
        
        elif N >= 3:
            # 计算Shannon熵来评估混合均匀性
            mx_means = [summary[f'Mx_{i+1}']['mean'] for i in range(N)]
            shannon_entropy = -sum(p * np.log(p) for p in mx_means if p > 1e-10)
            max_entropy = np.log(N)
            entropy_ratio = shannon_entropy / max_entropy
            
            if entropy_ratio > 0.9:
                f.write("混合模式: 高度均匀的多人混合\n")
                f.write("法医学意义: 各贡献者信号相对均衡，但解析复杂度高。\n")
            elif entropy_ratio > 0.7:
                f.write("混合模式: 中等均匀的多人混合\n")
                f.write("法医学意义: 存在主要和次要贡献者，需要分层分析。\n")
            else:
                f.write("混合模式: 高度不平衡的多人混合\n")
                f.write("法医学意义: 主要贡献者信号显著，次要贡献者可能难以检测。\n")
        
        f.write("\n")
        
        f.write("7. 不确定性量化\n")
        f.write("-" * 50 + "\n")
        
        for i in range(N):
            mx_stats = summary[f'Mx_{i+1}']
            ci_width = mx_stats['credible_interval_95'][1] - mx_stats['credible_interval_95'][0]
            relative_uncertainty = (mx_stats['std'] / mx_stats['mean']) * 100 if mx_stats['mean'] > 0 else 0
            
            f.write(f"Mx_{i+1} 不确定性分析:\n")
            f.write(f"  95%置信区间宽度: {ci_width:.4f}\n")
            f.write(f"  相对不确定性: {relative_uncertainty:.1f}%\n")
            
            if relative_uncertainty < 10:
                f.write("  ✓ 估计精度高\n")
            elif relative_uncertainty < 25:
                f.write("  ○ 估计精度中等\n")
            else:
                f.write("  ⚠ 估计不确定性较大\n")
            f.write("\n")
        
        f.write("8. 方法学优势与局限性\n")
        f.write("-" * 50 + "\n")
        f.write("优势:\n")
        f.write("• 边缘化处理避免了基因型采样的维度灾难\n")
        f.write("• 集成V5特征提供了数据驱动的参数估计\n")
        f.write("• K-top策略平衡了计算精度与效率\n")
        f.write("• 提供了完整的不确定性量化\n\n")
        
        f.write("局限性:\n")
        f.write("• 计算复杂度仍随贡献者数量和位点数指数增长\n")
        f.write("• 依赖于等位基因频率的准确性\n")
        f.write("• 假设等位基因之间相互独立\n")
        f.write("• 未考虑位点间的连锁不平衡\n\n")
        
        f.write("9. 建议与后续分析\n")
        f.write("-" * 50 + "\n")
        
        if diagnostics['min_ess'] < 100:
            f.write("• 建议增加MCMC迭代次数以提高统计精度\n")
        
        if results['acceptance_rate'] < 0.2 or results['acceptance_rate'] > 0.6:
            f.write("• 建议调整MCMC提议分布的步长参数\n")
        
        if N >= 4:
            f.write("• 对于高复杂度混合样本，建议结合其他证据进行综合判断\n")
        
        f.write("• 建议进行敏感性分析，评估先验假设的影响\n")
        f.write("• 可考虑进行后验预测检验验证模型拟合质量\n")
        f.write("• 建议结合基因型推断(问题3)进行综合分析\n\n")
        
        f.write("10. 技术规范与质量控制\n")
        f.write("-" * 50 + "\n")
        f.write("本分析遵循以下技术规范:\n")
        f.write("• MCMC收敛性诊断: Geweke检验、有效样本量评估\n")
        f.write("• 不确定性量化: 95%置信区间、HPDI区间\n")
        f.write("• 数值稳定性: LogSumExp技巧、自适应步长\n")
        f.write("• 结果可重现性: 随机种子控制、完整参数记录\n\n")
        
        quality_score = 0
        if diagnostics['min_ess'] >= 100:
            quality_score += 25
        if 0.2 <= results['acceptance_rate'] <= 0.6:
            quality_score += 25
        if diagnostics['convergence_status'] == 'Good':
            quality_score += 25
        if results.get('converged', False):
            quality_score += 25
        
        f.write(f"分析质量评分: {quality_score}/100\n")
        if quality_score >= 75:
            f.write("✓ 高质量分析结果\n")
        elif quality_score >= 50:
            f.write("○ 中等质量分析结果\n")
        else:
            f.write("⚠ 分析质量有待提高\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("报告生成完成。如有疑问，请参考技术文档或联系分析人员。\n")
        f.write("=" * 100 + "\n")
    
    print(f"详细分析报告已保存到: {report_path}")
    
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


class PseudoFrequencyCalculator:
    """
    伪等位基因频率计算器 - 基于附件2数据推算位点特异性频率
    """
    
    def __init__(self):
        self.frequency_cache = {}
    
    def calculate_pseudo_frequencies(self, locus: str, 
                                   all_observed_alleles_in_att2: List[str]) -> Dict[str, float]:
        """
        计算位点的伪等位基因频率
        
        Args:
            locus: 位点名称
            all_observed_alleles_in_att2: 附件2中该位点观测到的所有等位基因
            
        Returns:
            等位基因频率字典
        """
        cache_key = (locus, tuple(sorted(all_observed_alleles_in_att2)))
        if cache_key in self.frequency_cache:
            return self.frequency_cache[cache_key]
        
        # 统计等位基因出现次数
        allele_counts = {}
        for allele in all_observed_alleles_in_att2:
            allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        # 计算频率
        total_count = sum(allele_counts.values())
        frequencies = {}
        
        for allele, count in allele_counts.items():
            frequencies[allele] = count / total_count
        
        # 为未观测到但可能存在的等位基因分配最小频率
        min_freq = 1 / (2 * len(all_observed_alleles_in_att2) + len(allele_counts))
        
        # 标准化频率，确保总和为1
        freq_sum = sum(frequencies.values())
        for allele in frequencies:
            frequencies[allele] = frequencies[allele] / freq_sum * (1 - min_freq * 2)
        
        self.frequency_cache[cache_key] = frequencies
        return frequencies
    
    def calculate_genotype_prior(self, genotype: Tuple[str, str], 
                               frequencies: Dict[str, float]) -> float:
        """
        基于HWE计算基因型的先验概率
        
        Args:
            genotype: 基因型 (allele1, allele2)
            frequencies: 等位基因频率
            
        Returns:
            基因型先验概率
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
            genotype_set: 基因型组合
            frequencies: 等位基因频率
            
        Returns:
            组合先验概率
        """
        log_prior = 0.0
        for genotype in genotype_set:
            if genotype is not None:
                prior = self.calculate_genotype_prior(genotype, frequencies)
                log_prior += np.log(max(prior, 1e-10))
        
        return log_prior


class MCMCMixtureInference_Corrected:
    """
    修正版MCMC混合比例推断类 - 实现正确的基因型边缘化
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
        
        # 初始化辅助类
        self.genotype_enumerator = GenotypeEnumerator()
        self.pseudo_freq_calculator = PseudoFrequencyCalculator()
        
        # MCMC结果存储
        self.mcmc_results = {}
        self.convergence_diagnostics = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
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
        # 基础参数 - 来自V5特征的动态计算
        self.k_gamma = self.global_params.get("k_gamma", 1.0)
        self.beta = self.global_params.get("beta", 1.5)
        
        # 方差参数
        self.sigma_var_base = self.global_params.get("sigma_var_base", 0.1)
        self.c1 = self.global_params.get("c1", 0.5)
        self.c2 = self.global_params.get("c2", 0.3)
        self.c3 = self.global_params.get("c3", 0.2)
        
        # Stutter CV参数
        self.cv_hs_base = self.global_params.get("cv_hs_base", 0.25)
        self.A_s = self.global_params.get("A_s", 0.1)
        self.B_s = self.global_params.get("B_s", 0.001)
        
        # ADO参数
        self.H_50 = self.global_params.get("ado_h50", 200.0)
        self.s_ado = self.global_params.get("ado_slope", 0.01)
        
        # 降解参数
        self.k_deg_0 = self.global_params.get("k_deg_0", 0.001)
        self.size_ref = self.global_params.get("size_ref", 100.0)
        self.alpha = self.global_params.get("alpha", 1.0)
        
        # MCMC参数
        self.n_chains = self.global_params.get("n_chains", 4)
        self.n_iterations = self.global_params.get("n_iterations", 50000)
        self.n_warmup = self.global_params.get("n_warmup", 12500)
        self.thinning = self.global_params.get("thinning", 5)
        
        # K-top参数
        self.K_top = self.global_params.get("K_top", 500)
    
    def calculate_gamma_l(self, locus: str, v5_features: Dict) -> float:
        """
        基于V5特征计算位点特异性放大效率 γ_l
        
        Args:
            locus: 位点名称
            v5_features: V5特征数据
            
        Returns:
            位点放大效率
        """
        # 从V5特征获取平均峰高
        avg_height = v5_features.get('avg_peak_height', 1000.0)
        
        # 基础放大效率
        gamma_base = self.k_gamma * avg_height
        
        # 获取位点间平衡熵
        inter_locus_entropy = v5_features.get('inter_locus_balance_entropy', 1.0)
        
        # 计算权重
        L_exp = len(self.marker_params) if self.marker_params else 10
        if L_exp > 1:
            w_entropy = (1 - inter_locus_entropy / np.log(L_exp)) ** self.beta
            
            # 位点特异性调整 (简化：假设均匀分布)
            P_l = 1.0 / L_exp
            gamma_l = gamma_base * (1 + w_entropy * ((P_l * L_exp) - 1))
        else:
            gamma_l = gamma_base
            
        return max(gamma_l, 1e-3)
    
    def calculate_sigma_var_l(self, locus: str, v5_features: Dict) -> float:
        """
        基于V5特征计算位点方差参数 σ_var,l
        
        Args:
            locus: 位点名称
            v5_features: V5特征数据
            
        Returns:
            方差参数
        """
        # 从V5特征获取相关参数
        avg_height = v5_features.get('avg_peak_height', 1000.0)
        R_PHR = v5_features.get('ratio_severe_imbalance_loci', 0.0)
        gamma_1 = v5_features.get('skewness_peak_height', 0.0)
        H_a_bar = max(v5_features.get('avg_locus_allele_entropy', 1.0), 1e-6)
        
        # Sigmoid函数参数
        A_f = self.global_params.get("A_f", 1.0)
        B_f = self.global_params.get("B_f", 0.001)
        h_0f = self.global_params.get("h_0f", 1000.0)
        
        # 计算f(h_bar)
        f_h = 1 + A_f / (1 + np.exp(B_f * (avg_height - h_0f)))
        
        # 计算最终方差
        sigma_var = (self.sigma_var_base * 
                    (1 + self.c1 * R_PHR + self.c2 * abs(gamma_1) + self.c3 * (1 / H_a_bar)) * 
                    f_h)
        
        return max(sigma_var, 0.01)
    
    def calculate_degradation_factor(self, allele_size: float, 
                                   height_size_correlation: float) -> float:
        """
        计算降解因子 D_F
        
        Args:
            allele_size: 等位基因片段大小
            height_size_correlation: 峰高-片段大小相关性
            
        Returns:
            降解因子
        """
        # 计算降解系数
        k_deg = self.k_deg_0 * max(0, -height_size_correlation) ** self.alpha
        
        # 计算降解因子
        D_F = np.exp(-k_deg * max(0, allele_size - self.size_ref))
        
        return max(D_F, 1e-6)
    
    def calculate_expected_height(self, allele: str, locus: str,
                                genotype_set: List[Tuple[str, str]],
                                mixture_ratios: np.ndarray,
                                gamma_l: float, v5_features: Dict) -> float:
        """
        计算等位基因期望峰高 μ_exp,j,l
        
        Args:
            allele: 目标等位基因
            locus: 位点名称
            genotype_set: 基因型组合
            mixture_ratios: 混合比例
            gamma_l: 位点放大效率
            v5_features: V5特征数据
            
        Returns:
            期望峰高
        """
        mu_allele = 0.0
        
        # 计算直接等位基因贡献
        for i, genotype in enumerate(genotype_set):
            if genotype is not None:
                # 计算拷贝数
                C_copy = self._calculate_copy_number(allele, genotype)
                
                if C_copy > 0:
                    # 获取片段大小
                    allele_size = self._get_allele_size(allele, locus)
                    
                    # 计算降解因子
                    height_size_corr = v5_features.get('height_size_correlation', 0.0)
                    D_F = self.calculate_degradation_factor(allele_size, height_size_corr)
                    
                    # 累加贡献
                    mu_allele += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        # 计算Stutter贡献
        mu_stutter = self._calculate_stutter_contribution(
            allele, locus, genotype_set, mixture_ratios, gamma_l, v5_features)
        
        return mu_allele + mu_stutter
    
    def _calculate_copy_number(self, allele: str, genotype: Tuple[str, str]) -> float:
        """计算等位基因在基因型中的拷贝数"""
        if genotype is None:
            return 0.0
        
        count = sum(1 for gt_allele in genotype if gt_allele == allele)
        
        # 对于纯合子，应用f_homo因子
        if len(set(genotype)) == 1 and allele in genotype:
            f_homo = self.global_params.get("f_homo", 1.0)
            return 2.0 * f_homo
        
        return float(count)
    
    def _get_allele_size(self, allele: str, locus: str) -> float:
        """获取等位基因片段大小"""
        try:
            allele_num = float(allele)
            # 从marker参数获取基础信息
            marker_info = self.marker_params.get(locus, {})
            base_size = marker_info.get('base_size', 100.0)
            repeat_length = marker_info.get('L_repeat', 4.0)
            return base_size + allele_num * repeat_length
        except ValueError:
            # 对于非数字等位基因（如X, Y, OL），返回默认大小
            return 150.0
    
    def _calculate_stutter_contribution(self, target_allele: str, locus: str,
                                      genotype_set: List[Tuple[str, str]],
                                      mixture_ratios: np.ndarray, gamma_l: float,
                                      v5_features: Dict) -> float:
        """计算Stutter贡献到目标等位基因的期望峰高"""
        mu_stutter = 0.0
        
        # 获取Stutter参数
        stutter_params = self.marker_params.get(locus, {}).get('n_minus_1_Stutter', {})
        if stutter_params.get('SR_model_type') == 'N/A':
            return 0.0
        
        try:
            target_allele_num = float(target_allele)
        except ValueError:
            return 0.0
        
        # 寻找可能的亲代等位基因（n-1 Stutter）
        parent_allele_num = target_allele_num + 1
        parent_allele = str(int(parent_allele_num)) if parent_allele_num.is_integer() else str(parent_allele_num)
        
        # 计算亲代等位基因的总贡献
        mu_parent = 0.0
        for i, genotype in enumerate(genotype_set):
            if genotype is not None:
                C_copy = self._calculate_copy_number(parent_allele, genotype)
                if C_copy > 0:
                    parent_size = self._get_allele_size(parent_allele, locus)
                    height_size_corr = v5_features.get('height_size_correlation', 0.0)
                    D_F = self.calculate_degradation_factor(parent_size, height_size_corr)
                    mu_parent += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        if mu_parent > 1e-6:
            # 计算期望Stutter比率
            if stutter_params.get('SR_model_type') == 'Allele Regression':
                m = stutter_params.get('SR_m', 0.0)
                c = stutter_params.get('SR_c', 0.0)
                e_SR = m * parent_allele_num + c
            elif stutter_params.get('SR_model_type') == 'Allele Average':
                e_SR = stutter_params.get('SR_c', 0.0)
            else:
                e_SR = 0.0
            
            e_SR = max(0.0, min(e_SR, 0.5))  # 限制Stutter比率在合理范围内
            mu_stutter = e_SR * mu_parent
        
        return mu_stutter
    
    def calculate_ado_probability(self, expected_height: float) -> float:
        """计算等位基因缺失(ADO)概率"""
        if expected_height <= 0:
            return 0.99  # 如果期望峰高为0，ADO概率很高
        
        P_ado = 1.0 / (1.0 + np.exp(self.s_ado * (expected_height - self.H_50)))
        return np.clip(P_ado, 1e-6, 0.99)
    
    def calculate_marginalized_likelihood_locus(self, locus_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              v5_features: Dict,
                                              att2_data: Dict = None) -> float:
        """
        计算单个位点的边缘化似然函数 L_l(Mx, θ)
        
        这是修正后的核心函数，实现了对基因型的完全边缘化
        
        Args:
            locus_data: 位点观测数据 {'locus': str, 'alleles': List[str], 'heights': Dict}
            N: 贡献者数量
            mixture_ratios: 混合比例向量
            v5_features: V5特征数据
            att2_data: 附件2数据（用于计算伪频率）
            
        Returns:
            边缘化对数似然值
        """
        locus = locus_data['locus']
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        # 步骤1: 枚举所有与观测等位基因兼容的基因型组合
        K_top = self.K_top if N >= 4 else None
        valid_genotype_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
            observed_alleles, N, K_top)
        
        if not valid_genotype_sets:
            return -1e10  # 如果没有有效的基因型组合，返回极小值
        
        # 步骤2: 计算伪等位基因频率
        if att2_data and locus in att2_data:
            att2_alleles = att2_data[locus]
        else:
            # 如果没有附件2数据，使用观测等位基因作为基础
            att2_alleles = observed_alleles
        
        pseudo_frequencies = self.pseudo_freq_calculator.calculate_pseudo_frequencies(
            locus, att2_alleles)
        
        # 计算位点参数
        gamma_l = self.calculate_gamma_l(locus, v5_features)
        sigma_var_l = self.calculate_sigma_var_l(locus, v5_features)
        
        # 步骤3 & 4: 对所有基因型组合进行边缘化
        log_marginal_likelihood = -np.inf
        likelihood_terms = []
        
        for genotype_set in valid_genotype_sets:
            # 计算该基因型组合的伪先验概率
            log_prior = self.pseudo_freq_calculator.calculate_genotype_set_prior(
                genotype_set, pseudo_frequencies)
            
            # 计算该基因型组合的条件似然
            log_conditional_likelihood = self._calculate_conditional_likelihood_locus(
                observed_alleles, observed_heights, genotype_set, mixture_ratios,
                gamma_l, sigma_var_l, locus, v5_features)
            
            # 计算联合概率（对数空间）
            log_joint = log_prior + log_conditional_likelihood
            likelihood_terms.append(log_joint)
        
        # 使用logsumexp技巧计算边缘化似然
        if likelihood_terms:
            log_marginal_likelihood = self._logsumexp(likelihood_terms)
        
        return log_marginal_likelihood
    
    def _calculate_conditional_likelihood_locus(self, observed_alleles: List[str],
                                              observed_heights: Dict[str, float],
                                              genotype_set: List[Tuple[str, str]],
                                              mixture_ratios: np.ndarray,
                                              gamma_l: float, sigma_var_l: float,
                                              locus: str, v5_features: Dict) -> float:
        """
        计算给定基因型组合的条件似然 P(E_obs,l | N, Mx, {G_i}_l, θ)
        
        Args:
            observed_alleles: 观测等位基因
            observed_heights: 观测峰高
            genotype_set: 特定基因型组合
            mixture_ratios: 混合比例
            gamma_l: 位点放大效率
            sigma_var_l: 位点方差参数
            locus: 位点名称
            v5_features: V5特征数据
            
        Returns:
            条件对数似然值
        """
        log_likelihood = 0.0
        
        # 1. 计算观测等位基因的峰高似然
        for allele in observed_alleles:
            observed_height = observed_heights.get(allele, 0.0)
            if observed_height > 0:
                # 计算期望峰高
                mu_exp = self.calculate_expected_height(
                    allele, locus, genotype_set, mixture_ratios, gamma_l, v5_features)
                
                if mu_exp > 1e-6:
                    # 对数正态分布似然：h_j ~ LogN(ln(μ_exp) - σ²/2, σ²)
                    log_mu = np.log(mu_exp) - sigma_var_l**2 / 2
                    log_likelihood += stats.lognorm.logpdf(
                        observed_height, sigma_var_l, scale=np.exp(log_mu))
                else:
                    log_likelihood += -1e6  # 期望峰高为0但观测到峰，给予极大惩罚
        
        # 2. 计算ADO的似然（对于基因型中存在但未观测到的等位基因）
        genotype_alleles = set()
        for genotype in genotype_set:
            if genotype is not None:
                genotype_alleles.update(genotype)
        
        # 找出应该存在但未观测到的等位基因（发生了ADO）
        dropped_alleles = genotype_alleles - set(observed_alleles)
        for allele in dropped_alleles:
            # 计算该等位基因的期望峰高
            mu_exp_ado = self.calculate_expected_height(
                allele, locus, genotype_set, mixture_ratios, gamma_l, v5_features)
            
            # 计算ADO概率
            P_ado = self.calculate_ado_probability(mu_exp_ado)
            log_likelihood += np.log(max(P_ado, 1e-10))
        
        return log_likelihood
    
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
                                              v5_features: Dict,
                                              att2_data: Dict = None) -> float:
        """
        计算总的边缘化似然函数 L_total(Mx, θ)
        
        Args:
            observed_data: 所有位点的观测数据
            N: 贡献者数量
            mixture_ratios: 混合比例
            v5_features: V5特征数据
            att2_data: 附件2数据
            
        Returns:
            总对数似然值
        """
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self.calculate_marginalized_likelihood_locus(
                locus_data, N, mixture_ratios, v5_features, att2_data)
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def calculate_prior_mixture_ratios(self, mixture_ratios: np.ndarray,
                                     v5_features: Dict = None) -> float:
        """
        计算混合比例的先验概率
        
        Args:
            mixture_ratios: 混合比例
            v5_features: V5特征（用于自适应先验）
            
        Returns:
            对数先验概率
        """
        # 基础Dirichlet先验
        if v5_features:
            # 基于V5特征调整先验参数
            skewness = v5_features.get('skewness_peak_height', 0.0)
            modality = v5_features.get('modality_peak_height', 1.0)
            
            # 如果偏度较大或多峰性强，使用更分散的先验
            if abs(skewness) > 1.0 or modality > 2:
                alpha = np.ones(len(mixture_ratios)) * 0.5  # 更分散
            else:
                alpha = np.ones(len(mixture_ratios))  # 均匀先验
        else:
            alpha = np.ones(len(mixture_ratios))
        
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
        concentration = np.maximum(concentration, 0.1)  # 避免过小的参数
        
        # 从Dirichlet分布采样
        new_ratios = np.random.dirichlet(concentration)
        
        # 确保比例在合理范围内
        new_ratios = np.maximum(new_ratios, 1e-6)
        new_ratios = new_ratios / np.sum(new_ratios)  # 重新标准化
        
        return new_ratios
    
    def mcmc_sampler_corrected(self, observed_data: Dict, N: int,
                             v5_features: Dict, att2_data: Dict = None) -> Dict:
        """
        修正版MCMC采样器 - 仅采样混合比例Mx，基因型已边缘化
        
        Args:
            observed_data: 观测数据
            N: 贡献者数量
            v5_features: V5特征数据
            att2_data: 附件2数据
            
        Returns:
            MCMC采样结果
        """
        print(f"开始边缘化MCMC采样，贡献者数量: {N}")
        print(f"总迭代次数: {self.n_iterations}, 预热次数: {self.n_warmup}")
        
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
            observed_data, N, mixture_ratios, v5_features, att2_data)
        current_log_prior = self.calculate_prior_mixture_ratios(mixture_ratios, v5_features)
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
                print(f"迭代 {iteration}/{self.n_iterations}, "
                      f"当前接受率: {acceptance_rate:.3f}, "
                      f"当前似然: {current_log_likelihood:.2f}")
            
            # 提议新的混合比例
            proposed_ratios = self.propose_mixture_ratios(mixture_ratios, step_size)
            
            # 计算提议状态的概率
            proposed_log_likelihood = self.calculate_total_marginalized_likelihood(
                observed_data, N, proposed_ratios, v5_features, att2_data)
            proposed_log_prior = self.calculate_prior_mixture_ratios(proposed_ratios, v5_features)
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
                    step_size *= 0.9  # 减小步长
                elif recent_acceptance > target_acceptance + 0.05:
                    step_size *= 1.1  # 增大步长
                step_size = np.clip(step_size, 0.01, 0.2)
                
                if iteration % (adaptation_interval * 4) == 0:
                    print(f"  步长调整为: {step_size:.4f}, 最近接受率: {recent_acceptance:.3f}")
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
        
        final_acceptance_rate = n_accepted / self.n_iterations
        print(f"MCMC完成，总接受率: {final_acceptance_rate:.3f}")
        print(f"有效样本数: {len(samples['mixture_ratios'])}")
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'n_samples': len(samples['mixture_ratios']),
            'acceptance_details': acceptance_details,
            'final_step_size': step_size,
            'converged': final_acceptance_rate > 0.15 and final_acceptance_rate < 0.6
        }
    
    def analyze_convergence(self, samples: Dict) -> Dict:
        """
        分析MCMC收敛性
        
        Args:
            samples: MCMC样本
            
        Returns:
            收敛性诊断结果
        """
        diagnostics = {}
        
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 有效样本量 (ESS)
        ess_values = []
        for i in range(n_components):
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i])
            # 计算积分自相关时间
            tau_int = 1 + 2 * np.sum(autocorr[1:min(20, len(autocorr))])
            ess = n_samples / max(tau_int, 1)
            ess_values.append(max(ess, 1))
        
        diagnostics['effective_sample_size'] = ess_values
        diagnostics['min_ess'] = min(ess_values)
        
        # 2. Geweke诊断（简化版）
        # 比较前10%和后50%的样本均值
        split_point = max(n_samples // 10, 1)
        geweke_scores = []
        
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
        
        # 3. 轨迹统计
        diagnostics['trace_stats'] = {
            'mean_mixture_ratios': np.mean(mixture_samples, axis=0).tolist(),
            'std_mixture_ratios': np.std(mixture_samples, axis=0).tolist(),
            'median_mixture_ratios': np.median(mixture_samples, axis=0).tolist()
        }
        
        # 4. 收敛评估
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
    
    def generate_posterior_summary(self, results: Dict) -> Dict:
        """
        生成后验分布摘要统计
        
        Args:
            results: MCMC结果
            
        Returns:
            后验摘要
        """
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
            ci_80 = np.percentile(component_samples, [10, 90])
            
            # 最高后验密度区间（简化版）
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
                'credible_interval_80': ci_80.tolist(),
                'hpdi_95': hpdi_95,
                'min': np.min(component_samples),
                'max': np.max(component_samples)
            }
        
        # 后验概率统计
        if 'log_posterior' in samples:
            log_posterior = np.array(samples['log_posterior'])
            summary['log_posterior'] = {
                'mean': np.mean(log_posterior),
                'std': np.std(log_posterior),
                'max': np.max(log_posterior),
                'min': np.min(log_posterior)
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
        """估计样本的众数（使用核密度估计）"""
        try:
            from scipy.stats import gaussian_kde
            
            if len(samples) < 10:
                return np.median(samples)
            
            # 创建核密度估计
            kde = gaussian_kde(samples)
            
            # 在样本范围内评估密度
            x_range = np.linspace(np.min(samples), np.max(samples), 200)
            density = kde(x_range)
            
            # 找到密度最大的点
            mode_idx = np.argmax(density)
            return x_range[mode_idx]
            
        except ImportError:
            # 如果没有scipy，使用直方图方法
            hist, bin_edges = np.histogram(samples, bins=20)
            max_bin = np.argmax(hist)
            return (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
        except:
            # 出错时返回中位数
            return np.median(samples)
    
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
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
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
            
            # 添加均值线
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
            
            # 统计信息
            mean_val = np.mean(mixture_samples[:, i])
            median_val = np.median(mixture_samples[:, i])
            mode_val = self._estimate_mode(mixture_samples[:, i])
            
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'均值: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2,
                           label=f'中位数: {median_val:.3f}')
            axes[i].axvline(mode_val, color='orange', linestyle='--', linewidth=2,
                           label=f'众数: {mode_val:.3f}')
            
            # 置信区间
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
        
        # 3. 混合比例联合分布（对于2-3个组分）
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(mixture_samples[:, 0], mixture_samples[:, 1], 
                       alpha=0.6, s=2, color='blue')
            plt.xlabel('Mx_1', fontsize=12)
            plt.ylabel('Mx_2', fontsize=12)
            plt.title('混合比例联合后验分布 (Mx_1 vs Mx_2)', fontsize=14)
            
            # 添加对角线（Mx_1 + Mx_2 = 1的约束）
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
        
        elif n_components == 3:
            # 三角图（ternary plot的简化版本）
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            combinations = [(0, 1), (0, 2), (1, 2)]
            labels = [('Mx_1', 'Mx_2'), ('Mx_1', 'Mx_3'), ('Mx_2', 'Mx_3')]
            
            for i, ((idx1, idx2), (label1, label2)) in enumerate(zip(combinations, labels)):
                axes[i].scatter(mixture_samples[:, idx1], mixture_samples[:, idx2], 
                               alpha=0.6, s=2)
                axes[i].set_xlabel(label1, fontsize=12)
                axes[i].set_ylabel(label2, fontsize=12)
                axes[i].set_title(f'{label1} vs {label2}', fontsize=12)
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('三组分混合比例联合后验分布投影', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'joint_posterior_3d_projections.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. 自相关函数图
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i], max_lag=100)
            lags = np.arange(len(autocorr))
            
            axes[i].plot(lags, autocorr, linewidth=1.5)
            axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[i].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, 
                           label='0.1阈值')
            axes[i].axhline(y=-0.1, color='r', linestyle='--', alpha=0.7)
            
            # 计算有效样本量相关信息
            tau_int = 1 + 2 * np.sum(autocorr[1:min(20, len(autocorr))])
            ess = len(mixture_samples) / max(tau_int, 1)
            
            axes[i].set_title(f'Mx_{i+1} 自相关函数 (τ_int≈{tau_int:.1f}, ESS≈{ess:.0f})', 
                             fontsize=12)
            axes[i].set_xlabel('滞后 (Lag)')
            axes[i].set_ylabel('自相关')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'autocorrelation_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 接受率和似然轨迹
        if 'log_likelihood' in samples:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 似然轨迹
            log_likelihood = np.array(samples['log_likelihood'])
            ax1.plot(log_likelihood, alpha=0.8, linewidth=0.8)
            ax1.set_title('对数似然轨迹', fontsize=12)
            ax1.set_xlabel('迭代次数 (thinned)')
            ax1.set_ylabel('对数似然')
            ax1.grid(True, alpha=0.3)
            
            # 后验概率轨迹
            if 'log_posterior' in samples:
                log_posterior = np.array(samples['log_posterior'])
                ax2.plot(log_posterior, alpha=0.8, linewidth=0.8, color='green')
                ax2.set_title('对数后验概率轨迹', fontsize=12)
            else:
                ax2.set_title('对数似然轨迹（副图）', fontsize=12)
                ax2.plot(log_likelihood, alpha=0.8, linewidth=0.8, color='orange')
            
            ax2.set_xlabel('迭代次数 (thinned)')
            ax2.set_ylabel('对数后验概率')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'likelihood_trace.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. 接受率统计图
        if 'acceptance_details' in results:
            acceptance_details = results['acceptance_details']
            if len(acceptance_details) > 100:
                # 计算滑动窗口接受率
                window_size = len(acceptance_details) // 20
                acceptance_rates = []
                iterations = []
                
                for i in range(window_size, len(acceptance_details), window_size):
                    window_data = acceptance_details[i-window_size:i]
                    acceptance_rate = np.mean([d['accepted'] for d in window_data])
                    acceptance_rates.append(acceptance_rate)
                    iterations.append(i)
                
                plt.figure(figsize=(12, 6))
                plt.plot(iterations, acceptance_rates, marker='o', linewidth=2, markersize=4)
                plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, 
                           label='目标接受率 (0.4)')
                plt.xlabel('迭代次数')
                plt.ylabel('接受率')
                plt.title('MCMC接受率演化', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'acceptance_rate_evolution.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"所有图表已保存到目录: {output_dir}")
    
    def save_results(self, results: Dict, summary: Dict, 
                    output_path: str = './mcmc_results.json') -> None:
        """
        保存MCMC结果到JSON文件
        
        Args:
            results: MCMC结果
            summary: 后验摘要
            output_path: 输出文件路径
        """
        # 准备保存的数据
        save_data = {
            'posterior_summary': summary,
            'convergence_diagnostics': self.convergence_diagnostics,
            'mcmc_settings': {
                'n_iterations': self.n_iterations,
                'n_warmup': self.n_warmup,
                'n_chains': self.n_chains,
                'thinning': self.thinning,
                'K_top': self.K_top
            },
            'model_parameters': {
                'k_gamma': self.k_gamma,
                'beta': self.beta,
                'sigma_var_base': self.sigma_var_base,
                'cv_hs_base': self.cv_hs_base,
                'H_50': self.H_50,
                's_ado': self.s_ado,
                'k_deg_0': self.k_deg_0,
                'size_ref': self.size_ref,
                'alpha': self.alpha
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'Marginalized MCMC (P2 V10.1)'
        }
        
        # 保存部分样本数据（避免文件过大）
        if results['n_samples'] > 1000:
            indices = np.random.choice(results['n_samples'], 1000, replace=False)
            mixture_samples = np.array(results['samples']['mixture_ratios'])
            save_data['sample_mixture_ratios'] = mixture_samples[indices].tolist()
        else:
            save_data['sample_mixture_ratios'] = results['samples']['mixture_ratios']
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")


def load_v5_features(features_file_path: str) -> Dict:
    """
    从问题1的V5特征文件中加载特征数据
    
    Args:
        features_file_path: V5特征文件路径
        
    Returns:
        V5特征字典
    """
    try:
        df = pd.read_csv(features_file_path, encoding='utf-8-sig')
        
        # 假设我们分析第一个样本，实际使用时应该根据样本ID选择
        if len(df) > 0:
            sample_features = df.iloc[0].to_dict()
            
            # 确保所有需要的特征都存在，如果不存在则使用默认值
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
            # 如果文件为空，返回默认特征
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
        print(f"加载V5特征文件失败: {e}")
        print("使用默认V5特征值")
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
    """
    从问题1的结果中加载NoC预测
    
    Args:
        noc_file_path: 问题1结果文件路径
        
    Returns:
        预测的贡献者数量
    """
    try:
        df = pd.read_csv(noc_file_path, encoding='utf-8-sig')
        if 'baseline_pred' in df.columns and len(df) > 0:
            return int(df['baseline_pred'].iloc[0])
        elif 'corrected_pred' in df.columns and len(df) > 0:
            return int(df['corrected_pred'].iloc[0])
        elif 'NoC_True' in df.columns and len(df) > 0:
            return int(df['NoC_True'].iloc[0])
        else:
            return 2  # 默认值
    except Exception as e:
        print(f"加载NoC预测失败: {e}")
        return 2  # 默认值


def create_synthetic_str_data(N: int = 2, n_loci: int = 5) -> Dict:
    """
    创建合成STR数据用于测试边缘化MCMC
    
    Args:
        N: 贡献者数量
        n_loci: 位点数量
        
    Returns:
        合成观测数据
    """
    # 常见STR位点
    common_loci = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11', 
                   'D18S51', 'D5S818', 'D13S317', 'D7S820', 'D16S539']
    
    loci = common_loci[:n_loci]
    
    # 每个位点的等位基因库
    allele_pools = {
        'D3S1358': [str(i) for i in range(12, 20)],
        'vWA': [str(i) for i in range(11, 22)],
        'FGA': [str(i) + '.3' if i % 3 == 0 else str(i) for i in range(18, 28)],
        'D8S1179': [str(i) for i in range(7, 17)],
        'D21S11': [str(i) for i in range(24, 39)],
        'D18S51': [str(i) for i in range(9, 27)],
        'D5S818': [str(i) for i in range(7, 17)],
        'D13S317': [str(i) for i in range(8, 16)],
        'D7S820': [str(i) for i in range(6, 15)],
        'D16S539': [str(i) for i in range(5, 16)]
    }
    
    observed_data = {}
    
    # 模拟真实混合比例（用于生成数据）
    true_mixture_ratios = np.random.dirichlet(np.ones(N) * 2)  # 稍微不均匀的先验
    print(f"模拟数据的真实混合比例: {true_mixture_ratios}")
    
    for locus in loci:
        if locus not in allele_pools:
            # 对于未定义的位点，生成通用等位基因
            allele_pools[locus] = [str(i) for i in range(8, 20)]
        
        pool = allele_pools[locus]
        
        # 为每个贡献者生成基因型
        contributor_genotypes = []
        for i in range(N):
            # 随机选择两个等位基因
            genotype = tuple(sorted(np.random.choice(pool, 2, replace=True)))
            contributor_genotypes.append(genotype)
        
        # 收集所有可能观测到的等位基因
        all_alleles_from_genotypes = set()
        for genotype in contributor_genotypes:
            all_alleles_from_genotypes.update(genotype)
        
        # 模拟ADO：随机丢失一些等位基因
        ado_prob = 0.1  # 10%的ADO概率
        observed_alleles = []
        for allele in all_alleles_from_genotypes:
            if np.random.random() > ado_prob:
                observed_alleles.append(allele)
        
        # 确保至少有2个等位基因被观测到
        if len(observed_alleles) < 2:
            observed_alleles = list(all_alleles_from_genotypes)[:2]
        
        # 为观测到的等位基因生成峰高
        heights = {}
        for allele in observed_alleles:
            # 计算该等位基因的期望贡献
            total_contribution = 0
            for i, genotype in enumerate(contributor_genotypes):
                copy_number = sum(1 for a in genotype if a == allele)
                total_contribution += true_mixture_ratios[i] * copy_number
            
            # 基础峰高（模拟amplification efficiency）
            base_height = 1000 * total_contribution
            
            # 添加噪声和随机效应
            noise_factor = np.random.lognormal(0, 0.2)  # 对数正态噪声
            observed_height = max(50, base_height * noise_factor)
            
            heights[allele] = observed_height
        
        observed_data[locus] = {
            'locus': locus,
            'alleles': observed_alleles,
            'heights': heights,
            'true_genotypes': contributor_genotypes  # 仅用于验证，实际分析中不可用
        }
    
    return observed_data


def create_att2_mock_data(observed_data: Dict) -> Dict:
    """
    创建模拟的附件2数据（用于计算伪等位基因频率）
    
    Args:
        observed_data: 观测数据
        
    Returns:
        模拟的附件2数据
    """
    att2_data = {}
    
    for locus, data in observed_data.items():
        # 扩展观测等位基因列表，模拟在更大数据集中观测到的等位基因
        observed_alleles = data['alleles']
        
        # 添加一些相邻的等位基因（模拟population diversity）
        extended_alleles = observed_alleles.copy()
        
        for allele in observed_alleles:
            try:
                allele_num = float(allele.split('.')[0])  # 处理如 "21.3" 的情况
                
                # 添加相邻等位基因
                for offset in [-2, -1, 1, 2]:
                    new_allele_num = allele_num + offset
                    if new_allele_num > 0:
                        if '.' in allele:  # 保持小数格式
                            decimal_part = allele.split('.')[1]
                            new_allele = f"{int(new_allele_num)}.{decimal_part}"
                        else:
                            new_allele = str(int(new_allele_num))
                        
                        # 以一定概率添加
                        if np.random.random() < 0.3:
                            extended_alleles.append(new_allele)
            except:
                # 对于无法解析的等位基因，跳过
                continue
        
        # 模拟每个等位基因在附件2中出现的次数
        allele_counts = []
        for allele in extended_alleles:
            # 观测到的等位基因有更高的出现频率
            if allele in observed_alleles:
                count = np.random.randint(2, 10)
            else:
                count = np.random.randint(1, 3)
            
            allele_counts.extend([allele] * count)
        
        att2_data[locus] = allele_counts
    
    return att2_data


def main():
    """
    主函数 - 演示修正版问题2的MCMC混合比例推断
    """
    print("=" * 80)
    print("问题2：MCMC混合比例推断 (修正版 V10.1 - 边缘化实现)")
    print("=" * 80)
    
    # 1. 初始化修正版MCMC推断器
    config_path = './config_params.json'
    mcmc_inferencer = MCMCMixtureInference_Corrected(config_path)
    
    # 2. 加载问题1的结果
    print("\n=== 加载问题1结果 ===")
    try:
        # 尝试加载NoC预测
        N_predicted = load_problem1_noc_prediction('./prob1_features_v2.9.csv')
        print(f"从问题1加载的预测贡献者数量: {N_predicted}")
        
        # 尝试加载V5特征
        v5_features = load_v5_features('./prob1_features_enhanced.csv')
        print("成功加载V5特征数据")
        print(f"主要特征: 平均峰高={v5_features['avg_peak_height']:.1f}, "
              f"位点间熵={v5_features['inter_locus_balance_entropy']:.3f}")
        
    except Exception as e:
        print(f"加载问题1结果失败: {e}")
        N_predicted = 3
        v5_features = load_v5_features('')  # 使用默认值
        print(f"使用默认设置: N={N_predicted}, 默认V5特征")
    
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
    
    # 5. 运行边缘化MCMC推断
    print(f"\n=== 开始边缘化MCMC推断 ===")
    print(f"贡献者数量: N = {N_predicted}")
    print(f"MCMC设置: {mcmc_inferencer.n_iterations} 次迭代, "
          f"{mcmc_inferencer.n_warmup} 次预热, "
          f"thinning = {mcmc_inferencer.thinning}")
    
    if N_predicted >= 4:
        print(f"使用K-top采样策略: K = {mcmc_inferencer.K_top}")
    
    start_time = time.time()
    
    mcmc_results = mcmc_inferencer.mcmc_sampler_corrected(
        observed_data, N_predicted, v5_features, att2_data)
    
    end_time = time.time()
    print(f"\nMCMC推断完成，总耗时: {end_time - start_time:.1f} 秒")
    
    # 6. 分析收敛性
    print(f"\n=== 分析MCMC收敛性 ===")
    convergence_diagnostics = mcmc_inferencer.analyze_convergence(mcmc_results['samples'])
    mcmc_inferencer.convergence_diagnostics = convergence_diagnostics
    
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
    posterior_summary = mcmc_inferencer.generate_posterior_summary(mcmc_results)
    
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
    
    # 从合成数据中获取真实比例（如果可用）
    if 'true_mixture_ratios' in locals():
        print("真实混合比例 vs 后验估计:")
        true_ratios = eval(input("请输入生成数据时显示的真实混合比例（格式如 [0.6, 0.4]）: ") or "[0.5, 0.5]")
        
        for i in range(N_predicted):
            true_val = true_ratios[i] if i < len(true_ratios) else 0
            estimated_mean = posterior_summary[f'Mx_{i+1}']['mean']
            error = abs(estimated_mean - true_val)
            
            print(f"  Mx_{i+1}: 真实={true_val:.4f}, 估计={estimated_mean:.4f}, "
                  f"误差={error:.4f}")
    
    # 10. 绘制结果图表
    print(f"\n=== 生成结果图表 ===")
    output_dir = './problem2_corrected_plots'
    mcmc_inferencer.plot_results(mcmc_results, output_dir)
    
    # 11. 保存结果
    print(f"\n=== 保存分析结果 ===")
    mcmc_inferencer.save_results(
        mcmc_results, posterior_summary, './problem2_corrected_mcmc_results.json')
    
    # 12. 生成详细报告
    print(f"\n=== 生成分析报告 ===")
    generate_detailed_analysis_report(
        mcmc_results, posterior_summary, convergence_diagnostics, 
        N_predicted, v5_features)
    
    print("\n" + "=" * 80)
    print("问题2修正版分析完成！")
    print("主要改进:")
    print("✓ 实现了基因型的完全边缘化处理")
    print("✓ 不再直接采样基因型，避免了维度灾难")
    print("✓ 基于附件2数据计算伪等位基因频率")
    print("✓ 集成V5特征计算位点特异性参数")
    print("✓ 针对N>=4使用K-top采样优化")
    print("=" * 80)


def generate_detailed_analysis_report(results: Dict, summary: Dict,
                                     diagnostics: Dict, N: int,
                                     v5_features: Dict) -> None:
    """
    生成详细的分析报告
    
    Args:
        results: MCMC结果
        summary: 后验摘要
        diagnostics: 收敛性诊断
        N: 贡献者数量
        v5_features: V5特征数据
    """
    report_path = './problem2_corrected_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("问题2：STR混合样本的混合比例推断分析报告（修正版 V10.1）\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"方法: 边缘化MCMC (Marginalized MCMC)\n")
        f.write(f"贡献者数量: {N}\n")
        f.write(f"MCMC样本数: {results['n_samples']}\n")
        f.write(f"总接受率: {results['acceptance_rate']:.3f}\n")
        f.write(f"收敛状态: {results.get('converged', False)}\n\n")
        
        f.write("1. 方法论创新\n")
        f.write("-" * 50 + "\n")
        f.write("本分析实现了以下关键改进：\n")
        f.write("• 基因型完全边缘化：不直接采样基因型，而是在似然函数中对所有可能的\n")
        f.write("  基因型组合进行积分，避免了高维度基因型空间的采样困难。\n")
        f.write("• 位点特异性参数：基于V5特征动态计算每个位点的放大效率和方差参数。\n")
        f.write("• K-top采样优化：对于N≥4的情况，使用K-top策略限制基因型枚举的\n")
        f.write("  计算复杂度。\n")
        f.write("• 伪频率先验：基于附件2数据计算位点特异性的等位基因频率。\n\n")
        
        f.write("2. 输入数据特征\n")
        f.write("-" * 50 + "\n")
        f.write("V5特征摘要：\n")
        f.write(f"• 平均峰高: {v5_features['avg_peak_height']:.1f} RFU\n")
        f.write(f"• 位点间平衡熵: {v5_features['inter_locus_balance_entropy']:.3f}\n")
        f.write(f"• 严重失衡位点比例: {v5_features['ratio_severe_imbalance_loci']:.3f}\n")
        f.write(f"• 峰高偏度: {v5_features['skewness_peak_height']:.3f}\n")
        f.write(f"• 峰高-片段大小相关性: {v5_features['height_size_correlation']:.3f}\n\n")
        
        f.write("3. 混合比例后验估计\n")
        f.write("-" * 50 + "\n")
        
        contributor_ranking = summary['contributor_ranking']
        for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
            mx_stats = summary[f'Mx_{contributor_id}']
            f.write(f"贡献者 {contributor_id} (排名 {rank}):\n")
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
        
        f.write("4. 模型收敛性诊断\n")
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
        
        f.write("5. 模型质量评估\n")
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
            f.write("⚠ 接受率偏低，可能需要减小提议步长\n")
        else:
            f.write("⚠ 接受率偏高，可能需要增大提议步长\n")
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("报告生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 100 + "\n")
        
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断了程序执行")
    # except Exception as e:
    #     print(f"\n程序执行过程中发生错误: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     print("\n请检查输入数据和配置参数，或联系技术支持")# 对于N>=4，使用K-top采样策略
    #     valid_sets = self._enumerate_k_top_combinations(observed_alleles, N, K_top)
        
    #     self.memo[cache_key] = valid_sets
    #     return valid_sets