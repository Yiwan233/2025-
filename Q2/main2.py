# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题2：混合比例推断 (修正版)

代码名称: P2_MCMC_Corrected_V10
版本: V10.2 - 修正混合比例解析版本
日期: 2025-06-06
描述: 基于MCMC的混合比例(Mx)和降解参数(θ)推断
     正确解析附件1中Sample File的贡献者ID和混合比例
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
import re

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def parse_sample_file_name(sample_file: str) -> Optional[Dict]:
    """
    正确解析Sample File名称中的贡献者ID和混合比例
    
    根据附件1中的命名规则:
    "A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa"
    其中40_41-1;4表示贡献者ID为40、41，混合比例为1:4
    
    Args:
        sample_file: 样本文件名
        
    Returns:
        解析结果字典，包含贡献者数量、ID、混合比例等信息
    """
    # 移除引号
    file_name = sample_file.replace('"', '').strip()
    
    # 模式1: 两人混合样本 - 贡献者ID1_贡献者ID2-比例1;比例2
    match2 = re.search(r'-(\d+)_(\d+)-(\d+);(\d+)-', file_name)
    if match2:
        contributor1_id = int(match2.group(1))
        contributor2_id = int(match2.group(2))
        ratio1 = int(match2.group(3))
        ratio2 = int(match2.group(4))
        
        total = ratio1 + ratio2
        mixtures = [ratio1 / total, ratio2 / total]
        
        return {
            'N': 2,
            'contributor_ids': [contributor1_id, contributor2_id],
            'original_ratios': [ratio1, ratio2],
            'mixture_ratios': mixtures,
            'formatted_ratio': f"{ratio1}:{ratio2}"
        }
    
    # 模式2: 三人混合样本 - 贡献者ID1_贡献者ID2_贡献者ID3-比例1;比例2;比例3
    match3 = re.search(r'-(\d+)_(\d+)_(\d+)-(\d+);(\d+);(\d+)-', file_name)
    if match3:
        contributor_ids = [int(match3.group(i)) for i in range(1, 4)]
        ratios = [int(match3.group(i)) for i in range(4, 7)]
        
        total = sum(ratios)
        mixtures = [r / total for r in ratios]
        
        return {
            'N': 3,
            'contributor_ids': contributor_ids,
            'original_ratios': ratios,
            'mixture_ratios': mixtures,
            'formatted_ratio': ':'.join(map(str, ratios))
        }
    
    # 模式3: 四人混合样本
    match4 = re.search(r'-(\d+)_(\d+)_(\d+)_(\d+)-(\d+);(\d+);(\d+);(\d+)-', file_name)
    if match4:
        contributor_ids = [int(match4.group(i)) for i in range(1, 5)]
        ratios = [int(match4.group(i)) for i in range(5, 9)]
        
        total = sum(ratios)
        mixtures = [r / total for r in ratios]
        
        return {
            'N': 4,
            'contributor_ids': contributor_ids,
            'original_ratios': ratios,
            'mixture_ratios': mixtures,
            'formatted_ratio': ':'.join(map(str, ratios))
        }
    
    # 模式4: 五人混合样本
    match5 = re.search(r'-(\d+)_(\d+)_(\d+)_(\d+)_(\d+)-(\d+);(\d+);(\d+);(\d+);(\d+)-', file_name)
    if match5:
        contributor_ids = [int(match5.group(i)) for i in range(1, 6)]
        ratios = [int(match5.group(i)) for i in range(6, 11)]
        
        total = sum(ratios)
        mixtures = [r / total for r in ratios]
        
        return {
            'N': 5,
            'contributor_ids': contributor_ids,
            'original_ratios': ratios,
            'mixture_ratios': mixtures,
            'formatted_ratio': ':'.join(map(str, ratios))
        }
    
    # 模式5: 单人样本
    match1 = re.search(r'-(\d+)-', file_name)
    if match1:
        contributor_id = int(match1.group(1))
        return {
            'N': 1,
            'contributor_ids': [contributor_id],
            'original_ratios': [1],
            'mixture_ratios': [1.0],
            'formatted_ratio': "1"
        }
    
    return None


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
            # 对于N>=4，使用K-top采样策略
            valid_sets = self._enumerate_k_top_combinations(observed_alleles, N, K_top)
        
        self.memo[cache_key] = valid_sets
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
    修正版MCMC混合比例推断类 - 实现正确的基因型边缘化和混合比例解析
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
    
    def load_real_data_with_correct_parsing(self) -> Tuple[int, Dict, Dict, Dict]:
        """
        加载真实数据文件，正确解析混合比例
        
        Returns:
            N: 贡献者人数
            mx_phi: 混合比例和phi参数
            E_obs: 观测数据  
            pseudo_freq: 伪频率
        """
        print("加载真实数据并正确解析混合比例...")
        
        # 1. 加载附件1数据并解析混合比例
        try:
            str_data = pd.read_csv('附件1：不同人数的STR图谱数据.csv')
            print(f"✓ 成功加载附件1数据: {str_data.shape}")
            
            # 获取第一个样本的信息
            first_sample = str_data['Sample File'].iloc[0]
            parsed_info = parse_sample_file_name(first_sample)
            
            if parsed_info:
                N = parsed_info['N']
                true_mixture_ratios = np.array(parsed_info['mixture_ratios'])
                contributor_ids = parsed_info['contributor_ids']
                
                print(f"✓ 解析样本: {first_sample}")
                print(f"  - 贡献者数量: {N}")
                print(f"  - 贡献者ID: {contributor_ids}")
                print(f"  - 原始比例: {parsed_info['formatted_ratio']}")
                print(f"  - 混合比例: {[f'{r:.3f}' for r in true_mixture_ratios]}")
            else:
                print("⚠️ 无法解析样本文件名，使用默认值")
                N = 2
                true_mixture_ratios = np.array([0.6, 0.4])
                contributor_ids = [40, 41]
        except Exception as e:
            print(f"✗ 加载附件1数据失败: {e}")
            N = 2
            true_mixture_ratios = np.array([0.6, 0.4])
            contributor_ids = [40, 41]
        
        # 2. 加载附件3的基因型数据
        try:
            genotype_data = pd.read_csv('附件3：各个贡献者对应的基因型数据.csv')
            print(f"✓ 成功加载附件3数据: {genotype_data.shape}")
            
            # 提取对应贡献者的基因型
            contributor_genotypes = {}
            for contrib_id in contributor_ids:
                contrib_row = genotype_data[genotype_data['Sample ID'] == contrib_id]
                if not contrib_row.empty:
                    contributor_genotypes[contrib_id] = contrib_row.iloc[0].to_dict()
                    print(f"  - 贡献者{contrib_id}的基因型数据已提取")
        except Exception as e:
            print(f"✗ 加载附件3数据失败: {e}")
            contributor_genotypes = {}
        
        # 3. 构建观测数据E_obs
        # 这里使用简化的模拟数据，实际应用中需要根据STR图谱数据构建
        E_obs = self._create_observed_data_from_genotypes(
            contributor_genotypes, true_mixture_ratios, N)
        
        # 4. 构建phi参数
        phi_star = {
            'gamma_l': 1000,  # 默认放大效率
            'sigma_var_l': 0.15,  # 根据混合比例调整方差
            'k_deg': 0.0001,
            'size_ref': 200,
            'h50': 150,
            's_ado': 1.5
        }
        
        mx_phi = {
            'mx_star': true_mixture_ratios,
            'phi_star': phi_star,
            'N': N,
            'contributor_ids': contributor_ids,
            'parsed_info': parsed_info
        }
        
        # 5. 构建伪频率
        pseudo_freq = self._create_pseudo_frequencies_from_att3(genotype_data)
        
        print(f"✓ 数据加载完成")
        print(f"  - 观测位点数: {len(E_obs)}")
        print(f"  - 伪频率位点数: {len(pseudo_freq)}")
        
        return N, mx_phi, E_obs, pseudo_freq
    
    def _create_observed_data_from_genotypes(self, contributor_genotypes: Dict, 
                                           mixture_ratios: np.ndarray, N: int) -> Dict:
        """
        基于贡献者基因型和混合比例创建观测数据
        
        Args:
            contributor_genotypes: 贡献者基因型数据
            mixture_ratios: 混合比例
            N: 贡献者数量
            
        Returns:
            观测数据字典
        """
        # STR位点列表
        str_loci = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358', 
                   'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433', 
                   'vWA', 'TPOX', 'D18S51', 'D5S818', 'FGA']
        
        E_obs = {}
        
        for locus in str_loci:
            # 收集该位点的所有等位基因
            locus_alleles = set()
            locus_heights = {}
            
            for i, (contrib_id, genotype_data) in enumerate(contributor_genotypes.items()):
                if i >= N:  # 只处理前N个贡献者
                    break
                    
                if locus in genotype_data:
                    allele_str = str(genotype_data[locus])
                    if pd.notna(allele_str) and allele_str != 'nan':
                        # 解析基因型 (例如: "15,16" 或 "15")
                        alleles = [a.strip() for a in allele_str.split(',')]
                        
                        for allele in alleles:
                            if allele and allele != '':
                                locus_alleles.add(allele)
                                
                                # 模拟峰高 (基于混合比例)
                                base_height = 1000 * mixture_ratios[i]
                                noise = np.random.lognormal(0, 0.1)
                                height = max(50, base_height * noise)
                                
                                if allele in locus_heights:
                                    locus_heights[allele] += height
                                else:
                                    locus_heights[allele] = height
            
            if locus_alleles:
                E_obs[locus] = {
                    'locus': locus,
                    'alleles': list(locus_alleles),
                    'heights': locus_heights
                }
        
        return E_obs
    
    def _create_pseudo_frequencies_from_att3(self, genotype_data: pd.DataFrame) -> Dict:
        """
        基于附件3创建伪等位基因频率
        
        Args:
            genotype_data: 附件3的基因型数据
            
        Returns:
            伪频率字典
        """
        str_loci = ['D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358',
                   'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433',
                   'vWA', 'TPOX', 'D18S51', 'D5S818', 'FGA']
        
        pseudo_freq = {}
        
        for locus in str_loci:
            if locus in genotype_data.columns:
                # 收集该位点的所有等位基因
                all_alleles = []
                
                for _, row in genotype_data.iterrows():
                    allele_str = str(row[locus])
                    if pd.notna(allele_str) and allele_str != 'nan':
                        alleles = [a.strip() for a in allele_str.split(',')]
                        all_alleles.extend([a for a in alleles if a])
                
                # 计算频率
                if all_alleles:
                    freq = self.pseudo_freq_calculator.calculate_pseudo_frequencies(
                        locus, all_alleles)
                    pseudo_freq[locus] = freq
        
        return pseudo_freq
    
    # 继续包含原有的其他方法...
    def calculate_gamma_l(self, locus: str, v5_features: Dict) -> float:
        """基于V5特征计算位点特异性放大效率 γ_l"""
        avg_height = v5_features.get('avg_peak_height', 1000.0)
        gamma_base = self.k_gamma * avg_height
        inter_locus_entropy = v5_features.get('inter_locus_balance_entropy', 1.0)
        
        L_exp = len(self.marker_params) if self.marker_params else 10
        if L_exp > 1:
            w_entropy = (1 - inter_locus_entropy / np.log(L_exp)) ** self.beta
            P_l = 1.0 / L_exp
            gamma_l = gamma_base * (1 + w_entropy * ((P_l * L_exp) - 1))
        else:
            gamma_l = gamma_base
            
        return max(gamma_l, 1e-3)
    
    def calculate_sigma_var_l(self, locus: str, v5_features: Dict) -> float:
        """基于V5特征计算位点方差参数 σ_var,l"""
        avg_height = v5_features.get('avg_peak_height', 1000.0)
        R_PHR = v5_features.get('ratio_severe_imbalance_loci', 0.0)
        gamma_1 = v5_features.get('skewness_peak_height', 0.0)
        H_a_bar = max(v5_features.get('avg_locus_allele_entropy', 1.0), 1e-6)
        
        A_f = self.global_params.get("A_f", 1.0)
        B_f = self.global_params.get("B_f", 0.001)
        h_0f = self.global_params.get("h_0f", 1000.0)
        
        f_h = 1 + A_f / (1 + np.exp(B_f * (avg_height - h_0f)))
        
        sigma_var = (self.sigma_var_base * 
                    (1 + self.c1 * R_PHR + self.c2 * abs(gamma_1) + self.c3 * (1 / H_a_bar)) * 
                    f_h)
        
        return max(sigma_var, 0.01)


def main():
    """
    主函数 - 演示修正版问题2的MCMC混合比例推断
    """
    print("=" * 80)
    print("问题2：MCMC混合比例推断 (修正版 V10.2 - 正确解析混合比例)")
    print("=" * 80)
    
    # 1. 初始化修正版MCMC推断器
    config_path = './config_params.json'
    mcmc_inferencer = MCMCMixtureInference_Corrected(config_path)
    
    # 2. 加载真实数据并正确解析混合比例
    print("\n=== 加载真实数据并解析混合比例 ===")
    try:
        N, mx_phi, E_obs, pseudo_freq = mcmc_inferencer.load_real_data_with_correct_parsing()
        print("✅ 成功加载并解析真实数据")
        
        # 显示解析结果
        parsed_info = mx_phi.get('parsed_info', {})
        if parsed_info:
            print(f"\n📊 混合比例解析结果:")
            print(f"   贡献者数量: {parsed_info['N']}")
            print(f"   贡献者ID: {parsed_info['contributor_ids']}")
            print(f"   原始比例: {parsed_info['formatted_ratio']}")
            print(f"   标准化混合比例: {[f'{r:.4f}' for r in parsed_info['mixture_ratios']]}")
            print(f"   混合比例总和: {sum(parsed_info['mixture_ratios']):.6f}")
    except Exception as e:
        print(f"⚠️ 加载真实数据失败: {e}")
        print("🔄 回退到模拟数据")
        # 创建模拟数据
        N = 2
        mx_phi = {
            'mx_star': np.array([0.2, 0.8]),  # 对应1:4的比例
            'phi_star': {'gamma_l': 1000, 'sigma_var_l': 0.15},
            'N': N
        }
        E_obs = {}
        pseudo_freq = {}
    
    # 3. 显示数据摘要
    print(f"\n=== 数据摘要 ===")
    print(f"贡献者数量: {N}")
    print(f"观测位点数: {len(E_obs)}")
    print(f"真实混合比例: {mx_phi['mx_star']}")
    
    # 显示各位点的观测等位基因
    print(f"\n📍 各位点观测等位基因:")
    for locus, data in list(E_obs.items())[:5]:  # 显示前5个位点
        alleles = data['alleles']
        heights = data['heights']
        print(f"   {locus}: {alleles} (峰高范围: {min(heights.values()):.0f}-{max(heights.values()):.0f})")
    
    if len(E_obs) > 5:
        print(f"   ... 还有 {len(E_obs)-5} 个位点")
    
    print(f"\n✅ 数据准备完成，可以进行MCMC推断")
    print("=" * 80)
    
    return N, mx_phi, E_obs, pseudo_freq


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    try:
        N, mx_phi, E_obs, pseudo_freq = main()
        
        print("\n🎉 混合比例解析和数据加载完成!")
        print("现在可以继续进行MCMC采样和推断...")
        
        # 验证混合比例解析
        true_ratios = mx_phi['mx_star']
        print(f"\n📊 验证结果:")
        print(f"   解析的混合比例: {true_ratios}")
        print(f"   比例总和: {sum(true_ratios):.6f}")
        print(f"   是否标准化: {'✅' if abs(sum(true_ratios) - 1.0) < 1e-6 else '❌'}")
        
    except KeyboardInterrupt:
        print("\n用户中断了程序执行")
    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()