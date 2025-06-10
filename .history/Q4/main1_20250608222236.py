# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题四：基于UPG-M的STR混合样本降噪系统

版本: V11.0 - Unified Probabilistic Genotyping with MCMC Denoising (UPG-M)
日期: 2025-06-08
描述: 统一概率基因分型，同时推断NoC、Mx、基因型和降噪
核心创新:
1. 整体性(Holism)：同时推断NoC、混合比例、基因型和噪声参数
2. Trans-dimensional MCMC (RJMCMC)：动态调整贡献者人数
3. 基于V5特征的自适应降噪策略
4. Stutter和ADO建模的综合噪声处理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import loggamma, gammaln
from scipy.signal import find_peaks, savgol_filter
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from collections import defaultdict
import itertools
from math import comb
import logging
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("问题四：基于UPG-M的STR混合样本统一概率基因分型降噪系统")
print("Unified Probabilistic Genotyping with MCMC-based Noise Reduction")
print("=" * 80)

# =====================
# 1. 核心配置与常量
# =====================
class Config:
    """系统配置类"""
    # 文件路径配置
    DATA_DIR = './'
    ATTACHMENT4_PATH = os.path.join(DATA_DIR, '附件4：混合STR图谱数据.csv')
    ATTACHMENT1_PATH = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
    ATTACHMENT3_PATH = os.path.join(DATA_DIR, '附件3：各个贡献者对应的基因型数据.csv')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'q4_upg_denoising_results')
    Q1_MODEL_PATH = os.path.join(DATA_DIR, 'noc_optimized_random_forest_model.pkl')
    
    # STR分析参数
    HEIGHT_THRESHOLD = 50
    SATURATION_THRESHOLD = 30000
    CTA_THRESHOLD = 0.5
    PHR_IMBALANCE_THRESHOLD = 0.6
    
    # 降噪参数
    NOISE_THRESHOLD = 100  # 噪声阈值
    STUTTER_RATIO_RANGE = (0.02, 0.15)  # Stutter比例范围
    ADO_PROBABILITY_THRESHOLD = 0.3  # ADO概率阈值
    
    # UPG-M MCMC参数
    N_ITERATIONS = 20000
    N_WARMUP = 8000
    N_CHAINS = 4
    THINNING = 5
    MAX_NOC = 5  # 最大贡献者数量
    
    # 模型参数
    RANDOM_STATE = 42
    N_JOBS = -1

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# =====================
# 2. 噪声检测与表征模块
# =====================
class NoiseDetector:
    """噪声检测器，识别和表征各种类型的噪声"""
    
    def __init__(self):
        self.noise_models = {}
        self.baseline_estimators = {}
        logger.info("噪声检测器初始化完成")
    
    def detect_baseline_noise(self, heights: np.ndarray, window_size: int = 5) -> Dict:
        """检测基线噪声水平"""
        try:
            # 使用移动窗口估计基线噪声
            baseline_noise = np.convolve(heights, np.ones(window_size)/window_size, mode='same')
            
            # 计算噪声统计量
            noise_level = np.std(heights - baseline_noise)
            signal_to_noise = np.mean(heights) / max(noise_level, 1e-6)
            
            # 检测异常峰
            z_scores = np.abs((heights - np.mean(heights)) / max(np.std(heights), 1e-6))
            outlier_peaks = np.sum(z_scores > 3)
            
            return {
                'baseline_noise_level': noise_level,
                'signal_to_noise_ratio': signal_to_noise,
                'outlier_peaks_count': outlier_peaks,
                'baseline_estimate': baseline_noise
            }
        except Exception as e:
            logger.warning(f"基线噪声检测失败: {e}")
            return {
                'baseline_noise_level': 0.0,
                'signal_to_noise_ratio': 1.0,
                'outlier_peaks_count': 0,
                'baseline_estimate': np.zeros_like(heights)
            }
    
    def detect_stutter_artifacts(self, locus_data: Dict) -> Dict:
        """检测Stutter伪影"""
        alleles = locus_data['alleles']
        heights = locus_data['heights']
        sizes = locus_data.get('sizes', {})
        
        stutter_candidates = []
        parent_alleles = []
        
        # 按峰高排序等位基因
        sorted_alleles = sorted(alleles, key=lambda x: heights.get(x, 0), reverse=True)
        
        for i, allele in enumerate(sorted_alleles):
            try:
                allele_num = float(allele)
                parent_num = allele_num + 1  # n-1 stutter
                parent_allele = str(int(parent_num)) if parent_num.is_integer() else str(parent_num)
                
                if parent_allele in heights:
                    parent_height = heights[parent_allele]
                    current_height = heights[allele]
                    
                    # 计算stutter比例
                    stutter_ratio = current_height / max(parent_height, 1e-6)
                    
                    # 判断是否为stutter
                    if config.STUTTER_RATIO_RANGE[0] <= stutter_ratio <= config.STUTTER_RATIO_RANGE[1]:
                        stutter_candidates.append({
                            'stutter_allele': allele,
                            'parent_allele': parent_allele,
                            'stutter_ratio': stutter_ratio,
                            'stutter_height': current_height,
                            'parent_height': parent_height
                        })
                        parent_alleles.append(parent_allele)
                
            except (ValueError, TypeError):
                continue
        
        return {
            'stutter_candidates': stutter_candidates,
            'stutter_count': len(stutter_candidates),
            'parent_alleles': parent_alleles,
            'estimated_stutter_ratio': np.mean([s['stutter_ratio'] for s in stutter_candidates]) if stutter_candidates else 0.0
        }
    
    def detect_allele_dropout(self, locus_data: Dict, expected_alleles: List[str] = None) -> Dict:
        """检测等位基因缺失（ADO）"""
        observed_alleles = set(locus_data['alleles'])
        heights = locus_data['heights']
        
        # 估计可能的等位基因缺失
        dropout_candidates = []
        
        if expected_alleles:
            expected_set = set(expected_alleles)
            missing_alleles = expected_set - observed_alleles
            
            for missing_allele in missing_alleles:
                # 估计缺失等位基因的期望高度
                neighbor_heights = []
                
                try:
                    missing_num = float(missing_allele)
                    for allele in observed_alleles:
                        try:
                            allele_num = float(allele)
                            if abs(allele_num - missing_num) <= 2.0:  # 附近的等位基因
                                neighbor_heights.append(heights[allele])
                        except (ValueError, TypeError):
                            continue
                    
                    if neighbor_heights:
                        expected_height = np.mean(neighbor_heights)
                        dropout_probability = 1.0 / (1.0 + np.exp(0.01 * (expected_height - 200)))
                        
                        dropout_candidates.append({
                            'missing_allele': missing_allele,
                            'expected_height': expected_height,
                            'dropout_probability': dropout_probability
                        })
                
                except (ValueError, TypeError):
                    continue
        
        # 检测低峰高可能的ADO
        low_height_alleles = []
        for allele, height in heights.items():
            if height < config.HEIGHT_THRESHOLD * 2:
                ado_prob = 1.0 / (1.0 + np.exp(0.01 * (height - 200)))
                low_height_alleles.append({
                    'allele': allele,
                    'height': height,
                    'ado_probability': ado_prob
                })
        
        return {
            'dropout_candidates': dropout_candidates,
            'low_height_alleles': low_height_alleles,
            'estimated_ado_rate': len(dropout_candidates) / max(len(expected_alleles or observed_alleles), 1)
        }
    
    def detect_peak_artifacts(self, locus_data: Dict) -> Dict:
        """检测峰伪影（如肩峰、分裂峰等）"""
        alleles = locus_data['alleles']
        heights = locus_data['heights']
        sizes = locus_data.get('sizes', {})
        
        artifacts = {
            'shoulder_peaks': [],
            'split_peaks': [],
            'pull_up_peaks': [],
            'spike_artifacts': []
        }
        
        # 检测肩峰
        for allele in alleles:
            height = heights[allele]
            
            # 肩峰检测：相邻大小范围内有明显更高的峰
            if allele in sizes:
                current_size = sizes[allele]
                
                for other_allele in alleles:
                    if other_allele != allele and other_allele in sizes:
                        other_size = sizes[other_allele]
                        other_height = heights[other_allele]
                        
                        # 如果在附近有显著更高的峰，可能是肩峰
                        if (abs(current_size - other_size) < 5.0 and 
                            other_height > height * 3 and 
                            height < config.HEIGHT_THRESHOLD * 1.5):
                            artifacts['shoulder_peaks'].append({
                                'artifact_allele': allele,
                                'main_allele': other_allele,
                                'height_ratio': height / other_height
                            })
            
            # 尖峰伪影检测：非常窄且高的峰
            if height > config.SATURATION_THRESHOLD * 0.8:
                artifacts['spike_artifacts'].append({
                    'allele': allele,
                    'height': height,
                    'artifact_type': 'spike'
                })
        
        return artifacts
    
    def estimate_noise_parameters(self, sample_data: pd.DataFrame) -> Dict:
        """估计样本的整体噪声参数"""
        noise_params = {
            'global_baseline_noise': 0.0,
            'average_snr': 0.0,
            'stutter_rate': 0.0,
            'ado_rate': 0.0,
            'artifact_density': 0.0
        }
        
        all_heights = []
        all_stutter_ratios = []
        all_ado_rates = []
        total_artifacts = 0
        total_alleles = 0
        
        # 按位点分析
        for locus, locus_group in sample_data.groupby('Marker'):
            # 准备位点数据
            locus_data = {
                'locus': locus,
                'alleles': locus_group['Allele'].tolist(),
                'heights': dict(zip(locus_group['Allele'], locus_group['Height'])),
                'sizes': dict(zip(locus_group['Allele'], locus_group.get('Size', locus_group['Allele'])))
            }
            
            heights = list(locus_data['heights'].values())
            all_heights.extend(heights)
            
            # 基线噪声检测
            baseline_info = self.detect_baseline_noise(np.array(heights))
            
            # Stutter检测
            stutter_info = self.detect_stutter_artifacts(locus_data)
            if stutter_info['stutter_count'] > 0:
                all_stutter_ratios.extend([s['stutter_ratio'] for s in stutter_info['stutter_candidates']])
            
            # ADO检测
            ado_info = self.detect_allele_dropout(locus_data)
            all_ado_rates.append(ado_info['estimated_ado_rate'])
            
            # 伪影检测
            artifacts = self.detect_peak_artifacts(locus_data)
            artifact_count = sum(len(artifacts[key]) for key in artifacts)
            total_artifacts += artifact_count
            total_alleles += len(locus_data['alleles'])
        
        # 汇总噪声参数
        if all_heights:
            noise_params['global_baseline_noise'] = np.std(all_heights) * 0.1
            noise_params['average_snr'] = np.mean(all_heights) / max(noise_params['global_baseline_noise'], 1e-6)
        
        if all_stutter_ratios:
            noise_params['stutter_rate'] = np.mean(all_stutter_ratios)
        
        if all_ado_rates:
            noise_params['ado_rate'] = np.mean(all_ado_rates)
        
        if total_alleles > 0:
            noise_params['artifact_density'] = total_artifacts / total_alleles
        
        return noise_params

# =====================
# 3. V5特征驱动的参数估计器
# =====================
class V5ParameterEstimator:
    """基于V5特征的自适应参数估计器"""
    
    def __init__(self):
        self.parameter_models = {}
        self.feature_cache = {}
        logger.info("V5参数估计器初始化完成")
    
    def extract_v5_features_enhanced(self, sample_peaks: pd.DataFrame) -> Dict:
        """提取增强的V5特征集"""
        if sample_peaks.empty:
            return self._get_default_v5_features()
        
        features = {}
        
        # 基础统计特征
        total_peaks = len(sample_peaks)
        all_heights = sample_peaks['Height'].values
        
        features['total_peak_count'] = total_peaks
        features['avg_peak_height'] = np.mean(all_heights) if total_peaks > 0 else 0
        features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
        features['max_peak_height'] = np.max(all_heights) if total_peaks > 0 else 0
        features['min_peak_height'] = np.min(all_heights) if total_peaks > 0 else 0
        
        # 按位点分组特征
        locus_groups = sample_peaks.groupby('Marker')
        alleles_per_locus = locus_groups['Allele'].nunique()
        
        features['max_alleles_per_locus'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
        features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
        
        # 峰高分布特征
        if total_peaks > 2:
            features['skewness_peak_height'] = stats.skew(all_heights)
            features['kurtosis_peak_height'] = stats.kurtosis(all_heights)
        else:
            features['skewness_peak_height'] = 0
            features['kurtosis_peak_height'] = 0
        
        # PHR特征
        phr_values = []
        for locus, locus_group in locus_groups:
            if len(locus_group) == 2:
                heights = locus_group['Height'].values
                phr = min(heights) / max(heights) if max(heights) > 0 else 0
                phr_values.append(phr)
        
        if phr_values:
            features['avg_phr'] = np.mean(phr_values)
            features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
            features['min_phr'] = np.min(phr_values)
            features['num_severe_imbalance'] = sum(phr <= config.PHR_IMBALANCE_THRESHOLD for phr in phr_values)
            features['ratio_severe_imbalance'] = features['num_severe_imbalance'] / len(phr_values)
        else:
            features.update({
                'avg_phr': 0, 'std_phr': 0, 'min_phr': 0,
                'num_severe_imbalance': 0, 'ratio_severe_imbalance': 0
            })
        
        # 位点间平衡特征
        locus_heights = locus_groups['Height'].sum()
        if len(locus_heights) > 1:
            total_height = locus_heights.sum()
            if total_height > 0:
                locus_probs = locus_heights / total_height
                features['inter_locus_entropy'] = -np.sum(locus_probs * np.log(locus_probs + 1e-10))
            else:
                features['inter_locus_entropy'] = 0
        else:
            features['inter_locus_entropy'] = 0
        
        # 饱和特征
        saturated_count = (sample_peaks['Height'] >= config.SATURATION_THRESHOLD).sum()
        features['num_saturated_peaks'] = saturated_count
        features['ratio_saturated_peaks'] = saturated_count / total_peaks if total_peaks > 0 else 0
        
        return features
    
    def _get_default_v5_features(self) -> Dict:
        """获取默认V5特征"""
        return {
            'total_peak_count': 0, 'avg_peak_height': 0, 'std_peak_height': 0,
            'max_peak_height': 0, 'min_peak_height': 0, 'max_alleles_per_locus': 0,
            'avg_alleles_per_locus': 0, 'std_alleles_per_locus': 0,
            'skewness_peak_height': 0, 'kurtosis_peak_height': 0,
            'avg_phr': 0, 'std_phr': 0, 'min_phr': 0,
            'num_severe_imbalance': 0, 'ratio_severe_imbalance': 0,
            'inter_locus_entropy': 0, 'num_saturated_peaks': 0, 'ratio_saturated_peaks': 0
        }
    
    def estimate_gamma_parameters(self, v5_features: Dict) -> Dict:
        """基于V5特征估计gamma参数（放大效率）"""
        # 基础放大效率
        base_gamma = 1000.0  # 基础RFU单位
        
        # 根据平均峰高调整
        avg_height = v5_features.get('avg_peak_height', 1000.0)
        height_factor = np.sqrt(avg_height / 1000.0)
        
        # 根据位点间平衡调整
        inter_locus_entropy = v5_features.get('inter_locus_entropy', 1.0)
        balance_factor = 1.0 + 0.2 * inter_locus_entropy
        
        # 根据饱和情况调整
        saturation_ratio = v5_features.get('ratio_saturated_peaks', 0.0)
        saturation_factor = 1.0 - 0.3 * saturation_ratio
        
        gamma_global = base_gamma * height_factor * balance_factor * saturation_factor
        
        return {
            'gamma_global': max(gamma_global, 100.0),
            'gamma_std': gamma_global * 0.2,  # 20%变异
            'gamma_locus_variation': 0.3  # 位点间变异系数
        }
    
    def estimate_sigma_parameters(self, v5_features: Dict) -> Dict:
        """基于V5特征估计sigma参数（方差参数）"""
        # 基础方差参数
        base_sigma = 0.15
        
        # 根据峰高分布调整
        std_height = v5_features.get('std_peak_height', 0.0)
        avg_height = max(v5_features.get('avg_peak_height', 1000.0), 1.0)
        cv_height = std_height / avg_height
        
        # 根据PHR失衡调整
        ratio_imbalance = v5_features.get('ratio_severe_imbalance', 0.0)
        
        # 根据偏度调整
        skewness = abs(v5_features.get('skewness_peak_height', 0.0))
        
        # 综合调整
        sigma_var = base_sigma * (1.0 + 0.5 * cv_height + 0.3 * ratio_imbalance + 0.2 * skewness)
        
        return {
            'sigma_var_global': max(sigma_var, 0.05),
            'sigma_var_locus_factor': 1.2,  # 位点特异性因子
            'sigma_var_degradation_factor': 0.001  # 降解影响因子
        }
    
    def estimate_noise_thresholds(self, v5_features: Dict, noise_params: Dict) -> Dict:
        """基于V5特征和噪声参数估计阈值"""
        # 基础阈值
        base_threshold = config.HEIGHT_THRESHOLD
        
        # 根据SNR调整
        snr = noise_params.get('average_snr', 10.0)
        snr_factor = max(0.5, min(2.0, 10.0 / snr))
        
        # 根据基线噪声调整
        baseline_noise = noise_params.get('global_baseline_noise', 50.0)
        noise_factor = max(1.0, baseline_noise / 50.0)
        
        # 根据伪影密度调整
        artifact_density = noise_params.get('artifact_density', 0.1)
        artifact_factor = 1.0 + artifact_density
        
        # 计算动态阈值
        dynamic_threshold = base_threshold * snr_factor * noise_factor * artifact_factor
        
        return {
            'peak_calling_threshold': max(dynamic_threshold, 30.0),
            'noise_threshold': baseline_noise * 2,
            'stutter_threshold': dynamic_threshold * 0.6,
            'ado_threshold': dynamic_threshold * 1.5
        }

# =====================
# 4. UPG-M核心推断引擎
# =====================
class UPG_M_Core:
    """UPG-M统一概率基因分型核心引擎"""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.v5_estimator = V5ParameterEstimator()
        self.current_state = None
        self.mcmc_samples = []
        
        logger.info("UPG-M核心引擎初始化完成")
    
    def initialize_state(self, observed_data: Dict, max_noc: int = 5) -> Dict:
        """初始化MCMC状态"""
        # 初始化贡献者人数（从2开始）
        N = 2
        
        # 初始化混合比例
        Mx = np.random.dirichlet(np.ones(N))
        
        # 初始化基因型
        genotypes = {}
        for locus, locus_data in observed_data.items():
            # 简单初始化：为每个贡献者随机分配观测的等位基因
            alleles = locus_data['alleles']
            locus_genotypes = []
            
            for i in range(N):
                if len(alleles) >= 2:
                    genotype = tuple(np.random.choice(alleles, 2, replace=True))
                else:
                    genotype = (alleles[0], alleles[0]) if alleles else ('0', '0')
                locus_genotypes.append(genotype)
            
            genotypes[locus] = locus_genotypes
        
        # 初始化其他参数
        theta = {
            'gamma_global': 1000.0,
            'sigma_var_global': 0.15,
            'stutter_ratio': 0.05,
            'ado_probability': 0.1
        }
        
        return {
            'N': N,
            'Mx': Mx,
            'genotypes': genotypes,
            'theta': theta
        }
    
    def calculate_likelihood(self, state: Dict, observed_data: Dict, 
                           v5_features: Dict, noise_params: Dict) -> float:
        """计算当前状态的似然函数"""
        total_log_likelihood = 0.0
        
        N = state['N']
        Mx = state['Mx']
        genotypes = state['genotypes']
        theta = state['theta']
        
        # 获取V5驱动的参数
        gamma_params = self.v5_estimator.estimate_gamma_parameters(v5_features)
        sigma_params = self.v5_estimator.estimate_sigma_parameters(v5_features)
        thresholds = self.v5_estimator.estimate_noise_thresholds(v5_features, noise_params)
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self._calculate_locus_likelihood(
                locus, locus_data, state, gamma_params, sigma_params, thresholds
            )
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def _calculate_locus_likelihood(self, locus: str, locus_data: Dict, 
                                  state: Dict, gamma_params: Dict, 
                                  sigma_params: Dict, thresholds: Dict) -> float:
        """计算单个位点的似然函数"""
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        N = state['N']
        Mx = state['Mx']
        locus_genotypes = state['genotypes'][locus]
        
        log_likelihood = 0.0
        
        # 位点特异性参数
        gamma_l = gamma_params['gamma_global'] * np.random.normal(1.0, gamma_params['gamma_locus_variation'])
        sigma_l = sigma_params['sigma_var_global'] * sigma_params['sigma_var_locus_factor']
        
        # 观测等位基因的似然
        for allele in observed_alleles:
            observed_height = observed_heights[allele]
            
            # 计算期望峰高
            expected_height = self._calculate_expected_peak_height(
                allele, locus, locus_genotypes, Mx, gamma_l, state['theta']
            )
            
            # 峰高似然（对数正态分布）
            if expected_height > 1e-6:
                log_mu = np.log(expected_height) - sigma_l**2 / 2
                log_likelihood += stats.lognorm.logpdf(observed_height, sigma_l, scale=np.exp(log_mu))
            else:
                # 如果期望峰高很小，给一个较低的似然
                log_likelihood += -1e6
        
        # 未观测等位基因的ADO似然
        all_genotype_alleles = set()
        for genotype in locus_genotypes:
            all_genotype_alleles.update(genotype)
        
        missing_alleles = all_genotype_alleles - set(observed_alleles)
        for missing_allele in missing_alleles:
            expected_height = self._calculate_expected_peak_height(
                missing_allele, locus, locus_genotypes, Mx, gamma_l, state['theta']
            )
            
            # ADO概率
            ado_prob = self._calculate_ado_probability(expected_height, state['theta'])
            log_likelihood += np.log(max(ado_prob, 1e-10))
        
        return log_likelihood
    
    def _calculate_expected_peak_height(self, allele: str, locus: str, 
                                      genotypes: List[Tuple], Mx: np.ndarray,
                                      gamma_l: float, theta: Dict) -> float:
        """计算等位基因的期望峰高"""
        total_contribution = 0.0
        
        # 直接贡献
        for i, genotype in enumerate(genotypes):
            copy_number = self._get_copy_number(allele, genotype)
            if copy_number > 0:
                # 降解因子
                degradation_factor = self._calculate_degradation_factor(allele, locus)
                total_contribution += gamma_l * Mx[i] * copy_number * degradation_factor
        
        # Stutter贡献
        stutter_contribution = self._calculate_stutter_contribution(
            allele, locus, genotypes, Mx, gamma_l, theta
        )
        
        return total_contribution + stutter_contribution
    
    def _get_copy_number(self, allele: str, genotype: Tuple[str, str]) -> float:
        """计算等位基因在基因型中的拷贝数"""
        if genotype is None or len(genotype) != 2:
            return 0.0
        
        count = sum(1 for gt_allele in genotype if gt_allele == allele)
        
        # 对于纯合子，考虑preferential amplification
        if genotype[0] == genotype[1] and allele in genotype:
            return 2.0  # 可以调整为其他值来模拟偏好性放大
        
        return float(count)
    
    def _calculate_degradation_factor(self, allele: str, locus: str) -> float:
        """计算降解因子"""
        try:
            # 简化的片段大小计算
            allele_num = float(allele)
            size = 150.0 + allele_num * 4.0  # 估算片段大小
            
            # 指数降解模型
            k_deg = 0.001  # 降解常数
            size_ref = 200.0  # 参考大小
            
            degradation_factor = np.exp(-k_deg * max(0, size - size_ref))
            return max(degradation_factor, 0.1)
            
        except (ValueError, TypeError):
            return 1.0
    
    def _calculate_stutter_contribution(self, target_allele: str, locus: str,
                                      genotypes: List[Tuple], Mx: np.ndarray,
                                      gamma_l: float, theta: Dict) -> float:
        """计算Stutter贡献"""
        stutter_ratio = theta.get('stutter_ratio', 0.05)
        
        try:
            target_num = float(target_allele)
            parent_num = target_num + 1  # n-1 stutter
            parent_allele = str(int(parent_num)) if parent_num.is_integer() else str(parent_num)
            
            # 计算parent allele的贡献
            parent_contribution = 0.0
            for i, genotype in enumerate(genotypes):
                copy_number = self._get_copy_number(parent_allele, genotype)
                if copy_number > 0:
                    degradation_factor = self._calculate_degradation_factor(parent_allele, locus)
                    parent_contribution += gamma_l * Mx[i] * copy_number * degradation_factor
            
            return stutter_ratio * parent_contribution
            
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_ado_probability(self, expected_height: float, theta: Dict) -> float:
        """计算等位基因缺失概率"""
        if expected_height <= 0:
            return 0.99
        
        # Logistic模型
        H_50 = 200.0  # 50%缺失的峰高阈值
        slope = 0.01  # 坡度参数
        
        ado_prob = 1.0 / (1.0 + np.exp(slope * (expected_height - H_50)))
        return np.clip(ado_prob, 1e-6, 0.99)
    
    def calculate_priors(self, state: Dict) -> float:
        """计算先验概率"""
        log_prior = 0.0
        
        N = state['N']
        Mx = state['Mx']
        theta = state['theta']
        
        # NoC先验 (假设均匀分布在2-5之间)
        log_prior += np.log(1.0 / 4.0)  # P(N) for N in {2,3,4,5}
        
        # 混合比例先验 (Dirichlet)
        alpha = np.ones(N) * 1.0  # 可以根据V5特征调整
        log_prior += (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                     np.sum((alpha - 1) * np.log(Mx + 1e-10)))
        
        # 参数先验
        log_prior += stats.gamma.logpdf(theta['gamma_global'], a=2, scale=500)  # Gamma先验
        log_prior += stats.beta.logpdf(theta['stutter_ratio'], a=2, b=10)  # Beta先验
        log_prior += stats.beta.logpdf(theta['ado_probability'], a=1, b=9)  # Beta先验
        
        return log_prior
    
    def propose_new_state(self, current_state: Dict, observed_data: Dict) -> Dict:
        """提议新状态"""
        new_state = current_state.copy()
        new_state['genotypes'] = {k: [gt for gt in v] for k, v in current_state['genotypes'].items()}
        new_state['theta'] = current_state['theta'].copy()
        
        # 随机选择更新类型
        update_type = np.random.choice(['N', 'Mx', 'genotypes', 'theta'], p=[0.1, 0.3, 0.4, 0.2])
        
        if update_type == 'N':
            # Trans-dimensional move (RJMCMC)
            new_state = self._propose_noc_change(new_state, observed_data)
        elif update_type == 'Mx':
            # 更新混合比例
            new_state = self._propose_mixture_ratio_change(new_state)
        elif update_type == 'genotypes':
            # 更新基因型
            new_state = self._propose_genotype_change(new_state, observed_data)
        elif update_type == 'theta':
            # 更新参数
            new_state = self._propose_parameter_change(new_state)
        
        return new_state
    
    def _propose_noc_change(self, state: Dict, observed_data: Dict) -> Dict:
        """提议NoC变化（RJMCMC）"""
        current_N = state['N']
        
        # 限制NoC范围
        if current_N == 2:
            new_N = 3
        elif current_N == config.MAX_NOC:
            new_N = config.MAX_NOC - 1
        else:
            new_N = current_N + np.random.choice([-1, 1])
        
        new_state = state.copy()
        new_state['N'] = new_N
        
        if new_N > current_N:
            # 增加贡献者
            # 重新分配混合比例
            old_Mx = state['Mx']
            split_idx = np.random.randint(len(old_Mx))
            split_ratio = np.random.beta(2, 2)  # 分割比例
            
            new_Mx = np.zeros(new_N)
            new_Mx[:current_N] = old_Mx
            new_Mx[split_idx] *= split_ratio
            new_Mx[current_N] = old_Mx[split_idx] * (1 - split_ratio)
            
            new_state['Mx'] = new_Mx
            
            # 为新贡献者添加基因型
            for locus in observed_data:
                alleles = observed_data[locus]['alleles']
                if len(alleles) >= 2:
                    new_genotype = tuple(np.random.choice(alleles, 2, replace=True))
                else:
                    new_genotype = (alleles[0], alleles[0]) if alleles else ('0', '0')
                
                new_state['genotypes'][locus].append(new_genotype)
        
        else:
            # 减少贡献者
            remove_idx = np.random.randint(current_N)
            
            # 合并混合比例
            old_Mx = state['Mx']
            merge_idx = remove_idx if remove_idx == 0 else remove_idx - 1
            
            new_Mx = np.delete(old_Mx, remove_idx)
            if merge_idx < len(new_Mx):
                new_Mx[merge_idx] += old_Mx[remove_idx]
            
            # 重新归一化
            new_Mx = new_Mx / np.sum(new_Mx)
            new_state['Mx'] = new_Mx
            
            # 移除基因型
            for locus in observed_data:
                new_state['genotypes'][locus].pop(remove_idx)
        
        return new_state
    
    def _propose_mixture_ratio_change(self, state: Dict) -> Dict:
        """提议混合比例变化"""
        new_state = state.copy()
        
        # 使用Dirichlet提议
        current_Mx = state['Mx']
        concentration = current_Mx * 100  # 集中度参数
        concentration = np.maximum(concentration, 0.1)
        
        new_Mx = np.random.dirichlet(concentration)
        new_state['Mx'] = new_Mx
        
        return new_state
    
    def _propose_genotype_change(self, state: Dict, observed_data: Dict) -> Dict:
        """提议基因型变化"""
        new_state = state.copy()
        new_state['genotypes'] = {k: [gt for gt in v] for k, v in state['genotypes'].items()}
        
        # 随机选择位点和贡献者
        locus = np.random.choice(list(observed_data.keys()))
        contributor_idx = np.random.randint(state['N'])
        
        # 从观测的等位基因中重新采样
        alleles = observed_data[locus]['alleles']
        if len(alleles) >= 1:
            new_genotype = tuple(np.random.choice(alleles, 2, replace=True))
            new_state['genotypes'][locus][contributor_idx] = new_genotype
        
        return new_state
    
    def _propose_parameter_change(self, state: Dict) -> Dict:
        """提议参数变化"""
        new_state = state.copy()
        new_state['theta'] = state['theta'].copy()
        
        # 随机选择参数更新
        param_name = np.random.choice(['gamma_global', 'sigma_var_global', 'stutter_ratio', 'ado_probability'])
        
        if param_name == 'gamma_global':
            # 对数正态提议
            current_val = state['theta'][param_name]
            new_val = current_val * np.exp(np.random.normal(0, 0.1))
            new_state['theta'][param_name] = max(new_val, 100.0)
        
        elif param_name == 'sigma_var_global':
            # 截断正态提议
            current_val = state['theta'][param_name]
            new_val = current_val + np.random.normal(0, 0.02)
            new_state['theta'][param_name] = np.clip(new_val, 0.05, 0.5)
        
        elif param_name in ['stutter_ratio', 'ado_probability']:
            # Beta分布的logit变换
            current_val = state['theta'][param_name]
            logit_current = np.log(current_val / (1 - current_val))
            logit_new = logit_current + np.random.normal(0, 0.1)
            new_val = 1 / (1 + np.exp(-logit_new))
            new_state['theta'][param_name] = np.clip(new_val, 0.001, 0.999)
        
        return new_state
    
    def mcmc_sampler(self, observed_data: Dict, v5_features: Dict, 
                    noise_params: Dict, n_iterations: int = None) -> Dict:
        """UPG-M主MCMC采样器"""
        if n_iterations is None:
            n_iterations = config.N_ITERATIONS
        
        logger.info(f"开始UPG-M MCMC采样，迭代次数: {n_iterations}")
        
        # 初始化状态
        current_state = self.initialize_state(observed_data)
        
        # 计算初始概率
        current_log_likelihood = self.calculate_likelihood(current_state, observed_data, v5_features, noise_params)
        current_log_prior = self.calculate_priors(current_state)
        current_log_posterior = current_log_likelihood + current_log_prior
        
        # 存储样本
        samples = {
            'states': [],
            'log_likelihood': [],
            'log_posterior': [],
            'acceptance_info': []
        }
        
        n_accepted = 0
        acceptance_details = []
        
        # MCMC主循环
        for iteration in range(n_iterations):
            if iteration % 1000 == 0:
                acceptance_rate = n_accepted / max(iteration, 1)
                logger.info(f"UPG-M迭代 {iteration}/{n_iterations}, "
                          f"接受率: {acceptance_rate:.3f}, "
                          f"当前NoC: {current_state['N']}, "
                          f"似然: {current_log_likelihood:.2f}")
            
            # 提议新状态
            proposed_state = self.propose_new_state(current_state, observed_data)
            
            # 计算提议状态的概率
            proposed_log_likelihood = self.calculate_likelihood(proposed_state, observed_data, v5_features, noise_params)
            proposed_log_prior = self.calculate_priors(proposed_state)
            proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
            
            # Metropolis-Hastings接受/拒绝
            log_ratio = proposed_log_posterior - current_log_posterior
            
            # 对于trans-dimensional moves，需要考虑Jacobian
            if proposed_state['N'] != current_state['N']:
                # 简化的Jacobian处理
                jacobian_correction = 0.0  # 可以进一步完善
                log_ratio += jacobian_correction
            
            accept_prob = min(1.0, np.exp(log_ratio))
            
            if np.random.random() < accept_prob:
                current_state = proposed_state
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
                'proposed_noc': proposed_state['N'],
                'current_noc': current_state['N'],
                'log_ratio': log_ratio,
                'accept_prob': accept_prob
            })
            
            # 存储样本（预热后）
            if iteration >= config.N_WARMUP and iteration % config.THINNING == 0:
                # 深拷贝状态
                sample_state = {
                    'N': current_state['N'],
                    'Mx': current_state['Mx'].copy(),
                    'genotypes': {k: [gt for gt in v] for k, v in current_state['genotypes'].items()},
                    'theta': current_state['theta'].copy()
                }
                
                samples['states'].append(sample_state)
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
        
        final_acceptance_rate = n_accepted / n_iterations
        logger.info(f"UPG-M MCMC完成，总接受率: {final_acceptance_rate:.3f}")
        logger.info(f"有效样本数: {len(samples['states'])}")
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'n_samples': len(samples['states']),
            'acceptance_details': acceptance_details,
            'converged': 0.15 <= final_acceptance_rate <= 0.6
        }

# =====================
# 5. 基因型降噪后处理器
# =====================
class GenotypeDenoiser:
    """基因型降噪后处理器"""
    
    def __init__(self):
        self.denoising_strategies = {}
        logger.info("基因型降噪器初始化完成")
    
    def apply_posterior_denoising(self, mcmc_results: Dict, observed_data: Dict, 
                                 noise_params: Dict) -> Dict:
        """应用后验降噪"""
        samples = mcmc_results['samples']
        states = samples['states']
        
        denoised_results = {
            'consensus_noc': self._estimate_consensus_noc(states),
            'consensus_genotypes': self._estimate_consensus_genotypes(states, observed_data),
            'denoised_mixture_ratios': self._estimate_consensus_mixture_ratios(states),
            'noise_classification': self._classify_noise_sources(states, observed_data, noise_params),
            'confidence_intervals': self._calculate_confidence_intervals(states)
        }
        
        return denoised_results
    
    def _estimate_consensus_noc(self, states: List[Dict]) -> Dict:
        """估计一致性NoC"""
        noc_counts = {}
        for state in states:
            noc = state['N']
            noc_counts[noc] = noc_counts.get(noc, 0) + 1
        
        total_samples = len(states)
        noc_probabilities = {noc: count/total_samples for noc, count in noc_counts.items()}
        
        # 最大后验估计
        map_noc = max(noc_probabilities.items(), key=lambda x: x[1])[0]
        
        return {
            'map_noc': map_noc,
            'noc_probabilities': noc_probabilities,
            'confidence': noc_probabilities[map_noc]
        }
    
    def _estimate_consensus_genotypes(self, states: List[Dict], observed_data: Dict) -> Dict:
        """估计一致性基因型"""
        consensus_genotypes = {}
        
        # 只考虑MAP NoC的样本
        map_noc = self._estimate_consensus_noc(states)['map_noc']
        filtered_states = [state for state in states if state['N'] == map_noc]
        
        for locus in observed_data:
            locus_consensus = []
            
            for contributor_idx in range(map_noc):
                # 收集该贡献者在该位点的所有基因型
                genotype_counts = {}
                
                for state in filtered_states:
                    if (locus in state['genotypes'] and 
                        contributor_idx < len(state['genotypes'][locus])):
                        
                        genotype = tuple(sorted(state['genotypes'][locus][contributor_idx]))
                        genotype_counts[genotype] = genotype_counts.get(genotype, 0) + 1
                
                # 选择最频繁的基因型
                if genotype_counts:
                    map_genotype = max(genotype_counts.items(), key=lambda x: x[1])[0]
                    confidence = genotype_counts[map_genotype] / len(filtered_states)
                    
                    locus_consensus.append({
                        'genotype': map_genotype,
                        'confidence': confidence,
                        'alternatives': genotype_counts
                    })
                else:
                    locus_consensus.append({
                        'genotype': ('0', '0'),
                        'confidence': 0.0,
                        'alternatives': {}
                    })
            
            consensus_genotypes[locus] = locus_consensus
        
        return consensus_genotypes
    
    def _estimate_consensus_mixture_ratios(self, states: List[Dict]) -> Dict:
        """估计一致性混合比例"""
        map_noc = self._estimate_consensus_noc(states)['map_noc']
        filtered_states = [state for state in states if state['N'] == map_noc]
        
        if not filtered_states:
            return {'map_ratios': [], 'credible_intervals': []}
        
        # 收集混合比例样本
        mx_samples = np.array([state['Mx'] for state in filtered_states])
        
        # 计算统计量
        map_ratios = np.mean(mx_samples, axis=0)
        
        # 计算95%置信区间
        credible_intervals = []
        for i in range(map_noc):
            ci = np.percentile(mx_samples[:, i], [2.5, 97.5])
            credible_intervals.append(ci.tolist())
        
        return {
            'map_ratios': map_ratios.tolist(),
            'credible_intervals': credible_intervals,
            'std_ratios': np.std(mx_samples, axis=0).tolist()
        }
    
    def _classify_noise_sources(self, states: List[Dict], observed_data: Dict, 
                               noise_params: Dict) -> Dict:
        """分类噪声源"""
        noise_classification = {
            'true_alleles': {},
            'stutter_artifacts': {},
            'noise_peaks': {},
            'ado_alleles': {}
        }
        
        # 基于后验样本分析每个观测峰的性质
        for locus, locus_data in observed_data.items():
            alleles = locus_data['alleles']
            heights = locus_data['heights']
            
            for allele in alleles:
                # 分析该等位基因在后验样本中的出现频率
                true_allele_count = 0
                stutter_candidate_count = 0
                
                for state in states:
                    if locus in state['genotypes']:
                        # 检查是否为真实等位基因
                        is_true_allele = any(allele in genotype 
                                           for genotype in state['genotypes'][locus])
                        
                        if is_true_allele:
                            true_allele_count += 1
                        else:
                            # 检查是否可能是stutter
                            try:
                                allele_num = float(allele)
                                parent_num = allele_num + 1
                                parent_allele = str(int(parent_num)) if parent_num.is_integer() else str(parent_num)
                                
                                is_stutter = any(parent_allele in genotype 
                                               for genotype in state['genotypes'][locus])
                                
                                if is_stutter:
                                    stutter_candidate_count += 1
                            except:
                                pass
                
                total_samples = len(states)
                true_probability = true_allele_count / total_samples
                stutter_probability = stutter_candidate_count / total_samples
                
                # 分类
                if true_probability > 0.5:
                    noise_classification['true_alleles'][f"{locus}:{allele}"] = {
                        'probability': true_probability,
                        'height': heights[allele]
                    }
                elif stutter_probability > 0.3:
                    noise_classification['stutter_artifacts'][f"{locus}:{allele}"] = {
                        'probability': stutter_probability,
                        'height': heights[allele]
                    }
                elif heights[allele] < config.HEIGHT_THRESHOLD * 2:
                    noise_classification['noise_peaks'][f"{locus}:{allele}"] = {
                        'probability': 1 - true_probability - stutter_probability,
                        'height': heights[allele]
                    }
        
        return noise_classification
    
    def _calculate_confidence_intervals(self, states: List[Dict]) -> Dict:
        """计算置信区间"""
        # 参数的置信区间
        parameter_samples = {
            'gamma_global': [],
            'sigma_var_global': [],
            'stutter_ratio': [],
            'ado_probability': []
        }
        
        for state in states:
            theta = state['theta']
            for param_name in parameter_samples:
                if param_name in theta:
                    parameter_samples[param_name].append(theta[param_name])
        
        confidence_intervals = {}
        for param_name, samples in parameter_samples.items():
            if samples:
                ci_95 = np.percentile(samples, [2.5, 97.5])
                confidence_intervals[param_name] = {
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'ci_95': ci_95.tolist()
                }
        
        return confidence_intervals

# =====================
# 6. 可视化模块
# =====================
class UPG_Visualizer:
    """UPG-M结果可视化器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"可视化器初始化，输出目录: {output_dir}")
    
    def plot_denoising_results(self, sample_file: str, original_data: Dict, 
                              denoised_results: Dict, mcmc_results: Dict):
        """绘制降噪结果"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. NoC后验分布
        ax1 = plt.subplot(3, 4, 1)
        noc_probs = denoised_results['consensus_noc']['noc_probabilities']
        nocs = list(noc_probs.keys())
        probs = list(noc_probs.values())
        
        bars = ax1.bar(nocs, probs, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('贡献者人数 (NoC)')
        ax1.set_ylabel('后验概率')
        ax1.set_title('NoC后验分布')
        ax1.grid(True, alpha=0.3)
        
        # 标注最大后验值
        map_noc = max(noc_probs.items(), key=lambda x: x[1])[0]
        for bar, noc in zip(bars, nocs):
            if noc == map_noc:
                bar.set_color('orange')
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'MAP: {probs[nocs.index(noc)]:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. 混合比例后验分布
        ax2 = plt.subplot(3, 4, 2)
        mx_results = denoised_results['denoised_mixture_ratios']
        map_ratios = mx_results['map_ratios']
        ci_ratios = mx_results['credible_intervals']
        
        x_pos = np.arange(len(map_ratios))
        bars = ax2.bar(x_pos, map_ratios, 
                      yerr=[[r[1] - map_ratios[i] for i, r in enumerate(ci_ratios)],
                            [map_ratios[i] - r[0] for i, r in enumerate(ci_ratios)]],
                      capsize=5, color='lightgreen', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('贡献者编号')
        ax2.set_ylabel('混合比例')
        ax2.set_title('混合比例后验估计')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'C{i+1}' for i in x_pos])
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (ratio, ci) in enumerate(zip(map_ratios, ci_ratios)):
            ax2.text(i, ratio + 0.02, f'{ratio:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]',
                    ha='center', va='bottom', fontsize=8)
        
        # 3. MCMC轨迹图 (NoC)
        ax3 = plt.subplot(3, 4, 3)
        states = mcmc_results['samples']['states']
        noc_trace = [state['N'] for state in states]
        
        ax3.plot(noc_trace, alpha=0.7, linewidth=1)
        ax3.set_xlabel('MCMC迭代 (thinned)')
        ax3.set_ylabel('NoC')
        ax3.set_title('NoC MCMC轨迹')
        ax3.grid(True, alpha=0.3)
        
        # 4. 混合比例轨迹图
        ax4 = plt.subplot(3, 4, 4)
        if len(map_ratios) > 0:
            for i in range(min(3, len(map_ratios))):  # 最多显示3个贡献者
                mx_trace = []
                for state in states:
                    if state['N'] > i:
                        mx_trace.append(state['Mx'][i])
                    else:
                        mx_trace.append(0)
                
                ax4.plot(mx_trace, label=f'贡献者{i+1}', alpha=0.7, linewidth=1)
        
        ax4.set_xlabel('MCMC迭代 (thinned)')
        ax4.set_ylabel('混合比例')
        ax4.set_title('混合比例MCMC轨迹')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 噪声分类饼图
        ax5 = plt.subplot(3, 4, 5)
        noise_classification = denoised_results['noise_classification']
        
        categories = ['真实等位基因', 'Stutter伪影', '噪声峰']
        counts = [
            len(noise_classification['true_alleles']),
            len(noise_classification['stutter_artifacts']),
            len(noise_classification['noise_peaks'])
        ]
        
        colors = ['lightgreen', 'orange', 'lightcoral']
        wedges, texts, autotexts = ax5.pie(counts, labels=categories, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax5.set_title('峰分类结果')
        
        # 6. 降噪前后对比 (选择一个代表性位点)
        ax6 = plt.subplot(3, 4, 6)
        
        # 选择第一个位点作为示例
        locus_names = list(original_data.keys())
        if locus_names:
            example_locus = locus_names[0]
            locus_data = original_data[example_locus]
            
            alleles = locus_data['alleles']
            heights = [locus_data['heights'][allele] for allele in alleles]
            
            # 根据噪声分类给峰着色
            colors = []
            for allele in alleles:
                peak_id = f"{example_locus}:{allele}"
                if peak_id in noise_classification['true_alleles']:
                    colors.append('green')
                elif peak_id in noise_classification['stutter_artifacts']:
                    colors.append('orange')
                elif peak_id in noise_classification['noise_peaks']:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            bars = ax6.bar(range(len(alleles)), heights, color=colors, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('等位基因')
            ax6.set_ylabel('峰高 (RFU)')
            ax6.set_title(f'位点 {example_locus} 降噪结果')
            ax6.set_xticks(range(len(alleles)))
            ax6.set_xticklabels(alleles, rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='真实等位基因'),
                Patch(facecolor='orange', alpha=0.7, label='Stutter伪影'),
                Patch(facecolor='red', alpha=0.7, label='噪声峰'),
                Patch(facecolor='gray', alpha=0.7, label='未分类')
            ]
            ax6.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 7. 参数后验分布
        ax7 = plt.subplot(3, 4, 7)
        theta_samples = []
        for state in states:
            theta_samples.append(state['theta']['stutter_ratio'])
        
        ax7.hist(theta_samples, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        ax7.set_xlabel('Stutter比例')
        ax7.set_ylabel('密度')
        ax7.set_title('Stutter比例后验分布')
        ax7.axvline(np.mean(theta_samples), color='red', linestyle='--', 
                   label=f'均值: {np.mean(theta_samples):.3f}')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. ADO概率后验分布
        ax8 = plt.subplot(3, 4, 8)
        ado_samples = []
        for state in states:
            ado_samples.append(state['theta']['ado_probability'])
        
        ax8.hist(ado_samples, bins=30, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
        ax8.set_xlabel('ADO概率')
        ax8.set_ylabel('密度')
        ax8.set_title('ADO概率后验分布')
        ax8.axvline(np.mean(ado_samples), color='red', linestyle='--',
                   label=f'均值: {np.mean(ado_samples):.3f}')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. 收敛性诊断
        ax9 = plt.subplot(3, 4, 9)
        log_likelihood = mcmc_results['samples']['log_likelihood']
        
        ax9.plot(log_likelihood, alpha=0.7, linewidth=1, color='purple')
        ax9.set_xlabel('MCMC迭代 (thinned)')
        ax9.set_ylabel('对数似然')
        ax9.set_title('MCMC收敛性诊断')
        ax9.grid(True, alpha=0.3)
        
        # 添加移动平均线
        if len(log_likelihood) > 50:
            window = min(50, len(log_likelihood) // 10)
            moving_avg = np.convolve(log_likelihood, np.ones(window)/window, mode='valid')
            ax9.plot(range(window-1, len(log_likelihood)), moving_avg, 
                    color='red', linewidth=2, label=f'移动平均({window})')
            ax9.legend()
        
        # 10. 基因型置信度热图
        ax10 = plt.subplot(3, 4, 10)
        consensus_genotypes = denoised_results['consensus_genotypes']
        
        if consensus_genotypes:
            loci = list(consensus_genotypes.keys())[:5]  # 最多显示5个位点
            contributors = range(len(map_ratios))
            
            confidence_matrix = np.zeros((len(loci), len(contributors)))
            
            for i, locus in enumerate(loci):
                for j, contributor in enumerate(contributors):
                    if j < len(consensus_genotypes[locus]):
                        confidence_matrix[i, j] = consensus_genotypes[locus][j]['confidence']
            
            im = ax10.imshow(confidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax10.set_xticks(range(len(contributors)))
            ax10.set_xticklabels([f'C{i+1}' for i in contributors])
            ax10.set_yticks(range(len(loci)))
            ax10.set_yticklabels(loci)
            ax10.set_xlabel('贡献者')
            ax10.set_ylabel('位点')
            ax10.set_title('基因型推断置信度')
            
            # 添加数值标注
            for i in range(len(loci)):
                for j in range(len(contributors)):
                    text = ax10.text(j, i, f'{confidence_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax10, shrink=0.8)
        
        # 11. 信噪比改善
        ax11 = plt.subplot(3, 4, 11)
        
        # 计算降噪前后的信噪比
        original_snr = []
        denoised_snr = []
        
        for locus, locus_data in original_data.items():
            heights = list(locus_data['heights'].values())
            if heights:
                # 原始SNR
                signal = np.max(heights)
                noise = np.std(heights)
                orig_snr = signal / max(noise, 1e-6)
                original_snr.append(orig_snr)
                
                # 降噪后SNR (移除噪声峰)
                true_heights = []
                for allele, height in locus_data['heights'].items():
                    peak_id = f"{locus}:{allele}"
                    if peak_id in noise_classification['true_alleles']:
                        true_heights.append(height)
                
                if true_heights:
                    signal_denoised = np.max(true_heights)
                    noise_denoised = np.std(true_heights)
                    denoised_snr_val = signal_denoised / max(noise_denoised, 1e-6)
                    denoised_snr.append(denoised_snr_val)
                else:
                    denoised_snr.append(orig_snr)
        
        if original_snr and denoised_snr:
            x_pos = np.arange(len(original_snr))
            width = 0.35
            
            ax11.bar(x_pos - width/2, original_snr, width, label='降噪前', 
                    color='lightcoral', alpha=0.7)
            ax11.bar(x_pos + width/2, denoised_snr, width, label='降噪后', 
                    color='lightgreen', alpha=0.7)
            
            ax11.set_xlabel('位点')
            ax11.set_ylabel('信噪比')
            ax11.set_title('降噪效果对比')
            ax11.set_xticks(x_pos)
            ax11.set_xticklabels([f'L{i+1}' for i in range(len(original_snr))])
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        
        # 12. 汇总统计
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # 创建汇总文本
        summary_text = f"""
UPG-M降噪汇总报告

样本ID: {sample_file}

推断结果:
• MAP NoC: {denoised_results['consensus_noc']['map_noc']}
• NoC置信度: {denoised_results['consensus_noc']['confidence']:.3f}
• 主要贡献者比例: {map_ratios[0]:.3f}

降噪效果:
• 真实等位基因: {len(noise_classification['true_alleles'])}个
• Stutter伪影: {len(noise_classification['stutter_artifacts'])}个  
• 噪声峰: {len(noise_classification['noise_peaks'])}个

MCMC质量:
• 接受率: {mcmc_results['acceptance_rate']:.3f}
• 有效样本: {mcmc_results['n_samples']}
• 收敛状态: {'良好' if mcmc_results['converged'] else '需检查'}

参数估计:
• Stutter比例: {np.mean([s['theta']['stutter_ratio'] for s in states]):.3f}
• ADO概率: {np.mean([s['theta']['ado_probability'] for s in states]):.3f}
        """
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{sample_file}_upg_denoising_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"样本 {sample_file} 的降噪结果图表已保存")
    
    def plot_comparison_original_vs_denoised(self, sample_file: str, original_data: Dict, 
                                           denoised_results: Dict):
        """绘制原始数据与降噪结果对比"""
        noise_classification = denoised_results['noise_classification']
        
        # 选择几个代表性位点进行对比
        loci_to_plot = list(original_data.keys())[:6]  # 最多6个位点
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, locus in enumerate(loci_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            locus_data = original_data[locus]
            
            alleles = locus_data['alleles']
            heights = [locus_data['heights'][allele] for allele in alleles]
            
            # 创建颜色映射
            colors_original = ['lightblue'] * len(alleles)
            colors_denoised = []
            
            for i, allele in enumerate(alleles):
                peak_id = f"{locus}:{allele}"
                if peak_id in noise_classification['true_alleles']:
                    colors_denoised.append('green')
                elif peak_id in noise_classification['stutter_artifacts']:
                    colors_denoised.append('orange')
                elif peak_id in noise_classification['noise_peaks']:
                    colors_denoised.append('red')
                else:
                    colors_denoised.append('gray')
            
            # 绘制对比条形图
            x_pos = np.arange(len(alleles))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, heights, width, label='原始', 
                          color=colors_original, alpha=0.7, edgecolor='black')
            
            # 降噪后的高度（噪声峰设为0）
            denoised_heights = []
            for i, allele in enumerate(alleles):
                peak_id = f"{locus}:{allele}"
                if peak_id in noise_classification['noise_peaks']:
                    denoised_heights.append(0)  # 噪声峰去除
                else:
                    denoised_heights.append(heights[i])
            
            bars2 = ax.bar(x_pos + width/2, denoised_heights, width, label='降噪后',
                          color=colors_denoised, alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('等位基因')
            ax.set_ylabel('峰高 (RFU)')
            ax.set_title(f'位点 {locus}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(alleles, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加峰高标注
            for bar, height in zip(bars2, denoised_heights):
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                           max(heights) * 0.01, f'{int(height)}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的子图
        for idx in range(len(loci_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{sample_file}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"样本 {sample_file} 的对比图表已保存")

# =====================
# 7. 主分析流水线
# =====================
class UPG_M_Pipeline:
    """UPG-M完整分析流水线"""
    
    def __init__(self):
        self.upg_core = UPG_M_Core()
        self.denoiser = GenotypeDenoiser()
        self.visualizer = UPG_Visualizer(config.OUTPUT_DIR)
        
        logger.info("UPG-M流水线初始化完成")
    
    def load_attachment4_data(self) -> Dict[str, pd.DataFrame]:
        """加载附件4数据"""
        try:
            df_att4 = pd.read_csv(config.ATTACHMENT4_PATH, encoding='utf-8')
            logger.info(f"成功加载附件4数据，形状: {df_att4.shape}")
            
            # 按样本分组
            sample_groups = {}
            for sample_file, group in df_att4.groupby('Sample File'):
                sample_groups[sample_file] = group
            
            return sample_groups
        except Exception as e:
            logger.error(f"附件4数据加载失败: {e}")
            return {}
    
    def extract_mixing_info_from_filename(self, filename: str) -> Dict:
        """从文件名提取混合信息"""
        # 解析文件名格式: A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa
        mixing_info = {
            'contributor_ids': [],
            'mixing_ratios': [],
            'total_contributors': 0
        }
        
        try:
            # 查找贡献者ID和混合比例模式
            pattern = r'-(\d+(?:_\d+)*)-(\d+(?:;\d+)*)-'
            match = re.search(pattern, filename)
            
            if match:
                # 提取贡献者ID
                ids_str = match.group(1)
                contributor_ids = [int(id_val) for id_val in ids_str.split('_') if id_val.isdigit()]
                
                # 提取混合比例
                ratios_str = match.group(2)
                mixing_ratios = [int(ratio) for ratio in ratios_str.split(';') if ratio.isdigit()]
                
                if len(contributor_ids) == len(mixing_ratios):
                    # 归一化混合比例
                    total_ratio = sum(mixing_ratios)
                    normalized_ratios = [ratio / total_ratio for ratio in mixing_ratios]
                    
                    mixing_info = {
                        'contributor_ids': contributor_ids,
                        'mixing_ratios': normalized_ratios,
                        'total_contributors': len(contributor_ids),
                        'raw_ratios': mixing_ratios
                    }
        
        except Exception as e:
            logger.warning(f"文件名解析失败 {filename}: {e}")
        
        return mixing_info
    
    def process_sample_peaks(self, sample_data: pd.DataFrame) -> pd.DataFrame:
        """处理样本峰数据"""
        processed_peaks = []
        
        for _, row in sample_data.iterrows():
            sample_file = row['Sample File']
            marker = row['Marker']
            
            # 处理所有峰
            for i in range(1, 101):
                allele = row.get(f'Allele {i}')
                size = row.get(f'Size {i}')
                height = row.get(f'Height {i}')
                
                if pd.notna(allele) and pd.notna(size) and pd.notna(height):
                    if float(height) >= config.HEIGHT_THRESHOLD:
                        processed_peaks.append({
                            'Sample File': sample_file,
                            'Marker': marker,
                            'Allele': str(allele),
                            'Size': float(size),
                            'Height': float(height)
                        })
        
        return pd.DataFrame(processed_peaks)
    
    def prepare_observed_data(self, sample_peaks: pd.DataFrame) -> Dict:
        """准备观测数据"""
        observed_data = {}
        
        for locus, locus_group in sample_peaks.groupby('Marker'):
            alleles = locus_group['Allele'].tolist()
            heights = dict(zip(locus_group['Allele'], locus_group['Height']))
            sizes = dict(zip(locus_group['Allele'], locus_group['Size']))
            
            observed_data[locus] = {
                'locus': locus,
                'alleles': alleles,
                'heights': heights,
                'sizes': sizes
            }
        
        return observed_data
    
    def analyze_sample(self, sample_file: str, sample_data: pd.DataFrame) -> Dict:
        """分析单个样本"""
        logger.info(f"开始UPG-M分析样本: {sample_file}")
        start_time = time.time()
        
        # 提取真实混合信息（如果文件名包含）
        true_mixing_info = self.extract_mixing_info_from_filename(sample_file)
        
        # 处理峰数据
        sample_peaks = self.process_sample_peaks(sample_data)
        
        if sample_peaks.empty:
            logger.warning(f"样本 {sample_file} 没有有效峰数据")
            return self._get_default_result(sample_file, true_mixing_info)
        
        # 噪声检测
        noise_params = self.upg_core.noise_detector.estimate_noise_parameters(sample_peaks)
        
        # 提取V5特征
        v5_features = self.upg_core.v5_estimator.extract_v5_features_enhanced(sample_peaks)
        
        # 准备观测数据
        observed_data = self.prepare_observed_data(sample_peaks)
        
        # UPG-M MCMC推断
        mcmc_results = self.upg_core.mcmc_sampler(observed_data, v5_features, noise_params)
        
        # 后验降噪处理
        denoised_results = self.denoiser.apply_posterior_denoising(
            mcmc_results, observed_data, noise_params)
        
        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(
            denoised_results, true_mixing_info)
        
        end_time = time.time()
        
        result = {
            'sample_file': sample_file,
            'true_mixing_info': true_mixing_info,
            'noise_params': noise_params,
            'v5_features': v5_features,
            'observed_data': observed_data,
            'mcmc_results': mcmc_results,
            'denoised_results': denoised_results,
            'performance_metrics': performance_metrics,
            'computation_time': end_time - start_time
        }
        
        logger.info(f"样本 {sample_file} 分析完成，耗时: {end_time - start_time:.1f}秒")
        return result
    
    def _get_default_result(self, sample_file: str, true_mixing_info: Dict) -> Dict:
        """获取默认结果"""
        return {
            'sample_file': sample_file,
            'true_mixing_info': true_mixing_info,
            'noise_params': {},
            'v5_features': {},
            'observed_data': {},
            'mcmc_results': None,
            'denoised_results': {
                'consensus_noc': {'map_noc': 2, 'confidence': 0.5},
                'denoised_mixture_ratios': {'map_ratios': [0.5, 0.5]},
                'noise_classification': {
                    'true_alleles': {}, 'stutter_artifacts': {}, 'noise_peaks': {}
                }
            },
            'performance_metrics': {},
            'computation_time': 0.0
        }
    
    def _calculate_performance_metrics(self, denoised_results: Dict, 
                                     true_mixing_info: Dict) -> Dict:
        """计算性能指标"""
        metrics = {}
        
        if true_mixing_info.get('total_contributors', 0) > 0:
            # NoC准确性
            true_noc = true_mixing_info['total_contributors']
            predicted_noc = denoised_results['consensus_noc']['map_noc']
            
            metrics['noc_accuracy'] = 1.0 if true_noc == predicted_noc else 0.0
            metrics['noc_error'] = abs(true_noc - predicted_noc)
            
            # 混合比例准确性
            if 'mixing_ratios' in true_mixing_info:
                true_ratios = true_mixing_info['mixing_ratios']
                predicted_ratios = denoised_results['denoised_mixture_ratios']['map_ratios']
                
                if len(true_ratios) == len(predicted_ratios):
                    # 需要处理贡献者排序问题
                    # 简化处理：按比例大小排序后比较
                    true_sorted = sorted(true_ratios, reverse=True)
                    pred_sorted = sorted(predicted_ratios, reverse=True)
                    
                    mse = np.mean([(t - p)**2 for t, p in zip(true_sorted, pred_sorted)])
                    mae = np.mean([abs(t - p) for t, p in zip(true_sorted, pred_sorted)])
                    
                    metrics['mixing_ratio_mse'] = mse
                    metrics['mixing_ratio_mae'] = mae
                    metrics['mixing_ratio_accuracy'] = 1 - mae  # 简化的准确性度量
        
        # 降噪效果指标
        noise_classification = denoised_results['noise_classification']
        total_peaks = (len(noise_classification['true_alleles']) + 
                      len(noise_classification['stutter_artifacts']) + 
                      len(noise_classification['noise_peaks']))
        
        if total_peaks > 0:
            metrics['noise_reduction_rate'] = len(noise_classification['noise_peaks']) / total_peaks
            metrics['stutter_detection_rate'] = len(noise_classification['stutter_artifacts']) / total_peaks
            metrics['true_allele_retention_rate'] = len(noise_classification['true_alleles']) / total_peaks
        
        return metrics
    
    def save_results(self, result: Dict, output_path: str):
        """保存结果"""
        # 简化结果用于JSON序列化
        simplified_result = {
            'sample_file': result['sample_file'],
            'true_mixing_info': result['true_mixing_info'],
            'denoised_results': {
                'consensus_noc': result['denoised_results']['consensus_noc'],
                'denoised_mixture_ratios': result['denoised_results']['denoised_mixture_ratios'],
                'noise_classification_summary': {
                    'true_alleles_count': len(result['denoised_results']['noise_classification']['true_alleles']),
                    'stutter_count': len(result['denoised_results']['noise_classification']['stutter_artifacts']),
                    'noise_peaks_count': len(result['denoised_results']['noise_classification']['noise_peaks'])
                }
            },
            'performance_metrics': result['performance_metrics'],
            'computation_time': result['computation_time'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加MCMC质量信息
        if result['mcmc_results']:
            simplified_result['mcmc_quality'] = {
                'acceptance_rate': result['mcmc_results']['acceptance_rate'],
                'n_samples': result['mcmc_results']['n_samples'],
                'converged': result['mcmc_results']['converged']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_result, f, ensure_ascii=False, indent=2)

# =====================
# 8. 主函数和应用接口
# =====================
def analyze_single_sample_upg(sample_id: str, att4_path: str = None) -> Dict:
    """使用UPG-M方法分析附件4中的单个样本"""
    if att4_path is None:
        att4_path = config.ATTACHMENT4_PATH
    
    # 初始化流水线
    pipeline = UPG_M_Pipeline()
    
    #