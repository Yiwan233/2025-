# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题四：统一概率基因分型降噪系统

版本: V1.0 - UPG-M (Unified Probabilistic Genotyping with Denoising)
日期: 2025-06-09
描述: 基于统一概率基因分型的混合STR图谱降噪算法
核心创新:
1. 统一概率框架同时推断NoC、Mx、基因型和降噪
2. 自适应峰分类器区分信号、Stutter和噪声
3. V5特征驱动的动态参数调整
4. RJMCMC跨维采样处理变维问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import loggamma, gammaln, logit, expit
from scipy.signal import find_peaks, peak_widths
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 100)
print("                     问题四：基于UPG-M的混合STR图谱智能降噪系统")
print("        Unified Probabilistic Genotyping with Machine Learning Denoising")
print("=" * 100)

# =====================
# 1. 系统配置与常量定义
# =====================
class UPGConfig:
    """UPG-M系统配置类"""
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
    MIN_PEAK_HEIGHT_FOR_CONSIDERATION = 30
    CTA_THRESHOLD = 0.5
    PHR_IMBALANCE_THRESHOLD = 0.6
    
    # 降噪参数
    NOISE_THRESHOLD = 100  # 噪声峰高阈值
    SIGNAL_NOISE_RATIO_MIN = 2.0  # 最小信噪比
    STUTTER_RATIO_MAX = 0.15  # 最大Stutter比例
    
    # UPG-M MCMC参数
    N_ITERATIONS = 20000
    N_WARMUP = 8000
    N_CHAINS = 4
    THINNING = 5
    K_TOP = 1000  # 用于N>=4的采样策略
    
    # 跨维MCMC参数
    N_MIN = 1
    N_MAX = 5
    TRANS_DIMENSIONAL_PROB = 0.1  # 跨维转移概率
    
    # 模型参数
    RANDOM_STATE = 42
    N_JOBS = -1

config = UPGConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# =====================
# 2. V5特征工程模块 (继承Q1)
# =====================
class V5FeatureExtractor:
    """V5特征提取器，继承Q1的特征工程"""
    
    def __init__(self):
        self.feature_cache = {}
        logger.info("V5特征提取器初始化完成")
    
    def extract_noc_from_filename(self, filename: str):
        """从文件名提取贡献者人数（NoC）"""
        match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
        if match:
            ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
            return len(ids) if len(ids) > 0 else np.nan
        return np.nan
    
    def extract_mixture_ratios_from_filename(self, filename):
        """从文件名提取真实的贡献者信息和混合比例"""
        # 改进的正则表达式匹配模式
        # 匹配格式：-贡献者IDs-比例(可能包含分号)-
        match = re.search(r'-(\d+(?:_\d+)*)-([^-]*?)-M\d', str(filename))
        if not match:
            logger.warning(f"无法从文件名解析贡献者信息: {filename}")
            return None, None
        
        contributor_ids = match.group(1).split('_')
        ratio_part = match.group(2)
        
        logger.info(f"解析文件名: 贡献者IDs={contributor_ids}, 比例部分='{ratio_part}'")
        
        # 解析比例部分
        if ';' in ratio_part:
            try:
                ratio_values = [float(x) for x in ratio_part.split(';')]
                logger.info(f"解析到的比例值: {ratio_values}")
                
                # 确保比例数量与贡献者数量匹配
                if len(ratio_values) == len(contributor_ids):
                    # 标准化为概率
                    total = sum(ratio_values)
                    if total > 0:
                        true_ratios = [r/total for r in ratio_values]
                        logger.info(f"标准化后的真实比例: {true_ratios}")
                        return contributor_ids, true_ratios
                    else:
                        logger.warning("比例值总和为0")
                else:
                    logger.warning(f"比例数量({len(ratio_values)})与贡献者数量({len(contributor_ids)})不匹配")
            except ValueError as e:
                logger.warning(f"解析比例值失败: {e}")
        else:
            logger.info(f"比例部分无分号，尝试解析单个数值: '{ratio_part}'")
            try:
                # 可能是单个数值，表示主要贡献者的权重
                single_ratio = float(ratio_part)
                if len(contributor_ids) == 1:
                    return contributor_ids, [1.0]
                else:
                    # 如果有多个贡献者但只有一个比例值，可能需要特殊处理
                    logger.warning(f"单个比例值{single_ratio}对应多个贡献者，使用等比例")
            except ValueError:
                logger.warning(f"无法解析比例部分: '{ratio_part}'")
        
        # 如果解析失败，假设等比例
        true_ratios = [1.0/len(contributor_ids)] * len(contributor_ids)
        logger.info(f"使用等比例假设: {true_ratios}")
        return contributor_ids, true_ratios
    
    def calculate_entropy(self, probabilities):
        """计算香农熵"""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) == 0:
            return 0.0
        return -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    def process_peaks_with_metadata(self, sample_data):
        """处理峰数据并保留元数据"""
        processed_data = []
        
        for _, sample_row in sample_data.iterrows():
            sample_file = sample_row['Sample File']
            marker = sample_row['Marker']
            
            # 提取所有峰
            peaks = []
            for i in range(1, 101):
                allele = sample_row.get(f'Allele {i}')
                size = sample_row.get(f'Size {i}')
                height = sample_row.get(f'Height {i}')
                
                if pd.notna(allele) and pd.notna(size) and pd.notna(height):
                    original_height = float(height)
                    
                    # 保留更多原始信息，包括低峰
                    if original_height >= config.MIN_PEAK_HEIGHT_FOR_CONSIDERATION:
                        peaks.append({
                            'allele': str(allele),
                            'size': float(size),
                            'height': min(original_height, config.SATURATION_THRESHOLD),
                            'original_height': original_height,
                            'peak_index': i,
                            'is_saturated': original_height >= config.SATURATION_THRESHOLD
                        })
            
            if not peaks:
                continue
                
            # 计算峰的质量指标
            peaks.sort(key=lambda x: x['height'], reverse=True)
            
            for idx, peak in enumerate(peaks):
                height_rank = idx + 1
                total_peaks = len(peaks)
                max_height = peaks[0]['height']
                
                # 计算相对强度
                relative_intensity = peak['height'] / max_height if max_height > 0 else 0
                
                # 计算初步CTA（用于降噪前的初始评估）
                if height_rank == 1:
                    preliminary_cta = 0.95
                elif height_rank <= 3:
                    preliminary_cta = max(0.3, relative_intensity * 0.9)
                else:
                    preliminary_cta = max(0.1, relative_intensity * 0.6)
                
                processed_data.append({
                    'Sample File': sample_file,
                    'Marker': marker,
                    'Allele': peak['allele'],
                    'Size': peak['size'],
                    'Height': peak['height'],
                    'Original_Height': peak['original_height'],
                    'Peak_Index': peak['peak_index'],
                    'Height_Rank': height_rank,
                    'Relative_Intensity': relative_intensity,
                    'Preliminary_CTA': preliminary_cta,
                    'Is_Saturated': peak['is_saturated']
                })
        
        return pd.DataFrame(processed_data)
    
    def extract_enhanced_v5_features(self, sample_file, sample_peaks):
        """提取增强的V5特征集（降噪专用）"""
        if sample_peaks.empty:
            return self._get_default_features(sample_file)
        
        features = {'Sample File': sample_file}
        
        # 基础数据准备
        total_peaks = len(sample_peaks)
        all_heights = sample_peaks['Height'].values
        all_sizes = sample_peaks['Size'].values
        all_relative_intensities = sample_peaks['Relative_Intensity'].values
        
        # 按位点分组统计
        locus_groups = sample_peaks.groupby('Marker')
        alleles_per_locus = locus_groups['Allele'].nunique()
        locus_heights = locus_groups['Height'].sum()
        
        # A类：图谱层面统计特征（降噪增强版）
        features['mac_profile'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
        features['total_distinct_alleles'] = sample_peaks['Allele'].nunique()
        features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
        
        # MGTN系列
        for N in [3, 4, 5, 6]:
            features[f'loci_gt{N}_alleles'] = (alleles_per_locus >= N).sum()
        
        # 等位基因计数分布熵
        if len(alleles_per_locus) > 0:
            counts = alleles_per_locus.value_counts(normalize=True)
            features['allele_count_dist_entropy'] = self.calculate_entropy(counts.values)
        else:
            features['allele_count_dist_entropy'] = 0
        
        # B类：峰高和信号质量特征（降噪专用）
        if total_peaks > 0:
            features['avg_peak_height'] = np.mean(all_heights)
            features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
            features['min_peak_height'] = np.min(all_heights)
            features['max_peak_height'] = np.max(all_heights)
            features['median_peak_height'] = np.median(all_heights)
            
            # 信号质量指标
            features['signal_to_noise_ratio'] = features['max_peak_height'] / features['min_peak_height'] if features['min_peak_height'] > 0 else 0
            features['dynamic_range'] = np.log10(features['signal_to_noise_ratio'] + 1)
            
            # 峰分布特征
            features['skewness_peak_height'] = stats.skew(all_heights) if total_peaks > 2 else 0
            features['kurtosis_peak_height'] = stats.kurtosis(all_heights, fisher=False) if total_peaks > 3 else 0
            
            # 相对强度分析
            features['avg_relative_intensity'] = np.mean(all_relative_intensities)
            features['std_relative_intensity'] = np.std(all_relative_intensities) if total_peaks > 1 else 0
            
            # 弱信号统计
            weak_signals = (all_heights < config.NOISE_THRESHOLD).sum()
            features['weak_signal_count'] = weak_signals
            features['weak_signal_ratio'] = weak_signals / total_peaks
            
            # 强信号统计
            strong_signals = (all_heights > 1000).sum()
            features['strong_signal_count'] = strong_signals
            features['strong_signal_ratio'] = strong_signals / total_peaks
            
            # PHR相关特征（降噪增强）
            phr_values = []
            imbalance_scores = []
            
            for marker, marker_group in locus_groups:
                if len(marker_group) >= 2:
                    heights = marker_group['Height'].values
                    sizes = marker_group['Size'].values
                    
                    # 计算PHR
                    if len(heights) == 2:
                        phr = min(heights) / max(heights) if max(heights) > 0 else 0
                        phr_values.append(phr)
                    
                    # 计算位点内不平衡度
                    if len(heights) > 1:
                        height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
                        imbalance_scores.append(height_cv)
            
            if phr_values:
                features['avg_phr'] = np.mean(phr_values)
                features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
                features['min_phr'] = np.min(phr_values)
                features['median_phr'] = np.median(phr_values)
                features['num_loci_with_phr'] = len(phr_values)
                features['num_severe_imbalance_loci'] = sum(phr <= config.PHR_IMBALANCE_THRESHOLD for phr in phr_values)
                features['ratio_severe_imbalance_loci'] = features['num_severe_imbalance_loci'] / len(phr_values)
            else:
                for key in ['avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr', 
                           'num_severe_imbalance_loci', 'ratio_severe_imbalance_loci']:
                    features[key] = 0
            
            # 位点不平衡特征
            if imbalance_scores:
                features['avg_locus_imbalance'] = np.mean(imbalance_scores)
                features['max_locus_imbalance'] = np.max(imbalance_scores)
            else:
                features['avg_locus_imbalance'] = 0
                features['max_locus_imbalance'] = 0
            
            # 峰模式复杂度
            try:
                log_heights = np.log(all_heights + 1)
                hist, _ = np.histogram(log_heights, bins=min(10, total_peaks))
                peaks_found, _ = find_peaks(hist)
                features['modality_peak_height'] = len(peaks_found)
            except:
                features['modality_peak_height'] = 1
                
            # 饱和效应
            saturated_peaks = sample_peaks['Is_Saturated'].sum()
            features['num_saturated_peaks'] = saturated_peaks
            features['ratio_saturated_peaks'] = saturated_peaks / total_peaks
        
        # C类：信息论特征（降噪优化）
        if len(locus_heights) > 0:
            total_height = locus_heights.sum()
            if total_height > 0:
                locus_probs = locus_heights / total_height
                features['inter_locus_balance_entropy'] = self.calculate_entropy(locus_probs.values)
            else:
                features['inter_locus_balance_entropy'] = 0
            
            # 平均位点等位基因分布熵
            locus_entropies = []
            for marker, marker_group in locus_groups:
                if len(marker_group) > 1:
                    heights = marker_group['Height'].values
                    height_sum = heights.sum()
                    if height_sum > 0:
                        probs = heights / height_sum
                        entropy = self.calculate_entropy(probs)
                        locus_entropies.append(entropy)
            
            features['avg_locus_allele_entropy'] = np.mean(locus_entropies) if locus_entropies else 0
            
            # 峰高分布熵
            if total_peaks > 0:
                log_heights = np.log(all_heights + 1)
                hist, _ = np.histogram(log_heights, bins=min(15, total_peaks))
                hist_probs = hist / hist.sum()
                hist_probs = hist_probs[hist_probs > 0]
                features['peak_height_entropy'] = self.calculate_entropy(hist_probs)
            else:
                features['peak_height_entropy'] = 0
            
            # 图谱完整性
            effective_loci_count = len(locus_groups)
            features['num_loci_with_effective_alleles'] = effective_loci_count
            features['num_loci_no_effective_alleles'] = max(0, 20 - effective_loci_count)
        
        # D类：DNA降解与技术噪声特征
        if total_peaks > 1 and len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            try:
                corr_matrix = np.corrcoef(all_heights, all_sizes)
                features['height_size_correlation'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0
            except:
                features['height_size_correlation'] = 0
            
            # 降解斜率
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(all_sizes, all_heights)
                features['height_size_slope'] = slope
                features['degradation_r_squared'] = r_value ** 2
            except:
                features['height_size_slope'] = 0
                features['degradation_r_squared'] = 0
        else:
            features['height_size_correlation'] = 0
            features['height_size_slope'] = 0
            features['degradation_r_squared'] = 0
        
        # E类：降噪专用特征
        # 噪声模式识别
        if total_peaks > 0:
            # 基线噪声估计
            baseline_noise = np.percentile(all_heights, 10)
            features['baseline_noise_level'] = baseline_noise
            features['signal_above_baseline'] = np.mean(all_heights > baseline_noise * 3)
            
            # 异常峰检测
            height_median = np.median(all_heights)
            height_mad = np.median(np.abs(all_heights - height_median))
            outlier_threshold = height_median + 3 * height_mad
            features['outlier_peak_count'] = np.sum(all_heights > outlier_threshold)
            features['outlier_peak_ratio'] = features['outlier_peak_count'] / total_peaks
            
            # 信号一致性
            features['signal_consistency'] = 1 - (features['std_peak_height'] / features['avg_peak_height']) if features['avg_peak_height'] > 0 else 0
        
        return features
    
    def _get_default_features(self, sample_file):
        """获取默认特征值"""
        default_features = {
            'Sample File': sample_file,
            'mac_profile': 0, 'total_distinct_alleles': 0, 'avg_alleles_per_locus': 0,
            'std_alleles_per_locus': 0, 'loci_gt3_alleles': 0, 'loci_gt4_alleles': 0,
            'loci_gt5_alleles': 0, 'loci_gt6_alleles': 0, 'allele_count_dist_entropy': 0,
            'avg_peak_height': 0, 'std_peak_height': 0, 'min_peak_height': 0,
            'max_peak_height': 0, 'median_peak_height': 0, 'signal_to_noise_ratio': 0,
            'dynamic_range': 0, 'skewness_peak_height': 0, 'kurtosis_peak_height': 0,
            'avg_relative_intensity': 0, 'std_relative_intensity': 0,
            'weak_signal_count': 0, 'weak_signal_ratio': 0,
            'strong_signal_count': 0, 'strong_signal_ratio': 0,
            'avg_phr': 0, 'std_phr': 0, 'min_phr': 0, 'median_phr': 0,
            'num_loci_with_phr': 0, 'num_severe_imbalance_loci': 0,
            'ratio_severe_imbalance_loci': 0, 'avg_locus_imbalance': 0,
            'max_locus_imbalance': 0, 'modality_peak_height': 1,
            'num_saturated_peaks': 0, 'ratio_saturated_peaks': 0,
            'inter_locus_balance_entropy': 0, 'avg_locus_allele_entropy': 0,
            'peak_height_entropy': 0, 'num_loci_with_effective_alleles': 0,
            'num_loci_no_effective_alleles': 20, 'height_size_correlation': 0,
            'height_size_slope': 0, 'degradation_r_squared': 0,
            'baseline_noise_level': 0, 'signal_above_baseline': 0,
            'outlier_peak_count': 0, 'outlier_peak_ratio': 0, 'signal_consistency': 0
        }
        return default_features

# =====================
# 3. 智能峰分类器 (基于梯度提升)
# =====================
class IntelligentPeakClassifier:
    """智能峰分类器，区分真实信号、Stutter和噪声"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        logger.info("智能峰分类器初始化完成")
    
    def extract_peak_features(self, peak_data, locus_context):
        """提取单个峰的特征用于分类"""
        features = {}
        
        # 基础峰特征
        features['height'] = peak_data['Height']
        features['size'] = peak_data['Size']
        features['relative_intensity'] = peak_data['Relative_Intensity']
        features['height_rank'] = peak_data['Height_Rank']
        features['is_saturated'] = int(peak_data['Is_Saturated'])
        
        # 上下文特征（位点层面）
        locus_peaks = locus_context
        features['locus_peak_count'] = len(locus_peaks)
        features['locus_max_height'] = locus_peaks['Height'].max()
        features['locus_mean_height'] = locus_peaks['Height'].mean()
        features['locus_height_cv'] = locus_peaks['Height'].std() / locus_peaks['Height'].mean() if locus_peaks['Height'].mean() > 0 else 0
        
        # 相对位置特征
        all_sizes_in_locus = sorted(locus_peaks['Size'].tolist())
        current_size = peak_data['Size']
        
        if len(all_sizes_in_locus) > 1:
            size_diffs = [abs(current_size - s) for s in all_sizes_in_locus if s != current_size]
            features['min_size_diff'] = min(size_diffs) if size_diffs else 0
            features['is_size_outlier'] = int(features['min_size_diff'] > 20)  # 20bp阈值
        else:
            features['min_size_diff'] = 0
            features['is_size_outlier'] = 0
        
        # Stutter模式特征
        potential_parent_heights = []
        for _, other_peak in locus_peaks.iterrows():
            if other_peak['Size'] > current_size:
                size_diff = other_peak['Size'] - current_size
                if 3 <= size_diff <= 5:  # 典型的n-1 stutter
                    potential_parent_heights.append(other_peak['Height'])
        
        if potential_parent_heights:
            max_parent_height = max(potential_parent_heights)
            features['stutter_ratio'] = peak_data['Height'] / max_parent_height
            features['has_potential_parent'] = 1
        else:
            features['stutter_ratio'] = 0
            features['has_potential_parent'] = 0
        
        # 噪声模式特征
        features['height_log'] = np.log10(peak_data['Height'] + 1)
        features['size_normalized'] = (current_size - 100) / 300  # 标准化片段大小
        
        # 信号质量评估
        features['signal_quality_score'] = self._calculate_signal_quality(peak_data, locus_context)
        
        return features
    
    def _calculate_signal_quality(self, peak_data, locus_context):
        """计算信号质量评分"""
        score = 0.0
        
        # 峰高权重
        height = peak_data['Height']
        if height > 1000:
            score += 0.3
        elif height > 500:
            score += 0.2
        elif height > 200:
            score += 0.1
        
        # 相对强度权重
        rel_intensity = peak_data['Relative_Intensity']
        if rel_intensity > 0.5:
            score += 0.2
        elif rel_intensity > 0.2:
            score += 0.1
        
        # 位点一致性权重
        locus_peak_count = len(locus_context)
        if locus_peak_count <= 4:  # 合理的等位基因数
            score += 0.2
        elif locus_peak_count <= 6:
            score += 0.1
        
        # 片段大小合理性
        size = peak_data['Size']
        if 100 <= size <= 500:  # 合理的STR片段大小范围
            score += 0.2
        
        # 饱和度惩罚
        if peak_data['Is_Saturated']:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def prepare_training_data(self, processed_peaks, ground_truth_labels=None):
        """准备训练数据"""
        training_features = []
        training_labels = []
        
        # 按样本和位点分组
        for (sample_file, marker), locus_group in processed_peaks.groupby(['Sample File', 'Marker']):
            for _, peak in locus_group.iterrows():
                # 提取特征
                peak_features = self.extract_peak_features(peak, locus_group)
                training_features.append(list(peak_features.values()))
                
                if not self.feature_names:
                    self.feature_names = list(peak_features.keys())
                
                # 分配标签（如果没有真实标签，使用启发式规则）
                if ground_truth_labels is not None:
                    # 使用真实标签
                    label = ground_truth_labels.get((sample_file, marker, peak['Allele']), 'Unknown')
                else:
                    # 使用启发式规则生成伪标签
                    label = self._assign_heuristic_label(peak, locus_group)
                
                training_labels.append(label)
        
        return np.array(training_features), np.array(training_labels)
    
    def _assign_heuristic_label(self, peak, locus_group):
        """使用启发式规则分配标签"""
        height = peak['Height']
        rel_intensity = peak['Relative_Intensity']
        stutter_ratio = 0
        
        # 检查是否为潜在的Stutter
        for _, other_peak in locus_group.iterrows():
            if other_peak['Size'] > peak['Size']:
                size_diff = other_peak['Size'] - peak['Size']
                if 3 <= size_diff <= 5:  # n-1 stutter
                    stutter_ratio = height / other_peak['Height']
                    break
        
        # 分类规则
        if height < config.NOISE_THRESHOLD and rel_intensity < 0.1:
            return 'Noise'
        elif 0 < stutter_ratio <= config.STUTTER_RATIO_MAX:
            return 'Stutter'
        elif height >= config.HEIGHT_THRESHOLD and rel_intensity > 0.2:
            return 'True_Allele'
        else:
            return 'Unknown'
    
    def train_classifier(self, training_features, training_labels):
        """训练峰分类器"""
        logger.info("开始训练智能峰分类器...")
        
        # 过滤掉Unknown标签
        valid_indices = training_labels != 'Unknown'
        X = training_features[valid_indices]
        y = training_labels[valid_indices]
        
        if len(X) == 0:
            logger.warning("没有有效的训练数据")
            return
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
        )
        
        # 训练梯度提升分类器
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=config.RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"峰分类器训练完成，测试准确率: {accuracy:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("前10个重要特征:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def classify_peak(self, peak_data, locus_context):
        """分类单个峰"""
        if not self.is_trained:
            # 如果模型未训练，使用启发式规则
            return self._assign_heuristic_label(peak_data, locus_context), 0.5
        
        # 提取特征
        peak_features = self.extract_peak_features(peak_data, locus_context)
        X = np.array(list(peak_features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # 预测
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def classify_sample_peaks(self, processed_peaks):
        """对样本的所有峰进行分类"""
        results = []
        
        for (sample_file, marker), locus_group in processed_peaks.groupby(['Sample File', 'Marker']):
            for _, peak in locus_group.iterrows():
                classification, confidence = self.classify_peak(peak, locus_group)
                
                result = {
                    'Sample File': sample_file,
                    'Marker': marker,
                    'Allele': peak['Allele'],
                    'Height': peak['Height'],
                    'Size': peak['Size'],
                    'Classification': classification,
                    'Confidence': confidence,
                    'Original_CTA': peak.get('Preliminary_CTA', 0.5)
                }
                results.append(result)
        
        return pd.DataFrame(results)

# =====================
# 4. 自适应CTA计算器
# =====================
class AdaptiveCTACalculator:
    """自适应CTA计算器，基于峰分类结果动态调整CTA值"""
    
    def __init__(self, peak_classifier):
        self.peak_classifier = peak_classifier
        self.v5_integrator = None
        logger.info("自适应CTA计算器初始化完成")
    
    def set_v5_features(self, v5_features):
        """设置V5特征"""
        self.v5_integrator = V5FeatureIntegrator(v5_features)
    
    def calculate_adaptive_cta(self, peak_data, locus_context, sample_v5_features):
        """计算自适应CTA值"""
        # 峰分类结果
        classification, confidence = self.peak_classifier.classify_peak(peak_data, locus_context)
        
        # 基础CTA计算
        base_cta = self._calculate_base_cta(peak_data, locus_context)
        
        # 基于分类结果的调整
        if classification == 'True_Allele':
            classification_weight = 1.0 + 0.3 * confidence
        elif classification == 'Stutter':
            classification_weight = 0.3 + 0.2 * confidence
        elif classification == 'Noise':
            classification_weight = 0.1 * (1 - confidence)
        else:
            classification_weight = 0.5
        
        # 基于V5特征的调整
        v5_weight = self._calculate_v5_adjustment(sample_v5_features, peak_data)
        
        # 基于位点上下文的调整
        context_weight = self._calculate_context_adjustment(peak_data, locus_context)
        
        # 综合CTA计算
        adaptive_cta = base_cta * classification_weight * v5_weight * context_weight
        adaptive_cta = np.clip(adaptive_cta, 0.0, 1.0)
        
        return adaptive_cta, {
            'base_cta': base_cta,
            'classification': classification,
            'classification_confidence': confidence,
            'classification_weight': classification_weight,
            'v5_weight': v5_weight,
            'context_weight': context_weight
        }
    
    def _calculate_base_cta(self, peak_data, locus_context):
        """计算基础CTA值"""
        height = peak_data['Height']
        rel_intensity = peak_data['Relative_Intensity']
        height_rank = peak_data['Height_Rank']
        
        # 高度权重
        if height > 2000:
            height_weight = 0.9
        elif height > 1000:
            height_weight = 0.8
        elif height > 500:
            height_weight = 0.6
        elif height > 200:
            height_weight = 0.4
        else:
            height_weight = 0.2
        
        # 相对强度权重
        intensity_weight = min(0.9, rel_intensity * 1.5)
        
        # 排名权重
        if height_rank == 1:
            rank_weight = 0.95
        elif height_rank == 2:
            rank_weight = 0.8
        elif height_rank <= 4:
            rank_weight = 0.6
        else:
            rank_weight = 0.3
        
        # 综合基础CTA
        base_cta = (height_weight * 0.4 + intensity_weight * 0.4 + rank_weight * 0.2)
        
        return base_cta
    
    def _calculate_v5_adjustment(self, v5_features, peak_data):
        """基于V5特征计算调整权重"""
        if not v5_features:
            return 1.0
        
        # 信噪比调整
        snr = v5_features.get('signal_to_noise_ratio', 1.0)
        snr_weight = min(1.2, 0.8 + 0.4 * np.log10(snr + 1))
        
        # 降解调整
        degradation_corr = abs(v5_features.get('height_size_correlation', 0.0))
        if degradation_corr > 0.3:  # 明显降解
            degradation_weight = 0.9
        elif degradation_corr > 0.1:
            degradation_weight = 0.95
        else:
            degradation_weight = 1.0
        
        # 样本质量调整
        weak_signal_ratio = v5_features.get('weak_signal_ratio', 0.0)
        quality_weight = 1.0 - 0.3 * weak_signal_ratio
        
        return snr_weight * degradation_weight * quality_weight
    
    def _calculate_context_adjustment(self, peak_data, locus_context):
        """基于位点上下文计算调整权重"""
        locus_peak_count = len(locus_context)
        
        # 位点等位基因数合理性
        if locus_peak_count <= 2:
            count_weight = 1.0
        elif locus_peak_count <= 4:
            count_weight = 0.9
        elif locus_peak_count <= 6:
            count_weight = 0.7
        else:
            count_weight = 0.5  # 过多等位基因，可能有噪声
        
        # 位点内平衡性
        heights = locus_context['Height'].values
        if len(heights) > 1:
            height_cv = np.std(heights) / np.mean(heights)
            if height_cv < 0.5:  # 较好的平衡性
                balance_weight = 1.0
            elif height_cv < 1.0:
                balance_weight = 0.9
            else:
                balance_weight = 0.8
        else:
            balance_weight = 1.0
        
        return count_weight * balance_weight

class V5FeatureIntegrator:
    """V5特征集成器（降噪专用）"""
    
    def __init__(self, v5_features):
        self.v5_features = v5_features
    
    def calculate_noise_threshold(self, locus: str) -> float:
        """基于V5特征计算动态噪声阈值"""
        base_threshold = config.NOISE_THRESHOLD
        
        # 基于平均峰高调整
        avg_height = self.v5_features.get('avg_peak_height', 1000.0)
        height_factor = max(0.5, min(2.0, avg_height / 1000.0))
        
        # 基于信噪比调整
        snr = self.v5_features.get('signal_to_noise_ratio', 10.0)
        snr_factor = max(0.8, min(1.5, np.log10(snr + 1)))
        
        # 基于样本质量调整
        weak_signal_ratio = self.v5_features.get('weak_signal_ratio', 0.1)
        quality_factor = 1.0 - 0.3 * weak_signal_ratio
        
        dynamic_threshold = base_threshold * height_factor * snr_factor * quality_factor
        
        return max(30, min(200, dynamic_threshold))

# =====================
# 5. UPG-M核心推断引擎
# =====================
class UPGMInferenceEngine:
    """统一概率基因分型推断引擎"""
    
    def __init__(self):
        self.peak_classifier = IntelligentPeakClassifier()
        self.cta_calculator = AdaptiveCTACalculator(self.peak_classifier)
        self.v5_extractor = V5FeatureExtractor()
        self.current_v5_features = None
        
        # MCMC参数
        self.n_iterations = config.N_ITERATIONS
        self.n_warmup = config.N_WARMUP
        self.n_chains = config.N_CHAINS
        self.thinning = config.THINNING
        
        logger.info("UPG-M推断引擎初始化完成")
    
    def preprocess_sample(self, sample_data):
        """预处理样本数据"""
        sample_file = sample_data.iloc[0]['Sample File']
        logger.info(f"预处理样本: {sample_file}")
        
        # 处理峰数据
        processed_peaks = self.v5_extractor.process_peaks_with_metadata(sample_data)
        
        if processed_peaks.empty:
            logger.warning(f"样本 {sample_file} 没有有效峰数据")
            return None, None, None
        
        # 提取V5特征
        v5_features = self.v5_extractor.extract_enhanced_v5_features(sample_file, processed_peaks)
        self.current_v5_features = v5_features
        self.cta_calculator.set_v5_features(v5_features)
        
        # 峰分类（如果分类器已训练）
        peak_classifications = self.peak_classifier.classify_sample_peaks(processed_peaks)
        
        return processed_peaks, v5_features, peak_classifications
    
    def train_peak_classifier_on_dataset(self, dataset_path):
        """在数据集上训练峰分类器"""
        logger.info(f"在数据集上训练峰分类器: {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path, encoding='utf-8')
            
            all_training_features = []
            all_training_labels = []
            
            # 处理所有样本
            for sample_file, sample_group in df.groupby('Sample File'):
                processed_peaks = self.v5_extractor.process_peaks_with_metadata(sample_group)
                
                if not processed_peaks.empty:
                    # 准备训练数据
                    features, labels = self.peak_classifier.prepare_training_data(processed_peaks)
                    all_training_features.extend(features)
                    all_training_labels.extend(labels)
            
            if all_training_features:
                training_features = np.array(all_training_features)
                training_labels = np.array(all_training_labels)
                
                # 训练分类器
                training_results = self.peak_classifier.train_classifier(training_features, training_labels)
                logger.info("峰分类器训练完成")
                return training_results
            else:
                logger.warning("没有足够的训练数据")
                return None
                
        except Exception as e:
            logger.error(f"训练峰分类器失败: {e}")
            return None
    
    def denoise_sample(self, processed_peaks, peak_classifications):
        """对样本进行降噪处理"""
        if processed_peaks.empty:
            return processed_peaks
        
        denoised_peaks = []
        
        # 按位点处理
        for (sample_file, marker), locus_group in processed_peaks.groupby(['Sample File', 'Marker']):
            locus_denoised = []
            
            for _, peak in locus_group.iterrows():
                # 计算自适应CTA
                adaptive_cta, cta_details = self.cta_calculator.calculate_adaptive_cta(
                    peak, locus_group, self.current_v5_features
                )
                
                # 应用CTA阈值
                if adaptive_cta >= config.CTA_THRESHOLD:
                    peak_copy = peak.copy()
                    peak_copy['Adaptive_CTA'] = adaptive_cta
                    peak_copy['CTA_Details'] = str(cta_details)
                    peak_copy['Denoising_Status'] = 'Retained'
                    locus_denoised.append(peak_copy)
                else:
                    # 记录被过滤的峰
                    peak_copy = peak.copy()
                    peak_copy['Adaptive_CTA'] = adaptive_cta
                    peak_copy['CTA_Details'] = str(cta_details)
                    peak_copy['Denoising_Status'] = 'Filtered'
                    # 可选择性保留，用于分析
            
            denoised_peaks.extend(locus_denoised)
        
        denoised_df = pd.DataFrame(denoised_peaks) if denoised_peaks else pd.DataFrame()
        
        logger.info(f"降噪前峰数: {len(processed_peaks)}, 降噪后峰数: {len(denoised_df)}")
        
        return denoised_df
    
    def calculate_likelihood(self, observed_data, N, mixture_ratios, genotype_sets, theta):
        """计算观测数据的似然函数"""
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self._calculate_locus_likelihood(
                locus_data, N, mixture_ratios, genotype_sets.get(locus, []), theta, locus
            )
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def _calculate_locus_likelihood(self, locus_data, N, mixture_ratios, genotype_set, theta, locus):
        """计算单个位点的似然函数"""
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        observed_ctas = locus_data.get('ctas', {})
        
        log_likelihood = 0.0
        
        # 为每个观测等位基因计算似然
        for allele in observed_alleles:
            height = observed_heights.get(allele, 0.0)
            cta = observed_ctas.get(allele, 0.5)
            
            if height > 0:
                # 期望峰高计算
                expected_height = self._calculate_expected_height(
                    allele, locus, genotype_set, mixture_ratios, theta
                )
                
                if expected_height > 1e-6:
                    # 对数正态分布似然
                    sigma = theta.get('sigma_var', {}).get(locus, 0.3)
                    log_mu = np.log(expected_height) - sigma**2 / 2
                    
                    # CTA权重调整
                    weighted_height = height * cta
                    log_likelihood += stats.lognorm.logpdf(weighted_height, sigma, scale=np.exp(log_mu))
                else:
                    log_likelihood += -1e6  # 惩罚项
        
        return log_likelihood
    
    def _calculate_expected_height(self, allele, locus, genotype_set, mixture_ratios, theta):
        """计算等位基因的期望峰高"""
        gamma_l = theta.get('gamma', {}).get(locus, 1000.0)
        
        expected_height = 0.0
        
        # 计算直接贡献
        for i, genotype in enumerate(genotype_set):
            if genotype is not None:
                copy_number = self._get_copy_number(allele, genotype)
                if copy_number > 0:
                    # 降解因子
                    allele_size = self._get_allele_size(allele)
                    degradation_factor = self._calculate_degradation_factor(allele_size, theta)
                    
                    expected_height += gamma_l * mixture_ratios[i] * copy_number * degradation_factor
        
        # 添加Stutter贡献
        stutter_contribution = self._calculate_stutter_contribution(
            allele, locus, genotype_set, mixture_ratios, theta
        )
        
        return expected_height + stutter_contribution
    
    def _get_copy_number(self, allele, genotype):
        """获取等位基因在基因型中的拷贝数"""
        if genotype is None:
            return 0.0
        return float(genotype.count(allele))
    
    def _get_allele_size(self, allele):
        """获取等位基因的片段大小"""
        try:
            # 简化的大小计算
            allele_num = float(allele.split('.')[0])
            return 150.0 + allele_num * 4.0
        except:
            return 200.0
    
    def _calculate_degradation_factor(self, size, theta):
        """计算降解因子"""
        k_deg = theta.get('k_degradation', 0.001)
        reference_size = 200.0
        return np.exp(-k_deg * max(0, size - reference_size))
    
    def _calculate_stutter_contribution(self, allele, locus, genotype_set, mixture_ratios, theta):
        """计算Stutter贡献"""
        stutter_ratios = theta.get('stutter_ratios', {})
        default_stutter_ratio = 0.05
        
        try:
            allele_num = float(allele.split('.')[0])
            parent_allele_num = allele_num + 1
            parent_allele = str(int(parent_allele_num)) if parent_allele_num.is_integer() else str(parent_allele_num)
            
            parent_expected = 0.0
            gamma_l = theta.get('gamma', {}).get(locus, 1000.0)
            
            for i, genotype in enumerate(genotype_set):
                if genotype is not None:
                    copy_number = self._get_copy_number(parent_allele, genotype)
                    if copy_number > 0:
                        parent_size = self._get_allele_size(parent_allele)
                        degradation_factor = self._calculate_degradation_factor(parent_size, theta)
                        parent_expected += gamma_l * mixture_ratios[i] * copy_number * degradation_factor
            
            stutter_ratio = stutter_ratios.get(locus, default_stutter_ratio)
            return stutter_ratio * parent_expected
            
        except:
            return 0.0
    
    def mcmc_sampler(self, observed_data, initial_params=None):
        """UPG-M的RJMCMC采样器"""
        logger.info("开始UPG-M RJMCMC采样")
        
        # 初始化参数
        if initial_params is None:
            current_N = 2
            current_Mx = np.array([0.5, 0.5])
            current_genotypes = self._initialize_genotypes(observed_data, current_N)
            current_theta = self._initialize_theta(observed_data)
        else:
            current_N = initial_params['N']
            current_Mx = initial_params['Mx']
            current_genotypes = initial_params['genotypes']
            current_theta = initial_params['theta']
        
        # 计算初始似然
        current_log_likelihood = self.calculate_likelihood(
            observed_data, current_N, current_Mx, current_genotypes, current_theta
        )
        current_log_prior = self._calculate_prior(current_N, current_Mx, current_genotypes, current_theta)
        current_log_posterior = current_log_likelihood + current_log_prior
        
        # 存储样本
        samples = {
            'N': [],
            'Mx': [],
            'log_likelihood': [],
            'log_posterior': [],
            'acceptance_info': []
        }
        
        n_accepted = 0
        n_trans_dimensional_attempts = 0
        n_trans_dimensional_accepted = 0
        
        # MCMC主循环
        for iteration in range(self.n_iterations):
            if iteration % 2000 == 0:
                acceptance_rate = n_accepted / max(iteration, 1)
                logger.info(f"RJMCMC迭代 {iteration}/{self.n_iterations}, "
                          f"接受率: {acceptance_rate:.3f}, "
                          f"当前N: {current_N}, "
                          f"当前似然: {current_log_likelihood:.2f}")
            
            # 决定更新类型
            if np.random.random() < config.TRANS_DIMENSIONAL_PROB and iteration > self.n_warmup // 2:
                # 跨维移动（改变N）
                n_trans_dimensional_attempts += 1
                proposed_N, proposed_Mx, proposed_genotypes, log_proposal_ratio = self._propose_trans_dimensional_move(
                    current_N, current_Mx, current_genotypes, observed_data
                )
                
                if proposed_N is not None:
                    # 计算提议状态的概率
                    proposed_log_likelihood = self.calculate_likelihood(
                        observed_data, proposed_N, proposed_Mx, proposed_genotypes, current_theta
                    )
                    proposed_log_prior = self._calculate_prior(
                        proposed_N, proposed_Mx, proposed_genotypes, current_theta
                    )
                    proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
                    
                    # RJMCMC接受概率
                    log_ratio = (proposed_log_posterior - current_log_posterior + log_proposal_ratio)
                    accept_prob = min(1.0, np.exp(log_ratio))
                    
                    if np.random.random() < accept_prob:
                        current_N = proposed_N
                        current_Mx = proposed_Mx
                        current_genotypes = proposed_genotypes
                        current_log_likelihood = proposed_log_likelihood
                        current_log_prior = proposed_log_prior
                        current_log_posterior = proposed_log_posterior
                        n_accepted += 1
                        n_trans_dimensional_accepted += 1
                        accepted = True
                    else:
                        accepted = False
                else:
                    accepted = False
            
            else:
                # 固定维度移动
                update_type = np.random.choice(['Mx', 'genotypes', 'theta'], p=[0.4, 0.4, 0.2])
                
                if update_type == 'Mx':
                    proposed_Mx = self._propose_mixture_ratios(current_Mx)
                    proposed_log_likelihood = self.calculate_likelihood(
                        observed_data, current_N, proposed_Mx, current_genotypes, current_theta
                    )
                    proposed_log_prior = self._calculate_prior(
                        current_N, proposed_Mx, current_genotypes, current_theta
                    )
                    
                elif update_type == 'genotypes':
                    proposed_genotypes = self._propose_genotypes(current_genotypes, observed_data, current_N)
                    proposed_log_likelihood = self.calculate_likelihood(
                        observed_data, current_N, current_Mx, proposed_genotypes, current_theta
                    )
                    proposed_log_prior = self._calculate_prior(
                        current_N, current_Mx, proposed_genotypes, current_theta
                    )
                    
                else:  # theta
                    proposed_theta = self._propose_theta(current_theta)
                    proposed_log_likelihood = self.calculate_likelihood(
                        observed_data, current_N, current_Mx, current_genotypes, proposed_theta
                    )
                    proposed_log_prior = self._calculate_prior(
                        current_N, current_Mx, current_genotypes, proposed_theta
                    )
                
                proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
                
                # Metropolis-Hastings接受/拒绝
                log_ratio = proposed_log_posterior - current_log_posterior
                accept_prob = min(1.0, np.exp(log_ratio))
                
                if np.random.random() < accept_prob:
                    if update_type == 'Mx':
                        current_Mx = proposed_Mx
                    elif update_type == 'genotypes':
                        current_genotypes = proposed_genotypes
                    else:
                        current_theta = proposed_theta
                    
                    current_log_likelihood = proposed_log_likelihood
                    current_log_prior = proposed_log_prior
                    current_log_posterior = proposed_log_posterior
                    n_accepted += 1
                    accepted = True
                else:
                    accepted = False
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['N'].append(current_N)
                samples['Mx'].append(current_Mx.copy())
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
        
        final_acceptance_rate = n_accepted / self.n_iterations
        trans_dimensional_acceptance_rate = (n_trans_dimensional_accepted / max(n_trans_dimensional_attempts, 1) 
                                           if n_trans_dimensional_attempts > 0 else 0)
        
        logger.info(f"RJMCMC完成，总接受率: {final_acceptance_rate:.3f}")
        logger.info(f"跨维接受率: {trans_dimensional_acceptance_rate:.3f}")
        logger.info(f"有效样本数: {len(samples['N'])}")
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'trans_dimensional_acceptance_rate': trans_dimensional_acceptance_rate,
            'n_samples': len(samples['N']),
            'converged': 0.15 <= final_acceptance_rate <= 0.6
        }
    
    def _initialize_genotypes(self, observed_data, N):
        """初始化基因型"""
        genotypes = {}
        
        for locus, locus_data in observed_data.items():
            alleles = locus_data['alleles']
            locus_genotypes = []
            
            for i in range(N):
                if len(alleles) >= 2:
                    # 随机选择两个等位基因
                    genotype = tuple(np.random.choice(alleles, 2, replace=True))
                elif len(alleles) == 1:
                    genotype = (alleles[0], alleles[0])
                else:
                    genotype = None
                
                locus_genotypes.append(genotype)
            
            genotypes[locus] = locus_genotypes
        
        return genotypes
    
    def _initialize_theta(self, observed_data):
        """初始化模型参数"""
        theta = {
            'gamma': {},
            'sigma_var': {},
            'stutter_ratios': {},
            'k_degradation': 0.001
        }
        
        for locus, locus_data in observed_data.items():
            heights = list(locus_data['heights'].values())
            
            if heights:
                avg_height = np.mean(heights)
                theta['gamma'][locus] = avg_height * 0.8
                theta['sigma_var'][locus] = 0.3
                theta['stutter_ratios'][locus] = 0.05
            else:
                theta['gamma'][locus] = 1000.0
                theta['sigma_var'][locus] = 0.3
                theta['stutter_ratios'][locus] = 0.05
        
        return theta
    
    def _calculate_prior(self, N, Mx, genotypes, theta):
        """计算先验概率"""
        log_prior = 0.0
        
        # N的先验（基于V5特征预测）
        if self.current_v5_features:
            # 这里可以集成Q1的NoC预测结果
            predicted_noc = self._predict_noc_from_v5(self.current_v5_features)
            if N == predicted_noc:
                log_prior += np.log(0.6)
            else:
                log_prior += np.log(0.4 / (config.N_MAX - 1))
        else:
            # 均匀先验
            log_prior += np.log(1.0 / config.N_MAX)
        
        # Mx的Dirichlet先验
        alpha = np.ones(N) * 1.0
        log_prior += (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                     np.sum((alpha - 1) * np.log(Mx + 1e-10)))
        
        # 基因型的先验（基于HWE）
        for locus, locus_genotypes in genotypes.items():
            for genotype in locus_genotypes:
                if genotype is not None:
                    # 简化的HWE先验
                    log_prior += np.log(0.25)  # 均匀先验
        
        # theta参数的先验
        for locus in theta['gamma']:
            # gamma的对数正态先验
            log_prior += stats.lognorm.logpdf(theta['gamma'][locus], s=0.5, scale=1000.0)
            # sigma的逆伽马先验
            log_prior += stats.invgamma.logpdf(theta['sigma_var'][locus], a=2, scale=0.1)
        
        return log_prior
    
    def _predict_noc_from_v5(self, v5_features):
        """基于V5特征预测NoC（简化版本）"""
        # 这里可以集成Q1训练的模型
        mac_profile = v5_features.get('mac_profile', 2)
        avg_alleles = v5_features.get('avg_alleles_per_locus', 2)
        
        # 简化的启发式规则
        if mac_profile <= 2 and avg_alleles <= 2.2:
            return 1
        elif mac_profile <= 3 and avg_alleles <= 2.8:
            return 2
        elif mac_profile <= 5 and avg_alleles <= 3.5:
            return 3
        elif mac_profile <= 7 and avg_alleles <= 4.2:
            return 4
        else:
            return 5
    
    def _propose_trans_dimensional_move(self, current_N, current_Mx, current_genotypes, observed_data):
        """提议跨维移动"""
        # 决定增加还是减少贡献者
        if current_N == config.N_MIN:
            move_type = 'birth'
        elif current_N == config.N_MAX:
            move_type = 'death'
        else:
            move_type = np.random.choice(['birth', 'death'])
        
        if move_type == 'birth':
            # 增加一个贡献者
            proposed_N = current_N + 1
            
            # 拆分一个现有的混合比例
            split_idx = np.random.randint(current_N)
            split_ratio = np.random.beta(1, 1)  # 拆分比例
            
            proposed_Mx = np.zeros(proposed_N)
            for i in range(current_N):
                if i == split_idx:
                    proposed_Mx[i] = current_Mx[i] * split_ratio
                    proposed_Mx[current_N] = current_Mx[i] * (1 - split_ratio)
                elif i < split_idx:
                    proposed_Mx[i] = current_Mx[i]
                else:
                    proposed_Mx[i] = current_Mx[i]
            
            # 为新贡献者生成基因型
            proposed_genotypes = {}
            for locus, locus_genotypes in current_genotypes.items():
                new_locus_genotypes = locus_genotypes.copy()
                
                # 为新贡献者随机生成基因型
                alleles = observed_data[locus]['alleles']
                if len(alleles) >= 2:
                    new_genotype = tuple(np.random.choice(alleles, 2, replace=True))
                elif len(alleles) == 1:
                    new_genotype = (alleles[0], alleles[0])
                else:
                    new_genotype = None
                
                new_locus_genotypes.append(new_genotype)
                proposed_genotypes[locus] = new_locus_genotypes
            
            # 计算提议比率（birth move的雅可比行列式）
            log_proposal_ratio = np.log(split_ratio) + np.log(1 - split_ratio)
            
        else:  # death
            # 移除一个贡献者
            proposed_N = current_N - 1
            
            if proposed_N == 0:
                return None, None, None, 0  # 不允许移除所有贡献者
            
            # 选择要移除的贡献者（倾向于移除贡献比例小的）
            remove_idx = np.random.choice(current_N, p=1 - current_Mx / np.sum(current_Mx))
            
            # 将移除贡献者的比例分配给其他贡献者
            proposed_Mx = np.delete(current_Mx, remove_idx)
            proposed_Mx = proposed_Mx / np.sum(proposed_Mx)  # 重新归一化
            
            # 移除对应的基因型
            proposed_genotypes = {}
            for locus, locus_genotypes in current_genotypes.items():
                new_locus_genotypes = [gt for i, gt in enumerate(locus_genotypes) if i != remove_idx]
                proposed_genotypes[locus] = new_locus_genotypes
            
            # 计算提议比率（death move的雅可比行列式）
            log_proposal_ratio = -np.log(current_Mx[remove_idx])
        
        return proposed_N, proposed_Mx, proposed_genotypes, log_proposal_ratio
    
    def _propose_mixture_ratios(self, current_Mx, step_size=0.05):
        """提议新的混合比例"""
        # 使用Dirichlet分布提议
        concentration = current_Mx / step_size
        concentration = np.maximum(concentration, 0.1)
        
        proposed_Mx = np.random.dirichlet(concentration)
        return proposed_Mx
    
    def _propose_genotypes(self, current_genotypes, observed_data, N):
        """提议新的基因型组合"""
        proposed_genotypes = {}
        
        for locus, locus_genotypes in current_genotypes.items():
            proposed_locus_genotypes = locus_genotypes.copy()
            
            # 随机选择一个个体更新其基因型
            individual_idx = np.random.randint(N)
            alleles = observed_data[locus]['alleles']
            
            if len(alleles) >= 2:
                new_genotype = tuple(np.random.choice(alleles, 2, replace=True))
            elif len(alleles) == 1:
                new_genotype = (alleles[0], alleles[0])
            else:
                new_genotype = None
            
            proposed_locus_genotypes[individual_idx] = new_genotype
            proposed_genotypes[locus] = proposed_locus_genotypes
        
        return proposed_genotypes
    
    def _propose_theta(self, current_theta):
        """提议新的模型参数"""
        proposed_theta = {}
        
        # 复制当前参数
        for key, value in current_theta.items():
            if isinstance(value, dict):
                proposed_theta[key] = value.copy()
            else:
                proposed_theta[key] = value
        
        # 随机更新一个参数
        param_type = np.random.choice(['gamma', 'sigma_var', 'stutter_ratios', 'k_degradation'])
        
        if param_type in ['gamma', 'sigma_var', 'stutter_ratios']:
            # 选择一个位点
            loci = list(proposed_theta[param_type].keys())
            if loci:
                locus = np.random.choice(loci)
                current_value = proposed_theta[param_type][locus]
                
                if param_type == 'gamma':
                    # 对数正态提议
                    log_value = np.log(current_value) + np.random.normal(0, 0.1)
                    proposed_theta[param_type][locus] = np.exp(log_value)
                elif param_type == 'sigma_var':
                    # 在合理范围内提议
                    proposed_theta[param_type][locus] = max(0.01, min(1.0, 
                        current_value + np.random.normal(0, 0.05)))
                else:  # stutter_ratios
                    # 在[0, 0.2]范围内提议
                    proposed_theta[param_type][locus] = max(0.0, min(0.2, 
                        current_value + np.random.normal(0, 0.01)))
        
        else:  # k_degradation
            # 全局降解参数
            current_value = proposed_theta[param_type]
            log_value = np.log(current_value + 1e-6) + np.random.normal(0, 0.1)
            proposed_theta[param_type] = max(1e-6, min(0.01, np.exp(log_value)))
        
        return proposed_theta

# =====================
# 6. 评估与可视化模块
# =====================
class DenoisingEvaluator:
    """降噪效果评估器"""
    
    def __init__(self):
        self.metrics = {}
        logger.info("降噪评估器初始化完成")
    
    def evaluate_denoising_performance(self, original_peaks, denoised_peaks, ground_truth=None):
        """评估降噪性能"""
        metrics = {}
        
        # 基础统计
        original_count = len(original_peaks)
        denoised_count = len(denoised_peaks)
        filtered_count = original_count - denoised_count
        
        metrics['original_peak_count'] = original_count
        metrics['denoised_peak_count'] = denoised_count
        metrics['filtered_peak_count'] = filtered_count
        metrics['filtering_ratio'] = filtered_count / original_count if original_count > 0 else 0
        
        # 信噪比改善
        if not original_peaks.empty and not denoised_peaks.empty:
            original_snr = self._calculate_snr(original_peaks)
            denoised_snr = self._calculate_snr(denoised_peaks)
            
            metrics['original_snr'] = original_snr
            metrics['denoised_snr'] = denoised_snr
            metrics['snr_improvement'] = denoised_snr / original_snr if original_snr > 0 else 1
        
        # 信号质量评估
        if not denoised_peaks.empty:
            signal_quality = self._assess_signal_quality(denoised_peaks)
            metrics.update(signal_quality)
        
        # 与真实标签比较（如果有的话）
        if ground_truth is not None:
            classification_metrics = self._evaluate_classification(denoised_peaks, ground_truth)
            metrics.update(classification_metrics)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_snr(self, peaks_df):
        """计算信噪比"""
        if peaks_df.empty:
            return 0
        
        heights = peaks_df['Height'].values
        if len(heights) < 2:
            return 1
        
        # 信号：前25%的峰
        signal_heights = heights[heights >= np.percentile(heights, 75)]
        # 噪声：后25%的峰
        noise_heights = heights[heights <= np.percentile(heights, 25)]
        
        if len(signal_heights) > 0 and len(noise_heights) > 0:
            signal_power = np.mean(signal_heights)
            noise_power = np.mean(noise_heights)
            return signal_power / noise_power if noise_power > 0 else 1
        
        return 1
    
    def _assess_signal_quality(self, peaks_df):
        """评估信号质量"""
        quality_metrics = {}
        
        if peaks_df.empty:
            return quality_metrics
        
        heights = peaks_df['Height'].values
        
        # 动态范围
        quality_metrics['dynamic_range'] = np.max(heights) / np.min(heights) if np.min(heights) > 0 else 1
        
        # 峰高一致性
        quality_metrics['height_consistency'] = 1 - (np.std(heights) / np.mean(heights)) if np.mean(heights) > 0 else 0
        
        # 强信号比例
        strong_signal_threshold = np.percentile(heights, 75)
        quality_metrics['strong_signal_ratio'] = np.mean(heights >= strong_signal_threshold)
        
        # 位点完整性
        locus_counts = peaks_df.groupby('Marker').size()
        quality_metrics['avg_alleles_per_locus'] = locus_counts.mean()
        quality_metrics['locus_count'] = len(locus_counts)
        
        return quality_metrics
    
    def _evaluate_classification(self, peaks_df, ground_truth):
        """评估分类性能"""
        classification_metrics = {}
        
        # 这里需要根据具体的ground_truth格式来实现
        # 暂时返回空字典
        
        return classification_metrics
    
    def plot_denoising_results(self, original_peaks, denoised_peaks, output_dir, sample_name):
        """绘制降噪结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 峰高分布对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if not original_peaks.empty:
            ax1.hist(original_peaks['Height'], bins=50, alpha=0.7, label='原始峰', color='red')
        if not denoised_peaks.empty:
            ax1.hist(denoised_peaks['Height'], bins=50, alpha=0.7, label='降噪后', color='blue')
        
        ax1.set_xlabel('峰高 (RFU)')
        ax1.set_ylabel('峰数量')
        ax1.set_title('峰高分布对比')
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. 各位点峰数量对比
        if not original_peaks.empty and not denoised_peaks.empty:
            original_counts = original_peaks.groupby('Marker').size()
            denoised_counts = denoised_peaks.groupby('Marker').size()
            
            markers = sorted(list(set(original_counts.index) | set(denoised_counts.index)))
            original_values = [original_counts.get(m, 0) for m in markers]
            denoised_values = [denoised_counts.get(m, 0) for m in markers]
            
            x = np.arange(len(markers))
            width = 0.35
            
            ax2.bar(x - width/2, original_values, width, label='原始峰', color='red', alpha=0.7)
            ax2.bar(x + width/2, denoised_values, width, label='降噪后', color='blue', alpha=0.7)
            
            ax2.set_xlabel('STR位点')
            ax2.set_ylabel('峰数量')
            ax2.set_title('各位点峰数量对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(markers, rotation=45)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_denoising_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 降噪效果指标图
        if self.metrics:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # SNR改善
            if 'snr_improvement' in self.metrics:
                snr_data = ['原始SNR', '降噪后SNR']
                snr_values = [self.metrics['original_snr'], self.metrics['denoised_snr']]
                ax1.bar(snr_data, snr_values, color=['red', 'blue'], alpha=0.7)
                ax1.set_ylabel('信噪比')
                ax1.set_title('信噪比改善')
                
                # 添加改善比例文本
                improvement = self.metrics['snr_improvement']
                ax1.text(0.5, max(snr_values) * 0.8, f'改善: {improvement:.2f}x', 
                        ha='center', fontsize=12, fontweight='bold')
            
            # 峰数量变化
            count_data = ['原始峰数', '降噪后峰数', '过滤峰数']
            count_values = [self.metrics['original_peak_count'], 
                           self.metrics['denoised_peak_count'],
                           self.metrics['filtered_peak_count']]
            ax2.bar(count_data, count_values, color=['gray', 'blue', 'red'], alpha=0.7)
            ax2.set_ylabel('峰数量')
            ax2.set_title('峰数量变化')
            
            # 信号质量指标
            if 'dynamic_range' in self.metrics:
                quality_metrics = ['动态范围', '峰高一致性', '强信号比例']
                quality_values = [min(self.metrics.get('dynamic_range', 0), 100),  # 限制动态范围显示
                                self.metrics.get('height_consistency', 0),
                                self.metrics.get('strong_signal_ratio', 0)]
                ax3.bar(quality_metrics, quality_values, color='green', alpha=0.7)
                ax3.set_ylabel('质量评分')
                ax3.set_title('信号质量指标')
                ax3.tick_params(axis='x', rotation=45)
            
            # 位点统计
            if 'locus_count' in self.metrics:
                locus_stats = ['有效位点数', '平均等位基因数']
                locus_values = [self.metrics.get('locus_count', 0),
                               self.metrics.get('avg_alleles_per_locus', 0)]
                ax4.bar(locus_stats, locus_values, color='orange', alpha=0.7)
                ax4.set_ylabel('数量')
                ax4.set_title('位点统计')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{sample_name}_metrics.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"降噪结果图表已保存到: {output_dir}")

# =====================
# 7. 主分析流水线
# =====================
class UPGMPipeline:
    """UPG-M完整分析流水线"""
    
    def __init__(self):
        self.inference_engine = UPGMInferenceEngine()
        self.evaluator = DenoisingEvaluator()
        self.results = {}
        logger.info("UPG-M分析流水线初始化完成")
    
    def train_system(self, training_data_path):
        """训练系统组件"""
        logger.info("开始训练UPG-M系统组件...")
        
        # 训练峰分类器
        training_results = self.inference_engine.train_peak_classifier_on_dataset(training_data_path)
        
        if training_results:
            logger.info("系统训练完成")
            return training_results
        else:
            logger.warning("系统训练失败，将使用默认规则")
            return None
    
    def analyze_sample(self, sample_data, enable_mcmc=True):
        """分析单个样本"""
        sample_file = sample_data.iloc[0]['Sample File']
        logger.info(f"开始UPG-M分析样本: {sample_file}")
        
        start_time = time.time()
        
        # 步骤1: 预处理
        processed_peaks, v5_features, peak_classifications = self.inference_engine.preprocess_sample(sample_data)
        
        if processed_peaks is None:
            logger.warning(f"样本 {sample_file} 预处理失败")
            return self._get_default_result(sample_file, time.time() - start_time)
        
        # 步骤2: 降噪
        denoised_peaks = self.inference_engine.denoise_sample(processed_peaks, peak_classifications)
        
        # 步骤3: 评估降噪效果
        denoising_metrics = self.evaluator.evaluate_denoising_performance(processed_peaks, denoised_peaks)
        
        # 步骤4: 准备观测数据用于MCMC
        observed_data = self._prepare_observed_data(denoised_peaks)
        
        # 步骤5: MCMC推断（可选）
        mcmc_results = None
        posterior_summary = None
        
        if enable_mcmc and observed_data:
            try:
                mcmc_results = self.inference_engine.mcmc_sampler(observed_data)
                posterior_summary = self._generate_posterior_summary(mcmc_results)
            except Exception as e:
                logger.warning(f"MCMC推断失败: {e}")
        
        end_time = time.time()
        
        # 整合结果
        result = {
            'sample_file': sample_file,
            'v5_features': v5_features,
            'original_peaks': processed_peaks,
            'denoised_peaks': denoised_peaks,
            'peak_classifications': peak_classifications,
            'denoising_metrics': denoising_metrics,
            'observed_data': observed_data,
            'mcmc_results': mcmc_results,
            'posterior_summary': posterior_summary,
            'computation_time': end_time - start_time,
            'success': True
        }
        
        logger.info(f"样本 {sample_file} 分析完成，耗时: {end_time - start_time:.1f}秒")
        return result
    
    def _prepare_observed_data(self, denoised_peaks):
        """准备观测数据用于MCMC推断"""
        if denoised_peaks.empty:
            return {}
        
        observed_data = {}
        
        for marker, locus_group in denoised_peaks.groupby('Marker'):
            alleles = locus_group['Allele'].tolist()
            heights = dict(zip(locus_group['Allele'], locus_group['Height']))
            ctas = dict(zip(locus_group['Allele'], locus_group.get('Adaptive_CTA', [0.5] * len(alleles))))
            
            observed_data[marker] = {
                'alleles': alleles,
                'heights': heights,
                'ctas': ctas
            }
        
        return observed_data
    
    def _generate_posterior_summary(self, mcmc_results):
        """生成后验分布摘要"""
        if not mcmc_results or not mcmc_results['samples']:
            return None
        
        samples = mcmc_results['samples']
        
        # NoC后验分布
        noc_samples = samples['N']
        noc_counts = pd.Series(noc_samples).value_counts()
        noc_posterior = noc_counts / len(noc_samples)
        
        # 混合比例后验统计
        mx_samples = samples['Mx']
        
        # 计算每个可能NoC值的混合比例统计
        mx_summary = {}
        for noc in noc_counts.index:
            noc_indices = [i for i, n in enumerate(noc_samples) if n == noc]
            if noc_indices:
                noc_mx_samples = [mx_samples[i] for i in noc_indices]
                
                # 转换为数组进行统计
                mx_array = np.array(noc_mx_samples)
                
                mx_stats = {}
                for i in range(noc):
                    component_samples = mx_array[:, i]
                    mx_stats[f'Mx_{i+1}'] = {
                        'mean': np.mean(component_samples),
                        'std': np.std(component_samples),
                        'median': np.median(component_samples),
                        'credible_interval_95': np.percentile(component_samples, [2.5, 97.5]).tolist()
                    }
                
                mx_summary[f'NoC_{noc}'] = mx_stats
        
        return {
            'noc_posterior': noc_posterior.to_dict(),
            'most_probable_noc': noc_counts.index[0],
            'noc_confidence': noc_counts.iloc[0] / len(noc_samples),
            'mixture_ratio_summary': mx_summary,
            'model_quality': {
                'acceptance_rate': mcmc_results['acceptance_rate'],
                'n_samples': mcmc_results['n_samples'],
                'converged': mcmc_results['converged']
            }
        }
    
    def _get_default_result(self, sample_file, computation_time):
        """获取默认结果"""
        return {
            'sample_file': sample_file,
            'v5_features': {},
            'original_peaks': pd.DataFrame(),
            'denoised_peaks': pd.DataFrame(),
            'peak_classifications': pd.DataFrame(),
            'denoising_metrics': {},
            'observed_data': {},
            'mcmc_results': None,
            'posterior_summary': None,
            'computation_time': computation_time,
            'success': False
        }
    
    def analyze_attachment4_dataset(self, max_samples=None):
        """分析附件4数据集"""
        logger.info("开始分析附件4混合STR图谱数据集")
        
        if not os.path.exists(config.ATTACHMENT4_PATH):
            logger.error(f"附件4文件不存在: {config.ATTACHMENT4_PATH}")
            return {}
        
        # 加载数据
        try:
            df = pd.read_csv(config.ATTACHMENT4_PATH, encoding='utf-8')
            logger.info(f"成功加载附件4数据，形状: {df.shape}")
        except Exception as e:
            logger.error(f"加载附件4数据失败: {e}")
            return {}
        
        # 首先在数据集上训练系统
        training_results = self.train_system(config.ATTACHMENT4_PATH)
        
        # 分析样本
        all_results = {}
        sample_files = df['Sample File'].unique()
        
        if max_samples:
            sample_files = sample_files[:max_samples]
        
        logger.info(f"计划分析 {len(sample_files)} 个样本")
        
        for idx, sample_file in enumerate(sample_files, 1):
            logger.info(f"--- 分析进度: {idx}/{len(sample_files)} - 样本: {sample_file} ---")
            
            try:
                sample_data = df[df['Sample File'] == sample_file]
                result = self.analyze_sample(sample_data, enable_mcmc=True)
                all_results[sample_file] = result
                
                # 保存单个样本结果
                self.save_sample_result(result)
                
                # 绘制结果
                self.plot_sample_result(result)
                
                # 打印简要结果
                self._print_sample_summary(result)
                
            except Exception as e:
                logger.error(f"样本 {sample_file} 分析失败: {e}")
                continue
        
        # 生成批量分析摘要
        self._generate_batch_summary(all_results)
        
        logger.info(f"附件4数据集分析完成！成功分析 {len(all_results)} 个样本")
        return all_results
    
    def save_sample_result(self, result):
        """保存单个样本结果"""
        sample_file = result['sample_file']
        
        # 创建输出目录
        sample_dir = os.path.join(config.OUTPUT_DIR, f"sample_{sample_file}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存降噪后的峰数据
        if not result['denoised_peaks'].empty:
            denoised_path = os.path.join(sample_dir, f"{sample_file}_denoised_peaks.csv")
            result['denoised_peaks'].to_csv(denoised_path, index=False, encoding='utf-8-sig')
        
        # 保存峰分类结果
        if not result['peak_classifications'].empty:
            classification_path = os.path.join(sample_dir, f"{sample_file}_peak_classifications.csv")
            result['peak_classifications'].to_csv(classification_path, index=False, encoding='utf-8-sig')
        
        # 保存分析摘要
        summary = {
            'sample_file': result['sample_file'],
            'computation_time': result['computation_time'],
            'success': result['success'],
            'denoising_metrics': result['denoising_metrics'],
            'v5_features': result['v5_features'],
            'posterior_summary': result['posterior_summary'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(sample_dir, f"{sample_file}_analysis_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"样本 {sample_file} 结果已保存到: {sample_dir}")
    
    def plot_sample_result(self, result):
        """绘制单个样本的分析结果"""
        sample_file = result['sample_file']
        sample_dir = os.path.join(config.OUTPUT_DIR, f"sample_{sample_file}")
        
        # 降噪效果图
        self.evaluator.plot_denoising_results(
            result['original_peaks'], 
            result['denoised_peaks'], 
            sample_dir, 
            sample_file
        )
        
        # MCMC结果图（如果有的话）
        if result['mcmc_results'] is not None:
            self._plot_mcmc_results(result['mcmc_results'], result['posterior_summary'], sample_dir, sample_file)
    
    def _plot_mcmc_results(self, mcmc_results, posterior_summary, output_dir, sample_name):
        """绘制MCMC推断结果"""
        if not mcmc_results or not posterior_summary:
            return
        
        samples = mcmc_results['samples']
        
        # 1. NoC后验分布
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # NoC后验分布
        noc_samples = samples['N']
        noc_counts = pd.Series(noc_samples).value_counts().sort_index()
        
        ax1.bar(noc_counts.index, noc_counts.values / len(noc_samples), 
               alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('贡献者人数 (NoC)')
        ax1.set_ylabel('后验概率')
        ax1.set_title('贡献者人数后验分布')
        ax1.grid(True, alpha=0.3)
        
        # 添加最可能的NoC
        most_probable_noc = noc_counts.index[0]
        confidence = noc_counts.iloc[0] / len(noc_samples)
        ax1.text(0.7, 0.9, f'最可能NoC: {most_probable_noc}\n置信度: {confidence:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 2. MCMC轨迹（NoC）
        ax2.plot(noc_samples, alpha=0.7, linewidth=0.5)
        ax2.set_xlabel('MCMC迭代')
        ax2.set_ylabel('NoC')
        ax2.set_title('NoC MCMC轨迹')
        ax2.grid(True, alpha=0.3)
        
        # 3. 对数似然轨迹
        if 'log_likelihood' in samples:
            ax3.plot(samples['log_likelihood'], alpha=0.7, linewidth=0.5, color='red')
            ax3.set_xlabel('MCMC迭代')
            ax3.set_ylabel('对数似然')
            ax3.set_title('似然函数轨迹')
            ax3.grid(True, alpha=0.3)
        
        # 4. 接受率统计
        acceptance_rate = mcmc_results['acceptance_rate']
        trans_acceptance_rate = mcmc_results.get('trans_dimensional_acceptance_rate', 0)
        
        rates = ['整体接受率', '跨维接受率']
        values = [acceptance_rate, trans_acceptance_rate]
        colors = ['green' if v > 0.15 else 'red' for v in values]
        
        bars = ax4.bar(rates, values, color=colors, alpha=0.7)
        ax4.set_ylabel('接受率')
        ax4.set_title('MCMC接受率统计')
        ax4.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_mcmc_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 混合比例结果（针对最可能的NoC）
        if most_probable_noc in [2, 3, 4, 5]:  # 只为合理的NoC值绘制
            self._plot_mixture_ratios(samples, most_probable_noc, output_dir, sample_name)
    
    def _plot_mixture_ratios(self, samples, target_noc, output_dir, sample_name):
        """绘制特定NoC的混合比例分布"""
        noc_samples = samples['N']
        mx_samples = samples['Mx']
        
        # 过滤出目标NoC的样本
        target_indices = [i for i, n in enumerate(noc_samples) if n == target_noc]
        
        if not target_indices:
            return
        
        target_mx_samples = [mx_samples[i] for i in target_indices]
        mx_array = np.array(target_mx_samples)
        
        # 创建子图
        fig, axes = plt.subplots(target_noc, 1, figsize=(12, 4*target_noc))
        if target_noc == 1:
            axes = [axes]
        
        for i in range(target_noc):
            component_samples = mx_array[:, i]
            
            # 直方图
            axes[i].hist(component_samples, bins=50, alpha=0.7, density=True, 
                        color=f'C{i}', edgecolor='black', linewidth=0.5)
            
            # 统计量
            mean_val = np.mean(component_samples)
            median_val = np.median(component_samples)
            ci_95 = np.percentile(component_samples, [2.5, 97.5])
            
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'均值: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2,
                           label=f'中位数: {median_val:.3f}')
            axes[i].axvspan(ci_95[0], ci_95[1], alpha=0.2, color='gray',
                           label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
            
            axes[i].set_xlabel(f'混合比例 Mx_{i+1}')
            axes[i].set_ylabel('概率密度')
            axes[i].set_title(f'贡献者 {i+1} 混合比例后验分布 (NoC={target_noc})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_mixture_ratios_NoC{target_noc}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果是2个贡献者，绘制联合分布
        if target_noc == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(mx_array[:, 0], mx_array[:, 1], alpha=0.6, s=2, color='blue')
            plt.xlabel('Mx_1', fontsize=12)
            plt.ylabel('Mx_2', fontsize=12)
            plt.title(f'样本 {sample_name} 混合比例联合后验分布 (NoC=2)', fontsize=14)
            
            # 添加约束线
            x_line = np.linspace(0, 1, 100)
            y_line = 1 - x_line
            plt.plot(x_line, y_line, 'r--', alpha=0.7, label='Mx_1 + Mx_2 = 1')
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.axis('equal')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, f'{sample_name}_joint_mixture_ratios.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _print_sample_summary(self, result):
        """打印样本分析摘要"""
        sample_file = result['sample_file']
        success = result['success']
        computation_time = result['computation_time']
        
        print(f"  分析状态: {'成功' if success else '失败'}")
        print(f"  计算时间: {computation_time:.1f}秒")
        
        if success:
            # 降噪效果
            metrics = result['denoising_metrics']
            if metrics:
                original_count = metrics.get('original_peak_count', 0)
                denoised_count = metrics.get('denoised_peak_count', 0)
                filtering_ratio = metrics.get('filtering_ratio', 0)
                snr_improvement = metrics.get('snr_improvement', 1)
                
                print(f"  降噪效果: {original_count} → {denoised_count} 峰 (过滤率: {filtering_ratio:.1%})")
                print(f"  信噪比改善: {snr_improvement:.2f}x")
            
            # MCMC结果
            if result['posterior_summary']:
                summary = result['posterior_summary']
                most_probable_noc = summary['most_probable_noc']
                noc_confidence = summary['noc_confidence']
                
                print(f"  预测NoC: {most_probable_noc} (置信度: {noc_confidence:.3f})")
                
                # 混合比例
                if f'NoC_{most_probable_noc}' in summary['mixture_ratio_summary']:
                    mx_summary = summary['mixture_ratio_summary'][f'NoC_{most_probable_noc}']
                    print(f"  混合比例估计:")
                    for i in range(most_probable_noc):
                        mx_stats = mx_summary[f'Mx_{i+1}']
                        mean_val = mx_stats['mean']
                        ci = mx_stats['credible_interval_95']
                        print(f"    贡献者{i+1}: {mean_val:.3f} (95%CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
                
                # 模型质量
                model_quality = summary['model_quality']
                acceptance_rate = model_quality['acceptance_rate']
                converged = model_quality['converged']
                print(f"  模型质量: 接受率={acceptance_rate:.3f}, 收敛={'是' if converged else '否'}")
            else:
                print(f"  ⚠️  MCMC推断未执行或失败")
        else:
            print(f"  ⚠️  样本分析失败")
    
    def _generate_batch_summary(self, all_results):
        """生成批量分析摘要"""
        if not all_results:
            return
        
        summary_data = []
        
        for sample_file, result in all_results.items():
            sample_summary = {
                'Sample_File': sample_file,
                'Success': result['success'],
                'Computation_Time': result['computation_time']
            }
            
            # 降噪指标
            if result['denoising_metrics']:
                metrics = result['denoising_metrics']
                sample_summary.update({
                    'Original_Peak_Count': metrics.get('original_peak_count', 0),
                    'Denoised_Peak_Count': metrics.get('denoised_peak_count', 0),
                    'Filtering_Ratio': metrics.get('filtering_ratio', 0),
                    'SNR_Improvement': metrics.get('snr_improvement', 1),
                    'Dynamic_Range': metrics.get('dynamic_range', 0),
                    'Signal_Quality': metrics.get('height_consistency', 0)
                })
            
            # MCMC结果
            if result['posterior_summary']:
                summary = result['posterior_summary']
                sample_summary.update({
                    'Predicted_NoC': summary['most_probable_noc'],
                    'NoC_Confidence': summary['noc_confidence'],
                    'MCMC_Success': True,
                    'MCMC_Acceptance_Rate': summary['model_quality']['acceptance_rate'],
                    'MCMC_Converged': summary['model_quality']['converged']
                })
                
                # 添加混合比例（针对最可能的NoC）
                most_probable_noc = summary['most_probable_noc']
                if f'NoC_{most_probable_noc}' in summary['mixture_ratio_summary']:
                    mx_summary = summary['mixture_ratio_summary'][f'NoC_{most_probable_noc}']
                    for i in range(most_probable_noc):
                        mx_stats = mx_summary[f'Mx_{i+1}']
                        sample_summary[f'Mx_{i+1}_Mean'] = mx_stats['mean']
                        sample_summary[f'Mx_{i+1}_CI_Width'] = (mx_stats['credible_interval_95'][1] - 
                                                              mx_stats['credible_interval_95'][0])
            else:
                sample_summary.update({
                    'Predicted_NoC': None,
                    'NoC_Confidence': 0,
                    'MCMC_Success': False,
                    'MCMC_Acceptance_Rate': 0,
                    'MCMC_Converged': False
                })
            
            summary_data.append(sample_summary)
        
        # 保存摘要表格
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(config.OUTPUT_DIR, 'batch_analysis_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        # 生成统计报告
        self._print_batch_statistics(summary_df)
        
        # 保存详细统计
        stats_path = os.path.join(config.OUTPUT_DIR, 'batch_statistics.json')
        batch_stats = self._calculate_batch_statistics(summary_df)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"批量分析摘要已保存: {summary_path}")
    
    def _print_batch_statistics(self, summary_df):
        """打印批量统计结果"""
        print(f"\n{'='*80}")
        print("                      UPG-M批量分析统计摘要")
        print(f"{'='*80}")
        
        total_samples = len(summary_df)
        successful_samples = summary_df['Success'].sum()
        
        print(f"\n📊 总体统计:")
        print(f"   总样本数: {total_samples}")
        print(f"   成功分析: {successful_samples} ({successful_samples/total_samples*100:.1f}%)")
        
        if successful_samples > 0:
            success_df = summary_df[summary_df['Success']]
            
            # 降噪效果统计
            print(f"\n🔧 降噪效果统计:")
            if 'SNR_Improvement' in success_df.columns:
                avg_snr_improvement = success_df['SNR_Improvement'].mean()
                print(f"   平均信噪比改善: {avg_snr_improvement:.2f}x")
            
            if 'Filtering_Ratio' in success_df.columns:
                avg_filtering_ratio = success_df['Filtering_Ratio'].mean()
                print(f"   平均过滤率: {avg_filtering_ratio:.1%}")
            
            # MCMC统计
            mcmc_success = success_df['MCMC_Success'].sum() if 'MCMC_Success' in success_df.columns else 0
            print(f"\n🎯 MCMC推断统计:")
            print(f"   MCMC成功率: {mcmc_success}/{successful_samples} ({mcmc_success/successful_samples*100:.1f}%)")
            
            if mcmc_success > 0:
                mcmc_df = success_df[success_df['MCMC_Success']]
                avg_acceptance = mcmc_df['MCMC_Acceptance_Rate'].mean()
                converged_count = mcmc_df['MCMC_Converged'].sum()
                
                print(f"   平均MCMC接受率: {avg_acceptance:.3f}")
                print(f"   MCMC收敛率: {converged_count}/{mcmc_success} ({converged_count/mcmc_success*100:.1f}%)")
                
                # NoC分布
                if 'Predicted_NoC' in mcmc_df.columns:
                    noc_distribution = mcmc_df['Predicted_NoC'].value_counts().sort_index()
                    print(f"   NoC分布: {noc_distribution.to_dict()}")
            
            # 计算时间统计
            avg_time = success_df['Computation_Time'].mean()
            print(f"\n⏱️  性能统计:")
            print(f"   平均计算时间: {avg_time:.1f}秒")
        
        print(f"\n💾 结果文件:")
        print(f"   输出目录: {config.OUTPUT_DIR}")
        print(f"   批量摘要: batch_analysis_summary.csv")
        print(f"   统计报告: batch_statistics.json")
    
    def _calculate_batch_statistics(self, summary_df):
        """计算详细的批量统计"""
        stats = {
            'total_samples': len(summary_df),
            'successful_samples': int(summary_df['Success'].sum()),
            'success_rate': float(summary_df['Success'].mean()),
            'avg_computation_time': float(summary_df['Computation_Time'].mean()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 成功样本的详细统计
        if stats['successful_samples'] > 0:
            success_df = summary_df[summary_df['Success']]
            
            # 降噪统计
            if 'SNR_Improvement' in success_df.columns:
                stats['denoising_stats'] = {
                    'avg_snr_improvement': float(success_df['SNR_Improvement'].mean()),
                    'median_snr_improvement': float(success_df['SNR_Improvement'].median()),
                    'snr_improvement_std': float(success_df['SNR_Improvement'].std()),
                    'avg_filtering_ratio': float(success_df['Filtering_Ratio'].mean()) if 'Filtering_Ratio' in success_df.columns else 0
                }
            
            # MCMC统计
            if 'MCMC_Success' in success_df.columns:
                mcmc_success_count = int(success_df['MCMC_Success'].sum())
                stats['mcmc_stats'] = {
                    'mcmc_success_count': mcmc_success_count,
                    'mcmc_success_rate': float(success_df['MCMC_Success'].mean()),
                }
                
                if mcmc_success_count > 0:
                    mcmc_df = success_df[success_df['MCMC_Success']]
                    stats['mcmc_stats'].update({
                        'avg_acceptance_rate': float(mcmc_df['MCMC_Acceptance_Rate'].mean()),
                        'convergence_rate': float(mcmc_df['MCMC_Converged'].mean()),
                        'noc_distribution': mcmc_df['Predicted_NoC'].value_counts().to_dict() if 'Predicted_NoC' in mcmc_df.columns else {}
                    })
        
        return stats

# =====================
# 8. 主程序入口和示例
# =====================
def analyze_single_sample_from_attachment4(sample_id: str, enable_mcmc: bool = True) -> Dict:
    """分析附件4中的单个样本"""
    
    # 初始化流水线
    pipeline = UPGMPipeline()
    
    # 加载附件4数据
    if not os.path.exists(config.ATTACHMENT4_PATH):
        raise FileNotFoundError(f"附件4文件不存在: {config.ATTACHMENT4_PATH}")
    
    df = pd.read_csv(config.ATTACHMENT4_PATH, encoding='utf-8')
    
    # 查找目标样本
    sample_data = df[df['Sample File'] == sample_id]
    if sample_data.empty:
        raise ValueError(f"样本 {sample_id} 不存在于附件4数据中")
    
    # 训练系统（首次运行）
    logger.info("在附件4数据集上训练系统组件...")
    training_results = pipeline.train_system(config.ATTACHMENT4_PATH)
    
    # 分析目标样本
    result = pipeline.analyze_sample(sample_data, enable_mcmc=enable_mcmc)
    
    # 保存结果
    pipeline.save_sample_result(result)
    pipeline.plot_sample_result(result)
    
    return result

def main():
    """主程序"""
    print("UPG-M混合STR图谱智能降噪系统")
    print("基于统一概率基因分型的噪声消除与信号增强")
    print("="*80)
    
    # 检查必要文件
    if not os.path.exists(config.ATTACHMENT4_PATH):
        print(f"错误: 附件4文件不存在 - {config.ATTACHMENT4_PATH}")
        print("请确保文件在当前目录下")
        return
    
    # 设置随机种子
    np.random.seed(config.RANDOM_STATE)
    
    # 选择分析模式
    print("\n请选择分析模式:")
    print("1. 分析单个样本（快速测试）")
    print("2. 批量分析所有样本")
    print("3. 批量分析前N个样本")
    print("4. 系统训练和验证")
    
    try:
        choice = input("请输入选择 (1/2/3/4): ").strip()
        
        if choice == "1":
            # 单样本分析
            df = pd.read_csv(config.ATTACHMENT4_PATH, encoding='utf-8')
            available_samples = df['Sample File'].unique()
            
            print(f"\n可用样本 ({len(available_samples)}个):")
            for i, sample_id in enumerate(available_samples[:10], 1):
                print(f"  {i}. {sample_id}")
            if len(available_samples) > 10:
                print(f"  ... 还有{len(available_samples)-10}个样本")
            
            sample_id = input("\n请输入样本ID: ").strip()
            
            if sample_id in available_samples:
                print(f"\n开始分析样本: {sample_id}")
                result = analyze_single_sample_from_attachment4(sample_id, enable_mcmc=True)
                
                print("\n=== 分析完成 ===")
                pipeline = UPGMPipeline()
                pipeline._print_sample_summary(result)
            else:
                print("样本ID不存在")
        
        elif choice == "2":
            # 批量分析所有样本
            pipeline = UPGMPipeline()
            all_results = pipeline.analyze_attachment4_dataset()
        
        elif choice == "3":
            # 批量分析前N个样本
            try:
                max_samples = int(input("请输入要分析的样本数量: ").strip())
                pipeline = UPGMPipeline()
                all_results = pipeline.analyze_attachment4_dataset(max_samples=max_samples)
            except ValueError:
                print("输入的样本数量无效")
        
        elif choice == "4":
            # 系统训练和验证
            print("\n开始系统训练和验证...")
            pipeline = UPGMPipeline()
            training_results = pipeline.train_system(config.ATTACHMENT4_PATH)
            
            if training_results:
                print("系统训练完成")
                print(f"峰分类器准确率: {training_results['accuracy']:.4f}")
            else:
                print("系统训练失败")
        
        else:
            print("无效的选择")
    
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("UPG-M分析完成！")
    print(f"结果保存在目录: {config.OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()