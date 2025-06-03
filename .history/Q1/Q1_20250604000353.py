# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别 (Bootstrap数据扩增版)

版本: V4.0 - Bootstrap Enhanced
日期: 2025-06-03
描述: 基于附件一真实数据，使用Bootstrap方法生成更多训练样本，提高模型性能
主要改进:
1. Bootstrap重采样生成更多训练数据
2. 保持原有的高质量特征工程
3. 多模型集成提高预测准确率
4. 完整的模型评估和解释性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
import re
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter
from time import time
import random

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample

# 可解释性分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP不可用，将跳过可解释性分析")
    SHAP_AVAILABLE = False

# 配置
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=== 法医混合STR图谱NoC智能识别系统 (Bootstrap增强版) ===")
print("基于附件一数据进行Bootstrap扩增的完整实现")

# =====================
# 1. 文件路径与基础设置
# =====================
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'noc_bootstrap_plots')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# 关键参数设置
HEIGHT_THRESHOLD = 50
SATURATION_THRESHOLD = 30000
CTA_THRESHOLD = 0.5
PHR_IMBALANCE_THRESHOLD = 0.6

# Bootstrap参数
BOOTSTRAP_MULTIPLIER = 3  # 降低扩增倍数，避免过度拟合
BOOTSTRAP_RANDOM_STATE = 42

# =====================
# 2. 辅助函数定义
# =====================
def extract_noc_from_filename(filename):
    """从文件名提取贡献者人数（NoC）"""
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
    if match:
        ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return len(ids) if len(ids) > 0 else np.nan
    return np.nan

def calculate_entropy(probabilities):
    """计算香农熵"""
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

def calculate_ols_slope(x, y):
    """计算OLS回归斜率"""
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    
    x = np.array(x)
    y = np.array(y)
    
    if len(np.unique(x)) < 2:
        return 0.0
    
    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    except:
        return 0.0

# =====================
# 3. Bootstrap数据扩增类
# =====================
class STRBootstrapGenerator:
    """STR数据Bootstrap生成器"""
    
    def __init__(self, original_data, multiplier=1.5, random_state=42):
        """
        初始化Bootstrap生成器
        
        Args:
            original_data: 原始数据DataFrame
            multiplier: 扩增倍数（建议2-5倍）
            random_state: 随机种子
        """
        self.original_data = original_data
        self.multiplier = multiplier
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        print(f"Bootstrap生成器初始化 - 扩增倍数: {multiplier}x")
        
        # 验证数据格式
        required_cols = ['Sample File', 'NoC_True', 'Marker']
        missing_cols = [col for col in required_cols if col not in original_data.columns]
        if missing_cols:
            raise ValueError(f"原始数据缺少必要列: {missing_cols}")
        
        print(f"原始数据验证通过 - 形状: {original_data.shape}")
        
    def generate_bootstrap_samples(self):
        """
        生成Bootstrap扩增样本
        
        Returns:
            bootstrap_data: 扩增后的数据
        """
        print("\n=== 开始Bootstrap数据扩增 ===")
        
        # 按NoC分组
        grouped_by_noc = self.original_data.groupby('NoC_True')
        all_bootstrap_samples = []
        
        for noc, group_data in grouped_by_noc:
            print(f"\n处理NoC={noc}的样本 (原始数量: {len(group_data)})")
            
            # 计算需要生成的样本数
            original_count = len(group_data)
            target_count = original_count * self.multiplier
            bootstrap_count = target_count - original_count
            
            print(f"目标样本数: {target_count}, 需要Bootstrap生成: {bootstrap_count}")
            
            # 添加原始样本
            all_bootstrap_samples.append(group_data)
            
            # 生成Bootstrap样本
            if bootstrap_count > 0:
                bootstrap_samples = self._generate_noc_bootstrap_samples(
                    group_data, bootstrap_count, noc)
                all_bootstrap_samples.append(bootstrap_samples)
                
                print(f"✓ 成功生成 {len(bootstrap_samples)} 个Bootstrap样本")
        
        # 合并所有样本
        bootstrap_data = pd.concat(all_bootstrap_samples, ignore_index=True)
        
        print(f"\n=== Bootstrap数据扩增完成 ===")
        print(f"原始样本数: {len(self.original_data)}")
        print(f"扩增后样本数: {len(bootstrap_data)}")
        print(f"扩增倍数: {len(bootstrap_data) / len(self.original_data):.1f}x")
        
        # 显示各NoC的样本分布
        print("\n各NoC类别的样本分布:")
        noc_dist = bootstrap_data['NoC_True'].value_counts().sort_index()
        for noc, count in noc_dist.items():
            original_count = len(self.original_data[self.original_data['NoC_True'] == noc])
            print(f"  {noc}人: {count} 样本 (原始: {original_count}, 增加: {count - original_count})")
        
        return bootstrap_data
    
    def _generate_noc_bootstrap_samples(self, noc_group_data, n_samples, noc):
        """
        为特定NoC生成Bootstrap样本
        
        Args:
            noc_group_data: 该NoC的原始数据
            n_samples: 需要生成的样本数
            noc: NoC值
            
        Returns:
            bootstrap_samples: Bootstrap样本
        """
        bootstrap_samples = []
        
        # 获取所有样本的基本信息
        unique_samples = noc_group_data['Sample File'].unique()
        
        for i in range(n_samples):
            # Bootstrap重采样：随机选择一个原始样本作为模板
            template_sample = np.random.choice(unique_samples)
            template_data = noc_group_data[noc_group_data['Sample File'] == template_sample]
            
            # 生成新的样本名称
            new_sample_name = f"Bootstrap_{noc}P_{i+1:04d}_{template_sample}"
            
            # 创建新样本数据
            new_sample_data = template_data.copy()
            new_sample_data['Sample File'] = new_sample_name
            
            # 对峰高数据添加噪声（模拟实验变异）
            new_sample_data = self._add_realistic_noise(new_sample_data)
            
            bootstrap_samples.append(new_sample_data)
        
        # 合并所有Bootstrap样本
        if bootstrap_samples:
            return pd.concat(bootstrap_samples, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _add_realistic_noise(self, sample_data):
        """
        为样本数据添加现实的噪声
        
        Args:
            sample_data: 样本数据
            
        Returns:
            noisy_data: 添加噪声后的数据
        """
        noisy_data = sample_data.copy()
        
        # 对Height列添加对数正态噪声（符合峰高的分布特性）
        for i in range(1, 101):  # Height 1 到 Height 100
            height_col = f'Height {i}'
            if height_col in noisy_data.columns:
                original_heights = noisy_data[height_col].values
                
                # 只对非零、非NaN的值添加噪声
                mask = pd.notna(original_heights) & (original_heights > 0)
                if mask.any():
                    # 对数正态噪声，变异系数约为10-20%（更保守）
                    noise_cv = np.random.uniform(0.10, 0.20)
                    
                    # 生成对数正态分布的乘性噪声
                    sigma = np.sqrt(np.log(1 + noise_cv**2))
                    noise_factor = np.random.lognormal(
                        mean=-sigma**2/2,  # 调整均值使期望值为1
                        sigma=sigma,
                        size=len(original_heights)
                    )
                    
                    noisy_heights = original_heights.copy().astype(float)
                    noisy_heights[mask] = original_heights[mask] * noise_factor[mask]
                    
                    # 确保峰高在合理范围内
                    noisy_heights[mask] = np.clip(
                        noisy_heights[mask],
                        original_heights[mask] * 0.7,  # 不小于原值的70%
                        original_heights[mask] * 1.5   # 不大于原值的150%
                    )
                    
                    # 四舍五入到整数（RFU值通常是整数）
                    noisy_heights = np.round(noisy_heights).astype(int)
                    
                    # 应用饱和阈值
                    noisy_heights = np.minimum(noisy_heights, SATURATION_THRESHOLD)
                    
                    # 确保不小于分析阈值
                    noisy_heights[mask & (noisy_heights < HEIGHT_THRESHOLD)] = HEIGHT_THRESHOLD
                    
                    noisy_data[height_col] = noisy_heights
        
        return noisy_data

# =====================
# 4. 数据加载与预处理
# =====================
print("\n=== 步骤1: 数据加载与预处理 ===")

# 加载数据
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"成功加载数据，形状: {df.shape}")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 提取NoC标签
df['NoC_True'] = df['Sample File'].apply(extract_noc_from_filename)
df = df.dropna(subset=['NoC_True'])
df['NoC_True'] = df['NoC_True'].astype(int)

print(f"原始数据NoC分布: {df['NoC_True'].value_counts().sort_index().to_dict()}")
print(f"原始样本数: {df['Sample File'].nunique()}")

# =====================
# 5. Bootstrap数据扩增
# =====================
print("\n=== 步骤2: Bootstrap数据扩增 ===")

# 创建Bootstrap生成器
bootstrap_generator = STRBootstrapGenerator(
    df, 
    multiplier=BOOTSTRAP_MULTIPLIER, 
    random_state=BOOTSTRAP_RANDOM_STATE
)

# 生成扩增数据
df_bootstrap = bootstrap_generator.generate_bootstrap_samples()

# 保存扩增后的数据
bootstrap_data_path = os.path.join(DATA_DIR, 'bootstrap_enhanced_str_data.csv')
try:
    df_bootstrap.to_csv(bootstrap_data_path, index=False, encoding='utf-8-sig')
    print(f"Bootstrap扩增数据已保存到: {bootstrap_data_path}")
except Exception as e:
    print(f"保存Bootstrap数据失败: {e}")

# =====================
# 6. 简化的峰处理
# =====================
print("\n=== 步骤3: 峰处理与信号表征 ===")

def process_peaks_with_cta(sample_data):
    """简化的峰处理，包含基础CTA评估"""
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
                corrected_height = min(original_height, SATURATION_THRESHOLD)
                
                if corrected_height >= HEIGHT_THRESHOLD:
                    peaks.append({
                        'allele': allele,
                        'size': float(size),
                        'height': corrected_height,
                        'original_height': original_height
                    })
        
        if not peaks:
            continue
            
        # 简化的CTA评估
        peaks.sort(key=lambda x: x['height'], reverse=True)
        
        for peak in peaks:
            height_rank = peaks.index(peak) + 1
            total_peaks = len(peaks)
            
            if height_rank == 1:
                cta = 0.95
            elif height_rank == 2 and total_peaks >= 2:
                height_ratio = peak['height'] / peaks[0]['height']
                cta = 0.8 if height_ratio > 0.3 else 0.6
            else:
                height_ratio = peak['height'] / peaks[0]['height']
                cta = max(0.1, min(0.8, height_ratio))
            
            if cta >= CTA_THRESHOLD:
                processed_data.append({
                    'Sample File': sample_file,
                    'Marker': marker,
                    'Allele': peak['allele'],
                    'Size': peak['size'],
                    'Height': peak['height'],
                    'Original_Height': peak['original_height'],
                    'CTA': cta
                })
    
    return pd.DataFrame(processed_data)

# 处理所有样本（包括Bootstrap样本）
all_processed_peaks = []
unique_samples = df_bootstrap['Sample File'].nunique()
processed_count = 0

print(f"开始处理 {unique_samples} 个样本的峰数据...")

for sample_file, group in df_bootstrap.groupby('Sample File'):
    processed_count += 1
    if processed_count % 50 == 0 or processed_count == unique_samples:
        print(f"处理进度: {processed_count}/{unique_samples} ({processed_count/unique_samples*100:.1f}%)")
    
    sample_peaks = process_peaks_with_cta(group)
    if not sample_peaks.empty:
        all_processed_peaks.append(sample_peaks)

df_peaks = pd.concat(all_processed_peaks, ignore_index=True) if all_processed_peaks else pd.DataFrame()
print(f"处理后的峰数据形状: {df_peaks.shape}")

# =====================
# 7. 综合特征工程
# =====================
print("\n=== 步骤4: 综合特征工程 ===")

def extract_comprehensive_features_v5(sample_file, sample_peaks):
    """基于文档第4节的综合特征体系构建"""
    if sample_peaks.empty:
        return {}
    
    features = {'Sample File': sample_file}
    
    # 基础数据准备
    total_peaks = len(sample_peaks)
    all_heights = sample_peaks['Height'].values
    all_sizes = sample_peaks['Size'].values
    
    # 按位点分组统计
    locus_groups = sample_peaks.groupby('Marker')
    alleles_per_locus = locus_groups['Allele'].nunique()
    locus_heights = locus_groups['Height'].sum()
    
    expected_autosomal_count = 23
    
    # A类：谱图层面基础统计特征
    features['mac_profile'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['total_distinct_alleles'] = sample_peaks['Allele'].nunique()
    
    if expected_autosomal_count > 0:
        all_locus_counts = np.zeros(expected_autosomal_count)
        all_locus_counts[:len(alleles_per_locus)] = alleles_per_locus.values
        features['avg_alleles_per_locus'] = np.mean(all_locus_counts)
        features['std_alleles_per_locus'] = np.std(all_locus_counts)
    else:
        features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    
    # MGTN系列
    for N in [2, 3, 4, 5, 6]:
        features[f'loci_gt{N}_alleles'] = (alleles_per_locus >= N).sum()
    
    # 等位基因计数的熵
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['allele_count_dist_entropy'] = calculate_entropy(counts.values)
    else:
        features['allele_count_dist_entropy'] = 0
    
    # B类：峰高、平衡性及随机效应特征
    if total_peaks > 0:
        features['avg_peak_height'] = np.mean(all_heights)
        features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
        features['min_peak_height'] = np.min(all_heights)
        features['max_peak_height'] = np.max(all_heights)
        
        # 峰高比(PHR)相关统计
        phr_values = []
        for marker, marker_group in locus_groups:
            if len(marker_group) == 2:
                heights = marker_group['Height'].values
                phr = min(heights) / max(heights) if max(heights) > 0 else 0
                phr_values.append(phr)
        
        if phr_values:
            features['avg_phr'] = np.mean(phr_values)
            features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
            features['min_phr'] = np.min(phr_values)
            features['median_phr'] = np.median(phr_values)
            features['num_loci_with_phr'] = len(phr_values)
            features['num_severe_imbalance_loci'] = sum(phr <= PHR_IMBALANCE_THRESHOLD for phr in phr_values)
            features['ratio_severe_imbalance_loci'] = features['num_severe_imbalance_loci'] / len(phr_values)
        else:
            for key in ['avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr', 
                       'num_severe_imbalance_loci', 'ratio_severe_imbalance_loci']:
                features[key] = 0
        
        # 峰高分布统计矩
        if total_peaks > 2:
            features['skewness_peak_height'] = stats.skew(all_heights)
            features['kurtosis_peak_height'] = stats.kurtosis(all_heights, fisher=False)
        else:
            features['skewness_peak_height'] = 0
            features['kurtosis_peak_height'] = 0
        
        # 峰高多峰性
        try:
            log_heights = np.log(all_heights + 1)
            if len(np.unique(log_heights)) > 1:
                hist, _ = np.histogram(log_heights, bins=min(10, total_peaks))
                peaks_found, _ = find_peaks(hist)
                features['modality_peak_height'] = len(peaks_found)
            else:
                features['modality_peak_height'] = 1
        except:
            features['modality_peak_height'] = 1
        
        # 饱和效应
        saturated_peaks = (sample_peaks['Original_Height'] >= SATURATION_THRESHOLD).sum()
        features['num_saturated_peaks'] = saturated_peaks
        features['ratio_saturated_peaks'] = saturated_peaks / total_peaks
    else:
        for key in ['avg_peak_height', 'std_peak_height', 'min_peak_height', 'max_peak_height',
                   'avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr',
                   'num_severe_imbalance_loci', 'ratio_severe_imbalance_loci',
                   'skewness_peak_height', 'kurtosis_peak_height', 'modality_peak_height',
                   'num_saturated_peaks', 'ratio_saturated_peaks']:
            features[key] = 0
    
    # C类：信息论及谱图复杂度特征
    if len(locus_heights) > 0:
        total_height = locus_heights.sum()
        if total_height > 0:
            locus_probs = locus_heights / total_height
            features['inter_locus_balance_entropy'] = calculate_entropy(locus_probs.values)
        else:
            features['inter_locus_balance_entropy'] = 0
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
                entropy = calculate_entropy(probs)
                locus_entropies.append(entropy)
    
    features['avg_locus_allele_entropy'] = np.mean(locus_entropies) if locus_entropies else 0
    
    # 样本整体峰高分布熵
    if total_peaks > 0:
        log_heights = np.log(all_heights + 1)
        hist, _ = np.histogram(log_heights, bins=min(15, total_peaks))
        hist_probs = hist / hist.sum()
        hist_probs = hist_probs[hist_probs > 0]
        features['peak_height_entropy'] = calculate_entropy(hist_probs)
    else:
        features['peak_height_entropy'] = 0
    
    # 图谱完整性指标
    effective_loci_count = len(locus_groups)
    features['num_loci_with_effective_alleles'] = effective_loci_count
    features['num_loci_no_effective_alleles'] = max(0, expected_autosomal_count - effective_loci_count)
    
    # D类：DNA降解与信息丢失特征
    if total_peaks > 1:
        if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1]
        else:
            features['height_size_correlation'] = 0
        
        features['height_size_slope'] = calculate_ols_slope(all_sizes, all_heights)
        
        try:
            weights = all_heights / all_heights.sum()
            weighted_correlation = np.average(all_sizes, weights=weights)
            features['weighted_height_size_slope'] = calculate_ols_slope(all_sizes, all_heights)
        except:
            features['weighted_height_size_slope'] = 0
        
        if len(phr_values) > 1:
            phr_sizes = []
            for marker, marker_group in locus_groups:
                if len(marker_group) == 2:
                    avg_size = marker_group['Size'].mean()
                    phr_sizes.append(avg_size)
            
            if len(phr_sizes) == len(phr_values) and len(phr_sizes) > 1:
                features['phr_size_slope'] = calculate_ols_slope(phr_sizes, phr_values)
            else:
                features['phr_size_slope'] = 0
        else:
            features['phr_size_slope'] = 0
        
    else:
        for key in ['height_size_correlation', 'height_size_slope', 'weighted_height_size_slope', 'phr_size_slope']:
            features[key] = 0
    
    # 其他降解相关特征
    dropout_score = features['num_loci_no_effective_alleles'] / expected_autosomal_count if expected_autosomal_count > 0 else 0
    features['locus_dropout_score_weighted_by_size'] = dropout_score
    
    if len(locus_groups) > 1:
        locus_max_heights = []
        locus_avg_sizes = []
        for marker, marker_group in locus_groups:
            max_height = marker_group['Height'].max()
            avg_size = marker_group['Size'].mean()
            locus_max_heights.append(max_height)
            locus_avg_sizes.append(avg_size)
        
        features['degradation_index_rfu_per_bp'] = calculate_ols_slope(locus_avg_sizes, locus_max_heights)
    else:
        features['degradation_index_rfu_per_bp'] = 0
    
    # 简化的信息完整度比率
    small_fragment_loci = 0
    large_fragment_loci = 0
    small_fragment_effective = 0
    large_fragment_effective = 0
    
    for marker, marker_group in locus_groups:
        avg_size = marker_group['Size'].mean()
        if avg_size < 200:
            small_fragment_loci += 1
            small_fragment_effective += 1
        else:
            large_fragment_loci += 1
            large_fragment_effective += 1
    
    total_small_expected = expected_autosomal_count // 2
    total_large_expected = expected_autosomal_count - total_small_expected
    
    small_completeness = small_fragment_effective / total_small_expected if total_small_expected > 0 else 0
    large_completeness = large_fragment_effective / total_large_expected if total_large_expected > 0 else 0
    
    if large_completeness > 0:
        features['info_completeness_ratio_small_large'] = small_completeness / large_completeness
    else:
        features['info_completeness_ratio_small_large'] = small_completeness / 0.001
    
    return features

# 提取所有样本的特征
print("开始特征提取...")
start_time = time()

all_features = []
unique_samples = df_peaks['Sample File'].nunique() if not df_peaks.empty else 0
processed_count = 0

if df_peaks.empty:
    print("警告: 处理后的峰数据为空，使用原始数据进行特征提取")
    # 如果峰处理失败，回退到使用原始数据
    for sample_file in df_bootstrap['Sample File'].unique():
        features = {'Sample File': sample_file}
        # 添加默认特征值
        for feature_name in ['mac_profile', 'total_distinct_alleles', 'avg_alleles_per_locus']:
            features[feature_name] = 0
        all_features.append(features)
else:
    for sample_file, group in df_peaks.groupby('Sample File'):
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == unique_samples:
            print(f"特征提取进度: {processed_count}/{unique_samples} ({processed_count/unique_samples*100:.1f}%)")
        
        features = extract_comprehensive_features_v5(sample_file, group)
        if features:  # 确保特征不为空
            all_features.append(features)

if not all_features:
    print("错误: 无法提取特征，请检查数据格式")
    exit()

df_features = pd.DataFrame(all_features)

# 合并NoC标签
noc_map = df_bootstrap.groupby('Sample File')['NoC_True'].first().to_dict()
df_features['NoC_True'] = df_features['Sample File'].map(noc_map)
df_features = df_features.dropna(subset=['NoC_True'])

# 验证数据质量
if df_features.empty:
    print("错误: 特征数据为空，请检查数据处理流程")
    exit()

# 填充缺失值
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
df_features[numeric_cols] = df_features[numeric_cols].fillna(0)

print(f"特征提取完成，耗时: {time() - start_time:.2f}秒")
print(f"特征数据形状: {df_features.shape}")

# 检查特征数量
feature_cols_check = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
print(f"实际特征数量: {len(feature_cols_check)}")

if len(feature_cols_check) == 0:
    print("错误: 没有可用的特征，请检查特征提取函数")
    exit()

# =====================
# 8. 特征选择与模型训练
# =====================
print("\n=== 步骤5: 模型训练与验证 ===")

# 准备特征和标签
feature_cols = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
X = df_features[feature_cols]
y = df_features['NoC_True']

print(f"原始特征数量: {len(feature_cols)}")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 编码标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 使用LassoCV进行特征选择
print("执行Lasso特征选择...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000)
lasso_cv.fit(X_scaled, y_encoded)

# 基于Lasso系数选择特征
selector = SelectFromModel(lasso_cv, prefit=True)
X_selected = selector.transform(X_scaled)
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

print(f"Lasso选择的特征数量: {len(selected_features)}")
print(f"特征选择比例: {len(selected_features)/len(feature_cols):.2%}")

# 使用选择的特征
X_final = pd.DataFrame(X_selected, columns=selected_features)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 交叉验证设置
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===== Gradient Boosting 主模型 =====
print("\n训练Gradient Boosting模型...")

# 超参数网格
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 9],
    'learning_rate': [0.1, 0.5, 0.01],
    'subsample': [ 0.8, 0.9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
gb_grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("执行网格搜索...")
gb_grid_search.fit(X_train, y_train)

best_gb_model = gb_grid_search.best_estimator_
print(f"最佳参数: {gb_grid_search.best_params_}")
print(f"最佳CV分数: {gb_grid_search.best_score_:.4f}")

# 评估模型
y_pred_gb = best_gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting测试准确率: {gb_accuracy:.4f}")

# ===== 对比模型 =====
print("\n训练对比模型...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_scores = cross_val_score(rf_model, X_final, y_encoded, cv=cv, scoring='accuracy')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest CV: {rf_scores.mean():.4f}±{rf_scores.std():.4f}, Test: {rf_accuracy:.4f}")

# ===== 集成模型 =====
print("\n构建集成模型...")

ensemble_model = VotingClassifier(
    estimators=[
        ('gb', best_gb_model),
        ('rf', rf_model)
    ],
    voting='soft',
    weights=[2, 1]  # 给GB更高权重
)

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"集成模型测试准确率: {ensemble_accuracy:.4f}")

# 选择最佳模型
models = {
    'Gradient Boosting': (best_gb_model, gb_accuracy, y_pred_gb),
    'Random Forest': (rf_model, rf_accuracy, y_pred_rf),
    'Ensemble': (ensemble_model, ensemble_accuracy, y_pred_ensemble)
}

best_model_name = max(models.keys(), key=lambda x: models[x][1])
best_model, best_accuracy, best_predictions = models[best_model_name]

print(f"\n最佳模型: {best_model_name} (准确率: {best_accuracy:.4f})")

# =====================
# 8. 详细评估与可视化
# =====================
print("\n=== 步骤6: 详细评估与可视化 ===")

# 转换标签用于显示
y_test_orig = label_encoder.inverse_transform(y_test)
best_predictions_orig = label_encoder.inverse_transform(best_predictions)

# 分类报告
class_names = [str(x) for x in sorted(label_encoder.classes_)]
print(f"\n{best_model_name} 详细分类报告:")
print(classification_report(y_test_orig, best_predictions_orig, target_names=[f"{x}人" for x in class_names]))

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_orig, best_predictions_orig)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=[f"{x}人" for x in class_names], 
           yticklabels=[f"{x}人" for x in class_names])
plt.title(f'{best_model_name} 混淆矩阵')
plt.ylabel('真实NoC')
plt.xlabel('预测NoC')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

# 模型性能对比
plt.figure(figsize=(12, 6))
model_names = list(models.keys())
accuracies = [models[name][1] for name in model_names]

colors = ['#d62728' if name == best_model_name else '#1f77b4' for name in model_names]
bars = plt.bar(model_names, accuracies, color=colors)
plt.ylim(0, 1.1)
plt.ylabel('测试准确率')
plt.title('模型性能对比')
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
plt.close()

# 特征重要性分析
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 显示前15个重要特征
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'{best_model_name} 特征重要性 (前15)')
    plt.xlabel('特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"\n{best_model_name} Top 10 重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:35} {row['importance']:.4f}")

# 学习曲线
from sklearn.model_selection import learning_curve

try:
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_final, y_encoded, cv=cv, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )

    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练集')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='验证集')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')

    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.title(f'{best_model_name} 学习曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'), dpi=300)
    plt.close()
except Exception as e:
    print(f"学习曲线生成失败: {e}")

# =====================
# 9. SHAP可解释性分析
# =====================
if SHAP_AVAILABLE and hasattr(best_model, 'feature_importances_'):
    print("\n=== 步骤7: SHAP可解释性分析 ===")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(best_model)
        
        # 计算SHAP值（使用小样本）
        shap_sample_size = min(20, len(X_test))
        X_shap = X_test.iloc[:shap_sample_size]
        shap_values = explainer.shap_values(X_shap)
        
        # 对于多分类，取第一个类别的SHAP值或计算平均
        if isinstance(shap_values, list):
            shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values_mean = np.abs(shap_values)
        
        # SHAP特征重要性
        feature_shap_importance = np.mean(shap_values_mean, axis=0)
        shap_importance_df = pd.DataFrame({
            'feature': selected_features,
            'shap_importance': feature_shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_shap_features = shap_importance_df.head(10)
        sns.barplot(data=top_shap_features, x='shap_importance', y='feature')
        plt.title('SHAP特征重要性 (前10)')
        plt.xlabel('平均SHAP重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'shap_importance.png'), dpi=300)
        plt.close()
        
        print("SHAP Top 10 重要特征:")
        for idx, row in shap_importance_df.head(10).iterrows():
            print(f"  {row['feature']:35} {row['shap_importance']:.4f}")
        
        # 尝试生成SHAP摘要图
        try:
            if isinstance(shap_values, list):
                # 多分类情况，使用第一个类别
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_shap, feature_names=selected_features, 
                            plot_type="bar", show=False)
            plt.title("SHAP摘要图")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"生成SHAP摘要图失败: {e}")
            
    except Exception as e:
        print(f"SHAP分析失败: {e}")

# =====================
# 10. 应用模型进行预测
# =====================
print("\n=== 步骤8: 应用模型进行全样本预测 ===")

# 对所有样本进行预测
y_pred_all_encoded = best_model.predict(X_final)
y_pred_all = label_encoder.inverse_transform(y_pred_all_encoded)

# 添加预测结果到特征数据框
df_features['Predicted_NoC'] = y_pred_all

# 计算整体准确率
overall_accuracy = (df_features['Predicted_NoC'] == df_features['NoC_True']).mean()
print(f"整体预测准确率: {overall_accuracy:.4f}")

# 各NoC类别的准确率
noc_accuracy = df_features.groupby('NoC_True').apply(
    lambda x: (x['Predicted_NoC'] == x['NoC_True']).mean()
).reset_index(name='Accuracy')

print("\n各NoC类别预测准确率:")
for _, row in noc_accuracy.iterrows():
    print(f"  {int(row['NoC_True'])}人: {row['Accuracy']:.4f}")

# NoC预测准确率可视化
plt.figure(figsize=(10, 6))
sns.barplot(data=noc_accuracy, x='NoC_True', y='Accuracy')
plt.ylim(0, 1.1)
plt.xlabel('真实NoC')
plt.ylabel('预测准确率')
plt.title('各NoC类别预测准确率')

for i, row in noc_accuracy.iterrows():
    plt.text(i, row['Accuracy'] + 0.03, f"{row['Accuracy']:.3f}", 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'noc_accuracy_by_class.png'), dpi=300)
plt.close()

# =====================
# 11. 保存结果
# =====================
print("\n=== 步骤9: 保存结果 ===")

# 保存特征数据
df_features.to_csv(os.path.join(DATA_DIR, 'noc_features_with_predictions.csv'), 
                   index=False, encoding='utf-8-sig')

# 保存模型性能摘要
summary = {
    'model_info': {
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'overall_accuracy': float(overall_accuracy),
        'feature_selection_method': 'LassoCV',
        'selected_features_count': len(selected_features)
    },
    'data_info': {
        'total_samples': len(df_features),
        'feature_count_original': len(feature_cols),
        'feature_count_selected': len(selected_features),
        'noc_distribution': df_features['NoC_True'].value_counts().sort_index().to_dict()
    },
    'model_performance': {
        name: {'accuracy': float(acc)} for name, (_, acc, _) in models.items()
    },
    'noc_class_accuracy': {
        int(row['NoC_True']): float(row['Accuracy']) 
        for _, row in noc_accuracy.iterrows()
    },
    'selected_features': selected_features,
    'hyperparameters': gb_grid_search.best_params_ if best_model_name == 'Gradient Boosting' else {},
    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(DATA_DIR, 'noc_analysis_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 保存最佳模型
import joblib
try:
    joblib.dump(best_model, os.path.join(DATA_DIR, f'best_noc_model_{best_model_name.lower().replace(" ", "_")}.pkl'))
    joblib.dump(scaler, os.path.join(DATA_DIR, 'feature_scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(DATA_DIR, 'label_encoder.pkl'))
    print("模型和预处理器已保存")
except Exception as e:
    print(f"保存模型时出错: {e}")

# =====================
# 12. 最终报告
# =====================
print("\n" + "="*60)
print("           法医混合STR图谱NoC识别 - 最终报告")
print("="*60)

print(f"\n📊 数据概况:")
print(f"   • 总样本数: {len(df_features)}")
print(f"   • NoC分布: {dict(df_features['NoC_True'].value_counts().sort_index())}")
print(f"   • 原始特征数: {len(feature_cols)}")
print(f"   • 选择特征数: {len(selected_features)}")

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   • 测试集准确率: {best_accuracy:.4f}")
print(f"   • 整体准确率: {overall_accuracy:.4f}")

if best_model_name == 'Gradient Boosting':
    print(f"   • 最佳超参数: {gb_grid_search.best_params_}")

print(f"\n📈 各类别表现:")
for _, row in noc_accuracy.iterrows():
    noc = int(row['NoC_True'])
    acc = row['Accuracy']
    print(f"   • {noc}人混合样本: {acc:.4f}")

print(f"\n🔍 Top 5 重要特征:")
if hasattr(best_model, 'feature_importances_'):
    top_5_features = feature_importance.head(5)
    for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
        print(f"   {i}. {row['feature']:30} ({row['importance']:.4f})")

print(f"\n💾 保存的文件:")
print(f"   • 特征数据: noc_features_with_predictions.csv")
print(f"   • 分析摘要: noc_analysis_summary.json")
print(f"   • 最佳模型: best_noc_model_{best_model_name.lower().replace(' ', '_')}.pkl")
print(f"   • 图表目录: {PLOTS_DIR}")

print(f"\n📋 模型解释:")
print(f"   • 本模型基于{len(selected_features)}个精选特征")
print(f"   • 特征涵盖图谱统计、峰高分布、平衡性、信息熵、降解指标等")
print(f"   • 使用{best_model_name}算法实现NoC自动识别")
print(f"   • 整体准确率达到{overall_accuracy:.1%}，具有良好的实用价值")

if SHAP_AVAILABLE:
    print(f"   • 已生成SHAP可解释性分析，增强模型透明度")

print("\n✅ 分析完成！")
print("="*60)