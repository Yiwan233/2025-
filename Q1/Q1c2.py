# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别 (随机森林优化版)

版本: V8.0 - Random Forest Optimization
日期: 2025-06-06
描述: 基于RFECV特征选择 + 随机森林深度优化 + 详细NoC性能分析
主要特点:
1. 使用随机森林替代梯度提升机
2. 三阶段参数优化（粗调→细调→微调）
3. 详细的各NoC类别性能分析
4. 袋外(OOB)评估增强模型可靠性
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
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, 
                                   GridSearchCV, RandomizedSearchCV, validation_curve,
                                   cross_validate, RepeatedStratifiedKFold)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, make_scorer, f1_score, balanced_accuracy_score,
                           precision_recall_curve, average_precision_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

# 可解释性分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP不可用，将跳过可解释性分析")
    SHAP_AVAILABLE = False

# 配置
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=== 法医混合STR图谱NoC智能识别系统 V8.0 (随机森林版) ===")
print("RFECV特征选择 + 随机森林深度优化 + 详细性能分析")

# =====================
# 1. 文件路径与基础设置
# =====================
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'noc_rf_optimization')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# 关键参数设置
HEIGHT_THRESHOLD = 50
SATURATION_THRESHOLD = 30000
CTA_THRESHOLD = 0.5
PHR_IMBALANCE_THRESHOLD = 0.6

# =====================
# 2. 中文特征名称映射
# =====================
FEATURE_NAME_MAPPING = {
    # A类：图谱层面基础统计特征
    'mac_profile': '样本最大等位基因数',
    'total_distinct_alleles': '样本总特异等位基因数',
    'avg_alleles_per_locus': '每位点平均等位基因数',
    'std_alleles_per_locus': '每位点等位基因数标准差',
    'loci_gt2_alleles': '等位基因数大于2的位点数',
    'loci_gt3_alleles': '等位基因数大于3的位点数',
    'loci_gt4_alleles': '等位基因数大于4的位点数',
    'loci_gt5_alleles': '等位基因数大于5的位点数',
    'loci_gt6_alleles': '等位基因数大于6的位点数',
    'allele_count_dist_entropy': '等位基因计数分布熵',
    
    # B类：峰高、平衡性及随机效应特征
    'avg_peak_height': '平均峰高',
    'std_peak_height': '峰高标准差',
    'min_peak_height': '最小峰高',
    'max_peak_height': '最大峰高',
    'peak_height_cv': '峰高变异系数',
    'avg_phr': '平均峰高比',
    'std_phr': '峰高比标准差',
    'min_phr': '最小峰高比',
    'median_phr': '峰高比中位数',
    'num_loci_with_phr': '可计算峰高比的位点数',
    'num_severe_imbalance_loci': '严重失衡位点数',
    'ratio_severe_imbalance_loci': '严重失衡位点比例',
    'skewness_peak_height': '峰高分布偏度',
    'kurtosis_peak_height': '峰高分布峭度',
    'modality_peak_height': '峰高分布多峰性',
    'num_saturated_peaks': '饱和峰数量',
    'ratio_saturated_peaks': '饱和峰比例',
    
    # C类：信息论及图谱复杂度特征
    'inter_locus_balance_entropy': '位点间平衡熵',
    'avg_locus_allele_entropy': '平均位点等位基因熵',
    'peak_height_entropy': '峰高分布熵',
    'num_loci_with_effective_alleles': '有效等位基因位点数',
    'num_loci_no_effective_alleles': '无有效等位基因位点数',
    
    # D类：DNA降解与信息丢失特征
    'height_size_correlation': '峰高片段大小相关性',
    'height_size_slope': '峰高片段大小回归斜率',
    'weighted_height_size_slope': '加权峰高片段大小斜率',
    'phr_size_slope': '峰高比片段大小斜率',
    'locus_dropout_score_weighted_by_size': '片段大小加权位点丢失评分',
    'degradation_index_rfu_per_bp': 'RFU每碱基对降解指数',
    'info_completeness_ratio_small_large': '小大片段信息完整度比率',
    
    # 新增高级特征
    'peak_height_quantile_ratio': '峰高分位数比率',
    'allele_diversity_index': '等位基因多样性指数',
    'peak_pattern_complexity': '峰模式复杂度',
    'heterozygosity_rate': '杂合率',
    'peak_clustering_coefficient': '峰聚类系数'
}

def get_chinese_name(feature_name):
    """获取特征的中文名称"""
    return FEATURE_NAME_MAPPING.get(feature_name, feature_name)

# =====================
# 3. 辅助函数定义
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
# 5. 简化峰处理与CTA评估
# =====================
print("\n=== 步骤2: 简化峰处理与信号表征 ===")

def process_peaks_simplified(sample_data):
    """简化的峰处理函数"""
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
            
            # 简化的CTA计算
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

# 处理所有样本
print("处理峰数据...")
all_processed_peaks = []
for sample_file, group in df.groupby('Sample File'):
    sample_peaks = process_peaks_simplified(group)
    if not sample_peaks.empty:
        all_processed_peaks.append(sample_peaks)

df_peaks = pd.concat(all_processed_peaks, ignore_index=True) if all_processed_peaks else pd.DataFrame()
print(f"处理后的峰数据形状: {df_peaks.shape}")

# =====================
# 6. 增强版特征工程
# =====================
print("\n=== 步骤3: 增强版特征工程 ===")

def extract_enhanced_features(sample_file, sample_peaks):
    """提取增强的特征集，包括新的高级特征"""
    if sample_peaks.empty:
        return {}
    
    features = {'样本文件': sample_file}
    
    # 基础数据准备
    total_peaks = len(sample_peaks)
    all_heights = sample_peaks['Height'].values
    all_sizes = sample_peaks['Size'].values
    
    # 按位点分组统计
    locus_groups = sample_peaks.groupby('Marker')
    alleles_per_locus = locus_groups['Allele'].nunique()
    locus_heights = locus_groups['Height'].sum()
    
    # A类：图谱层面基础统计特征
    features['样本最大等位基因数'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['样本总特异等位基因数'] = sample_peaks['Allele'].nunique()
    features['每位点平均等位基因数'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
    features['每位点等位基因数标准差'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    
    # MGTN系列
    for N in [2, 3, 4, 5, 6]:
        features[f'等位基因数大于{N}的位点数'] = (alleles_per_locus >= N).sum()
    
    # 等位基因计数分布的熵
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['等位基因计数分布熵'] = calculate_entropy(counts.values)
    else:
        features['等位基因计数分布熵'] = 0
    
    # B类：峰高、平衡性及随机效应特征
    if total_peaks > 0:
        # 基础峰高统计
        features['平均峰高'] = np.mean(all_heights)
        features['峰高标准差'] = np.std(all_heights) if total_peaks > 1 else 0
        features['最小峰高'] = np.min(all_heights)
        features['最大峰高'] = np.max(all_heights)
        features['峰高变异系数'] = features['峰高标准差'] / features['平均峰高'] if features['平均峰高'] > 0 else 0
        
        # 新增：峰高分位数比率
        if total_peaks >= 4:
            q25 = np.percentile(all_heights, 25)
            q75 = np.percentile(all_heights, 75)
            features['峰高分位数比率'] = q75 / q25 if q25 > 0 else 0
        else:
            features['峰高分位数比率'] = 1.0
        
        # 峰高比相关特征
        phr_values = []
        for marker, marker_group in locus_groups:
            if len(marker_group) == 2:
                heights = marker_group['Height'].values
                phr = min(heights) / max(heights) if max(heights) > 0 else 0
                phr_values.append(phr)
        
        if phr_values:
            features['平均峰高比'] = np.mean(phr_values)
            features['峰高比标准差'] = np.std(phr_values) if len(phr_values) > 1 else 0
            features['最小峰高比'] = np.min(phr_values)
            features['峰高比中位数'] = np.median(phr_values)
            features['可计算峰高比的位点数'] = len(phr_values)
            features['严重失衡位点数'] = sum(phr <= PHR_IMBALANCE_THRESHOLD for phr in phr_values)
            features['严重失衡位点比例'] = features['严重失衡位点数'] / len(phr_values)
        else:
            for key in ['平均峰高比', '峰高比标准差', '最小峰高比', '峰高比中位数', 
                       '可计算峰高比的位点数', '严重失衡位点数', '严重失衡位点比例']:
                features[key] = 0
        
        # 峰高分布形状特征
        if total_peaks > 2:
            features['峰高分布偏度'] = stats.skew(all_heights)
            features['峰高分布峭度'] = stats.kurtosis(all_heights, fisher=False)
        else:
            features['峰高分布偏度'] = 0
            features['峰高分布峭度'] = 0
        
        # 峰高多峰性指标
        try:
            log_heights = np.log(all_heights + 1)
            if len(np.unique(log_heights)) > 1:
                hist, _ = np.histogram(log_heights, bins=min(10, total_peaks))
                peaks_found, _ = find_peaks(hist)
                features['峰高分布多峰性'] = len(peaks_found)
            else:
                features['峰高分布多峰性'] = 1
        except:
            features['峰高分布多峰性'] = 1
        
        # 饱和效应特征
        saturated_peaks = (sample_peaks['Original_Height'] >= SATURATION_THRESHOLD).sum()
        features['饱和峰数量'] = saturated_peaks
        features['饱和峰比例'] = saturated_peaks / total_peaks
    else:
        # 空值填充
        for key in ['平均峰高', '峰高标准差', '最小峰高', '最大峰高', '峰高变异系数',
                   '峰高分位数比率', '平均峰高比', '峰高比标准差', '最小峰高比', '峰高比中位数',
                   '可计算峰高比的位点数', '严重失衡位点数', '严重失衡位点比例',
                   '峰高分布偏度', '峰高分布峭度', '峰高分布多峰性',
                   '饱和峰数量', '饱和峰比例']:
            features[key] = 0
    
    # C类：信息论及图谱复杂度特征
    if len(locus_heights) > 0:
        total_height = locus_heights.sum()
        if total_height > 0:
            locus_probs = locus_heights / total_height
            features['位点间平衡熵'] = calculate_entropy(locus_probs.values)
        else:
            features['位点间平衡熵'] = 0
    else:
        features['位点间平衡熵'] = 0
    
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
    
    features['平均位点等位基因熵'] = np.mean(locus_entropies) if locus_entropies else 0
    
    # 样本整体峰高分布熵
    if total_peaks > 0:
        log_heights = np.log(all_heights + 1)
        hist, _ = np.histogram(log_heights, bins=min(15, total_peaks))
        hist_probs = hist / hist.sum()
        hist_probs = hist_probs[hist_probs > 0]
        features['峰高分布熵'] = calculate_entropy(hist_probs)
    else:
        features['峰高分布熵'] = 0
    
    # 图谱完整性指标
    effective_loci_count = len(locus_groups)
    features['有效等位基因位点数'] = effective_loci_count
    features['无有效等位基因位点数'] = max(0, 20 - effective_loci_count)  # 假设20个位点
    
    # D类：DNA降解与信息丢失特征
    if total_peaks > 1:
        # 峰高与片段大小的相关性
        if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['峰高片段大小相关性'] = np.corrcoef(all_heights, all_sizes)[0, 1]
        else:
            features['峰高片段大小相关性'] = 0
        
        # 线性回归斜率
        features['峰高片段大小回归斜率'] = calculate_ols_slope(all_sizes, all_heights)
        
        # 加权回归斜率（简化版）
        try:
            weights = all_heights / all_heights.sum()
            features['加权峰高片段大小斜率'] = calculate_ols_slope(all_sizes, all_heights)
        except:
            features['加权峰高片段大小斜率'] = 0
        
        # PHR随片段大小变化的斜率
        if len(phr_values) > 1:
            phr_sizes = []
            for marker, marker_group in locus_groups:
                if len(marker_group) == 2:
                    avg_size = marker_group['Size'].mean()
                    phr_sizes.append(avg_size)
            
            if len(phr_sizes) == len(phr_values) and len(phr_sizes) > 1:
                features['峰高比片段大小斜率'] = calculate_ols_slope(phr_sizes, phr_values)
            else:
                features['峰高比片段大小斜率'] = 0
        else:
            features['峰高比片段大小斜率'] = 0
    else:
        for key in ['峰高片段大小相关性', '峰高片段大小回归斜率', '加权峰高片段大小斜率', '峰高比片段大小斜率']:
            features[key] = 0
    
    # 位点丢失评分（简化版）
    dropout_score = features['无有效等位基因位点数'] / 20  # 基于20个位点的假设
    features['片段大小加权位点丢失评分'] = dropout_score
    
    # RFU每碱基对衰减指数（简化版）
    if len(locus_groups) > 1:
        locus_max_heights = []
        locus_avg_sizes = []
        for marker, marker_group in locus_groups:
            max_height = marker_group['Height'].max()
            avg_size = marker_group['Size'].mean()
            locus_max_heights.append(max_height)
            locus_avg_sizes.append(avg_size)
        
        features['RFU每碱基对降解指数'] = calculate_ols_slope(locus_avg_sizes, locus_max_heights)
    else:
        features['RFU每碱基对降解指数'] = 0
    
    # 小大片段信息完整度比率（简化版）
    small_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() < 200)
    large_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() >= 200)
    
    if large_fragment_effective > 0:
        features['小大片段信息完整度比率'] = small_fragment_effective / large_fragment_effective
    else:
        features['小大片段信息完整度比率'] = small_fragment_effective / 0.001
    
    # 新增高级特征
    # 1. 等位基因多样性指数 (Simpson's diversity index)
    if total_peaks > 0:
        allele_counts = sample_peaks['Allele'].value_counts()
        total_alleles = allele_counts.sum()
        diversity = 1 - sum((n/total_alleles)**2 for n in allele_counts.values)
        features['等位基因多样性指数'] = diversity
    else:
        features['等位基因多样性指数'] = 0
    
    # 2. 峰模式复杂度
    if len(locus_groups) > 0:
        pattern_complexity = 0
        for marker, group in locus_groups:
            n_alleles = len(group)
            if n_alleles > 2:
                # 计算峰高的标准差作为复杂度的一部分
                height_std = group['Height'].std()
                pattern_complexity += height_std / group['Height'].mean() if group['Height'].mean() > 0 else 0
        features['峰模式复杂度'] = pattern_complexity / len(locus_groups)
    else:
        features['峰模式复杂度'] = 0
    
    # 3. 杂合率
    heterozygous_loci = sum(1 for marker, group in locus_groups if len(group) == 2)
    features['杂合率'] = heterozygous_loci / len(locus_groups) if len(locus_groups) > 0 else 0
    
    # 4. 峰聚类系数（测量峰的聚集程度）
    if total_peaks > 2:
        # 使用峰高的变异系数和峰间距离
        height_cv = np.std(all_heights) / np.mean(all_heights) if np.mean(all_heights) > 0 else 0
        size_range = np.max(all_sizes) - np.min(all_sizes)
        features['峰聚类系数'] = height_cv * (1 - size_range/1000)  # 归一化
    else:
        features['峰聚类系数'] = 0
    
    return features

# 提取所有样本的特征
print("开始增强版特征提取...")
all_features = []

if df_peaks.empty:
    print("警告: 处理后的峰数据为空，使用默认特征")
    for sample_file in df['Sample File'].unique():
        features = {'样本文件': sample_file, '样本最大等位基因数': 0}
        all_features.append(features)
else:
    for sample_file, group in df_peaks.groupby('Sample File'):
        features = extract_enhanced_features(sample_file, group)
        if features:
            all_features.append(features)

df_features = pd.DataFrame(all_features)

# 合并NoC标签
noc_map = df.groupby('Sample File')['NoC_True'].first().to_dict()
df_features['贡献者人数'] = df_features['样本文件'].map(noc_map)
df_features = df_features.dropna(subset=['贡献者人数'])

# 填充缺失值
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
df_features[numeric_cols] = df_features[numeric_cols].fillna(0)

print(f"特征数据形状: {df_features.shape}")
print(f"特征数量: {len([col for col in df_features.columns if col not in ['样本文件', '贡献者人数']])}")

# =====================
# 7. RFECV特征选择
# =====================
print("\n=== 步骤4: RFECV递归特征消除 ===")

# 准备数据
feature_cols = [col for col in df_features.columns if col not in ['样本文件', '贡献者人数']]
X = df_features[feature_cols].fillna(0)
y = df_features['贡献者人数']

print(f"原始特征数: {len(feature_cols)}")
print(f"样本数: {len(X)}")
print(f"NoC分布: {y.value_counts().sort_index().to_dict()}")

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据平衡性检查
print("\n数据平衡性检查:")
class_distribution = pd.Series(y_encoded).value_counts().sort_index()
for cls, count in class_distribution.items():
    original_cls = label_encoder.inverse_transform([cls])[0]
    print(f"  {original_cls}人: {count}个样本")

min_samples = class_distribution.min()
imbalance_ratio = class_distribution.max() / min_samples
print(f"不平衡比例: {imbalance_ratio:.2f}")

# 处理类别不平衡
if imbalance_ratio > 3:
    try:
        from imblearn.over_sampling import SMOTE
        print("使用SMOTE处理类别不平衡...")
        k_neighbors = min(3, min_samples - 1) if min_samples > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_scaled, y_encoded = smote.fit_resample(X_scaled, y_encoded)
        print(f"SMOTE后样本数: {len(X_scaled)}")
    except ImportError:
        print("未安装imblearn，跳过SMOTE处理")
    except Exception as e:
        print(f"SMOTE处理失败: {e}")

# 设置交叉验证
min_class_size = pd.Series(y_encoded).value_counts().min()
cv_folds = min(5, min_class_size)
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

print(f"\n开始RFECV特征选择（{cv_folds}折交叉验证）...")

# 创建基础随机森林用于RFECV
base_estimator = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 自定义评分函数（平衡准确率）
from sklearn.metrics import balanced_accuracy_score
balanced_scorer = make_scorer(balanced_accuracy_score)

# 执行RFECV
print("执行递归特征消除交叉验证...")
start_time = time()

rfecv = RFECV(
    estimator=base_estimator,
    step=1,
    cv=cv,
    scoring=balanced_scorer,
    n_jobs=-1,
    verbose=1
)

rfecv.fit(X_scaled, y_encoded)

elapsed_time = time() - start_time
print(f"RFECV完成，耗时: {elapsed_time:.1f}秒")

# 获取选择的特征
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
selected_indices = [i for i in range(len(feature_cols)) if rfecv.support_[i]]

print(f"\nRFECV选择的特征数: {len(selected_features)}")
print(f"最优特征数: {rfecv.n_features_}")

# 获取最佳分数
if hasattr(rfecv, 'cv_results_'):
    best_score = max(rfecv.cv_results_['mean_test_score'])
elif hasattr(rfecv, 'grid_scores_'):
    best_score = max(rfecv.grid_scores_)
else:
    best_score = 0.0

print(f"最佳交叉验证分数: {best_score:.4f}")

print("\nRFECV选择的特征:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feature}")

# 创建最终的特征矩阵
X_selected = X_scaled[:, selected_indices]

# 可视化RFECV结果
plt.figure(figsize=(12, 8))

if hasattr(rfecv, 'cv_results_'):
    scores = rfecv.cv_results_['mean_test_score']
elif hasattr(rfecv, 'grid_scores_'):
    scores = rfecv.grid_scores_
else:
    scores = [0.0] * len(feature_cols)

plt.plot(range(1, len(scores) + 1), scores, 'bo-')
plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', 
           label=f'最优特征数: {rfecv.n_features_}')
plt.xlabel('特征数量')
plt.ylabel('交叉验证分数')
plt.title('RFECV特征选择结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'RFECV结果.png'), dpi=300, bbox_inches='tight')
plt.close()

# =====================
# 8. 随机森林优化器
# =====================
class RandomForestOptimizer:
    """随机森林优化器，针对不同NoC使用不同策略"""
    
    def __init__(self):
        self.noc_specific_configs = {
            2: {
                'class_weight': {2: 1.0},
                'feature_selection_params': {'min_features_to_select': 8},
                'model_params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True
                }
            },
            3: {
                'class_weight': {3: 1.2},
                'feature_selection_params': {'min_features_to_select': 10},
                'model_params': {
                    'n_estimators': 300,
                    'max_depth': 12,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True
                }
            },
            4: {
                'class_weight': {4: 2.0},
                'feature_selection_params': {'min_features_to_select': 12},
                'model_params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'oob_score': True
                }
            },
            5: {
                'class_weight': {5: 3.0},
                'feature_selection_params': {'min_features_to_select': 15},
                'model_params': {
                    'n_estimators': 800,
                    'max_depth': 20,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'oob_score': True
                }
            }
        }
    
    def get_sample_weights(self, y):
        """计算样本权重，对少数类给予更高权重"""
        # 基础权重
        base_weights = compute_sample_weight('balanced', y)
        
        # 对4人和5人样本额外加权
        enhanced_weights = base_weights.copy()
        for i, label in enumerate(y):
            if label == 4:
                enhanced_weights[i] *= 2.5
            elif label == 5:
                enhanced_weights[i] *= 4.0
        
        return enhanced_weights

# =====================
# 9. 随机森林深度优化
# =====================
print("\n=== 步骤5: 随机森林深度优化 ===")

# 自定义分层划分
def custom_stratified_split(X, y, test_size=0.25, random_state=42):
    """确保每个类别在训练集和测试集中都有代表"""
    np.random.seed(random_state)
    unique_classes = np.unique(y)
    
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = len(cls_indices)
        
        if n_cls <= 2:
            n_test = 1
            n_train = n_cls - 1
        else:
            n_test = max(1, int(n_cls * test_size))
            n_train = n_cls - n_test
        
        np.random.shuffle(cls_indices)
        test_indices = cls_indices[:n_test]
        train_indices = cls_indices[n_test:n_test+n_train]
        
        X_train_list.append(X[train_indices])
        X_test_list.append(X[test_indices])
        y_train_list.append(y[train_indices])
        y_test_list.append(y[test_indices])
    
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.hstack(y_train_list)
    y_test = np.hstack(y_test_list)
    
    return X_train, X_test, y_train, y_test

# 划分数据集
X_train, X_test, y_train, y_test = custom_stratified_split(X_selected, y_encoded, test_size=0.25)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"训练集标签分布: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
print(f"测试集标签分布: {pd.Series(y_test).value_counts().sort_index().to_dict()}")

# 初始化优化器
rf_optimizer = RandomForestOptimizer()

# 计算类别权重
class_weights = rf_optimizer.get_sample_weights(y_train)

# 设置优化的交叉验证
min_class_size_train = pd.Series(y_train).value_counts().min()
cv_folds_opt = min(5, min_class_size_train)
cv_opt = StratifiedKFold(n_splits=cv_folds_opt, shuffle=True, random_state=42)

# === 阶段1: 粗调参数 ===
print("\n阶段1: 随机森林粗调参数...")

# 粗调参数网格
coarse_param_grid = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True],
    'oob_score': [True]
}

# 粗调搜索
print("执行粗调随机搜索...")
coarse_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    coarse_param_grid,
    n_iter=50,
    cv=cv_opt,
    scoring=balanced_scorer,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

coarse_search.fit(X_train, y_train, sample_weight=class_weights)

print(f"粗调最佳参数: {coarse_search.best_params_}")
print(f"粗调最佳分数: {coarse_search.best_score_:.4f}")

# === 阶段2: 细调参数 ===
print("\n阶段2: 基于粗调结果进行细调...")

# 提取粗调最佳参数
best_coarse = coarse_search.best_params_

# 细调参数网格
fine_param_grid = {}

# n_estimators 细调
if best_coarse['n_estimators'] == 100:
    fine_param_grid['n_estimators'] = [80, 100, 150, 200]
elif best_coarse['n_estimators'] == 800:
    fine_param_grid['n_estimators'] = [600, 800, 1000, 1200]
else:
    base_n = best_coarse['n_estimators']
    fine_param_grid['n_estimators'] = [base_n - 100, base_n, base_n + 100, base_n + 200]

# max_depth 细调
if best_coarse['max_depth'] is None:
    fine_param_grid['max_depth'] = [15, 20, 25, None]
else:
    base_depth = best_coarse['max_depth']
    fine_param_grid['max_depth'] = [max(5, base_depth - 2), base_depth, base_depth + 2, base_depth + 5]

# 其他参数细调
fine_param_grid.update({
    'min_samples_split': [max(2, best_coarse['min_samples_split'] - 1), 
                         best_coarse['min_samples_split'],
                         best_coarse['min_samples_split'] + 2],
    'min_samples_leaf': [max(1, best_coarse['min_samples_leaf'] - 1),
                        best_coarse['min_samples_leaf'],
                        best_coarse['min_samples_leaf'] + 1],
    'max_features': [best_coarse['max_features']],
    'bootstrap': [True],
    'oob_score': [True]
})

print("执行细调网格搜索...")
fine_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    fine_param_grid,
    cv=cv_opt,
    scoring=balanced_scorer,
    n_jobs=-1,
    verbose=1
)

fine_search.fit(X_train, y_train, sample_weight=class_weights)

print(f"细调最佳参数: {fine_search.best_params_}")
print(f"细调最佳分数: {fine_search.best_score_:.4f}")

# === 阶段3: 最终优化 ===
print("\n阶段3: 最终参数微调...")

# 最终微调参数
final_params = fine_search.best_params_.copy()

# 针对类别不平衡的最终优化
validation_param_grid = {
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [None, 50, 100, 200]
}

# 固定其他最优参数
base_estimator_final = RandomForestClassifier(
    n_estimators=final_params['n_estimators'],
    max_depth=final_params['max_depth'],
    min_samples_split=final_params['min_samples_split'],
    min_samples_leaf=final_params['min_samples_leaf'],
    max_features=final_params['max_features'],
    bootstrap=final_params['bootstrap'],
    oob_score=final_params['oob_score'],
    random_state=42,
    n_jobs=-1
)

print("执行最终微调...")
final_search = GridSearchCV(
    base_estimator_final,
    validation_param_grid,
    cv=cv_opt,
    scoring=balanced_scorer,
    n_jobs=-1,
    verbose=1
)

final_search.fit(X_train, y_train, sample_weight=class_weights)

print(f"最终最佳参数: {final_search.best_params_}")
print(f"最终最佳分数: {final_search.best_score_:.4f}")

# 创建最优模型
final_params.update(final_search.best_params_)
optimal_rf_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=-1)
optimal_rf_model.fit(X_train, y_train, sample_weight=class_weights)

# 在测试集上评估
y_pred_rf = optimal_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
balanced_acc = balanced_accuracy_score(y_test, y_pred_rf)
f1_weighted = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\n最优随机森林测试集性能:")
print(f"  准确率: {rf_accuracy:.4f}")
print(f"  平衡准确率: {balanced_acc:.4f}")
print(f"  加权F1分数: {f1_weighted:.4f}")

# 如果模型支持OOB评估
if hasattr(optimal_rf_model, 'oob_score_') and optimal_rf_model.oob_score_ is not None:
    print(f"  袋外(OOB)评估分数: {optimal_rf_model.oob_score_:.4f}")

# =====================
# 10. 验证曲线分析
# =====================
print("\n=== 步骤6: 随机森林验证曲线分析 ===")

# 分析随机森林关键超参数的影响
key_params = ['n_estimators', 'max_depth', 'max_features']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, param in enumerate(key_params):
    print(f"分析参数: {param}")
    
    # 设置参数范围
    if param == 'n_estimators':
        param_range = [50, 100, 200, 300, 500, 800]
    elif param == 'max_depth':
        param_range = [5, 10, 15, 20, 25, None]
    elif param == 'max_features':
        param_range = ['sqrt', 'log2', None, 0.3, 0.5, 0.8]
    
    # 创建基础模型
    base_params = final_params.copy()
    base_params.pop(param, None)
    
    try:
        # 处理None值的特殊情况
        if param == 'max_depth' and None in param_range:
            # 对于max_depth，特别处理None值
            param_range_for_validation = []
            param_labels = []
            for p in param_range:
                if p is None:
                    param_range_for_validation.append(100)  # 用一个大数代替None
                    param_labels.append('None')
                else:
                    param_range_for_validation.append(p)
                    param_labels.append(str(p))
            
            train_scores, validation_scores = validation_curve(
                RandomForestClassifier(**{k: v for k, v in base_params.items() if k != 'max_depth'}, 
                                     random_state=42, n_jobs=-1),
                X_train, y_train,
                param_name='max_depth', 
                param_range=param_range,
                cv=cv_opt, scoring=balanced_scorer,
                n_jobs=-1
            )
        else:
            train_scores, validation_scores = validation_curve(
                RandomForestClassifier(**base_params, random_state=42, n_jobs=-1),
                X_train, y_train,
                param_name=param, 
                param_range=param_range,
                cv=cv_opt, scoring=balanced_scorer,
                n_jobs=-1
            )
            param_labels = [str(p) for p in param_range]
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = np.mean(validation_scores, axis=1)
        validation_std = np.std(validation_scores, axis=1)
        
        x_axis = range(len(param_range))
        
        axes[i].plot(x_axis, train_mean, 'o-', color='blue', label='训练集')
        axes[i].plot(x_axis, validation_mean, 'o-', color='red', label='验证集')
        axes[i].fill_between(x_axis, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        axes[i].fill_between(x_axis, validation_mean - validation_std, validation_mean + validation_std, alpha=0.1, color='red')
        
        # 设置x轴标签
        axes[i].set_xticks(x_axis)
        axes[i].set_xticklabels(param_labels, rotation=45)
        
        # 标记最优值
        optimal_value = final_params.get(param)
        if optimal_value in param_range:
            optimal_idx = param_range.index(optimal_value)
            axes[i].axvline(x=optimal_idx, color='green', linestyle='--', alpha=0.7, 
                           label=f'最优值: {optimal_value}')
        
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('平衡准确率')
        axes[i].set_title(f'{param} 验证曲线')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"参数 {param} 的验证曲线生成失败: {e}")
        axes[i].text(0.5, 0.5, f'{param}\n验证曲线生成失败', 
                    ha='center', va='center', transform=axes[i].transAxes)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_validation_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# =====================
# 11. 详细的NoC性能分析
# =====================
def detailed_noc_performance_analysis(y_true, y_pred, label_encoder):
    """详细的NoC性能分析"""
    
    # 转换为原始标签
    y_true_orig = label_encoder.inverse_transform(y_true)
    y_pred_orig = label_encoder.inverse_transform(y_pred)
    
    print("\n" + "="*80)
    print("                      各贡献者人数详细性能分析")
    print("="*80)
    
    # 总体性能
    overall_accuracy = accuracy_score(y_true_orig, y_pred_orig)
    print(f"\n📊 总体性能指标:")
    print(f"   整体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # 各类别详细分析
    unique_classes = sorted(list(set(y_true_orig)))
    
    print(f"\n📈 各贡献者人数类别详细性能:")
    print("-" * 100)
    print(f"{'贡献者人数':^12} {'样本数':^8} {'正确预测':^10} {'准确率':^10} {'精确率':^10} {'召回率':^10} {'F1分数':^10} {'性能等级':^12}")
    print("-" * 100)
    
    # 计算各类别指标
    precision_scores = precision_score(y_true_orig, y_pred_orig, average=None, labels=unique_classes, zero_division=0)
    recall_scores = recall_score(y_true_orig, y_pred_orig, average=None, labels=unique_classes, zero_division=0)
    f1_scores = f1_score(y_true_orig, y_pred_orig, average=None, labels=unique_classes, zero_division=0)
    
    performance_data = []
    
    for i, noc in enumerate(unique_classes):
        # 统计信息
        true_mask = (y_true_orig == noc)
        total_samples = true_mask.sum()
        correct_predictions = ((y_true_orig == noc) & (y_pred_orig == noc)).sum()
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        precision = precision_scores[i]
        recall = recall_scores[i]
        f1 = f1_scores[i]
        
        # 性能等级评定
        if accuracy >= 0.95:
            grade = "🟢 优秀"
        elif accuracy >= 0.85:
            grade = "🟡 良好" 
        elif accuracy >= 0.70:
            grade = "🟠 一般"
        elif accuracy >= 0.50:
            grade = "🔴 较差"
        else:
            grade = "⚫ 很差"
        
        print(f"{noc:^12}人 {total_samples:^8} {correct_predictions:^10} {accuracy:^10.4f} {precision:^10.4f} {recall:^10.4f} {f1:^10.4f} {grade:^12}")
        
        performance_data.append({
            'noc': noc,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'grade': grade
        })
    
    print("-" * 100)
    
    # 混淆矩阵分析
    print(f"\n🔍 预测错误详细分析:")
    cm = confusion_matrix(y_true_orig, y_pred_orig, labels=unique_classes)
    
    for i, true_noc in enumerate(unique_classes):
        errors = []
        for j, pred_noc in enumerate(unique_classes):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / cm[i].sum() * 100
                errors.append(f"{pred_noc}人({cm[i, j]}次, {error_rate:.1f}%)")
        
        if errors:
            print(f"   {true_noc}人 → 误判为: {', '.join(errors)}")
        else:
            print(f"   {true_noc}人 → 无误判 ✅")
    
    # 特殊关注少数类
    print(f"\n⚠️  少数类别特别关注:")
    minority_classes = [data for data in performance_data if data['total_samples'] < 20]
    
    if minority_classes:
        for data in minority_classes:
            print(f"   {data['noc']}人混合样本 (样本数: {data['total_samples']}):")
            print(f"      准确率: {data['accuracy']:.4f} - {data['grade']}")
            if data['accuracy'] < 0.8:
                print(f"      ⚠️  性能偏低，建议:")
                print(f"         - 增加训练样本")
                print(f"         - 调整类别权重") 
                print(f"         - 使用专门的少数类学习策略")
    else:
        print("   所有类别样本数量充足 ✅")
    
    # 宏观指标
    macro_precision = np.mean(precision_scores)
    macro_recall = np.mean(recall_scores) 
    macro_f1 = np.mean(f1_scores)
    
    weighted_precision = precision_score(y_true_orig, y_pred_orig, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true_orig, y_pred_orig, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true_orig, y_pred_orig, average='weighted', zero_division=0)
    
    print(f"\n📊 宏观性能指标:")
    print(f"   宏平均 - 精确率: {macro_precision:.4f}, 召回率: {macro_recall:.4f}, F1分数: {macro_f1:.4f}")
    print(f"   加权平均 - 精确率: {weighted_precision:.4f}, 召回率: {weighted_recall:.4f}, F1分数: {weighted_f1:.4f}")
    
    return performance_data

# =====================
# 12. 结果分析与可视化
# =====================
print("\n=== 步骤7: 结果分析与可视化 ===")

# 调用详细性能分析
performance_results = detailed_noc_performance_analysis(y_test, y_pred_rf, label_encoder)

# 转换标签用于显示
y_test_orig = label_encoder.inverse_transform(y_test)
y_pred_orig = label_encoder.inverse_transform(y_pred_rf)

# 分类报告
class_names = [f"{x}人" for x in sorted(label_encoder.classes_)]
print(f"\n随机森林详细分类报告:")
print(classification_report(y_test_orig, y_pred_orig, target_names=class_names))

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_orig, y_pred_orig)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names)
plt.title('优化随机森林混淆矩阵')
plt.ylabel('真实NoC')
plt.xlabel('预测NoC')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '混淆矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()

# 特征重要性分析
plt.figure(figsize=(14, 10))
feature_importance = pd.DataFrame({
    '特征': selected_features,
    '重要性': optimal_rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

# 显示所有选择的特征
sns.barplot(data=feature_importance, x='重要性', y='特征')
plt.title(f'随机森林特征重要性排名 (RFECV选择的{len(selected_features)}个特征)')
plt.xlabel('特征重要性')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '特征重要性.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n随机森林特征重要性排名:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['特征']:35} {row['重要性']:.4f}")

# 学习曲线
from sklearn.model_selection import learning_curve

try:
    train_sizes, train_scores, val_scores = learning_curve(
        optimal_rf_model, X_selected, y_encoded, cv=cv_opt, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=balanced_scorer, random_state=42,
        n_jobs=-1
    )

    plt.figure(figsize=(12, 8))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='red', label='训练集')
    plt.plot(train_sizes, val_mean, 'o-', color='green', label='验证集')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')

    plt.xlabel('训练样本数')
    plt.ylabel('平衡准确率')
    plt.title('优化随机森林学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '学习曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"学习曲线生成失败: {e}")

# 优化过程可视化
optimization_scores = [
    coarse_search.best_score_,
    fine_search.best_score_,
    final_search.best_score_
]

optimization_stages = ['粗调', '细调', '最终微调']

plt.figure(figsize=(10, 6))
plt.plot(optimization_stages, optimization_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('优化阶段')
plt.ylabel('交叉验证分数')
plt.title('随机森林参数优化过程')
plt.grid(True, alpha=0.3)

for i, score in enumerate(optimization_scores):
    plt.text(i, score + 0.005, f'{score:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '优化过程.png'), dpi=300, bbox_inches='tight')
plt.close()

# =====================
# 13. SHAP可解释性分析
# =====================
if SHAP_AVAILABLE:
    print("\n=== 步骤8: SHAP可解释性分析 ===")
    
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(optimal_rf_model)
        
        # 计算SHAP值
        shap_sample_size = min(30, len(X_test))
        X_shap = X_test[:shap_sample_size]
        shap_values = explainer.shap_values(X_shap)
        
        # 处理多分类情况
        if isinstance(shap_values, list):
            shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values_mean = np.abs(shap_values)
        
        # SHAP特征重要性
        feature_shap_importance = np.mean(shap_values_mean, axis=0)
        shap_importance_df = pd.DataFrame({
            '特征': selected_features,
            'SHAP重要性': feature_shap_importance
        }).sort_values('SHAP重要性', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=shap_importance_df, x='SHAP重要性', y='特征')
        plt.title('SHAP特征重要性排名')
        plt.xlabel('平均SHAP重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'SHAP重要性.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("SHAP特征重要性排名:")
        for idx, row in shap_importance_df.iterrows():
            print(f"  {row['特征']:35} {row['SHAP重要性']:.4f}")
        
        # SHAP摘要图
        try:
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_plot, X_shap, feature_names=selected_features, 
                            plot_type="bar", show=False)
            plt.title("SHAP摘要图")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'SHAP摘要图.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"SHAP摘要图生成失败: {e}")
            
    except Exception as e:
        print(f"SHAP分析失败: {e}")

# =====================
# 14. 模型预测与保存
# =====================
print("\n=== 步骤9: 模型预测与保存 ===")

# 对所有样本进行预测
y_pred_all = optimal_rf_model.predict(X_selected)
y_pred_all_orig = label_encoder.inverse_transform(y_pred_all)

# 添加预测结果到特征数据框
df_features['预测贡献者人数'] = y_pred_all_orig

# 计算整体准确率
overall_accuracy = (df_features['预测贡献者人数'] == df_features['贡献者人数']).mean()
print(f"整体预测准确率: {overall_accuracy:.4f}")

# 各NoC类别的准确率
noc_accuracy = df_features.groupby('贡献者人数').apply(
    lambda x: (x['预测贡献者人数'] == x['贡献者人数']).mean()
).reset_index(name='准确率')

print("\n各NoC类别预测准确率:")
for _, row in noc_accuracy.iterrows():
    print(f"  {int(row['贡献者人数'])}人: {row['准确率']:.4f}")

# 可视化各类别准确率
plt.figure(figsize=(10, 6))
sns.barplot(data=noc_accuracy, x='贡献者人数', y='准确率')
plt.ylim(0, 1.1)
plt.xlabel('真实NoC')
plt.ylabel('预测准确率')
plt.title('各NoC类别预测准确率')

for i, row in noc_accuracy.iterrows():
    plt.text(i, row['准确率'] + 0.03, f"{row['准确率']:.3f}", 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '各类别准确率.png'), dpi=300, bbox_inches='tight')
plt.close()

# 保存结果
df_features.to_csv(os.path.join(DATA_DIR, 'NoC识别结果_RFECV_RF优化版.csv'), 
                   index=False, encoding='utf-8-sig')

# 保存模型
import joblib
model_filename = os.path.join(DATA_DIR, 'noc_optimized_random_forest_model.pkl')
joblib.dump({
    'model': optimal_rf_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'selected_features': selected_features,
    'selected_indices': selected_indices,
    'rfecv': rfecv,
    'optimization_history': {
        'coarse_best': coarse_search.best_params_,
        'fine_best': fine_search.best_params_,
        'final_best': final_search.best_params_,
        'scores': optimization_scores
    }
}, model_filename)

print(f"优化模型已保存至: {model_filename}")

# 保存详细摘要
summary = {
    '模型信息': {
        '模型类型': 'OptimizedRandomForestClassifier',
        '特征选择方法': 'RFECV',
        '优化阶段': ['粗调', '细调', '最终微调'],
        '测试集准确率': float(rf_accuracy),
        '平衡准确率': float(balanced_acc),
        '加权F1分数': float(f1_weighted),
        '整体准确率': float(overall_accuracy)
    },
    '特征选择结果': {
        '原始特征数': len(feature_cols),
        '最终特征数': len(selected_features),
        'RFECV最优特征数': int(rfecv.n_features_),
        'RFECV最佳分数': float(best_score),
        '选择的特征': selected_features
    },
    '优化过程': {
        '粗调最佳参数': coarse_search.best_params_,
        '粗调最佳分数': float(coarse_search.best_score_),
        '细调最佳参数': fine_search.best_params_,
        '细调最佳分数': float(fine_search.best_score_),
        '最终最佳参数': final_search.best_params_,
        '最终最佳分数': float(final_search.best_score_)
    },
    '最终模型参数': final_params,
    '数据信息': {
        '总样本数': len(df_features),
        'NoC分布': df_features['贡献者人数'].value_counts().sort_index().to_dict(),
        '训练集大小': len(X_train),
        '测试集大小': len(X_test)
    },
    '各类别准确率': {
        int(row['贡献者人数']): float(row['准确率']) 
        for _, row in noc_accuracy.iterrows()
    },
    '特征重要性前10': [
        {
            '特征': row['特征'],
            '重要性': float(row['重要性'])
        }
        for _, row in feature_importance.head(10).iterrows()
    ],
    '时间戳': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

# 如果有OOB分数，添加到摘要中
if hasattr(optimal_rf_model, 'oob_score_') and optimal_rf_model.oob_score_ is not None:
    summary['模型信息']['OOB评估分数'] = float(optimal_rf_model.oob_score_)

with open(os.path.join(DATA_DIR, 'NoC分析摘要_RFECV_RF优化版.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# =====================
# 15. 最终报告
# =====================
print("\n" + "="*80)
print("         法医混合STR图谱NoC识别 - RFECV+随机森林优化版最终报告")
print("="*80)

print(f"\n📊 数据概况:")
print(f"   • 总样本数: {len(df_features)}")
print(f"   • NoC分布: {dict(df_features['贡献者人数'].value_counts().sort_index())}")
print(f"   • 原始特征数: {len(feature_cols)}")
print(f"   • RFECV选择特征数: {len(selected_features)}")

print(f"\n🔧 技术创新:")
print(f"   • 使用RFECV递归特征消除进行特征选择")
print(f"   • 三阶段随机森林参数优化（粗调→细调→微调）")
print(f"   • 平衡准确率作为优化目标，处理类别不平衡")
print(f"   • 验证曲线分析关键超参数影响")
print(f"   • 利用袋外(OOB)评估提高模型可靠性")

print(f"\n🎯 RFECV特征选择结果:")
print(f"   • 最优特征数: {rfecv.n_features_}")
print(f"   • 最佳交叉验证分数: {best_score:.4f}")
print(f"   • 特征减少率: {(1 - len(selected_features)/len(feature_cols)):.1%}")

print(f"\n🚀 随机森林优化历程:")
print(f"   • 粗调阶段: {coarse_search.best_score_:.4f}")
print(f"   • 细调阶段: {fine_search.best_score_:.4f}")
print(f"   • 最终微调阶段: {final_search.best_score_:.4f}")
print(f"   • 最终模型参数: {final_params}")
print(f"   • 测试集准确率: {rf_accuracy:.4f}")
print(f"   • 平衡准确率: {balanced_acc:.4f}")
print(f"   • 加权F1分数: {f1_weighted:.4f}")
print(f"   • 整体预测准确率: {overall_accuracy:.4f}")

if hasattr(optimal_rf_model, 'oob_score_') and optimal_rf_model.oob_score_ is not None:
    print(f"   • 袋外评估分数: {optimal_rf_model.oob_score_:.4f}")

print(f"\n📈 各类别性能表现:")
performance_summary = []
for _, row in noc_accuracy.iterrows():
    noc = int(row['贡献者人数'])
    acc = row['准确率']
    sample_count = len(df_features[df_features['贡献者人数'] == noc])
    
    if acc >= 0.9:
        performance = "🟢 优秀"
    elif acc >= 0.8:
        performance = "🟡 良好"
    elif acc >= 0.6:
        performance = "🟠 一般"
    else:
        performance = "🔴 需改进"
    
    print(f"   • {noc}人混合样本: {acc:.4f} ({acc*100:.1f}%) - {performance} ({sample_count}个样本)")
    performance_summary.append((noc, acc, sample_count))

print(f"\n🔍 前5位最重要特征:")
top_5_features = feature_importance.head(5)
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    feature_cn = FEATURE_NAME_MAPPING.get(row['特征'], row['特征'])
    print(f"   {i}. {feature_cn:<25} (重要性: {row['重要性']:.4f})")

print(f"\n📋 模型特点说明:")
print(f"   • 基于 {len(selected_features)} 个精选生物特征进行预测")
print(f"   • 特征涵盖: 图谱统计特性、峰高分布特征、位点平衡性、")
print(f"     信息熵指标、DNA降解标志等多个维度")
print(f"   • 采用随机森林算法，利用Bootstrap聚合和特征随机采样")
print(f"   • 针对少数类样本进行特殊优化，使用类别权重平衡")
print(f"   • 整体预测准确率达到 {overall_accuracy:.1%}，具有较好的实用价值")

# 数据质量评估
print(f"\n📊 数据质量评估:")
noc_distribution = df_features['贡献者人数'].value_counts().sort_index()
max_samples = noc_distribution.max()
min_samples = noc_distribution.min()
imbalance_ratio = max_samples / min_samples

print(f"   • 样本不平衡程度: {imbalance_ratio:.1f}:1")
if imbalance_ratio > 10:
    print(f"   • ⚠️  数据严重不平衡，已采用加权采样策略")
elif imbalance_ratio > 5:
    print(f"   • ⚠️  数据中度不平衡，已采用权重平衡策略")
else:
    print(f"   • ✅ 数据平衡性良好")

# 随机森林特有优势
print(f"\n🌲 随机森林算法优势:")
print(f"   • Bootstrap聚合减少过拟合风险")
print(f"   • 特征随机采样提高泛化能力")
print(f"   • 对噪声和异常值具有较强鲁棒性")
print(f"   • 提供特征重要性排名，增强可解释性")
if hasattr(optimal_rf_model, 'oob_score_') and optimal_rf_model.oob_score_ is not None:
    print(f"   • 袋外评估提供无偏性能估计")

# 改进建议
print(f"\n💡 改进建议:")
low_performance_classes = [noc for noc, acc, _ in performance_summary if acc < 0.8]
if low_performance_classes:
    print(f"   • 针对 {', '.join(map(str, low_performance_classes))} 人混合样本:")
    print(f"     - 增加训练样本数量")
    print(f"     - 调整随机森林的class_weight参数")
    print(f"     - 考虑使用cost-sensitive learning")
    print(f"     - 尝试集成学习方法")
else:
    print(f"   • ✅ 所有类别性能均达到良好水平")

print(f"\n💾 输出文件:")
print(f"   • 特征数据文件: NoC识别结果_RFECV_RF优化版.csv")
print(f"   • 模型性能摘要: NoC分析摘要_RFECV_RF优化版.json")
print(f"   • 训练好的模型: noc_optimized_random_forest_model.pkl")
print(f"   • 图表输出目录: {PLOTS_DIR}")

if SHAP_AVAILABLE:
    print(f"   • SHAP可解释性分析图表已生成，提升模型透明度")

print(f"\n⏰ 分析完成时间: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
print("\n✅ 法医混合STR图谱NoC智能识别分析完成！")
print("="*80)