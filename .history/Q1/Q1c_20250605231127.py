# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别 (中文优化版)

版本: V5.0 - Chinese Features + RFECV + Optimized GradientBoosting
日期: 2025-06-03
描述: 中文特征名称 + RFECV特征选择 + 深度调参的梯度提升机
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

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

print("=== 法医混合STR图谱NoC智能识别系统 (中文优化版) ===")
print("基于RFECV特征选择 + 深度调参梯度提升机")

# =====================
# 1. 文件路径与基础设置
# =====================
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'noc_chinese_plots')
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
    'info_completeness_ratio_small_large': '小大片段信息完整度比率'
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
# 5. 峰处理与CTA评估
# =====================
print("\n=== 步骤2: 峰处理与信号表征 ===")

def process_peaks_with_cta(sample_data):
    """峰处理，包含CTA评估"""
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
            
        # CTA评估
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

# 处理所有样本
all_processed_peaks = []
unique_samples = df['Sample File'].nunique()
processed_count = 0

print(f"开始处理 {unique_samples} 个样本的峰数据...")

for sample_file, group in df.groupby('Sample File'):
    processed_count += 1
    if processed_count % 50 == 0 or processed_count == unique_samples:
        print(f"处理进度: {processed_count}/{unique_samples} ({processed_count/unique_samples*100:.1f}%)")
    
    sample_peaks = process_peaks_with_cta(group)
    if not sample_peaks.empty:
        all_processed_peaks.append(sample_peaks)

df_peaks = pd.concat(all_processed_peaks, ignore_index=True) if all_processed_peaks else pd.DataFrame()
print(f"处理后的峰数据形状: {df_peaks.shape}")

# =====================
# 6. 综合特征工程（返回中文名称）
# =====================
print("\n=== 步骤3: 综合特征工程 ===")

def extract_comprehensive_features_chinese(sample_file, sample_peaks):
    """提取综合特征，返回中文特征名"""
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
    
    expected_autosomal_count = 23
    
    # A类：图谱层面基础统计特征
    features['样本最大等位基因数'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['样本总特异等位基因数'] = sample_peaks['Allele'].nunique()
    
    if expected_autosomal_count > 0:
        all_locus_counts = np.zeros(expected_autosomal_count)
        all_locus_counts[:len(alleles_per_locus)] = alleles_per_locus.values
        features['每位点平均等位基因数'] = np.mean(all_locus_counts)
        features['每位点等位基因数标准差'] = np.std(all_locus_counts)
    else:
        features['每位点平均等位基因数'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['每位点等位基因数标准差'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    
    # MGTN系列
    for N in [2, 3, 4, 5, 6]:
        features[f'等位基因数大于{N}的位点数'] = (alleles_per_locus >= N).sum()
    
    # 等位基因计数的熵
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['等位基因计数分布熵'] = calculate_entropy(counts.values)
    else:
        features['等位基因计数分布熵'] = 0
    
    # B类：峰高、平衡性及随机效应特征
    if total_peaks > 0:
        features['平均峰高'] = np.mean(all_heights)
        features['峰高标准差'] = np.std(all_heights) if total_peaks > 1 else 0
        features['最小峰高'] = np.min(all_heights)
        features['最大峰高'] = np.max(all_heights)
        
        # 峰高比(PHR)相关统计
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
            for key in ['平均峰高比', '峰高比标准差', '最小峰高比', '峰高比中位数', '可计算峰高比的位点数', 
                       '严重失衡位点数', '严重失衡位点比例']:
                features[key] = 0
        
        # 峰高分布统计矩
        if total_peaks > 2:
            features['峰高分布偏度'] = stats.skew(all_heights)
            features['峰高分布峭度'] = stats.kurtosis(all_heights, fisher=False)
        else:
            features['峰高分布偏度'] = 0
            features['峰高分布峭度'] = 0
        
        # 峰高多峰性
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
        
        # 饱和效应
        saturated_peaks = (sample_peaks['Original_Height'] >= SATURATION_THRESHOLD).sum()
        features['饱和峰数量'] = saturated_peaks
        features['饱和峰比例'] = saturated_peaks / total_peaks
    else:
        for key in ['平均峰高', '峰高标准差', '最小峰高', '最大峰高',
                   '平均峰高比', '峰高比标准差', '最小峰高比', '峰高比中位数', '可计算峰高比的位点数',
                   '严重失衡位点数', '严重失衡位点比例',
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
    features['无有效等位基因位点数'] = max(0, expected_autosomal_count - effective_loci_count)
    
    # D类：DNA降解与信息丢失特征
    if total_peaks > 1:
        if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['峰高片段大小相关性'] = np.corrcoef(all_heights, all_sizes)[0, 1]
        else:
            features['峰高片段大小相关性'] = 0
        
        features['峰高片段大小回归斜率'] = calculate_ols_slope(all_sizes, all_heights)
        
        try:
            weights = all_heights / all_heights.sum()
            features['加权峰高片段大小斜率'] = calculate_ols_slope(all_sizes, all_heights)
        except:
            features['加权峰高片段大小斜率'] = 0
        
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
    
    # 其他降解相关特征
    dropout_score = features['无有效等位基因位点数'] / expected_autosomal_count if expected_autosomal_count > 0 else 0
    features['片段大小加权位点丢失评分'] = dropout_score
    
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
        features['小大片段信息完整度比率'] = small_completeness / large_completeness
    else:
        features['小大片段信息完整度比率'] = small_completeness / 0.001
    
    return features

# 提取所有样本的特征
print("开始特征提取...")
start_time = time()

all_features = []
unique_samples = df_peaks['Sample File'].nunique() if not df_peaks.empty else 0
processed_count = 0

if df_peaks.empty:
    print("警告: 处理后的峰数据为空，使用默认特征")
    for sample_file in df['Sample File'].unique():
        features = {'样本文件': sample_file, '样本最大等位基因数': 0}
        all_features.append(features)
else:
    for sample_file, group in df_peaks.groupby('Sample File'):
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == unique_samples:
            print(f"特征提取进度: {processed_count}/{unique_samples} ({processed_count/unique_samples*100:.1f}%)")
        
        features = extract_comprehensive_features_chinese(sample_file, group)
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

print(f"特征提取完成，耗时: {time() - start_time:.2f}秒")
print(f"特征数据形状: {df_features.shape}")

# =====================
# 7. RFECV特征选择（修复版）
# =====================
print("\n=== 步骤4: RFECV特征选择 ===")

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

# 创建用于RFECV的基础估计器（不使用sample_weight）
base_estimator = GradientBoostingClassifier(
    n_estimators=50,  # 减少估计器数量以加快RFECV速度
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# 使用RFECV进行特征选择（不传递sample_weight）
print("使用RFECV进行特征选择...")
rfecv = RFECV(
    estimator=base_estimator,
    step=1,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # 减少CV折数
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 执行RFECV（不传递sample_weight参数）
rfecv.fit(X_scaled, y_encoded)

# 获取选择的特征
selected_features_mask = rfecv.support_
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_features_mask[i]]
X_selected = X_scaled[:, selected_features_mask]

print(f"RFECV选择的特征数: {len(selected_features)}")
print(f"最优特征数: {rfecv.n_features_}")

# 显示选择的特征
print("\nRFECV选择的特征:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feature}")

# =====================
# 8. 深度调参的梯度提升机
# =====================
print("\n=== 步骤5: 深度调参梯度提升机 ===")

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
            # 对于样本数极少的类别，至少保留1个在测试集
            n_test = 1
            n_train = n_cls - 1
        else:
            n_