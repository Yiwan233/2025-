# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别 (修复版)

版本: V4.1 - Fixed RFECV Issue
日期: 2025-06-03
描述: 修复RFECV sample_weight参数问题，使用替代方案进行特征选择
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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
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
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=== 法医混合STR图谱NoC智能识别系统 (修复版) ===")
print("基于附件一数据的完整实现")

# =====================
# 1. 文件路径与基础设置
# =====================
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'noc_fixed_plots')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# 关键参数设置
HEIGHT_THRESHOLD = 50
SATURATION_THRESHOLD = 30000
CTA_THRESHOLD = 0.5
PHR_IMBALANCE_THRESHOLD = 0.6

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
# 3. 数据加载与预处理
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
# 4. 简化的峰处理
# =====================
print("\n=== 步骤2: 峰处理与信号表征 ===")

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
# 5. 综合特征工程
# =====================
print("\n=== 步骤3: 综合特征工程 ===")

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
    for sample_file in df['Sample File'].unique():
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
noc_map = df.groupby('Sample File')['NoC_True'].first().to_dict()
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
# 6. 修复后的特征选择
# =====================
print("\n=== 步骤4: 改进的特征选择方法 ===")

# 准备特征和标签
feature_cols = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
X = df_features[feature_cols].fillna(0)
y = df_features['NoC_True']

print(f"原始特征数: {len(feature_cols)}")
print(f"样本数: {len(X)}")
print(f"NoC分布: {y.value_counts().sort_index().to_dict()}")

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算样本权重（用于后续模型训练）
sample_weights = compute_sample_weight('balanced', y_encoded)

# 对4人和5人样本额外加权
enhanced_weights = sample_weights.copy()
for i, label in enumerate(y_encoded):
    if label_encoder.inverse_transform([label])[0] == 4:
        enhanced_weights[i] *= 2.5  # 4人样本权重×2.5
    elif label_encoder.inverse_transform([label])[0] == 5:
        enhanced_weights[i] *= 4.0  # 5人样本权重×4.0

print(f"样本权重范围: {enhanced_weights.min():.3f} - {enhanced_weights.max():.3f}")

# ===== 方法1: 基于单变量统计的特征选择 =====
print("\n使用基于单变量统计的特征选择...")

# 使用f_classif进行单变量特征选择
selector_univariate = SelectKBest(score_func=f_classif, k=20)  # 选择前20个特征
X_selected_univariate = selector_univariate.fit_transform(X_scaled, y_encoded)

selected_features_univariate = [feature_cols[i] for i in range(len(feature_cols)) 
                               if selector_univariate.get_support()[i]]

print(f"单变量选择的特征数: {len(selected_features_univariate)}")

# ===== 方法2: 基于LassoCV的特征选择 =====
print("\n使用LassoCV进行特征选择...")

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso_cv.fit(X_scaled, y_encoded)

# 选择非零系数的特征
selector_lasso = SelectFromModel(lasso_cv, prefit=True)
X_selected_lasso = selector_lasso.transform(X_scaled)
selected_features_lasso = [feature_cols[i] for i in range(len(feature_cols)) 
                          if selector_lasso.get_support()[i]]

print(f"LassoCV选择的特征数: {len(selected_features_lasso)}")

# ===== 方法3: 基于模型重要性的特征选择 =====
print("\n使用基于模型重要性的特征选择...")

# 创建一个支持样本权重的随机森林进行特征选择
rf_selector = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)

# 训练模型并获取特征重要性
rf_selector.fit(X_scaled, y_encoded, sample_weight=enhanced_weights)

# 基于重要性选择特征
importance_threshold = np.percentile(rf_selector.feature_importances_, 70)  # 选择前30%的特征
important_features_mask = rf_selector.feature_importances_ >= importance_threshold
selected_features_importance = [feature_cols[i] for i in range(len(feature_cols)) 
                               if important_features_mask[i]]

print(f"基于重要性选择的特征数: {len(selected_features_importance)}")

# ===== 综合特征选择策略 =====
print("\n综合特征选择策略...")

# 取三种方法的交集作为最终特征
selected_features_final = list(set(selected_features_univariate) & 
                              set(selected_features_lasso) & 
                              set(selected_features_importance))

# 如果交集太少，使用并集的前15个最重要特征
if len(selected_features_final) < 10:
    print("交集特征数量较少，使用重要性排序的并集策略...")
    
    # 合并所有选择的特征
    all_selected = list(set(selected_features_univariate + selected_features_lasso + selected_features_importance))
    
    # 按随机森林重要性排序
    feature_importance_dict = {feature_cols[i]: rf_selector.feature_importances_[i] 
                              for i in range(len(feature_cols))}
    
    all_selected_sorted = sorted(all_selected, key=lambda x: feature_importance_dict[x], reverse=True)
    selected_features_final = all_selected_sorted[:15]  # 选择前15个

print(f"最终选择的特征数: {len(selected_features_final)}")

# 显示选择的特征
print("\n最终选择的特征:")
for i, feature in enumerate(selected_features_final, 1):
    importance = feature_importance_dict.get(feature, 0)
    print(f"  {i:2d}. {feature:35} (重要性: {importance:.4f})")

# 使用最终选择的特征
feature_indices = [feature_cols.index(f) for f in selected_features_final]
X_final = X_scaled[:, feature_indices]
X_final = pd.DataFrame(X_final, columns=selected_features_final)

# =====================
# 7. 模型训练与评估
# =====================
print("\n=== 步骤5: 模型训练与验证 ===")

# 自定义分层划分，确保少数类在训练集和测试集中都有足够样本
def custom_train_test_split(X, y, test_size=0.3, random_state=42):
    """自定义训练测试集划分，确保少数类样本分布"""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    
    for cls, count in zip(unique