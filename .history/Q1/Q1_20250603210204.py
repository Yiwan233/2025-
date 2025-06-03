# -*- coding: utf-8 -*-
"""
法医混合STR图谱贡献者人数（NoC）智能识别模型
基于文档思路的完整实现 - Gradient Boosting版本

参考文档：法医混合STR图谱贡献者人数（NoC）智能识别模型研究
核心思路：
1. 精细化数据预处理与信号表征（简化Stutter处理）
2. 基于文档V5方案的综合特征体系构建
3. LassoCV特征选择
4. Gradient Boosting模型训练与集成
5. SHAP可解释性分析
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

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# 可解释性分析
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP不可用，将跳过可解释性分析")
    SHAP_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=== 法医混合STR图谱NoC智能识别模型 ===")
print("基于Gradient Boosting的完整实现")

# =====================
# 1. 文件路径与基础设置
# =====================
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'noc_analysis_plots')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# 关键参数设置（基于文档）
HEIGHT_THRESHOLD = 50  # 分析考虑阈值
SATURATION_THRESHOLD = 30000  # 饱和阈值
CTA_THRESHOLD = 0.5  # 真实等位基因置信度阈值
PHR_IMBALANCE_THRESHOLD = 0.6  # 严重不平衡阈值

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

print(f"NoC分布: {df['NoC_True'].value_counts().sort_index().to_dict()}")
print(f"样本数: {df['Sample File'].nunique()}")

# =====================
# 4. 简化的峰处理与CTA评估
# =====================
print("\n=== 步骤2: 峰处理与信号表征 ===")

def process_peaks_with_cta(sample_data):
    """
    简化的峰处理，包含基础CTA评估
    基于文档3.1和3.2节的思路，但简化Stutter模型
    """
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
                # 饱和校正
                corrected_height = min(original_height, SATURATION_THRESHOLD)
                
                # 分析考虑阈值过滤
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
        # 基于峰高相对大小进行简单的Stutter可能性评估
        peaks.sort(key=lambda x: x['height'], reverse=True)
        
        for peak in peaks:
            # 简化的CTA计算：基于峰高在位点内的相对位置
            height_rank = peaks.index(peak) + 1
            total_peaks = len(peaks)
            
            # 简单的启发式CTA评估
            if height_rank == 1:  # 最高峰
                cta = 0.95
            elif height_rank == 2 and total_peaks >= 2:  # 第二高峰
                height_ratio = peak['height'] / peaks[0]['height']
                cta = 0.8 if height_ratio > 0.3 else 0.6
            else:  # 其他峰
                height_ratio = peak['height'] / peaks[0]['height']
                cta = max(0.1, min(0.8, height_ratio))
            
            # 应用CTA阈值过滤
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
for sample_file, group in df.groupby('Sample File'):
    sample_peaks = process_peaks_with_cta(group)
    all_processed_peaks.append(sample_peaks)

df_peaks = pd.concat(all_processed_peaks, ignore_index=True) if all_processed_peaks else pd.DataFrame()
print(f"处理后的峰数据形状: {df_peaks.shape}")

# =====================
# 5. 综合特征工程（基于文档V5方案）
# =====================
print("\n=== 步骤3: 综合特征工程 ===")

def extract_comprehensive_features_v5(sample_file, sample_peaks):
    """
    基于文档第4节的综合特征体系构建
    实现A、B、C、D类特征
    """
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
    
    # 预期常染色体位点数（假设值，实际应从MARKER_PARAMS获取）
    expected_autosomal_count = 23  # 可根据实际试剂盒调整
    
    # ===============================
    # A类：谱图层面基础统计特征
    # ===============================
    
    # A.1 MACP - 样本最大等位基因数
    features['mac_profile'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    
    # A.2 TDA - 样本总特异等位基因数
    features['total_distinct_alleles'] = sample_peaks['Allele'].nunique()
    
    # A.3 AAP - 每位点平均等位基因数
    if expected_autosomal_count > 0:
        # 包含缺失位点（计为0）
        all_locus_counts = np.zeros(expected_autosomal_count)
        all_locus_counts[:len(alleles_per_locus)] = alleles_per_locus.values
        features['avg_alleles_per_locus'] = np.mean(all_locus_counts)
        features['std_alleles_per_locus'] = np.std(all_locus_counts)
    else:
        features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    
    # A.5 MGTN系列 - 等位基因数≥N的位点数
    for N in [2, 3, 4, 5, 6]:
        features[f'loci_gt{N}_alleles'] = (alleles_per_locus >= N).sum()
    
    # A.6 等位基因计数的熵
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['allele_count_dist_entropy'] = calculate_entropy(counts.values)
    else:
        features['allele_count_dist_entropy'] = 0
    
    # ===============================
    # B类：峰高、平衡性及随机效应特征
    # ===============================
    
    if total_peaks > 0:
        # B.1 基础峰高统计
        features['avg_peak_height'] = np.mean(all_heights)
        features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
        features['min_peak_height'] = np.min(all_heights)
        features['max_peak_height'] = np.max(all_heights)
        
        # B.2 峰高比(PHR)相关统计
        phr_values = []
        for marker, marker_group in locus_groups:
            if len(marker_group) == 2:  # 恰好两个等位基因的位点
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
        
        # B.3 峰高分布统计矩
        if total_peaks > 2:
            features['skewness_peak_height'] = stats.skew(all_heights)
            features['kurtosis_peak_height'] = stats.kurtosis(all_heights, fisher=False)
        else:
            features['skewness_peak_height'] = 0
            features['kurtosis_peak_height'] = 0
        
        # B.3+ 峰高多峰性
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
        
        # B.4 饱和效应
        saturated_peaks = (sample_peaks['Original_Height'] >= SATURATION_THRESHOLD).sum()
        features['num_saturated_peaks'] = saturated_peaks
        features['ratio_saturated_peaks'] = saturated_peaks / total_peaks
    else:
        # 空值填充
        for key in ['avg_peak_height', 'std_peak_height', 'min_peak_height', 'max_peak_height',
                   'avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr',
                   'num_severe_imbalance_loci', 'ratio_severe_imbalance_loci',
                   'skewness_peak_height', 'kurtosis_peak_height', 'modality_peak_height',
                   'num_saturated_peaks', 'ratio_saturated_peaks']:
            features[key] = 0
    
    # ===============================
    # C类：信息论及谱图复杂度特征
    # ===============================
    
    # C.1 位点间平衡性的香农熵
    if len(locus_heights) > 0:
        total_height = locus_heights.sum()
        if total_height > 0:
            locus_probs = locus_heights / total_height
            features['inter_locus_balance_entropy'] = calculate_entropy(locus_probs.values)
        else:
            features['inter_locus_balance_entropy'] = 0
    else:
        features['inter_locus_balance_entropy'] = 0
    
    # C.2 平均位点等位基因分布熵
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
    
    # C.3 样本整体峰高分布熵
    if total_peaks > 0:
        log_heights = np.log(all_heights + 1)
        hist, _ = np.histogram(log_heights, bins=min(15, total_peaks))
        hist_probs = hist / hist.sum()
        hist_probs = hist_probs[hist_probs > 0]
        features['peak_height_entropy'] = calculate_entropy(hist_probs)
    else:
        features['peak_height_entropy'] = 0
    
    # C.4 图谱完整性指标
    effective_loci_count = len(locus_groups)
    features['num_loci_with_effective_alleles'] = effective_loci_count
    features['num_loci_no_effective_alleles'] = max(0, expected_autosomal_count - effective_loci_count)
    
    # ===============================
    # D类：DNA降解与信息丢失特征
    # ===============================
    
    if total_peaks > 1:
        # D.1 峰高与片段大小的相关性
        if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1]
        else:
            features['height_size_correlation'] = 0
        
        # D.2 峰高与片段大小的线性回归斜率
        features['height_size_slope'] = calculate_ols_slope(all_sizes, all_heights)
        
        # D.3 加权回归斜率（权重为峰高）
        try:
            # 简化版加权回归
            weights = all_heights / all_heights.sum()
            weighted_correlation = np.average(all_sizes, weights=weights)
            features['weighted_height_size_slope'] = calculate_ols_slope(all_sizes, all_heights)
        except:
            features['weighted_height_size_slope'] = 0
        
        # D.4 PHR随片段大小变化的斜率
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
    
    # D.5 位点丢失评分（简化版）
    dropout_score = features['num_loci_no_effective_alleles'] / expected_autosomal_count if expected_autosomal_count > 0 else 0
    features['locus_dropout_score_weighted_by_size'] = dropout_score
    
    # D.6 RFU每碱基对衰减指数（简化版）
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
    
    # D.7 小片段与大片段信息完整度比率（简化版）
    # 假设片段大小<200bp为小片段，>=200bp为大片段
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
    
    # 估算总的小/大片段位点数
    total_small_expected = expected_autosomal_count // 2  # 假设一半是小片段
    total_large_expected = expected_autosomal_count - total_small_expected
    
    small_completeness = small_fragment_effective / total_small_expected if total_small_expected > 0 else 0
    large_completeness = large_fragment_effective / total_large_expected if total_large_expected > 0 else 0
    
    if large_completeness > 0:
        features['info_completeness_ratio_small_large'] = small_completeness / large_completeness
    else:
        features['info_completeness_ratio_small_large'] = small_completeness / 0.001  # 避免除零
    
    return features

# 提取所有样本的特征
print("开始特征提取...")
start_time = time()

all_features = []
for sample_file, group in df_peaks.groupby('Sample File'):
    features = extract_comprehensive_features_v5(sample_file, group)
    all_features.append(features)

df_features = pd.DataFrame(all_features)

# 合并NoC标签
noc_map = df.groupby('Sample File')['NoC_True'].first().to_dict()
df_features['NoC_True'] = df_features['Sample File'].map(noc_map)
df_features = df_features.dropna(subset=['NoC_True'])

print(f"特征提取完成，耗时: {time() - start_time:.2f}秒")
print(f"特征数据形状: {df_features.shape}")
print(f"特征数量: {len([col for col in df_features.columns if col not in ['Sample File', 'NoC_True']])}")

# =====================
# 6. 特征选择（LassoCV）
# =====================
print("\n=== 步骤4: 特征选择 ===")

# 准备数据
feature_cols = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
X = df_features[feature_cols].fillna(0)
y = df_features['NoC_True']

print(f"原始特征数: {len(feature_cols)}")
print(f"样本数: {len(X)}")
print(f"NoC分布: {y.value_counts().sort_index().to_dict()}")

# 标签编码（为了兼容性）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# LassoCV特征选择
print("使用LassoCV进行特征选择...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000)
lasso_cv.fit(X_scaled, y_encoded)

# 选择非零系数的特征
selector = SelectFromModel(lasso_cv, prefit=True)
X_selected = selector.transform(X_scaled)
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]

print(f"LassoCV选择的特征数: {len(selected_features)}")
print("选择的特征:")
for i, feature in enumerate(selected_features, 1):
    coef = lasso_cv.coef_[feature_cols.index(feature)]
    print(f"  {i:2d}. {feature:35} (系数: {coef:8.4f})")

# 如果选择的特征太少，保留重要特征
if len(selected_features) < 5:
    print("警告: LassoCV选择的特征太少，使用基于重要性的备选方案...")
    # 基于绝对系数值选择前15个特征
    feature_importance = [(i, abs(coef)) for i, coef in enumerate(lasso_cv.coef_)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in feature_importance[:15]]
    selected_features = [feature_cols[i] for i in selected_indices]
    X_selected = X_scaled[:, selected_indices]
    print(f"备选方案选择了 {len(selected_features)} 个特征")

# =====================
# 7. 模型训练与验证
# =====================
print("\n=== 步骤5: 模型训练与验证 ===")

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
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
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
    'Gradient Boosting': (best_gb_model, gb_accuracy, y