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
# 2. 扩展的中文特征名称映射 (90+个特征)
# =====================
FEATURE_NAME_MAPPING = {
    # A类：图谱层面基础统计特征 (20个)
    '样本最大等位基因数': '样本最大等位基因数',
    '样本总特异等位基因数': '样本总特异等位基因数',
    '每位点平均等位基因数': '每位点平均等位基因数',
    '每位点等位基因数标准差': '每位点等位基因数标准差',
    '每位点等位基因数中位数': '每位点等位基因数中位数',
    '每位点等位基因数最大值': '每位点等位基因数最大值',
    '每位点等位基因数最小值': '每位点等位基因数最小值',
    '等位基因数变异系数': '等位基因数变异系数',
    '等位基因计数分布熵': '等位基因计数分布熵',
    '等位基因计数分布偏度': '等位基因计数分布偏度',
    '等位基因计数分布峭度': '等位基因计数分布峭度',
    
    # MGTN系列扩展
    '等位基因数大于等于1的位点数': '≥1等位基因位点数',
    '等位基因数大于等于2的位点数': '≥2等位基因位点数', 
    '等位基因数大于等于3的位点数': '≥3等位基因位点数',
    '等位基因数大于等于4的位点数': '≥4等位基因位点数',
    '等位基因数大于等于5的位点数': '≥5等位基因位点数',
    '等位基因数大于等于6的位点数': '≥6等位基因位点数',
    '等位基因数大于等于7的位点数': '≥7等位基因位点数',
    '等位基因数大于等于8的位点数': '≥8等位基因位点数',
    '等位基因数大于等于1的位点比例': '≥1等位基因位点比例',
    '等位基因数大于等于2的位点比例': '≥2等位基因位点比例',
    '等位基因数大于等于3的位点比例': '≥3等位基因位点比例',
    '等位基因数大于等于4的位点比例': '≥4等位基因位点比例',
    '等位基因数大于等于5的位点比例': '≥5等位基因位点比例',
    '等位基因数大于等于6的位点比例': '≥6等位基因位点比例',
    '等位基因数大于等于7的位点比例': '≥7等位基因位点比例',
    '等位基因数大于等于8的位点比例': '≥8等位基因位点比例',
    
    # B类：峰高、平衡性及随机效应特征 (35个)
    '平均峰高': '平均峰高',
    '峰高标准差': '峰高标准差',
    '最小峰高': '最小峰高',
    '最大峰高': '最大峰高',
    '峰高中位数': '峰高中位数',
    '峰高变异系数': '峰高变异系数',
    '峰高范围': '峰高范围',
    '峰高四分位距': '峰高四分位距',
    
    # 峰高分位数特征
    '峰高第10百分位数': '峰高P10',
    '峰高第20百分位数': '峰高P20',
    '峰高第25百分位数': '峰高P25',
    '峰高第30百分位数': '峰高P30',
    '峰高第40百分位数': '峰高P40',
    '峰高第50百分位数': '峰高P50',
    '峰高第60百分位数': '峰高P60',
    '峰高第70百分位数': '峰高P70',
    '峰高第75百分位数': '峰高P75',
    '峰高第80百分位数': '峰高P80',
    '峰高第90百分位数': '峰高P90',
    '峰高第95百分位数': '峰高P95',
    
    # 峰高比值特征
    '峰高分位数比率_75_25': '峰高四分位距比',
    '峰高分位数比率_90_10': '峰高90/10分位比',
    '峰高分位数比率_95_5': '峰高95/5分位比',
    '最大最小峰高比': '最大最小峰高比',
    '平均最小峰高比': '平均最小峰高比',
    
    # PHR相关特征扩展
    '平均峰高比': '平均峰高比',
    '峰高比标准差': '峰高比标准差',
    '最小峰高比': '最小峰高比',
    '最大峰高比': '最大峰高比',
    '峰高比中位数': '峰高比中位数',
    '峰高比范围': '峰高比范围',
    '可计算峰高比的位点数': '可计算PHR位点数',
    '可计算峰高比的位点比例': '可计算PHR位点比例',
    '严重失衡位点数': '严重失衡位点数',
    '严重失衡位点比例': '严重失衡位点比例',
    '轻度失衡位点数': '轻度失衡位点数',
    '平衡位点数': '平衡位点数',
    
    # 峰高分布特征
    '峰高分布偏度': '峰高分布偏度',
    '峰高分布峭度': '峰高分布峭度',
    '峰高分布多峰性': '峰高分布多峰性',
    '饱和峰数量': '饱和峰数量',
    '饱和峰比例': '饱和峰比例',
    '高峰数量': '高峰数量',
    '低峰数量': '低峰数量',
    
    # C类：信息论及图谱复杂度特征 (15个)
    '位点间平衡熵': '位点间平衡熵',
    '位点间平衡基尼系数': '位点间平衡基尼系数',
    '位点间高度标准差': '位点间高度标准差',
    '位点间高度变异系数': '位点间高度变异系数',
    '位点间高度范围': '位点间高度范围',
    '位点间高度偏度': '位点间高度偏度',
    '位点间高度峭度': '位点间高度峭度',
    '平均位点等位基因熵': '平均位点等位基因熵',
    '位点等位基因熵标准差': '位点等位基因熵标准差',
    '平均位点基尼系数': '平均位点基尼系数',
    '峰高分布熵': '峰高分布熵',
    '有效等位基因位点数': '有效等位基因位点数',
    '无有效等位基因位点数': '无有效等位基因位点数',
    '图谱完整度': '图谱完整度',
    
    # D类：DNA降解与信息丢失特征 (15个)
    '峰高片段大小相关性': '峰高片段大小相关性',
    '峰高片段大小相关性平方': '峰高片段大小相关性²',
    '峰高片段大小回归斜率': '峰高片段大小回归斜率',
    '峰高片段大小回归斜率绝对值': '峰高片段大小回归斜率绝对值',
    '加权峰高片段大小斜率': '加权峰高片段大小斜率',
    '片段大小范围': '片段大小范围',
    '片段大小标准差': '片段大小标准差',
    '片段大小变异系数': '片段大小变异系数',
    '峰高比片段大小斜率': 'PHR片段大小斜率',
    '片段大小加权位点丢失评分': '片段大小加权位点丢失评分',
    'RFU每碱基对降解指数': 'RFU每碱基对降解指数',
    'RFU每碱基对降解指数绝对值': 'RFU每碱基对降解指数绝对值',
    '小大片段信息完整度比率': '小大片段信息完整度比率',
    
    # E类：高级生物学特征 (20个)
    '等位基因辛普森多样性指数': '等位基因辛普森多样性指数',
    '等位基因香农多样性指数': '等位基因香农多样性指数',
    '纯合子位点数': '纯合子位点数',
    '杂合子位点数': '杂合子位点数',
    '多等位基因位点数': '多等位基因位点数',
    '纯合率': '纯合率',
    '杂合率': '杂合率',
    '多等位基因率': '多等位基因率',
    '峰模式复杂度': '峰模式复杂度',
    '峰聚类系数': '峰聚类系数',
    '位点间等位基因数一致性': '位点间等位基因数一致性',
    '相邻峰高比均值': '相邻峰高比均值',
    '相邻峰高比标准差': '相邻峰高比标准差',
    '相邻峰高比最大值': '相邻峰高比最大值',
    '相邻峰高比最小值': '相邻峰高比最小值',
    
    # F类：位点特异性特征 (15个)
    '位点mean高度均值': '位点平均高度均值',
    '位点mean高度标准差': '位点平均高度标准差',
    '位点mean高度范围': '位点平均高度范围',
    '位点max高度均值': '位点最大高度均值',
    '位点max高度标准差': '位点最大高度标准差',
    '位点max高度范围': '位点最大高度范围',
    '位点min高度均值': '位点最小高度均值',
    '位点min高度标准差': '位点最小高度标准差',
    '位点min高度范围': '位点最小高度范围',
    '位点std高度均值': '位点标准差高度均值',
    '位点std高度标准差': '位点标准差高度标准差',
    '位点std高度范围': '位点标准差高度范围',
    '最高效位点高度': '最高效位点高度',
    '最低效位点高度': '最低效位点高度',
    '位点效率差异': '位点效率差异',
    
    # G类：信号质量和噪声特征 (10个)
    '动态范围': '动态范围',
    '信号强度指数': '信号强度指数',
    '高质量峰数量_1000': '高质量峰数量(≥1000)',
    '中等质量峰数量_500': '中等质量峰数量(≥500)',
    '可接受峰数量_200': '可接受峰数量(≥200)',
    '低质量峰数量_100': '低质量峰数量(<100)',
    '高质量峰比例': '高质量峰比例',
    '低质量峰比例': '低质量峰比例',
    
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
    """提取增强的特征集，扩展到90+个特征"""
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
    locus_mean_heights = locus_groups['Height'].mean()
    locus_max_heights = locus_groups['Height'].max()
    locus_min_heights = locus_groups['Height'].min()
    locus_std_heights = locus_groups['Height'].std()
    
    # A类：图谱层面基础统计特征 (20个特征)
    features['样本最大等位基因数'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['样本总特异等位基因数'] = sample_peaks['Allele'].nunique()
    features['每位点平均等位基因数'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
    features['每位点等位基因数标准差'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    features['每位点等位基因数中位数'] = alleles_per_locus.median() if len(alleles_per_locus) > 0 else 0
    features['每位点等位基因数最大值'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['每位点等位基因数最小值'] = alleles_per_locus.min() if len(alleles_per_locus) > 0 else 0
    features['等位基因数变异系数'] = features['每位点等位基因数标准差'] / features['每位点平均等位基因数'] if features['每位点平均等位基因数'] > 0 else 0
    
    # MGTN系列 - 扩展版本
    for N in [1, 2, 3, 4, 5, 6, 7, 8]:
        features[f'等位基因数大于等于{N}的位点数'] = (alleles_per_locus >= N).sum()
        features[f'等位基因数大于等于{N}的位点比例'] = (alleles_per_locus >= N).mean()
    
    # 等位基因计数分布的统计量
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['等位基因计数分布熵'] = calculate_entropy(counts.values)
        features['等位基因计数分布偏度'] = stats.skew(alleles_per_locus.values) if len(alleles_per_locus) > 2 else 0
        features['等位基因计数分布峭度'] = stats.kurtosis(alleles_per_locus.values) if len(alleles_per_locus) > 3 else 0
    else:
        features['等位基因计数分布熵'] = 0
        features['等位基因计数分布偏度'] = 0
        features['等位基因计数分布峭度'] = 0
    
    # B类：峰高、平衡性及随机效应特征 (35个特征)
    if total_peaks > 0:
        # 基础峰高统计 - 扩展版本
        features['平均峰高'] = np.mean(all_heights)
        features['峰高标准差'] = np.std(all_heights) if total_peaks > 1 else 0
        features['最小峰高'] = np.min(all_heights)
        features['最大峰高'] = np.max(all_heights)
        features['峰高中位数'] = np.median(all_heights)
        features['峰高变异系数'] = features['峰高标准差'] / features['平均峰高'] if features['平均峰高'] > 0 else 0
        features['峰高范围'] = features['最大峰高'] - features['最小峰高']
        features['峰高四分位距'] = np.percentile(all_heights, 75) - np.percentile(all_heights, 25)
        
        # 峰高分位数特征
        percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
        for p in percentiles:
            features[f'峰高第{p}百分位数'] = np.percentile(all_heights, p)
        
        # 峰高比值特征
        if total_peaks >= 2:
            features['峰高分位数比率_75_25'] = np.percentile(all_heights, 75) / np.percentile(all_heights, 25) if np.percentile(all_heights, 25) > 0 else 0
            features['峰高分位数比率_90_10'] = np.percentile(all_heights, 90) / np.percentile(all_heights, 10) if np.percentile(all_heights, 10) > 0 else 0
            features['峰高分位数比率_95_5'] = np.percentile(all_heights, 95) / np.percentile(all_heights, 5) if np.percentile(all_heights, 5) > 0 else 0
            features['最大最小峰高比'] = features['最大峰高'] / features['最小峰高'] if features['最小峰高'] > 0 else 0
            features['平均最小峰高比'] = features['平均峰高'] / features['最小峰高'] if features['最小峰高'] > 0 else 0
        
        # PHR相关特征 - 扩展版本
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
            features['最大峰高比'] = np.max(phr_values)
            features['峰高比中位数'] = np.median(phr_values)
            features['峰高比范围'] = np.max(phr_values) - np.min(phr_values)
            features['可计算峰高比的位点数'] = len(phr_values)
            features['可计算峰高比的位点比例'] = len(phr_values) / len(locus_groups) if len(locus_groups) > 0 else 0
            features['严重失衡位点数'] = sum(phr <= PHR_IMBALANCE_THRESHOLD for phr in phr_values)
            features['严重失衡位点比例'] = features['严重失衡位点数'] / len(phr_values)
            features['轻度失衡位点数'] = sum(0.6 < phr <= 0.8 for phr in phr_values)
            features['平衡位点数'] = sum(phr > 0.8 for phr in phr_values)
        else:
            for key in ['平均峰高比', '峰高比标准差', '最小峰高比', '最大峰高比', '峰高比中位数', '峰高比范围',
                       '可计算峰高比的位点数', '可计算峰高比的位点比例', '严重失衡位点数', '严重失衡位点比例',
                       '轻度失衡位点数', '平衡位点数']:
                features[key] = 0
        
        # 峰高分布形状特征
        if total_peaks > 2:
            features['峰高分布偏度'] = stats.skew(all_heights)
            features['峰高分布峭度'] = stats.kurtosis(all_heights, fisher=False)
        else:
            features['峰高分布偏度'] = 0
            features['峰高分布峭度'] = 0
        
        # 峰高多峰性和分布特征
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
        features['高峰数量'] = (all_heights >= np.percentile(all_heights, 90)).sum()
        features['低峰数量'] = (all_heights <= np.percentile(all_heights, 10)).sum()
        
    else:
        # 空值填充
        for key in ['平均峰高', '峰高标准差', '最小峰高', '最大峰高', '峰高中位数', '峰高变异系数',
                   '峰高范围', '峰高四分位距', '峰高分布偏度', '峰高分布峭度', '峰高分布多峰性',
                   '饱和峰数量', '饱和峰比例', '高峰数量', '低峰数量']:
            features[key] = 0
    
    # C类：信息论及图谱复杂度特征 (15个特征)
    if len(locus_heights) > 0:
        total_height = locus_heights.sum()
        if total_height > 0:
            locus_probs = locus_heights / total_height
            features['位点间平衡熵'] = calculate_entropy(locus_probs.values)
            features['位点间平衡基尼系数'] = 1 - sum(locus_probs.values ** 2)
        else:
            features['位点间平衡熵'] = 0
            features['位点间平衡基尼系数'] = 0
        
        # 位点高度统计
        features['位点间高度标准差'] = locus_heights.std() if len(locus_heights) > 1 else 0
        features['位点间高度变异系数'] = features['位点间高度标准差'] / locus_heights.mean() if locus_heights.mean() > 0 else 0
        features['位点间高度范围'] = locus_heights.max() - locus_heights.min()
        features['位点间高度偏度'] = stats.skew(locus_heights.values) if len(locus_heights) > 2 else 0
        features['位点间高度峭度'] = stats.kurtosis(locus_heights.values) if len(locus_heights) > 3 else 0
    else:
        for key in ['位点间平衡熵', '位点间平衡基尼系数', '位点间高度标准差', '位点间高度变异系数',
                   '位点间高度范围', '位点间高度偏度', '位点间高度峭度']:
            features[key] = 0
    
    # 平均位点等位基因分布熵和相关统计
    locus_entropies = []
    locus_gini_coeffs = []
    for marker, marker_group in locus_groups:
        if len(marker_group) > 1:
            heights = marker_group['Height'].values
            height_sum = heights.sum()
            if height_sum > 0:
                probs = heights / height_sum
                entropy = calculate_entropy(probs)
                gini = 1 - sum(probs ** 2)
                locus_entropies.append(entropy)
                locus_gini_coeffs.append(gini)
    
    features['平均位点等位基因熵'] = np.mean(locus_entropies) if locus_entropies else 0
    features['位点等位基因熵标准差'] = np.std(locus_entropies) if len(locus_entropies) > 1 else 0
    features['平均位点基尼系数'] = np.mean(locus_gini_coeffs) if locus_gini_coeffs else 0
    
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
    features['无有效等位基因位点数'] = max(0, 20 - effective_loci_count)
    features['图谱完整度'] = effective_loci_count / 20 if 20 > 0 else 0
    
    # D类：DNA降解与信息丢失特征 (15个特征)
    if total_peaks > 1:
        # 峰高与片段大小的相关性
        if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['峰高片段大小相关性'] = np.corrcoef(all_heights, all_sizes)[0, 1]
            features['峰高片段大小相关性平方'] = features['峰高片段大小相关性'] ** 2
        else:
            features['峰高片段大小相关性'] = 0
            features['峰高片段大小相关性平方'] = 0
        
        # 线性回归斜率和截距
        slope = calculate_ols_slope(all_sizes, all_heights)
        features['峰高片段大小回归斜率'] = slope
        features['峰高片段大小回归斜率绝对值'] = abs(slope)
        
        # 加权回归斜率
        try:
            weights = all_heights / all_heights.sum()
            features['加权峰高片段大小斜率'] = calculate_ols_slope(all_sizes, all_heights)
        except:
            features['加权峰高片段大小斜率'] = 0
        
        # 片段大小统计
        features['片段大小范围'] = np.max(all_sizes) - np.min(all_sizes)
        features['片段大小标准差'] = np.std(all_sizes) if len(all_sizes) > 1 else 0
        features['片段大小变异系数'] = features['片段大小标准差'] / np.mean(all_sizes) if np.mean(all_sizes) > 0 else 0
        
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
        for key in ['峰高片段大小相关性', '峰高片段大小相关性平方', '峰高片段大小回归斜率', 
                   '峰高片段大小回归斜率绝对值', '加权峰高片段大小斜率', '片段大小范围',
                   '片段大小标准差', '片段大小变异系数', '峰高比片段大小斜率']:
            features[key] = 0
    
    # 位点丢失和降解评分
    dropout_score = features['无有效等位基因位点数'] / 20
    features['片段大小加权位点丢失评分'] = dropout_score
    
    # RFU每碱基对衰减指数
    if len(locus_groups) > 1:
        locus_max_heights = []
        locus_avg_sizes = []
        for marker, marker_group in locus_groups:
            max_height = marker_group['Height'].max()
            avg_size = marker_group['Size'].mean()
            locus_max_heights.append(max_height)
            locus_avg_sizes.append(avg_size)
        
        features['RFU每碱基对降解指数'] = calculate_ols_slope(locus_avg_sizes, locus_max_heights)
        features['RFU每碱基对降解指数绝对值'] = abs(features['RFU每碱基对降解指数'])
    else:
        features['RFU每碱基对降解指数'] = 0
        features['RFU每碱基对降解指数绝对值'] = 0
    
    # 小大片段信息完整度比率
    small_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() < 200)
    large_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() >= 200)
    
    if large_fragment_effective > 0:
        features['小大片段信息完整度比率'] = small_fragment_effective / large_fragment_effective
    else:
        features['小大片段信息完整度比率'] = small_fragment_effective / 0.001
    
    # E类：高级生物学特征 (20个特征)
    # 1. 等位基因多样性指数 (Simpson's diversity index)
    if total_peaks > 0:
        allele_counts = sample_peaks['Allele'].value_counts()
        total_alleles = allele_counts.sum()
        simpson_diversity = 1 - sum((n/total_alleles)**2 for n in allele_counts.values)
        shannon_diversity = calculate_entropy(allele_counts.values / total_alleles)
        features['等位基因辛普森多样性指数'] = simpson_diversity
        features['等位基因香农多样性指数'] = shannon_diversity
    else:
        features['等位基因辛普森多样性指数'] = 0
        features['等位基因香农多样性指数'] = 0
    
    # 2. 基因型模式特征
    homozygous_loci = sum(1 for marker, group in locus_groups if len(group) == 1)
    heterozygous_loci = sum(1 for marker, group in locus_groups if len(group) == 2)
    multi_allelic_loci = sum(1 for marker, group in locus_groups if len(group) > 2)
    
    total_analyzed_loci = len(locus_groups)
    if total_analyzed_loci > 0:
        features['纯合子位点数'] = homozygous_loci
        features['杂合子位点数'] = heterozygous_loci
        features['多等位基因位点数'] = multi_allelic_loci
        features['纯合率'] = homozygous_loci / total_analyzed_loci
        features['杂合率'] = heterozygous_loci / total_analyzed_loci
        features['多等位基因率'] = multi_allelic_loci / total_analyzed_loci
    else:
        for key in ['纯合子位点数', '杂合子位点数', '多等位基因位点数', '纯合率', '杂合率', '多等位基因率']:
            features[key] = 0
    
    # 3. 峰模式复杂度和聚类特征
    if len(locus_groups) > 0:
        pattern_complexity = 0
        cluster_coefficient = 0
        
        for marker, group in locus_groups:
            n_alleles = len(group)
            if n_alleles > 1:
                heights = group['Height'].values
                sizes = group['Size'].values
                
                # 位点内复杂度
                height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0
                pattern_complexity += height_cv
                
                # 位点内聚类系数
                if n_alleles > 2:
                    size_range = np.max(sizes) - np.min(sizes)
                    cluster_coeff = 1 - (size_range / 100) if size_range < 100 else 0
                    cluster_coefficient += cluster_coeff
        
        features['峰模式复杂度'] = pattern_complexity / len(locus_groups)
        features['峰聚类系数'] = cluster_coefficient / len(locus_groups)
        
        # 位点间一致性指标
        allele_counts_per_locus = [len(group) for marker, group in locus_groups]
        features['位点间等位基因数一致性'] = 1 - (np.std(allele_counts_per_locus) / np.mean(allele_counts_per_locus)) if np.mean(allele_counts_per_locus) > 0 else 0
    else:
        features['峰模式复杂度'] = 0
        features['峰聚类系数'] = 0
        features['位点间等位基因数一致性'] = 0
    
    # 4. 峰高网络特征
    if total_peaks > 1:
        # 峰高相邻比特征
        sorted_heights = np.sort(all_heights)[::-1]  # 降序排列
        adjacent_ratios = []
        
        for i in range(len(sorted_heights) - 1):
            if sorted_heights[i+1] > 0:
                ratio = sorted_heights[i] / sorted_heights[i+1]
                adjacent_ratios.append(ratio)
        
        if adjacent_ratios:
            features['相邻峰高比均值'] = np.mean(adjacent_ratios)
            features['相邻峰高比标准差'] = np.std(adjacent_ratios) if len(adjacent_ratios) > 1 else 0
            features['相邻峰高比最大值'] = np.max(adjacent_ratios)
            features['相邻峰高比最小值'] = np.min(adjacent_ratios)
        else:
            for key in ['相邻峰高比均值', '相邻峰高比标准差', '相邻峰高比最大值', '相邻峰高比最小值']:
                features[key] = 1.0
    else:
        for key in ['相邻峰高比均值', '相邻峰高比标准差', '相邻峰高比最大值', '相邻峰高比最小值']:
            features[key] = 1.0
    
    # F类：位点特异性特征 (15个特征)
    if len(locus_groups) > 0:
        # 修复：更安全地处理位点高度分布特征
        try:
            locus_height_stats = {
                'mean': locus_mean_heights.values if hasattr(locus_mean_heights, 'values') else np.array(locus_mean_heights),
                'max': locus_max_heights.values if hasattr(locus_max_heights, 'values') else np.array(locus_max_heights),
                'min': locus_min_heights.values if hasattr(locus_min_heights, 'values') else np.array(locus_min_heights),
                'std': locus_std_heights.fillna(0).values if hasattr(locus_std_heights, 'values') else np.array(locus_std_heights.fillna(0))
            }
        except:
            # 如果上述方法失败，重新计算
            locus_height_stats = {
                'mean': [group['Height'].mean() for _, group in locus_groups],
                'max': [group['Height'].max() for _, group in locus_groups],
                'min': [group['Height'].min() for _, group in locus_groups],
                'std': [group['Height'].std() if len(group) > 1 else 0 for _, group in locus_groups]
            }
        
        for stat_name, values in locus_height_stats.items():
            if len(values) > 0:
                features[f'位点{stat_name}高度均值'] = np.mean(values)
                features[f'位点{stat_name}高度标准差'] = np.std(values) if len(values) > 1 else 0
                features[f'位点{stat_name}高度范围'] = np.max(values) - np.min(values)
        
        # 位点效率指标
        features['最高效位点高度'] = locus_heights.max() if hasattr(locus_heights, 'max') else max(locus_heights)
        features['最低效位点高度'] = locus_heights.min() if hasattr(locus_heights, 'min') else min(locus_heights)
        features['位点效率差异'] = features['最高效位点高度'] - features['最低效位点高度']
    else:
        for key in ['位点mean高度均值', '位点mean高度标准差', '位点mean高度范围',
                   '位点max高度均值', '位点max高度标准差', '位点max高度范围',
                   '位点min高度均值', '位点min高度标准差', '位点min高度范围',
                   '位点std高度均值', '位点std高度标准差', '位点std高度范围',
                   '最高效位点高度', '最低效位点高度', '位点效率差异']:
            features[key] = 0
    
    # G类：信号质量和噪声特征 (10个特征)
    if total_peaks > 0:
        # 信噪比相关特征
        min_height = np.min(all_heights)
        max_height = np.max(all_heights)
        mean_height = np.mean(all_heights)
        
        features['动态范围'] = max_height / min_height if min_height > 0 else 0
        features['信号强度指数'] = mean_height / min_height if min_height > 0 else 0
        
        # 阈值相关特征
        above_threshold_1000 = (all_heights >= 1000).sum()
        above_threshold_500 = (all_heights >= 500).sum()
        above_threshold_200 = (all_heights >= 200).sum()
        below_threshold_100 = (all_heights < 100).sum()
        
        features['高质量峰数量_1000'] = above_threshold_1000
        features['中等质量峰数量_500'] = above_threshold_500
        features['可接受峰数量_200'] = above_threshold_200
        features['低质量峰数量_100'] = below_threshold_100
        
        features['高质量峰比例'] = above_threshold_1000 / total_peaks
        features['低质量峰比例'] = below_threshold_100 / total_peaks
    else:
        for key in ['动态范围', '信号强度指数', '高质量峰数量_1000', '中等质量峰数量_500',
                   '可接受峰数量_200', '低质量峰数量_100', '高质量峰比例', '低质量峰比例']:
            features[key] = 0
    
    # 验证特征数量
    print(f"样本 {sample_file} 提取了 {len(features) - 1} 个特征")  # 减1是因为包含了样本文件名
    
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
    min_features_to_select=8,  # 设置最少特征数为8
    n_jobs=-1,
    verbose=1
)

rfecv.fit(X_scaled, y_encoded)

elapsed_time = time() - start_time
print(f"RFECV完成，耗时: {elapsed_time:.1f}秒")

# 分析RFECV结果，选择合适的特征数
print("\n分析不同特征数的性能...")

if hasattr(rfecv, 'cv_results_'):
    scores = rfecv.cv_results_['mean_test_score']
    scores_std = rfecv.cv_results_['std_test_score']
elif hasattr(rfecv, 'grid_scores_'):
    scores = rfecv.grid_scores_
    scores_std = [0] * len(scores)
else:
    scores = [0.0] * len(feature_cols)
    scores_std = [0] * len(feature_cols)

# 寻找最优特征数（基于准确率分析）
def find_optimal_features(scores, scores_std, min_features=8, max_features=None):
    """
    基于准确率分析寻找最优特征数
    
    策略：
    1. 首先找到最高分数
    2. 在最高分数的1个标准差范围内寻找特征数较少的点
    3. 确保特征数不少于min_features
    4. 如果最高分数对应的特征数已经合理，就选择它
    """
    if max_features is None:
        max_features = len(scores)
    
    scores = np.array(scores)
    scores_std = np.array(scores_std)
    
    # 找到最高分数及其位置
    max_score = np.max(scores)
    max_score_idx = np.argmax(scores)
    optimal_n_features = max_score_idx + 1
    
    print(f"最高分数: {max_score:.4f} (特征数: {optimal_n_features})")
    
    # 如果最优特征数太少，寻找合理的替代
    if optimal_n_features < min_features:
        # 在min_features以上寻找最好的点
        valid_range = range(min_features-1, min(max_features, len(scores)))
        if valid_range:
            best_idx_in_range = min_features - 1 + np.argmax(scores[min_features-1:])
            optimal_n_features = best_idx_in_range + 1
            print(f"最优特征数过少，调整为: {optimal_n_features} (分数: {scores[best_idx_in_range]:.4f})")
    
    # 检查是否有在1个标准差内的更简单模型
    threshold = max_score - scores_std[max_score_idx]
    
    # 从少到多检查特征数
    for i in range(min_features-1, len(scores)):
        if scores[i] >= threshold:
            if i + 1 < optimal_n_features:  # 找到更简单但性能相当的模型
                print(f"发现更简单的模型: {i+1}个特征，分数: {scores[i]:.4f} (在1σ范围内)")
                # 但我们还是倾向于使用更多特征来确保稳健性
                if optimal_n_features - (i + 1) <= 5:  # 如果差异不大，还是用原来的
                    break
            break
    
    return optimal_n_features

# 寻找最优特征数
optimal_n_features = find_optimal_features(scores, scores_std, min_features=10, max_features=min(50, len(feature_cols)))

print(f"确定的最优特征数: {optimal_n_features}")

# 如果RFECV自动选择的特征数与我们分析的不同，重新选择特征
if optimal_n_features != rfecv.n_features_:
    print(f"RFECV选择了{rfecv.n_features_}个特征，但基于性能分析，我们选择{optimal_n_features}个特征")
    
    # 根据特征重要性排名重新选择
    feature_ranking = rfecv.ranking_
    
    # 选择排名前optimal_n_features的特征
    selected_indices = np.where(feature_ranking <= optimal_n_features)[0]
    selected_features = [feature_cols[i] for i in selected_indices]
    
    # 确保选择了正确数量的特征
    if len(selected_features) != optimal_n_features:
        # 如果数量不匹配，按排名选择
        sorted_indices = np.argsort(feature_ranking)
        selected_indices = sorted_indices[:optimal_n_features]
        selected_features = [feature_cols[i] for i in selected_indices]
    
    print(f"重新选择了{len(selected_features)}个特征")
else:
    # 使用RFECV的原始选择
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
    selected_indices = [i for i in range(len(feature_cols)) if rfecv.support_[i]]

print(f"\n最终选择的特征数: {len(selected_features)}")
print(f"最终特征列表:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feature}")

# 重新计算最佳分数
if len(selected_features) <= len(scores):
    best_score = scores[len(selected_features) - 1]
else:
    best_score = max(scores)

print(f"对应的交叉验证分数: {best_score:.4f}")

# 创建最终的特征矩阵
X_selected = X_scaled[:, selected_indices]

# 可视化RFECV结果和我们的选择
plt.figure(figsize=(14, 8))

# 绘制所有分数
plt.subplot(1, 2, 1)
plt.plot(range(1, len(scores) + 1), scores, 'bo-', markersize=4)
plt.fill_between(range(1, len(scores) + 1), 
                 np.array(scores) - np.array(scores_std),
                 np.array(scores) + np.array(scores_std), 
                 alpha=0.2)

# 标记RFECV选择的特征数
plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', 
           label=f'RFECV选择: {rfecv.n_features_}个特征')

# 标记我们选择的特征数
plt.axvline(x=len(selected_features), color='green', linestyle='-', linewidth=2,
           label=f'最终选择: {len(selected_features)}个特征')

plt.xlabel('特征数量')
plt.ylabel('交叉验证分数')
plt.title('RFECV特征选择结果分析')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示性能稳定区间
plt.subplot(1, 2, 2)
if len(scores) >= 20:
    # 显示前20个特征的详细分数
    detail_range = range(1, min(21, len(scores) + 1))
    detail_scores = scores[:len(detail_range)]
    detail_std = scores_std[:len(detail_range)]
    
    plt.plot(detail_range, detail_scores, 'bo-', markersize=6)
    plt.fill_between(detail_range, 
                     np.array(detail_scores) - np.array(detail_std),
                     np.array(detail_scores) + np.array(detail_std), 
                     alpha=0.2)
    
    plt.axvline(x=len(selected_features), color='green', linestyle='-', linewidth=2,
               label=f'选择: {len(selected_features)}个特征')
    
    plt.xlabel('特征数量')
    plt.ylabel('交叉验证分数')
    plt.title('前20个特征的详细性能')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'RFECV结果分析.png'), dpi=300, bbox_inches='tight')
plt.close()

# 输出特征选择的详细分析
print(f"\n📊 特征选择详细分析:")
print(f"   • 原始特征总数: {len(feature_cols)}")
print(f"   • RFECV推荐特征数: {rfecv.n_features_}")
print(f"   • 最终选择特征数: {len(selected_features)}")
print(f"   • 特征保留比例: {len(selected_features)/len(feature_cols)*100:.1f}%")
print(f"   • 对应性能分数: {best_score:.4f}")

# 显示性能变化趋势
if len(scores) >= 5:
    print(f"\n📈 前几个关键特征数的性能:")
    key_points = [5, 10, 15, 20, len(selected_features)]
    key_points = [p for p in key_points if p <= len(scores)]
    key_points = sorted(set(key_points))
    
    for n_feat in key_points:
        score = scores[n_feat - 1]
        print(f"   • {n_feat:2d}个特征: {score:.4f}")
```