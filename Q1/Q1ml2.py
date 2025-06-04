# -*- coding: utf-8 -*-
"""
增强版 法医DNA分析 - 问题1：贡献者人数识别
版本: 4.0 (增强版)
新增功能:
1. 详细的分类报告 - 每个模型对不同人数的精确表现
2. 多种聚类算法测试
3. 无监督学习结果与有监督学习对比
4. 更全面的性能评估
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
from collections import Counter

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 聚类算法
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# 可选的高级算法
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost不可用，将跳过")
    XGBOOST_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'q1_enhanced_plots_v4')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

print("=== 增强版 NoC 识别系统 v4.0 ===")
print("新增: 详细分类报告 + 聚类分析")

# =====================
# 1. 数据加载与预处理 (保持不变)
# =====================
print("\n1. 数据加载与预处理...")

def extract_noc_from_filename(filename):
    """从文件名提取贡献者人数"""
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
    if match:
        ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return len(ids) if len(ids) > 0 else np.nan
    return np.nan

# 加载数据
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"成功加载数据，形状: {df.shape}")
except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 提取NoC
df['NoC_True'] = df['Sample File'].apply(extract_noc_from_filename)
df = df.dropna(subset=['NoC_True'])
df['NoC_True'] = df['NoC_True'].astype(int)

print(f"NoC分布: {df['NoC_True'].value_counts().sort_index().to_dict()}")
print(f"样本数: {df['Sample File'].nunique()}")

# =====================
# 2. 简化的峰处理 (保持不变)
# =====================
print("\n2. 简化的峰处理...")

def process_peaks_simplified(sample_data, height_threshold=50, saturation_threshold=30000):
    """简化的峰处理函数"""
    processed_data = []
    
    for _, sample_row in sample_data.iterrows():
        sample_file = sample_row['Sample File']
        marker = sample_row['Marker']
        
        peaks = []
        for i in range(1, 101):
            allele = sample_row.get(f'Allele {i}')
            size = sample_row.get(f'Size {i}')
            height = sample_row.get(f'Height {i}')
            
            if pd.notna(allele) and pd.notna(size) and pd.notna(height):
                corrected_height = min(float(height), saturation_threshold)
                
                if corrected_height >= height_threshold:
                    peaks.append({
                        'Sample File': sample_file,
                        'Marker': marker,
                        'Allele': allele,
                        'Size': float(size),
                        'Height': corrected_height,
                        'Original_Height': float(height)
                    })
        
        processed_data.extend(peaks)
    
    return pd.DataFrame(processed_data)

# 处理所有样本
all_processed_peaks = []
for sample_file, group in df.groupby('Sample File'):
    sample_peaks = process_peaks_simplified(group)
    all_processed_peaks.append(sample_peaks)

df_peaks = pd.concat(all_processed_peaks, ignore_index=True)
print(f"处理后的峰数据形状: {df_peaks.shape}")

# =====================
# 3. 增强特征工程 (保持不变)
# =====================
print("\n3. 增强特征工程...")

def extract_comprehensive_features(sample_file, sample_peaks, marker_info=None):
    """基于V5方案的综合特征提取"""
    if sample_peaks.empty:
        return {}
    
    features = {'Sample File': sample_file}
    
    # 基础统计
    total_peaks = len(sample_peaks)
    all_heights = sample_peaks['Height'].values
    all_sizes = sample_peaks['Size'].values
    
    # A. 图谱层面基础计数与统计特征
    alleles_per_locus = sample_peaks.groupby('Marker')['Allele'].nunique()
    
    features['mac_profile'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
    features['total_distinct_alleles'] = sample_peaks['Allele'].nunique()
    features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
    features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
    features['num_effective_loci'] = len(alleles_per_locus)
    
    # MGTN系列
    for n in [2, 3, 4, 5, 6]:
        features[f'loci_gt{n}_alleles'] = (alleles_per_locus >= n).sum()
    
    # 等位基因数分布的熵
    if len(alleles_per_locus) > 0:
        counts = alleles_per_locus.value_counts(normalize=True)
        features['allele_count_entropy'] = -np.sum(counts * np.log(counts + 1e-10))
    else:
        features['allele_count_entropy'] = 0
    
    # B. 峰高相关特征
    if total_peaks > 0:
        features['avg_peak_height'] = np.mean(all_heights)
        features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
        features['cv_peak_height'] = features['std_peak_height'] / features['avg_peak_height'] if features['avg_peak_height'] > 0 else 0
        features['skewness_peak_height'] = stats.skew(all_heights) if total_peaks > 2 else 0
        features['kurtosis_peak_height'] = stats.kurtosis(all_heights) if total_peaks > 3 else 0
        features['min_peak_height'] = np.min(all_heights)
        features['max_peak_height'] = np.max(all_heights)
        features['peak_height_range'] = features['max_peak_height'] - features['min_peak_height']
    else:
        for key in ['avg_peak_height', 'std_peak_height', 'cv_peak_height', 'skewness_peak_height', 
                   'kurtosis_peak_height', 'min_peak_height', 'max_peak_height', 'peak_height_range']:
            features[key] = 0
    
    # 峰高分布特征
    if total_peaks > 0:
        hist, _ = np.histogram(np.log(all_heights + 1), bins=min(10, total_peaks))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        features['peak_height_entropy'] = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        
        height_threshold_80 = np.percentile(all_heights, 80)
        features['high_peaks_ratio'] = (all_heights >= height_threshold_80).mean()
    else:
        features['peak_height_entropy'] = 0
        features['high_peaks_ratio'] = 0
    
    # C. 峰高比(PHR)相关特征
    phr_values = []
    for marker, marker_group in sample_peaks.groupby('Marker'):
        if len(marker_group) == 2:
            heights = marker_group['Height'].values
            phr = min(heights) / max(heights)
            phr_values.append(phr)
    
    if phr_values:
        features['avg_phr'] = np.mean(phr_values)
        features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
        features['min_phr'] = np.min(phr_values)
        features['median_phr'] = np.median(phr_values)
        features['num_loci_with_phr'] = len(phr_values)
        features['imbalance_ratio'] = (np.array(phr_values) < 0.6).mean()
    else:
        for key in ['avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr', 'imbalance_ratio']:
            features[key] = 0
    
    # D. 位点间平衡性
    locus_heights = sample_peaks.groupby('Marker')['Height'].sum()
    if len(locus_heights) > 0:
        total_height = locus_heights.sum()
        locus_probs = locus_heights / total_height
        features['inter_locus_entropy'] = -np.sum(locus_probs * np.log(locus_probs + 1e-10))
        features['locus_height_cv'] = locus_heights.std() / locus_heights.mean() if locus_heights.mean() > 0 else 0
    else:
        features['inter_locus_entropy'] = 0
        features['locus_height_cv'] = 0
    
    # E. 位点内等位基因分布熵
    locus_entropies = []
    for marker, marker_group in sample_peaks.groupby('Marker'):
        heights = marker_group['Height'].values
        if len(heights) > 1:
            probs = heights / heights.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            locus_entropies.append(entropy)
    
    features['avg_locus_allele_entropy'] = np.mean(locus_entropies) if locus_entropies else 0
    
    # F. 大小相关特征
    if total_peaks > 1:
        features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1] if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1 else 0
        
        if len(np.unique(all_sizes)) > 1:
            slope, _, _, _, _ = stats.linregress(all_sizes, all_heights)
            features['height_size_slope'] = slope
        else:
            features['height_size_slope'] = 0
            
        features['size_range'] = all_sizes.max() - all_sizes.min()
        features['avg_size'] = np.mean(all_sizes)
        features['std_size'] = np.std(all_sizes)
    else:
        features['height_size_correlation'] = 0
        features['height_size_slope'] = 0
        features['size_range'] = 0
        features['avg_size'] = 0
        features['std_size'] = 0
    
    # G. 复杂度指标
    features['total_peaks'] = total_peaks
    features['unique_markers'] = sample_peaks['Marker'].nunique()
    features['peaks_per_marker'] = total_peaks / features['unique_markers'] if features['unique_markers'] > 0 else 0
    
    # H. 复合特征
    features['complexity_score'] = features['mac_profile'] * features['avg_alleles_per_locus']
    features['balance_score'] = features['avg_phr'] * (1 - features['locus_height_cv'])
    features['quality_score'] = features['avg_peak_height'] / (features['std_peak_height'] + 1)
    
    return features

# 提取所有样本的特征
print("提取特征中...")
all_features = []
for sample_file, group in df_peaks.groupby('Sample File'):
    features = extract_comprehensive_features(sample_file, group)
    all_features.append(features)

df_features = pd.DataFrame(all_features)

# 合并NoC标签
noc_map = df.groupby('Sample File')['NoC_True'].first().to_dict()
df_features['NoC_True'] = df_features['Sample File'].map(noc_map)
df_features = df_features.dropna(subset=['NoC_True'])

print(f"特征数据形状: {df_features.shape}")
print(f"特征列数: {len([col for col in df_features.columns if col not in ['Sample File', 'NoC_True']])}")

# =====================
# 4. 多种机器学习算法测试 (增强版)
# =====================
print("\n4. 多种机器学习算法测试...")

# 准备数据
feature_cols = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
X = df_features[feature_cols].fillna(0)
y = df_features['NoC_True']

print(f"原始NoC标签: {sorted(y.unique())}")

# 为算法重新编码标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_original = y.copy()

print(f"编码后标签: {sorted(np.unique(y_encoded))} (对应原始: {sorted(y.unique())})")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 同时保存原始标签用于显示
_, _, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_original, test_size=0.3, random_state=42, stratify=y_encoded
)

# 交叉验证设置
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义算法
algorithms = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

if XGBOOST_AVAILABLE:
    algorithms['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# 测试所有算法并生成详细报告
results = {}
detailed_reports = {}

print("测试各种算法...")
print("="*80)

for name, clf in algorithms.items():
    try:
        # 交叉验证
        cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        # 训练并测试
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        # 转换预测结果回原始标签
        y_test_orig_current = label_encoder.inverse_transform(y_test)
        y_pred_orig = label_encoder.inverse_transform(y_pred)
        
        # 生成详细分类报告
        class_names = [f"{x}人" for x in sorted(label_encoder.classes_)]
        detailed_report = classification_report(
            y_test_orig_current, y_pred_orig, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # 计算每个类别的指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_orig_current, y_pred_orig, 
            labels=sorted(label_encoder.classes_),
            zero_division=0
        )
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': test_acc,
            'model': clf,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support
        }
        
        detailed_reports[name] = detailed_report
        
        # 打印详细结果
        print(f"\n{name} 详细分类报告:")
        report_str = classification_report(
            y_test_orig_current, y_pred_orig, 
            target_names=class_names,
            zero_division=0
        )
        print(report_str)
        
        print(f"{name:20} CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} Test: {test_acc:.4f}")
        print("-"*60)
        
    except Exception as e:
        print(f"{name} 失败: {e}")

# =====================
# 5. 聚类算法测试 (新增)
# =====================
print("\n5. 聚类算法测试...")
print("="*50)

# 聚类算法字典
clustering_algorithms = {
    'K-Means': KMeans(n_clusters=len(y.unique()), random_state=42, n_init=10),
    'Gaussian Mixture': GaussianMixture(n_components=len(y.unique()), random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=len(y.unique())),
    'Spectral Clustering': SpectralClustering(n_clusters=len(y.unique()), random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

clustering_results = {}

print(f"真实NoC类别数: {len(y.unique())}")
print(f"聚类目标: {sorted(y.unique())}")

for name, clusterer in clustering_algorithms.items():
    try:
        print(f"\n测试 {name}...")
        
        # 执行聚类
        if name == 'DBSCAN':
            # DBSCAN需要调参
            cluster_labels = clusterer.fit_predict(X_scaled)
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  DBSCAN发现 {n_clusters_found} 个聚类，{n_noise} 个噪声点")
        else:
            cluster_labels = clusterer.fit_predict(X_scaled)
            n_clusters_found = len(set(cluster_labels))
        
        # 计算聚类评估指标
        if len(set(cluster_labels)) > 1:  # 确保有多个聚类
            ari = adjusted_rand_score(y_encoded, cluster_labels)
            nmi = normalized_mutual_info_score(y_encoded, cluster_labels)
            homogeneity = homogeneity_score(y_encoded, cluster_labels)
            completeness = completeness_score(y_encoded, cluster_labels)
            v_measure = v_measure_score(y_encoded, cluster_labels)
            
            # 轮廓系数
            if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < len(cluster_labels):
                silhouette = silhouette_score(X_scaled, cluster_labels)
            else:
                silhouette = -1  # 无法计算
            
            clustering_results[name] = {
                'n_clusters_found': n_clusters_found,
                'ari': ari,
                'nmi': nmi,
                'homogeneity': homogeneity,
                'completeness': completeness,
                'v_measure': v_measure,
                'silhouette': silhouette,
                'cluster_labels': cluster_labels
            }
            
            print(f"  发现聚类数: {n_clusters_found}")
            print(f"  调整兰德指数 (ARI): {ari:.4f}")
            print(f"  标准化互信息 (NMI): {nmi:.4f}")
            print(f"  同质性: {homogeneity:.4f}")
            print(f"  完整性: {completeness:.4f}")
            print(f"  V-measure: {v_measure:.4f}")
            print(f"  轮廓系数: {silhouette:.4f}")
            
        else:
            print(f"  {name} 只找到一个聚类，跳过评估")
            
    except Exception as e:
        print(f"  {name} 失败: {e}")

# =====================
# 6. 模型集成 (保持原有逻辑)
# =====================
print("\n6. 模型集成...")

# 选择top 3模型进行集成
top_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
print("Top 3 模型:")
for name, result in top_models:
    print(f"  {name}: {result['cv_mean']:.4f}")

# 创建投票分类器
voting_models = [(name, result['model']) for name, result in top_models]
ensemble_model = VotingClassifier(
    estimators=voting_models,
    voting='soft',
    weights=[3, 2, 1]
)

# 训练集成模型
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"集成模型测试准确率: {ensemble_acc:.4f}")

# 集成模型的详细报告
y_pred_ensemble_orig = label_encoder.inverse_transform(y_pred_ensemble)
y_test_orig_ensemble = label_encoder.inverse_transform(y_test)

print(f"\n集成模型详细分类报告:")
class_names = [f"{x}人" for x in sorted(label_encoder.classes_)]
ensemble_report = classification_report(
    y_test_orig_ensemble, y_pred_ensemble_orig, 
    target_names=class_names,
    zero_division=0
)
print(ensemble_report)

# 更新结果
results['Ensemble'] = {
    'cv_mean': cross_val_score(ensemble_model, X_scaled, y_encoded, cv=cv, scoring='accuracy').mean(),
    'test_acc': ensemble_acc,
    'model': ensemble_model
}

# =====================
# 7. 增强的可视化分析
# =====================
print("\n7. 增强的可视化分析...")

# 7.1 模型性能比较图
plt.figure(figsize=(16, 10))
model_names = list(results.keys())
cv_scores = [results[name]['cv_mean'] for name in model_names]
test_scores = [results[name]['test_acc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8, color='skyblue')
bars2 = plt.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8, color='lightcoral')

# 添加数值标签
for i, (cv_score, test_score) in enumerate(zip(cv_scores, test_scores)):
    plt.text(i - width/2, cv_score + 0.01, f'{cv_score:.3f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width/2, test_score + 0.01, f'{test_score:.3f}', ha='center', va='bottom', fontsize=9)

plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('各模型性能比较')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7.2 每个模型的详细性能热图
if len(results) > 2:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 选择前6个模型
    top_6_models = list(results.keys())[:6]
    
    for idx, model_name in enumerate(top_6_models):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        if 'precision_per_class' in results[model_name]:
            # 创建性能矩阵
            metrics = ['Precision', 'Recall', 'F1-Score']
            classes = [f"{x}人" for x in sorted(label_encoder.classes_)]
            
            perf_matrix = np.array([
                results[model_name]['precision_per_class'],
                results[model_name]['recall_per_class'], 
                results[model_name]['f1_per_class']
            ])
            
            im = ax.imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # 设置标签
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(metrics)
            
            # 添加数值
            for i in range(len(metrics)):
                for j in range(len(classes)):
                    text = ax.text(j, i, f'{perf_matrix[i, j]:.3f}', 
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_title(f'{model_name}')
        else:
            ax.text(0.5, 0.5, f'{model_name}\n无详细指标', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # 隐藏多余的子图
    for idx in range(len(top_6_models), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 7.3 聚类结果可视化
if clustering_results:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 使用PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_methods = list(clustering_results.keys())[:6]  # 最多显示6个聚类方法
    
    for idx, method in enumerate(cluster_methods):
        if idx >= 6:
            break
            
        ax = axes[idx]
        cluster_labels = clustering_results[method]['cluster_labels']
        
        # 绘制聚类结果
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
        ax.set_title(f'{method}\nARI: {clustering_results[method]["ari"]:.3f}, '
                    f'NMI: {clustering_results[method]["nmi"]:.3f}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.grid(True, alpha=0.3)
    
    # 真实标签作为参考
    if len(cluster_methods) < 6:
        ax = axes[len(cluster_methods)]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='tab10', alpha=0.7, s=30)
        ax.set_title('真实NoC标签 (参考)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NoC')
    
    # 隐藏多余的子图
    for idx in range(len(cluster_methods) + 1, 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clustering_results_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 7.4 聚类性能评估对比
if clustering_results:
    metrics = ['ari', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'silhouette']
    methods = list(clustering_results.keys())
    
    # 创建评估矩阵
    eval_matrix = np.zeros((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            eval_matrix[i, j] = clustering_results[method][metric]
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(eval_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    # 设置标签
    plt.xticks(range(len(metrics)), [m.upper() for m in metrics])
    plt.yticks(range(len(methods)), methods)
    
    # 添加数值
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = plt.text(j, i, f'{eval_matrix[i, j]:.3f}', 
                           ha="center", va="center", 
                           color="white" if abs(eval_matrix[i, j]) > 0.5 else "black",
                           fontweight='bold')
    
    plt.title('聚类算法性能评估对比')
    plt.colorbar(im, label='评估分数')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'clustering_evaluation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 7.5 混淆矩阵对比 - 最佳模型
best_model_name = max(results.keys(), key=lambda x: results[x]['test_acc'])
best_predictions = results[best_model_name]['model'].predict(X_test)

y_test_display = label_encoder.inverse_transform(y_test)
best_predictions_display = label_encoder.inverse_transform(best_predictions)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_display, best_predictions_display)
class_names = [f"{x}人" for x in sorted(label_encoder.classes_)]

# 计算准确率百分比
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 绘制混淆矩阵
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           cbar_kws={'label': '百分比 (%)'})

# 在每个格子里添加绝对数量
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                ha='center', va='center', color='red', fontsize=10, fontweight='bold')

plt.title(f'{best_model_name} 混淆矩阵\n总体准确率: {results[best_model_name]["test_acc"]:.4f}')
plt.ylabel('真实标签 (NoC)')
plt.xlabel('预测标签 (NoC)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_best_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7.6 特征重要性分析
if hasattr(results[best_model_name]['model'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': results[best_model_name]['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(14, 10))
    top_features = feature_importance.head(20)  # 显示前20个特征
    
    # 创建颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'{best_model_name} 特征重要性排名 (前20位)')
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(importance + 0.001, i, f'{importance:.4f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{best_model_name} Top 15 重要特征:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:35} {row['importance']:.6f}")

# =====================
# 8. 综合性能分析
# =====================
print("\n8. 综合性能分析...")

# 8.1 有监督学习 vs 无监督学习对比
print("\n有监督学习 vs 无监督学习性能对比:")
print("="*60)

print("有监督学习最佳结果:")
print(f"  最佳模型: {best_model_name}")
print(f"  测试准确率: {results[best_model_name]['test_acc']:.4f}")
print(f"  交叉验证准确率: {results[best_model_name]['cv_mean']:.4f}±{results[best_model_name]['cv_std']:.4f}")

if clustering_results:
    print("\n无监督学习最佳结果:")
    best_clustering = max(clustering_results.items(), key=lambda x: x[1]['ari'])
    best_clustering_name, best_clustering_result = best_clustering
    print(f"  最佳聚类: {best_clustering_name}")
    print(f"  调整兰德指数 (ARI): {best_clustering_result['ari']:.4f}")
    print(f"  标准化互信息 (NMI): {best_clustering_result['nmi']:.4f}")
    print(f"  V-measure: {best_clustering_result['v_measure']:.4f}")

# 8.2 各NoC类别详细分析
print(f"\n各NoC类别详细分析 (基于{best_model_name}):")
print("="*60)

best_model_results = results[best_model_name]
if 'precision_per_class' in best_model_results:
    class_labels = sorted(label_encoder.classes_)
    
    analysis_df = pd.DataFrame({
        'NoC': [f"{x}人" for x in class_labels],
        'Precision': best_model_results['precision_per_class'],
        'Recall': best_model_results['recall_per_class'],
        'F1-Score': best_model_results['f1_per_class'],
        'Support': best_model_results['support_per_class'].astype(int)
    })
    
    print(analysis_df.to_string(index=False, float_format='%.4f'))
    
    # 找出表现最好和最差的类别
    best_f1_idx = np.argmax(best_model_results['f1_per_class'])
    worst_f1_idx = np.argmin(best_model_results['f1_per_class'])
    
    print(f"\n表现最佳类别: {class_labels[best_f1_idx]}人 (F1: {best_model_results['f1_per_class'][best_f1_idx]:.4f})")
    print(f"表现最差类别: {class_labels[worst_f1_idx]}人 (F1: {best_model_results['f1_per_class'][worst_f1_idx]:.4f})")

# =====================
# 9. 保存结果
# =====================
print("\n9. 保存结果...")

# 保存特征数据
df_features.to_csv(os.path.join(DATA_DIR, 'q1_features_enhanced_v4.csv'), index=False, encoding='utf-8-sig')

# 应用最佳模型到所有数据
best_model_final = results[best_model_name]['model']
y_pred_all_encoded = best_model_final.predict(X_scaled)
y_pred_all = label_encoder.inverse_transform(y_pred_all_encoded)
df_features['predicted_noc'] = y_pred_all

# 保存预测结果
df_features.to_csv(os.path.join(DATA_DIR, 'q1_predictions_enhanced_v4.csv'), index=False, encoding='utf-8-sig')

# 保存详细的性能总结
detailed_summary = {
    'analysis_metadata': {
        'version': '4.0',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': best_model_name,
        'best_test_accuracy': float(results[best_model_name]['test_acc']),
        'feature_count': len(feature_cols),
        'sample_count': len(df_features),
        'noc_distribution': y_original.value_counts().sort_index().to_dict()
    },
    'supervised_learning_results': {
        name: {
            'cv_mean': float(result['cv_mean']), 
            'cv_std': float(result['cv_std']),
            'test_acc': float(result['test_acc']),
            'precision_per_class': result.get('precision_per_class', []).tolist() if hasattr(result.get('precision_per_class', []), 'tolist') else result.get('precision_per_class', []),
            'recall_per_class': result.get('recall_per_class', []).tolist() if hasattr(result.get('recall_per_class', []), 'tolist') else result.get('recall_per_class', []),
            'f1_per_class': result.get('f1_per_class', []).tolist() if hasattr(result.get('f1_per_class', []), 'tolist') else result.get('f1_per_class', []),
            'support_per_class': result.get('support_per_class', []).tolist() if hasattr(result.get('support_per_class', []), 'tolist') else result.get('support_per_class', [])
        } 
        for name, result in results.items() if name != 'Ensemble'  # Ensemble模型对象无法序列化
    },
    'unsupervised_learning_results': {
        name: {
            'n_clusters_found': int(result['n_clusters_found']),
            'ari': float(result['ari']),
            'nmi': float(result['nmi']),
            'homogeneity': float(result['homogeneity']),
            'completeness': float(result['completeness']),
            'v_measure': float(result['v_measure']),
            'silhouette': float(result['silhouette'])
        }
        for name, result in clustering_results.items()
    } if clustering_results else {},
    'detailed_classification_reports': detailed_reports
}

with open(os.path.join(DATA_DIR, 'q1_detailed_summary_v4.json'), 'w', encoding='utf-8') as f:
    json.dump(detailed_summary, f, ensure_ascii=False, indent=2)

# =====================
# 10. 最终报告
# =====================
print("\n" + "="*80)
print("增强版 NoC 识别系统 v4.0 - 最终报告")
print("="*80)

print(f"\n📊 数据概况:")
print(f"   • 样本总数: {len(df_features)}")
print(f"   • NoC分布: {dict(y_original.value_counts().sort_index())}")
print(f"   • 特征总数: {len(feature_cols)}")

print(f"\n🏆 有监督学习最佳结果:")
print(f"   • 最佳模型: {best_model_name}")
print(f"   • 测试集准确率: {results[best_model_name]['test_acc']:.4f}")
print(f"   • 交叉验证: {results[best_model_name]['cv_mean']:.4f}±{results[best_model_name]['cv_std']:.4f}")

if clustering_results:
    best_clustering = max(clustering_results.items(), key=lambda x: x[1]['ari'])
    print(f"\n🔍 无监督学习最佳结果:")
    print(f"   • 最佳聚类: {best_clustering[0]}")
    print(f"   • ARI指数: {best_clustering[1]['ari']:.4f}")
    print(f"   • NMI指数: {best_clustering[1]['nmi']:.4f}")

print(f"\n📈 各类别最佳表现 (基于{best_model_name}):")
if 'precision_per_class' in results[best_model_name]:
    class_labels = sorted(label_encoder.classes_)
    for i, noc in enumerate(class_labels):
        precision = results[best_model_name]['precision_per_class'][i]
        recall = results[best_model_name]['recall_per_class'][i]
        f1 = results[best_model_name]['f1_per_class'][i]
        support = results[best_model_name]['support_per_class'][i]
        print(f"   • {noc}人: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n={int(support)})")

print(f"\n💾 保存的文件:")
print(f"   • 增强特征数据: q1_features_enhanced_v4.csv")
print(f"   • 预测结果: q1_predictions_enhanced_v4.csv")
print(f"   • 详细性能报告: q1_detailed_summary_v4.json")
print(f"   • 图表目录: {PLOTS_DIR}/")

print(f"\n📊 生成的可视化:")
print(f"   • 模型性能对比图")
print(f"   • 详细性能热图")
print(f"   • 聚类结果PCA可视化")
print(f"   • 聚类评估矩阵")
print(f"   • 增强混淆矩阵")
print(f"   • 特征重要性分析")

print(f"\n🎯 主要改进:")
print(f"   ✓ 详细的分类报告 - 每个模型对各NoC的精确表现")
print(f"   ✓ 多种聚类算法 - K-Means, GMM, 层次聚类等")
print(f"   ✓ 有监督 vs 无监督对比分析")
print(f"   ✓ 增强的可视化 - 性能热图、PCA聚类图等")
print(f"   ✓ 综合性能评估 - ARI, NMI, V-measure等指标")

print(f"\n✅ 分析完成！请查看生成的图表和报告。")
print("="*80)