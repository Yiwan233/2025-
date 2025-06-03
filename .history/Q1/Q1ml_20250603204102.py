# -*- coding: utf-8 -*-
"""
简化版 法医DNA分析 - 问题1：贡献者人数识别

版本: 3.0 (简化版)
描述: 移除复杂的Stutter判断逻辑，专注于特征工程和机器学习预测
主要改进:
1. 简化峰处理 - 只保留基本的阈值过滤
2. 增强特征工程 - 基于文档中的V5特征方案
3. 测试多种机器学习算法
4. 添加模型集成和超参数优化
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# 可选的高级算法 - 如果导入失败则跳过
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost不可用，将跳过")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM不可用，将跳过")
    LIGHTGBM_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 文件路径
DATA_DIR = './'
file_path = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'q1_plots_v3')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

print("=== 简化版 Q1 代码开始执行 ===")

# =====================
# 1. 数据加载与预处理
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
# 2. 简化的峰处理
# =====================
print("\n2. 简化的峰处理...")

def process_peaks_simplified(sample_data, height_threshold=50, saturation_threshold=30000):
    """
    简化的峰处理函数
    只保留基本的阈值过滤和饱和校正
    """
    processed_data = []
    
    for _, sample_row in sample_data.iterrows():
        sample_file = sample_row['Sample File']
        marker = sample_row['Marker']
        
        # 提取所有峰
        peaks = []
        for i in range(1, 101):  # 假设最多100个峰
            allele = sample_row.get(f'Allele {i}')
            size = sample_row.get(f'Size {i}')
            height = sample_row.get(f'Height {i}')
            
            if pd.notna(allele) and pd.notna(size) and pd.notna(height):
                # 饱和校正
                corrected_height = min(float(height), saturation_threshold)
                
                # 应用阈值过滤
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
# 3. 增强特征工程 (基于V5方案)
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
    # 按位点统计等位基因数
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
        # 峰高分箱后的熵
        hist, _ = np.histogram(np.log(all_heights + 1), bins=min(10, total_peaks))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        features['peak_height_entropy'] = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        
        # 高峰比例（前20%的峰）
        height_threshold_80 = np.percentile(all_heights, 80)
        features['high_peaks_ratio'] = (all_heights >= height_threshold_80).mean()
    else:
        features['peak_height_entropy'] = 0
        features['high_peaks_ratio'] = 0
    
    # C. 峰高比(PHR)相关特征 - 简化版
    phr_values = []
    for marker, marker_group in sample_peaks.groupby('Marker'):
        if len(marker_group) == 2:  # 只计算有两个峰的位点
            heights = marker_group['Height'].values
            phr = min(heights) / max(heights)
            phr_values.append(phr)
    
    if phr_values:
        features['avg_phr'] = np.mean(phr_values)
        features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
        features['min_phr'] = np.min(phr_values)
        features['median_phr'] = np.median(phr_values)
        features['num_loci_with_phr'] = len(phr_values)
        # 不平衡位点比例
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
    
    # F. 大小相关特征 (DNA降解指标)
    if total_peaks > 1:
        features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1] if len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1 else 0
        
        # 线性回归斜率
        if len(np.unique(all_sizes)) > 1:
            slope, _, _, _, _ = stats.linregress(all_sizes, all_heights)
            features['height_size_slope'] = slope
        else:
            features['height_size_slope'] = 0
            
        # 大小分布特征
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
    
    # H. 新增复合特征
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
# 4. 多种机器学习算法测试
# =====================
print("\n4. 多种机器学习算法测试...")

# 准备数据
feature_cols = [col for col in df_features.columns if col not in ['Sample File', 'NoC_True']]
X = df_features[feature_cols].fillna(0)
y = df_features['NoC_True']

print(f"原始NoC标签: {sorted(y.unique())}")

# 为XGBoost重新编码标签 (从0开始)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_original = y.copy()  # 保存原始标签用于后续显示

print(f"编码后标签: {sorted(np.unique(y_encoded))} (对应原始: {sorted(y.unique())})")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# 划分数据集 - 使用编码后的标签
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 同时保存原始标签用于显示
_, _, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_original, test_size=0.3, random_state=42, stratify=y_encoded
)

# 交叉验证设置 - 使用编码后的标签
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

# 添加可选的高级算法
if XGBOOST_AVAILABLE:
    algorithms['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

if LIGHTGBM_AVAILABLE:
    algorithms['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)

# 测试所有算法
results = {}
print("测试各种算法...")

for name, clf in algorithms.items():
    try:
        # 交叉验证 - 使用编码后的标签
        cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        # 训练并测试
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': test_acc,
            'model': clf
        }
        
        print(f"{name:20} CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} Test: {test_acc:.4f}")
        
    except Exception as e:
        print(f"{name} 失败: {e}")

# =====================
# 5. 超参数优化最佳模型
# =====================
print("\n5. 超参数优化...")

# 选择最佳算法进行超参数优化
best_algo = max(results.keys(), key=lambda x: results[x]['cv_mean'])
print(f"最佳基础算法: {best_algo}")

# 为最佳算法定义超参数网格
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    },
    'Extra Trees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
}

# 添加可选算法的参数网格
if XGBOOST_AVAILABLE:
    param_grids['XGBoost'] = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

if LIGHTGBM_AVAILABLE:
    param_grids['LightGBM'] = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, -1],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    }

if best_algo in param_grids:
    print(f"对 {best_algo} 进行网格搜索...")
    
    # 创建新的模型实例
    if best_algo == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    elif best_algo == 'XGBoost' and XGBOOST_AVAILABLE:
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    elif best_algo == 'LightGBM' and LIGHTGBM_AVAILABLE:
        base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    elif best_algo == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
    elif best_algo == 'Extra Trees':
        base_model = ExtraTreesClassifier(random_state=42, class_weight='balanced')
    else:
        print(f"没有为 {best_algo} 定义超参数优化，跳过...")
        base_model = None
    
    if base_model is not None:
        # 网格搜索
        grid_search = GridSearchCV(
            base_model, 
            param_grids[best_algo],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 更新最佳模型
        best_model = grid_search.best_estimator_
        results[best_algo]['model'] = best_model
        results[best_algo]['best_params'] = grid_search.best_params_
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳CV分数: {grid_search.best_score_:.4f}")
    else:
        print("跳过超参数优化")

# =====================
# 6. 模型集成
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
    weights=[3, 2, 1]  # 给最佳模型更高权重
)

# 训练集成模型 - 使用编码后的标签
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"集成模型测试准确率: {ensemble_acc:.4f}")

# 更新结果
results['Ensemble'] = {
    'cv_mean': cross_val_score(ensemble_model, X_scaled, y_encoded, cv=cv, scoring='accuracy').mean(),
    'test_acc': ensemble_acc,
    'model': ensemble_model
}

# =====================
# 7. 结果分析与可视化
# =====================
print("\n7. 结果分析与可视化...")

# 性能比较图
plt.figure(figsize=(14, 8))
model_names = list(results.keys())
cv_scores = [results[name]['cv_mean'] for name in model_names]
test_scores = [results[name]['test_acc'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)

plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('各模型性能比较')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
plt.close()

# 最佳模型混淆矩阵
best_final_model = max(results.keys(), key=lambda x: results[x]['test_acc'])
best_predictions = results[best_final_model]['model'].predict(X_test)

# 将编码后的标签转换回原始标签用于显示
y_test_display = label_encoder.inverse_transform(y_test)
best_predictions_display = label_encoder.inverse_transform(best_predictions)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_display, best_predictions_display)
class_names = [str(x) for x in sorted(label_encoder.classes_)]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names)
plt.title(f'{best_final_model} 混淆矩阵')
plt.ylabel('真实标签 (NoC)')
plt.xlabel('预测标签 (NoC)')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_best.png'), dpi=300)
plt.close()

# 特征重要性
if hasattr(results[best_final_model]['model'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': results[best_final_model]['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'{best_final_model} 特征重要性 (Top 15)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print("\nTop 10 重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")

# =====================
# 8. 保存结果
# =====================
print("\n8. 保存结果...")

# 保存特征数据
df_features.to_csv(os.path.join(DATA_DIR, 'q1_features_v3.csv'), index=False, encoding='utf-8-sig')

# 应用最佳模型到所有数据
best_model_final = results[best_final_model]['model']
y_pred_all_encoded = best_model_final.predict(X_scaled)
# 转换回原始标签
y_pred_all = label_encoder.inverse_transform(y_pred_all_encoded)
df_features['predicted_noc'] = y_pred_all

# 保存预测结果
df_features.to_csv(os.path.join(DATA_DIR, 'q1_predictions_v3.csv'), index=False, encoding='utf-8-sig')

# 保存模型性能总结
summary = {
    'best_model': best_final_model,
    'best_test_accuracy': float(results[best_final_model]['test_acc']),
    'all_results': {name: {'cv_mean': float(result['cv_mean']), 
                          'test_acc': float(result['test_acc'])} 
                   for name, result in results.items()},
    'feature_count': len(feature_cols),
    'sample_count': len(df_features)
}

with open(os.path.join(DATA_DIR, 'q1_summary_v3.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 最终报告
print(f"\n=== 最终结果 ===")
print(f"最佳模型: {best_final_model}")
print(f"测试集准确率: {results[best_final_model]['test_acc']:.4f}")
print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(df_features)}")
print(f"NoC分布: {y_original.value_counts().sort_index().to_dict()}")

print(f"\n文件已保存到:")
print(f"  - 特征数据: q1_features_v3.csv")
print(f"  - 预测结果: q1_predictions_v3.csv") 
print(f"  - 性能总结: q1_summary_v3.json")
print(f"  - 图表目录: {PLOTS_DIR}")

print("\n=== 简化版 Q1 代码执行完成 ===")