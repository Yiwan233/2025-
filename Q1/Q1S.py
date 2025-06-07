# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 模型为核心的全面参数敏感性分析脚本

版本: V4.2 (None值安全处理版)
日期: 2025-06-08
描述:
本版本修正了V4.1中因最优参数值为None而导致的绘图错误。
在绘制代表最优值的垂直线时，增加了对None值的检查，确保程序能够稳健运行。

分析参数:
  - 预处理: HEIGHT_THRESHOLD
  - 随机森林: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
"""
# =====================
# 0. 关键修正：设置Matplotlib后端
# =====================
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from time import time

# 导入您的原始脚本作为一个模块
try:
    import Q1c2
except ImportError:
    print("❌ 错误: 无法导入 'Q1c2.py'。请确保该文件与本脚本在同一目录下。")
    exit()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

# =====================
# 1. 配置与加载
# =====================
BASE_DIR = './'
DATA_FILE_PATH = os.path.join(BASE_DIR, '附件1：不同人数的STR图谱数据.csv')
MODEL_FILE_PATH = os.path.join(BASE_DIR, 'noc_optimized_random_forest_model.pkl')
PLOTS_DIR = os.path.join(BASE_DIR, 'noc_rf_optimization_sensitivity_analysis')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("=== 高密度参数敏感性分析脚本 V4.2 (None值安全处理版) ===")
print("⚠️ 注意: 本次分析点数密集，预计运行时间较长。")

# --- 加载区 ---
try:
    model_package = joblib.load(MODEL_FILE_PATH)
    optimal_rf_model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    selected_features_indices = model_package['selected_indices']
    
    df_raw = pd.read_csv(DATA_FILE_PATH, encoding='utf-8')
    df_raw['NoC_True'] = df_raw['Sample File'].apply(Q1c2.extract_noc_from_filename)
    df_raw = df_raw.dropna(subset=['NoC_True'])
    df_raw['NoC_True'] = df_raw['NoC_True'].astype(int)
    print("✅ 模型包和原始数据加载成功。")
except Exception as e:
    print(f"❌ 加载文件时发生错误: {e}")
    exit()

# 准备一份固定的、经过原始参数处理的特征数据
print("正在生成用于分析的基准特征数据...")
if not hasattr(Q1c2, 'all_features') or not Q1c2.all_features:
    print("错误：无法从 Q1c2.py 获取 all_features 列表。请确保Q1c2.py在主流程中生成了此列表。")
    exit()
    
df_features_base = pd.DataFrame(Q1c2.all_features)
df_features_base['贡献者人数'] = df_features_base['样本文件'].map(
    df_raw.groupby('Sample File')['NoC_True'].first().to_dict()
)
df_features_base = df_features_base.dropna(subset=['贡献者人数'])
X_base_raw = df_features_base[Q1c2.feature_cols].fillna(0)
y_base_encoded = label_encoder.transform(df_features_base['贡献者人数'])
X_base_scaled = scaler.transform(X_base_raw)
X_base_selected = X_base_scaled[:, selected_features_indices]
X_train_base, X_test_base, y_train_base, y_test_base = Q1c2.custom_stratified_split(X_base_selected, y_base_encoded)
print("✅ 基准特征数据准备完毕。")


# ==============================================================
# 2. 定义分析函数 (与V4.1版本相同)
# ==============================================================

def run_feature_param_sensitivity(param_name, param_value):
    """分析【预处理/特征工程参数】"""
    print(f"  分析 {param_name} = {param_value:.2f}")
    original_value = getattr(Q1c2, param_name)
    setattr(Q1c2, param_name, param_value)
    
    all_processed_peaks_sens = [p for _, g in df_raw.groupby('Sample File') if not (p := Q1c2.process_peaks_simplified(g)).empty]
    df_peaks_sens = pd.concat(all_processed_peaks_sens, ignore_index=True) if all_processed_peaks_sens else pd.DataFrame()

    all_features_sens = [Q1c2.extract_enhanced_features(sf, g) for sf, g in df_peaks_sens.groupby('Sample File') if g is not None] if not df_peaks_sens.empty else []
    
    setattr(Q1c2, param_name, original_value)
    if not all_features_sens: return None

    df_features_sens = pd.DataFrame([f for f in all_features_sens if f])
    if df_features_sens.empty: return None
    df_features_sens['贡献者人数'] = df_features_sens['样本文件'].map(df_raw.groupby('Sample File')['NoC_True'].first().to_dict())
    df_features_sens = df_features_sens.dropna(subset=['贡献者人数'])
    X_sens_raw = df_features_sens[Q1c2.feature_cols].fillna(0)
    y_sens_encoded = label_encoder.transform(df_features_sens['贡献者人数'])
    
    X_sens_scaled = scaler.transform(X_sens_raw)
    X_sens_selected = X_sens_scaled[:, selected_features_indices]
    _, X_test_sens, _, y_test_sens = Q1c2.custom_stratified_split(X_sens_selected, y_sens_encoded)
    
    if len(X_test_sens) == 0: return None
    y_pred_sens = optimal_rf_model.predict(X_test_sens)
    return balanced_accuracy_score(y_test_sens, y_pred_sens)

def run_model_param_sensitivity(param_name, param_value):
    """分析【模型超参数】"""
    if isinstance(param_value, float):
        print(f"  分析 {param_name} = {param_value:.3f}")
    else:
        print(f"  分析 {param_name} = {param_value}")
        
    params = optimal_rf_model.get_params()
    params[param_name] = param_value
    
    temp_model = RandomForestClassifier(**params)
    temp_model.fit(X_train_base, y_train_base)
    y_pred = temp_model.predict(X_test_base)
    return balanced_accuracy_score(y_test_base, y_pred)


# =====================
# 3. 执行高密度敏感性分析
# =====================
if __name__ == "__main__":
    total_start_time = time()
    results = {}
    
    param_configs = {
        'HEIGHT_THRESHOLD': {'type': 'feature', 'range': np.linspace(30, 150, 30)},
        'n_estimators': {'type': 'model', 'range': np.linspace(50, 1000, 30).astype(int)},
        'max_depth': {'type': 'model', 'range': np.linspace(5, 50, 30).astype(int)},
        'min_samples_split': {'type': 'model', 'range': np.linspace(2, 40, 30).astype(int)},
        'min_samples_leaf': {'type': 'model', 'range': np.linspace(1, 40, 30).astype(int)},
        'max_features': {'type': 'model', 'range': np.linspace(0.1, 1.0, 30)}
    }

    for name, config in param_configs.items():
        print(f"\n--- 正在分析参数: {name} ---")
        param_start_time = time()
        
        if config['type'] == 'feature':
            scores = [run_feature_param_sensitivity(name, val) for val in config['range']]
            original_val = getattr(Q1c2, name)
        else:
            scores = [run_model_param_sensitivity(name, val) for val in config['range']]
            original_val = optimal_rf_model.get_params()[name]
            
        results[name] = {'range': config['range'], 'scores': scores, 'original': original_val}
        print(f"参数 {name} 分析完成，耗时: {time() - param_start_time:.2f} 秒。")

    # =====================
    # 4. 可视化与保存结果
    # =====================
    print("\n--- 正在可视化高密度分析结果 ---")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('模型核心参数高密度敏感性分析', fontsize=24, y=0.97)
    axes = axes.flatten()
    
    plot_titles = {
        'HEIGHT_THRESHOLD': '预处理 - 峰高阈值',
        'n_estimators': '随机森林 - 决策树数量',
        'max_depth': '随机森林 - 最大深度',
        'min_samples_split': '随机森林 - 最小分裂样本数',
        'min_samples_leaf': '随机森林 - 最小叶片样本数',
        'max_features': '随机森林 - 最大特征比例'
    }

    for i, name in enumerate(param_configs.keys()):
        ax = axes[i]
        res = results[name]
        
        valid_indices = [j for j, s in enumerate(res['scores']) if s is not None]
        valid_range = [res['range'][j] for j in valid_indices]
        valid_scores = [res['scores'][j] for j in valid_indices]

        if not valid_scores:
            ax.text(0.5, 0.5, '分析失败', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        
        sns.regplot(x=valid_range, y=valid_scores, ax=ax, scatter_kws={'alpha':0.5, 's':50}, line_kws={'color':'red', 'linewidth':2.5}, lowess=True, label='平滑趋势 (LOWESS)')
        
        # =================================================================
        # ==== 关键修正：在绘图前检查 res['original'] 是否为 None ====
        # =================================================================
        original_val = res['original']
        if original_val is not None:
            # 根据值的类型选择不同的格式化方式
            if isinstance(original_val, float):
                label_text = f'原始最优值: {original_val:.2f}'
            else: # 适用于整数和None
                label_text = f'原始最优值: {original_val}'
            ax.axvline(x=original_val, color='black', linestyle='--', linewidth=2, label=label_text)
        else:
            # 如果原始值为None，我们可以在图上标注出来
             ax.text(0.05, 0.05, '原始最优值: None', transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        # =================================================================
        
        ax.set_xlabel(name, fontsize=14)
        ax.set_ylabel('测试集平衡准确率', fontsize=14)
        ax.set_title(plot_titles[name], fontsize=16, pad=15)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sensitivity_plot_path = os.path.join(PLOTS_DIR, '高密度核心参数敏感性分析.png')
    plt.savefig(sensitivity_plot_path, dpi=300, bbox_inches='tight')
    
    total_time = time() - total_start_time
    print(f"\n✅ 分析全部完成！总耗时: {total_time // 60:.0f} 分 {total_time % 60:.0f} 秒。")
    print(f"高密度敏感性分析图表已保存至: {sensitivity_plot_path}")