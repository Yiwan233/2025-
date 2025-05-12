# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.10_Dynamic_AT
版本: 2.10
日期: 2025-05-12
描述: 实现了基于样本最高峰百分比的动态分析阈值（Dynamic AT）策略。
     - 为每个样本计算一个特定的AT值用于初始峰筛选。
     - 引入了新的全局参数来控制动态AT的计算。
     - Stutter核心计算逻辑沿用V2.8的软边界SR处理。
     - 确保 'Sample File' 列被添加到 stutter_debug_log 中。
"""
# --- 基础库与机器学习库导入 ---
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from math import ceil, exp, sqrt
import json

# --- 配置与环境设置 (版本 2.10) ---
print("--- 脚本初始化与配置 (版本 2.10) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体已尝试设置为 'SimHei'。")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}。")

# --- Stutter比率阈值和惩罚因子定义 (与V2.8一致) ---
RS_MAX_K_IDEAL = 0.25
RS_MAX_K_ACCEPTABLE_MAX = 0.45
RS_MAX_K_HARD_CAP = 0.60
PENALTY_FACTOR_EXTENDED_SR = 0.3
PENALTY_FACTOR_VERY_HIGH_SR_MODEL = 0.05

# --- 新增：动态AT计算相关参数 (这些值需要你根据数据进行优化和验证) ---
DYNAMIC_AT_PERCENTAGE = 0.10 # 例如，样本最高峰的10%作为动态AT
MINIMUM_RELIABLE_PEAK_FOR_DYNAMIC_AT_CALC_RFU = 75 # 如果样本最高峰低于此值，则不使用百分比计算
FALLBACK_AT_FOR_LOW_SIGNAL_RFU = 30 # 当最高峰过低时，使用的备用AT值
MINIMUM_DYNAMIC_AT_FLOOR_RFU = 20 # 动态AT计算结果的最低下限值
USE_JSON_AT_AS_UPPER_CAP_FOR_DYNAMIC_AT = True # 是否用JSON中的AT值作为动态AT的上限

# --- 文件与目录路径定义 ---
DATA_DIR = './'
CONFIG_FILE_PATH = os.path.join(DATA_DIR, 'config_params.json')
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.10_dynamic_at')
processed_peaks_filename = os.path.join(DATA_DIR, 'prob1_processed_peaks_v2.10.csv')
stutter_debug_log_filename = os.path.join(DATA_DIR, 'prob1_stutter_debug_log_v2.10.csv')
feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v2.10.csv')

if not os.path.exists(PLOTS_DIR):
    try: os.makedirs(PLOTS_DIR); print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir: PLOTS_DIR = DATA_DIR; print(f"警告: 创建绘图目录失败: {e_dir}.")

# --- 从JSON文件加载全局参数 (版本 2.10) ---
print(f"\n--- 从JSON文件加载全局参数 (版本 2.10) ---")
config = {}
# 定义脚本内部的默认值
DEFAULT_SATURATION_THRESHOLD = 30000.0
DEFAULT_SIZE_TOLERANCE_BP = 0.5
DEFAULT_STUTTER_CV_HS_GLOBAL = 0.35 # 使用你认为效果较好的CV值
DEFAULT_GLOBAL_AT_JSON = 50.0 # JSON中定义的全局AT，用作参考或上限
DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD = 0.5

SATURATION_THRESHOLD = DEFAULT_SATURATION_THRESHOLD
SIZE_TOLERANCE_BP = DEFAULT_SIZE_TOLERANCE_BP
STUTTER_CV_HS_GLOBAL = DEFAULT_STUTTER_CV_HS_GLOBAL
GLOBAL_AT_JSON = DEFAULT_GLOBAL_AT_JSON
TRUE_ALLELE_CONFIDENCE_THRESHOLD = DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD
DYE_AT_VALUES_JSON = {'UNKNOWN': GLOBAL_AT_JSON} # JSON中定义的染料特异性AT
MARKER_PARAMS = {}

try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f: config = json.load(f)
    print(f"INFO: 已成功从 '{CONFIG_FILE_PATH}' 加载配置参数。")
    global_params_json = config.get("global_parameters", {})
    SATURATION_THRESHOLD = float(global_params_json.get("saturation_threshold_rfu", DEFAULT_SATURATION_THRESHOLD))
    SIZE_TOLERANCE_BP = float(global_params_json.get("size_tolerance_bp", DEFAULT_SIZE_TOLERANCE_BP))
    STUTTER_CV_HS_GLOBAL = float(global_params_json.get("stutter_cv_hs_global_n_minus_1", DEFAULT_STUTTER_CV_HS_GLOBAL))
    GLOBAL_AT_JSON = float(global_params_json.get("default_at_unknown_dye", DEFAULT_GLOBAL_AT_JSON))
    TRUE_ALLELE_CONFIDENCE_THRESHOLD = float(global_params_json.get("true_allele_confidence_threshold", DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD))
    
    print(f"INFO: 使用的 STUTTER_CV_HS_GLOBAL (全局Stutter峰高变异系数): {STUTTER_CV_HS_GLOBAL}")
    print(f"INFO: JSON中定义的 GLOBAL_AT: {GLOBAL_AT_JSON}")

    dye_at_json = config.get("dye_specific_at", {})
    DYE_AT_VALUES_JSON = {k.upper(): float(v) for k, v in dye_at_json.items()}
    if 'UNKNOWN' not in DYE_AT_VALUES_JSON: DYE_AT_VALUES_JSON['UNKNOWN'] = GLOBAL_AT_JSON
    print(f"INFO: JSON中定义的 DYE_AT_VALUES: {DYE_AT_VALUES_JSON}")
    
    MARKER_PARAMS = config.get("marker_specific_params", {})
    if not MARKER_PARAMS: print("警告: JSON 'marker_specific_params' 为空。Stutter评估将不准确。")
    else:
        for marker, params_val in MARKER_PARAMS.items():
            if "Dye" in params_val: MARKER_PARAMS[marker]["Dye"] = str(params_val["Dye"]).upper()
            if "n_minus_1_Stutter" not in MARKER_PARAMS[marker]: MARKER_PARAMS[marker]["n_minus_1_Stutter"] = {"SR_model_type": "N/A", "SR_m": 0.0, "SR_c": 0.0, "RS_max_k": 0.0}
except FileNotFoundError: print(f"警告: 配置文件 '{CONFIG_FILE_PATH}' 未找到。脚本将使用内部定义的默认全局参数。");
except json.JSONDecodeError as e: print(f"警告: 配置文件 '{CONFIG_FILE_PATH}' 格式错误: {e}。脚本将使用内部定义的默认全局参数。");
except Exception as e_config: print(f"加载配置文件错误: {e_config}。脚本将使用内部定义的默认全局参数。");
print("全局参数加载完成（或使用默认值）。")

# --- 中文翻译映射表定义 (保持不变) ---
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件', 'Marker': '标记', 'Dye': '染料',
    'Allele': '等位基因', 'Allele_Numeric': '数值型等位基因',
    'Size': '片段大小(bp)', 'Height': '校正后峰高(RFU)', 'Original_Height': '原始峰高(RFU)',
    'CTA': '真实等位基因置信度', 'Is_Stutter_Suspect': '高度可疑Stutter',
    'Stutter_Score_N_Minus_1': '作为n-1 Stutter的得分',
    'NoC_True': '真实贡献人数', 'max_allele_per_sample': '样本内最大有效等位基因数',
    'total_alleles_per_sample': '样本内总有效等位基因数', 'avg_alleles_per_marker': '每标记平均有效等位基因数',
    'markers_gt2_alleles': '有效等位基因数>2的标记数', 'markers_gt3_alleles': '有效等位基因数>3的标记数',
    'markers_gt4_alleles': '有效等位基因数>4的标记数', 'avg_peak_height': '有效等位基因平均峰高',
    'std_peak_height': '有效等位基因峰高标准差', 'baseline_pred': '基线模型预测NoC'
}
DESCRIBE_INDEX_TRANSLATION_MAP = {
    'count': '计数', 'mean': '均值', 'std': '标准差', 'min': '最小值',
    '25%': '25%分位数', '50%': '中位数(50%)', '75%': '75%分位数', 'max': '最大值'
}

# --- 辅助函数定义 (版本 2.10) ---
print(f"\n--- 辅助函数定义 (版本 2.10) ---")
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    print(f"\n{title}:")
    df_display = df_to_print.copy();
    if col_map: df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns]
    if index_item_map: df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index]
    if index_name_map and df_display.index.name is not None: df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 120): print(df_display)

def extract_true_noc_v2_6(filename_str): # 逻辑不变
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def get_allele_numeric_v2_6(allele_val_str): # 逻辑不变
    try: return float(allele_val_str)
    except ValueError:
        if isinstance(allele_val_str, str):
            allele_upper = allele_val_str.upper()
            if allele_upper == 'X': return -1.0
            if allele_upper == 'Y': return -2.0
            if allele_upper == 'OL': return -3.0
        return np.nan

# --- 核心：峰处理与Stutter评估函数 (版本 2.10 - 使用传入的动态AT进行预过滤) ---
def calculate_peak_confidence_v2_10(locus_peaks_df_input,
                                   sample_file_name_param, 
                                   marker_name, 
                                   marker_params_dict,
                                   current_sample_at_val, # <--- 新增：当前样本使用的动态AT值
                                   sat_threshold,
                                   size_tolerance, 
                                   cv_hs_n_minus_1_global_param,
                                   true_allele_conf_thresh,
                                   debug_log_list_param=None):
    locus_peaks_df = locus_peaks_df_input.copy()
    processed_peaks_output_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1']

    params = marker_params_dict.get(marker_name)
    if not params:
        params = {'L_repeat': 0, 
                  'n_minus_1_Stutter': {'SR_model_type': 'N/A', 'SR_m':0.0, 'SR_c':0.0, 'RS_max_k':0.0}}

    l_repeat = params.get('L_repeat', 0)
    stutter_params_n_minus_1 = params.get('n_minus_1_Stutter', {})
    json_rs_max_k = stutter_params_n_minus_1.get('RS_max_k', 0.0) # 仍然记录JSON中的RS_max_k作为参考

    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, sat_threshold)
        
        # MODIFICATION: 使用传入的 current_sample_at_val 进行峰筛选
        if height_adj >= current_sample_at_val: 
            allele_numeric = get_allele_numeric_v2_6(peak_row['Allele'])
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original,
                'Stutter_Score_N_Minus_1': 0.0, 
            })
    if not candidate_peaks_list: return pd.DataFrame(columns=processed_peaks_output_cols)
    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index(drop=True)

    model_type_for_early_exit = stutter_params_n_minus_1.get('SR_model_type', 'N/A')
    if l_repeat == 0 or model_type_for_early_exit == 'N/A':
        peaks_df['CTA'] = 1.0
        peaks_df['Is_Stutter_Suspect'] = False
        return peaks_df[processed_peaks_output_cols]

    cv_hs_to_use = cv_hs_n_minus_1_global_param

    temp_max_stutter_scores_for_each_peak = [0.0] * len(peaks_df)
    for i in range(len(peaks_df)): 
        pc_row = peaks_df.iloc[i]
        max_score_this_pc_is_stutter_from_any_parent = 0.0
        
        for j in range(len(peaks_df)): 
            if i == j: continue
            pp_row = peaks_df.iloc[j]
            if pc_row['Height'] >= pp_row['Height'] * 1.01 : continue
            
            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)

            current_stutter_score_for_pair = 0.0
            debug_data_for_this_pair = {}
            log_this_pair_detailed = False

            if is_pos_match_n_minus_1:
                log_this_pair_detailed = True
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 1e-9 else np.inf

                debug_data_for_this_pair.update({
                    'Parent_Allele_Raw': pp_row['Allele'], 'Parent_Allele_Numeric': pp_row['Allele_Numeric'],
                    'Parent_Size': pp_row['Size'], 'Parent_Height_Corrected': pp_row['Height'],
                    'Candidate_Stutter_Allele': pc_row['Allele'], 'Candidate_Stutter_Size': pc_row['Size'],
                    'Candidate_Stutter_Height_Corrected': pc_row['Height'],
                    'Observed_SR': sr_obs, 'Marker_L_Repeat': l_repeat,
                    'Marker_SR_Max_N_Minus_1': json_rs_max_k 
                })
                
                model_type = stutter_params_n_minus_1.get('SR_model_type', 'N/A')
                m_val = stutter_params_n_minus_1.get('SR_m', np.nan)
                c_val = stutter_params_n_minus_1.get('SR_c', np.nan)
                debug_data_for_this_pair.update({'ESR_Model_Type': model_type, 'ESR_m': m_val, 'ESR_c': c_val})

                e_sr = np.nan; e_hs = np.nan
                debug_data_for_this_pair['Expected_SR'] = np.nan
                debug_data_for_this_pair['Expected_Stutter_Height'] = np.nan
                debug_data_for_this_pair['Assumed_CV_Hs'] = np.nan
                debug_data_for_this_pair['Calculated_Sigma_Hs'] = np.nan
                debug_data_for_this_pair['Z_Score'] = np.nan
                
                if sr_obs > RS_MAX_K_HARD_CAP:
                    current_stutter_score_for_pair = 0.0 
                else: 
                    parent_an = pp_row['Allele_Numeric']
                    if pd.notna(parent_an) and parent_an >= 0:
                        if model_type == 'Allele Regression' or model_type == 'LUS Regression':
                            if pd.notna(m_val) and pd.notna(c_val): e_sr = m_val * parent_an + c_val
                        elif model_type == 'Allele Average':
                            if pd.notna(c_val): e_sr = c_val
                    
                    if pd.notna(e_sr): e_sr = max(0.001, e_sr)
                    debug_data_for_this_pair['Expected_SR'] = e_sr

                    if pd.notna(e_sr):
                        e_hs = e_sr * pp_row['Height']
                        debug_data_for_this_pair['Expected_Stutter_Height'] = e_hs
                        
                        if e_hs > 1e-6 :
                            sigma_hs = cv_hs_to_use * e_hs
                            debug_data_for_this_pair['Assumed_CV_Hs'] = cv_hs_to_use
                            debug_data_for_this_pair['Calculated_Sigma_Hs'] = sigma_hs
                            
                            if sigma_hs > 1e-6:
                                z_score = (pc_row['Height'] - e_hs) / sigma_hs
                                debug_data_for_this_pair['Z_Score'] = z_score
                                base_score = exp(-0.5 * (z_score**2))
                                
                                if sr_obs <= RS_MAX_K_IDEAL:
                                    current_stutter_score_for_pair = base_score
                                elif sr_obs <= RS_MAX_K_ACCEPTABLE_MAX:
                                    current_stutter_score_for_pair = base_score * PENALTY_FACTOR_EXTENDED_SR
                                else: 
                                    current_stutter_score_for_pair = base_score * PENALTY_FACTOR_VERY_HIGH_SR_MODEL
                            else: 
                                current_stutter_score_for_pair = 1.0 if abs(pc_row['Height'] - e_hs) < (0.001 * e_hs + 1e-6) else 0.0
                        else: 
                            current_stutter_score_for_pair = 1.0 if pc_row['Height'] < 1e-6 else 0.0 
                    else: 
                        current_stutter_score_for_pair = 0.0
            
            if log_this_pair_detailed and debug_log_list_param is not None:
                final_debug_entry = {
                    'Sample File': sample_file_name_param, 
                    'Marker': marker_name, 
                    **debug_data_for_this_pair,
                    'Calculated_Stutter_Score_for_this_Pair': current_stutter_score_for_pair
                }
                debug_log_list_param.append(final_debug_entry)
            
            max_score_this_pc_is_stutter_from_any_parent = max(max_score_this_pc_is_stutter_from_any_parent, current_stutter_score_for_pair)
        
        temp_max_stutter_scores_for_each_peak[i] = max_score_this_pc_is_stutter_from_any_parent

    peaks_df['Stutter_Score_N_Minus_1'] = temp_max_stutter_scores_for_each_peak
    peaks_df['CTA'] = 1.0 - peaks_df['Stutter_Score_N_Minus_1']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    peaks_df['Is_Stutter_Suspect'] = peaks_df['Stutter_Score_N_Minus_1'] >= true_allele_conf_thresh
    return peaks_df[processed_peaks_output_cols]

# --- 主逻辑 ---
# 步骤 1: 数据加载与NoC提取 (与V2.8.1一致)
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.10) ---")
df_prob1 = None
try:
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    if df_prob1.empty: raise ValueError("文件加载后为空或格式不正确。")
    print(f"成功加载文件: '{file_path_prob1}'")
    if 'Sample File' not in df_prob1.columns:
        print(f"严重错误: 原始数据文件 '{file_path_prob1}' 缺少 'Sample File' 列。脚本将退出。")
        exit()
except FileNotFoundError: print(f"严重错误: 原始数据文件 '{file_path_prob1}' 未找到。"); exit()
except Exception as e: print(f"加载原始数据文件错误: {e}"); exit()

unique_files = df_prob1['Sample File'].dropna().unique()
noc_map = {filename: extract_true_noc_v2_6(filename) for filename in unique_files}
df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)
df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
if df_prob1.empty: print("错误: 提取NoC并移除无效行后，数据为空。"); exit()
df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
print(f"NoC 提取完成。数据维度: {df_prob1.shape}")
print("--- 步骤 1 完成 ---")

# --- 步骤 2: 峰识别与Stutter概率化评估 (引入动态AT) ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.10 - 动态AT) ---")
if df_prob1.empty: print("错误: df_prob1 为空。"); exit()
if not MARKER_PARAMS: print("错误: MARKER_PARAMS (从JSON加载) 为空。请检查JSON文件。"); exit()

processed_peak_data_all_samples = []
stutter_debug_log_list = [] 
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

# 预先提取所有样本的峰数据，以便计算每个样本的最高峰
all_samples_peak_data_for_max_h = {}
print("正在预处理所有样本以确定样本内最高峰...")
for sample_file_name_iter in unique_files:
    sample_data = df_prob1[df_prob1['Sample File'] == sample_file_name_iter]
    sample_peaks = []
    for _, locus_row in sample_data.iterrows():
        for i in range(1, 101):
            h_val = locus_row.get(f'Height {i}')
            if pd.notna(h_val):
                try:
                    sample_peaks.append(float(h_val))
                except ValueError:
                    pass
    if sample_peaks:
        all_samples_peak_data_for_max_h[sample_file_name_iter] = sample_peaks
print("样本最高峰预处理完成。")


for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    if unique_sample_files_total > 0 : progress = sample_files_processed_count / unique_sample_files_total * 100; print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total} - {progress:.1f}%)", end='\r')
    
    # --- 计算当前样本的动态AT ---
    current_sample_at = FALLBACK_AT_FOR_LOW_SIGNAL_RFU # 默认使用低信号备用AT
    
    sample_peaks_heights = all_samples_peak_data_for_max_h.get(sample_file_name, [])
    if sample_peaks_heights:
        max_height_in_sample = np.max(sample_peaks_heights) if sample_peaks_heights else 0
        
        # 获取该样本第一个位点的染料颜色作为代表（或更复杂的逻辑）
        # 这是一个简化，理想情况是动态AT也考虑染料
        first_marker_for_sample = group_data_per_sample['Marker'].iloc[0]
        marker_dye = MARKER_PARAMS.get(first_marker_for_sample, {}).get('Dye', 'UNKNOWN')
        json_at_for_dye = DYE_AT_VALUES_JSON.get(marker_dye, GLOBAL_AT_JSON)

        if max_height_in_sample >= MINIMUM_RELIABLE_PEAK_FOR_DYNAMIC_AT_CALC_RFU:
            dynamic_at_calculated = max_height_in_sample * DYNAMIC_AT_PERCENTAGE
            current_sample_at = max(dynamic_at_calculated, MINIMUM_DYNAMIC_AT_FLOOR_RFU)
            if USE_JSON_AT_AS_UPPER_CAP_FOR_DYNAMIC_AT:
                current_sample_at = min(current_sample_at, json_at_for_dye)
        else: # 最高峰太低，使用备用AT，但也受JSON AT上限约束 (如果启用)
            current_sample_at = FALLBACK_AT_FOR_LOW_SIGNAL_RFU
            if USE_JSON_AT_AS_UPPER_CAP_FOR_DYNAMIC_AT:
                current_sample_at = min(current_sample_at, json_at_for_dye)
    
    # 确保 current_sample_at 不会低于一个非常小的值，例如我们之前用的 MIN_PEAK_HEIGHT_FOR_CONSIDERATION
    current_sample_at = max(current_sample_at, 1.0) # 确保至少为1 RFU
    # print(f"样本 {sample_file_name} 使用的动态AT: {current_sample_at:.2f} RFU") # 调试信息
    # --- 动态AT计算结束 ---

    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]
        for i in range(1, 101):
            allele_val, size_val, height_val = row_marker_data.get(f'Allele {i}'), row_marker_data.get(f'Size {i}'), row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val) and float(height_val) > 0: current_locus_peaks_list.append({'Allele': allele_val, 'Size': size_val, 'Height': height_val, 'Dye': str(row_marker_data.get('Dye', 'UNKNOWN')).upper()})
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)

        processed_locus_df = calculate_peak_confidence_v2_10( # 调用新版函数
            locus_peaks_for_filter_df,
            sample_file_name, 
            marker_name_actual,
            MARKER_PARAMS, 
            current_sample_at_val=current_sample_at, # <--- 传递计算好的动态AT
            sat_threshold=SATURATION_THRESHOLD,
            size_tolerance=SIZE_TOLERANCE_BP, 
            cv_hs_n_minus_1_global_param=STUTTER_CV_HS_GLOBAL,
            true_allele_conf_thresh=TRUE_ALLELE_CONFIDENCE_THRESHOLD, 
            debug_log_list_param=stutter_debug_log_list
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name 
            processed_locus_df['Marker'] = marker_name_actual
            processed_peak_data_all_samples.append(processed_locus_df)

print("\n所有样本位点处理完成。")
if not processed_peak_data_all_samples:
    print("警告: 处理所有样本后，列表 processed_peak_data_all_samples 为空。")
    df_processed_peaks = pd.DataFrame()
else:
    df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)

print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个独立样本。")
if not df_processed_peaks.empty:
    print(f"处理后的总峰条目数 (df_processed_peaks): {len(df_processed_peaks)}")
    print_df_in_chinese(df_processed_peaks.head(), col_map=COLUMN_TRANSLATION_MAP, title="处理后峰数据示例 (df_processed_peaks)")
else:
    print("df_processed_peaks 为空，没有峰通过处理流程。")

if stutter_debug_log_list:
    df_stutter_debug_log = pd.DataFrame(stutter_debug_log_list)
    if not df_stutter_debug_log.empty:
        try:
            df_stutter_debug_log.to_csv(stutter_debug_log_filename, index=False, encoding='utf-8-sig');
            print(f"Stutter评估详细日志已保存到: {stutter_debug_log_filename}")
        except Exception as e: print(f"错误: 保存Stutter日志失败: {e}")
    else:
        print("Stutter评估日志列表为空，未保存日志文件。")
else:
    print("没有生成Stutter评估日志。")

if not df_processed_peaks.empty:
    try:
        df_processed_peaks.to_csv(processed_peaks_filename, index=False, encoding='utf-8-sig');
        print(f"已处理的峰数据已保存到: {processed_peaks_filename}")
    except Exception as e: print(f"错误: 保存处理后的峰数据失败: {e}")
print("--- 步骤 2 完成 ---")

# --- 步骤 3: 特征工程 (版本 2.10) ---
print(f"\n--- 步骤 3: 特征工程 (版本 2.10) ---")
df_features_v2_10 = pd.DataFrame() # MODIFIED: DataFrame name

try:
    print(f"用于筛选真实等位基因的置信度阈值 (TRUE_ALLELE_CONFIDENCE_THRESHOLD): {TRUE_ALLELE_CONFIDENCE_THRESHOLD}")
    if not df_processed_peaks.empty and 'CTA' in df_processed_peaks.columns and \
       'Sample File' in df_processed_peaks.columns and 'Marker' in df_processed_peaks.columns and \
       'Allele' in df_processed_peaks.columns and 'Height' in df_processed_peaks.columns:
        df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= TRUE_ALLELE_CONFIDENCE_THRESHOLD
        df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
        if df_effective_alleles.empty: print("警告: 应用CTA阈值后，没有剩余有效等位基因。特征将主要为0或NaN。")
    else:
        print("警告: df_processed_peaks 为空或缺少必要列，无法进行有效等位基因筛选。")
        cols_for_empty_effective = ['Allele', 'Marker', 'Height', 'Sample File', 'Is_Effective_Allele']
        base_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1', 'Sample File', 'Marker']
        available_cols = [col for col in base_cols if col in df_processed_peaks.columns] if not df_processed_peaks.empty else ['Allele', 'Marker', 'Height', 'Sample File']
        df_effective_alleles = pd.DataFrame(columns= available_cols + ['Is_Effective_Allele'])

    all_sample_files_index_df = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')
    df_features_v2_10 = pd.DataFrame(index=all_sample_files_index_df.index)
    df_features_v2_10['NoC_True'] = all_sample_files_index_df['NoC_True']

    feature_cols_to_init = [
        'max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker',
        'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
        'avg_peak_height', 'std_peak_height'
    ]
    for col in feature_cols_to_init:
        df_features_v2_10[col] = 0.0

    if not df_effective_alleles.empty and \
       'Sample File' in df_effective_alleles.columns and \
       'Marker' in df_effective_alleles.columns and \
       'Allele' in df_effective_alleles.columns:
        
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].nunique().rename('N_eff_alleles')
        grouped_by_sample_eff = n_eff_alleles_per_locus.groupby('Sample File')

        df_features_v2_10['max_allele_per_sample'] = grouped_by_sample_eff.max().reindex(df_features_v2_10.index)
        df_features_v2_10['total_alleles_per_sample'] = grouped_by_sample_eff.sum().reindex(df_features_v2_10.index)
        df_features_v2_10['avg_alleles_per_marker'] = grouped_by_sample_eff.mean().reindex(df_features_v2_10.index)
        df_features_v2_10['markers_gt2_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 2).sum()).reindex(df_features_v2_10.index)
        df_features_v2_10['markers_gt3_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 3).sum()).reindex(df_features_v2_10.index)
        df_features_v2_10['markers_gt4_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 4).sum()).reindex(df_features_v2_10.index)
        
        if 'Height' in df_effective_alleles.columns:
            grouped_heights_eff = df_effective_alleles.groupby('Sample File')['Height']
            df_features_v2_10['avg_peak_height'] = grouped_heights_eff.mean().reindex(df_features_v2_10.index)
            df_features_v2_10['std_peak_height'] = grouped_heights_eff.std().reindex(df_features_v2_10.index)
        else:
            print("警告: df_effective_alleles 缺少 'Height' 列，峰高相关特征无法计算。")
    
    df_features_v2_10.fillna(0, inplace=True)
    df_features_v2_10.reset_index(inplace=True)

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features_v2_10 维度: {df_features_v2_10.shape}")
    print_df_in_chinese(df_features_v2_10.head(), col_map=COLUMN_TRANSLATION_MAP, title="新特征数据框 (df_features_v2_10) 前5行")
    
    try:
        df_features_v2_10.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
        print(f"特征数据已保存到: {feature_filename_prob1}")
    except Exception as e:
        print(f"错误: 保存特征数据失败: {e}")

except Exception as e_feat_eng_v2_10:
    print(f"严重错误: 在特征工程阶段发生错误: {e_feat_eng_v2_10}")
    import traceback
    traceback.print_exc();
print("--- 步骤 3 完成 ---")

# --- 步骤 4 & 5: 模型评估 (框架) ---
print(f"\n--- 步骤 4 & 5: 模型评估 (版本 2.10) ---")
# <<<<< 在此插入或确保你的模型训练和评估代码, 输入为 df_features_v2_10 >>>>>
print("--- 步骤 4 & 5 (框架) 完成 ---")

print(f"\n脚本 {os.path.basename(__file__)} (版本 2.10) 执行完毕。")
