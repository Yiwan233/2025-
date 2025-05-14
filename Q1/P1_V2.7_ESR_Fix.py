# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.7_ESR_Fix
版本: 2.7
日期: 2025-05-12
描述: 修正了 calculate_peak_confidence 函数中 LUS Regression 模型类型匹配问题。
     原为 'LUS Regression (fallback)'，现修正为 'LUS Regression' 以匹配JSON。
     保留了当 sr_obs > sr_max_n_minus_1 时跳过 E[SR] 计算的逻辑，
     因为这是当前设计的预期行为，此时 Expected_SR 会记录为 NaN (空白)。
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

# --- 配置与环境设置 (版本 2.7) --- # MODIFIED: Version
print("--- 脚本初始化与配置 (版本 2.7) ---") # MODIFIED: Version
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}.")

# --- 文件与目录路径定义 ---
DATA_DIR = './'
CONFIG_FILE_PATH = os.path.join(DATA_DIR, 'config_params.json')
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.7_esr_fix') # MODIFIED: Plot dir name
processed_peaks_filename = os.path.join(DATA_DIR, 'prob1_processed_peaks_v2.7.csv') # MODIFIED: Output filename
stutter_debug_log_filename = os.path.join(DATA_DIR, 'prob1_stutter_debug_log_v2.7.csv') # MODIFIED: Output filename
feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v2.7.csv') # MODIFIED: Output filename

if not os.path.exists(PLOTS_DIR):
    try: os.makedirs(PLOTS_DIR); print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir: PLOTS_DIR = DATA_DIR; print(f"警告: 创建绘图目录失败: {e_dir}.")

# --- 从JSON文件加载全局参数 (版本 2.7) ---
print(f"\n--- 从JSON文件加载全局参数 (版本 2.7) ---")
config = {}
SATURATION_THRESHOLD = 30000.0; SIZE_TOLERANCE_BP = 0.5; STUTTER_CV_HS_GLOBAL = 0.25;
GLOBAL_AT = 50.0; TRUE_ALLELE_CONFIDENCE_THRESHOLD = 0.5;
DYE_AT_VALUES = {'UNKNOWN': GLOBAL_AT}; MARKER_PARAMS = {}
try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f: config = json.load(f)
    print(f"INFO: 已成功从 '{CONFIG_FILE_PATH}' 加载配置参数。")
    global_params_json = config.get("global_parameters", {}); SATURATION_THRESHOLD = float(global_params_json.get("saturation_threshold_rfu", SATURATION_THRESHOLD)); SIZE_TOLERANCE_BP = float(global_params_json.get("size_tolerance_bp", SIZE_TOLERANCE_BP)); STUTTER_CV_HS_GLOBAL = float(global_params_json.get("stutter_cv_hs_global_n_minus_1", STUTTER_CV_HS_GLOBAL)); GLOBAL_AT = float(global_params_json.get("default_at_unknown_dye", GLOBAL_AT)); TRUE_ALLELE_CONFIDENCE_THRESHOLD = float(global_params_json.get("true_allele_confidence_threshold", TRUE_ALLELE_CONFIDENCE_THRESHOLD))
    dye_at_json = config.get("dye_specific_at", {}); DYE_AT_VALUES = {k.upper(): float(v) for k, v in dye_at_json.items()};
    if 'UNKNOWN' not in DYE_AT_VALUES: DYE_AT_VALUES['UNKNOWN'] = GLOBAL_AT
    MARKER_PARAMS = config.get("marker_specific_params", {})
    if not MARKER_PARAMS: print("警告: JSON 'marker_specific_params' 为空。Stutter评估将不准确。")
    else:
        for marker, params_val in MARKER_PARAMS.items():
            if "Dye" in params_val: MARKER_PARAMS[marker]["Dye"] = str(params_val["Dye"]).upper()
            if "n_minus_1_Stutter" not in MARKER_PARAMS[marker]: MARKER_PARAMS[marker]["n_minus_1_Stutter"] = {"SR_model_type": "N/A", "SR_m": 0.0, "SR_c": 0.0, "RS_max_k": 0.0}
except FileNotFoundError: print(f"严重错误: 配置文件 '{CONFIG_FILE_PATH}' 未找到。"); exit()
except json.JSONDecodeError as e: print(f"严重错误: 配置文件 '{CONFIG_FILE_PATH}' 格式错误: {e}。"); exit()
except Exception as e_config: print(f"严重错误: 加载配置文件错误: {e_config}。"); exit()
print("全局参数加载完成。")

# --- 中文翻译映射表定义 ---
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
CLASSIFICATION_REPORT_METRICS_MAP = {'precision': '精确率', 'recall': '召回率', 'f1-score': 'F1分数', 'support': '样本数'}
CLASSIFICATION_REPORT_AVG_MAP = {'accuracy': '准确率(整体)', 'macro avg': '宏平均', 'weighted avg': '加权平均'}


# --- 辅助函数定义 (版本 2.7) ---
print(f"\n--- 辅助函数定义 (版本 2.7) ---")
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    print(f"\n{title}:")
    df_display = df_to_print.copy();
    if col_map: df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns]
    if index_item_map: df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index]
    if index_name_map and df_display.index.name is not None: df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 120): print(df_display)

def extract_true_noc_v2_6(filename_str): # 版本号保持不变，因为逻辑不变
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def get_allele_numeric_v2_6(allele_val_str): # 版本号保持不变，因为逻辑不变
    try: return float(allele_val_str)
    except ValueError:
        if isinstance(allele_val_str, str):
            allele_upper = allele_val_str.upper()
            if allele_upper == 'X': return -1.0
            if allele_upper == 'Y': return -2.0
            if allele_upper == 'OL': return -3.0
        return np.nan

# --- 核心：峰处理与Stutter评估函数 (版本 2.7 - 修正LUS Regression匹配) ---
# stutter_debug_log_list = [] # 全局变量已在主逻辑中定义
def calculate_peak_confidence_v2_7(locus_peaks_df_input, marker_name, marker_params_dict, # MODIFIED: Function version
                                   dye_at_dict, global_at_val, sat_threshold,
                                   size_tolerance, cv_hs_n_minus_1_global_param,
                                   debug_log_list_param=None):
    locus_peaks_df = locus_peaks_df_input.copy()
    processed_peaks_output_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1']

    params = marker_params_dict.get(marker_name)
    if not params:
        default_dye = locus_peaks_df['Dye'].iloc[0].upper() if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN'
        params = {'L_repeat': 0, 'Dye': default_dye,
                  'n_minus_1_Stutter': {'SR_model_type': 'N/A', 'RS_max_k':0.0}}

    l_repeat = params.get('L_repeat', 0)
    marker_dye_from_params = str(params.get('Dye', 'UNKNOWN')).upper()
    data_dye = locus_peaks_df['Dye'].iloc[0].upper() if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN'
    final_dye_for_at = marker_dye_from_params if marker_dye_from_params != 'UNKNOWN' else data_dye
    at_val = dye_at_dict.get(final_dye_for_at, global_at_val)

    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, sat_threshold)
        if height_adj >= at_val:
            allele_numeric = get_allele_numeric_v2_6(peak_row['Allele'])
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original, 'Stutter_Score_N_Minus_1': 0.0,
            })
    if not candidate_peaks_list: return pd.DataFrame(columns=processed_peaks_output_cols)
    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index(drop=True)

    stutter_params_n_minus_1 = params.get('n_minus_1_Stutter', {})
    if l_repeat == 0 or \
       stutter_params_n_minus_1.get('SR_model_type') == 'N/A' or \
       pd.isna(stutter_params_n_minus_1.get('RS_max_k', np.nan)) or \
       stutter_params_n_minus_1.get('RS_max_k', 0.0) <= 1e-9:
        peaks_df['CTA'] = 1.0; peaks_df['Is_Stutter_Suspect'] = False
        return peaks_df[processed_peaks_output_cols]

    sr_max_n_minus_1 = stutter_params_n_minus_1.get('RS_max_k')
    cv_hs_to_use = cv_hs_n_minus_1_global_param

    temp_max_stutter_scores_for_each_peak = [0.0] * len(peaks_df)
    for i in range(len(peaks_df)): # pc_row is potential stutter
        pc_row = peaks_df.iloc[i]; max_score_this_pc_is_stutter_from_any_parent = 0.0
        for j in range(len(peaks_df)): # pp_row is potential parent
            if i == j: continue
            pp_row = peaks_df.iloc[j]
            if pc_row['Height'] >= pp_row['Height'] * 1.01 : continue # Stutter peak must be smaller
            
            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)

            score_stutter_for_this_parent_pair = 0.0
            debug_data_for_this_pair = {}
            log_this_pair_detailed = False

            # (在 for j in range(len(peaks_df)): 循环内部 )
# ...
            if is_pos_match_n_minus_1:
                log_this_pair_detailed = True
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 1e-9 else np.inf

                debug_data_for_this_pair.update({
                    'Parent_Allele_Raw': pp_row['Allele'], 
                    'Parent_Allele_Numeric': pp_row['Allele_Numeric'],
                    'Parent_Size': pp_row['Size'], 
                    'Parent_Height_Corrected': pp_row['Height'],
                    'Candidate_Stutter_Allele': pc_row['Allele'], 
                    'Candidate_Stutter_Size': pc_row['Size'],
                    'Candidate_Stutter_Height_Corrected': pc_row['Height'],
                    'Observed_SR': sr_obs, 
                    'Marker_L_Repeat': l_repeat,
                    'Marker_SR_Max_N_Minus_1': sr_max_n_minus_1
                })
                
                # --- MODIFICATION START ---
                # 从JSON获取模型参数并立即存入debug_data_for_this_pair
                # 无论 sr_obs 是否超限，这些参数都应该被记录（如果is_pos_match_n_minus_1为True）
                model_type_from_json = stutter_params_n_minus_1.get('SR_model_type', 'N/A')
                m_val_from_json = stutter_params_n_minus_1.get('SR_m', np.nan)
                c_val_from_json = stutter_params_n_minus_1.get('SR_c', np.nan)
                debug_data_for_this_pair.update({
                    'ESR_Model_Type': model_type_from_json,
                    'ESR_m': m_val_from_json,
                    'ESR_c': c_val_from_json
                })
                # --- MODIFICATION END ---

                e_sr = np.nan 
                e_hs = np.nan
                # Initialize other calculation-dependent fields that might not be set if sr_obs is high
                debug_data_for_this_pair['Assumed_CV_Hs'] = np.nan # Will be set if e_hs is calculated
                debug_data_for_this_pair['Calculated_Sigma_Hs'] = np.nan
                debug_data_for_this_pair['Z_Score'] = np.nan


                if sr_obs > sr_max_n_minus_1:
                    score_stutter_for_this_parent_pair = 0.0
                    # e_sr, e_hs 保持 np.nan
                else:
                    parent_an = pp_row['Allele_Numeric'] # 已在上面 debug_data_for_this_pair 中
                    # model_type, m_val, c_val 现在用上面定义的 _from_json 版本
                    
                    if pd.notna(parent_an) and parent_an >= 0:
                        if model_type_from_json == 'Allele Regression' or model_type_from_json == 'LUS Regression': # 确保已修正LUS Regression typo
                            if pd.notna(m_val_from_json) and pd.notna(c_val_from_json): 
                                e_sr = m_val_from_json * parent_an + c_val_from_json
                        elif model_type_from_json == 'Allele Average':
                            if pd.notna(c_val_from_json): 
                                e_sr = c_val_from_json
                    
                    if pd.notna(e_sr): e_sr = max(0.001, e_sr)
                    
                    e_hs = e_sr * pp_row['Height'] if pd.notna(e_sr) else np.nan
                    
                    current_calc_score = 0.0
                    if pd.notna(e_hs) and e_hs > 1e-6 :
                        sigma_hs = cv_hs_to_use * e_hs
                        debug_data_for_this_pair['Assumed_CV_Hs'] = cv_hs_to_use # Set it here
                        debug_data_for_this_pair['Calculated_Sigma_Hs'] = sigma_hs
                        if sigma_hs > 1e-6:
                            z_score = (pc_row['Height'] - e_hs) / sigma_hs
                            debug_data_for_this_pair['Z_Score'] = z_score
                            current_calc_score = exp(-0.5 * (z_score**2))
                            if pd.notna(e_sr) and e_sr > 1e-6 :
                                if sr_obs > e_sr * 1.8 : current_calc_score *= 0.5
                                if sr_obs > e_sr * 2.5 : current_calc_score *= 0.1
                        else: current_calc_score = 1.0 if abs(pc_row['Height'] - e_hs) < (0.001 * e_hs + 1e-6) else 0.0
                    elif pd.notna(e_hs) and pc_row['Height'] < 1e-6 : current_calc_score = 1.0
                    
                    score_stutter_for_this_parent_pair = current_calc_score

                debug_data_for_this_pair.update({'Expected_SR': e_sr, 'Expected_Stutter_Height': e_hs})
                # Assumed_CV_Hs, Calculated_Sigma_Hs, Z_Score are updated within the 'else' if e_hs path is taken.
                # If e_hs remains nan, they remain nan (their initialized value).

            if log_this_pair_detailed and debug_log_list_param is not None:
                # Ensure all potential keys are at least present with NaN if not calculated
                # This makes DataFrame creation more robust if some paths don't set all keys.
                # However, the current structure with debug_data_for_this_pair being progressively updated
                # and then spread into final_debug_entry should handle missing keys by them not appearing for that row.
                final_debug_entry = {'Marker': marker_name, **debug_data_for_this_pair,
                                     'Calculated_Stutter_Score_for_this_Pair': score_stutter_for_this_parent_pair}
                debug_log_list_param.append(final_debug_entry)
# ...

            max_score_this_pc_is_stutter_from_any_parent = max(max_score_this_pc_is_stutter_from_any_parent, score_stutter_for_this_parent_pair)
        temp_max_stutter_scores_for_each_peak[i] = max_score_this_pc_is_stutter_from_any_parent

    peaks_df['Stutter_Score_N_Minus_1'] = temp_max_stutter_scores_for_each_peak
    peaks_df['CTA'] = 1.0 - peaks_df['Stutter_Score_N_Minus_1']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    peaks_df['Is_Stutter_Suspect'] = peaks_df['Stutter_Score_N_Minus_1'] >= TRUE_ALLELE_CONFIDENCE_THRESHOLD
    return peaks_df[processed_peaks_output_cols]

# --- 主逻辑 ---
# 步骤 1: 数据加载与NoC提取
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.7) ---")
df_prob1 = None; load_successful = False
try:
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    if df_prob1.empty: raise ValueError("文件加载后为空或格式不正确。")
    load_successful = True; print(f"成功加载文件: '{file_path_prob1}'")
except Exception as e: print(f"加载文件错误: {e}"); exit()
unique_files = df_prob1['Sample File'].dropna().unique()
noc_map = {filename: extract_true_noc_v2_6(filename) for filename in unique_files}
df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)
df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
if df_prob1.empty: print("错误: 提取NoC并移除无效行后，数据为空。"); exit()
df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
print(f"NoC 提取完成。数据维度: {df_prob1.shape}")
print("--- 步骤 1 完成 ---")

# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.7) ---")
if df_prob1.empty: print("错误: df_prob1 为空。"); exit()
if not MARKER_PARAMS: print("错误: MARKER_PARAMS (从JSON加载) 为空。请检查JSON文件。"); exit()

processed_peak_data_all_samples = []
stutter_debug_log_list = [] # 全局的 stutter_debug_log_list
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    if unique_sample_files_total > 0 : progress = sample_files_processed_count / unique_sample_files_total * 100; print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total} - {progress:.1f}%)", end='\r')
    # sample_all_loci_processed_peaks = [] # This was defined inside the loop, but processed_peak_data_all_samples is extended directly
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]
        for i in range(1, 101): # Assuming max 100 allele calls per marker from wide format
            allele_val, size_val, height_val = row_marker_data.get(f'Allele {i}'), row_marker_data.get(f'Size {i}'), row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val) and float(height_val) > 0: current_locus_peaks_list.append({'Allele': allele_val, 'Size': size_val, 'Height': height_val, 'Dye': str(row_marker_data.get('Dye', 'UNKNOWN')).upper()})
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)

        processed_locus_df = calculate_peak_confidence_v2_7( # MODIFIED: Call updated function
            locus_peaks_for_filter_df, marker_name_actual,
            MARKER_PARAMS, DYE_AT_VALUES, GLOBAL_AT, SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, STUTTER_CV_HS_GLOBAL, stutter_debug_log_list # Pass the global list
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name
            processed_locus_df['Marker'] = marker_name_actual
            processed_peak_data_all_samples.append(processed_locus_df) # Append directly to the main list
    # No need for sample_all_loci_processed_peaks list if appending directly

print("\n所有样本位点处理完成。")
if not processed_peak_data_all_samples:
    print("警告: 处理所有样本后，列表 processed_peak_data_all_samples 为空 (可能所有峰都被过滤或参数问题)。")
    df_processed_peaks = pd.DataFrame() # 创建空DF以避免后续错误
else:
    df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)

print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个独立样本。")
if not df_processed_peaks.empty:
    print(f"处理后的总峰条目数 (df_processed_peaks): {len(df_processed_peaks)}")
    print_df_in_chinese(df_processed_peaks.head(), col_map=COLUMN_TRANSLATION_MAP, title="处理后峰数据示例 (df_processed_peaks)")
else:
    print("df_processed_peaks 为空，没有峰通过处理流程。")

if stutter_debug_log_list:
    df_stutter_debug_log = pd.DataFrame(stutter_debug_log_list);
    try:
        df_stutter_debug_log.to_csv(stutter_debug_log_filename, index=False, encoding='utf-8-sig');
        print(f"Stutter评估详细日志已保存到: {stutter_debug_log_filename}")
    except Exception as e: print(f"错误: 保存Stutter日志失败: {e}")
else:
    print("没有生成Stutter评估日志 (可能没有峰对进入详细的Stutter判断逻辑，或者 stutter_debug_log_list 未被正确传递/填充)。")

if not df_processed_peaks.empty:
    try:
        df_processed_peaks.to_csv(processed_peaks_filename, index=False, encoding='utf-8-sig');
        print(f"已处理的峰数据已保存到: {processed_peaks_filename}")
    except Exception as e: print(f"错误: 保存处理后的峰数据失败: {e}")
print("--- 步骤 2 完成 ---")

# --- 步骤 3: 特征工程 (版本 2.7 - 基于新的 df_processed_peaks) ---
print(f"\n--- 步骤 3: 特征工程 (版本 2.7) ---")
df_features_v2_7 = pd.DataFrame() # MODIFIED: DataFrame name for clarity

try:
    print(f"用于筛选真实等位基因的置信度阈值 (TRUE_ALLELE_CONFIDENCE_THRESHOLD): {TRUE_ALLELE_CONFIDENCE_THRESHOLD}") # Corrected variable name
    if not df_processed_peaks.empty:
        df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= TRUE_ALLELE_CONFIDENCE_THRESHOLD
        df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
        if df_effective_alleles.empty: print("警告: 应用CTA阈值后，没有剩余有效等位基因。特征将主要为0或NaN。") # MODIFIED: Warning
    else:
        df_effective_alleles = pd.DataFrame(columns= (df_processed_peaks.columns.tolist() if not df_processed_peaks.empty else ['Allele', 'Marker', 'Height']) + ['Is_Effective_Allele']) # MODIFIED: Handle truly empty df_processed_peaks
        print("警告: df_processed_peaks 为空，无法进行有效等位基因筛选。特征将主要为0或NaN。") # MODIFIED: Warning

    # Ensure all samples from original data are in the features DataFrame, even if they have no peaks/effective alleles
    all_sample_files_index_df = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')
    df_features_v2_7 = pd.DataFrame(index=all_sample_files_index_df.index) # Use original samples as index
    df_features_v2_7['NoC_True'] = all_sample_files_index_df['NoC_True'] # Add true NoC

    # Initialize feature columns to ensure they exist for all samples
    feature_cols_to_init = [
        'max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker',
        'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
        'avg_peak_height', 'std_peak_height'
    ]
    for col in feature_cols_to_init:
        df_features_v2_7[col] = 0.0 # Initialize with float 0.0 or np.nan as appropriate

    if not df_effective_alleles.empty:
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].nunique().rename('N_eff_alleles') # Use nunique for distinct alleles
        grouped_by_sample_eff = n_eff_alleles_per_locus.groupby('Sample File')

        # Calculate features only for samples that have effective alleles
        df_features_v2_7['max_allele_per_sample'] = grouped_by_sample_eff.max().reindex(df_features_v2_7.index)
        df_features_v2_7['total_alleles_per_sample'] = grouped_by_sample_eff.sum().reindex(df_features_v2_7.index) # Sum of distinct alleles per locus
        df_features_v2_7['avg_alleles_per_marker'] = grouped_by_sample_eff.mean().reindex(df_features_v2_7.index)
        df_features_v2_7['markers_gt2_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 2).sum()).reindex(df_features_v2_7.index)
        df_features_v2_7['markers_gt3_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 3).sum()).reindex(df_features_v2_7.index)
        df_features_v2_7['markers_gt4_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 4).sum()).reindex(df_features_v2_7.index)
        
        grouped_heights_eff = df_effective_alleles.groupby('Sample File')['Height']
        df_features_v2_7['avg_peak_height'] = grouped_heights_eff.mean().reindex(df_features_v2_7.index)
        df_features_v2_7['std_peak_height'] = grouped_heights_eff.std().reindex(df_features_v2_7.index)
    
    df_features_v2_7.fillna(0, inplace=True) # Fill any remaining NaNs (e.g. std for single peak sample) with 0
    df_features_v2_7.reset_index(inplace=True) # Move 'Sample File' from index to column

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features_v2_7 维度: {df_features_v2_7.shape}") # MODIFIED: DataFrame name
    print_df_in_chinese(df_features_v2_7.head(), col_map=COLUMN_TRANSLATION_MAP, title="新特征数据框 (df_features_v2_7) 前5行") # MODIFIED: DataFrame name
    
    # Save features
    try:
        df_features_v2_7.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
        print(f"特征数据已保存到: {feature_filename_prob1}")
    except Exception as e:
        print(f"错误: 保存特征数据失败: {e}")

except Exception as e_feat_eng_v2_7: # MODIFIED: Version in error message
    print(f"严重错误: 在特征工程阶段发生错误: {e_feat_eng_v2_7}")
    import traceback
    traceback.print_exc();
print("--- 步骤 3 完成 ---")

# --- 步骤 4 & 5: 模型评估 (您需要从V1.15复制并适配这里的代码，使用df_features_v2_7) ---
print(f"\n--- 步骤 4 & 5: 模型评估 (版本 2.7) ---") # MODIFIED: Version
# <<<<< 在此插入或确保 V1.15 中 Step 4 和 Step 5 的完整、可成功运行的代码, 输入为 df_features_v2_7 >>>>>
# 示例: (你需要替换成你的实际模型训练和评估代码)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# if not df_features_v2_7.empty and 'NoC_True' in df_features_v2_7.columns and len(df_features_v2_7) > 1:
#     X = df_features_v2_7.drop(columns=['Sample File', 'NoC_True'])
#     y = df_features_v2_7['NoC_True']
    
#     if len(X) > 0 and len(y) > 0 and len(y.unique()) > 1 : # Ensure there's data and more than one class for splitting
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None)

#         model = RandomForestClassifier(random_state=42)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         print("\n模型评估结果:")
#         print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
#         # ... (此处添加更完整的 classification_report 和 confusion_matrix 的中文打印)
#     else:
#         print("数据不足或类别单一，无法进行模型训练和评估。")
# else:
#     print("特征数据框为空或缺少必要列，跳过模型评估。")
print("--- 步骤 4 & 5 (框架) 完成 ---")

print(f"\n脚本 {os.path.basename(__file__)} (版本 2.7) 执行完毕。") # MODIFIED: Version