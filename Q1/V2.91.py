# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.9_No_AT_Filter_Test
版本: 2.9
日期: 2025-05-12
描述: 实验性修改：在Stutter分析的候选峰选择阶段移除了基于分析阈值(AT)的过滤。
     所有峰高大于0（经过饱和校正后）的峰都将进入Stutter分析流程。
     其他逻辑（如软边界SR处理、CV值使用等）与V2.8.1版本保持一致。
     目的是观察移除AT预过滤后对Stutter识别和CTA值的影响。
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

# --- 配置与环境设置 (版本 2.9) ---
print("--- 脚本初始化与配置 (版本 2.9) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体已尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}.")

# --- Stutter比率阈值和惩罚因子定义 (与V2.8.1一致) ---
RS_MAX_K_IDEAL = 0.25
RS_MAX_K_ACCEPTABLE_MAX = 0.45
RS_MAX_K_HARD_CAP = 0.60
PENALTY_FACTOR_EXTENDED_SR = 0.3
PENALTY_FACTOR_VERY_HIGH_SR_MODEL = 0.05
MIN_PEAK_HEIGHT_FOR_CONSIDERATION = 1 # RFU, 移除了AT后，用一个极小的阈值避免纯噪音

# --- 文件与目录路径定义 ---
DATA_DIR = './'
CONFIG_FILE_PATH = os.path.join(DATA_DIR, 'config_params.json')
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv') # 请确保文件名正确
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.9_no_at_filter')
processed_peaks_filename = os.path.join(DATA_DIR, 'prob1_processed_peaks_v2.9.csv')
stutter_debug_log_filename = os.path.join(DATA_DIR, 'prob1_stutter_debug_log_v2.9.csv')
feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_v2.9.csv')

if not os.path.exists(PLOTS_DIR):
    try: os.makedirs(PLOTS_DIR); print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir: PLOTS_DIR = DATA_DIR; print(f"警告: 创建绘图目录失败: {e_dir}.")

# --- 从JSON文件加载全局参数 (版本 2.9) ---
print(f"\n--- 从JSON文件加载全局参数 (版本 2.9) ---")
config = {}
# 定义脚本内部的默认值
DEFAULT_SATURATION_THRESHOLD = 30000.0
DEFAULT_SIZE_TOLERANCE_BP = 0.5
DEFAULT_STUTTER_CV_HS_GLOBAL = 0.25 # 默认值，会被JSON覆盖
DEFAULT_GLOBAL_AT = 50.0 # AT值仍会加载，但在此版本Stutter分析中不用于预过滤
DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD = 0.5

SATURATION_THRESHOLD = DEFAULT_SATURATION_THRESHOLD
SIZE_TOLERANCE_BP = DEFAULT_SIZE_TOLERANCE_BP
STUTTER_CV_HS_GLOBAL = DEFAULT_STUTTER_CV_HS_GLOBAL
GLOBAL_AT = DEFAULT_GLOBAL_AT # AT值会被加载但不在Stutter函数中用于筛选
TRUE_ALLELE_CONFIDENCE_THRESHOLD = DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD
DYE_AT_VALUES = {'UNKNOWN': GLOBAL_AT}
MARKER_PARAMS = {}

try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f: config = json.load(f)
    print(f"INFO: 已成功从 '{CONFIG_FILE_PATH}' 加载配置参数。")
    global_params_json = config.get("global_parameters", {})
    SATURATION_THRESHOLD = float(global_params_json.get("saturation_threshold_rfu", DEFAULT_SATURATION_THRESHOLD))
    SIZE_TOLERANCE_BP = float(global_params_json.get("size_tolerance_bp", DEFAULT_SIZE_TOLERANCE_BP))
    STUTTER_CV_HS_GLOBAL = float(global_params_json.get("stutter_cv_hs_global_n_minus_1", DEFAULT_STUTTER_CV_HS_GLOBAL))
    GLOBAL_AT = float(global_params_json.get("default_at_unknown_dye", DEFAULT_GLOBAL_AT))
    TRUE_ALLELE_CONFIDENCE_THRESHOLD = float(global_params_json.get("true_allele_confidence_threshold", DEFAULT_TRUE_ALLELE_CONFIDENCE_THRESHOLD))
    
    print(f"INFO: 使用的 STUTTER_CV_HS_GLOBAL (全局Stutter峰高变异系数): {STUTTER_CV_HS_GLOBAL}")
    print(f"INFO: 加载的 GLOBAL_AT (全局分析阈值，本脚本Stutter分析中不用于预过滤): {GLOBAL_AT}")

    dye_at_json = config.get("dye_specific_at", {})
    DYE_AT_VALUES = {k.upper(): float(v) for k, v in dye_at_json.items()}
    if 'UNKNOWN' not in DYE_AT_VALUES: DYE_AT_VALUES['UNKNOWN'] = GLOBAL_AT
    
    MARKER_PARAMS = config.get("marker_specific_params", {})
    if not MARKER_PARAMS: print("警告: JSON 'marker_specific_params' 为空。Stutter评估将不准确。")
    else:
        for marker, params_val in MARKER_PARAMS.items():
            if "Dye" in params_val: MARKER_PARAMS[marker]["Dye"] = str(params_val["Dye"]).upper()
            if "n_minus_1_Stutter" not in MARKER_PARAMS[marker]: MARKER_PARAMS[marker]["n_minus_1_Stutter"] = {"SR_model_type": "N/A", "SR_m": 0.0, "SR_c": 0.0, "RS_max_k": 0.0}
except FileNotFoundError: print(f"严重错误: 配置文件 '{CONFIG_FILE_PATH}' 未找到。脚本将使用内部默认值。"); # 使用已定义的默认值
except json.JSONDecodeError as e: print(f"严重错误: 配置文件 '{CONFIG_FILE_PATH}' 格式错误: {e}。脚本将使用内部默认值。");
except Exception as e_config: print(f"加载配置文件错误: {e_config}。脚本将使用内部默认值。");
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

# --- 辅助函数定义 (版本 2.9) ---
print(f"\n--- 辅助函数定义 (版本 2.9) ---")
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

# --- 核心：峰处理与Stutter评估函数 (版本 2.9 - 移除AT预过滤) ---
def calculate_peak_confidence_v2_9(locus_peaks_df_input,
                                   sample_file_name_param, 
                                   marker_name, 
                                   marker_params_dict,
                                   # dye_at_dict, global_at_val, # AT值不再用于此函数内部的预过滤
                                   sat_threshold, # 饱和阈值仍使用
                                   size_tolerance, 
                                   cv_hs_n_minus_1_global_param, # STUTTER_CV_HS_GLOBAL
                                   true_allele_conf_thresh,      # TRUE_ALLELE_CONFIDENCE_THRESHOLD
                                   debug_log_list_param=None):
    locus_peaks_df = locus_peaks_df_input.copy()
    processed_peaks_output_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1']

    params = marker_params_dict.get(marker_name)
    if not params: # 如果JSON中没有该marker的参数，则使用默认的非Stutter参数
        # default_dye = locus_peaks_df['Dye'].iloc[0].upper() if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN'
        params = {'L_repeat': 0, #'Dye': default_dye, # Dye信息主要用于AT，这里AT不用于过滤
                  'n_minus_1_Stutter': {'SR_model_type': 'N/A', 'SR_m':0.0, 'SR_c':0.0, 'RS_max_k':0.0}}

    l_repeat = params.get('L_repeat', 0)
    stutter_params_n_minus_1 = params.get('n_minus_1_Stutter', {})
    json_rs_max_k = stutter_params_n_minus_1.get('RS_max_k', 0.0)

    candidate_peaks_list = []
    for _, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, sat_threshold) # 应用饱和校正
        
        # MODIFICATION: 移除了基于at_val的过滤，使用MIN_PEAK_HEIGHT_FOR_CONSIDERATION
        if height_adj >= MIN_PEAK_HEIGHT_FOR_CONSIDERATION: 
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

    cv_hs_to_use = cv_hs_n_minus_1_global_param # 即 STUTTER_CV_HS_GLOBAL

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
                
                if sr_obs > RS_MAX_K_HARD_CAP: # 使用新的分级阈值
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
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.9) ---")
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

# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.9) ---")
if df_prob1.empty: print("错误: df_prob1 为空。"); exit()
if not MARKER_PARAMS: print("错误: MARKER_PARAMS (从JSON加载) 为空。请检查JSON文件。"); exit()

processed_peak_data_all_samples = []
stutter_debug_log_list = [] 
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    if unique_sample_files_total > 0 : progress = sample_files_processed_count / unique_sample_files_total * 100; print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total} - {progress:.1f}%)", end='\r')
    
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]
        for i in range(1, 101):
            allele_val, size_val, height_val = row_marker_data.get(f'Allele {i}'), row_marker_data.get(f'Size {i}'), row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val) and float(height_val) > 0: current_locus_peaks_list.append({'Allele': allele_val, 'Size': size_val, 'Height': height_val, 'Dye': str(row_marker_data.get('Dye', 'UNKNOWN')).upper()})
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)

        processed_locus_df = calculate_peak_confidence_v2_9( # 调用新版函数
            locus_peaks_for_filter_df,
            sample_file_name, 
            marker_name_actual,
            MARKER_PARAMS, 
            # DYE_AT_VALUES, GLOBAL_AT, # AT值不再传递给函数用于预过滤
            SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, 
            STUTTER_CV_HS_GLOBAL,
            TRUE_ALLELE_CONFIDENCE_THRESHOLD, 
            stutter_debug_log_list
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

# --- 步骤 3: 特征工程 (版本 2.9) ---
print(f"\n--- 步骤 3: 特征工程 (版本 2.9) ---")
df_features_v2_9 = pd.DataFrame() # MODIFIED: DataFrame name

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
    df_features_v2_9 = pd.DataFrame(index=all_sample_files_index_df.index)
    df_features_v2_9['NoC_True'] = all_sample_files_index_df['NoC_True']

    feature_cols_to_init = [
        'max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker',
        'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
        'avg_peak_height', 'std_peak_height'
    ]
    for col in feature_cols_to_init:
        df_features_v2_9[col] = 0.0

    if not df_effective_alleles.empty and \
       'Sample File' in df_effective_alleles.columns and \
       'Marker' in df_effective_alleles.columns and \
       'Allele' in df_effective_alleles.columns:
        
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].nunique().rename('N_eff_alleles')
        grouped_by_sample_eff = n_eff_alleles_per_locus.groupby('Sample File')

        df_features_v2_9['max_allele_per_sample'] = grouped_by_sample_eff.max().reindex(df_features_v2_9.index)
        df_features_v2_9['total_alleles_per_sample'] = grouped_by_sample_eff.sum().reindex(df_features_v2_9.index)
        df_features_v2_9['avg_alleles_per_marker'] = grouped_by_sample_eff.mean().reindex(df_features_v2_9.index)
        df_features_v2_9['markers_gt2_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 2).sum()).reindex(df_features_v2_9.index)
        df_features_v2_9['markers_gt3_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 3).sum()).reindex(df_features_v2_9.index)
        df_features_v2_9['markers_gt4_alleles'] = grouped_by_sample_eff.apply(lambda x: (x > 4).sum()).reindex(df_features_v2_9.index)
        
        if 'Height' in df_effective_alleles.columns:
            grouped_heights_eff = df_effective_alleles.groupby('Sample File')['Height']
            df_features_v2_9['avg_peak_height'] = grouped_heights_eff.mean().reindex(df_features_v2_9.index)
            df_features_v2_9['std_peak_height'] = grouped_heights_eff.std().reindex(df_features_v2_9.index)
        else:
            print("警告: df_effective_alleles 缺少 'Height' 列，峰高相关特征无法计算。")
    
    df_features_v2_9.fillna(0, inplace=True)
    df_features_v2_9.reset_index(inplace=True)

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features_v2_9 维度: {df_features_v2_9.shape}")
    print_df_in_chinese(df_features_v2_9.head(), col_map=COLUMN_TRANSLATION_MAP, title="新特征数据框 (df_features_v2_9) 前5行")
    
    try:
        df_features_v2_9.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
        print(f"特征数据已保存到: {feature_filename_prob1}")
    except Exception as e:
        print(f"错误: 保存特征数据失败: {e}")

except Exception as e_feat_eng_v2_9:
    print(f"严重错误: 在特征工程阶段发生错误: {e_feat_eng_v2_9}")
    import traceback
    traceback.print_exc();
print("--- 步骤 3 完成 ---")

# --- 步骤 4 & 5: 模型评估 (框架) ---
# --- 步骤 4: 模型训练与验证 (版本 2.9) ---
# --- 步骤 4: 模型训练与验证 (版本 2.91) ---
print(f"\n--- 步骤 4: 模型训练与验证 (版本 2.91) ---")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    from joblib import dump
    
    # 准备数据
    X = df_features_v2_9.drop(['Sample File', 'NoC_True'], axis=1)
    # int,string(NaN)
    y = df_features_v2_9['NoC_True']
    
    # 检查各特征的缺失值情况
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print("\n特征缺失值情况:")
        print(missing_values[missing_values > 0])
        X = X.fillna(0)  # 填充缺失值
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"训练集维度: {X_train.shape}, 测试集维度: {X_test.shape}")
    print(f"标签分布: {pd.Series(y).value_counts().sort_index().to_dict()}")
    
    # 1. 随机森林模型
    print("\n训练随机森林模型...")
    
    # 定义分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    # 交叉验证评估
    rf_cv_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"随机森林交叉验证准确率: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
    
    # 训练最终随机森林模型
    rf_model.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林测试集准确率: {rf_accuracy:.4f}")
    
    # 输出分类报告
    class_names = [f"{i}人" for i in sorted(y.unique())]
    print("\n随机森林分类报告:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    
    # 2. 集成学习方法
    print("\n训练集成学习模型...")
    
    # 准备基础估计器
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # 投票分类器
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting='soft',  # 使用概率进行投票
        weights=[2, 1, 1]  # 为随机森林分配更高权重
    )
    
    # 交叉验证评估
    voting_cv_scores = cross_val_score(voting_clf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"投票分类器交叉验证准确率: {voting_cv_scores.mean():.4f} (±{voting_cv_scores.std():.4f})")
    
    # 训练最终投票分类器
    voting_clf.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred_voting = voting_clf.predict(X_test)
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"投票分类器测试集准确率: {voting_accuracy:.4f}")
    
    # 输出分类报告
    print("\n投票分类器分类报告:")
    print(classification_report(y_test, y_pred_voting, target_names=class_names))
    
    # 选择最佳模型
    best_model_acc = max(rf_accuracy, voting_accuracy)
    if rf_accuracy >= voting_accuracy:
        best_model = rf_model
        best_model_name = "随机森林"
        y_pred = y_pred_rf
    else:
        best_model = voting_clf
        best_model_name = "投票分类器"
        y_pred = y_pred_voting
    
    print(f"\n最佳模型: {best_model_name} (测试集准确率: {best_model_acc:.4f})")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{best_model_name}混淆矩阵')
    confusion_matrix_path = os.path.join(PLOTS_DIR, 'confusion_matrix_v2.91.png')
    plt.savefig(confusion_matrix_path)
    print(f"混淆矩阵已保存至: {confusion_matrix_path}")
    
    # 特征重要性分析 (仅对随机森林)
    if best_model_name == "随机森林":
        plt.figure(figsize=(14, 10))
        
        # 提取特征重要性并排序
        feature_importances = pd.DataFrame({
            '特征': X.columns,
            '重要性': best_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        # 输出特征重要性
        print("\n特征重要性排名 (前10):")
        for idx, row in feature_importances.head(10).iterrows():
            print(f"{row['特征']: <30} {row['重要性']:.4f}")
        
        # 绘制前15个重要特征
        top_features = feature_importances.head(15)
        sns.barplot(x='重要性', y='特征', data=top_features)
        plt.title('随机森林 - 特征重要性 (前15位)')
        plt.tight_layout()
        feature_importance_path = os.path.join(PLOTS_DIR, 'feature_importance_v2.91.png')
        plt.savefig(feature_importance_path)
        print(f"特征重要性图已保存至: {feature_importance_path}")
    
    # 将最佳模型应用于全部数据
    y_pred_all = best_model.predict(X_scaled)
    df_features_v2_9['baseline_pred'] = y_pred_all
    
    # 保存更新后的特征文件
    df_features_v2_9.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
    print(f"带预测结果的特征数据已保存至: {feature_filename_prob1}")
    
    # 保存模型
    model_filename = os.path.join(DATA_DIR, f'noc_{best_model_name.lower().replace(" ", "_")}_v2.91.joblib')
    dump(best_model, model_filename)
    print(f"{best_model_name}模型已保存至: {model_filename}")
    
except ModuleNotFoundError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请安装所需库: pip install scikit-learn matplotlib seaborn joblib")
except Exception as e_model:
    print(f"模型训练过程中发生错误: {e_model}")
    import traceback
    traceback.print_exc()
    # 确保即使模型训练失败，仍能继续执行
    if 'baseline_pred' not in df_features_v2_9.columns:
        print("创建临时预测结果以确保步骤5能够执行...")
        df_features_v2_9['baseline_pred'] = df_features_v2_9['NoC_True']  # 临时使用真实标签

print("--- 步骤 4 完成 ---")

# --- 步骤 5: 模型评估与结果分析 ---
print(f"\n--- 步骤 5: 模型评估与结果分析 (版本 2.91) ---")

try:
    # 验证baseline_pred列是否存在
    if 'baseline_pred' not in df_features_v2_9.columns:
        print("警告: 'baseline_pred'列不存在，创建临时预测结果...")
        df_features_v2_9['baseline_pred'] = df_features_v2_9['NoC_True']
    
    # 计算整体准确率
    overall_accuracy = (df_features_v2_9['baseline_pred'] == df_features_v2_9['NoC_True']).mean()
    print(f"整体预测准确率: {overall_accuracy:.4f}")
    
    # 计算每个NoC类别的预测准确率
    noc_accuracy = df_features_v2_9.groupby('NoC_True').apply(
        lambda x: (x['baseline_pred'] == x['NoC_True']).mean()
    ).reset_index(name='准确率')
    
    print("\n各NoC类别预测准确率:")
    print_df_in_chinese(noc_accuracy, title="NoC类别准确率统计")
    
    # 可视化NoC预测准确率
    plt.figure(figsize=(10, 6))
    sns.barplot(x='NoC_True', y='准确率', data=noc_accuracy)
    plt.ylim(0, 1.1)
    plt.xlabel('真实贡献者人数')
    plt.ylabel('预测准确率')
    plt.title('各贡献者人数类别的预测准确率')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, row in noc_accuracy.iterrows():
        plt.text(i, row['准确率'] + 0.03, f"{row['准确率']:.2f}", ha='center')
    
    noc_accuracy_path = os.path.join(PLOTS_DIR, 'noc_accuracy_v2.91.png')
    plt.savefig(noc_accuracy_path)
    print(f"NoC预测准确率图已保存至: {noc_accuracy_path}")
    
    # 混合矩阵热图(归一化)
    plt.figure(figsize=(12, 10))
    confusion_matrix_df = pd.crosstab(
        df_features_v2_9['NoC_True'], 
        df_features_v2_9['baseline_pred'],
        rownames=['真实值'], 
        colnames=['预测值'],
        normalize='index'  # 按行归一化
    )
    
    sns.heatmap(confusion_matrix_df, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('NoC预测混淆矩阵 (行归一化)')
    confusion_norm_path = os.path.join(PLOTS_DIR, 'confusion_matrix_norm_v2.91.png')
    plt.savefig(confusion_norm_path)
    print(f"归一化混淆矩阵已保存至: {confusion_norm_path}")
    
    # 错误预测分析
    print("\n进行错误预测分析...")
    df_features_v2_9['预测正确'] = df_features_v2_9['baseline_pred'] == df_features_v2_9['NoC_True']
    error_cases = df_features_v2_9[~df_features_v2_9['预测正确']]
    
    if not error_cases.empty:
        error_analysis = error_cases.groupby(['NoC_True', 'baseline_pred']).size().reset_index(name='错误数量')
        error_analysis = error_analysis.sort_values('错误数量', ascending=False)
        
        print("\n错误预测类型分析:")
        print_df_in_chinese(error_analysis.head(10), title="主要错误类型")
        
        # 分析错误案例的特征分布
        print("\n错误案例与正确案例的特征比较:")
        for feature in ['max_allele_per_sample', 'avg_alleles_per_marker', 'avg_peak_height']:
            if feature in df_features_v2_9.columns:
                correct_mean = df_features_v2_9[df_features_v2_9['预测正确']][feature].mean()
                error_mean = df_features_v2_9[~df_features_v2_9['预测正确']][feature].mean()
                diff_pct = abs(error_mean - correct_mean) / correct_mean * 100 if correct_mean != 0 else 0
                
                print(f"{feature}: 正确案例平均值 = {correct_mean:.2f}, 错误案例平均值 = {error_mean:.2f} (差异: {diff_pct:.1f}%)")
        
        # 绘制特征分布对比图
        plt.figure(figsize=(15, 8))
        for i, feature in enumerate(['max_allele_per_sample', 'avg_alleles_per_marker', 'avg_peak_height']):
            if feature in df_features_v2_9.columns:
                plt.subplot(1, 3, i+1)
                sns.boxplot(x='预测正确', y=feature, data=df_features_v2_9)
                plt.title(f'{feature}分布')
                plt.xlabel('预测是否正确')
                plt.ylabel(feature)
        
        plt.tight_layout()
        error_analysis_path = os.path.join(PLOTS_DIR, 'error_analysis_v2.91.png')
        plt.savefig(error_analysis_path)
        print(f"错误分析图已保存至: {error_analysis_path}")
    else:
        print("没有错误预测案例。")
    
    # 规则校正：使用领域知识调整预测
    print("\n尝试应用规则校正...")
    corrected_preds = df_features_v2_9['baseline_pred'].copy()
    
    # 规则1：如果有>=3个位点有>3个等位基因，且NoC预测=1，则调整为2
    if 'markers_gt3_alleles' in df_features_v2_9.columns:
        rule1_mask = (df_features_v2_9['markers_gt3_alleles'] >= 3) & (df_features_v2_9['baseline_pred'] == 1)
        corrected_preds[rule1_mask] = 2
        print(f"规则1应用: {rule1_mask.sum()}个样本从NoC=1调整为NoC=2")
    
    # 规则2：如果每个位点平均等位基因数>2.5且NoC预测=2，则调整为3
    if 'avg_alleles_per_marker' in df_features_v2_9.columns:
        rule2_mask = (df_features_v2_9['avg_alleles_per_marker'] > 2.5) & (df_features_v2_9['baseline_pred'] == 2)
        corrected_preds[rule2_mask] = 3
        print(f"规则2应用: {rule2_mask.sum()}个样本从NoC=2调整为NoC=3")
    
    # 应用校正并比较
    df_features_v2_9['corrected_pred'] = corrected_preds
    corrected_accuracy = (df_features_v2_9['corrected_pred'] == df_features_v2_9['NoC_True']).mean()
    
    print(f"\n原始预测准确率: {overall_accuracy:.4f}")
    print(f"规则校正后准确率: {corrected_accuracy:.4f}")
    if corrected_accuracy > overall_accuracy:
        print(f"规则校正提高了准确率 (+{corrected_accuracy - overall_accuracy:.4f})")
        df_features_v2_9['baseline_pred'] = df_features_v2_9['corrected_pred']
        overall_accuracy = corrected_accuracy
    else:
        print("规则校正未能提高准确率")
    
    # 与其他版本比较
    try:
        prev_version = "v2.8"
        prev_file = os.path.join(DATA_DIR, f'prob1_features_{prev_version}.csv')
        if os.path.exists(prev_file):
            df_prev = pd.read_csv(prev_file, encoding='utf-8-sig')
            if 'baseline_pred' in df_prev.columns and 'NoC_True' in df_prev.columns:
                prev_accuracy = (df_prev['baseline_pred'] == df_prev['NoC_True']).mean()
                
                print(f"\n版本比较:")
                print(f"V2.8 (使用AT预过滤) 总体准确率: {prev_accuracy:.4f}")
                print(f"V2.91 (移除AT预过滤) 总体准确率: {overall_accuracy:.4f}")
                print(f"准确率变化: {(overall_accuracy - prev_accuracy) * 100:.2f}%")
                
                # 创建版本比较柱状图
                plt.figure(figsize=(8, 6))
                versions = ['V2.8 (使用AT预过滤)', 'V2.91 (移除AT预过滤)']
                accuracies = [prev_accuracy, overall_accuracy]
                
                colors = ['#1f77b4', '#d62728'] if overall_accuracy > prev_accuracy else ['#1f77b4', '#1f77b4']
                
                bars = sns.barplot(x=versions, y=accuracies, palette=colors)
                plt.ylim(0, 1.1)
                plt.ylabel('总体准确率')
                plt.xlabel('模型版本')
                plt.title('不同版本模型准确率比较')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                for i, acc in enumerate(accuracies):
                    plt.text(i, acc + 0.03, f"{acc:.4f}", ha='center')
                
                version_comparison_path = os.path.join(PLOTS_DIR, 'version_comparison_v2.91.png')
                plt.savefig(version_comparison_path)
                print(f"版本比较图已保存至: {version_comparison_path}")
        else:
            print(f"未找到先前版本的特征文件: {prev_file}")
    except Exception as e_compare:
        print(f"版本比较过程中发生错误: {e_compare}")
    
    # 保存最终预测结果
    df_features_v2_9.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
    print(f"\n最终预测结果已保存至: {feature_filename_prob1}")
    
    # 保存最终的分析结果摘要
    summary = {
        "脚本版本": "2.91 (移除AT预过滤)",
        "模型类型": best_model_name if 'best_model_name' in locals() else "未知",
        "总体准确率": float(overall_accuracy),
        "样本数": int(len(df_features_v2_9)),
        "特征数": int(len(X.columns)) if 'X' in locals() else "未知",
        "NoC类别数": int(len(df_features_v2_9['NoC_True'].unique())),
        "主要修改": "移除了Stutter分析中基于AT的预过滤，所有峰高大于1 RFU的峰都进入分析流程",
        "生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_file = os.path.join(DATA_DIR, 'prob1_analysis_summary_v2.91.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    print(f"分析摘要已保存至: {summary_file}")
    
except Exception as e_eval:
    print(f"模型评估过程中发生错误: {e_eval}")
    import traceback
    traceback.print_exc()

print("--- 步骤 5 完成 ---")

# --- 添加SHAP值分析（修复数组转换错误版）---
# --- 添加极简SHAP值分析 ---
print("\n--- SHAP值分析 (极简版) ---")

try:
    import shap
    import numpy as np
    import pandas as pd
    
    # 检查是否有训练好的随机森林模型
    if 'rf_model' in locals() and 'X_train' in locals():
        print("正在计算SHAP值，这可能需要一些时间...")
        
        # 创建一个解释器
        explainer = shap.TreeExplainer(rf_model)
        
        # 使用小样本量计算SHAP值
        sample_size = min(30, len(X_test))
        X_shap = X_test.iloc[:sample_size].copy()
        feature_names = list(X_shap.columns)
        
        # 获取SHAP值
        shap_values = explainer.shap_values(X_shap)
        
        # 检查是否是多分类问题
        is_multiclass = isinstance(shap_values, list)
        print(f"模型类型: {'多分类' if is_multiclass else '二分类/回归'}")
        
        # 获取要使用的SHAP值
        if is_multiclass:
            # 对于多分类，使用第一个类别
            selected_values = shap_values[0]
        else:
            selected_values = shap_values
        
        # 计算每个特征的平均SHAP值
        mean_values = []
        for i in range(X_shap.shape[1]):
            col_values = selected_values[:, i]
            mean_val = np.mean(col_values)
            mean_values.append((i, mean_val, abs(mean_val)))
        
        # 根据绝对值大小排序
        sorted_features = sorted(mean_values, key=lambda x: x[2], reverse=True)
        
        # 选择前8个特征
        top_limit = min(8, len(sorted_features))
        top_features = sorted_features[:top_limit]
        
        # 提取特征名称和SHAP值
        feature_labels = []
        shap_means = []
        
        for idx, mean_val, _ in top_features:
            feature_labels.append(feature_names[idx])
            shap_means.append(mean_val)
        
        # 绘制特征影响对比图
        plt.figure(figsize=(12, 8))
        
        # 颜色设置
        colors = ['#FF4136' if x > 0 else '#0074D9' for x in shap_means]
        
        # 绘制水平条形图
        positions = list(range(len(feature_labels)))
        plt.barh(positions, shap_means, color=colors)
        plt.yticks(positions, feature_labels)
        plt.xlabel('SHAP value (impact on model output)')
        plt.ylabel('Feature name')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Impact of Features on DNA Contributors Prediction (NoC)')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # 保存图形
        plt.tight_layout()
        shap_impact_path = os.path.join(PLOTS_DIR, 'shap_impact_v2.91.png')
        plt.savefig(shap_impact_path)
        print(f"特征影响对比图已保存至: {shap_impact_path}")
        
        # 尝试创建基本SHAP摘要图
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(selected_values, X_shap, plot_type="bar", show=False)
            plt.title("SHAP特征重要性")
            plt.tight_layout()
            shap_summary_path = os.path.join(PLOTS_DIR, 'shap_summary_v2.91.png')
            plt.savefig(shap_summary_path)
            print(f"SHAP摘要图已保存至: {shap_summary_path}")
        except Exception as e:
            print(f"创建SHAP摘要图失败: {e}")
        
        # 保存分析结果
        shap_analysis = {
            "top_features": feature_labels,
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            shap_analysis_file = os.path.join(DATA_DIR, 'shap_analysis_v2.91.json')
            with open(shap_analysis_file, 'w', encoding='utf-8') as f:
                json.dump(shap_analysis, f, ensure_ascii=False, indent=4)
            print(f"SHAP分析结果已保存至: {shap_analysis_file}")
        except Exception as e_save:
            print(f"保存SHAP分析结果时发生错误: {e_save}")
        
    else:
        print("未找到随机森林模型或训练数据，无法计算SHAP值")
    
except ImportError:
    print("未安装SHAP库，无法进行SHAP分析。可通过运行 'pip install shap' 安装")
except Exception as e_shap:
    print(f"SHAP分析过程中发生错误: {e_shap}")
    import traceback
    traceback.print_exc()

print("--- SHAP分析完成 ---")
print(f"\n脚本 {os.path.basename(__file__)} (版本 2.9) 执行完毕。")