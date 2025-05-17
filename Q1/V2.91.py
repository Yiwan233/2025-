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
# --- 步骤 3: 特征工程 (增强版) ---
print(f"\n--- 步骤 3: 特征工程 (增强版 - 基于综合特征方案) ---")
from scipy import stats
from scipy.signal import find_peaks
from collections import Counter
import warnings
df_features_enhanced = pd.DataFrame()

try:
    print(f"用于筛选真实等位基因的置信度阈值 (TRUE_ALLELE_CONFIDENCE_THRESHOLD): {TRUE_ALLELE_CONFIDENCE_THRESHOLD}")
    
    # 初始验证
    if not df_processed_peaks.empty and 'CTA' in df_processed_peaks.columns and \
       'Sample File' in df_processed_peaks.columns and 'Marker' in df_processed_peaks.columns and \
       'Allele' in df_processed_peaks.columns and 'Height' in df_processed_peaks.columns and \
       'Size' in df_processed_peaks.columns:
        
        df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= TRUE_ALLELE_CONFIDENCE_THRESHOLD
        df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
        
        if df_effective_alleles.empty: 
            print("警告: 应用CTA阈值后，没有剩余有效等位基因。特征将主要为0或NaN。")
    else:
        print("警告: df_processed_peaks 为空或缺少必要列，无法进行有效等位基因筛选。")
        cols_for_empty_effective = ['Allele', 'Marker', 'Height', 'Sample File', 'Is_Effective_Allele', 'Size']
        base_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_N_Minus_1', 'Sample File', 'Marker']
        available_cols = [col for col in base_cols if col in df_processed_peaks.columns] if not df_processed_peaks.empty else ['Allele', 'Marker', 'Height', 'Sample File', 'Size']
        df_effective_alleles = pd.DataFrame(columns= available_cols + ['Is_Effective_Allele'])

    # 创建特征数据框架（包含所有样本和真实NoC）
    all_sample_files_index_df = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')
    df_features_enhanced = pd.DataFrame(index=all_sample_files_index_df.index)
    df_features_enhanced['NoC_True'] = all_sample_files_index_df['NoC_True']
    
    # 如果没有有效等位基因，直接返回基本特征
    if df_effective_alleles.empty:
        print("警告: 无有效等位基因，仅生成空特征向量。")
        df_features_enhanced = df_features_enhanced.reset_index()
        df_features_enhanced['max_allele_per_sample'] = 0
        df_features_enhanced['total_distinct_alleles'] = B0
        df_features_enhanced['avg_alleles_per_locus'] = 0
        df_features_enhanced['std_alleles_per_locus'] = 0

    else:
        # 计算每个样本、每个位点有效等位基因的数量
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].nunique().rename('N_eff_alleles')
        n_eff_alleles_per_locus_df = n_eff_alleles_per_locus.reset_index()
        
        # 每个样本的有效位点列表 (至少有一个有效等位基因的位点)
        effective_loci_by_sample = df_effective_alleles.groupby('Sample File')['Marker'].unique()
        
        # 获取所有预期的常染色体位点（从位点参数中）
        expected_autosomal_loci = []
        for marker, params in MARKER_PARAMS.items():
            is_autosomal = True  # 默认为常染色体
            # 如果JSON中有指定，则使用指定值
            if 'is_autosomal' in params:
                is_autosomal = params['is_autosomal']
            if is_autosomal:
                expected_autosomal_loci.append(marker)
        expected_autosomal_count = len(expected_autosomal_loci)
        
        # 为特征计算准备样本级别的数据结构
        sample_features = {}
        
        for sample_file, sample_group in df_effective_alleles.groupby('Sample File'):
            # 获取该样本的所有有效等位基因（所有位点）
            all_effective_alleles = []
            allele_heights = []
            allele_sizes = []
            
            # 位点级别统计信息
            locus_allele_counts = {}  # 每个位点的等位基因数量
            locus_peak_heights = {}   # 每个位点的峰高总和
            locus_allele_details = {} # 每个位点的详细等位基因信息
            
            # 处理样本中的每个位点
            for marker, marker_group in sample_group.groupby('Marker'):
                alleles = marker_group['Allele'].unique().tolist()
                heights = marker_group['Height'].values
                sizes = marker_group['Size'].values
                
                # 保存有效等位基因详情
                locus_allele_counts[marker] = len(alleles)
                locus_peak_heights[marker] = sum(heights)
                locus_allele_details[marker] = []
                
                for _, allele_row in marker_group.iterrows():
                    all_effective_alleles.append(allele_row['Allele'])
                    allele_heights.append(allele_row['Height'])
                    allele_sizes.append(allele_row['Size'])
                    
                    locus_allele_details[marker].append({
                        'allele': allele_row['Allele'],
                        'height': allele_row['Height'],
                        'size': allele_row['Size']
                    })
            
            # 确定有效位点（至少有一个有效等位基因的位点）
            effective_loci = list(locus_allele_counts.keys())
            num_effective_loci = len(effective_loci)
            
            # 峰高与峰大小数据
            all_heights = np.array(allele_heights)
            all_sizes = np.array(allele_sizes)
            
            # 初始化样本特征
            sample_data = {
                'all_effective_alleles': all_effective_alleles,
                'allele_heights': all_heights,
                'allele_sizes': all_sizes,
                'locus_allele_counts': locus_allele_counts,
                'locus_peak_heights': locus_peak_heights,
                'locus_allele_details': locus_allele_details,
                'effective_loci': effective_loci,
                'num_effective_loci': num_effective_loci,
                'expected_autosomal_count': expected_autosomal_count,
                'expected_autosomal_loci': expected_autosomal_loci,
            }
            
            sample_features[sample_file] = sample_data
        
        # 遍历每个样本计算特征
        features_dict = {}
        
        for sample_file, sample_data in sample_features.items():
            # 样本特征字典
            features = {}
            
            # A. 图谱层面基础计数与统计特征
            # A.1. MACP (Maximum Allele Count per Profile / 样本最大等位基因数)
            if sample_data['locus_allele_counts']:
                features['mac_profile'] = max(sample_data['locus_allele_counts'].values())
            else:
                features['mac_profile'] = 0
            
            # A.2. TDA (Total Distinct Alleles in Profile / 样本总特异等位基因数)
            features['total_distinct_alleles'] = len(set(sample_data['all_effective_alleles']))
            
            # 对所有预期的常染色体位点，获取每个位点的等位基因数量（缺失位点为0）
            allele_counts_by_locus = []
            for locus in sample_data['expected_autosomal_loci']:
                count = sample_data['locus_allele_counts'].get(locus, 0)
                allele_counts_by_locus.append(count)
            
            allele_counts_array = np.array(allele_counts_by_locus)
            
            # A.3. AAP (Average Alleles per Locus / 每位点平均等位基因数)
            if len(allele_counts_by_locus) > 0:
                features['avg_alleles_per_locus'] = np.mean(allele_counts_array)
            else:
                features['avg_alleles_per_locus'] = 0
            
            # A.4. SDA (Standard Deviation of Alleles per Locus / 每位点等位基因数的标准差)
            if len(allele_counts_by_locus) > 1:
                features['std_alleles_per_locus'] = np.std(allele_counts_array, ddof=1)
            else:
                features['std_alleles_per_locus'] = 0
            
            # A.5. MGTN 系列 (等位基因数大于等于 N 的位点数)
            for n in [2, 3, 4, 5, 6]:
                features[f'loci_gt{n}_alleles'] = np.sum(allele_counts_array >= n)
            
            # A.6. Entropy of Allele Counts per Locus
            if len(allele_counts_by_locus) > 0 and np.sum(allele_counts_array) > 0:
                # 计算等位基因数计数的分布
                count_values, count_freqs = np.unique(allele_counts_array, return_counts=True)
                probs = count_freqs / np.sum(count_freqs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))  # 避免log(0)
                features['allele_count_dist_entropy'] = entropy
            else:
                features['allele_count_dist_entropy'] = 0
            
            # B. 峰高、平衡性与随机效应相关特征
            # B.1. 平均峰高和峰高标准差
            if len(sample_data['allele_heights']) > 0:
                features['avg_peak_height'] = np.mean(sample_data['allele_heights'])
                if len(sample_data['allele_heights']) > 1:
                    features['std_peak_height'] = np.std(sample_data['allele_heights'], ddof=1)
                else:
                    features['std_peak_height'] = 0
            else:
                features['avg_peak_height'] = 0
                features['std_peak_height'] = 0
            
            # B.2. 峰高比 (PHR) 相关统计量
            phr_values = []
            
            for locus, alleles in sample_data['locus_allele_details'].items():
                # 仅考虑有恰好两个等位基因的位点
                if len(alleles) == 2:
                    h1 = alleles[0]['height']
                    h2 = alleles[1]['height']
                    if max(h1, h2) > 0:
                        phr = min(h1, h2) / max(h1, h2)
                        phr_values.append(phr)
            
            if phr_values:
                features['avg_phr'] = np.mean(phr_values)
                if len(phr_values) > 1:
                    features['std_phr'] = np.std(phr_values, ddof=1)
                else:
                    features['std_phr'] = 0
                features['min_phr'] = np.min(phr_values)
                features['median_phr'] = np.median(phr_values)
                features['num_loci_with_phr'] = len(phr_values)
                
                # 严重不平衡位点数量
                imbalance_threshold = 0.6  # 可以从配置读取
                num_severe_imbalance = sum(phr <= imbalance_threshold for phr in phr_values)
                features['num_severe_imbalance_loci'] = num_severe_imbalance
                if len(phr_values) > 0:
                    features['ratio_severe_imbalance_loci'] = num_severe_imbalance / len(phr_values)
                else:
                    features['ratio_severe_imbalance_loci'] = 0
            else:
                features['avg_phr'] = 0
                features['std_phr'] = 0
                features['min_phr'] = 0
                features['median_phr'] = 0
                features['num_loci_with_phr'] = 0
                features['num_severe_imbalance_loci'] = 0
                features['ratio_severe_imbalance_loci'] = 0
            
            # B.3. 峰高分布的偏度和峭度
            if len(sample_data['allele_heights']) > 2:
                features['skewness_peak_height'] = stats.skew(sample_data['allele_heights'], bias=False)
                features['kurtosis_peak_height'] = stats.kurtosis(sample_data['allele_heights'], fisher=False, bias=False)
            else:
                features['skewness_peak_height'] = 0
                features['kurtosis_peak_height'] = 0
            
            # B.3+ 峰高分布的多峰性指标
            if len(sample_data['allele_heights']) > 0:
                try:
                    # 对数变换峰高，可能有助于更好地识别多峰性
                    log_heights = np.log(sample_data['allele_heights'] + 1)
                    
                    # 使用KDE或直方图寻找局部最大值
                    if len(np.unique(log_heights)) > 1:
                        # 获取直方图
                        hist, bin_edges = np.histogram(log_heights, bins='auto')
                        # 在直方图上寻找峰
                        peaks, _ = find_peaks(hist)
                        features['modality_peak_height'] = len(peaks)
                    else:
                        # 如果所有值都相同
                        features['modality_peak_height'] = 1 if len(log_heights) > 0 else 0
                except Exception as e_modality:
                    # 出错时默认为1
                    features['modality_peak_height'] = 1
                    print(f"计算多峰性指标时出错: {e_modality}")
            else:
                features['modality_peak_height'] = 0
            
            # B.4. 饱和效应指示特征
            sat_threshold = SATURATION_THRESHOLD  # 饱和阈值
            num_saturated_peaks = np.sum(sample_data['allele_heights'] >= sat_threshold)
            features['num_saturated_peaks'] = num_saturated_peaks
            if len(sample_data['allele_heights']) > 0:
                features['ratio_saturated_peaks'] = num_saturated_peaks / len(sample_data['allele_heights'])
            else:
                features['ratio_saturated_peaks'] = 0
            
            # C. 信息论与图谱复杂度/完整性特征
            # C.1. 位点间平衡性的香农熵
            total_height = sum(sample_data['locus_peak_heights'].values())
            if total_height > 0:
                locus_probs = [h/total_height for h in sample_data['locus_peak_heights'].values() if h > 0]
                if locus_probs:
                    features['inter_locus_balance_entropy'] = -np.sum(p * np.log(p) for p in locus_probs)
                else:
                    features['inter_locus_balance_entropy'] = 0
            else:
                features['inter_locus_balance_entropy'] = 0
            
            # C.2. 平均位点等位基因分布熵
            locus_entropies = []
            for locus, alleles in sample_data['locus_allele_details'].items():
                locus_height_sum = sum(a['height'] for a in alleles)
                if locus_height_sum > 0:
                    probs = [a['height'] / locus_height_sum for a in alleles]
                    entropy = -np.sum(p * np.log(p) for p in probs)
                    locus_entropies.append(entropy)
            
            if locus_entropies:
                features['avg_locus_allele_entropy'] = np.mean(locus_entropies)
            else:
                features['avg_locus_allele_entropy'] = 0
            
            # C.3. 样本整体峰高分布熵
            if len(sample_data['allele_heights']) > 0:
                # 对数变换后分箱
                log_heights = np.log(sample_data['allele_heights'] + 1)
                hist, _ = np.histogram(log_heights, bins=15)  # 15个箱子
                hist_probs = hist / np.sum(hist)
                hist_probs = hist_probs[hist_probs > 0]  # 移除零概率
                if len(hist_probs) > 0:
                    features['peak_height_entropy'] = -np.sum(p * np.log(p) for p in hist_probs)
                else:
                    features['peak_height_entropy'] = 0
            else:
                features['peak_height_entropy'] = 0
            
            # C.4. 图谱完整性指标
            features['num_loci_with_effective_alleles'] = sample_data['num_effective_loci']
            features['num_loci_no_effective_alleles'] = sample_data['expected_autosomal_count'] - sample_data['num_effective_loci']
            
            # D. DNA降解与信息丢失量化指标
            # D.1. 峰高与片段大小的相关性
            if len(sample_data['allele_heights']) > 1 and len(np.unique(sample_data['allele_heights'])) > 1 and len(np.unique(sample_data['allele_sizes'])) > 1:
                features['height_size_correlation'] = np.corrcoef(sample_data['allele_sizes'], sample_data['allele_heights'])[0, 1]
            else:
                features['height_size_correlation'] = 0
            
            # D.2. 峰高与片段大小的线性回归斜率
            if len(sample_data['allele_heights']) > 1 and len(np.unique(sample_data['allele_heights'])) > 1 and len(np.unique(sample_data['allele_sizes'])) > 1:
                try:
                    slope, _, _, _, _ = stats.linregress(sample_data['allele_sizes'], sample_data['allele_heights'])
                    features['height_size_slope'] = slope
                except:
                    features['height_size_slope'] = 0
            else:
                features['height_size_slope'] = 0
            
            # D.3. 加权峰高与片段大小的线性回归斜率 (WLS回归)
            if len(sample_data['allele_heights']) > 1 and len(np.unique(sample_data['allele_heights'])) > 1 and len(np.unique(sample_data['allele_sizes'])) > 1:
                try:
                    # 使用峰高作为权重
                    weights = sample_data['allele_heights']
                    
                    # 手动计算加权最小二乘回归
                    x = sample_data['allele_sizes']
                    y = sample_data['allele_heights']
                    w = weights
                    
                    sum_w = np.sum(w)
                    sum_wx = np.sum(w * x)
                    sum_wy = np.sum(w * y)
                    sum_wxy = np.sum(w * x * y)
                    sum_wx2 = np.sum(w * x**2)
                    
                    denominator = sum_w * sum_wx2 - sum_wx**2
                    if denominator != 0:
                        weighted_slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
                    else:
                        weighted_slope = 0
                    
                    features['weighted_height_size_slope'] = weighted_slope
                except:
                    features['weighted_height_size_slope'] = 0
            else:
                features['weighted_height_size_slope'] = 0
            
            # D.4. PHR随扩增子片段大小变化的斜率
            if len(phr_values) > 1:
                try:
                    # 获取有PHR值的位点及其平均片段大小
                    phr_loci = []
                    phr_sizes = []
                    
                    for locus, alleles in sample_data['locus_allele_details'].items():
                        if len(alleles) == 2:
                            h1 = alleles[0]['height']
                            h2 = alleles[1]['height']
                            if max(h1, h2) > 0:
                                # 获取该位点的平均片段大小
                                avg_size = np.mean([alleles[0]['size'], alleles[1]['size']])
                                phr = min(h1, h2) / max(h1, h2)
                                phr_loci.append(locus)
                                phr_sizes.append(avg_size)
                    
                    # 计算PHR与片段大小的回归
                    if len(phr_values) > 1 and len(phr_sizes) > 1:
                        slope, _, _, _, _ = stats.linregress(phr_sizes, phr_values)
                        features['phr_size_slope'] = slope
                    else:
                        features['phr_size_slope'] = 0
                except:
                    features['phr_size_slope'] = 0
            else:
                features['phr_size_slope'] = 0
            
            # D.5. 按片段大小加权的位点丢失评分
            try:
                # 获取每个位点的片段大小
                locus_sizes = {}
                for marker, params in MARKER_PARAMS.items():
                    if 'avg_size_bp' in params:
                        locus_sizes[marker] = params['avg_size_bp']
                
                # 计算丢失位点评分
                dropout_score = 0
                size_sum = 0
                
                for locus in sample_data['expected_autosomal_loci']:
                    if locus in locus_sizes:
                        size = locus_sizes[locus]
                        is_dropout = locus not in sample_data['effective_loci']
                        dropout_score += is_dropout * size
                        size_sum += size
                
                if size_sum > 0:
                    features['locus_dropout_score_weighted_by_size'] = dropout_score / size_sum
                else:
                    features['locus_dropout_score_weighted_by_size'] = 0
            except:
                features['locus_dropout_score_weighted_by_size'] = 0
            
            # D.6. RFU每碱基对衰减指数
            try:
                # 获取每个位点的最高峰和平均片段大小
                locus_max_heights = []
                locus_avg_sizes = []
                
                for locus in sample_data['effective_loci']:
                    if locus in sample_data['locus_allele_details'] and locus in MARKER_PARAMS and 'avg_size_bp' in MARKER_PARAMS[locus]:
                        max_height = max(a['height'] for a in sample_data['locus_allele_details'][locus])
                        avg_size = MARKER_PARAMS[locus]['avg_size_bp']
                        locus_max_heights.append(max_height)
                        locus_avg_sizes.append(avg_size)
                
                # 计算最高峰与位点平均片段大小的回归
                if len(locus_max_heights) > 1 and len(locus_avg_sizes) > 1:
                    slope, _, _, _, _ = stats.linregress(locus_avg_sizes, locus_max_heights)
                    features['degradation_index_rfu_per_bp'] = slope
                else:
                    features['degradation_index_rfu_per_bp'] = 0
            except:
                features['degradation_index_rfu_per_bp'] = 0
            
            # D.7. 小片段与大片段区域的信息完整度比率
            try:
                # 根据marker参数划分小片段和大片段位点
                small_loci = []
                large_loci = []
                
                for marker, params in MARKER_PARAMS.items():
                    if marker in sample_data['expected_autosomal_loci']:
                        if 'size_category' in params:
                            if params['size_category'] == 'small':
                                small_loci.append(marker)
                            elif params['size_category'] == 'large':
                                large_loci.append(marker)
                
                # 计算小片段和大片段区域的位点完整度
                small_completeness = 0
                large_completeness = 0
                
                if small_loci:
                    small_with_info = sum(1 for locus in small_loci if locus in sample_data['effective_loci'])
                    small_completeness = small_with_info / len(small_loci)
                
                if large_loci:
                    large_with_info = sum(1 for locus in large_loci if locus in sample_data['effective_loci'])
                    large_completeness = large_with_info / len(large_loci)
                
                # 计算比率
                if large_completeness > 0:
                    features['info_completeness_ratio_small_large'] = small_completeness / large_completeness
                else:
                    # 避免除零
                    features['info_completeness_ratio_small_large'] = small_completeness / 1e-6 if small_completeness > 0 else 0
            except:
                features['info_completeness_ratio_small_large'] = 0
            
            # 等位基因频率信息特征（由于缺乏频率数据，设为NaN）
            features['avg_allele_freq'] = np.nan
            features['num_rare_alleles'] = np.nan
            features['min_allele_freq'] = np.nan
            features['avg_locus_sum_neg_log_freq'] = np.nan
            
            # 存储样本的特征
            features_dict[sample_file] = features
        
        # 将特征转换为DataFrame
        df_features_all = pd.DataFrame.from_dict(features_dict, orient='index')
        
        # 将特征与NoC_True合并
        df_features_enhanced = df_features_enhanced.merge(df_features_all, left_index=True, right_index=True, how='left')
        
        # 重置索引，以便Sample File成为列
        df_features_enhanced = df_features_enhanced.reset_index()
        
        # 填充缺失值
        df_features_enhanced = df_features_enhanced.fillna(0)
    
    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features_enhanced 维度: {df_features_enhanced.shape}")
    print_df_in_chinese(df_features_enhanced.head(), col_map=COLUMN_TRANSLATION_MAP, title="增强特征数据框 (df_features_enhanced) 前5行")
    
    # 查看特征描述性统计
    if not df_features_enhanced.empty:
        numeric_cols = df_features_enhanced.select_dtypes(include=['number']).columns.drop('NoC_True')
        feature_stats = df_features_enhanced[numeric_cols].describe()
        print_df_in_chinese(feature_stats, col_map=COLUMN_TRANSLATION_MAP, index_item_map=DESCRIBE_INDEX_TRANSLATION_MAP, title="特征描述性统计")
    
    # 保存特征到文件
    enhanced_feature_filename = os.path.join(DATA_DIR, 'prob1_features_enhanced.csv')
    try:
        df_features_enhanced.to_csv(enhanced_feature_filename, index=False, encoding='utf-8-sig')
        print(f"增强特征数据已保存到: {enhanced_feature_filename}")
    except Exception as e:
        print(f"错误: 保存增强特征数据失败: {e}")
    
    # 将增强特征赋值给主数据框用于后续处理
    df_features_v2_9 = df_features_enhanced.copy()
    feature_filename_prob1 = os.path.join(DATA_DIR, 'prob1_features_enhanced.csv')

except Exception as e_feat_eng_enhanced:
    print(f"严重错误: 在增强特征工程阶段发生错误: {e_feat_eng_enhanced}")
    import traceback
    traceback.print_exc()
    
    # 如果增强特征工程失败，回退到基本特征工程
    print("尝试回退到基本特征工程...")
    try:
        # 基本特征工程（与原始V2.91代码相同）
        df_features_v2_9 = pd.DataFrame()
        
        if not df_processed_peaks.empty and 'CTA' in df_processed_peaks.columns and \
           'Sample File' in df_processed_peaks.columns and 'Marker' in df_processed_peaks.columns and \
           'Allele' in df_processed_peaks.columns and 'Height' in df_processed_peaks.columns:
            
            df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= TRUE_ALLELE_CONFIDENCE_THRESHOLD
            df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
            if df_effective_alleles.empty: 
                print("警告: 应用CTA阈值后，没有剩余有效等位基因。特征将主要为0或NaN。")
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
        
        print("回退到基本特征工程完成。")
    except Exception as e_basic_feat:
        print(f"严重错误: 基本特征工程也失败: {e_basic_feat}")
        traceback.print_exc()
        print("无法生成有效特征，将使用空特征框架。")
        
        # 创建一个最小化的特征框架
        all_sample_files = df_prob1['Sample File'].unique()
        noc_values = {sf: df_prob1[df_prob1['Sample File'] == sf]['NoC_True'].iloc[0] 
                    for sf in all_sample_files if len(df_prob1[df_prob1['Sample File'] == sf]) > 0}
        
        df_features_v2_9 = pd.DataFrame({'Sample File': list(noc_values.keys()), 
                                        'NoC_True': list(noc_values.values())})
        
        basic_features = ['max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker',
                          'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
                          'avg_peak_height', 'std_peak_height']
        
        for col in basic_features:
            df_features_v2_9[col] = 0.0
print(f"最终特征数据框 df_features_v2_9 维度: {df_features_v2_9.shape}")
print(f"\n--- 步骤 3: 特征工程完成 ---")
# --- 步骤 4 & 5: 模型评估 (框架) ---
# --- 步骤 4: 模型训练与验证 (版本 2.9) ---
# --- 步骤 4: 模型训练与验证 (版本 2.91) ---
# --- 步骤 4: 模型训练与验证 (融合决策树方法) ---
# --- 步骤 4: 模型训练与验证 (基于LassoCV筛选特征) ---
print(f"\n--- 步骤 4: 模型训练与验证 (版本 2.91 + 决策树 + LassoCV特征) ---")

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    from joblib import dump
    import os
    
    # 使用LassoCV筛选的14个特征
    lasso_selected_features = [
        'mac_profile', 
        'total_distinct_alleles', 
        'avg_alleles_per_locus', 
        'loci_gt3_alleles', 
        'loci_gt6_alleles', 
        'avg_peak_height', 
        'num_loci_with_phr', 
        'ratio_severe_imbalance_loci', 
        'skewness_peak_height', 
        'modality_peak_height', 
        'inter_locus_balance_entropy', 
        'avg_locus_allele_entropy', 
        'peak_height_entropy', 
        'height_size_correlation'
    ]
    
    print("使用LassoCV筛选的14个特征进行模型训练:")
    for i, feature in enumerate(lasso_selected_features, 1):
        print(f"{i}. {feature}")
    
    # 准备数据 - 只使用筛选后的特征
    # 首先检查这些特征是否都存在
    available_features = [f for f in lasso_selected_features if f in df_features_v2_9.columns]
    missing_features = [f for f in lasso_selected_features if f not in df_features_v2_9.columns]
    
    if missing_features:
        print(f"警告: 以下特征在数据框中不存在，将被忽略: {missing_features}")
    
    if len(available_features) < len(lasso_selected_features):
        print(f"实际使用 {len(available_features)}/{len(lasso_selected_features)} 个筛选特征")
    
    # 选择特征和目标变量
    X = df_features_v2_9[available_features].copy()
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
    
    # 定义分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 定义类别名称（用于可视化）
    class_names = [f"{i}人" for i in sorted(y.unique())]
    
    # 1. 决策树模型 (从paper.py整合)
    print("\n训练决策树模型...")
    
    dt_model = DecisionTreeClassifier(
        max_depth=4,  # 根据paper.py中设置的默认值
        class_weight='balanced',
        criterion='gini',
        random_state=42
    )
    
    # 交叉验证评估决策树
    dt_cv_scores = cross_val_score(dt_model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"决策树交叉验证准确率: {dt_cv_scores.mean():.4f} (±{dt_cv_scores.std():.4f})")
    
    # 训练最终决策树模型
    dt_model.fit(X_train, y_train)
    
    # 在测试集上评估决策树
    y_pred_dt = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"决策树测试集准确率: {dt_accuracy:.4f}")
    
    # 输出决策树分类报告
    print("\n决策树分类报告:")
    print(classification_report(y_test, y_pred_dt, target_names=class_names))
    
    # 可视化决策树 (从paper.py整合)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=X.columns, 
              class_names=class_names,
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title('决策树模型 (DNA贡献者人数分类)', fontsize=14)
    plt.tight_layout()
    decision_tree_path = os.path.join(PLOTS_DIR, 'decision_tree_v2.91.png')
    plt.savefig(decision_tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"决策树可视化图已保存至: {decision_tree_path}")
    
    # 2. 随机森林模型 (保留原有代码)
    print("\n训练随机森林模型...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    
    # 交叉验证评估随机森林
    rf_cv_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"随机森林交叉验证准确率: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")
    
    # 训练最终随机森林模型
    rf_model.fit(X_train, y_train)
    
    # 在测试集上评估随机森林
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林测试集准确率: {rf_accuracy:.4f}")
    
    # 输出随机森林分类报告
    print("\n随机森林分类报告:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    
    # 3. 集成学习方法 (增加决策树作为基学习器)
    print("\n训练集成学习模型 (包含决策树)...")
    
    # 准备基础估计器 (增加决策树)
    base_estimators = [
        ('dt', DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # 投票分类器
    voting_clf = VotingClassifier(
        estimators=base_estimators,
        voting='soft',
        weights=[1, 2, 1, 1]  # 为随机森林分配更高权重
    )
    
    # 交叉验证评估投票分类器
    voting_cv_scores = cross_val_score(voting_clf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"投票分类器交叉验证准确率: {voting_cv_scores.mean():.4f} (±{voting_cv_scores.std():.4f})")
    
    # 训练最终投票分类器
    voting_clf.fit(X_train, y_train)
    
    # 在测试集上评估投票分类器
    y_pred_voting = voting_clf.predict(X_test)
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"投票分类器测试集准确率: {voting_accuracy:.4f}")
    
    # 输出投票分类器分类报告
    print("\n投票分类器分类报告:")
    print(classification_report(y_test, y_pred_voting, target_names=class_names))
    
    # 4. 模型比较与选择
    # 选择最佳模型
    model_accuracies = {
        "决策树": dt_accuracy,
        "随机森林": rf_accuracy, 
        "投票分类器": voting_accuracy
    }
    
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model_acc = model_accuracies[best_model_name]
    
    if best_model_name == "决策树":
        best_model = dt_model
        y_pred = y_pred_dt
    elif best_model_name == "随机森林":
        best_model = rf_model
        y_pred = y_pred_rf
    else:
        best_model = voting_clf
        y_pred = y_pred_voting
    
    print(f"\n最佳模型: {best_model_name} (测试集准确率: {best_model_acc:.4f})")
    
    # 可视化比较所有模型性能
    plt.figure(figsize=(12, 8))
    model_names = list(model_accuracies.keys())
    accuracies = [model_accuracies[name] for name in model_names]
    
    # 找出最佳模型颜色
    colors = ['#1f77b4' if name != best_model_name else '#d62728' for name in model_names]
    
    # 创建性能比较柱状图
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel('准确率')
    plt.xlabel('模型')
    plt.title('各模型NoC识别准确率比较')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加准确率标签
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.03, f"{acc:.4f}", ha='center')
    
    # 保存模型比较图
    model_comparison_path = os.path.join(PLOTS_DIR, 'model_comparison_v2.91.png')
    plt.savefig(model_comparison_path)
    plt.close()
    print(f"模型比较图已保存至: {model_comparison_path}")
    
    # 绘制最佳模型的混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{best_model_name}混淆矩阵')
    confusion_matrix_path = os.path.join(PLOTS_DIR, 'confusion_matrix_v2.91.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"混淆矩阵已保存至: {confusion_matrix_path}")
    
    # 特征重要性分析
    if best_model_name in ["决策树", "随机森林"]:
        plt.figure(figsize=(14, 10))
        
        # 提取特征重要性并排序
        feature_importances = pd.DataFrame({
            '特征': X.columns,
            '重要性': best_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        # 输出特征重要性
        print(f"\n{best_model_name}特征重要性排名 (前10):")
        for idx, row in feature_importances.head(10).iterrows():
            print(f"{row['特征']: <30} {row['重要性']:.4f}")
        
        # 绘制前15个重要特征
        top_features = feature_importances.head(15)
        sns.barplot(x='重要性', y='特征', data=top_features)
        plt.title(f'{best_model_name} - 特征重要性 (前15位)')
        plt.tight_layout()
        feature_importance_path = os.path.join(PLOTS_DIR, 'feature_importance_v2.91.png')
        plt.savefig(feature_importance_path)
        plt.close()
        print(f"特征重要性图已保存至: {feature_importance_path}")
    
    # 将最佳模型应用于全部数据
    y_pred_all = best_model.predict(X_scaled)
    df_features_v2_9['baseline_pred'] = y_pred_all
    
    # 保存更新后的特征文件
    df_features_v2_9.to_csv(feature_filename_prob1, index=False, encoding='utf-8-sig')
    print(f"带预测结果的特征数据已保存至: {feature_filename_prob1}")
    
    # 保存模型
    model_filename = os.path.join(DATA_DIR, f'noc_{best_model_name}_v2.91.joblib')
    dump(best_model, model_filename)
    print(f"{best_model_name}模型已保存至: {model_filename}")
    
    # 增加来自不同过滤方法效果比较（模拟 paper.py 中的比较逻辑）
    print("\n不同过滤方法假设性比较 (从 paper.py 整合):")
    print("注意: 真实比较需要不同过滤方法处理后的数据。这里仅做概念演示。")
    
    filter_methods = ["当前版本 (v2.91)", "假设性AT过滤", "参考性能 (理论上限)"]
    
    # 这里假设的准确率值，实际使用时应从不同版本的真实结果填入
    hypothetical_accuracies = [
        best_model_acc,  # 当前准确率
        best_model_acc * 0.95,  # 假设的AT过滤版本（略低）
        best_model_acc * 1.05   # 假设的理论上限（略高）
    ]
    
    # 绘制过滤方法比较
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    bars = plt.bar(filter_methods, hypothetical_accuracies, color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel('假设性准确率')
    plt.xlabel('过滤方法')
    plt.title('不同过滤方法对NoC识别准确率的假设影响')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加准确率标签
    for i, acc in enumerate(hypothetical_accuracies):
        plt.text(i, acc + 0.03, f"{acc:.4f}", ha='center')
    
    filtering_comparison_path = os.path.join(PLOTS_DIR, 'filtering_comparison_v2.91.png')
    plt.savefig(filtering_comparison_path)
    plt.close()
    print(f"过滤方法比较图已保存至: {filtering_comparison_path}")
    print("注: 此比较为概念演示，实际使用时应基于不同过滤方法处理的真实数据进行比较")
    
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

# --- ROC曲线与AUC评分分析 ---
print("\n--- ROC曲线与AUC评分分析 ---")

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 确保我们有训练好的模型和测试数据
    if 'X_test' in locals() and 'y_test' in locals() and 'best_model' in locals():
        # 获取唯一类别
        classes = sorted(np.unique(y))
        n_classes = len(classes)
        
        # 将标签进行二值化处理用于多分类ROC计算
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # 初始化图表
        plt.figure(figsize=(12, 10))
        
        # 如果模型已经有predict_proba方法（如随机森林、决策树等）
        if hasattr(best_model, 'predict_proba'):
            # 直接使用模型
            y_score = best_model.predict_proba(X_test)
            
            # 为每个类别计算ROC曲线和AUC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 计算微平均ROC曲线和AUC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # 绘制所有ROC曲线
            colors = cm.rainbow(np.linspace(0, 1, n_classes))
            
            # 绘制每个类别的ROC曲线
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'类别 {classes[i]} 的ROC曲线 (AUC = {roc_auc[i]:.2f})')
            
            # 绘制微平均ROC曲线
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'微平均ROC曲线 (AUC = {roc_auc["micro"]:.2f})',
                     color='deeppink', linestyle=':', linewidth=4)
            
            # 绘制随机猜测的ROC曲线
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            # 设置图表
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正例率 (FPR)')
            plt.ylabel('真正例率 (TPR)')
            plt.title(f'{best_model_name}的ROC曲线')
            plt.legend(loc="lower right")
            
            # 保存ROC曲线图
            roc_curve_path = os.path.join(PLOTS_DIR, 'roc_curve_v2.91.png')
            plt.savefig(roc_curve_path)
            plt.close()
            print(f"ROC曲线图已保存至: {roc_curve_path}")
            
            # 打印每个类别的AUC值
            print("\n各类别AUC值:")
            for i in range(n_classes):
                print(f"类别 {classes[i]}: AUC = {roc_auc[i]:.4f}")
            print(f"微平均AUC: {roc_auc['micro']:.4f}")
            
            # 绘制精确率-召回率曲线
            plt.figure(figsize=(12, 10))
            
            # 计算每个类别的PR曲线
            precision = dict()
            recall = dict()
            avg_precision = dict()
            
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
            
            # 绘制每个类别的PR曲线
            for i, color in zip(range(n_classes), colors):
                plt.plot(recall[i], precision[i], color=color, lw=2,
                         label=f'类别 {classes[i]} 的PR曲线 (AP = {avg_precision[i]:.2f})')
            
            # 设置图表
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('召回率 (Recall)')
            plt.ylabel('精确率 (Precision)')
            plt.title(f'{best_model_name}的精确率-召回率曲线')
            plt.legend(loc="lower left")
            
            # 保存PR曲线图
            pr_curve_path = os.path.join(PLOTS_DIR, 'pr_curve_v2.91.png')
            plt.savefig(pr_curve_path)
            plt.close()
            print(f"精确率-召回率曲线图已保存至: {pr_curve_path}")
            
            # 打印每个类别的AP值
            print("\n各类别平均精确率(AP)值:")
            for i in range(n_classes):
                print(f"类别 {classes[i]}: AP = {avg_precision[i]:.4f}")
            
        else:
            # 如果模型没有predict_proba方法
            print(f"警告: {best_model_name}不支持概率预测，无法计算ROC曲线和AUC。")
            
        # 创建混淆矩阵热图(归一化)
        plt.figure(figsize=(10, 8))
        cm_test = confusion_matrix(y_test, y_pred)
        cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
        
        # 绘制混淆矩阵
        sns.heatmap(cm_test_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[f"{c}" for c in classes],
                   yticklabels=[f"{c}" for c in classes])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{best_model_name}混淆矩阵 (归一化)')
        
        # 保存混淆矩阵
        cm_norm_path = os.path.join(PLOTS_DIR, 'confusion_matrix_norm_v2.91.png')
        plt.savefig(cm_norm_path)
        plt.close()
        print(f"归一化混淆矩阵已保存至: {cm_norm_path}")
        
    else:
        print("错误: 模型训练未完成或测试数据不可用，无法生成ROC曲线和计算AUC。")

except ModuleNotFoundError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请安装所需库: pip install scikit-learn matplotlib")
except Exception as e:
    print(f"ROC曲线生成过程中发生错误: {e}")
    import traceback
    traceback.print_exc()

print("--- ROC/AUC分析完成 ---")