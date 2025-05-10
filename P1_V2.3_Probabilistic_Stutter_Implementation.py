# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题1：贡献者人数识别

代码名称: P1_V2.3_Probabilistic_Stutter_Implementation
版本: 2.3
日期: 2025-05-10
描述: 整合了用户确认的参数和Stutter置信度打分逻辑。
      实现了核心函数 calculate_peak_confidence_v2_3 用于峰识别与Stutter概率化评估。
      脚本会加载数据，应用此函数处理所有峰，并生成 df_processed_peaks。
      后续的特征工程和模型评估步骤将基于 df_processed_peaks 进行。
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

# --- 配置与环境设置 (版本 2.3) ---
print("--- 脚本初始化与配置 (版本 2.3) ---")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}.")

DATA_DIR = './'
file_path_prob1 = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
PLOTS_DIR = os.path.join(DATA_DIR, 'prob1_plots_v2.3_stutter_eval')
# feature_filename_prob1_v2_3 = os.path.join(DATA_DIR, 'prob1_features_v2.3.csv') # 特征保存文件名(下一步骤生成)
processed_peaks_filename = os.path.join(DATA_DIR, 'prob1_processed_peaks_v2.3.csv') # 保存处理后峰数据

if not os.path.exists(PLOTS_DIR):
    try:
        os.makedirs(PLOTS_DIR)
        print(f"已创建绘图保存目录: {PLOTS_DIR}")
    except OSError as e_dir:
        PLOTS_DIR = DATA_DIR

# --- 全局参数定义 (版本 2.3) ---
print(f"\n--- 全局参数定义 (版本 2.3) ---")
SATURATION_THRESHOLD = 30000.0
SIZE_TOLERANCE_BP = 0.5 # 片段大小比较的容忍误差 (bp)
# 根据您的确认，全局 n-1 Stutter 峰高变异系数
GLOBAL_CV_HS_N_MINUS_1 = 0.25
STUTTER_CV_HS_GLOBAL = 0.25 # 全局Stutter峰高的假设变异系数 (n-1反向Stutter)
GLOBAL_AT = 50 # 备用全局AT
DYE_AT_VALUES = {
    'B': 75, 'BLUE': 75, 'G': 101, 'GREEN': 101, 'Y': 60, 'YELLOW': 60,
    'R': 69, 'RED': 69, 'P': 56, 'PURPLE': 56, # 'P' 对应 Purple
    'O': 50, 'ORANGE': 50, 'UNKNOWN': GLOBAL_AT
}

# MARKER_PARAMS 字典 (核心！由您根据文献详细填充)
# SR_c 对于 Allele Average 模型即为 SR_avg
# 对于 LUS Regression，如果无LUS数据，则回退使用提供的基于Allele的m,c
MARKER_PARAMS = {
    "AMEL":    {"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D3S1358": {"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.09860,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D1S1656": {"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "LUS Regression",  "SR_m": 0.00604,   "SR_c": 0.00132,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D2S441":  {"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.05268,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D10S1248":{"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01068,  "SR_c": -0.06292,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D13S317": {"Lrepeat_k": 4, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.05638,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "Penta E": {"Lrepeat_k": 5, "Dye": "Blue",   "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00398,  "SR_c": -0.01607,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D16S539": {"Lrepeat_k": 4, "Dye": "Green",  "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.0118,   "SR_c": -0.0595,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D18S51":  {"Lrepeat_k": 4, "Dye": "Green",  "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00879,  "SR_c": -0.04708,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D2S1338": {"Lrepeat_k": 4, "Dye": "Green",  "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.09165,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "CSF1PO":  {"Lrepeat_k": 4, "Dye": "Green",  "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01144,  "SR_c": -0.05766,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "Penta D": {"Lrepeat_k": 5, "Dye": "Green",  "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.01990,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "TH01":    {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "LUS Regression",  "SR_m": 0.00185,  "SR_c": 0.00801,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "VWA":     {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08928,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D21S11":  {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08990,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D7S820":  {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01048,  "SR_c": -0.05172,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D5S818":  {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.07378,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "TPOX":    {"Lrepeat_k": 4, "Dye": "Yellow", "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.00611,  "SR_c": -0.02772,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D8S1179": {"Lrepeat_k": 4, "Dye": "Red",    "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08246,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D12S391": {"Lrepeat_k": 4, "Dye": "Red",    "n-1_Stutter": {"SR_model_type": "Allele Regression","SR_m": 0.01063,  "SR_c": -0.10533,  "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D19S433": {"Lrepeat_k": 4, "Dye": "Red",    "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08044,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "SE33":    {"Lrepeat_k": 4, "Dye": "Red",    "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.12304,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "D22S1045":{"Lrepeat_k": 3, "Dye": "Red",    "n-1_Stutter": {"SR_model_type": "LUS Regression",  "SR_m": 0.01528,  "SR_c": -0.1354,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "DYS391":  {"Lrepeat_k": 4, "Dye": "Purple", "n-1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "FGA":     {"Lrepeat_k": 4, "Dye": "Purple", "n-1_Stutter": {"SR_model_type": "Allele Average",  "SR_m": 0,         "SR_c": 0.08296,   "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "DYS576":  {"Lrepeat_k": 4, "Dye": "Purple", "n-1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
    "DYS570":  {"Lrepeat_k": 4, "Dye": "Purple", "n-1_Stutter": {"SR_model_type": "N/A",             "SR_m": 0,         "SR_c": 0,         "RS_max_k": 0.3, "CV_Hs": GLOBAL_CV_HS_N_MINUS_1}},
}
# --- 中文翻译映射表 ---
# ... (同V1.15)
COLUMN_TRANSLATION_MAP = {
    'Sample File': '样本文件', 'Marker': '标记', 'Dye': '染料',
    'Allele': '等位基因', 'Allele_Numeric': '数值型等位基因',
    'Size': '片段大小(bp)', 'Height': '峰高(RFU)', 'Original_Height': '原始峰高(RFU)',
    'CTA': '真实等位基因置信度', 'Is_Stutter_Suspect': '高度可疑Stutter',
    'Stutter_Score_as_N_minus_1': '作为n-1 Stutter的得分',
    'NoC_True': '真实贡献人数',
    'max_allele_per_sample': '样本内最大有效等位基因数',
    'total_alleles_per_sample': '样本内总有效等位基因数',
    'avg_alleles_per_marker': '每标记平均有效等位基因数',
    'markers_gt2_alleles': '有效等位基因数>2的标记数',
    'markers_gt3_alleles': '有效等位基因数>3的标记数',
    'markers_gt4_alleles': '有效等位基因数>4的标记数',
    'avg_peak_height': '有效等位基因平均峰高',
    'std_peak_height': '有效等位基因峰高标准差',
    'baseline_pred': '基线模型预测NoC'
}
DESCRIBE_INDEX_TRANSLATION_MAP = { 
    'count': '计数', 'mean': '均值', 'std': '标准差', 'min': '最小值',
    '25%': '25%分位数', '50%': '中位数(50%)', '75%': '75%分位数', 'max': '最大值'
}
# --- 辅助函数 ---
def print_df_in_chinese(df_to_print, col_map=None, index_item_map=None, index_name_map=None, title="DataFrame 内容", float_format='{:.4f}'):
    # ... (同V1.15)
    print(f"\n{title}:")
    df_display = df_to_print.copy()
    if col_map: df_display.columns = [col_map.get(str(col), str(col)) for col in df_display.columns]
    if index_item_map: df_display.index = [index_item_map.get(str(idx), str(idx)) for idx in df_display.index]
    if index_name_map and df_display.index.name is not None: df_display.index.name = index_name_map.get(str(df_display.index.name), str(df_display.index.name))
    with pd.option_context('display.float_format', float_format.format, 'display.max_rows', None, 'display.max_columns', None, 'display.width', 120): print(df_display)

def extract_true_noc_v2_3(filename_str):
    # ... (同V2.2/V2.3) ...
    filename_str = str(filename_str)
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', filename_str)
    if match:
        ids_list = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return int(len(ids_list)) if len(ids_list) > 0 else np.nan
    return np.nan

def get_allele_numeric_v2_3(allele_val_str):
    # ... (同V2.3) ...
    try: return float(allele_val_str)
    except ValueError:
        if isinstance(allele_val_str, str):
            if allele_val_str.upper() == 'X': return -1.0 
            if allele_val_str.upper() == 'Y': return -2.0
        return np.nan

# --- 核心：峰处理与Stutter评估函数 (版本 2.3) ---
print(f"\n--- 核心函数定义: calculate_peak_confidence_v2_3 (版本 2.3) ---")
def calculate_peak_confidence_v2_3(locus_peaks_df_input, marker_name, marker_params_dict,
                                 dye_at_dict, global_at_val, sat_threshold,
                                 size_tolerance, cv_hs_n_minus_1_global):
    locus_peaks_df = locus_peaks_df_input.copy()
    processed_peaks_output_cols = ['Allele', 'Allele_Numeric', 'Size', 'Height', 'Original_Height', 'CTA', 'Is_Stutter_Suspect', 'Stutter_Score_as_N_minus_1']

    if marker_name not in marker_params_dict:
        params = {'L_repeat': 0, 'Dye': locus_peaks_df['Dye'].iloc[0] if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else 'UNKNOWN', 'n_minus_1_Stutter': {'SR_model_type': 'N/A'}}
    else:
        params = marker_params_dict[marker_name]

    l_repeat = params.get('L_repeat', 0)
    current_dye = locus_peaks_df['Dye'].iloc[0].upper() if not locus_peaks_df.empty and 'Dye' in locus_peaks_df.columns else params.get('Dye', 'UNKNOWN').upper()
    at_val = dye_at_dict.get(current_dye, global_at_val)

    candidate_peaks_list = []
    for idx, peak_row in locus_peaks_df.iterrows():
        height_original = float(peak_row['Height'])
        height_adj = min(height_original, sat_threshold)
        if height_adj >= at_val:
            allele_numeric = get_allele_numeric_v2_3(peak_row['Allele'])
            candidate_peaks_list.append({
                'Allele': peak_row['Allele'], 'Allele_Numeric': allele_numeric,
                'Size': float(peak_row['Size']), 'Height': height_adj,
                'Original_Height': height_original,
                'Stutter_Score_as_N_minus_1': 0.0,
            })
    
    if not candidate_peaks_list:
        return pd.DataFrame(columns=processed_peaks_output_cols)

    peaks_df = pd.DataFrame(candidate_peaks_list).sort_values(by='Height', ascending=False).reset_index(drop=True)
    
    stutter_params_n_minus_1 = params.get('n-1_Stutter', {})
    if l_repeat == 0 or stutter_params_n_minus_1.get('SR_model_type') == 'N/A' or pd.isna(stutter_params_n_minus_1.get('RS_max_k', np.nan)):
        peaks_df['CTA'] = 1.0
        peaks_df['Is_Stutter_Suspect'] = False
        # Stutter_Score_as_N_minus_1 is already 0.0
        return peaks_df[processed_peaks_output_cols]

    sr_max_n_minus_1 = stutter_params_n_minus_1.get('RS_max_k', 0.3) # Default if missing in specific entry

    temp_max_stutter_scores = [0.0] * len(peaks_df)
    for i in range(len(peaks_df)): # pc_row is potential stutter
        pc_row = peaks_df.iloc[i]
        current_max_score_pc_is_stutter_n_minus_1 = 0.0
        for j in range(len(peaks_df)): # pp_row is potential parent
            if i == j: continue
            pp_row = peaks_df.iloc[j]
            if pc_row['Height'] >= pp_row['Height']: continue

            is_pos_match_n_minus_1 = (abs((pp_row['Size'] - pc_row['Size']) - l_repeat) <= size_tolerance)
            
            if is_pos_match_n_minus_1:
                sr_obs = pc_row['Height'] / pp_row['Height'] if pp_row['Height'] > 1e-9 else np.inf

                if sr_obs > sr_max_n_minus_1:
                    score_stutter_for_this_parent = 0.0
                else:
                    e_sr = 0.0
                    parent_an = pp_row['Allele_Numeric']
                    if pd.notna(parent_an) and parent_an >= 0: # Ensure parent allele is valid for model
                        model_type = stutter_params_n_minus_1.get('SR_model_type')
                        m_val = stutter_params_n_minus_1.get('SR_m', 0)
                        c_val = stutter_params_n_minus_1.get('SR_c', 0)
                        if model_type == 'Allele Regression' or model_type == 'LUS Regression': # LUS falls back to Allele Reg.
                            e_sr = m_val * parent_an + c_val
                        elif model_type == 'Allele Average':
                            e_sr = c_val # c is SR_avg for this type
                    
                    e_sr = max(0.001, e_sr) # Prevent E[SR] from being zero or negative
                    e_hs = e_sr * pp_row['Height']
                    
                    current_score = 0.0
                    if e_hs > 1e-6:
                        sigma_hs = cv_hs_n_minus_1_global * e_hs
                        if sigma_hs > 1e-6:
                            z_score = (pc_row['Height'] - e_hs) / sigma_hs
                            current_score = exp(-0.5 * (z_score**2))
                            # Optional: Further penalize if SR_obs is much higher than E_SR but still under SR_max
                            # if sr_obs > e_sr * 1.8 and e_sr > 1e-6: current_score *= 0.5 
                        else: # sigma_hs is effectively zero, implies E_Hs is also very small or CV is zero
                            current_score = 1.0 if abs(pc_row['Height'] - e_hs) < (0.001 * e_hs + 1e-6) else 0.0 # Strict match if no variance
                    elif pc_row['Height'] < 1e-6: # If E_Hs is zero, candidate must also be zero
                        current_score = 1.0
                    
                    score_stutter_for_this_parent = current_score
                
                current_max_score_pc_is_stutter_n_minus_1 = max(current_max_score_pc_is_stutter_n_minus_1, score_stutter_for_this_parent)
        
        temp_max_stutter_scores[i] = current_max_score_pc_is_stutter_n_minus_1
        
    peaks_df['Stutter_Score_as_N_minus_1'] = temp_max_stutter_scores
    peaks_df['CTA'] = 1.0 - peaks_df['Stutter_Score_as_N_minus_1']
    peaks_df['CTA'] = peaks_df['CTA'].clip(lower=0.0, upper=1.0)
    peaks_df['Is_Stutter_Suspect'] = peaks_df['Stutter_Score_as_N_minus_1'] >= 0.5 # Example threshold

    return peaks_df[processed_peaks_output_cols]


# --- 步骤 1: 数据加载与NoC提取 ---
# (您需要在此处填入来自P1_V1.15版本并成功运行的完整 Step 1 代码)
print(f"\n--- 步骤 1：数据加载与 NoC 提取 (版本 2.2) ---")
# ... (与 V1.15 相同的完整数据加载和 NoC_True 提取代码块，确保 load_successful 和 df_prob1 被正确设置)
df_prob1 = None; load_successful = False
try:
    df_prob1 = pd.read_csv(file_path_prob1, encoding='utf-8', sep=',', on_bad_lines='skip')
    load_successful = True; print(f"成功加载文件: '{file_path_prob1}'")
except Exception as e: print(f"加载文件失败: {e}"); exit()
if df_prob1.empty: print("数据加载后为空"); exit()
if 'Sample File' not in df_prob1.columns: print("'Sample File' 列不存在"); exit()
unique_files = df_prob1['Sample File'].dropna().unique()
noc_map = {filename: extract_true_noc_v2_3(filename) for filename in unique_files}
df_prob1['NoC_True_Mapped'] = df_prob1['Sample File'].map(noc_map)
df_prob1.dropna(subset=['NoC_True_Mapped'], inplace=True)
if df_prob1.empty: print("处理NoC后数据为空"); exit()
df_prob1['NoC_True'] = df_prob1['NoC_True_Mapped'].astype(int)
df_prob1.drop(columns=['NoC_True_Mapped'], inplace=True)
print(f"NoC 提取完成。数据维度: {df_prob1.shape}")
# ----- 假设 Step 1 执行完毕，df_prob1 包含数据和正确的 NoC_True 列 -----
# 为了脚本能独立运行（即使没有Step1的实际数据加载），我将创建一个模拟的df_prob1结构
# 在实际使用中，您必须用真实的Step1代码替换这部分
if 'df_prob1' not in locals() or df_prob1.empty: # Check if df_prob1 was loaded by user's Step 1
    print("警告: df_prob1 未由用户提供的Step1代码加载。将创建模拟数据以继续。")
    # 创建非常小型的模拟 df_prob1，仅用于演示后续流程的运行
    # 这个模拟数据与实际附件1的结构和内容有极大差异，仅用于让脚本不报错地运行下去
    _sim_data = {
        'Sample File': ['S1-ID1_ID2-1;1-other', 'S1-ID1_ID2-1;1-other', 'S2-ID3_ID4_ID5-1;1;1-other', 'S2-ID3_ID4_ID5-1;1;1-other'],
        'Marker': ['M1', 'M2', 'M1', 'M2'],
        'Dye': ['B', 'G', 'B', 'R'],
        'Allele 1': [10, 12, 10, 15], 'Size 1': [100, 120, 100, 130], 'Height 1': [1000, 800, 1200, 900],
        'Allele 2': [11, 13, 9.3, 16], 'Size 2': [104, 124, 97, 134], 'Height 2': [900, 700, 30, 100], # 9.3 peak for M1 in S2 is low (stutter?)
        'Allele 3': [None, 14, 11, None], 'Size 3': [np.nan, 128, 104, np.nan], 'Height 3': [np.nan, 20, 1100, np.nan], # 14 is low stutter of 13 for M2 in S1?
    }
    # 为模拟数据填充到 Allele 100
    for i in range(4, 101):
        _sim_data[f'Allele {i}'] = [None] * 4
        _sim_data[f'Size {i}'] = [np.nan] * 4
        _sim_data[f'Height {i}'] = [np.nan] * 4

    df_prob1 = pd.DataFrame(_sim_data)
    # 为模拟数据添加NoC_True
    _noc_map_sim = {filename: extract_true_noc_v2_3(filename) for filename in df_prob1['Sample File'].unique()}
    df_prob1['NoC_True'] = df_prob1['Sample File'].map(_noc_map_sim)
    df_prob1.dropna(subset=['NoC_True'], inplace=True)
    df_prob1['NoC_True'] = df_prob1['NoC_True'].astype(int)
    print("模拟 df_prob1 已创建。")
    print(df_prob1.head())
    noc_distribution_sim = df_prob1['NoC_True'].value_counts().sort_index()
    print_df_in_chinese(noc_distribution_sim.reset_index().rename(columns={'index':'NoC_True', 'NoC_True':'count'}),
                        col_map=COLUMN_TRANSLATION_MAP, index_item_map=DESCRIBE_INDEX_TRANSLATION_MAP,
                        title="模拟 'NoC_True' 分布")
print("--- 步骤 1 完成 (或使用模拟数据) ---")


# --- 步骤 2: 峰识别与Stutter概率化评估 ---
print(f"\n--- 步骤 2: 峰识别与Stutter概率化评估 (版本 2.3) ---")
if df_prob1.empty: print("错误: df_prob1 为空，无法进行峰处理。"); exit()
if not MARKER_PARAMS: print("错误: MARKER_PARAMS 为空。请填充参数。"); exit()

processed_peak_data_all_samples = []
sample_files_processed_count = 0
unique_sample_files_total = df_prob1['Sample File'].nunique()

for sample_file_name, group_data_per_sample in df_prob1.groupby('Sample File'):
    sample_files_processed_count += 1
    if unique_sample_files_total > 10 and sample_files_processed_count % (unique_sample_files_total // 10) == 0: # 每处理10%打印一次
        print(f"正在处理样本: {sample_file_name} ({sample_files_processed_count}/{unique_sample_files_total})...", end='\r')
    
    sample_all_loci_processed_peaks = []
    for marker_name_actual, locus_data_from_groupby in group_data_per_sample.groupby('Marker'):
        current_locus_peaks_list = []
        if locus_data_from_groupby.empty: continue
        row_marker_data = locus_data_from_groupby.iloc[0]

        for i in range(1, 101):
            allele_val = row_marker_data.get(f'Allele {i}')
            size_val = row_marker_data.get(f'Size {i}')
            height_val = row_marker_data.get(f'Height {i}')
            if pd.notna(allele_val) and pd.notna(size_val) and pd.notna(height_val) and float(height_val) > 0:
                current_locus_peaks_list.append({
                    'Allele': allele_val, 'Size': size_val, 'Height': height_val,
                    'Dye': str(row_marker_data.get('Dye', 'UNKNOWN')).upper() # 确保Dye是大写
                })
        
        if not current_locus_peaks_list: continue
        locus_peaks_for_filter_df = pd.DataFrame(current_locus_peaks_list)
        
        processed_locus_df = calculate_peak_confidence_v2_3(
            locus_peaks_for_filter_df, marker_name_actual, MARKER_PARAMS,
            DYE_AT_VALUES, GLOBAL_AT, SATURATION_THRESHOLD,
            SIZE_TOLERANCE_BP, STUTTER_CV_HS_GLOBAL
        )
        if not processed_locus_df.empty:
            processed_locus_df['Sample File'] = sample_file_name
            processed_locus_df['Marker'] = marker_name_actual
            sample_all_loci_processed_peaks.append(processed_locus_df)
    
    if sample_all_loci_processed_peaks:
        processed_peak_data_all_samples.extend(sample_all_loci_processed_peaks)
print("\n所有样本位点处理完成。")

if not processed_peak_data_all_samples: print("错误: 处理所有样本后，列表 processed_peak_data_all_samples 为空。"); exit()
df_processed_peaks = pd.concat(processed_peak_data_all_samples, ignore_index=True)
print(f"峰处理与Stutter评估完成。共处理 {sample_files_processed_count} 个独立样本。")
print(f"处理后的总峰条目数 (df_processed_peaks): {len(df_processed_peaks)}")
print_df_in_chinese(df_processed_peaks.head(), col_map=COLUMN_TRANSLATION_MAP, title="处理后峰数据示例 (df_processed_peaks)")
print_df_in_chinese(df_processed_peaks.sample(min(10, len(df_processed_peaks))), col_map=COLUMN_TRANSLATION_MAP, title="随机抽样10条处理后峰数据")

try:
    df_processed_peaks.to_csv(processed_peaks_filename, index=False, encoding='utf-8-sig')
    print(f"已处理的峰数据已保存到: {processed_peaks_filename}")
except Exception as e_save_peaks:
    print(f"错误: 保存处理后的峰数据失败: {e_save_peaks}")
print("--- 步骤 2 完成 ---")


# --- 步骤 3: 特征工程 (版本 2.3 - 基于 df_processed_peaks) ---
print(f"\n--- 步骤 3: 特征工程 (版本 2.3) ---")
df_features_v2_3 = pd.DataFrame()
try:
    # ... (与V2.2版本相同的特征工程代码，确保输入是 df_processed_peaks，
    #      并使用其中的 'CTA' 列进行有效等位基因筛选) ...
    # <<<<< 在此插入或确保 V2.2 中 Step 3 的完整、可成功运行的代码, 确保使用的是 df_processed_peaks >>>>>
    CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE = 0.5 
    print(f"用于筛选真实等位基因的置信度阈值 (CTA_threshold): {CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE}")
    df_processed_peaks['Is_Effective_Allele'] = df_processed_peaks['CTA'] >= CONFIDENCE_THRESHOLD_FOR_TRUE_ALLELE
    df_effective_alleles = df_processed_peaks[df_processed_peaks['Is_Effective_Allele']]
    if df_effective_alleles.empty: print("警告: 应用CTA阈值后，没有剩余有效等位基因。")
    
    all_sample_files_index_df = df_prob1[['Sample File', 'NoC_True']].drop_duplicates().set_index('Sample File')
    df_features_v2_3 = pd.DataFrame(index=all_sample_files_index_df.index)
    df_features_v2_3['NoC_True'] = all_sample_files_index_df['NoC_True']

    if not df_effective_alleles.empty:
        n_eff_alleles_per_locus = df_effective_alleles.groupby(['Sample File', 'Marker'])['Allele'].count().rename('N_eff_alleles')
        grouped_by_sample_eff = n_eff_alleles_per_locus.groupby('Sample File')
        for col, series in {
            'max_allele_per_sample': grouped_by_sample_eff.max(),
            'total_alleles_per_sample': grouped_by_sample_eff.sum(),
            'avg_alleles_per_marker': grouped_by_sample_eff.mean(),
            'markers_gt2_alleles': grouped_by_sample_eff.apply(lambda x: (x > 2).sum()),
            'markers_gt3_alleles': grouped_by_sample_eff.apply(lambda x: (x > 3).sum()),
            'markers_gt4_alleles': grouped_by_sample_eff.apply(lambda x: (x > 4).sum())
        }.items(): df_features_v2_3[col] = series.reindex(df_features_v2_3.index)
        
        grouped_heights_eff = df_effective_alleles.groupby('Sample File')['Height']
        df_features_v2_3['avg_peak_height'] = grouped_heights_eff.mean().reindex(df_features_v2_3.index)
        df_features_v2_3['std_peak_height'] = grouped_heights_eff.std().reindex(df_features_v2_3.index)
    else: # 如果没有有效等位基因，则特征赋0
        for col in ['max_allele_per_sample', 'total_alleles_per_sample', 'avg_alleles_per_marker', 
                    'markers_gt2_alleles', 'markers_gt3_alleles', 'markers_gt4_alleles',
                    'avg_peak_height', 'std_peak_height']:
            df_features_v2_3[col] = 0
            
    df_features_v2_3.fillna(0, inplace=True)
    df_features_v2_3.reset_index(inplace=True)

    print("\n--- 特征工程完成 ---")
    print(f"最终特征数据框 df_features_v2_3 维度: {df_features_v2_3.shape}")
    print_df_in_chinese(df_features_v2_3.head(), col_map=COLUMN_TRANSLATION_MAP, title="新特征数据框 (df_features_v2_3) 前5行")
    # ... (打印 describe 的代码)

except Exception as e_feat_eng_v2_3:
    print(f"严重错误: 在特征工程阶段发生错误: {e_feat_eng_v2_3}"); import traceback; traceback.print_exc();

# --- 步骤 4 & 5: 模型评估 ---
# print(f"\n--- 步骤 4 & 5: 模型评估 (版本 2.3) ---")
# print("请将之前版本 (如V1.15) 的步骤4和5模型评估代码复制到此处，并确保输入是 df_features_v2_3。")
# ... (模型训练和评估代码) ...

print(f"\n脚本 {os.path.basename(__file__)} (版本 2.3) 执行完毕。")