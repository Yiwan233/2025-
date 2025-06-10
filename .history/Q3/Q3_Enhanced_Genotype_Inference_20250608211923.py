# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题三：基于V4强化的MCMC基因型推断系统

版本: V4.0 - Enhanced MCMC Genotype Inference
日期: 2025-06-07
描述: 结合Q1的RFECV特征选择、Q2的MGM-RF方法进行基因型推断
核心创新:
1. 集成Q1的V5特征工程和随机森林NoC预测
2. 采用Q2的MGM-M基因型边缘化MCMC方法
3. 增强的伪频率计算和先验建模
4. 多链MCMC收敛性诊断和质量控制
5. 基因型匹配准确性评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gammaln
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from collections import defaultdict, Counter
import itertools
from math import comb
import logging
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from copy import deepcopy

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("问题三：基于V4强化的MCMC基因型推断系统")
print("集成Q1特征工程 + Q2 MGM-RF方法 + 基因型匹配评估")
print("=" * 80)

# =====================
# 1. 系统配置
# =====================
class Config:
    """系统配置类"""
    # 文件路径配置
    DATA_DIR = './'
    ATTACHMENT1_PATH = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
    ATTACHMENT2_PATH = os.path.join(DATA_DIR, '附件2：混合STR图谱数据.csv')
    ATTACHMENT3_PATH = os.path.join(DATA_DIR, '附件3：各个贡献者的基因型.csv')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'q3_enhanced_results')
    Q1_MODEL_PATH = os.path.join(DATA_DIR, 'noc_optimized_random_forest_model.pkl')
    # 样本文件名解析参数
    SAMPLE_NAME_PATTERN = r'-(\d+(?:_\d+)*)-[\d;]+-(\d+(?:;\d+)*)-'  # 提取贡献者ID和比例
    CONTRIBUTOR_SEPARATOR = '_'  # 贡献者ID分隔符
    RATIO_SEPARATOR = ';'        # 混合比例分隔符
    
    # STR分析参数
    HEIGHT_THRESHOLD = 50
    SATURATION_THRESHOLD = 30000
    CTA_THRESHOLD = 0.5
    PHR_IMBALANCE_THRESHOLD = 0.6
    
    # MCMC参数
    N_ITERATIONS = 12000
    N_WARMUP = 4000
    N_CHAINS = 4
    THINNING = 3
    K_TOP = 600  # 用于N>=4的采样策略
    
    # 模型参数
    RANDOM_STATE = 42
    N_JOBS = -1

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# =====================
# 2. 继承Q1和Q2的核心组件
# =====================

# 从Q1继承特征工程
exec("""
def extract_noc_from_filename(filename):
    match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
    if match:
        ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
        return len(ids) if len(ids) > 0 else np.nan
    return np.nan

def calculate_entropy(probabilities):
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

def calculate_ols_slope(x, y):
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
""")

class Q1FeatureEngineering:
    """继承Q1的特征工程功能"""
    
    def __init__(self):
        self.feature_cache = {}
        logger.info("Q1特征工程模块初始化完成")
    
    def process_peaks_simplified(self, sample_data):
        """简化的峰处理函数（继承自Q1）"""
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
                    corrected_height = min(original_height, config.SATURATION_THRESHOLD)
                    
                    if corrected_height >= config.HEIGHT_THRESHOLD:
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
                
                if cta >= config.CTA_THRESHOLD:
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
    
    def extract_v5_features(self, sample_file, sample_peaks):
        """提取V5特征集（简化版）"""
        if sample_peaks.empty:
            return self._get_default_features(sample_file)
        
        features = {'Sample File': sample_file}
        
        # 基础数据准备
        total_peaks = len(sample_peaks)
        all_heights = sample_peaks['Height'].values
        all_sizes = sample_peaks['Size'].values
        
        # 按位点分组统计
        locus_groups = sample_peaks.groupby('Marker')
        alleles_per_locus = locus_groups['Allele'].nunique()
        locus_heights = locus_groups['Height'].sum()
        
        # A类：图谱层面基础统计特征
        features['mac_profile'] = alleles_per_locus.max() if len(alleles_per_locus) > 0 else 0
        features['total_distinct_alleles'] = sample_peaks['Allele'].nunique()
        features['avg_alleles_per_locus'] = alleles_per_locus.mean() if len(alleles_per_locus) > 0 else 0
        features['std_alleles_per_locus'] = alleles_per_locus.std() if len(alleles_per_locus) > 1 else 0
        
        # MGTN系列
        for N in [3, 4, 5, 6]:
            features[f'loci_gt{N}_alleles'] = (alleles_per_locus >= N).sum()
        
        # 等位基因计数分布熵
        if len(alleles_per_locus) > 0:
            counts = alleles_per_locus.value_counts(normalize=True)
            features['allele_count_dist_entropy'] = calculate_entropy(counts.values)
        else:
            features['allele_count_dist_entropy'] = 0
        
        # B类：峰高、平衡性特征
        if total_peaks > 0:
            features['avg_peak_height'] = np.mean(all_heights)
            features['std_peak_height'] = np.std(all_heights) if total_peaks > 1 else 0
            features['skewness_peak_height'] = stats.skew(all_heights) if total_peaks > 2 else 0
            features['kurtosis_peak_height'] = stats.kurtosis(all_heights, fisher=False) if total_peaks > 3 else 0
            
            # PHR相关特征
            phr_values = []
            for marker, marker_group in locus_groups:
                if len(marker_group) == 2:
                    heights = marker_group['Height'].values
                    phr = min(heights) / max(heights) if max(heights) > 0 else 0
                    phr_values.append(phr)
            
            if phr_values:
                features['avg_phr'] = np.mean(phr_values)
                features['std_phr'] = np.std(phr_values) if len(phr_values) > 1 else 0
                features['min_phr'] = np.min(phr_values)
                features['median_phr'] = np.median(phr_values)
                features['num_loci_with_phr'] = len(phr_values)
                features['num_severe_imbalance_loci'] = sum(phr <= config.PHR_IMBALANCE_THRESHOLD for phr in phr_values)
                features['ratio_severe_imbalance_loci'] = features['num_severe_imbalance_loci'] / len(phr_values)
            else:
                for key in ['avg_phr', 'std_phr', 'min_phr', 'median_phr', 'num_loci_with_phr', 
                           'num_severe_imbalance_loci', 'ratio_severe_imbalance_loci']:
                    features[key] = 0
        
        # C类：信息论特征
        if len(locus_heights) > 0:
            total_height = locus_heights.sum()
            if total_height > 0:
                locus_probs = locus_heights / total_height
                features['inter_locus_balance_entropy'] = calculate_entropy(locus_probs.values)
            else:
                features['inter_locus_balance_entropy'] = 0
            
            # 平均位点等位基因分布熵
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
            
            # 图谱完整性指标
            effective_loci_count = len(locus_groups)
            features['num_loci_with_effective_alleles'] = effective_loci_count
            features['num_loci_no_effective_alleles'] = max(0, 20 - effective_loci_count)
        
        # D类：降解特征
        if total_peaks > 1 and len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1]
        else:
            features['height_size_correlation'] = 0
        
        return features
    
    def _get_default_features(self, sample_file):
        """获取默认特征值"""
        default_features = {
            'Sample File': sample_file,
            'mac_profile': 0, 'total_distinct_alleles': 0, 'avg_alleles_per_locus': 0,
            'std_alleles_per_locus': 0, 'loci_gt3_alleles': 0, 'loci_gt4_alleles': 0,
            'loci_gt5_alleles': 0, 'loci_gt6_alleles': 0, 'allele_count_dist_entropy': 0,
            'avg_peak_height': 0, 'std_peak_height': 0, 'skewness_peak_height': 0,
            'kurtosis_peak_height': 0, 'avg_phr': 0, 'std_phr': 0, 'min_phr': 0,
            'median_phr': 0, 'num_loci_with_phr': 0, 'num_severe_imbalance_loci': 0,
            'ratio_severe_imbalance_loci': 0, 'inter_locus_balance_entropy': 0,
            'avg_locus_allele_entropy': 0, 'num_loci_with_effective_alleles': 0,
            'num_loci_no_effective_alleles': 20, 'height_size_correlation': 0
        }
        return default_features

class NoCPredictor:
    """NoC预测器，使用Q1训练的随机森林模型"""
    
    def __init__(self, model_path: str):
        self.model_data = None
        self.load_model(model_path)
        logger.info("NoC预测器初始化完成")
    
    def load_model(self, model_path: str):
        """加载Q1训练的模型"""
        try:
            self.model_data = joblib.load(model_path)
            logger.info(f"成功加载Q1模型: {model_path}")
        except Exception as e:
            logger.warning(f"Q1模型加载失败: {e}，将使用默认NoC=2")
            self.model_data = None
    
    def predict_noc(self, v5_features: Dict) -> Tuple[int, float]:
        """预测贡献者人数"""
        if self.model_data is None:
            return 2, 0.5
        
        try:
            model = self.model_data['model']
            scaler = self.model_data['scaler']
            label_encoder = self.model_data['label_encoder']
            selected_features = self.model_data['selected_features']
            
            # 准备特征向量
            feature_vector = []
            for feature_name in selected_features:
                if feature_name in v5_features:
                    feature_vector.append(v5_features[feature_name])
                else:
                    feature_vector.append(0.0)
            
            # 标准化
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # 预测
            y_pred_encoded = model.predict(X_scaled)[0]
            y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
            
            # 预测概率作为置信度
            y_proba = model.predict_proba(X_scaled)[0]
            confidence = np.max(y_proba)
            
            return int(y_pred), float(confidence)
            
        except Exception as e:
            logger.warning(f"NoC预测失败: {e}，使用默认值")
            return 2, 0.5

# =====================
# 3. 附件3基因型数据处理
# =====================
class Attachment3Processor:
    """附件3基因型数据处理器"""
    
    def __init__(self, att3_path: str):
        self.att3_path = att3_path
        self.genotype_data = {}
        self.load_attachment3_data()
        logger.info("附件3处理器初始化完成")
    
    def load_attachment3_data(self):
        """加载附件3基因型数据"""
        try:
            df_att3 = pd.read_csv(self.att3_path, encoding='utf-8')
            logger.info(f"成功加载附件3数据，形状: {df_att3.shape}")
            
            # 处理基因型数据
            for _, row in df_att3.iterrows():
                sample_id = row['Sample']
                contributor_id = row['Contributor']
                marker = row['Marker']
                
                # 提取基因型
                genotype = []
                for col in df_att3.columns:
                    if col.startswith('Allele'):
                        allele_val = row[col]
                        if pd.notna(allele_val):
                            genotype.append(str(allele_val))
                
                # 确保基因型是二倍体
                if len(genotype) == 1:
                    genotype.append(genotype[0])  # 纯合子
                elif len(genotype) > 2:
                    genotype = genotype[:2]  # 只取前两个等位基因
                
                # 标准化基因型表示
                genotype = tuple(sorted(genotype))
                
                # 存储
                if sample_id not in self.genotype_data:
                    self.genotype_data[sample_id] = {}
                if contributor_id not in self.genotype_data[sample_id]:
                    self.genotype_data[sample_id][contributor_id] = {}
                
                self.genotype_data[sample_id][contributor_id][marker] = genotype
            
            logger.info(f"成功处理{len(self.genotype_data)}个样本的基因型数据")
            
        except Exception as e:
            logger.error(f"附件3数据加载失败: {e}")
            self.genotype_data = {}
    
    def get_sample_genotypes(self, sample_id: str) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """获取指定样本的所有贡献者基因型"""
        return self.genotype_data.get(sample_id, {})
    
    def get_contributor_genotype(self, sample_id: str, contributor_id: str, marker: str) -> Optional[Tuple[str, str]]:
        """获取指定贡献者在指定位点的基因型"""
        try:
            return self.genotype_data[sample_id][contributor_id][marker]
        except KeyError:
            return None
    
    def get_available_samples(self) -> List[str]:
        """获取可用的样本ID列表"""
        return list(self.genotype_data.keys())
    
    def get_contributors_for_sample(self, sample_id: str) -> List[str]:
        """获取指定样本的所有贡献者ID"""
        return list(self.genotype_data.get(sample_id, {}).keys())

# =====================
# 4. 增强的伪频率计算器
# =====================
class EnhancedPseudoFrequencyCalculator:
    """增强的伪等位基因频率计算器，结合附件2和附件3数据"""
    
    def __init__(self, att2_data: Dict = None, att3_processor: Attachment3Processor = None):
        self.att2_data = att2_data or {}
        self.att3_processor = att3_processor
        self.frequency_cache = {}
        logger.info("增强伪频率计算器初始化完成")
    
    def calculate_pseudo_frequencies(self, locus: str, target_sample_id: str = None) -> Dict[str, float]:
        """计算位点的伪等位基因频率 w_l(a_k)"""
        cache_key = (locus, target_sample_id)
        if cache_key in self.frequency_cache:
            return self.frequency_cache[cache_key]
        
        # 收集等位基因
        all_alleles = []
        
        # 从附件2收集观测等位基因
        if self.att2_data:
            for sample_id, sample_peaks in self.att2_data.items():
                if not sample_peaks.empty:
                    locus_peaks = sample_peaks[sample_peaks['Marker'] == locus]
                    if not locus_peaks.empty:
                        alleles = locus_peaks['Allele'].tolist()
                        all_alleles.extend(alleles)
        
        # 从附件3收集基因型等位基因
        if self.att3_processor:
            for sample_id in self.att3_processor.get_available_samples():
                contributors = self.att3_processor.get_contributors_for_sample(sample_id)
                for contributor_id in contributors:
                    genotype = self.att3_processor.get_contributor_genotype(sample_id, contributor_id, locus)
                    if genotype:
                        all_alleles.extend(list(genotype))
        
        if not all_alleles:
            # 如果没有数据，返回均匀分布
            return {'10': 0.5, '11': 0.5}
        
        # 统计等位基因出现次数
        allele_counts = Counter(all_alleles)
        
        # 计算频率
        total_count = sum(allele_counts.values())
        frequencies = {}
        
        for allele, count in allele_counts.items():
            frequencies[allele] = count / total_count
        
        # 为未观测到但可能存在的等位基因分配最小频率
        unique_alleles = set(all_alleles)
        min_freq = 1 / (2 * len(unique_alleles) + len(allele_counts))
        
        # 标准化频率，为稀有等位基因保留概率空间
        freq_sum = sum(frequencies.values())
        if freq_sum > 0:
            adjustment_factor = (1 - min_freq * 2) / freq_sum
            for allele in frequencies:
                frequencies[allele] = frequencies[allele] * adjustment_factor + min_freq
        
        self.frequency_cache[cache_key] = frequencies
        return frequencies
    
    def calculate_genotype_prior(self, genotype: Tuple[str, str], 
                               frequencies: Dict[str, float]) -> float:
        """基于Hardy-Weinberg平衡计算基因型的先验概率"""
        a1, a2 = genotype
        f1 = frequencies.get(a1, 1e-6)
        f2 = frequencies.get(a2, 1e-6)
        
        if a1 == a2:  # 纯合子
            return f1 * f2
        else:  # 杂合子
            return 2 * f1 * f2

# =====================
# 5. 基因型采样和枚举器
# =====================
class EnhancedGenotypeEnumerator:
    """增强的基因型枚举器，结合已知基因型信息"""
    
    def __init__(self, att3_processor: Attachment3Processor = None, max_contributors: int = 5):
        self.att3_processor = att3_processor
        self.max_contributors = max_contributors
        self.memo = {}
        logger.info(f"增强基因型枚举器初始化，最大贡献者数：{max_contributors}")
    
    def enumerate_valid_genotype_sets(self, observed_alleles: List[str], 
                                    N: int, target_sample_id: str = None,
                                    locus: str = None, K_top: int = None) -> List[List[Tuple[str, str]]]:
        """枚举所有与观测等位基因兼容的基因型组合"""
        cache_key = (tuple(sorted(observed_alleles)), N, target_sample_id, locus, K_top)
        if cache_key in self.memo:
            return self.memo[cache_key]
        
        # 如果有已知基因型信息，优先使用
        if self.att3_processor and target_sample_id and locus:
            known_genotypes = self._get_known_genotypes(target_sample_id, locus)
            if known_genotypes:
                logger.debug(f"使用已知基因型信息，样本{target_sample_id}位点{locus}")
                valid_sets = self._enumerate_with_known_genotypes(
                    observed_alleles, N, known_genotypes)
            else:
                valid_sets = self._enumerate_unknown_genotypes(
                    observed_alleles, N, K_top)
        else:
            valid_sets = self._enumerate_unknown_genotypes(
                observed_alleles, N, K_top)
        
        self.memo[cache_key] = valid_sets
        logger.debug(f"为{N}个贡献者枚举了{len(valid_sets)}个有效基因型组合")
        return valid_sets
    
    def _get_known_genotypes(self, sample_id: str, locus: str) -> List[Tuple[str, str]]:
        """获取已知的基因型"""
        if not self.att3_processor:
            return []
        
        known_genotypes = []
        contributors = self.att3_processor.get_contributors_for_sample(sample_id)
        
        for contributor_id in contributors:
            genotype = self.att3_processor.get_contributor_genotype(sample_id, contributor_id, locus)
            if genotype:
                known_genotypes.append(genotype)
        
        return known_genotypes
    
    def _enumerate_with_known_genotypes(self, observed_alleles: List[str], 
                                      N: int, known_genotypes: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """使用已知基因型进行枚举"""
        # 如果已知基因型数量等于N，直接返回
        if len(known_genotypes) == N:
            if self._can_explain_alleles(known_genotypes, observed_alleles):
                return [known_genotypes]
            else:
                return []
        
        # 如果已知基因型数量少于N，需要推断剩余的
        if len(known_genotypes) < N:
            remaining_count = N - len(known_genotypes)
            
            # 生成可能的基因型
            A_l = observed_alleles
            possible_genotypes = []
            for i, a1 in enumerate(A_l):
                for j, a2 in enumerate(A_l):
                    if i <= j:
                        possible_genotypes.append((a1, a2))
            
            valid_sets = []
            
            # 枚举剩余基因型的所有组合
            for remaining_genotypes in itertools.combinations_with_replacement(possible_genotypes, remaining_count):
                full_genotype_set = known_genotypes + list(remaining_genotypes)
                if self._can_explain_alleles(full_genotype_set, observed_alleles):
                    valid_sets.append(full_genotype_set)
            
            return valid_sets
        
        # 如果已知基因型数量大于N，选择子集
        if len(known_genotypes) > N:
            valid_sets = []
            for genotype_subset in itertools.combinations(known_genotypes, N):
                if self._can_explain_alleles(list(genotype_subset), observed_alleles):
                    valid_sets.append(list(genotype_subset))
            return valid_sets
        
        return []
    
    def _enumerate_unknown_genotypes(self, observed_alleles: List[str], 
                                   N: int, K_top: int = None) -> List[List[Tuple[str, str]]]:
        """枚举未知基因型（使用Q2的方法）"""
        if N <= 3:
            return self._enumerate_all_combinations(observed_alleles, N)
        else:
            return self._enumerate_k_top_combinations(observed_alleles, N, K_top)
    
    def _enumerate_all_combinations(self, observed_alleles: List[str], N: int) -> List[List[Tuple[str, str]]]:
        """枚举所有可能的基因型组合（适用于N<=3）"""
        A_l = observed_alleles
        all_genotype_sets = []
        
        # 生成所有可能的基因型
        possible_genotypes = []
        for i, a1 in enumerate(A_l):
            for j, a2 in enumerate(A_l):
                if i <= j:
                    possible_genotypes.append((a1, a2))
        
        # 生成N个个体的所有基因型组合
        for genotype_combination in itertools.product(possible_genotypes, repeat=N):
            if self._can_explain_alleles(list(genotype_combination), A_l):
                all_genotype_sets.append(list(genotype_combination))
        
        return all_genotype_sets
    
    def _enumerate_k_top_combinations(self, observed_alleles: List[str], 
                                    N: int, K_top: int) -> List[List[Tuple[str, str]]]:
        """使用K-top采样策略（适用于N>=4）"""
        if K_top is None:
            K_top = min(800, max(100, len(observed_alleles) ** N))
        
        A_l = observed_alleles
        possible_genotypes = []
        
        # 生成所有可能的基因型
        for i, a1 in enumerate(A_l):
            for j, a2 in enumerate(A_l):
                if i <= j:
                    possible_genotypes.append((a1, a2))
        
        sampled_sets = []
        max_attempts = K_top * 10
        attempts = 0
        
        while len(sampled_sets) < K_top and attempts < max_attempts:
            attempts += 1
            
            # 随机选择N个基因型
            genotype_combination = [
                possible_genotypes[np.random.randint(len(possible_genotypes))]
                for _ in range(N)
            ]
            
            # 检查是否能解释观测等位基因
            if self._can_explain_alleles(genotype_combination, A_l):
                if genotype_combination not in sampled_sets:
                    sampled_sets.append(genotype_combination)
        
        return sampled_sets
    
    def _can_explain_alleles(self, genotype_set: List[Tuple[str, str]], 
                           observed_alleles: List[str]) -> bool:
        """检查基因型组合是否能解释所有观测到的等位基因"""
        # 从基因型组合中收集所有等位基因
        genotype_alleles = set()
        for genotype in genotype_set:
            if genotype is not None:
                genotype_alleles.update(genotype)
        
        # 检查所有观测等位基因是否都能被基因型组合解释
        observed_set = set(observed_alleles)
        return observed_set.issubset(genotype_alleles)

# =====================
# 6. 增强的MCMC基因型推断器
# =====================
class EnhancedMCMCGenotypeInferencer:
    """增强的MCMC基因型推断器，集成Q1、Q2方法和附件3信息"""
    
    def __init__(self, q1_model_path: str = None, att3_processor: Attachment3Processor = None):
        # 初始化组件
        self.att3_processor = att3_processor
        self.pseudo_freq_calculator = None
        self.genotype_enumerator = EnhancedGenotypeEnumerator(att3_processor)
        self.noc_predictor = NoCPredictor(q1_model_path) if q1_model_path else None
        self.q1_feature_engineering = Q1FeatureEngineering()
        self.v5_features = None
        
        # MCMC参数
        self.n_iterations = config.N_ITERATIONS
        self.n_warmup = config.N_WARMUP
        self.n_chains = config.N_CHAINS
        self.thinning = config.THINNING
        self.K_top = config.K_TOP
        
        logger.info("增强MCMC基因型推断器初始化完成")
    
    def set_att2_data(self, att2_data: Dict):
        """设置附件2数据"""
        self.pseudo_freq_calculator = EnhancedPseudoFrequencyCalculator(att2_data, self.att3_processor)
        logger.info("附件2数据已设置")
    
    def predict_noc_from_sample(self, sample_data: pd.DataFrame) -> Tuple[int, float, Dict]:
        """从样本数据预测NoC"""
        # 处理峰数据
        sample_peaks = self.q1_feature_engineering.process_peaks_simplified(sample_data)
        
        # 提取V5特征
        sample_file = sample_data.iloc[0]['Sample File']
        v5_features = self.q1_feature_engineering.extract_v5_features(sample_file, sample_peaks)
        self.v5_features = v5_features
        
        # 预测NoC
        if self.noc_predictor:
            predicted_noc, confidence = self.noc_predictor.predict_noc(v5_features)
        else:
            predicted_noc, confidence = 2, 0.5
        
        logger.info(f"样本 {sample_file} 预测NoC: {predicted_noc} (置信度: {confidence:.3f})")
        
        return predicted_noc, confidence, v5_features
    
    def calculate_locus_marginalized_likelihood(self, locus_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              target_sample_id: str = None) -> float:
        """计算单个位点的边缘化似然函数"""
        if self.pseudo_freq_calculator is None:
            raise ValueError("请先设置附件2数据")
        
        locus = locus_data['locus']
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        # 枚举有效基因型组合
        K_top = self.K_top if N >= 4 else None
        valid_genotype_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
            observed_alleles, N, target_sample_id, locus, K_top)
        
        if not valid_genotype_sets:
            logger.warning(f"位点{locus}没有有效的基因型组合")
            return -1e10
        
        # 计算伪等位基因频率
        pseudo_frequencies = self.pseudo_freq_calculator.calculate_pseudo_frequencies(
            locus, target_sample_id)
        
        # 计算位点特异性参数（基于V5特征）
        gamma_l = self._calculate_gamma_l(locus)
        sigma_var_l = self._calculate_sigma_var_l(locus)
        
        # 边缘化求和
        log_marginal_likelihood = -np.inf
        likelihood_terms = []
        
        for genotype_set in valid_genotype_sets:
            # 基因型组合先验概率
            log_prior = self._calculate_genotype_set_prior(genotype_set, pseudo_frequencies)
            
            # 条件似然
            log_conditional_likelihood = self._calculate_conditional_likelihood(
                observed_alleles, observed_heights, genotype_set, mixture_ratios,
                gamma_l, sigma_var_l, locus)
            
            # 联合概率
            log_joint = log_prior + log_conditional_likelihood
            likelihood_terms.append(log_joint)
        
        # logsumexp边缘化
        if likelihood_terms:
            log_marginal_likelihood = self._logsumexp(likelihood_terms)
        
        return log_marginal_likelihood
    
    def _calculate_gamma_l(self, locus: str) -> float:
        """基于V5特征计算位点特异性放大效率"""
        if self.v5_features is None:
            return 1000.0
        
        avg_height = self.v5_features.get('avg_peak_height', 1000.0)
        inter_locus_entropy = self.v5_features.get('inter_locus_balance_entropy', 1.0)
        
        k_gamma = 1.0
        gamma_base = k_gamma * avg_height
        
        L_exp = 20
        beta = 1.5
        
        if L_exp > 1:
            w_entropy = (1 - inter_locus_entropy / np.log(L_exp)) ** beta
            P_l = 1.0 / L_exp
            gamma_l = gamma_base * (1 + w_entropy * ((P_l * L_exp) - 1))
        else:
            gamma_l = gamma_base
            
        return max(gamma_l, 1e-3)
    
    def _calculate_sigma_var_l(self, locus: str) -> float:
        """基于V5特征计算位点方差参数"""
        if self.v5_features is None:
            return 0.1
        
        avg_height = self.v5_features.get('avg_peak_height', 1000.0)
        R_PHR = self.v5_features.get('ratio_severe_imbalance_loci', 0.0)
        gamma_1 = self.v5_features.get('skewness_peak_height', 0.0)
        H_a_bar = max(self.v5_features.get('avg_locus_allele_entropy', 1.0), 1e-6)
        
        sigma_var_base = 0.1
        c1, c2, c3 = 0.5, 0.3, 0.2
        
        A_f = 1.0
        B_f = 0.001
        h_0f = 1000.0
        
        f_h = 1 + A_f / (1 + np.exp(B_f * (avg_height - h_0f)))
        
        sigma_var = (sigma_var_base * 
                    (1 + c1 * R_PHR + c2 * abs(gamma_1) + c3 * (1 / H_a_bar)) * 
                    f_h)
        
        return max(sigma_var, 0.01)
    
    def _calculate_genotype_set_prior(self, genotype_set: List[Tuple[str, str]], 
                                    frequencies: Dict[str, float]) -> float:
        """计算基因型组合的先验概率"""
        log_prior = 0.0
        for genotype in genotype_set:
            if genotype is not None:
                prior = self.pseudo_freq_calculator.calculate_genotype_prior(genotype, frequencies)
                log_prior += np.log(max(prior, 1e-10))
        return log_prior
    
    def _calculate_conditional_likelihood(self, observed_alleles: List[str],
                                        observed_heights: Dict[str, float],
                                        genotype_set: List[Tuple[str, str]],
                                        mixture_ratios: np.ndarray,
                                        gamma_l: float, sigma_var_l: float,
                                        locus: str) -> float:
        """计算给定基因型组合的条件似然"""
        log_likelihood = 0.0
        
        # 观测等位基因的峰高似然
        for allele in observed_alleles:
            observed_height = observed_heights.get(allele, 0.0)
            if observed_height > 0:
                mu_exp = self._calculate_expected_height(
                    allele, locus, genotype_set, mixture_ratios, gamma_l)
                
                if mu_exp > 1e-6:
                    log_mu = np.log(mu_exp) - sigma_var_l**2 / 2
                    log_likelihood += stats.lognorm.logpdf(
                        observed_height, sigma_var_l, scale=np.exp(log_mu))
                else:
                    log_likelihood += -1e6
        
        # ADO似然
        genotype_alleles = set()
        for genotype in genotype_set:
            if genotype is not None:
                genotype_alleles.update(genotype)
        
        dropped_alleles = genotype_alleles - set(observed_alleles)
        for allele in dropped_alleles:
            mu_exp_ado = self._calculate_expected_height(
                allele, locus, genotype_set, mixture_ratios, gamma_l)
            P_ado = self._calculate_ado_probability(mu_exp_ado)
            log_likelihood += np.log(max(P_ado, 1e-10))
        
        return log_likelihood
    
    def _calculate_expected_height(self, allele: str, locus: str,
                                 genotype_set: List[Tuple[str, str]],
                                 mixture_ratios: np.ndarray,
                                 gamma_l: float) -> float:
        """计算等位基因的期望峰高"""
        mu_allele = 0.0
        
        # 直接等位基因贡献
        for i, genotype in enumerate(genotype_set):
            if genotype is not None:
                C_copy = self._get_copy_number(allele, genotype)
                
                if C_copy > 0:
                    allele_size = self._get_allele_size(allele, locus)
                    D_F = self._calculate_degradation_factor(allele_size)
                    mu_allele += gamma_l * mixture_ratios[i] * C_copy * D_F
        
        # Stutter贡献
        mu_stutter = self._calculate_stutter_contribution(
            allele, locus, genotype_set, mixture_ratios, gamma_l)
        
        return mu_allele + mu_stutter
    
    def _get_copy_number(self, allele: str, genotype: Tuple[str, str]) -> float:
        """计算等位基因在基因型中的拷贝数"""
        if genotype is None:
            return 0.0
        
        count = sum(1 for gt_allele in genotype if gt_allele == allele)
        
        if len(set(genotype)) == 1 and allele in genotype:
            f_homo = 1.0
            return 2.0 * f_homo
        
        return float(count)
    
    def _get_allele_size(self, allele: str, locus: str) -> float:
        """获取等位基因片段大小"""
        try:
            allele_num = float(allele)
            # 简化的片段大小计算
            base_size = 150.0
            repeat_length = 4.0
            return base_size + allele_num * repeat_length
        except ValueError:
            return 150.0
    
    def _calculate_degradation_factor(self, allele_size: float) -> float:
        """计算降解因子"""
        if self.v5_features is None:
            return 1.0
        
        k_deg_0 = 0.001
        size_ref = 200.0
        alpha = 1.0
        
        height_size_corr = self.v5_features.get('height_size_correlation', 0.0)
        k_deg = k_deg_0 * max(0, -height_size_corr) ** alpha
        
        D_F = np.exp(-k_deg * max(0, allele_size - size_ref))
        
        return max(D_F, 1e-6)
    
    def _calculate_stutter_contribution(self, target_allele: str, locus: str,
                                      genotype_set: List[Tuple[str, str]],
                                      mixture_ratios: np.ndarray, gamma_l: float) -> float:
        """计算Stutter贡献"""
        stutter_ratio = 0.05
        
        try:
            target_allele_num = float(target_allele)
            parent_allele_num = target_allele_num + 1
            parent_allele = str(int(parent_allele_num)) if parent_allele_num.is_integer() else str(parent_allele_num)
            
            mu_parent = 0.0
            for i, genotype in enumerate(genotype_set):
                if genotype is not None:
                    C_copy = self._get_copy_number(parent_allele, genotype)
                    if C_copy > 0:
                        parent_size = self._get_allele_size(parent_allele, locus)
                        D_F = self._calculate_degradation_factor(parent_size)
                        mu_parent += gamma_l * mixture_ratios[i] * C_copy * D_F
            
            return stutter_ratio * mu_parent
        except:
            return 0.0
    
    def _calculate_ado_probability(self, expected_height: float) -> float:
        """计算等位基因缺失概率"""
        if expected_height <= 0:
            return 0.99
        
        H_50 = 200.0
        s_ado = 0.01
        P_ado = 1.0 / (1.0 + np.exp(s_ado * (expected_height - H_50)))
        return np.clip(P_ado, 1e-6, 0.99)
    
    def _logsumexp(self, log_values: List[float]) -> float:
        """数值稳定的logsumexp计算"""
        if not log_values:
            return -np.inf
        
        max_val = max(log_values)
        if max_val == -np.inf:
            return -np.inf
        
        sum_exp = sum(np.exp(val - max_val) for val in log_values)
        return max_val + np.log(sum_exp)
    
    def calculate_total_marginalized_likelihood(self, observed_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              target_sample_id: str = None) -> float:
        """计算总的边缘化似然函数"""
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self.calculate_locus_marginalized_likelihood(
                locus_data, N, mixture_ratios, target_sample_id)
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def mcmc_genotype_sampler(self, observed_data: Dict, N: int, 
                            target_sample_id: str = None) -> Dict:
        """MCMC基因型采样器"""
        logger.info(f"开始MCMC基因型采样，样本: {target_sample_id}, 贡献者数量: {N}")
        logger.info(f"总迭代次数: {self.n_iterations}, 预热次数: {self.n_warmup}")
        
        # 初始化参数
        mixture_ratios = np.random.dirichlet(np.ones(N))
        
        # 初始化每个位点的基因型组合
        current_genotypes = {}
        for locus, locus_data in observed_data.items():
            observed_alleles = locus_data['alleles']
            valid_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
                observed_alleles, N, target_sample_id, locus, self.K_top)
            
            if valid_sets:
                current_genotypes[locus] = valid_sets[np.random.randint(len(valid_sets))]
            else:
                # 默认基因型
                default_genotype = [(observed_alleles[0], observed_alleles[0])] * N
                current_genotypes[locus] = default_genotype
        
        # 存储MCMC样本
        samples = {
            'mixture_ratios': [],
            'genotypes': [],
            'log_likelihood': [],
            'acceptance_info': []
        }
        
        # 计算初始似然
        current_log_likelihood = self.calculate_total_marginalized_likelihood(
            observed_data, N, mixture_ratios, target_sample_id)
        
        # MCMC主循环
        n_accepted_mx = 0
        n_accepted_gt = 0
        acceptance_details = []
        
        # 自适应步长
        step_size = 0.05
        adaptation_interval = 500
        target_acceptance = 0.35
        
        for iteration in range(self.n_iterations):
            if iteration % 1000 == 0:
                acceptance_rate_mx = n_accepted_mx / max(iteration, 1)
                acceptance_rate_gt = n_accepted_gt / max(iteration, 1)
                logger.info(f"迭代 {iteration}/{self.n_iterations}, "
                          f"混合比例接受率: {acceptance_rate_mx:.3f}, "
                          f"基因型接受率: {acceptance_rate_gt:.3f}, "
                          f"当前似然: {current_log_likelihood:.2f}")
            
            # 步骤1: 更新混合比例
            proposed_ratios = self._propose_mixture_ratios(mixture_ratios, step_size)
            
            proposed_log_likelihood = self.calculate_total_marginalized_likelihood(
                observed_data, N, proposed_ratios, target_sample_id)
            
            log_ratio_mx = proposed_log_likelihood - current_log_likelihood
            accept_prob_mx = min(1.0, np.exp(log_ratio_mx))
            
            if np.random.random() < accept_prob_mx:
                mixture_ratios = proposed_ratios
                current_log_likelihood = proposed_log_likelihood
                n_accepted_mx += 1
                accepted_mx = True
            else:
                accepted_mx = False
            
            # 步骤2: 更新基因型（每个位点）
            accepted_gt = False
            for locus in observed_data.keys():
                proposed_genotypes = current_genotypes.copy()
                
                # 为当前位点提议新的基因型组合
                observed_alleles = observed_data[locus]['alleles']
                valid_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
                    observed_alleles, N, target_sample_id, locus, self.K_top)
                
                if valid_sets and len(valid_sets) > 1:
                    new_genotype_set = valid_sets[np.random.randint(len(valid_sets))]
                    proposed_genotypes[locus] = new_genotype_set
                    
                    # 计算提议状态的似然（只需要重新计算这个位点）
                    old_genotype_set = current_genotypes[locus]
                    current_genotypes[locus] = new_genotype_set
                    
                    proposed_log_likelihood_gt = self.calculate_total_marginalized_likelihood(
                        observed_data, N, mixture_ratios, target_sample_id)
                    
                    log_ratio_gt = proposed_log_likelihood_gt - current_log_likelihood
                    accept_prob_gt = min(1.0, np.exp(log_ratio_gt))
                    
                    if np.random.random() < accept_prob_gt:
                        current_genotypes = proposed_genotypes
                        current_log_likelihood = proposed_log_likelihood_gt
                        n_accepted_gt += 1
                        accepted_gt = True
                    else:
                        # 恢复原来的基因型
                        current_genotypes[locus] = old_genotype_set
            
            # 记录接受信息
            acceptance_details.append({
                'iteration': iteration,
                'accepted_mx': accepted_mx,
                'accepted_gt': accepted_gt,
                'log_likelihood': current_log_likelihood
            })
            
            # 自适应步长调整
            if iteration > 0 and iteration % adaptation_interval == 0 and iteration < self.n_warmup:
                recent_acceptance_mx = np.mean([a['accepted_mx'] for a in acceptance_details[-adaptation_interval:]])
                if recent_acceptance_mx < target_acceptance - 0.05:
                    step_size *= 0.9
                elif recent_acceptance_mx > target_acceptance + 0.05:
                    step_size *= 1.1
                step_size = np.clip(step_size, 0.01, 0.2)
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['genotypes'].append(deepcopy(current_genotypes))
                samples['log_likelihood'].append(current_log_likelihood)
        
        final_acceptance_rate_mx = n_accepted_mx / self.n_iterations
        final_acceptance_rate_gt = n_accepted_gt / self.n_iterations
        
        logger.info(f"MCMC完成，混合比例接受率: {final_acceptance_rate_mx:.3f}, "
                   f"基因型接受率: {final_acceptance_rate_gt:.3f}")
        logger.info(f"有效样本数: {len(samples['mixture_ratios'])}")
        
        return {
            'samples': samples,
            'acceptance_rate_mx': final_acceptance_rate_mx,
            'acceptance_rate_gt': final_acceptance_rate_gt,
            'n_samples': len(samples['mixture_ratios']),
            'acceptance_details': acceptance_details,
            'final_step_size': step_size,
            'converged': 0.15 <= final_acceptance_rate_mx <= 0.6 and 0.05 <= final_acceptance_rate_gt <= 0.4
        }
    
    def _propose_mixture_ratios(self, current_ratios: np.ndarray, 
                               step_size: float = 0.05) -> np.ndarray:
        """使用Dirichlet分布提议新的混合比例"""
        concentration = current_ratios / step_size
        concentration = np.maximum(concentration, 0.1)
        
        new_ratios = np.random.dirichlet(concentration)
        new_ratios = np.maximum(new_ratios, 1e-6)
        new_ratios = new_ratios / np.sum(new_ratios)
        
        return new_ratios

# =====================
# 7. 基因型匹配评估器
# =====================
class GenotypeMatchEvaluator:
    """基因型匹配评估器，评估推断准确性"""
    
    def __init__(self, att3_processor: Attachment3Processor):
        self.att3_processor = att3_processor
        logger.info("基因型匹配评估器初始化完成")
    
    def evaluate_genotype_inference(self, sample_id: str, mcmc_results: Dict, 
                                  observed_loci: List[str]) -> Dict:
        """评估基因型推断的准确性"""
        if not mcmc_results['samples']['genotypes']:
            return {'error': 'No MCMC samples available'}
        
        # 获取真实基因型
        true_genotypes = self.att3_processor.get_sample_genotypes(sample_id)
        if not true_genotypes:
            return {'error': f'No true genotypes available for sample {sample_id}'}
        
        # 计算后验众数基因型
        posterior_genotypes = self._calculate_posterior_mode_genotypes(
            mcmc_results['samples']['genotypes'], observed_loci)
        
        # 计算匹配指标
        evaluation_results = {}
        
        # 1. 基因型一致性率 (Genotype Concordance Rate, GCR)
        gcr_results = self._calculate_genotype_concordance_rate(
            true_genotypes, posterior_genotypes, observed_loci)
        evaluation_results['genotype_concordance'] = gcr_results
        
        # 2. 等位基因一致性率 (Allele Concordance Rate, ACR)
        acr_results = self._calculate_allele_concordance_rate(
            true_genotypes, posterior_genotypes, observed_loci)
        evaluation_results['allele_concordance'] = acr_results
        
        # 3. 位点特异性分析
        locus_specific_results = self._analyze_locus_specific_performance(
            true_genotypes, posterior_genotypes, observed_loci)
        evaluation_results['locus_specific'] = locus_specific_results
        
        # 4. 贡献者特异性分析
        contributor_specific_results = self._analyze_contributor_specific_performance(
            true_genotypes, posterior_genotypes, observed_loci)
        evaluation_results['contributor_specific'] = contributor_specific_results
        
        # 5. 总体性能摘要
        overall_summary = self._generate_overall_summary(evaluation_results)
        evaluation_results['overall_summary'] = overall_summary
        
        return evaluation_results
    
    def _calculate_posterior_mode_genotypes(self, genotype_samples: List[Dict], 
                                          observed_loci: List[str]) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """计算后验众数基因型"""
        posterior_genotypes = {}
        
        for locus in observed_loci:
            locus_genotype_counts = defaultdict(lambda: defaultdict(int))
            
            # 统计每个贡献者在每个位点的基因型频次
            for sample in genotype_samples:
                if locus in sample:
                    genotype_set = sample[locus]
                    for i, genotype in enumerate(genotype_set):
                        if genotype is not None:
                            locus_genotype_counts[i][genotype] += 1
            
            # 找出每个贡献者的众数基因型
            posterior_genotypes[locus] = {}
            for contributor_idx, genotype_counts in locus_genotype_counts.items():
                if genotype_counts:
                    mode_genotype = max(genotype_counts.items(), key=lambda x: x[1])[0]
                    posterior_genotypes[locus][contributor_idx] = mode_genotype
        
        return posterior_genotypes
    
    def _calculate_genotype_concordance_rate(self, true_genotypes: Dict, 
                                           posterior_genotypes: Dict, 
                                           observed_loci: List[str]) -> Dict:
        """计算基因型一致性率"""
        total_comparisons = 0
        correct_genotypes = 0
        locus_results = {}
        
        for locus in observed_loci:
            if locus not in posterior_genotypes:
                continue
                
            locus_total = 0
            locus_correct = 0
            
            # 比较每个已知贡献者的基因型
            for contributor_id, true_genotype in true_genotypes.items():
                if locus in true_genotype:
                    true_gt = true_genotype[locus]
                    
                    # 寻找最匹配的推断基因型
                    best_match = False
                    for inferred_idx, inferred_gt in posterior_genotypes[locus].items():
                        if self._genotypes_match(true_gt, inferred_gt):
                            best_match = True
                            break
                    
                    locus_total += 1
                    total_comparisons += 1
                    
                    if best_match:
                        locus_correct += 1
                        correct_genotypes += 1
            
            if locus_total > 0:
                locus_results[locus] = {
                    'concordance_rate': locus_correct / locus_total,
                    'total_comparisons': locus_total,
                    'correct_matches': locus_correct
                }
        
        overall_gcr = correct_genotypes / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'overall_gcr': overall_gcr,
            'locus_specific': locus_results,
            'total_comparisons': total_comparisons,
            'correct_genotypes': correct_genotypes
        }
    
    def _calculate_allele_concordance_rate(self, true_genotypes: Dict, 
                                         posterior_genotypes: Dict, 
                                         observed_loci: List[str]) -> Dict:
        """计算等位基因一致性率"""
        total_alleles = 0
        correct_alleles = 0
        locus_results = {}
        
        for locus in observed_loci:
            if locus not in posterior_genotypes:
                continue
                
            locus_total = 0
            locus_correct = 0
            
            # 收集真实等位基因
            true_alleles = set()
            for contributor_id, true_genotype in true_genotypes.items():
                if locus in true_genotype:
                    true_gt = true_genotype[locus]
                    true_alleles.update(true_gt)
            
            # 收集推断等位基因
            inferred_alleles = set()
            for inferred_idx, inferred_gt in posterior_genotypes[locus].items():
                inferred_alleles.update(inferred_gt)
            
            # 计算等位基因层面的一致性
            for allele in true_alleles:
                locus_total += 1
                total_alleles += 1
                if allele in inferred_alleles:
                    locus_correct += 1
                    correct_alleles += 1
            
            if locus_total > 0:
                locus_results[locus] = {
                    'acr': locus_correct / locus_total,
                    'true_alleles': list(true_alleles),
                    'inferred_alleles': list(inferred_alleles),
                    'common_alleles': list(true_alleles & inferred_alleles)
                }
        
        overall_acr = correct_alleles / total_alleles if total_alleles > 0 else 0
        
        return {
            'overall_acr': overall_acr,
            'locus_specific': locus_results,
            'total_alleles': total_alleles,
            'correct_alleles': correct_alleles
        }
    
    def _genotypes_match(self, true_genotype: Tuple[str, str], 
                        inferred_genotype: Tuple[str, str]) -> bool:
        """检查两个基因型是否匹配"""
        true_set = set(true_genotype)
        inferred_set = set(inferred_genotype)
        return true_set == inferred_set
    
    def _analyze_locus_specific_performance(self, true_genotypes: Dict, 
                                          posterior_genotypes: Dict, 
                                          observed_loci: List[str]) -> Dict:
        """分析位点特异性性能"""
        locus_performance = {}
        
        for locus in observed_loci:
            if locus not in posterior_genotypes:
                continue
            
            # 统计该位点的性能
            performance_metrics = {
                'total_contributors': len(true_genotypes),
                'inferred_contributors': len(posterior_genotypes[locus]),
                'exact_matches': 0,
                'partial_matches': 0,
                'mismatches': 0
            }
            
            for contributor_id, true_genotype in true_genotypes.items():
                if locus in true_genotype:
                    true_gt = true_genotype[locus]
                    
                    # 寻找最佳匹配
                    best_match_score = 0
                    for inferred_idx, inferred_gt in posterior_genotypes[locus].items():
                        match_score = self._calculate_genotype_similarity(true_gt, inferred_gt)
                        best_match_score = max(best_match_score, match_score)
                    
                    if best_match_score == 1.0:
                        performance_metrics['exact_matches'] += 1
                    elif best_match_score > 0:
                        performance_metrics['partial_matches'] += 1
                    else:
                        performance_metrics['mismatches'] += 1
            
            # 计算位点特异性指标
            total_contributors = performance_metrics['total_contributors']
            if total_contributors > 0:
                performance_metrics['exact_match_rate'] = performance_metrics['exact_matches'] / total_contributors
                performance_metrics['partial_match_rate'] = performance_metrics['partial_matches'] / total_contributors
                performance_metrics['mismatch_rate'] = performance_metrics['mismatches'] / total_contributors
            
            locus_performance[locus] = performance_metrics
        
        return locus_performance
    
    def _analyze_contributor_specific_performance(self, true_genotypes: Dict, 
                                                posterior_genotypes: Dict, 
                                                observed_loci: List[str]) -> Dict:
        """分析贡献者特异性性能"""
        contributor_performance = {}
        
        for contributor_id, true_genotype in true_genotypes.items():
            performance_metrics = {
                'total_loci': len([l for l in observed_loci if l in true_genotype]),
                'exact_matches': 0,
                'partial_matches': 0,
                'mismatches': 0
            }
            
            for locus in observed_loci:
                if locus in true_genotype and locus in posterior_genotypes:
                    true_gt = true_genotype[locus]
                    
                    # 寻找最佳匹配
                    best_match_score = 0
                    for inferred_idx, inferred_gt in posterior_genotypes[locus].items():
                        match_score = self._calculate_genotype_similarity(true_gt, inferred_gt)
                        best_match_score = max(best_match_score, match_score)
                    
                    if best_match_score == 1.0:
                        performance_metrics['exact_matches'] += 1
                    elif best_match_score > 0:
                        performance_metrics['partial_matches'] += 1
                    else:
                        performance_metrics['mismatches'] += 1
            
            # 计算贡献者特异性指标
            total_loci = performance_metrics['total_loci']
            if total_loci > 0:
                performance_metrics['exact_match_rate'] = performance_metrics['exact_matches'] / total_loci
                performance_metrics['partial_match_rate'] = performance_metrics['partial_matches'] / total_loci
                performance_metrics['mismatch_rate'] = performance_metrics['mismatches'] / total_loci
            
            contributor_performance[contributor_id] = performance_metrics
        
        return contributor_performance
    
    def _calculate_genotype_similarity(self, gt1: Tuple[str, str], 
                                     gt2: Tuple[str, str]) -> float:
        """计算两个基因型的相似度"""
        set1 = set(gt1)
        set2 = set(gt2)
        
        if set1 == set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_overall_summary(self, evaluation_results: Dict) -> Dict:
        """生成总体性能摘要"""
        summary = {}
        
        # 基因型一致性摘要
        if 'genotype_concordance' in evaluation_results:
            gcr = evaluation_results['genotype_concordance']
            summary['genotype_concordance_rate'] = gcr['overall_gcr']
            summary['total_genotype_comparisons'] = gcr['total_comparisons']
        
        # 等位基因一致性摘要
        if 'allele_concordance' in evaluation_results:
            acr = evaluation_results['allele_concordance']
            summary['allele_concordance_rate'] = acr['overall_acr']
            summary['total_allele_comparisons'] = acr['total_alleles']
        
        # 位点性能摘要
        if 'locus_specific' in evaluation_results:
            locus_perf = evaluation_results['locus_specific']
            locus_exact_rates = [perf['exact_match_rate'] for perf in locus_perf.values() 
                               if 'exact_match_rate' in perf]
            if locus_exact_rates:
                summary['avg_locus_exact_match_rate'] = np.mean(locus_exact_rates)
                summary['std_locus_exact_match_rate'] = np.std(locus_exact_rates)
        
        # 贡献者性能摘要
        if 'contributor_specific' in evaluation_results:
            contrib_perf = evaluation_results['contributor_specific']
            contrib_exact_rates = [perf['exact_match_rate'] for perf in contrib_perf.values() 
                                 if 'exact_match_rate' in perf]
            if contrib_exact_rates:
                summary['avg_contributor_exact_match_rate'] = np.mean(contrib_exact_rates)
                summary['std_contributor_exact_match_rate'] = np.std(contrib_exact_rates)
        
        # 性能等级评定
        if 'genotype_concordance_rate' in summary:
            gcr = summary['genotype_concordance_rate']
            if gcr >= 0.95:
                summary['performance_grade'] = 'A (优秀)'
            elif gcr >= 0.85:
                summary['performance_grade'] = 'B (良好)'
            elif gcr >= 0.70:
                summary['performance_grade'] = 'C (一般)'
            elif gcr >= 0.50:
                summary['performance_grade'] = 'D (较差)'
            else:
                summary['performance_grade'] = 'F (很差)'
        
        return summary

# =====================
# 8. 主分析流水线
# =====================
class Q3EnhancedPipeline:
    """问题三增强分析流水线"""
    
    def __init__(self, q1_model_path: str = None, att3_path: str = None):
        # 初始化组件
        self.att3_processor = Attachment3Processor(att3_path) if att3_path else None
        self.mcmc_inferencer = EnhancedMCMCGenotypeInferencer(q1_model_path, self.att3_processor)
        self.match_evaluator = GenotypeMatchEvaluator(self.att3_processor) if self.att3_processor else None
        self.q1_feature_engineering = Q1FeatureEngineering()
        
        logger.info("Q3增强分析流水线初始化完成")
    
    def load_data(self, att1_path: str = None, att2_path: str = None):
        """加载数据"""
        self.att2_data = {}
        
        if att2_path and os.path.exists(att2_path):
            try:
                df_att2 = pd.read_csv(att2_path, encoding='utf-8')
                for sample_file, group in df_att2.groupby('Sample File'):
                    sample_peaks = self.q1_feature_engineering.process_peaks_simplified(group)
                    self.att2_data[sample_file] = sample_peaks
                
                self.mcmc_inferencer.set_att2_data(self.att2_data)
                logger.info(f"成功加载附件2数据，包含{len(self.att2_data)}个样本")
            except Exception as e:
                logger.error(f"附件2数据加载失败: {e}")
    
    def analyze_single_sample(self, sample_id: str, att1_or_att2_data: pd.DataFrame) -> Dict:
        """分析单个样本的基因型"""
        logger.info(f"开始分析样本: {sample_id}")
        
        # 步骤1: 预测NoC和提取V5特征
        predicted_noc, noc_confidence, v5_features = self.mcmc_inferencer.predict_noc_from_sample(att1_or_att2_data)
        
        # 步骤2: 处理STR数据
        sample_peaks = self.q1_feature_engineering.process_peaks_simplified(att1_or_att2_data)
        
        if sample_peaks.empty:
            logger.warning(f"样本 {sample_id} 没有有效的峰数据")
            return self._get_default_result(sample_id, predicted_noc, noc_confidence, v5_features)
        
        # 步骤3: 准备观测数据
        observed_data = {}
        observed_loci = []
        
        for locus, locus_group in sample_peaks.groupby('Marker'):
            alleles = locus_group['Allele'].tolist()
            heights = dict(zip(locus_group['Allele'], locus_group['Height']))
            
            observed_data[locus] = {
                'locus': locus,
                'alleles': alleles,
                'heights': heights
            }
            observed_loci.append(locus)
        
        logger.info(f"样本 {sample_id} 包含{len(observed_data)}个有效位点")
        
        # 步骤4: MCMC基因型推断
        start_time = time.time()
        mcmc_results = self.mcmc_inferencer.mcmc_genotype_sampler(
            observed_data, predicted_noc, sample_id)
        end_time = time.time()
        
        # 步骤5: 基因型匹配评估（如果有真实基因型）
        evaluation_results = None
        if self.match_evaluator and sample_id in self.att3_processor.get_available_samples():
            evaluation_results = self.match_evaluator.evaluate_genotype_inference(
                sample_id, mcmc_results, observed_loci)
        
        # 步骤6: 生成后验摘要
        posterior_summary = self.generate_genotype_posterior_summary(mcmc_results, predicted_noc, observed_loci)
        
        # 步骤7: 收敛性诊断
        convergence_diagnostics = self.analyze_mcmc_convergence(mcmc_results, predicted_noc)
        
        result = {
            'sample_id': sample_id,
            'predicted_noc': predicted_noc,
            'noc_confidence': noc_confidence,
            'v5_features': v5_features,
            'mcmc_results': mcmc_results,
            'posterior_summary': posterior_summary,
            'evaluation_results': evaluation_results,
            'convergence_diagnostics': convergence_diagnostics,
            'computation_time': end_time - start_time,
            'observed_data': observed_data,
            'observed_loci': observed_loci
        }
        
        logger.info(f"样本 {sample_id} 分析完成，耗时: {end_time - start_time:.1f}秒")
        return result
    
    def _get_default_result(self, sample_id: str, predicted_noc: int, 
                          noc_confidence: float, v5_features: Dict) -> Dict:
        """获取默认结果（当峰数据为空时）"""
        return {
            'sample_id': sample_id,
            'predicted_noc': predicted_noc,
            'noc_confidence': noc_confidence,
            'v5_features': v5_features,
            'mcmc_results': None,
            'posterior_summary': {},
            'evaluation_results': None,
            'convergence_diagnostics': {'status': 'No valid peaks'},
            'computation_time': 0.0,
            'observed_data': {},
            'observed_loci': []
        }
    
    def generate_genotype_posterior_summary(self, mcmc_results: Dict, N: int, 
                                          observed_loci: List[str]) -> Dict:
        """生成基因型后验分布摘要"""
        if not mcmc_results or not mcmc_results['samples']['genotypes']:
            return {}
        
        summary = {}
        genotype_samples = mcmc_results['samples']['genotypes']
        
        # 计算每个位点每个贡献者的基因型后验分布
        for locus in observed_loci:
            locus_summary = {}
            
            # 统计基因型频次
            for contributor_idx in range(N):
                genotype_counts = defaultdict(int)
                
                for sample in genotype_samples:
                    if locus in sample and contributor_idx < len(sample[locus]):
                        genotype = sample[locus][contributor_idx]
                        if genotype is not None:
                            genotype_counts[genotype] += 1
                
                if genotype_counts:
                    # 计算后验概率
                    total_count = sum(genotype_counts.values())
                    genotype_probs = {gt: count/total_count for gt, count in genotype_counts.items()}
                    
                    # 找出最可能的基因型
                    mode_genotype = max(genotype_probs.items(), key=lambda x: x[1])
                    
                    # 计算95%可信集合
                    sorted_genotypes = sorted(genotype_probs.items(), key=lambda x: x[1], reverse=True)
                    cumulative_prob = 0
                    credible_set_95 = []
                    
                    for gt, prob in sorted_genotypes:
                        cumulative_prob += prob
                        credible_set_95.append((gt, prob))
                        if cumulative_prob >= 0.95:
                            break
                    
                    locus_summary[f'contributor_{contributor_idx+1}'] = {
                        'mode_genotype': mode_genotype[0],
                        'mode_probability': mode_genotype[1],
                        'posterior_distribution': dict(genotype_probs),
                        'credible_set_95': credible_set_95,
                        'total_samples': total_count
                    }
            
            summary[locus] = locus_summary
        
        # 添加MCMC质量指标
        summary['mcmc_quality'] = {
            'acceptance_rate_mx': mcmc_results['acceptance_rate_mx'],
            'acceptance_rate_gt': mcmc_results['acceptance_rate_gt'],
            'n_effective_samples': mcmc_results['n_samples'],
            'converged': mcmc_results.get('converged', False)
        }
        
        return summary
    
    def analyze_mcmc_convergence(self, mcmc_results: Dict, N: int) -> Dict:
        """分析MCMC收敛性"""
        if not mcmc_results or not mcmc_results['samples']:
            return {'status': 'No MCMC samples'}
        
        diagnostics = {}
        
        # 混合比例收敛性
        mixture_samples = np.array(mcmc_results['samples']['mixture_ratios'])
        if len(mixture_samples) > 0:
            diagnostics['mixture_ratio_convergence'] = self._analyze_parameter_convergence(mixture_samples)
        
        # 接受率诊断
        mx_acceptance = mcmc_results['acceptance_rate_mx']
        gt_acceptance = mcmc_results['acceptance_rate_gt']
        
        diagnostics['acceptance_rates'] = {
            'mixture_ratios': mx_acceptance,
            'genotypes': gt_acceptance
        }
        
        # 收敛评估
        convergence_issues = []
        if mx_acceptance < 0.15 or mx_acceptance > 0.6:
            convergence_issues.append('Mixture ratio acceptance rate suboptimal')
        if gt_acceptance < 0.05 or gt_acceptance > 0.4:
            convergence_issues.append('Genotype acceptance rate suboptimal')
        
        if len(mixture_samples) < 100:
            convergence_issues.append('Insufficient effective samples')
        
        diagnostics['convergence_status'] = 'Good' if not convergence_issues else 'Poor'
        diagnostics['convergence_issues'] = convergence_issues
        
        return diagnostics
    
    def _analyze_parameter_convergence(self, samples: np.ndarray) -> Dict:
        """分析参数收敛性"""
        n_samples, n_params = samples.shape
        
        convergence_metrics = {}
        
        # 计算有效样本量
        ess_values = []
        for i in range(n_params):
            autocorr = self._calculate_autocorrelation(samples[:, i])
            tau_int = 1 + 2 * np.sum(autocorr[1:min(20, len(autocorr))])
            ess = n_samples / max(tau_int, 1)
            ess_values.append(max(ess, 1))
        
        convergence_metrics['effective_sample_sizes'] = ess_values
        convergence_metrics['min_ess'] = min(ess_values)
        
        # Geweke诊断
        geweke_scores = []
        split_point = max(n_samples // 10, 1)
        
        for i in range(n_params):
            first_part = samples[:split_point, i]
            last_part = samples[-n_samples//2:, i]
            
            if len(first_part) > 1 and len(last_part) > 1:
                mean_diff = np.mean(last_part) - np.mean(first_part)
                pooled_std = np.sqrt((np.var(first_part) + np.var(last_part)) / 2)
                if pooled_std > 1e-10:
                    geweke_score = abs(mean_diff / pooled_std)
                else:
                    geweke_score = 0
            else:
                geweke_score = 0
            
            geweke_scores.append(geweke_score)
        
        convergence_metrics['geweke_scores'] = geweke_scores
        convergence_metrics['max_geweke'] = max(geweke_scores) if geweke_scores else 0
        
        return convergence_metrics
    
    def _calculate_autocorrelation(self, x: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """计算自相关函数"""
        n = len(x)
        if n < 2:
            return np.array([1.0])
        
        x = x - np.mean(x)
        
        # 使用FFT计算自相关
        f = np.fft.fft(x, n=2*n-1)
        autocorr = np.fft.ifft(f * np.conj(f)).real
        autocorr = autocorr[:n] / autocorr[0]
        
        return autocorr[:min(max_lag, n)]
    
    def plot_results(self, results: Dict, output_dir: str) -> None:
        """绘制分析结果图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        sample_id = results['sample_id']
        
        if results['mcmc_results'] is None:
            logger.warning(f"样本 {sample_id} 没有MCMC结果，跳过绘图")
            return
        
        # 1. 混合比例轨迹图
        self._plot_mixture_ratio_traces(results, output_dir)
        
        # 2. 基因型后验分布图
        self._plot_genotype_posteriors(results, output_dir)
        
        # 3. 收敛性诊断图
        self._plot_convergence_diagnostics(results, output_dir)
        
        # 4. 性能评估图（如果有评估结果）
        if results['evaluation_results']:
            self._plot_performance_evaluation(results, output_dir)
        
        logger.info(f"样本 {sample_id} 的图表已保存到: {output_dir}")
    
    def _plot_mixture_ratio_traces(self, results: Dict, output_dir: str):
        """绘制混合比例轨迹图"""
        sample_id = results['sample_id']
        predicted_noc = results['predicted_noc']
        
        mixture_samples = np.array(results['mcmc_results']['samples']['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        fig, axes = plt.subplots(n_components, 1, figsize=(14, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].plot(mixture_samples[:, i], alpha=0.8, linewidth=0.5)
            axes[i].set_title(f'混合比例 Mx_{i+1} 的MCMC轨迹', fontsize=12)
            axes[i].set_xlabel('迭代次数 (thinned)')
            axes[i].set_ylabel(f'Mx_{i+1}')
            axes[i].grid(True, alpha=0.3)
            
            mean_val = np.mean(mixture_samples[:, i])
            axes[i].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                           label=f'均值: {mean_val:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_id}_mixture_traces.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_genotype_posteriors(self, results: Dict, output_dir: str):
        """绘制基因型后验分布图"""
        sample_id = results['sample_id']
        posterior_summary = results['posterior_summary']
        
        if not posterior_summary:
            return
        
        # 为每个位点创建基因型后验分布图
        for locus, locus_summary in posterior_summary.items():
            if locus == 'mcmc_quality':
                continue
            
            fig, axes = plt.subplots(1, len(locus_summary), figsize=(6*len(locus_summary), 5))
            if len(locus_summary) == 1:
                axes = [axes]
            
            for idx, (contributor, contrib_data) in enumerate(locus_summary.items()):
                if 'posterior_distribution' not in contrib_data:
                    continue
                
                posterior_dist = contrib_data['posterior_distribution']
                genotypes = list(posterior_dist.keys())
                probs = list(posterior_dist.values())
                
                bars = axes[idx].bar(range(len(genotypes)), probs, alpha=0.7)
                axes[idx].set_ylabel('后验概率')
                axes[idx].set_xticks(range(len(genotypes)))
                axes[idx].set_xticklabels([f'{gt[0]},{gt[1]}' for gt in genotypes], rotation=45)
                axes[idx].grid(True, alpha=0.3)
                
                # 标注最可能的基因型
                mode_idx = np.argmax(probs)
                axes[idx].text(mode_idx, probs[mode_idx] + 0.01, 
                              f'Mode\n{probs[mode_idx]:.3f}', 
                              ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{sample_id}_{locus}_genotype_posteriors.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_convergence_diagnostics(self, results: Dict, output_dir: str):
        """绘制收敛性诊断图"""
        sample_id = results['sample_id']
        mcmc_results = results['mcmc_results']
        
        # 绘制似然轨迹
        acceptance_details = mcmc_results['acceptance_details']
        iterations = [detail['iteration'] for detail in acceptance_details]
        likelihoods = [detail['log_likelihood'] for detail in acceptance_details]
        
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, likelihoods, alpha=0.8, linewidth=0.5)
        plt.title(f'样本 {sample_id} - 对数似然轨迹')
        plt.xlabel('迭代次数')
        plt.ylabel('对数似然')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{sample_id}_likelihood_trace.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制接受率图
        window_size = 500
        mx_acceptance = []
        gt_acceptance = []
        
        for i in range(window_size, len(acceptance_details), window_size):
            recent_details = acceptance_details[i-window_size:i]
            mx_rate = np.mean([d['accepted_mx'] for d in recent_details])
            gt_rate = np.mean([d['accepted_gt'] for d in recent_details])
            mx_acceptance.append(mx_rate)
            gt_acceptance.append(gt_rate)
        
        if mx_acceptance and gt_acceptance:
            plt.figure(figsize=(12, 6))
            window_centers = range(window_size, len(acceptance_details), window_size)
            plt.plot(window_centers[:len(mx_acceptance)], mx_acceptance, 
                    label='混合比例接受率', linewidth=2)
            plt.plot(window_centers[:len(gt_acceptance)], gt_acceptance, 
                    label='基因型接受率', linewidth=2)
            plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='目标接受率')
            plt.title(f'样本 {sample_id} - MCMC接受率变化')
            plt.xlabel('迭代次数')
            plt.ylabel('接受率')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{sample_id}_acceptance_rates.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_evaluation(self, results: Dict, output_dir: str):
        """绘制性能评估图"""
        sample_id = results['sample_id']
        evaluation_results = results['evaluation_results']
        
        if not evaluation_results or 'error' in evaluation_results:
            return
        
        # 基因型一致性率图
        if 'genotype_concordance' in evaluation_results:
            gcr_data = evaluation_results['genotype_concordance']
            
            if 'locus_specific' in gcr_data:
                loci = list(gcr_data['locus_specific'].keys())
                concordance_rates = [gcr_data['locus_specific'][locus]['concordance_rate'] 
                                   for locus in loci]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(loci, concordance_rates, alpha=0.7, color='skyblue')
                plt.title(f'样本 {sample_id} - 各位点基因型一致性率')
                plt.xlabel('STR位点')
                plt.ylabel('基因型一致性率')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, rate in zip(bars, concordance_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{rate:.3f}', ha='center', va='bottom')
                
                # 添加总体一致性率线
                overall_gcr = gcr_data['overall_gcr']
                plt.axhline(y=overall_gcr, color='red', linestyle='--', 
                           label=f'总体GCR: {overall_gcr:.3f}')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{sample_id}_genotype_concordance.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
        
        # 等位基因一致性率图
        if 'allele_concordance' in evaluation_results:
            acr_data = evaluation_results['allele_concordance']
            
            if 'locus_specific' in acr_data:
                loci = list(acr_data['locus_specific'].keys())
                acr_rates = [acr_data['locus_specific'][locus]['acr'] for locus in loci]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(loci, acr_rates, alpha=0.7, color='lightgreen')
                plt.title(f'样本 {sample_id} - 各位点等位基因一致性率')
                plt.xlabel('STR位点')
                plt.ylabel('等位基因一致性率')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, rate in zip(bars, acr_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{rate:.3f}', ha='center', va='bottom')
                
                # 添加总体一致性率线
                overall_acr = acr_data['overall_acr']
                plt.axhline(y=overall_acr, color='red', linestyle='--', 
                           label=f'总体ACR: {overall_acr:.3f}')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{sample_id}_allele_concordance.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """保存分析结果"""
        # 简化结果以便JSON序列化
        simplified_results = {
            'sample_id': results['sample_id'],
            'predicted_noc': results['predicted_noc'],
            'noc_confidence': results['noc_confidence'],
            'posterior_summary': results['posterior_summary'],
            'evaluation_results': results['evaluation_results'],
            'convergence_diagnostics': results['convergence_diagnostics'],
            'computation_time': results['computation_time'],
            'observed_loci': results['observed_loci'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加部分MCMC样本
        if results['mcmc_results'] is not None:
            mcmc_results = results['mcmc_results']
            if mcmc_results['n_samples'] > 200:
                indices = np.random.choice(mcmc_results['n_samples'], 200, replace=False)
                mixture_samples = np.array(mcmc_results['samples']['mixture_ratios'])
                simplified_results['sample_mixture_ratios'] = mixture_samples[indices].tolist()
                
                # 采样基因型样本
                genotype_samples = mcmc_results['samples']['genotypes']
                simplified_results['sample_genotypes'] = [genotype_samples[i] for i in indices]
            else:
                simplified_results['sample_mixture_ratios'] = mcmc_results['samples']['mixture_ratios']
                simplified_results['sample_genotypes'] = mcmc_results['samples']['genotypes']
            
            simplified_results['mcmc_quality'] = {
                'acceptance_rate_mx': mcmc_results['acceptance_rate_mx'],
                'acceptance_rate_gt': mcmc_results['acceptance_rate_gt'],
                'n_samples': mcmc_results['n_samples'],
                'converged': mcmc_results['converged']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {output_path}")

# =====================
# 9. 主函数和应用接口
# =====================
def analyze_single_sample_q3(sample_id: str, att1_or_att2_path: str, 
                            q1_model_path: str = None, att3_path: str = None) -> Dict:
    """分析单个样本的基因型推断"""
    
    print(f"\n{'='*80}")
    print(f"开始分析样本: {sample_id}")
    print(f"{'='*80}")
    
    # 初始化流水线
    pipeline = Q3EnhancedPipeline(q1_model_path, att3_path)
    
    # 加载数据
    pipeline.load_data(att2_path=config.ATTACHMENT2_PATH)
    
    # 读取样本数据
    try:
        df_samples = pd.read_csv(att1_or_att2_path, encoding='utf-8')
        sample_data = df_samples[df_samples['Sample File'] == sample_id]
        
        if sample_data.empty:
            raise ValueError(f"样本 {sample_id} 不存在于数据文件中")
        
    except Exception as e:
        print(f"样本数据加载失败: {e}")
        return {}
    
    # 分析样本
    result = pipeline.analyze_single_sample(sample_id, sample_data)
    
    # 保存结果
    output_file = os.path.join(config.OUTPUT_DIR, f'{sample_id}_genotype_analysis.json')
    pipeline.save_results(result, output_file)
    
    # 绘制图表
    pipeline.plot_results(result, config.OUTPUT_DIR)
    
    return result

def analyze_all_samples_q3(att1_or_att2_path: str, q1_model_path: str = None, 
                          att3_path: str = None, max_samples: int = None) -> Dict[str, Dict]:
    """批量分析所有样本的基因型推断"""
    
    print(f"\n{'='*80}")
    print("开始批量基因型推断分析")
    print(f"{'='*80}")
    
    # 初始化流水线
    pipeline = Q3EnhancedPipeline(q1_model_path, att3_path)
    
    # 加载数据
    pipeline.load_data(att2_path=config.ATTACHMENT2_PATH)
    
    # 读取样本数据
    try:
        df_samples = pd.read_csv(att1_or_att2_path, encoding='utf-8')
        sample_list = df_samples['Sample File'].unique().tolist()
        
        if max_samples:
            sample_list = sample_list[:max_samples]
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return {}
    
    # 批量分析
    all_results = {}
    
    print(f"计划分析 {len(sample_list)} 个样本")
    
    for idx, sample_id in enumerate(sample_list, 1):
        print(f"\n--- 分析进度: {idx}/{len(sample_list)} - 样本: {sample_id} ---")
        
        try:
            sample_data = df_samples[df_samples['Sample File'] == sample_id]
            result = pipeline.analyze_single_sample(sample_id, sample_data)
            all_results[sample_id] = result
            
            # 保存单个样本结果
            output_file = os.path.join(config.OUTPUT_DIR, f'{sample_id}_result.json')
            pipeline.save_results(result, output_file)
            
            # 绘制图表
            pipeline.plot_results(result, config.OUTPUT_DIR)
            
            # 打印简要结果
            print_sample_summary_q3(result)
            
        except Exception as e:
            print(f"样本 {sample_id} 分析失败: {e}")
            continue
    
    # 生成批量分析摘要
    generate_batch_summary_q3(all_results)
    
    print(f"\n{'='*80}")
    print(f"批量分析完成！成功分析 {len(all_results)} 个样本")
    print(f"结果保存到目录: {config.OUTPUT_DIR}")
    print(f"{'='*80}")
    
    return all_results

def print_sample_summary_q3(result: Dict):
    """打印样本分析摘要"""
    sample_id = result['sample_id']
    predicted_noc = result['predicted_noc']
    noc_confidence = result['noc_confidence']
    computation_time = result['computation_time']
    
    print(f"  预测NoC: {predicted_noc} (置信度: {noc_confidence:.3f})")
    print(f"  计算耗时: {computation_time:.1f}秒")
    
    if result['mcmc_results'] is not None:
        mcmc_results = result['mcmc_results']
        mx_acceptance = mcmc_results['acceptance_rate_mx']
        gt_acceptance = mcmc_results['acceptance_rate_gt']
        converged = mcmc_results['converged']
        
        print(f"  MCMC接受率: 混合比例={mx_acceptance:.3f}, 基因型={gt_acceptance:.3f}")
        print(f"  MCMC收敛: {'是' if converged else '否'}")
        
        # 打印基因型推断摘要
        posterior_summary = result['posterior_summary']
        observed_loci = result['observed_loci']
        
        if posterior_summary and observed_loci:
            print(f"  基因型推断位点数: {len(observed_loci)}")
            
            # 显示第一个位点的推断结果作为示例
            first_locus = observed_loci[0]
            if first_locus in posterior_summary:
                locus_summary = posterior_summary[first_locus]
                print(f"  示例位点 {first_locus}:")
                
                for contributor, contrib_data in locus_summary.items():
                    if 'mode_genotype' in contrib_data:
                        mode_gt = contrib_data['mode_genotype']
                        mode_prob = contrib_data['mode_probability']
                        print(f"    {contributor}: {mode_gt[0]},{mode_gt[1]} (prob={mode_prob:.3f})")
    
    # 打印评估结果（如果有）
    if result['evaluation_results'] and 'overall_summary' in result['evaluation_results']:
        eval_summary = result['evaluation_results']['overall_summary']
        if 'genotype_concordance_rate' in eval_summary:
            gcr = eval_summary['genotype_concordance_rate']
            grade = eval_summary.get('performance_grade', 'N/A')
            print(f"  基因型一致性率: {gcr:.3f} ({grade})")

def generate_batch_summary_q3(all_results: Dict[str, Dict]):
    """生成批量分析摘要报告"""
    if not all_results:
        return
    
    summary_data = []
    
    for sample_id, result in all_results.items():
        sample_summary = {
            'Sample_ID': sample_id,
            'Predicted_NoC': result['predicted_noc'],
            'NoC_Confidence': result['noc_confidence'],
            'Computation_Time': result['computation_time'],
            'MCMC_Success': result['mcmc_results'] is not None,
            'Observed_Loci_Count': len(result['observed_loci'])
        }
        
        if result['mcmc_results'] is not None:
            mcmc_results = result['mcmc_results']
            sample_summary.update({
                'MCMC_Acceptance_Rate_Mx': mcmc_results['acceptance_rate_mx'],
                'MCMC_Acceptance_Rate_Gt': mcmc_results['acceptance_rate_gt'],
                'MCMC_Converged': mcmc_results['converged'],
                'Effective_Samples': mcmc_results['n_samples']
            })
        else:
            sample_summary.update({
                'MCMC_Acceptance_Rate_Mx': None,
                'MCMC_Acceptance_Rate_Gt': None,
                'MCMC_Converged': False,
                'Effective_Samples': 0
            })
        
        # 添加评估结果
        if result['evaluation_results'] and 'overall_summary' in result['evaluation_results']:
            eval_summary = result['evaluation_results']['overall_summary']
            sample_summary.update({
                'Genotype_Concordance_Rate': eval_summary.get('genotype_concordance_rate', None),
                'Allele_Concordance_Rate': eval_summary.get('allele_concordance_rate', None),
                'Performance_Grade': eval_summary.get('performance_grade', None)
            })
        
        summary_data.append(sample_summary)
    
    # 保存摘要表格
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(config.OUTPUT_DIR, 'q3_batch_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    # 生成统计报告
    print(f"\n批量基因型推断分析统计摘要:")
    print(f"  总样本数: {len(all_results)}")
    
    noc_distribution = summary_df['Predicted_NoC'].value_counts().sort_index()
    print(f"  NoC分布: {noc_distribution.to_dict()}")
    
    successful_mcmc = summary_df['MCMC_Success'].sum()
    print(f"  MCMC成功率: {successful_mcmc}/{len(all_results)} ({successful_mcmc/len(all_results)*100:.1f}%)")
    
    if successful_mcmc > 0:
        mcmc_df = summary_df[summary_df['MCMC_Success']]
        avg_acceptance_mx = mcmc_df['MCMC_Acceptance_Rate_Mx'].mean()
        avg_acceptance_gt = mcmc_df['MCMC_Acceptance_Rate_Gt'].mean()
        converged_count = mcmc_df['MCMC_Converged'].sum()
        avg_time = mcmc_df['Computation_Time'].mean()
        avg_loci = mcmc_df['Observed_Loci_Count'].mean()
        
        print(f"  平均MCMC接受率: 混合比例={avg_acceptance_mx:.3f}, 基因型={avg_acceptance_gt:.3f}")
        print(f"  MCMC收敛率: {converged_count}/{successful_mcmc} ({converged_count/successful_mcmc*100:.1f}%)")
        print(f"  平均计算时间: {avg_time:.1f}秒")
        print(f"  平均观测位点数: {avg_loci:.1f}")
    
    # 评估结果统计
    eval_samples = summary_df.dropna(subset=['Genotype_Concordance_Rate'])
    if len(eval_samples) > 0:
        avg_gcr = eval_samples['Genotype_Concordance_Rate'].mean()
        avg_acr = eval_samples['Allele_Concordance_Rate'].mean()
        print(f"  平均基因型一致性率: {avg_gcr:.3f}")
        print(f"  平均等位基因一致性率: {avg_acr:.3f}")
        
        # 性能等级分布
        grade_distribution = eval_samples['Performance_Grade'].value_counts()
        print(f"  性能等级分布: {grade_distribution.to_dict()}")
    
    print(f"  摘要文件已保存: {summary_path}")

def main():
    """主程序"""
    print("=" * 80)
    print("问题三：基于V4强化的MCMC基因型推断系统")
    print("集成Q1特征工程 + Q2 MGM-RF方法 + 基因型匹配评估")
    print("=" * 80)
    
    # 检查必要文件
    required_files = [
        (config.ATTACHMENT1_PATH, "附件1"),
        (config.ATTACHMENT2_PATH, "附件2"), 
        (config.ATTACHMENT3_PATH, "附件3")
    ]
    
    missing_files = []
    for file_path, file_name in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"错误: 缺少必要文件 - {', '.join(missing_files)}")
        print("请确保以下文件在当前目录下:")
        for file_path, file_name in required_files:
            print(f"  {file_name}: {file_path}")
        return
    
    # 设置随机种子
    np.random.seed(config.RANDOM_STATE)
    
    # 选择分析模式
    print("\n请选择分析模式:")
    print("1. 分析单个样本（基于附件1或附件2）")
    print("2. 批量分析附件1中的所有样本")
    print("3. 批量分析附件2中的所有样本") 
    print("4. 批量分析前N个样本")
    
    choice = input("请输入选择 (1/2/3/4): ").strip()
    
    if choice == "1":
        # 单样本分析
        sample_id = input("请输入样本ID: ").strip()
        data_source = input("数据来源 (1=附件1, 2=附件2): ").strip()
        
        if data_source == "1":
            data_path = config.ATTACHMENT1_PATH
        elif data_source == "2":
            data_path = config.ATTACHMENT2_PATH
        else:
            print("无效的数据来源选择")
            return
        
        try:
            result = analyze_single_sample_q3(
                sample_id, 
                data_path,
                config.Q1_MODEL_PATH,
                config.ATTACHMENT3_PATH
            )
            
            print("\n=== 分析结果 ===")
            print_sample_summary_q3(result)
            
        except Exception as e:
            print(f"分析失败: {e}")
    
    elif choice == "2":
        # 批量分析附件1
        try:
            all_results = analyze_all_samples_q3(
                config.ATTACHMENT1_PATH,
                config.Q1_MODEL_PATH,
                config.ATTACHMENT3_PATH
            )
            
        except Exception as e:
            print(f"批量分析失败: {e}")
    
    elif choice == "3":
        # 批量分析附件2
        try:
            all_results = analyze_all_samples_q3(
                config.ATTACHMENT2_PATH,
                config.Q1_MODEL_PATH,
                config.ATTACHMENT3_PATH
            )
            
        except Exception as e:
            print(f"批量分析失败: {e}")
    
    elif choice == "4":
        # 批量分析前N个样本
        try:
            max_samples = int(input("请输入要分析的样本数量: ").strip())
            data_source = input("数据来源 (1=附件1, 2=附件2): ").strip()
            
            if data_source == "1":
                data_path = config.ATTACHMENT1_PATH
            elif data_source == "2":
                data_path = config.ATTACHMENT2_PATH
            else:
                print("无效的数据来源选择")
                return
            
            all_results = analyze_all_samples_q3(
                data_path,
                config.Q1_MODEL_PATH,
                config.ATTACHMENT3_PATH,
                max_samples
            )
            
        except ValueError:
            print("输入的样本数量无效")
        except Exception as e:
            print(f"批量分析失败: {e}")
    
    else:
        print("无效的选择")

if __name__ == "__main__":
    main()