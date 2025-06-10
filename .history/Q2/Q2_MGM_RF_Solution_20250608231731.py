# -*- coding: utf-8 -*-
"""
数学建模 - 法医DNA分析 - 问题二：基于随机森林特征工程的MGM-M混合比例推断

版本: V1.0 - MGM-RF (Mixture Ratio Inference based on Random Forest and MGM-M)
日期: 2025-06-07
描述: 结合Q1的随机森林特征工程和MGM-M方法的混合比例推断
核心创新:
1. 使用Q1的RFECV特征选择和随机森林算法
2. 集成MGM-M的基因型边缘化MCMC推断
3. V5特征驱动的自适应参数估计
4. 针对不同NoC的专门化策略
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import loggamma, gammaln
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from collections import defaultdict
import itertools
from math import comb
import logging
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.utils.class_weight import compute_sample_weight
# from sklearn.metrics import SCORERS
import joblib

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("问题二：基于随机森林特征工程的MGM-M混合比例推断系统")
print("集成Q1的RFECV特征选择 + MGM-M基因型边缘化MCMC方法")
print("=" * 80)

# =====================
# 1. 核心配置与常量
# =====================
class Config:
    """系统配置类"""
    # 文件路径配置
    DATA_DIR = './'
    ATTACHMENT1_PATH = os.path.join(DATA_DIR, '附件1：不同人数的STR图谱数据.csv')
    ATTACHMENT2_PATH = os.path.join(DATA_DIR, '附件2：混合STR图谱数据.csv')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'q2_mgm_rf_results')
    Q1_MODEL_PATH = os.path.join(DATA_DIR, 'noc_optimized_random_forest_model.pkl')
    
    # STR分析参数
    HEIGHT_THRESHOLD = 50
    SATURATION_THRESHOLD = 30000
    CTA_THRESHOLD = 0.5
    PHR_IMBALANCE_THRESHOLD = 0.6
    
    # MCMC参数
    N_ITERATIONS = 15000
    N_WARMUP = 5000
    N_CHAINS = 4
    THINNING = 5
    K_TOP = 800  # 用于N>=4的采样策略
    
    # 模型参数
    RANDOM_STATE = 42
    N_JOBS = -1

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

def extract_noc_from_filename(filename):
        """从文件名提取贡献者人数（NoC）"""
        match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
        if match:
            ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
            return len(ids) if len(ids) > 0 else np.nan
        return np.nan

# =====================
# 2. Q1特征工程模块继承
# =====================
class Q1FeatureEngineering:
    """继承Q1的特征工程功能"""
    
    def __init__(self):
        self.feature_cache = {}
        logger.info("Q1特征工程模块初始化完成")
    
    def extract_noc_from_filename(self, filename: str):
        """从文件名提取贡献者人数（NoC）"""
        match = re.search(r'-(\d+(?:_\d+)*)-[\d;]+-', str(filename))
        if match:
            ids = [id_val for id_val in match.group(1).split('_') if id_val.isdigit()]
            return len(ids) if len(ids) > 0 else np.nan
        return np.nan
    
    def calculate_entropy(self, probabilities):
        """计算香农熵"""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) == 0:
            return 0.0
        return -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    def calculate_ols_slope(self, x, y):
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
    
    def extract_true_mixture_info(self, filename):
        """从文件名提取真实的贡献者信息和混合比例"""
        # 匹配模式：贡献者ID_比例部分
        match = re.search(r'-(\d+(?:_\d+)*)-([^-]+)-', str(filename))
        if not match:
            return None, None
        
        contributor_ids = match.group(1).split('_')
        ratio_part = match.group(2)
        
        # 解析比例部分，如"1;4"表示1:4的比例
        if ';' in ratio_part:
            ratio_values = [float(x) for x in ratio_part.split(';')]
            # 标准化为概率
            total = sum(ratio_values)
            true_ratios = [r/total for r in ratio_values]
        else:
            # 如果没有比例信息，假设等比例
            true_ratios = [1.0/len(contributor_ids)] * len(contributor_ids)
        
        return contributor_ids, true_ratios
    
    def extract_v5_features(self, sample_file, sample_peaks):
        """提取V5特征集（继承并扩展Q1的特征提取）"""
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
            features['allele_count_dist_entropy'] = self.calculate_entropy(counts.values)
        else:
            features['allele_count_dist_entropy'] = 0
        
        # B类：峰高、平衡性及随机效应特征
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
            
            # 峰高分布多峰性
            try:
                log_heights = np.log(all_heights + 1)
                hist, _ = np.histogram(log_heights, bins=min(10, total_peaks))
                from scipy.signal import find_peaks
                peaks_found, _ = find_peaks(hist)
                features['modality_peak_height'] = len(peaks_found)
            except:
                features['modality_peak_height'] = 1
                
            # 饱和峰统计
            saturated_peaks = (sample_peaks['Original_Height'] >= config.SATURATION_THRESHOLD).sum()
            features['num_saturated_peaks'] = saturated_peaks
            features['ratio_saturated_peaks'] = saturated_peaks / total_peaks
        
        # C类：信息论及图谱复杂度特征
        if len(locus_heights) > 0:
            total_height = locus_heights.sum()
            if total_height > 0:
                locus_probs = locus_heights / total_height
                features['inter_locus_balance_entropy'] = self.calculate_entropy(locus_probs.values)
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
                        entropy = self.calculate_entropy(probs)
                        locus_entropies.append(entropy)
            
            features['avg_locus_allele_entropy'] = np.mean(locus_entropies) if locus_entropies else 0
            
            # 样本整体峰高分布熵
            if total_peaks > 0:
                log_heights = np.log(all_heights + 1)
                hist, _ = np.histogram(log_heights, bins=min(15, total_peaks))
                hist_probs = hist / hist.sum()
                hist_probs = hist_probs[hist_probs > 0]
                features['peak_height_entropy'] = self.calculate_entropy(hist_probs)
            else:
                features['peak_height_entropy'] = 0
            
            # 图谱完整性指标
            effective_loci_count = len(locus_groups)
            features['num_loci_with_effective_alleles'] = effective_loci_count
            features['num_loci_no_effective_alleles'] = max(0, 20 - effective_loci_count)
        
        # D类：DNA降解与信息丢失特征
        if total_peaks > 1 and len(np.unique(all_heights)) > 1 and len(np.unique(all_sizes)) > 1:
            features['height_size_correlation'] = np.corrcoef(all_heights, all_sizes)[0, 1]
            features['height_size_slope'] = self.calculate_ols_slope(all_sizes, all_heights)
            features['weighted_height_size_slope'] = self.calculate_ols_slope(all_sizes, all_heights)
            
            # PHR随片段大小变化的斜率
            if len(phr_values) > 1:
                phr_sizes = []
                for marker, marker_group in locus_groups:
                    if len(marker_group) == 2:
                        avg_size = marker_group['Size'].mean()
                        phr_sizes.append(avg_size)
                
                if len(phr_sizes) == len(phr_values) and len(phr_sizes) > 1:
                    features['phr_size_slope'] = self.calculate_ols_slope(phr_sizes, phr_values)
                else:
                    features['phr_size_slope'] = 0
            else:
                features['phr_size_slope'] = 0
        else:
            features['height_size_correlation'] = 0
            features['height_size_slope'] = 0
            features['weighted_height_size_slope'] = 0
            features['phr_size_slope'] = 0
        
        # 位点丢失和降解评分
        features['locus_dropout_score_weighted_by_size'] = features['num_loci_no_effective_alleles'] / 20
        
        # RFU每碱基对衰减指数
        if len(locus_groups) > 1:
            locus_max_heights = []
            locus_avg_sizes = []
            for marker, marker_group in locus_groups:
                max_height = marker_group['Height'].max()
                avg_size = marker_group['Size'].mean()
                locus_max_heights.append(max_height)
                locus_avg_sizes.append(avg_size)
            
            features['degradation_index_rfu_per_bp'] = self.calculate_ols_slope(locus_avg_sizes, locus_max_heights)
        else:
            features['degradation_index_rfu_per_bp'] = 0
        
        # 小大片段信息完整度比率
        small_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() < 200)
        large_fragment_effective = sum(1 for marker, group in locus_groups if group['Size'].mean() >= 200)
        
        if large_fragment_effective > 0:
            features['info_completeness_ratio_small_large'] = small_fragment_effective / large_fragment_effective
        else:
            features['info_completeness_ratio_small_large'] = small_fragment_effective / 0.001
        
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
            'ratio_severe_imbalance_loci': 0, 'modality_peak_height': 1,
            'num_saturated_peaks': 0, 'ratio_saturated_peaks': 0,
            'inter_locus_balance_entropy': 0, 'avg_locus_allele_entropy': 0,
            'peak_height_entropy': 0, 'num_loci_with_effective_alleles': 0,
            'num_loci_no_effective_alleles': 20, 'height_size_correlation': 0,
            'height_size_slope': 0, 'weighted_height_size_slope': 0,
            'phr_size_slope': 0, 'locus_dropout_score_weighted_by_size': 1,
            'degradation_index_rfu_per_bp': 0, 'info_completeness_ratio_small_large': 0
        }
        return default_features

# =====================
# 3. NoC预测模块（基于Q1训练的模型）
# =====================
class NoCPredictor:
    """NoC预测器，使用Q1训练的随机森林模型"""
    
    def __init__(self, model_path: str):
        self.model_data = None
        self.load_model(model_path)
        logger.info("NoC预测器初始化完成")
    
    def load_model(self, model_path: str):
        """加载Q1训练的模型，处理版本兼容性问题"""
        try:
            # 方法1：尝试直接加载
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_data = joblib.load(model_path)
            
            # 验证模型数据完整性
            required_keys = ['model', 'scaler', 'label_encoder', 'selected_features']
            if not all(key in self.model_data for key in required_keys):
                raise ValueError("模型文件缺少必要组件")
                
            logger.info(f"成功加载Q1模型: {model_path}")
            
        except Exception as e:
            logger.warning(f"标准加载失败: {e}")
            
            # 方法2：尝试手动重建评分器
            try:
                self._load_with_custom_scorer(model_path)
                logger.info("使用自定义评分器成功加载模型")
                
            except Exception as e2:
                logger.warning(f"自定义加载也失败: {e2}")
                
                # 方法3：尝试加载模型核心组件
                try:
                    self._load_core_components(model_path)
                    logger.info("成功加载模型核心组件")
                    
                except Exception as e3:
                    logger.error(f"所有加载方法都失败: {e3}")
                    self.model_data = None
                    
                    # 提供解决方案
                    logger.info("解决方案:")
                    logger.info("1. 重新运行Q1c2.py生成兼容的模型文件")
                    logger.info("2. 或者使用基于特征的简单NoC估计")
    
    def _load_with_custom_scorer(self, model_path: str):
        """使用自定义评分器加载模型"""
        import pickle
        
        # 创建自定义的_Scorer类
        class CustomScorer:
            def __init__(self, score_func):
                self._score_func = score_func
            
            def __call__(self, estimator, X, y, sample_weight=None):
                return self._score_func(y, estimator.predict(X))
        
        # 临时替换sklearn的_Scorer
        import sklearn.metrics._scorer as scorer_module
        original_scorer = getattr(scorer_module, '_Scorer', None)
        scorer_module._Scorer = CustomScorer
        
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
        finally:
            # 恢复原始的_Scorer
            if original_scorer is not None:
                scorer_module._Scorer = original_scorer
    
    def _load_core_components(self, model_path: str):
        """只加载模型的核心组件"""
        import pickle
        
        with open(model_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 提取核心组件
        core_components = {}
        
        if 'model' in raw_data:
            # 如果RandomForestClassifier可以直接使用
            if hasattr(raw_data['model'], 'predict'):
                core_components['model'] = raw_data['model']
            else:
                # 创建新的RandomForestClassifier
                from sklearn.ensemble import RandomForestClassifier
                rf_params = getattr(raw_data['model'], 'get_params', lambda: {})()
                core_components['model'] = RandomForestClassifier(**rf_params)
                
                # 复制训练好的树
                if hasattr(raw_data['model'], 'estimators_'):
                    core_components['model'].estimators_ = raw_data['model'].estimators_
                    core_components['model'].n_features_in_ = raw_data['model'].n_features_in_
                    core_components['model'].classes_ = raw_data['model'].classes_
                    core_components['model'].n_classes_ = raw_data['model'].n_classes_
        
        # 其他组件
        for key in ['scaler', 'label_encoder', 'selected_features', 'selected_indices']:
            if key in raw_data:
                core_components[key] = raw_data[key]
        
        self.model_data = core_components
    
    def _simple_noc_estimation(self, v5_features: Dict) -> int:
        """基于特征的增强版简单NoC估计"""
        # 增强的特征规则
        mac_profile = v5_features.get('mac_profile', 0)
        avg_alleles = v5_features.get('avg_alleles_per_locus', 0)
        total_alleles = v5_features.get('total_distinct_alleles', 0)
        loci_gt4 = v5_features.get('loci_gt4_alleles', 0)
        loci_gt5 = v5_features.get('loci_gt5_alleles', 0)
        
        # 规则1：基于MAC（最大等位基因数）
        if mac_profile >= 6:
            return min(5, max(3, int(mac_profile / 1.5)))
        elif mac_profile >= 5:
            return 3
        elif mac_profile >= 4:
            return 2
        elif mac_profile >= 3:
            return 2
        
        # 规则2：基于高等位基因数位点
        if loci_gt5 >= 3:
            return 4
        elif loci_gt5 >= 1 or loci_gt4 >= 5:
            return 3
        elif loci_gt4 >= 2:
            return 2
        
        # 规则3：基于总等位基因数
        if total_alleles >= 60:
            return 4
        elif total_alleles >= 45:
            return 3
        elif total_alleles >= 30:
            return 2
        
        # 默认返回2
        return 2
    
    def predict_noc(self, v5_features: Dict) -> Tuple[int, float]:
        """
        预测贡献者人数
        
        Args:
            v5_features: V5特征字典
            
        Returns:
            (predicted_noc, confidence)
        """
        if self.model_data is None:
            # 使用增强的特征估计
            noc_estimate = self._simple_noc_estimation(v5_features)
            confidence = 0.6  # 提高置信度
            logger.info(f"使用特征规则估计NoC: {noc_estimate} (置信度: {confidence})")
            return noc_estimate, confidence
        
        try:
            model = self.model_data['model']
            scaler = self.model_data.get('scaler')
            label_encoder = self.model_data['label_encoder']
            selected_features = self.model_data.get('selected_features', [])
            
            # 准备特征向量
            if selected_features:
                # 使用选定的特征
                feature_vector = []
                for feature_name in selected_features:
                    feature_vector.append(v5_features.get(feature_name, 0))
            else:
                # 使用所有数值特征
                feature_vector = []
                for key, value in v5_features.items():
                    if isinstance(value, (int, float)) and key != 'Sample File':
                        feature_vector.append(value)
            
            if not feature_vector:
                # 如果没有有效特征，回退到简单估计
                noc_estimate = self._simple_noc_estimation(v5_features)
                return noc_estimate, 0.5
            
            # 转换为numpy数组
            X = np.array(feature_vector).reshape(1, -1)
            
            # 特征标准化（如果有scaler）
            if scaler is not None:
                try:
                    X_scaled = scaler.transform(X)
                except:
                    X_scaled = X  # 如果标准化失败，使用原始特征
            else:
                X_scaled = X
            
            # 预测
            y_pred_encoded = model.predict(X_scaled)[0]
            y_pred = label_encoder.inverse_transform([y_pred_encoded])[0]
            
            # 预测概率作为置信度
            try:
                y_proba = model.predict_proba(X_scaled)[0]
                confidence = np.max(y_proba)
            except:
                confidence = 0.7  # 如果概率预测失败，使用固定置信度
            
            logger.info(f"模型预测NoC: {y_pred} (置信度: {confidence:.3f})")
            return int(y_pred), float(confidence)
            
        except Exception as e:
            logger.warning(f"NoC预测失败: {e}，使用特征规则估计")
            noc_estimate = self._simple_noc_estimation(v5_features)
            return noc_estimate, 0.5

# =====================
# 4. MGM-M核心组件
# =====================
class PseudoFrequencyCalculator:
    """伪等位基因频率计算器"""
    
    def __init__(self):
        self.frequency_cache = {}
        logger.info("伪频率计算器初始化完成")
    
    def calculate_pseudo_frequencies(self, locus: str, 
                                   att2_alleles: List[str]) -> Dict[str, float]:
        """计算位点的伪等位基因频率 w_l(a_k)"""
        cache_key = (locus, tuple(sorted(att2_alleles)))
        if cache_key in self.frequency_cache:
            return self.frequency_cache[cache_key]
        
        # 统计等位基因出现次数
        allele_counts = {}
        for allele in att2_alleles:
            allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        # 计算频率
        total_count = sum(allele_counts.values())
        frequencies = {}
        
        for allele, count in allele_counts.items():
            frequencies[allele] = count / total_count
        
        # 为未观测到但可能存在的等位基因分配最小频率
        min_freq = 1 / (2 * len(att2_alleles) + len(allele_counts))
        
        # 标准化频率
        freq_sum = sum(frequencies.values())
        if freq_sum > 0:
            for allele in frequencies:
                frequencies[allele] = frequencies[allele] / freq_sum * (1 - min_freq * 2)
        
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

class V5FeatureIntegrator:
    """V5特征集成器"""
    
    def __init__(self, v5_features: Dict):
        self.v5_features = v5_features
        logger.info("V5特征集成器初始化完成")
    
    def calculate_gamma_l(self, locus: str) -> float:
        """基于V5特征计算位点特异性放大效率"""
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
    
    def calculate_sigma_var_l(self, locus: str) -> float:
        """基于V5特征计算位点方差参数"""
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
    
    def calculate_degradation_factor(self, allele_size: float) -> float:
        """计算降解因子"""
        k_deg_0 = 0.001
        size_ref = 200.0
        alpha = 1.0
        
        height_size_corr = self.v5_features.get('height_size_correlation', 0.0)
        k_deg = k_deg_0 * max(0, -height_size_corr) ** alpha
        
        D_F = np.exp(-k_deg * max(0, allele_size - size_ref))
        
        return max(D_F, 1e-6)

class GenotypeEnumerator:
    """基因型枚举器"""
    
    def __init__(self, max_contributors: int = 5):
        self.max_contributors = max_contributors
        self.memo = {}
        logger.info(f"基因型枚举器初始化，最大贡献者数：{max_contributors}")
    
    def enumerate_valid_genotype_sets(self, observed_alleles: List[str], 
                                    N: int, K_top: int = None) -> List[List[Tuple[str, str]]]:
        """枚举所有与观测等位基因兼容的基因型组合"""
        cache_key = (tuple(sorted(observed_alleles)), N, K_top)
        if cache_key in self.memo:
            return self.memo[cache_key]
        
        if N <= 3:
            valid_sets = self._enumerate_all_combinations(observed_alleles, N)
        else:
            valid_sets = self._enumerate_k_top_combinations(observed_alleles, N, K_top)
        
        self.memo[cache_key] = valid_sets
        logger.debug(f"为{N}个贡献者枚举了{len(valid_sets)}个有效基因型组合")
        return valid_sets
    
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
            K_top = min(1000, max(100, len(observed_alleles) ** N))
        
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
# 5. MGM-RF主推断器
# =====================
class MGM_RF_Inferencer:
    """MGM-RF混合比例推断器，集成Q1的随机森林和MGM-M方法"""
    
    def __init__(self, q1_model_path: str = None, sample_id: str = None):
        # 初始化组件
        self.pseudo_freq_calculator = PseudoFrequencyCalculator()
        self.genotype_enumerator = GenotypeEnumerator(max_contributors=extract_noc_from_filename(sample_id) if sample_id else 5)
        self.v5_integrator = None
        self.noc_predictor = NoCPredictor(q1_model_path) if q1_model_path else None
        self.q1_feature_engineering = Q1FeatureEngineering()
        
        # MCMC参数
        self.n_iterations = config.N_ITERATIONS
        self.n_warmup = config.N_WARMUP
        self.n_chains = config.N_CHAINS
        self.thinning = config.THINNING
        self.K_top = config.K_TOP
        
        logger.info("MGM-RF推断器初始化完成")
    
    def set_v5_features(self, v5_features: Dict):
        """设置V5特征"""
        self.v5_integrator = V5FeatureIntegrator(v5_features)
        logger.info("V5特征已设置")
    
    def predict_noc_from_sample(self, sample_data: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        从样本数据预测NoC
        
        Args:
            sample_data: 单个样本的STR数据
            
        Returns:
            (predicted_noc, confidence, v5_features)
        """
        # 处理峰数据
        sample_peaks = self.q1_feature_engineering.process_peaks_simplified(sample_data)
        
        # 提取V5特征
        sample_file = sample_data.iloc[0]['Sample File']
        v5_features = self.q1_feature_engineering.extract_v5_features(sample_file, sample_peaks)
        
        # 预测NoC
        if self.noc_predictor:
            predicted_noc, confidence = self.noc_predictor.predict_noc(v5_features)
        else:
            predicted_noc, confidence = 2, 0.5
        
        logger.info(f"样本 {sample_file} 预测NoC: {predicted_noc} (置信度: {confidence:.3f})")
        
        return predicted_noc, confidence, v5_features
    
    def calculate_locus_marginalized_likelihood(self, locus_data: Dict, N: int,
                                              mixture_ratios: np.ndarray,
                                              att2_data: Dict = None) -> float:
        """计算单个位点的边缘化似然函数"""
        if self.v5_integrator is None:
            raise ValueError("请先设置V5特征")
        
        locus = locus_data['locus']
        observed_alleles = locus_data['alleles']
        observed_heights = locus_data['heights']
        
        # 枚举有效基因型组合
        K_top = self.K_top if N >= 4 else None
        valid_genotype_sets = self.genotype_enumerator.enumerate_valid_genotype_sets(
            observed_alleles, N, K_top)
        
        if not valid_genotype_sets:
            logger.warning(f"位点{locus}没有有效的基因型组合")
            return -1e10
        
        # 计算伪等位基因频率
        if att2_data and locus in att2_data:
            att2_alleles = att2_data[locus]
        else:
            att2_alleles = observed_alleles
        
        pseudo_frequencies = self.pseudo_freq_calculator.calculate_pseudo_frequencies(
            locus, att2_alleles)
        
        # 计算位点特异性参数
        gamma_l = self.v5_integrator.calculate_gamma_l(locus)
        sigma_var_l = self.v5_integrator.calculate_sigma_var_l(locus)
        
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
                    D_F = self.v5_integrator.calculate_degradation_factor(allele_size)
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
                        D_F = self.v5_integrator.calculate_degradation_factor(parent_size)
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
                                              att2_data: Dict = None) -> float:
        """计算总的边缘化似然函数"""
        total_log_likelihood = 0.0
        
        for locus, locus_data in observed_data.items():
            locus_likelihood = self.calculate_locus_marginalized_likelihood(
                locus_data, N, mixture_ratios, att2_data)
            total_log_likelihood += locus_likelihood
        
        return total_log_likelihood
    
    def calculate_prior_mixture_ratios(self, mixture_ratios: np.ndarray) -> float:
        """计算混合比例的先验概率"""
        alpha = np.ones(len(mixture_ratios))
        
        if self.v5_integrator:
            skewness = self.v5_integrator.v5_features.get('skewness_peak_height', 0.0)
            if abs(skewness) > 1.0:
                alpha = np.ones(len(mixture_ratios)) * 0.5
        
        log_prior = (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                    np.sum((alpha - 1) * np.log(mixture_ratios + 1e-10)))
        
        return log_prior
    
    def propose_mixture_ratios(self, current_ratios: np.ndarray, 
                             step_size: float = 0.05) -> np.ndarray:
        """使用Dirichlet分布提议新的混合比例"""
        concentration = current_ratios / step_size
        concentration = np.maximum(concentration, 0.1)
        
        new_ratios = np.random.dirichlet(concentration)
        new_ratios = np.maximum(new_ratios, 1e-6)
        new_ratios = new_ratios / np.sum(new_ratios)
        
        return new_ratios
    
    def mcmc_sampler(self, observed_data: Dict, N: int, 
                    att2_data: Dict = None) -> Dict:
        """MGM-RF方法的MCMC采样器"""
        if self.v5_integrator is None:
            raise ValueError("请先设置V5特征")
        
        logger.info(f"开始MGM-RF MCMC采样，贡献者数量: {N}")
        logger.info(f"总迭代次数: {self.n_iterations}, 预热次数: {self.n_warmup}")
        
        # 初始化混合比例
        mixture_ratios = np.random.dirichlet(np.ones(N))
        
        # 存储MCMC样本
        samples = {
            'mixture_ratios': [],
            'log_likelihood': [],
            'log_posterior': [],
            'acceptance_info': []
        }
        
        # 计算初始似然
        current_log_likelihood = self.calculate_total_marginalized_likelihood(
            observed_data, N, mixture_ratios, att2_data)
        current_log_prior = self.calculate_prior_mixture_ratios(mixture_ratios)
        current_log_posterior = current_log_likelihood + current_log_prior
        
        # MCMC主循环
        n_accepted = 0
        acceptance_details = []
        
        # 自适应步长
        step_size = 0.05
        adaptation_interval = 500
        target_acceptance = 0.4
        
        for iteration in range(self.n_iterations):
            if iteration % 1000 == 0:
                acceptance_rate = n_accepted / max(iteration, 1)
                logger.info(f"迭代 {iteration}/{self.n_iterations}, "
                          f"当前接受率: {acceptance_rate:.3f}, "
                          f"当前似然: {current_log_likelihood:.2f}")
            
            # 提议新的混合比例
            proposed_ratios = self.propose_mixture_ratios(mixture_ratios, step_size)
            
            # 计算提议状态的概率
            proposed_log_likelihood = self.calculate_total_marginalized_likelihood(
                observed_data, N, proposed_ratios, att2_data)
            proposed_log_prior = self.calculate_prior_mixture_ratios(proposed_ratios)
            proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
            
            # Metropolis-Hastings接受/拒绝
            log_ratio = proposed_log_posterior - current_log_posterior
            accept_prob = min(1.0, np.exp(log_ratio))
            
            if np.random.random() < accept_prob:
                mixture_ratios = proposed_ratios
                current_log_likelihood = proposed_log_likelihood
                current_log_prior = proposed_log_prior
                current_log_posterior = proposed_log_posterior
                n_accepted += 1
                accepted = True
            else:
                accepted = False
            
            # 记录接受信息
            acceptance_details.append({
                'iteration': iteration,
                'accepted': accepted,
                'log_ratio': log_ratio,
                'accept_prob': accept_prob
            })
            
            # 自适应步长调整
            if iteration > 0 and iteration % adaptation_interval == 0 and iteration < self.n_warmup:
                recent_acceptance = np.mean([a['accepted'] for a in acceptance_details[-adaptation_interval:]])
                if recent_acceptance < target_acceptance - 0.05:
                    step_size *= 0.9
                elif recent_acceptance > target_acceptance + 0.05:
                    step_size *= 1.1
                step_size = np.clip(step_size, 0.01, 0.2)
                
                if iteration % (adaptation_interval * 4) == 0:
                    logger.info(f"  步长调整为: {step_size:.4f}, 最近接受率: {recent_acceptance:.3f}")
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
        
        final_acceptance_rate = n_accepted / self.n_iterations
        logger.info(f"MCMC完成，总接受率: {final_acceptance_rate:.3f}")
        logger.info(f"有效样本数: {len(samples['mixture_ratios'])}")
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'n_samples': len(samples['mixture_ratios']),
            'acceptance_details': acceptance_details,
            'final_step_size': step_size,
            'converged': 0.15 <= final_acceptance_rate <= 0.6
        }

# =====================
# 6. 数据处理与分析流水线
# =====================
class MGM_RF_Pipeline:
    """MGM-RF完整分析流水线"""
    
    def __init__(self, q1_model_path: str = None, sample_id: str= None):
        self.mgm_rf_inferencer = MGM_RF_Inferencer(q1_model_path, sample_id)
        self.results = {}
        logger.info("MGM-RF流水线初始化完成")
    
    def load_attachment2_data(self, att2_path: str) -> Dict[str, pd.DataFrame]:
        """加载附件2数据"""
        try:
            df_att2 = pd.read_csv(att2_path, encoding='utf-8')
            logger.info(f"成功加载附件2数据，形状: {df_att2.shape}")
            
            # 按样本分组
            sample_groups = {}
            for sample_file, group in df_att2.groupby('Sample File'):
                sample_groups[sample_file] = group
            
            return sample_groups
        except Exception as e:
            logger.error(f"附件2数据加载失败: {e}")
            return {}
    
    def prepare_att2_frequency_data(self, att2_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """准备附件2的等位基因频率数据"""
        frequency_data = {}
        
        for sample_file, sample_df in att2_data.items():
            sample_peaks = self.mgm_rf_inferencer.q1_feature_engineering.process_peaks_simplified(sample_df)
            
            if not sample_peaks.empty:
                for locus, locus_group in sample_peaks.groupby('Marker'):
                    if locus not in frequency_data:
                        frequency_data[locus] = []
                    
                    # 收集该位点的所有等位基因
                    alleles = locus_group['Allele'].tolist()
                    frequency_data[locus].extend(alleles)
        
        logger.info(f"附件2频率数据覆盖{len(frequency_data)}个位点")
        return frequency_data
    
    def analyze_sample(self, sample_data: pd.DataFrame, att2_freq_data: Dict[str, List[str]] = None) -> Dict:
        """分析单个样本的混合比例"""
        sample_file = sample_data.iloc[0]['Sample File']
        logger.info(f"开始分析样本: {sample_file}")
        
        # 步骤1: 预测NoC
        predicted_noc, noc_confidence, v5_features = self.mgm_rf_inferencer.predict_noc_from_sample(sample_data)
        
        # 步骤1.5: 提取真实混合信息（如果可用）
        true_contributor_ids, true_ratios = self.mgm_rf_inferencer.q1_feature_engineering.extract_true_mixture_info(sample_file)
        if true_contributor_ids and true_ratios:
            logger.info(f"样本 {sample_file} 真实贡献者: {true_contributor_ids}, 真实比例: {true_ratios}")
        
        # 步骤2: 设置V5特征
        self.mgm_rf_inferencer.set_v5_features(v5_features)
        
        # 步骤3: 处理STR数据
        sample_peaks = self.mgm_rf_inferencer.q1_feature_engineering.process_peaks_simplified(sample_data)
        
        if sample_peaks.empty:
            logger.warning(f"样本 {sample_file} 没有有效的峰数据")
            return self._get_default_result(sample_file, predicted_noc, noc_confidence, v5_features)
        
        # 步骤4: 准备观测数据
        observed_data = {}
        for locus, locus_group in sample_peaks.groupby('Marker'):
            alleles = locus_group['Allele'].tolist()
            heights = dict(zip(locus_group['Allele'], locus_group['Height']))
            
            observed_data[locus] = {
                'locus': locus,
                'alleles': alleles,
                'heights': heights
            }
        
        logger.info(f"样本 {sample_file} 包含{len(observed_data)}个有效位点")
        
        # 步骤5: MCMC推断混合比例
        start_time = time.time()
        mcmc_results = self.mgm_rf_inferencer.mcmc_sampler(
            observed_data, predicted_noc, att2_freq_data)
        end_time = time.time()
        
        # 步骤6: 生成后验摘要
        posterior_summary = self.generate_posterior_summary(mcmc_results, predicted_noc)
        
        # 步骤7: 收敛性诊断
        convergence_diagnostics = self.analyze_convergence(mcmc_results['samples'], predicted_noc)
        
        result = {
            'sample_file': sample_file,
            'predicted_noc': predicted_noc,
            'noc_confidence': noc_confidence,
            'v5_features': v5_features,
            'mcmc_results': mcmc_results,
            'posterior_summary': posterior_summary,
            'convergence_diagnostics': convergence_diagnostics,
            'computation_time': end_time - start_time,
            'observed_data': observed_data,
            'true_contributor_ids': true_contributor_ids,  
            'true_mixture_ratios': true_ratios,          
        }
        
        logger.info(f"样本 {sample_file} 分析完成，耗时: {end_time - start_time:.1f}秒")
        return result
    
    def _get_default_result(self, sample_file: str, predicted_noc: int, noc_confidence: float, v5_features: Dict) -> Dict:
        """获取默认结果（当峰数据为空时）"""
        # 创建默认的混合比例
        if predicted_noc == 1:
            default_ratios = [1.0]
        else:
            default_ratios = [1.0/predicted_noc] * predicted_noc
        
        default_summary = {}
        for i in range(predicted_noc):
            default_summary[f'Mx_{i+1}'] = {
                'mean': default_ratios[i],
                'std': 0.0,
                'median': default_ratios[i],
                'mode': default_ratios[i],
                'credible_interval_95': [default_ratios[i], default_ratios[i]],
                'credible_interval_90': [default_ratios[i], default_ratios[i]],
                'hpdi_95': [default_ratios[i], default_ratios[i]]
            }
        
        return {
            'sample_file': sample_file,
            'predicted_noc': predicted_noc,
            'noc_confidence': noc_confidence,
            'v5_features': v5_features,
            'mcmc_results': None,
            'posterior_summary': default_summary,
            'convergence_diagnostics': {'status': 'No valid peaks'},
            'computation_time': 0.0,
            'observed_data': {}
        }
    
    def generate_posterior_summary(self, results: Dict, N: int) -> Dict:
        """生成后验分布摘要统计"""
        samples = results['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        
        summary = {}
        
        # 混合比例统计
        for i in range(N):
            component_samples = mixture_samples[:, i]
            
            # 基本统计量
            mean_val = np.mean(component_samples)
            std_val = np.std(component_samples)
            median_val = np.median(component_samples)
            
            # 置信区间
            ci_95 = np.percentile(component_samples, [2.5, 97.5])
            ci_90 = np.percentile(component_samples, [5, 95])
            
            # 最高后验密度区间
            sorted_samples = np.sort(component_samples)
            n_samples = len(sorted_samples)
            hpdi_95_width = int(0.95 * n_samples)
            
            min_width = np.inf
            hpdi_95 = [sorted_samples[0], sorted_samples[-1]]
            
            for start in range(n_samples - hpdi_95_width):
                end = start + hpdi_95_width
                width = sorted_samples[end] - sorted_samples[start]
                if width < min_width:
                    min_width = width
                    hpdi_95 = [sorted_samples[start], sorted_samples[end]]
            
            summary[f'Mx_{i+1}'] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'mode': self._estimate_mode(component_samples),
                'credible_interval_95': ci_95.tolist(),
                'credible_interval_90': ci_90.tolist(),
                'hpdi_95': hpdi_95,
                'min': np.min(component_samples),
                'max': np.max(component_samples)
            }
        
        # 模型质量指标
        summary['model_quality'] = {
            'acceptance_rate': results['acceptance_rate'],
            'n_effective_samples': results['n_samples'],
            'converged': results.get('converged', False),
            'final_step_size': results.get('final_step_size', 'N/A')
        }
        
        # 计算与真值的比较（如果有真值）
        if 'true_mixture_ratios' in results and results.get('true_mixture_ratios'):
            true_ratios = results['true_mixture_ratios']
            predicted_means = [summary[f'Mx_{i+1}']['mean'] for i in range(N)]
            
            # 计算误差指标
            if len(true_ratios) == len(predicted_means):
                mae = np.mean(np.abs(np.array(predicted_means) - np.array(true_ratios)))
                rmse = np.sqrt(np.mean((np.array(predicted_means) - np.array(true_ratios))**2))
                
                summary['accuracy_metrics'] = {
                    'mae': mae,
                    'rmse': rmse,
                    'true_ratios': true_ratios,
                    'predicted_ratios': predicted_means
                }
        # 贡献者排序（按后验均值）
        mx_means = [(i+1, summary[f'Mx_{i+1}']['mean']) for i in range(N)]
        mx_means.sort(key=lambda x: x[1], reverse=True)
        summary['contributor_ranking'] = mx_means
        
        return summary
    
    def _estimate_mode(self, samples: np.ndarray) -> float:
        """估计样本的众数"""
        try:
            from scipy.stats import gaussian_kde
            
            if len(samples) < 10:
                return np.median(samples)
            
            kde = gaussian_kde(samples)
            x_range = np.linspace(np.min(samples), np.max(samples), 200)
            density = kde(x_range)
            mode_idx = np.argmax(density)
            return x_range[mode_idx]
            
        except:
            hist, bin_edges = np.histogram(samples, bins=20)
            max_bin = np.argmax(hist)
            return (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    
    def analyze_convergence(self, samples: Dict, N: int) -> Dict:
        """分析MCMC收敛性"""
        diagnostics = {}
        
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 有效样本量 (ESS)
        ess_values = []
        for i in range(n_components):
            autocorr = self._calculate_autocorrelation(mixture_samples[:, i])
            tau_int = 1 + 2 * np.sum(autocorr[1:min(20, len(autocorr))])
            ess = n_samples / max(tau_int, 1)
            ess_values.append(max(ess, 1))
        
        diagnostics['effective_sample_size'] = ess_values
        diagnostics['min_ess'] = min(ess_values)
        
        # 2. Geweke诊断
        geweke_scores = []
        split_point = max(n_samples // 10, 1)
        
        for i in range(n_components):
            first_part = mixture_samples[:split_point, i]
            last_part = mixture_samples[-n_samples//2:, i]
            
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
        
        diagnostics['geweke_scores'] = geweke_scores
        diagnostics['max_geweke'] = max(geweke_scores) if geweke_scores else 0
        
        # 3. 收敛评估
        convergence_issues = []
        if diagnostics['min_ess'] < 50:
            convergence_issues.append('Low ESS')
        if diagnostics['max_geweke'] > 2:
            convergence_issues.append('Geweke test failed')
        
        diagnostics['convergence_status'] = 'Good' if not convergence_issues else 'Poor'
        diagnostics['convergence_issues'] = convergence_issues
        
        return diagnostics
    
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
        
        sample_file = results['sample_file']
        predicted_noc = results['predicted_noc']
        
        if results['mcmc_results'] is None:
            logger.warning(f"样本 {sample_file} 没有MCMC结果，跳过绘图")
            return
        
        samples = results['mcmc_results']['samples']
        mixture_samples = np.array(samples['mixture_ratios'])
        n_samples, n_components = mixture_samples.shape
        
        # 1. 轨迹图
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
        plt.savefig(os.path.join(output_dir, f'{sample_file}_mcmc_trace.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 后验分布密度图
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            axes[i].hist(mixture_samples[:, i], bins=50, alpha=0.7, density=True, 
                        color=f'C{i}', edgecolor='black', linewidth=0.5)
            
            mean_val = np.mean(mixture_samples[:, i])
            median_val = np.median(mixture_samples[:, i])
            mode_val = self._estimate_mode(mixture_samples[:, i])
            
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'均值: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2,
                           label=f'中位数: {median_val:.3f}')
            axes[i].axvline(mode_val, color='orange', linestyle='--', linewidth=2,
                           label=f'众数: {mode_val:.3f}')
            
            ci_95 = np.percentile(mixture_samples[:, i], [2.5, 97.5])
            axes[i].axvspan(ci_95[0], ci_95[1], alpha=0.2, color='gray',
                           label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
            
            axes[i].set_title(f'混合比例 Mx_{i+1} 的后验分布', fontsize=12)
            axes[i].set_xlabel(f'Mx_{i+1}')
            axes[i].set_ylabel('概率密度')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_file}_posterior_dist.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混合比例联合分布（对于2个组分）
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(mixture_samples[:, 0], mixture_samples[:, 1], 
                       alpha=0.6, s=2, color='blue')
            plt.xlabel('Mx_1', fontsize=12)
            plt.ylabel('Mx_2', fontsize=12)
            plt.title(f'样本 {sample_file} 混合比例联合后验分布', fontsize=14)
            
            # 添加约束线
            x_line = np.linspace(0, 1, 100)
            y_line = 1 - x_line
            plt.plot(x_line, y_line, 'r--', alpha=0.7, label='Mx_1 + Mx_2 = 1')
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.axis('equal')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, f'{sample_file}_joint_posterior.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"样本 {sample_file} 的图表已保存到: {output_dir}")
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """保存分析结果"""
        # 简化结果以便JSON序列化
        simplified_results = {
            'sample_file': results['sample_file'],
            'predicted_noc': results['predicted_noc'],
            'noc_confidence': results['noc_confidence'],
            'posterior_summary': results['posterior_summary'],
            'convergence_diagnostics': results['convergence_diagnostics'],
            'computation_time': results['computation_time'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加部分MCMC样本
        if results['mcmc_results'] is not None:
            mcmc_results = results['mcmc_results']
            if mcmc_results['n_samples'] > 500:
                indices = np.random.choice(mcmc_results['n_samples'], 500, replace=False)
                mixture_samples = np.array(mcmc_results['samples']['mixture_ratios'])
                simplified_results['sample_mixture_ratios'] = mixture_samples[indices].tolist()
            else:
                simplified_results['sample_mixture_ratios'] = mcmc_results['samples']['mixture_ratios']
            
            simplified_results['mcmc_quality'] = {
                'acceptance_rate': mcmc_results['acceptance_rate'],
                'n_samples': mcmc_results['n_samples'],
                'converged': mcmc_results['converged']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {output_path}")

# =====================
# 7. 主函数和应用接口
# =====================
def analyze_single_sample_from_att2(sample_id: str, att2_path: str, q1_model_path: str = None) -> Dict:
    """分析附件2中的单个样本"""
    
    # 初始化流水线
    pipeline = MGM_RF_Pipeline(q1_model_path, sample_id)
    
    # 加载附件2数据
    att2_data = pipeline.load_attachment2_data(att2_path)
    
    if sample_id not in att2_data:
        raise ValueError(f"样本 {sample_id} 不存在于附件2数据中")
    
    # 准备频率数据
    att2_freq_data = pipeline.prepare_att2_frequency_data(att2_data)
    
    # 分析目标样本
    sample_data = att2_data[sample_id]
    result = pipeline.analyze_sample(sample_data, att2_freq_data)
    
    # 保存结果
    output_file = os.path.join(config.OUTPUT_DIR, f'{sample_id}_analysis_result.json')
    pipeline.save_results(result, output_file)
    
    # 绘制图表
    pipeline.plot_results(result, config.OUTPUT_DIR)
    
    return result

def analyze_all_samples_from_att2(att2_path: str, q1_model_path: str = None, max_samples: int = None) -> Dict[str, Dict]:
    """分析附件2中的所有样本"""
    
    print(f"\n{'='*80}")
    print("开始批量分析附件2中的混合STR样本")
    print(f"{'='*80}")
    
    # 初始化流水线
    pipeline = MGM_RF_Pipeline(q1_model_path)
    
    # 加载附件2数据
    att2_data = pipeline.load_attachment2_data(att2_path)
    
    if not att2_data:
        print("没有有效的附件2数据")
        return {}
    
    # 准备频率数据
    att2_freq_data = pipeline.prepare_att2_frequency_data(att2_data)
    
    # 批量分析
    all_results = {}
    sample_list = list(att2_data.keys())
    
    if max_samples:
        sample_list = sample_list[:max_samples]
    
    print(f"计划分析 {len(sample_list)} 个样本")
    
    for idx, sample_id in enumerate(sample_list, 1):
        print(f"\n--- 分析进度: {idx}/{len(sample_list)} - 样本: {sample_id} ---")
        
        try:
            sample_data = att2_data[sample_id]
            result = pipeline.analyze_sample(sample_data, att2_freq_data)
            all_results[sample_id] = result
            
            # 保存单个样本结果
            output_file = os.path.join(config.OUTPUT_DIR, f'{sample_id}_result.json')
            pipeline.save_results(result, output_file)
            
            # 绘制图表
            pipeline.plot_results(result, config.OUTPUT_DIR)
            
            # 打印简要结果
            print_sample_summary(result)
            
        except Exception as e:
            print(f"样本 {sample_id} 分析失败: {e}")
            continue
    
    # 生成批量分析摘要
    generate_batch_summary(all_results)
    
    print(f"\n{'='*80}")
    print(f"批量分析完成！成功分析 {len(all_results)} 个样本")
    print(f"结果保存到目录: {config.OUTPUT_DIR}")
    print(f"{'='*80}")
    
    return all_results

def print_sample_summary(result: Dict):
    """打印样本分析摘要"""
    sample_file = result['sample_file']
    predicted_noc = result['predicted_noc']
    noc_confidence = result['noc_confidence']
    computation_time = result['computation_time']
    
    print(f"  预测NoC: {predicted_noc} (置信度: {noc_confidence:.3f})")
    print(f"  计算耗时: {computation_time:.1f}秒")
    
    if result['mcmc_results'] is not None:
        acceptance_rate = result['mcmc_results']['acceptance_rate']
        converged = result['mcmc_results']['converged']
        print(f"  MCMC接受率: {acceptance_rate:.3f}, 收敛: {'是' if converged else '否'}")
        
        # 打印混合比例
        posterior_summary = result['posterior_summary']
        contributor_ranking = posterior_summary['contributor_ranking']
        
        print(f"  混合比例估计:")
        for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking[:3], 1):  # 只显示前3个
            mx_stats = posterior_summary[f'Mx_{contributor_id}']
            ci_95 = mx_stats['credible_interval_95']
            print(f"    第{rank}主要贡献者: {mean_ratio:.3f} (95%CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}])")
    else:
        print(f"  ⚠️  无有效峰数据，使用默认比例")
    if result.get('true_mixture_ratios'):
        print(f"  真实混合比例: {result['true_mixture_ratios']}")
        if 'accuracy_metrics' in result.get('posterior_summary', {}):
            metrics = result['posterior_summary']['accuracy_metrics']
            print(f"  预测精度 - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

def generate_batch_summary(all_results: Dict[str, Dict]):
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
            'MCMC_Success': result['mcmc_results'] is not None
        }
        
        if result['mcmc_results'] is not None:
            sample_summary['MCMC_Acceptance_Rate'] = result['mcmc_results']['acceptance_rate']
            sample_summary['MCMC_Converged'] = result['mcmc_results']['converged']
            sample_summary['Effective_Samples'] = result['mcmc_results']['n_samples']
            
            # 提取主要贡献者比例
            posterior_summary = result['posterior_summary']
            contributor_ranking = posterior_summary['contributor_ranking']
            
            for i, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
                sample_summary[f'Mx_{i}_Mean'] = mean_ratio
                mx_stats = posterior_summary[f'Mx_{contributor_id}']
                sample_summary[f'Mx_{i}_CI_Lower'] = mx_stats['credible_interval_95'][0]
                sample_summary[f'Mx_{i}_CI_Upper'] = mx_stats['credible_interval_95'][1]
        else:
            sample_summary.update({
                'MCMC_Acceptance_Rate': None,
                'MCMC_Converged': False,
                'Effective_Samples': 0
            })
        
        summary_data.append(sample_summary)
    
    # 保存摘要表格
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(config.OUTPUT_DIR, 'batch_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    # 生成统计报告
    print(f"\n批量分析统计摘要:")
    print(f"  总样本数: {len(all_results)}")
    
    noc_distribution = summary_df['Predicted_NoC'].value_counts().sort_index()
    print(f"  NoC分布: {noc_distribution.to_dict()}")
    
    successful_mcmc = summary_df['MCMC_Success'].sum()
    print(f"  MCMC成功率: {successful_mcmc}/{len(all_results)} ({successful_mcmc/len(all_results)*100:.1f}%)")
    
    if successful_mcmc > 0:
        mcmc_df = summary_df[summary_df['MCMC_Success']]
        avg_acceptance = mcmc_df['MCMC_Acceptance_Rate'].mean()
        converged_count = mcmc_df['MCMC_Converged'].sum()
        avg_time = mcmc_df['Computation_Time'].mean()
        
        print(f"  平均MCMC接受率: {avg_acceptance:.3f}")
        print(f"  MCMC收敛率: {converged_count}/{successful_mcmc} ({converged_count/successful_mcmc*100:.1f}%)")
        print(f"  平均计算时间: {avg_time:.1f}秒")
    
    print(f"  摘要文件已保存: {summary_path}")

# =====================
# 8. 主程序入口
# =====================
def main():
    """主程序"""
    print("=" * 80)
    print("法医混合STR图谱贡献者比例推断系统 (MGM-RF方法)")
    print("基于Q1随机森林特征工程 + MGM-M基因型边缘化MCMC")
    print("=" * 80)
    
    # 检查必要文件
    if not os.path.exists(config.ATTACHMENT2_PATH):
        print(f"错误: 附件2文件不存在 - {config.ATTACHMENT2_PATH}")
        return
    
    # 设置随机种子
    np.random.seed(config.RANDOM_STATE)
    
    # 选择分析模式
    print("\n请选择分析模式:")
    print("1. 分析单个样本")
    print("2. 批量分析所有样本")
    print("3. 批量分析前N个样本")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 单样本分析
        sample_id = input("请输入样本ID: ").strip()
        
        try:
            result = analyze_single_sample_from_att2(
                sample_id, 
                config.ATTACHMENT2_PATH, 
                config.Q1_MODEL_PATH
            )
            
            print("\n=== 分析结果 ===")
            print_sample_summary(result)
            
        except Exception as e:
            print(f"分析失败: {e}")
    
    elif choice == "2":
        # 批量分析所有样本
        try:
            all_results = analyze_all_samples_from_att2(
                config.ATTACHMENT2_PATH,
                config.Q1_MODEL_PATH
            )
            
        except Exception as e:
            print(f"批量分析失败: {e}")
    
    elif choice == "3":
        # 批量分析前N个样本
        try:
            max_samples = int(input("请输入要分析的样本数量: ").strip())
            
            all_results = analyze_all_samples_from_att2(
                config.ATTACHMENT2_PATH,
                config.Q1_MODEL_PATH,
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