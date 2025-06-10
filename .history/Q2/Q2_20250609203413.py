# -*- coding: utf-8 -*-
"""
问题二优化版：基于V5特征的智能先验混合比例推断系统
版本: V2.0 - Enhanced Priors with V5 Feature-Driven Informative Priors
核心改进：
1. V5特征驱动的动态先验调整
2. 混合比例的不平衡性预测
3. 贡献者主次关系的先验建模
4. 增强的MCMC采样策略
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class EnhancedPriorCalculator:
    """增强的先验计算器：基于V5特征动态调整混合比例先验"""
    
    def __init__(self):
        self.feature_weights = {
            # 不平衡性指标
            'skewness_peak_height': 0.3,           # 峰高分布偏度
            'ratio_severe_imbalance_loci': 0.25,   # 严重失衡位点比例
            'avg_locus_allele_entropy': 0.2,       # 平均位点等位基因熵
            'inter_locus_balance_entropy': 0.15,   # 位点间平衡熵
            'mac_profile': 0.1                     # 样本最大等位基因数
        }
        
        # 混合模式分类阈值
        self.thresholds = {
            'highly_imbalanced': 0.75,    # 高度不平衡
            'moderately_imbalanced': 0.5, # 中度不平衡
            'balanced': 0.25              # 相对平衡
        }
        
        logger.info("增强先验计算器初始化完成")
    
    def predict_mixture_pattern(self, v5_features: Dict) -> Dict:
        """基于V5特征预测混合模式"""
        
        # 1. 计算不平衡性指标
        imbalance_score = 0.0
        feature_contributions = {}
        
        # 峰高分布偏度 (越大越不平衡)
        skewness = abs(v5_features.get('skewness_peak_height', 0.0))
        skew_score = min(1.0, skewness / 2.0)  # 标准化到[0,1]
        imbalance_score += skew_score * self.feature_weights['skewness_peak_height']
        feature_contributions['skewness'] = skew_score
        
        # 严重失衡位点比例 (越大越不平衡)
        severe_imbalance_ratio = v5_features.get('ratio_severe_imbalance_loci', 0.0)
        imbalance_score += severe_imbalance_ratio * self.feature_weights['ratio_severe_imbalance_loci']
        feature_contributions['severe_imbalance'] = severe_imbalance_ratio
        
        # 平均位点等位基因熵 (越小越不平衡)
        avg_entropy = v5_features.get('avg_locus_allele_entropy', 1.0)
        max_entropy = np.log(4)  # 假设最多4个等位基因
        entropy_score = 1.0 - min(1.0, avg_entropy / max_entropy)
        imbalance_score += entropy_score * self.feature_weights['avg_locus_allele_entropy']
        feature_contributions['low_entropy'] = entropy_score
        
        # 位点间平衡熵 (越小越不平衡)
        inter_entropy = v5_features.get('inter_locus_balance_entropy', 1.0)
        max_inter_entropy = np.log(20)  # 假设20个位点
        inter_entropy_score = 1.0 - min(1.0, inter_entropy / max_inter_entropy)
        imbalance_score += inter_entropy_score * self.feature_weights['inter_locus_balance_entropy']
        feature_contributions['low_inter_entropy'] = inter_entropy_score
        
        # 最大等位基因数 (越大可能越不平衡)
        mac = v5_features.get('mac_profile', 2)
        mac_score = min(1.0, (mac - 2) / 6)  # 标准化，2个等位基因是最平衡的
        imbalance_score += mac_score * self.feature_weights['mac_profile']
        feature_contributions['high_mac'] = mac_score
        
        # 2. 分类混合模式
        if imbalance_score >= self.thresholds['highly_imbalanced']:
            mixture_pattern = 'highly_imbalanced'
            pattern_description = "高度不平衡混合（主要贡献者占优）"
        elif imbalance_score >= self.thresholds['moderately_imbalanced']:
            mixture_pattern = 'moderately_imbalanced'
            pattern_description = "中度不平衡混合（存在主次贡献者）"
        else:
            mixture_pattern = 'balanced'
            pattern_description = "相对平衡混合（贡献者比例相近）"
        
        # 3. 预测主要贡献者比例
        if mixture_pattern == 'highly_imbalanced':
            predicted_major_ratio = 0.7 + 0.2 * (imbalance_score - self.thresholds['highly_imbalanced']) / (1.0 - self.thresholds['highly_imbalanced'])
        elif mixture_pattern == 'moderately_imbalanced':
            predicted_major_ratio = 0.5 + 0.2 * (imbalance_score - self.thresholds['moderately_imbalanced']) / (self.thresholds['highly_imbalanced'] - self.thresholds['moderately_imbalanced'])
        else:
            predicted_major_ratio = 0.4 + 0.1 * imbalance_score / self.thresholds['balanced']
        
        return {
            'mixture_pattern': mixture_pattern,
            'pattern_description': pattern_description,
            'imbalance_score': imbalance_score,
            'predicted_major_ratio': predicted_major_ratio,
            'feature_contributions': feature_contributions,
            'confidence': min(1.0, imbalance_score * 2)  # 预测置信度
        }
    
    def calculate_informative_prior_alpha(self, N: int, mixture_prediction: Dict) -> np.ndarray:
        """基于混合模式预测计算Dirichlet先验的alpha参数"""
        
        pattern = mixture_prediction['mixture_pattern']
        predicted_major_ratio = mixture_prediction['predicted_major_ratio']
        confidence = mixture_prediction['confidence']
        
        # 基础alpha值
        base_alpha = 1.0
        
        if pattern == 'highly_imbalanced':
            # 高度不平衡：强烈倾向于一个主要贡献者
            alpha = np.ones(N) * 0.3  # 次要贡献者的低alpha
            alpha[0] = base_alpha * 3.0  # 主要贡献者的高alpha
            
            # 进一步调整以反映预测的主要比例
            concentration_factor = confidence * 2.0
            alpha[0] = alpha[0] * (1 + concentration_factor)
            
        elif pattern == 'moderately_imbalanced':
            # 中度不平衡：存在主次关系但不极端
            alpha = np.ones(N) * 0.7
            alpha[0] = base_alpha * 1.8
            
            # 如果N=2，调整第二个贡献者
            if N == 2:
                ratio_factor = predicted_major_ratio / (1 - predicted_major_ratio)
                alpha[0] = base_alpha * ratio_factor
                alpha[1] = base_alpha
            
        else:
            # 相对平衡：使用接近均匀的先验，但仍略有偏向
            alpha = np.ones(N) * 0.8
            alpha[0] = base_alpha * 1.2
        
        # 应用置信度调整
        concentration_adjustment = 1 + confidence
        alpha = alpha * concentration_adjustment
        
        # 确保alpha值在合理范围内
        alpha = np.clip(alpha, 0.1, 10.0)
        
        logger.info(f"计算得到的Dirichlet先验alpha参数: {alpha}")
        logger.info(f"对应的期望比例: {alpha / np.sum(alpha)}")
        
        return alpha
    
    def calculate_enhanced_prior_log_probability(self, mixture_ratios: np.ndarray, 
                                               alpha: np.ndarray) -> float:
        """计算增强的混合比例先验对数概率"""
        
        # Dirichlet分布的对数概率密度
        log_prior = (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + 
                    np.sum((alpha - 1) * np.log(mixture_ratios + 1e-10)))
        
        return log_prior
    
    def sample_informative_prior(self, alpha: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """从信息先验中采样，用于验证先验合理性"""
        samples = np.random.dirichlet(alpha, n_samples)
        return samples
    
    def visualize_prior_distribution(self, alpha: np.ndarray, title: str = "先验分布"):
        """可视化先验分布"""
        n_samples = 5000
        samples = self.sample_informative_prior(alpha, n_samples)
        N = len(alpha)
        
        fig, axes = plt.subplots(1, N, figsize=(4*N, 4))
        if N == 1:
            axes = [axes]
        
        for i in range(N):
            axes[i].hist(samples[:, i], bins=50, alpha=0.7, density=True, 
                        color=f'C{i}', edgecolor='black', linewidth=0.5)
            axes[i].set_title(f'Mx_{i+1} 先验分布')
            axes[i].set_xlabel(f'Mx_{i+1}')
            axes[i].set_ylabel('概率密度')
            
            # 添加统计信息
            mean_val = np.mean(samples[:, i])
            median_val = np.median(samples[:, i])
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'均值: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', 
                           label=f'中位数: {median_val:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

class EnhancedMGM_RF_Inferencer:
    """增强的MGM-RF推断器，集成智能先验系统"""
    
    def __init__(self, original_inferencer, enhanced_prior_calculator):
        # 继承原始推断器的所有功能
        self.__dict__.update(original_inferencer.__dict__)
        
        # 添加增强的先验计算器
        self.enhanced_prior_calculator = enhanced_prior_calculator
        self.current_alpha = None
        self.mixture_prediction = None
        
        logger.info("增强MGM-RF推断器初始化完成")
    
    def predict_and_set_informative_prior(self, v5_features: Dict, N: int):
        """预测混合模式并设置信息先验"""
        
        # 预测混合模式
        self.mixture_prediction = self.enhanced_prior_calculator.predict_mixture_pattern(v5_features)
        
        # 计算先验参数
        self.current_alpha = self.enhanced_prior_calculator.calculate_informative_prior_alpha(
            N, self.mixture_prediction)
        
        logger.info(f"混合模式预测: {self.mixture_prediction['pattern_description']}")
        logger.info(f"不平衡评分: {self.mixture_prediction['imbalance_score']:.3f}")
        logger.info(f"预测主要贡献者比例: {self.mixture_prediction['predicted_major_ratio']:.3f}")
        logger.info(f"预测置信度: {self.mixture_prediction['confidence']:.3f}")
    
    def calculate_enhanced_prior_mixture_ratios(self, mixture_ratios: np.ndarray) -> float:
        """计算增强的混合比例先验概率"""
        if self.current_alpha is None:
            # 如果没有设置信息先验，使用原始的无信息先验
            alpha = np.ones(len(mixture_ratios))
            logger.warning("使用默认无信息先验")
        else:
            alpha = self.current_alpha
        
        return self.enhanced_prior_calculator.calculate_enhanced_prior_log_probability(
            mixture_ratios, alpha)
    
    def propose_enhanced_mixture_ratios(self, current_ratios: np.ndarray, 
                                      step_size: float = 0.05) -> np.ndarray:
        """增强的混合比例提议函数，考虑先验信息"""
        
        if self.current_alpha is not None:
            # 使用先验信息指导提议
            proposal_concentration = self.current_alpha * step_size * 10
            proposal_concentration = np.maximum(proposal_concentration, 0.1)
            
            # 混合当前状态和先验信息
            effective_concentration = (current_ratios * (1 - step_size) + 
                                     proposal_concentration * step_size)
            effective_concentration = np.maximum(effective_concentration, 0.1)
            
            new_ratios = np.random.dirichlet(effective_concentration)
        else:
            # 回退到原始提议方法
            concentration = current_ratios / step_size
            concentration = np.maximum(concentration, 0.1)
            new_ratios = np.random.dirichlet(concentration)
        
        new_ratios = np.maximum(new_ratios, 1e-6)
        new_ratios = new_ratios / np.sum(new_ratios)
        
        return new_ratios
    
    def enhanced_mcmc_sampler(self, observed_data: Dict, N: int, 
                            att2_data: Dict = None) -> Dict:
        """增强的MCMC采样器，使用智能先验"""
        
        if self.v5_integrator is None:
            raise ValueError("请先设置V5特征")
        
        # 确保已设置信息先验
        if self.current_alpha is None:
            logger.warning("未设置信息先验，使用V5特征自动设置")
            self.predict_and_set_informative_prior(self.v5_integrator.v5_features, N)
        
        logger.info(f"开始增强MGM-RF MCMC采样，贡献者数量: {N}")
        logger.info(f"使用信息先验: alpha = {self.current_alpha}")
        
        # 初始化混合比例 - 使用先验信息
        if self.current_alpha is not None:
            # 从先验分布采样初始值
            mixture_ratios = np.random.dirichlet(self.current_alpha)
        else:
            mixture_ratios = np.random.dirichlet(np.ones(N))
        
        logger.info(f"初始混合比例: {mixture_ratios}")
        
        # 存储MCMC样本
        samples = {
            'mixture_ratios': [],
            'log_likelihood': [],
            'log_posterior': [],
            'log_prior': [],
            'acceptance_info': []
        }
        
        # 计算初始概率
        current_log_likelihood = self.calculate_total_marginalized_likelihood(
            observed_data, N, mixture_ratios, att2_data)
        current_log_prior = self.calculate_enhanced_prior_mixture_ratios(mixture_ratios)
        current_log_posterior = current_log_likelihood + current_log_prior
        
        logger.info(f"初始似然: {current_log_likelihood:.2f}")
        logger.info(f"初始先验: {current_log_prior:.2f}")
        logger.info(f"初始后验: {current_log_posterior:.2f}")
        
        # MCMC主循环
        n_accepted = 0
        acceptance_details = []
        
        # 自适应步长 - 根据先验信息调整
        if self.mixture_prediction and self.mixture_prediction['confidence'] > 0.7:
            step_size = 0.03  # 高置信度时使用较小步长
        else:
            step_size = 0.05  # 默认步长
        
        adaptation_interval = 500
        target_acceptance = 0.4
        
        for iteration in range(self.n_iterations):
            if iteration % 1000 == 0:
                acceptance_rate = n_accepted / max(iteration, 1)
                logger.info(f"迭代 {iteration}/{self.n_iterations}, "
                          f"接受率: {acceptance_rate:.3f}, "
                          f"似然: {current_log_likelihood:.2f}, "
                          f"先验: {current_log_prior:.2f}")
            
            # 使用增强的提议函数
            proposed_ratios = self.propose_enhanced_mixture_ratios(mixture_ratios, step_size)
            
            # 计算提议状态的概率
            proposed_log_likelihood = self.calculate_total_marginalized_likelihood(
                observed_data, N, proposed_ratios, att2_data)
            proposed_log_prior = self.calculate_enhanced_prior_mixture_ratios(proposed_ratios)
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
            
            # 记录详细接受信息
            acceptance_details.append({
                'iteration': iteration,
                'accepted': accepted,
                'log_ratio': log_ratio,
                'accept_prob': accept_prob,
                'current_ratios': mixture_ratios.copy(),
                'log_likelihood': current_log_likelihood,
                'log_prior': current_log_prior
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
                    logger.info(f"  步长调整: {step_size:.4f}, 最近接受率: {recent_acceptance:.3f}")
            
            # 存储样本（预热后）
            if iteration >= self.n_warmup and iteration % self.thinning == 0:
                samples['mixture_ratios'].append(mixture_ratios.copy())
                samples['log_likelihood'].append(current_log_likelihood)
                samples['log_posterior'].append(current_log_posterior)
                samples['log_prior'].append(current_log_prior)
        
        final_acceptance_rate = n_accepted / self.n_iterations
        logger.info(f"增强MCMC完成，总接受率: {final_acceptance_rate:.3f}")
        logger.info(f"有效样本数: {len(samples['mixture_ratios'])}")
        
        # 分析先验vs后验
        self._analyze_prior_posterior_comparison(samples, N)
        
        return {
            'samples': samples,
            'acceptance_rate': final_acceptance_rate,
            'n_samples': len(samples['mixture_ratios']),
            'acceptance_details': acceptance_details,
            'final_step_size': step_size,
            'converged': 0.15 <= final_acceptance_rate <= 0.6,
            'prior_alpha': self.current_alpha.copy() if self.current_alpha is not None else None,
            'mixture_prediction': self.mixture_prediction
        }
    
    def _analyze_prior_posterior_comparison(self, samples: Dict, N: int):
        """分析先验与后验的对比"""
        if not samples['mixture_ratios']:
            return
        
        posterior_samples = np.array(samples['mixture_ratios'])
        posterior_means = np.mean(posterior_samples, axis=0)
        
        if self.current_alpha is not None:
            prior_means = self.current_alpha / np.sum(self.current_alpha)
            
            logger.info("先验vs后验对比:")
            for i in range(N):
                shift = posterior_means[i] - prior_means[i]
                logger.info(f"  Mx_{i+1}: 先验={prior_means[i]:.3f}, 后验={posterior_means[i]:.3f}, 偏移={shift:+.3f}")

def create_enhanced_pipeline(q1_model_path: str = None):
    """创建增强的MGM-RF流水线"""
    
    # 导入原始组件
    from Q2_MGM_RF_Solution import MGM_RF_Inferencer, MGM_RF_Pipeline
    
    # 创建原始推断器
    original_inferencer = MGM_RF_Inferencer(q1_model_path)
    
    # 创建增强先验计算器
    enhanced_prior_calculator = EnhancedPriorCalculator()
    
    # 创建增强推断器
    enhanced_inferencer = EnhancedMGM_RF_Inferencer(
        original_inferencer, enhanced_prior_calculator)
    
    # 创建增强流水线
    class EnhancedMGM_RF_Pipeline(MGM_RF_Pipeline):
        def __init__(self, enhanced_inferencer):
            self.mgm_rf_inferencer = enhanced_inferencer
            self.results = {}
            logger.info("增强MGM-RF流水线初始化完成")
        
        def analyze_sample(self, sample_data: pd.DataFrame, 
                         att2_freq_data: Dict[str, List[str]] = None) -> Dict:
            """分析单个样本，使用增强的先验系统"""
            sample_file = sample_data.iloc[0]['Sample File']
            logger.info(f"开始增强分析: {sample_file}")
            
            # 步骤1: 预测NoC和提取V5特征
            predicted_noc, noc_confidence, v5_features = \
                self.mgm_rf_inferencer.predict_noc_from_sample(sample_data)
            
            # 步骤2: 设置V5特征
            self.mgm_rf_inferencer.set_v5_features(v5_features)
            
            # 步骤3: 预测混合模式并设置信息先验
            self.mgm_rf_inferencer.predict_and_set_informative_prior(
                v5_features, predicted_noc)
            
            # 步骤4: 处理STR数据
            sample_peaks = self.mgm_rf_inferencer.q1_feature_engineering.process_peaks_simplified(sample_data)
            
            if sample_peaks.empty:
                logger.warning(f"样本 {sample_file} 没有有效峰数据")
                return self._get_enhanced_default_result(
                    sample_file, predicted_noc, noc_confidence, v5_features)
            
            # 步骤5: 准备观测数据
            observed_data = {}
            for locus, locus_group in sample_peaks.groupby('Marker'):
                alleles = locus_group['Allele'].tolist()
                heights = dict(zip(locus_group['Allele'], locus_group['Height']))
                
                observed_data[locus] = {
                    'locus': locus,
                    'alleles': alleles,
                    'heights': heights
                }
            
            # 步骤6: 增强MCMC推断
            import time
            start_time = time.time()
            mcmc_results = self.mgm_rf_inferencer.enhanced_mcmc_sampler(
                observed_data, predicted_noc, att2_freq_data)
            end_time = time.time()
            
            # 步骤7: 生成后验摘要和诊断
            posterior_summary = self.generate_posterior_summary(mcmc_results, predicted_noc)
            convergence_diagnostics = self.analyze_convergence(mcmc_results['samples'], predicted_noc)
            
            # 步骤8: 增强结果分析
            enhanced_analysis = self._analyze_enhanced_results(mcmc_results, v5_features)
            
            result = {
                'sample_file': sample_file,
                'predicted_noc': predicted_noc,
                'noc_confidence': noc_confidence,
                'v5_features': v5_features,
                'mixture_prediction': mcmc_results['mixture_prediction'],
                'mcmc_results': mcmc_results,
                'posterior_summary': posterior_summary,
                'convergence_diagnostics': convergence_diagnostics,
                'enhanced_analysis': enhanced_analysis,
                'computation_time': end_time - start_time,
                'observed_data': observed_data
            }
            
            logger.info(f"增强分析完成: {sample_file}, 耗时: {end_time - start_time:.1f}秒")
            return result
        
        def _get_enhanced_default_result(self, sample_file: str, predicted_noc: int, 
                                       noc_confidence: float, v5_features: Dict) -> Dict:
            """获取增强的默认结果"""
            # 使用先验预测的比例作为默认值
            mixture_prediction = self.mgm_rf_inferencer.enhanced_prior_calculator.predict_mixture_pattern(v5_features)
            
            if predicted_noc == 2 and mixture_prediction['mixture_pattern'] != 'balanced':
                major_ratio = mixture_prediction['predicted_major_ratio']
                default_ratios = [major_ratio, 1 - major_ratio]
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
                'mixture_prediction': mixture_prediction,
                'mcmc_results': None,
                'posterior_summary': default_summary,
                'convergence_diagnostics': {'status': 'No valid peaks - used prior prediction'},
                'enhanced_analysis': {'used_default': True},
                'computation_time': 0.0,
                'observed_data': {}
            }
        
        def _analyze_enhanced_results(self, mcmc_results: Dict, v5_features: Dict) -> Dict:
            """分析增强结果"""
            analysis = {
                'prior_effectiveness': {},
                'posterior_concentration': {},
                'prediction_accuracy': {}
            }
            
            if mcmc_results['samples']['mixture_ratios']:
                posterior_samples = np.array(mcmc_results['samples']['mixture_ratios'])
                
                # 分析后验分布的集中程度
                posterior_stds = np.std(posterior_samples, axis=0)
                analysis['posterior_concentration'] = {
                    'std_values': posterior_stds.tolist(),
                    'avg_std': np.mean(posterior_stds),
                    'concentration_level': 'high' if np.mean(posterior_stds) < 0.1 else 'medium' if np.mean(posterior_stds) < 0.2 else 'low'
                }
                
                # 分析先验的有效性
                if mcmc_results.get('prior_alpha') is not None:
                    prior_alpha = mcmc_results['prior_alpha']
                    prior_means = prior_alpha / np.sum(prior_alpha)
                    posterior_means = np.mean(posterior_samples, axis=0)
                    
                    # 计算先验和后验的KL散度
                    kl_divergence = self._calculate_kl_divergence(prior_means, posterior_means)
                    
                    analysis['prior_effectiveness'] = {
                        'prior_means': prior_means.tolist(),
                        'posterior_means': posterior_means.tolist(),
                        'mean_absolute_shift': np.mean(np.abs(posterior_means - prior_means)),
                        'kl_divergence': kl_divergence,
                        'prior_influence': 'strong' if kl_divergence < 0.1 else 'moderate' if kl_divergence < 0.5 else 'weak'
                    }
                
                # 分析预测准确性
                mixture_prediction = mcmc_results.get('mixture_prediction', {})
                if mixture_prediction:
                    predicted_major_ratio = mixture_prediction.get('predicted_major_ratio', 0.5)
                    actual_major_ratio = np.max(posterior_means)
                    
                    analysis['prediction_accuracy'] = {
                        'predicted_major_ratio': predicted_major_ratio,
                        'actual_major_ratio': actual_major_ratio,
                        'prediction_error': abs(actual_major_ratio - predicted_major_ratio),
                        'accuracy_level': 'excellent' if abs(actual_major_ratio - predicted_major_ratio) < 0.1 
                                        else 'good' if abs(actual_major_ratio - predicted_major_ratio) < 0.2 
                                        else 'fair'
                    }
            
            return analysis
        
        def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
            """计算KL散度"""
            p = np.clip(p, 1e-10, 1.0)
            q = np.clip(q, 1e-10, 1.0)
            return np.sum(p * np.log(p / q))
        
        def visualize_enhanced_results(self, results: Dict, output_dir: str):
            """可视化增强分析结果"""
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            sample_file = results['sample_file']
            
            # 1. 先验预测可视化
            if 'mixture_prediction' in results and results['mixture_prediction']:
                mixture_pred = results['mixture_prediction']
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # 特征贡献图
                contributions = mixture_pred['feature_contributions']
                features = list(contributions.keys())
                values = list(contributions.values())
                
                ax1.barh(features, values, color='skyblue', edgecolor='navy', alpha=0.7)
                ax1.set_xlabel('贡献分数')
                ax1.set_title('V5特征对不平衡性的贡献')
                ax1.grid(True, alpha=0.3)
                
                # 混合模式预测
                patterns = ['balanced', 'moderately_imbalanced', 'highly_imbalanced']
                pattern_scores = [0, 0, 0]
                current_pattern = mixture_pred['mixture_pattern']
                pattern_scores[patterns.index(current_pattern)] = mixture_pred['imbalance_score']
                
                colors = ['green', 'orange', 'red']
                ax2.bar(patterns, pattern_scores, color=colors, alpha=0.7, edgecolor='black')
                ax2.set_ylabel('不平衡评分')
                ax2.set_title(f'混合模式预测: {mixture_pred["pattern_description"]}')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # 先验分布可视化
                if results['mcmc_results'] and results['mcmc_results'].get('prior_alpha') is not None:
                    prior_alpha = results['mcmc_results']['prior_alpha']
                    N = len(prior_alpha)
                    
                    # 从先验采样
                    prior_samples = np.random.dirichlet(prior_alpha, 5000)
                    
                    for i in range(min(N, 2)):  # 最多显示2个组分
                        ax = ax3 if i == 0 else ax4
                        ax.hist(prior_samples[:, i], bins=50, alpha=0.7, density=True,
                               color=f'C{i}', edgecolor='black', linewidth=0.5)
                        ax.set_title(f'Mx_{i+1} 先验分布')
                        ax.set_xlabel(f'Mx_{i+1}')
                        ax.set_ylabel('概率密度')
                        ax.grid(True, alpha=0.3)
                        
                        # 添加预测值
                        predicted_val = mixture_pred['predicted_major_ratio'] if i == 0 else (1 - mixture_pred['predicted_major_ratio'])
                        ax.axvline(predicted_val, color='red', linestyle='--', linewidth=2,
                                  label=f'预测值: {predicted_val:.3f}')
                        ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{sample_file}_enhanced_prediction.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. 先验vs后验对比
            if (results['mcmc_results'] and results['mcmc_results']['samples']['mixture_ratios'] and
                results['mcmc_results'].get('prior_alpha') is not None):
                
                posterior_samples = np.array(results['mcmc_results']['samples']['mixture_ratios'])
                prior_alpha = results['mcmc_results']['prior_alpha']
                N = len(prior_alpha)
                
                fig, axes = plt.subplots(N, 1, figsize=(12, 4*N))
                if N == 1:
                    axes = [axes]
                
                # 从先验采样用于对比
                prior_samples = np.random.dirichlet(prior_alpha, len(posterior_samples))
                
                for i in range(N):
                    # 绘制先验和后验分布
                    axes[i].hist(prior_samples[:, i], bins=50, alpha=0.5, density=True,
                                color='blue', label='先验分布', edgecolor='navy')
                    axes[i].hist(posterior_samples[:, i], bins=50, alpha=0.7, density=True,
                                color='red', label='后验分布', edgecolor='darkred')
                    
                    # 添加统计信息
                    prior_mean = np.mean(prior_samples[:, i])
                    posterior_mean = np.mean(posterior_samples[:, i])
                    
                    axes[i].axvline(prior_mean, color='blue', linestyle='--', linewidth=2,
                                   label=f'先验均值: {prior_mean:.3f}')
                    axes[i].axvline(posterior_mean, color='red', linestyle='--', linewidth=2,
                                   label=f'后验均值: {posterior_mean:.3f}')
                    
                    axes[i].set_title(f'Mx_{i+1}: 先验 vs 后验分布对比')
                    axes[i].set_xlabel(f'Mx_{i+1}')
                    axes[i].set_ylabel('概率密度')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{sample_file}_prior_posterior_comparison.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # 调用原始的可视化方法
            super().plot_results(results, output_dir)
    
    return EnhancedMGM_RF_Pipeline(enhanced_inferencer)

# 使用示例和测试函数
def test_enhanced_prior_system():
    """测试增强先验系统"""
    print("="*80)
    print("增强先验系统测试")
    print("="*80)
    
    # 创建测试V5特征
    test_features = [
        {
            'name': '高度不平衡样本',
            'features': {
                'skewness_peak_height': 2.5,      # 高偏度
                'ratio_severe_imbalance_loci': 0.8, # 80%位点严重失衡
                'avg_locus_allele_entropy': 0.3,   # 低熵
                'inter_locus_balance_entropy': 1.0, # 低位点间熵
                'mac_profile': 6                    # 高最大等位基因数
            }
        },
        {
            'name': '中度不平衡样本',
            'features': {
                'skewness_peak_height': 1.2,
                'ratio_severe_imbalance_loci': 0.4,
                'avg_locus_allele_entropy': 0.8,
                'inter_locus_balance_entropy': 2.0,
                'mac_profile': 4
            }
        },
        {
            'name': '相对平衡样本',
            'features': {
                'skewness_peak_height': 0.3,
                'ratio_severe_imbalance_loci': 0.1,
                'avg_locus_allele_entropy': 1.2,
                'inter_locus_balance_entropy': 2.8,
                'mac_profile': 3
            }
        }
    ]
    
    # 创建增强先验计算器
    enhanced_prior_calc = EnhancedPriorCalculator()
    
    for test_case in test_features:
        print(f"\n--- {test_case['name']} ---")
        
        # 预测混合模式
        mixture_pred = enhanced_prior_calc.predict_mixture_pattern(test_case['features'])
        
        print(f"预测模式: {mixture_pred['pattern_description']}")
        print(f"不平衡评分: {mixture_pred['imbalance_score']:.3f}")
        print(f"预测主要贡献者比例: {mixture_pred['predicted_major_ratio']:.3f}")
        print(f"预测置信度: {mixture_pred['confidence']:.3f}")
        
        # 计算2人混合的先验参数
        alpha_2 = enhanced_prior_calc.calculate_informative_prior_alpha(2, mixture_pred)
        print(f"2人混合Dirichlet先验α: {alpha_2}")
        print(f"期望比例: {alpha_2 / np.sum(alpha_2)}")
        
        # 可视化先验分布
        fig = enhanced_prior_calc.visualize_prior_distribution(
            alpha_2, f"{test_case['name']} - 2人混合先验分布")
        plt.show()

def analyze_sample_with_enhanced_priors(sample_id: str, att2_path: str, 
                                      q1_model_path: str = None) -> Dict:
    """使用增强先验系统分析样本"""
    print(f"\n{'='*80}")
    print(f"使用增强先验系统分析样本: {sample_id}")
    print(f"{'='*80}")
    
    # 创建增强流水线
    enhanced_pipeline = create_enhanced_pipeline(q1_model_path)
    
    # 加载数据
    att2_data = enhanced_pipeline.load_attachment2_data(att2_path)
    
    if sample_id not in att2_data:
        raise ValueError(f"样本 {sample_id} 不存在")
    
    # 准备频率数据
    att2_freq_data = enhanced_pipeline.prepare_att2_frequency_data(att2_data)
    
    # 分析样本
    sample_data = att2_data[sample_id]
    result = enhanced_pipeline.analyze_sample(sample_data, att2_freq_data)
    
    # 保存结果
    output_dir = './enhanced_mgm_rf_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{sample_id}_enhanced_result.json')
    enhanced_pipeline.save_results(result, output_file)
    
    # 可视化结果
    enhanced_pipeline.visualize_enhanced_results(result, output_dir)
    
    # 打印详细结果
    print_enhanced_analysis_summary(result)
    
    return result

def print_enhanced_analysis_summary(result: Dict):
    """打印增强分析摘要"""
    print(f"\n{'='*60}")
    print(f"增强分析结果摘要")
    print(f"{'='*60}")
    
    sample_file = result['sample_file']
    predicted_noc = result['predicted_noc']
    noc_confidence = result['noc_confidence']
    
    print(f"样本ID: {sample_file}")
    print(f"预测NoC: {predicted_noc} (置信度: {noc_confidence:.3f})")
    
    # 混合模式预测
    if 'mixture_prediction' in result:
        mixture_pred = result['mixture_prediction']
        print(f"\n📊 混合模式预测:")
        print(f"  模式: {mixture_pred['pattern_description']}")
        print(f"  不平衡评分: {mixture_pred['imbalance_score']:.3f}")
        print(f"  预测主要贡献者比例: {mixture_pred['predicted_major_ratio']:.3f}")
        print(f"  预测置信度: {mixture_pred['confidence']:.3f}")
        
        print(f"\n🔍 V5特征贡献分析:")
        for feature, contribution in mixture_pred['feature_contributions'].items():
            print(f"  {feature}: {contribution:.3f}")
    
    # MCMC结果
    if result['mcmc_results'] is not None:
        mcmc_results = result['mcmc_results']
        print(f"\n⚙️ MCMC采样结果:")
        print(f"  接受率: {mcmc_results['acceptance_rate']:.3f}")
        print(f"  收敛状态: {'是' if mcmc_results['converged'] else '否'}")
        print(f"  有效样本数: {mcmc_results['n_samples']}")
        
        if mcmc_results.get('prior_alpha') is not None:
            prior_alpha = mcmc_results['prior_alpha']
            print(f"  使用的先验α: {prior_alpha}")
        
        # 后验结果
        posterior_summary = result['posterior_summary']
        contributor_ranking = posterior_summary['contributor_ranking']
        
        print(f"\n🎯 混合比例后验估计:")
        for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
            mx_stats = posterior_summary[f'Mx_{contributor_id}']
            ci_95 = mx_stats['credible_interval_95']
            std_val = mx_stats['std']
            print(f"  贡献者{contributor_id}: {mean_ratio:.4f} ± {std_val:.4f}")
            print(f"    95%置信区间: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    
    # 增强分析结果
    if 'enhanced_analysis' in result:
        enhanced_analysis = result['enhanced_analysis']
        
        if 'prior_effectiveness' in enhanced_analysis:
            prior_eff = enhanced_analysis['prior_effectiveness']
            print(f"\n📈 先验有效性分析:")
            print(f"  先验影响程度: {prior_eff.get('prior_influence', 'unknown')}")
            print(f"  平均偏移: {prior_eff.get('mean_absolute_shift', 0):.4f}")
            print(f"  KL散度: {prior_eff.get('kl_divergence', 0):.4f}")
        
        if 'prediction_accuracy' in enhanced_analysis:
            pred_acc = enhanced_analysis['prediction_accuracy']
            print(f"\n🎯 预测准确性:")
            print(f"  准确性水平: {pred_acc.get('accuracy_level', 'unknown')}")
            print(f"  预测误差: {pred_acc.get('prediction_error', 0):.4f}")
    
    print(f"\n⏱️ 计算时间: {result['computation_time']:.1f}秒")
    print(f"{'='*60}")

# 主程序接口
def main_enhanced():
    """增强系统主程序"""
    print("法医混合STR图谱贡献者比例推断系统 - 增强先验版本")
    print("基于V5特征驱动的智能先验 + MGM-M方法")
    print("="*80)
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 测试增强先验系统")
    print("2. 分析单个样本（增强版）")
    print("3. 对比分析（原版 vs 增强版）")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_enhanced_prior_system()
    
    elif choice == "2":
        att2_path = input("请输入附件2路径 (默认: 附件2：混合STR图谱数据.csv): ").strip()
        if not att2_path:
            att2_path = "附件2：混合STR图谱数据.csv"
        
        sample_id = input("请输入样本ID: ").strip()
        q1_model_path = input("请输入Q1模型路径 (可选，按Enter跳过): ").strip()
        
        try:
            result = analyze_sample_with_enhanced_priors(
                sample_id, att2_path, q1_model_path if q1_model_path else None)
        except Exception as e:
            print(f"分析失败: {e}")
    
    elif choice == "3":
        print("对比分析功能开发中...")
        # TODO: 实现原版vs增强版的对比分析
    
    else:
        print("无效选择")

if __name__ == "__main__":
    main_enhanced()