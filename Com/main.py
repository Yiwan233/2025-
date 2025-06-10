# -*- coding: utf-8 -*-
"""
法医DNA分析 - Q2/Q3/Q4混合比例预测均方误差(MSE)计算器

版本: V1.0
日期: 2025-06-10
描述: 计算Q2、Q3、Q4方法的混合比例预测均方误差
功能:
1. 从文件名解析真实混合比例
2. 加载各方法的预测结果
3. 计算MSE、MAE、RMSE等评估指标
4. 生成对比分析报告和可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import re
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("                Q2/Q3/Q4混合比例预测均方误差(MSE)计算器")
print("           Mixture Ratio Prediction MSE Calculator")
print("=" * 80)

class MixtureRatioMSECalculator:
    """混合比例MSE计算器"""
    
    def __init__(self):
        self.true_ratios = {}  # 真实混合比例
        self.q2_results = {}   # Q2预测结果
        self.q3_results = {}   # Q3预测结果  
        self.q4_results = {}   # Q4预测结果
        self.evaluation_results = {}
        
        # 结果保存目录
        self.output_dir = './mse_evaluation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("混合比例MSE计算器初始化完成")
    
    def parse_mixture_ratio_from_filename(self, filename: str) -> Tuple[List[str], List[float]]:
        """
        从文件名解析混合比例
        
        例如: "A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa"
        解析为: 贡献者IDs=['40', '41'], 混合比例=[0.2, 0.8] (1:4标准化后)
        """
        try:
            # 匹配模式: -贡献者IDs-比例-
            pattern = r'-(\d+(?:_\d+)*)-([^-]*?)-M\d'
            match = re.search(pattern, str(filename))
            
            if not match:
                logger.warning(f"无法从文件名解析贡献者信息: {filename}")
                return None, None
            
            contributor_ids = match.group(1).split('_')
            ratio_part = match.group(2)
            
            logger.debug(f"解析文件名: {filename}")
            logger.debug(f"  贡献者IDs: {contributor_ids}")
            logger.debug(f"  比例部分: '{ratio_part}'")
            
            # 解析比例部分
            if ';' in ratio_part:
                try:
                    ratio_values = [float(x) for x in ratio_part.split(';')]
                    
                    if len(ratio_values) == len(contributor_ids):
                        # 标准化为概率
                        total = sum(ratio_values)
                        if total > 0:
                            true_ratios = [r/total for r in ratio_values]
                            logger.debug(f"  解析得到的标准化比例: {true_ratios}")
                            return contributor_ids, true_ratios
                        else:
                            logger.warning(f"比例值总和为0: {filename}")
                    else:
                        logger.warning(f"比例数量({len(ratio_values)})与贡献者数量({len(contributor_ids)})不匹配")
                except ValueError as e:
                    logger.warning(f"解析比例值失败: {e}")
            
            # 如果解析失败，假设等比例
            true_ratios = [1.0/len(contributor_ids)] * len(contributor_ids)
            logger.debug(f"  使用等比例假设: {true_ratios}")
            return contributor_ids, true_ratios
            
        except Exception as e:
            logger.error(f"解析文件名时出错: {e}")
            return None, None
    
    def load_true_ratios_from_data(self, data_path: str):
        """从数据文件加载真实混合比例"""
        logger.info(f"从数据文件加载真实混合比例: {data_path}")
        
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
            sample_files = df['Sample File'].unique()
            
            logger.info(f"发现 {len(sample_files)} 个样本")
            
            for sample_file in sample_files:
                contributor_ids, true_ratios = self.parse_mixture_ratio_from_filename(sample_file)
                
                if contributor_ids and true_ratios:
                    self.true_ratios[sample_file] = {
                        'contributor_ids': contributor_ids,
                        'ratios': true_ratios,
                        'noc': len(contributor_ids)
                    }
                    logger.info(f"样本 {sample_file}: {len(contributor_ids)}个贡献者, 比例={true_ratios}")
            
            logger.info(f"成功解析 {len(self.true_ratios)} 个样本的真实混合比例")
            
        except Exception as e:
            logger.error(f"加载真实比例失败: {e}")
    
    def load_q2_results(self, q2_results_dir: str):
        """加载Q2方法的预测结果"""
        logger.info(f"加载Q2预测结果: {q2_results_dir}")
        
        try:
            q2_dir = Path(q2_results_dir)
            if not q2_dir.exists():
                logger.warning(f"Q2结果目录不存在: {q2_results_dir}")
                return
            
            # 查找Q2结果文件
            result_files = list(q2_dir.glob("*_result.json")) + list(q2_dir.glob("*_analysis_result.json"))
            
            logger.info(f"找到 {len(result_files)} 个Q2结果文件")
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    sample_file = result_data.get('sample_file', '')
                    if not sample_file:
                        # 尝试从文件名提取样本ID
                        sample_file = result_file.stem.replace('_result', '').replace('_analysis_result', '')
                    
                    # 提取混合比例
                    posterior_summary = result_data.get('posterior_summary', {})
                    if posterior_summary:
                        predicted_ratios = []
                        noc = result_data.get('predicted_noc', 0)
                        
                        for i in range(1, noc + 1):
                            mx_key = f'Mx_{i}'
                            if mx_key in posterior_summary:
                                ratio = posterior_summary[mx_key].get('mean', 0)
                                predicted_ratios.append(ratio)
                        
                        if predicted_ratios:
                            self.q2_results[sample_file] = {
                                'ratios': predicted_ratios,
                                'noc': noc,
                                'confidence': result_data.get('noc_confidence', 0),
                                'mcmc_converged': result_data.get('mcmc_quality', {}).get('converged', False)
                            }
                            logger.debug(f"Q2 - {sample_file}: {predicted_ratios}")
                
                except Exception as e:
                    logger.warning(f"加载Q2结果文件失败 {result_file}: {e}")
            
            logger.info(f"成功加载 {len(self.q2_results)} 个Q2预测结果")
            
        except Exception as e:
            logger.error(f"加载Q2结果失败: {e}")
    
    def load_q3_results(self, q3_results_dir: str):
        """加载Q3方法的预测结果"""
        logger.info(f"加载Q3预测结果: {q3_results_dir}")
        
        try:
            q3_dir = Path(q3_results_dir)
            if not q3_dir.exists():
                logger.warning(f"Q3结果目录不存在: {q3_results_dir}")
                return
            
            # 查找Q3结果文件
            result_files = list(q3_dir.glob("*_result.json")) + list(q3_dir.glob("*_analysis_result.json"))
            
            logger.info(f"找到 {len(result_files)} 个Q3结果文件")
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    sample_file = result_data.get('sample_id', '')
                    if not sample_file:
                        sample_file = result_file.stem.replace('_result', '').replace('_analysis_result', '')
                    
                    # Q3的结果可能包含基因型信息和混合比例
                    # 如果Q3也输出了混合比例，提取它们
                    mcmc_results = result_data.get('mcmc_results', {})
                    if mcmc_results and 'mixture_ratio_summary' in mcmc_results:
                        mx_summary = mcmc_results['mixture_ratio_summary']
                        predicted_ratios = []
                        noc = result_data.get('predicted_noc', 0)
                        
                        for i in range(1, noc + 1):
                            mx_key = f'Mx_{i}'
                            if mx_key in mx_summary:
                                ratio = mx_summary[mx_key].get('mean', 0)
                                predicted_ratios.append(ratio)
                        
                        if predicted_ratios:
                            self.q3_results[sample_file] = {
                                'ratios': predicted_ratios,
                                'noc': noc,
                                'confidence': result_data.get('noc_confidence', 0),
                                'mcmc_converged': mcmc_results.get('converged', False)
                            }
                            logger.debug(f"Q3 - {sample_file}: {predicted_ratios}")
                
                except Exception as e:
                    logger.warning(f"加载Q3结果文件失败 {result_file}: {e}")
            
            logger.info(f"成功加载 {len(self.q3_results)} 个Q3预测结果")
            
        except Exception as e:
            logger.error(f"加载Q3结果失败: {e}")
    
    def load_q4_results(self, q4_results_dir: str):
        """加载Q4方法的预测结果"""
        logger.info(f"加载Q4预测结果: {q4_results_dir}")
        
        try:
            q4_dir = Path(q4_results_dir)
            if not q4_dir.exists():
                logger.warning(f"Q4结果目录不存在: {q4_results_dir}")
                return
            
            # 查找Q4结果文件
            sample_dirs = [d for d in q4_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
            
            logger.info(f"找到 {len(sample_dirs)} 个Q4样本目录")
            
            for sample_dir in sample_dirs:
                try:
                    # 查找样本分析摘要文件
                    summary_files = list(sample_dir.glob("*_analysis_summary.json"))
                    
                    if not summary_files:
                        continue
                    
                    with open(summary_files[0], 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    sample_file = result_data.get('sample_file', '')
                    if not sample_file:
                        sample_file = sample_dir.name.replace('sample_', '')
                    
                    # Q4的结果主要是降噪，但如果有混合比例估计，提取它们
                    posterior_summary = result_data.get('posterior_summary', {})
                    if posterior_summary and 'mixture_ratio_summary' in posterior_summary:
                        mx_summary = posterior_summary['mixture_ratio_summary']
                        
                        # 找到最可能的NoC
                        most_probable_noc = posterior_summary.get('most_probable_noc', 2)
                        noc_key = f'NoC_{most_probable_noc}'
                        
                        if noc_key in mx_summary:
                            predicted_ratios = []
                            for i in range(1, most_probable_noc + 1):
                                mx_key = f'Mx_{i}'
                                if mx_key in mx_summary[noc_key]:
                                    ratio = mx_summary[noc_key][mx_key].get('mean', 0)
                                    predicted_ratios.append(ratio)
                            
                            if predicted_ratios:
                                self.q4_results[sample_file] = {
                                    'ratios': predicted_ratios,
                                    'noc': most_probable_noc,
                                    'confidence': posterior_summary.get('noc_confidence', 0),
                                    'mcmc_converged': posterior_summary.get('model_quality', {}).get('converged', False)
                                }
                                logger.debug(f"Q4 - {sample_file}: {predicted_ratios}")
                
                except Exception as e:
                    logger.warning(f"加载Q4结果失败 {sample_dir}: {e}")
            
            logger.info(f"成功加载 {len(self.q4_results)} 个Q4预测结果")
            
        except Exception as e:
            logger.error(f"加载Q4结果失败: {e}")
    
    def align_ratios(self, true_ratios: List[float], predicted_ratios: List[float]) -> Tuple[List[float], List[float]]:
        """
        对齐真实比例和预测比例
        处理贡献者数量不匹配的情况
        """
        true_ratios = np.array(true_ratios)
        predicted_ratios = np.array(predicted_ratios)
        
        # 如果长度相同，直接返回
        if len(true_ratios) == len(predicted_ratios):
            return true_ratios.tolist(), predicted_ratios.tolist()
        
        # 如果预测的贡献者数量少于真实数量，用0填充
        if len(predicted_ratios) < len(true_ratios):
            padded_predicted = np.zeros(len(true_ratios))
            padded_predicted[:len(predicted_ratios)] = predicted_ratios
            return true_ratios.tolist(), padded_predicted.tolist()
        
        # 如果预测的贡献者数量多于真实数量，截断
        if len(predicted_ratios) > len(true_ratios):
            # 按照预测比例大小排序，取前N个
            sorted_indices = np.argsort(predicted_ratios)[::-1]
            top_predicted = predicted_ratios[sorted_indices[:len(true_ratios)]]
            # 重新标准化
            top_predicted = top_predicted / np.sum(top_predicted)
            
            # 对应的真实比例也需要排序（按从大到小）
            true_sorted_indices = np.argsort(true_ratios)[::-1]
            aligned_true = true_ratios[true_sorted_indices]
            
            return aligned_true.tolist(), top_predicted.tolist()
        
        return true_ratios.tolist(), predicted_ratios.tolist()
    
    def calculate_mse_metrics(self, method_name: str, method_results: Dict) -> Dict:
        """计算单个方法的MSE等评估指标"""
        logger.info(f"计算 {method_name} 的评估指标")
        
        all_true_ratios = []
        all_predicted_ratios = []
        sample_wise_metrics = {}
        
        matched_samples = 0
        
        for sample_file, true_data in self.true_ratios.items():
            if sample_file not in method_results:
                continue
            
            true_ratios = true_data['ratios']
            predicted_data = method_results[sample_file]
            predicted_ratios = predicted_data['ratios']
            
            # 对齐比例
            aligned_true, aligned_predicted = self.align_ratios(true_ratios, predicted_ratios)
            
            # 计算样本级别的指标
            mse = mean_squared_error(aligned_true, aligned_predicted)
            mae = mean_absolute_error(aligned_true, aligned_predicted)
            rmse = np.sqrt(mse)
            
            # 计算最大绝对误差
            max_error = np.max(np.abs(np.array(aligned_true) - np.array(aligned_predicted)))
            
            sample_wise_metrics[sample_file] = {
                'true_ratios': aligned_true,
                'predicted_ratios': aligned_predicted,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'max_error': max_error,
                'true_noc': true_data['noc'],
                'predicted_noc': predicted_data['noc'],
                'noc_correct': true_data['noc'] == predicted_data['noc']
            }
            
            # 添加到总体计算
            all_true_ratios.extend(aligned_true)
            all_predicted_ratios.extend(aligned_predicted)
            matched_samples += 1
        
        if matched_samples == 0:
            logger.warning(f"{method_name} 没有匹配的样本")
            return {}
        
        # 计算总体指标
        overall_mse = mean_squared_error(all_true_ratios, all_predicted_ratios)
        overall_mae = mean_absolute_error(all_true_ratios, all_predicted_ratios)
        overall_rmse = np.sqrt(overall_mse)
        overall_max_error = np.max(np.abs(np.array(all_true_ratios) - np.array(all_predicted_ratios)))
        
        # 计算NoC准确率
        noc_correct_count = sum(1 for m in sample_wise_metrics.values() if m['noc_correct'])
        noc_accuracy = noc_correct_count / matched_samples
        
        # 计算不同误差范围的样本比例
        sample_mses = [m['mse'] for m in sample_wise_metrics.values()]
        excellent_samples = sum(1 for mse in sample_mses if mse < 0.01)  # MSE < 0.01
        good_samples = sum(1 for mse in sample_mses if 0.01 <= mse < 0.05)  # 0.01 <= MSE < 0.05
        fair_samples = sum(1 for mse in sample_mses if 0.05 <= mse < 0.1)   # 0.05 <= MSE < 0.1
        poor_samples = matched_samples - excellent_samples - good_samples - fair_samples
        
        overall_metrics = {
            'method_name': method_name,
            'matched_samples': matched_samples,
            'total_samples': len(self.true_ratios),
            'coverage_rate': matched_samples / len(self.true_ratios),
            
            # 总体误差指标
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_max_error': overall_max_error,
            
            # NoC准确性
            'noc_accuracy': noc_accuracy,
            
            # 样本级别误差分布
            'mean_sample_mse': np.mean(sample_mses),
            'median_sample_mse': np.median(sample_mses),
            'std_sample_mse': np.std(sample_mses),
            
            # 性能等级分布
            'excellent_samples': excellent_samples,
            'good_samples': good_samples,
            'fair_samples': fair_samples,
            'poor_samples': poor_samples,
            'excellent_rate': excellent_samples / matched_samples,
            'good_rate': good_samples / matched_samples,
            'fair_rate': fair_samples / matched_samples,
            'poor_rate': poor_samples / matched_samples,
            
            # 样本级别详细结果
            'sample_wise_metrics': sample_wise_metrics
        }
        
        logger.info(f"{method_name} 评估完成:")
        logger.info(f"  匹配样本数: {matched_samples}/{len(self.true_ratios)}")
        logger.info(f"  总体MSE: {overall_mse:.6f}")
        logger.info(f"  总体RMSE: {overall_rmse:.6f}")
        logger.info(f"  NoC准确率: {noc_accuracy:.3f}")
        
        return overall_metrics
    
    def calculate_all_metrics(self):
        """计算所有方法的评估指标"""
        logger.info("开始计算所有方法的评估指标")
        
        methods = {
            'Q2': self.q2_results,
            'Q3': self.q3_results,
            'Q4': self.q4_results
        }
        
        for method_name, method_results in methods.items():
            if method_results:
                metrics = self.calculate_mse_metrics(method_name, method_results)
                if metrics:
                    self.evaluation_results[method_name] = metrics
            else:
                logger.warning(f"{method_name} 没有预测结果")
        
        logger.info(f"完成 {len(self.evaluation_results)} 个方法的评估")
    
    def generate_comparison_report(self):
        """生成对比分析报告"""
        logger.info("生成对比分析报告")
        
        if not self.evaluation_results:
            logger.warning("没有评估结果，无法生成报告")
            return
        
        print(f"\n{'='*80}")
        print("                    Q2/Q3/Q4混合比例预测性能对比报告")
        print(f"{'='*80}")
        
        # 汇总表格
        print(f"\n📊 总体性能指标对比:")
        print(f"{'方法':<8} {'样本数':<8} {'覆盖率':<8} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'NoC准确率':<10}")
        print("-" * 80)
        
        for method_name, metrics in self.evaluation_results.items():
            print(f"{method_name:<8} "
                  f"{metrics['matched_samples']:<8} "
                  f"{metrics['coverage_rate']:<8.3f} "
                  f"{metrics['overall_mse']:<12.6f} "
                  f"{metrics['overall_rmse']:<12.6f} "
                  f"{metrics['overall_mae']:<12.6f} "
                  f"{metrics['noc_accuracy']:<10.3f}")
        
        # 性能等级分布
        print(f"\n📈 性能等级分布:")
        print(f"{'方法':<8} {'优秀(<0.01)':<12} {'良好(0.01-0.05)':<15} {'一般(0.05-0.1)':<15} {'较差(>0.1)':<12}")
        print("-" * 70)
        
        for method_name, metrics in self.evaluation_results.items():
            print(f"{method_name:<8} "
                  f"{metrics['excellent_rate']:<12.3f} "
                  f"{metrics['good_rate']:<15.3f} "
                  f"{metrics['fair_rate']:<15.3f} "
                  f"{metrics['poor_rate']:<12.3f}")
        
        # 最佳方法推荐
        print(f"\n🏆 最佳方法推荐:")
        
        best_mse_method = min(self.evaluation_results.keys(), 
                             key=lambda x: self.evaluation_results[x]['overall_mse'])
        best_noc_method = max(self.evaluation_results.keys(), 
                             key=lambda x: self.evaluation_results[x]['noc_accuracy'])
        best_coverage_method = max(self.evaluation_results.keys(), 
                                  key=lambda x: self.evaluation_results[x]['coverage_rate'])
        
        print(f"  • 最低MSE: {best_mse_method} (MSE={self.evaluation_results[best_mse_method]['overall_mse']:.6f})")
        print(f"  • 最高NoC准确率: {best_noc_method} (准确率={self.evaluation_results[best_noc_method]['noc_accuracy']:.3f})")
        print(f"  • 最高覆盖率: {best_coverage_method} (覆盖率={self.evaluation_results[best_coverage_method]['coverage_rate']:.3f})")
        
        # 详细样本分析
        print(f"\n📋 详细样本分析 (前5个样本):")
        
        # 找到所有方法都有结果的样本
        common_samples = set(self.true_ratios.keys())
        for method_results in [self.q2_results, self.q3_results, self.q4_results]:
            if method_results:
                common_samples &= set(method_results.keys())
        
        common_samples = list(common_samples)[:5]  # 只显示前5个
        
        for sample_file in common_samples:
            print(f"\n样本: {sample_file}")
            true_data = self.true_ratios[sample_file]
            print(f"  真实比例: {true_data['ratios']} (NoC={true_data['noc']})")
            
            for method_name, metrics in self.evaluation_results.items():
                if sample_file in metrics['sample_wise_metrics']:
                    sample_metrics = metrics['sample_wise_metrics'][sample_file]
                    print(f"  {method_name}: {sample_metrics['predicted_ratios']} "
                          f"(MSE={sample_metrics['mse']:.6f}, NoC={sample_metrics['predicted_noc']})")
    
    def plot_comparison_charts(self):
        """绘制对比分析图表"""
        logger.info("绘制对比分析图表")
        
        if not self.evaluation_results:
            logger.warning("没有评估结果，无法绘制图表")
            return
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 总体MSE对比
        plt.figure(figsize=(12, 8))
        
        methods = list(self.evaluation_results.keys())
        mse_values = [self.evaluation_results[m]['overall_mse'] for m in methods]
        rmse_values = [self.evaluation_results[m]['overall_rmse'] for m in methods]
        mae_values = [self.evaluation_results[m]['overall_mae'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, mse_values, width, label='MSE', alpha=0.8)
        plt.bar(x, rmse_values, width, label='RMSE', alpha=0.8)
        plt.bar(x + width, mae_values, width, label='MAE', alpha=0.8)
        
        plt.xlabel('方法', fontsize=12)
        plt.ylabel('误差值', fontsize=12)
        plt.title('Q2/Q3/Q4混合比例预测误差对比', fontsize=14, fontweight='bold')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mse, rmse, mae) in enumerate(zip(mse_values, rmse_values, mae_values)):
            plt.text(i - width, mse + max(mse_values) * 0.01, f'{mse:.4f}', 
                    ha='center', va='bottom', fontsize=10)
            plt.text(i, rmse + max(rmse_values) * 0.01, f'{rmse:.4f}', 
                    ha='center', va='bottom', fontsize=10)
            plt.text(i + width, mae + max(mae_values) * 0.01, f'{mae:.4f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overall_error_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 性能等级分布堆叠条形图
        plt.figure(figsize=(10, 6))
        
        performance_data = []
        for method in methods:
            metrics = self.evaluation_results[method]
            performance_data.append([
                metrics['excellent_rate'],
                metrics['good_rate'], 
                metrics['fair_rate'],
                metrics['poor_rate']
            ])
        
        performance_data = np.array(performance_data).T
        
        labels = ['优秀 (MSE<0.01)', '良好 (0.01≤MSE<0.05)', '一般 (0.05≤MSE<0.1)', '较差 (MSE≥0.1)']
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
        
        bottom = np.zeros(len(methods))
        for i, (data, label, color) in enumerate(zip(performance_data, labels, colors)):
            plt.bar(methods, data, bottom=bottom, label=label, color=color, alpha=0.8)
            
            # 添加百分比标签
            for j, value in enumerate(data):
                if value > 0.05:  # 只显示大于5%的标签
                    plt.text(j, bottom[j] + value/2, f'{value:.1%}', 
                            ha='center', va='center', fontweight='bold', fontsize=10)
            
            bottom += data
        
        plt.xlabel('方法', fontsize=12)
        plt.ylabel('样本比例', fontsize=12)
        plt.title('各方法性能等级分布', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. NoC准确率对比
        plt.figure(figsize=(8, 6))
        
        noc_accuracies = [self.evaluation_results[m]['noc_accuracy'] for m in methods]
        coverage_rates = [self.evaluation_results[m]['coverage_rate'] for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # NoC准确率
        bars1 = ax1.bar(methods, noc_accuracies, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax1.set_ylabel('NoC准确率', fontsize=12)
        ax1.set_title('贡献者人数预测准确率', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        for bar, acc in zip(bars1, noc_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 覆盖率
        bars2 = ax2.bar(methods, coverage_rates, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8)
        ax2.set_ylabel('覆盖率', fontsize=12)
        ax2.set_title('样本覆盖率', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        for bar, cov in zip(bars2, coverage_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{cov:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_coverage_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 样本级MSE分布箱线图
        plt.figure(figsize=(10, 6))
        
        mse_data = []
        labels = []
        
        for method in methods:
            metrics = self.evaluation_results[method]
            sample_mses = [m['mse'] for m in metrics['sample_wise_metrics'].values()]
            mse_data.append(sample_mses)
            labels.append(f"{method}\n(n={len(sample_mses)})")
        
        box_plot = plt.boxplot(mse_data, labels=labels, patch_artist=True)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('样本级MSE', fontsize=12)
        plt.title('各方法样本级MSE分布', fontsize=14, fontweight='bold')
        plt.yscale('log')  # 使用对数刻度
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_mse_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 散点图：真实比例 vs 预测比例
        fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            metrics = self.evaluation_results[method]
            
            all_true = []
            all_pred = []
            
            for sample_metrics in metrics['sample_wise_metrics'].values():
                all_true.extend(sample_metrics['true_ratios'])
                all_pred.extend(sample_metrics['predicted_ratios'])
            
            axes[i].scatter(all_true, all_pred, alpha=0.6, s=30)
            
            # 添加完美预测线
            min_val = min(min(all_true), min(all_pred))
            max_val = max(max(all_true), max(all_pred))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel('真实混合比例', fontsize=11)
            axes[i].set_ylabel('预测混合比例', fontsize=11)
            axes[i].set_title(f'{method}\nMSE={metrics["overall_mse"]:.4f}', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'true_vs_predicted_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图表已保存到: {self.output_dir}")
    
    def save_detailed_results(self):
        """保存详细评估结果"""
        logger.info("保存详细评估结果")
        
        # 保存总体评估结果
        output_file = os.path.join(self.output_dir, 'mse_evaluation_results.json')
        
        # 创建可序列化的结果
        serializable_results = {}
        
        for method_name, metrics in self.evaluation_results.items():
            serializable_metrics = metrics.copy()
            
            # 转换sample_wise_metrics为可序列化格式
            serializable_sample_metrics = {}
            for sample_file, sample_metrics in metrics['sample_wise_metrics'].items():
                serializable_sample_metrics[sample_file] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in sample_metrics.items()
                }
            
            serializable_metrics['sample_wise_metrics'] = serializable_sample_metrics
            serializable_results[method_name] = serializable_metrics
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式的汇总表
        summary_data = []
        for method_name, metrics in self.evaluation_results.items():
            summary_data.append({
                'Method': method_name,
                'Matched_Samples': metrics['matched_samples'],
                'Coverage_Rate': metrics['coverage_rate'],
                'Overall_MSE': metrics['overall_mse'],
                'Overall_RMSE': metrics['overall_rmse'],
                'Overall_MAE': metrics['overall_mae'],
                'NoC_Accuracy': metrics['noc_accuracy'],
                'Excellent_Rate': metrics['excellent_rate'],
                'Good_Rate': metrics['good_rate'],
                'Fair_Rate': metrics['fair_rate'],
                'Poor_Rate': metrics['poor_rate']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(self.output_dir, 'mse_summary.csv')
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        
        # 保存样本级详细结果
        sample_details = []
        for method_name, metrics in self.evaluation_results.items():
            for sample_file, sample_metrics in metrics['sample_wise_metrics'].items():
                true_data = self.true_ratios[sample_file]
                
                sample_details.append({
                    'Method': method_name,
                    'Sample_File': sample_file,
                    'True_NoC': true_data['noc'],
                    'Predicted_NoC': sample_metrics['predicted_noc'],
                    'NoC_Correct': sample_metrics['noc_correct'],
                    'True_Ratios': str(sample_metrics['true_ratios']),
                    'Predicted_Ratios': str(sample_metrics['predicted_ratios']),
                    'MSE': sample_metrics['mse'],
                    'MAE': sample_metrics['mae'],
                    'RMSE': sample_metrics['rmse'],
                    'Max_Error': sample_metrics['max_error']
                })
        
        details_df = pd.DataFrame(sample_details)
        details_csv = os.path.join(self.output_dir, 'sample_level_results.csv')
        details_df.to_csv(details_csv, index=False, encoding='utf-8-sig')
        
        logger.info(f"详细结果已保存:")
        logger.info(f"  - JSON格式: {output_file}")
        logger.info(f"  - 汇总CSV: {summary_csv}")
        logger.info(f"  - 详细CSV: {details_csv}")
    
    def run_full_evaluation(self, data_path: str, q2_dir: str = None, q3_dir: str = None, q4_dir: str = None):
        """运行完整的MSE评估流程"""
        logger.info("开始完整的MSE评估流程")
        
        # 1. 加载真实混合比例
        self.load_true_ratios_from_data(data_path)
        
        if not self.true_ratios:
            logger.error("没有加载到真实混合比例，无法进行评估")
            return
        
        # 2. 加载各方法的预测结果
        if q2_dir and os.path.exists(q2_dir):
            self.load_q2_results(q2_dir)
        else:
            logger.warning("Q2结果目录不存在或未指定")
        
        if q3_dir and os.path.exists(q3_dir):
            self.load_q3_results(q3_dir)
        else:
            logger.warning("Q3结果目录不存在或未指定")
        
        if q4_dir and os.path.exists(q4_dir):
            self.load_q4_results(q4_dir)
        else:
            logger.warning("Q4结果目录不存在或未指定")
        
        # 检查是否有任何方法的结果
        if not any([self.q2_results, self.q3_results, self.q4_results]):
            logger.error("没有加载到任何方法的预测结果")
            return
        
        # 3. 计算MSE等评估指标
        self.calculate_all_metrics()
        
        # 4. 生成对比报告
        self.generate_comparison_report()
        
        # 5. 绘制对比图表
        self.plot_comparison_charts()
        
        # 6. 保存详细结果
        self.save_detailed_results()
        
        logger.info("MSE评估流程完成！")

def create_demo_data():
    """创建演示数据"""
    print("\n创建演示数据用于测试...")
    
    # 创建模拟的真实数据
    demo_samples = [
        "A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa",
        "A02_RD14-0004-42_43-2;3-M3S30-0.075IP-Q4.0_002.5sec.fsa", 
        "A02_RD14-0005-44_45_46-1;2;1-M3S30-0.075IP-Q4.0_003.5sec.fsa"
    ]
    
    # 创建模拟数据文件
    demo_data = []
    markers = ['D3S1358', 'vWA', 'FGA']
    
    for sample in demo_samples:
        for marker in markers:
            demo_data.append({
                'Sample File': sample,
                'Marker': marker,
                'Allele 1': '14',
                'Size 1': 150.0,
                'Height 1': 1000.0
            })
    
    demo_df = pd.DataFrame(demo_data)
    demo_path = './demo_mixture_data.csv'
    demo_df.to_csv(demo_path, index=False, encoding='utf-8-sig')
    
    # 创建模拟的Q2结果
    os.makedirs('./demo_q2_results', exist_ok=True)
    for i, sample in enumerate(demo_samples):
        q2_result = {
            'sample_file': sample,
            'predicted_noc': 2 if i < 2 else 3,
            'noc_confidence': 0.9,
            'posterior_summary': {}
        }
        
        # 添加混合比例
        if i == 0:  # 第一个样本 1:4 比例
            q2_result['posterior_summary'] = {
                'Mx_1': {'mean': 0.22},
                'Mx_2': {'mean': 0.78}
            }
        elif i == 1:  # 第二个样本 2:3 比例
            q2_result['posterior_summary'] = {
                'Mx_1': {'mean': 0.42},
                'Mx_2': {'mean': 0.58}
            }
        else:  # 第三个样本 1:2:1 比例
            q2_result['posterior_summary'] = {
                'Mx_1': {'mean': 0.28},
                'Mx_2': {'mean': 0.48},
                'Mx_3': {'mean': 0.24}
            }
        
        result_file = f'./demo_q2_results/{sample}_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(q2_result, f, ensure_ascii=False, indent=2)
    
    print(f"演示数据已创建:")
    print(f"  - 数据文件: {demo_path}")
    print(f"  - Q2结果目录: ./demo_q2_results")
    print(f"  - 包含 {len(demo_samples)} 个混合样本")
    
    return demo_path, './demo_q2_results'

def main():
    """主函数"""
    print("欢迎使用Q2/Q3/Q4混合比例预测MSE计算器！")
    print("此工具将计算并对比各方法的预测精度")
    
    # 检查必要的数据文件
    data_files = [
        "附件1：不同人数的STR图谱数据.csv",
        "附件2：混合STR图谱数据.csv"
    ]
    
    data_path = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_path = file_path
            break
    
    # 检查结果目录
    result_dirs = {
        'Q2': './q2_mgm_rf_results',
        'Q3': './q3_enhanced_results', 
        'Q4': './q4_upg_denoising_results'
    }
    
    # 如果没有真实数据，提供演示选项
    if not data_path:
        print("\n没有找到必要的数据文件。")
        print("可用选项:")
        print("1. 创建演示数据进行测试")
        print("2. 退出程序")
        
        choice = input("请选择 (1/2): ").strip()
        if choice == "1":
            data_path, demo_q2_dir = create_demo_data()
            result_dirs['Q2'] = demo_q2_dir
        else:
            print("程序退出")
            return
    
    try:
        # 初始化MSE计算器
        calculator = MixtureRatioMSECalculator()
        
        # 运行完整评估
        calculator.run_full_evaluation(
            data_path=data_path,
            q2_dir=result_dirs['Q2'] if os.path.exists(result_dirs['Q2']) else None,
            q3_dir=result_dirs['Q3'] if os.path.exists(result_dirs['Q3']) else None,
            q4_dir=result_dirs['Q4'] if os.path.exists(result_dirs['Q4']) else None
        )
        
        print(f"\n{'='*80}")
        print("MSE评估完成！")
        print(f"结果已保存到: {calculator.output_dir}")
        print("生成的文件包括:")
        print("  - mse_evaluation_results.json (详细JSON结果)")
        print("  - mse_summary.csv (汇总表格)")  
        print("  - sample_level_results.csv (样本级详细结果)")
        print("  - *.png (对比图表)")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行")
    except Exception as e:
        print(f"\n程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()