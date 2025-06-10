# -*- coding: utf-8 -*-
"""
visualization_integration.py
法医DNA分析可视化集成模块

集成Q1、Q2、Q3、Q4所有方法的可视化结果
提供统一的图表生成和结果展示功能
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from pathlib import Path
import re
from collections import defaultdict

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualizationIntegrator:
    """可视化集成器 - 统一所有方法的结果展示"""
    
    def __init__(self):
        self.color_palette = {
            'Q1_NoC': '#FF6B6B',      # 红色 - NoC识别
            'Q2_Ratio': '#4ECDC4',    # 青色 - 混合比例
            'Q3_Genotype': '#45B7D1', # 蓝色 - 基因型推断
            'Q4_Denoise': '#96CEB4'   # 绿色 - 降噪
        }
        self.method_names = {
            'Q1': 'NoC识别 (RFECV+随机森林)',
            'Q2': '混合比例推断 (MGM-RF)',
            'Q3': '基因型推断 (增强MCMC)',
            'Q4': '降噪处理 (UPG-M)'
        }
        logger.info("可视化集成器初始化完成")
    
    def load_results_from_directory(self, results_dir: str, method_name: str) -> Dict:
        """从结果目录加载分析结果"""
        results = {}
        
        if not os.path.exists(results_dir):
            logger.warning(f"{method_name}结果目录不存在: {results_dir}")
            return results
        
        try:
            # 搜索JSON结果文件
            for file_path in Path(results_dir).rglob("*.json"):
                if any(keyword in file_path.name.lower() 
                      for keyword in ['result', 'summary', 'analysis']):
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 提取样本ID
                        sample_id = self._extract_sample_id(file_path.name, data)
                        if sample_id:
                            results[sample_id] = {
                                'data': data,
                                'file_path': str(file_path),
                                'method': method_name
                            }
                    
                    except Exception as e:
                        logger.warning(f"加载文件失败 {file_path}: {e}")
            
            logger.info(f"从{method_name}加载了{len(results)}个样本结果")
            
        except Exception as e:
            logger.error(f"加载{method_name}结果目录失败: {e}")
        
        return results
    
    def _extract_sample_id(self, filename: str, data: Dict) -> Optional[str]:
        """从文件名或数据中提取样本ID"""
        # 方法1：从数据中直接获取
        for key in ['sample_file', 'sample_id', 'Sample_File', 'Sample_ID']:
            if key in data and data[key]:
                return str(data[key])
        
        # 方法2：从文件名解析
        # 匹配模式如: sample_xxx_result.json, xxx_analysis.json
        patterns = [
            r'sample_(.+?)_result',
            r'(.+?)_analysis',
            r'(.+?)_result',
            r'sample_(.+?)\.json',
            r'([^_/\\]+)_[^_]*\.json'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def create_unified_comparison_plots(self, all_results: Dict, output_dir: str) -> int:
        """创建统一的比较图表"""
        os.makedirs(output_dir, exist_ok=True)
        plot_count = 0
        
        # 1. 方法覆盖率对比
        plot_count += self._plot_method_coverage(all_results, output_dir)
        
        # 2. NoC预测一致性分析
        plot_count += self._plot_noc_consistency(all_results, output_dir)
        
        # 3. 混合比例预测对比
        plot_count += self._plot_mixture_ratio_comparison(all_results, output_dir)
        
        # 4. 处理时间对比
        plot_count += self._plot_processing_time_comparison(all_results, output_dir)
        
        # 5. 整体性能雷达图
        plot_count += self._plot_performance_radar(all_results, output_dir)
        
        # 6. 样本复杂度分析
        plot_count += self._plot_sample_complexity_analysis(all_results, output_dir)
        
        return plot_count
    
    def _plot_method_coverage(self, all_results: Dict, output_dir: str) -> int:
        """绘制各方法的样本覆盖率"""
        try:
            methods = list(all_results.keys())
            coverage_data = []
            
            # 统计每个方法处理的样本数
            all_samples = set()
            for method_results in all_results.values():
                all_samples.update(method_results.keys())
            
            for method in methods:
                method_samples = set(all_results[method].keys())
                coverage_rate = len(method_samples) / len(all_samples) if all_samples else 0
                coverage_data.append({
                    'Method': self.method_names.get(method, method),
                    'Coverage_Rate': coverage_rate,
                    'Sample_Count': len(method_samples),
                    'Total_Samples': len(all_samples)
                })
            
            df_coverage = pd.DataFrame(coverage_data)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 覆盖率条形图
            bars = ax1.bar(df_coverage['Method'], df_coverage['Coverage_Rate'], 
                          color=[self.color_palette.get(f'{method}_NoC', '#666666') 
                                for method in methods])
            ax1.set_ylabel('样本覆盖率')
            ax1.set_title('各方法样本覆盖率对比')
            ax1.set_ylim(0, 1.1)
            
            # 添加数值标签
            for bar, rate in zip(bars, df_coverage['Coverage_Rate']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{rate:.1%}', ha='center', va='bottom')
            
            # 样本数量对比
            ax2.bar(df_coverage['Method'], df_coverage['Sample_Count'],
                   color=[self.color_palette.get(f'{method}_NoC', '#666666') 
                         for method in methods])
            ax2.set_ylabel('处理样本数')
            ax2.set_title('各方法处理样本数对比')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'method_coverage_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("方法覆盖率对比图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成方法覆盖率图失败: {e}")
            return 0
    
    def _plot_noc_consistency(self, all_results: Dict, output_dir: str) -> int:
        """绘制NoC预测一致性分析"""
        try:
            # 收集NoC预测结果
            noc_predictions = defaultdict(dict)
            
            for method, method_results in all_results.items():
                for sample_id, result in method_results.items():
                    data = result['data']
                    
                    # 提取NoC预测
                    noc = None
                    if method == 'Q1':
                        noc = data.get('贡献者人数') or data.get('predicted_noc')
                    else:
                        noc = data.get('predicted_noc') or data.get('noc_prediction')
                    
                    if noc is not None:
                        noc_predictions[sample_id][method] = int(noc)
            
            if not noc_predictions:
                logger.warning("没有找到NoC预测数据")
                return 0
            
            # 创建NoC一致性矩阵
            df_noc = pd.DataFrame(noc_predictions).T.fillna(0).astype(int)
            
            # 绘制NoC分布热力图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # NoC预测分布
            noc_counts = {}
            for method in df_noc.columns:
                if method in df_noc.columns:
                    noc_counts[self.method_names.get(method, method)] = df_noc[method].value_counts().sort_index()
            
            noc_df = pd.DataFrame(noc_counts).fillna(0)
            sns.heatmap(noc_df, annot=True, fmt='g', cmap='Blues', ax=ax1)
            ax1.set_title('各方法NoC预测分布热力图')
            ax1.set_xlabel('分析方法')
            ax1.set_ylabel('预测NoC值')
            
            # NoC一致性分析
            if len(df_noc.columns) >= 2:
                # 计算方法间的一致性
                methods = list(df_noc.columns)
                consistency_matrix = np.zeros((len(methods), len(methods)))
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i != j:
                            # 计算两个方法预测结果的一致性
                            common_samples = df_noc[(df_noc[method1] > 0) & (df_noc[method2] > 0)]
                            if len(common_samples) > 0:
                                consistency = (common_samples[method1] == common_samples[method2]).mean()
                                consistency_matrix[i, j] = consistency
                        else:
                            consistency_matrix[i, j] = 1.0
                
                method_labels = [self.method_names.get(m, m) for m in methods]
                sns.heatmap(consistency_matrix, 
                           xticklabels=method_labels,
                           yticklabels=method_labels,
                           annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
                ax2.set_title('方法间NoC预测一致性')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'noc_consistency_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("NoC一致性分析图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成NoC一致性图失败: {e}")
            return 0
    
    def _plot_mixture_ratio_comparison(self, all_results: Dict, output_dir: str) -> int:
        """绘制混合比例预测对比"""
        try:
            mixture_data = defaultdict(dict)
            
            # 收集混合比例数据
            for method, method_results in all_results.items():
                for sample_id, result in method_results.items():
                    data = result['data']
                    
                    # 提取混合比例
                    ratios = None
                    if method == 'Q2':
                        if 'posterior_summary' in data:
                            summary = data['posterior_summary']
                            ratios = []
                            i = 1
                            while f'Mx_{i}' in summary:
                                ratios.append(summary[f'Mx_{i}'].get('mean', 0))
                                i += 1
                    elif method == 'Q3':
                        # Q3中可能也有混合比例信息
                        if 'mixture_ratios' in data:
                            ratios = data['mixture_ratios']
                    
                    if ratios:
                        mixture_data[sample_id][method] = ratios
            
            if not mixture_data:
                logger.warning("没有找到混合比例数据")
                return 0
            
            # 绘制混合比例对比
            n_samples = min(6, len(mixture_data))  # 限制显示样本数
            if n_samples == 0:
                return 0
            
            sample_ids = list(mixture_data.keys())[:n_samples]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, sample_id in enumerate(sample_ids):
                ax = axes[idx]
                sample_ratios = mixture_data[sample_id]
                
                methods = list(sample_ratios.keys())
                x_pos = np.arange(len(methods))
                
                # 获取最大组分数
                max_components = max(len(ratios) for ratios in sample_ratios.values())
                
                # 绘制每个组分
                bottom = np.zeros(len(methods))
                for comp_idx in range(max_components):
                    values = []
                    for method in methods:
                        ratios = sample_ratios[method]
                        value = ratios[comp_idx] if comp_idx < len(ratios) else 0
                        values.append(value)
                    
                    ax.bar(x_pos, values, bottom=bottom, 
                          label=f'组分{comp_idx+1}', alpha=0.7)
                    bottom += values
                
                ax.set_title(f'样本 {sample_id}\n混合比例对比')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([self.method_names.get(m, m) for m in methods], 
                                  rotation=45)
                ax.set_ylabel('混合比例')
                ax.legend()
                ax.set_ylim(0, 1.1)
            
            # 隐藏多余的子图
            for idx in range(n_samples, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mixture_ratio_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("混合比例对比图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成混合比例对比图失败: {e}")
            return 0
    
    def _plot_processing_time_comparison(self, all_results: Dict, output_dir: str) -> int:
        """绘制处理时间对比"""
        try:
            time_data = []
            
            for method, method_results in all_results.items():
                times = []
                for sample_id, result in method_results.items():
                    data = result['data']
                    
                    # 提取计算时间
                    comp_time = data.get('computation_time')
                    if comp_time is not None:
                        times.append(float(comp_time))
                
                if times:
                    time_data.append({
                        'Method': self.method_names.get(method, method),
                        'Mean_Time': np.mean(times),
                        'Std_Time': np.std(times),
                        'Median_Time': np.median(times),
                        'Max_Time': np.max(times),
                        'Min_Time': np.min(times),
                        'Sample_Count': len(times)
                    })
            
            if not time_data:
                logger.warning("没有找到处理时间数据")
                return 0
            
            df_time = pd.DataFrame(time_data)
            
            # 创建处理时间对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 平均处理时间
            bars = ax1.bar(df_time['Method'], df_time['Mean_Time'], 
                          yerr=df_time['Std_Time'], capsize=5,
                          color=[self.color_palette.get(f'{method}_NoC', '#666666') 
                                for method in ['Q1', 'Q2', 'Q3', 'Q4']])
            ax1.set_ylabel('处理时间 (秒)')
            ax1.set_title('各方法平均处理时间对比')
            
            # 添加数值标签
            for bar, time_val in zip(bars, df_time['Mean_Time']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.1f}s', ha='center', va='bottom')
            
            # 处理时间分布箱线图
            time_distributions = []
            method_labels = []
            
            for method, method_results in all_results.items():
                times = []
                for result in method_results.values():
                    comp_time = result['data'].get('computation_time')
                    if comp_time is not None:
                        times.append(float(comp_time))
                
                if times:
                    time_distributions.append(times)
                    method_labels.append(self.method_names.get(method, method))
            
            if time_distributions:
                ax2.boxplot(time_distributions, labels=method_labels)
                ax2.set_ylabel('处理时间 (秒)')
                ax2.set_title('处理时间分布箱线图')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("处理时间对比图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成处理时间对比图失败: {e}")
            return 0
    
    def _plot_performance_radar(self, all_results: Dict, output_dir: str) -> int:
        """绘制整体性能雷达图"""
        try:
            # 计算各方法的性能指标
            performance_metrics = {}
            
            for method, method_results in all_results.items():
                if not method_results:
                    continue
                
                metrics = {
                    'success_rate': 0,
                    'average_confidence': 0,
                    'processing_speed': 0,
                    'result_quality': 0,
                    'convergence_rate': 0
                }
                
                successful_samples = 0
                confidences = []
                times = []
                convergences = []
                
                for result in method_results.values():
                    data = result['data']
                    
                    # 成功率
                    if data.get('success', True):
                        successful_samples += 1
                    
                    # 置信度
                    confidence = data.get('noc_confidence') or data.get('confidence')
                    if confidence is not None:
                        confidences.append(float(confidence))
                    
                    # 处理时间
                    comp_time = data.get('computation_time')
                    if comp_time is not None:
                        times.append(float(comp_time))
                    
                    # 收敛性
                    converged = data.get('converged')
                    if converged is not None:
                        convergences.append(bool(converged))
                
                # 计算标准化指标 (0-1范围)
                metrics['success_rate'] = successful_samples / len(method_results) if method_results else 0
                metrics['average_confidence'] = np.mean(confidences) if confidences else 0
                metrics['processing_speed'] = 1 / (1 + np.mean(times)) if times else 0  # 时间越短，速度分数越高
                metrics['convergence_rate'] = np.mean(convergences) if convergences else 0
                
                # 结果质量（基于多个因素的综合评分）
                quality_score = (metrics['success_rate'] + metrics['average_confidence'] + 
                               metrics['convergence_rate']) / 3
                metrics['result_quality'] = quality_score
                
                performance_metrics[method] = metrics
            
            if not performance_metrics:
                logger.warning("没有性能指标数据")
                return 0
            
            # 创建雷达图
            categories = ['成功率', '平均置信度', '处理速度', '结果质量', '收敛率']
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 设置角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for idx, (method, metrics) in enumerate(performance_metrics.items()):
                values = [
                    metrics['success_rate'],
                    metrics['average_confidence'],
                    metrics['processing_speed'],
                    metrics['result_quality'],
                    metrics['convergence_rate']
                ]
                values += values[:1]  # 闭合图形
                
                color = colors[idx % len(colors)]
                ax.plot(angles, values, 'o-', linewidth=2, label=self.method_names.get(method, method), color=color)
                ax.fill(angles, values, alpha=0.25, color=color)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('各方法整体性能雷达图', size=16, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_radar_chart.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("性能雷达图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成性能雷达图失败: {e}")
            return 0
    
    def _plot_sample_complexity_analysis(self, all_results: Dict, output_dir: str) -> int:
        """绘制样本复杂度分析"""
        try:
            complexity_data = []
            
            # 从Q1结果中提取复杂度相关特征
            q1_results = all_results.get('Q1', {})
            
            for sample_id, result in q1_results.items():
                data = result['data']
                
                # 提取复杂度指标
                complexity_indicators = {
                    'sample_id': sample_id,
                    'noc': data.get('贡献者人数', 0),
                    'mac_profile': 0,
                    'total_alleles': 0,
                    'avg_alleles_per_locus': 0,
                    'processing_difficulty': 'Unknown'
                }
                
                # 从特征中提取
                features = data
                complexity_indicators['mac_profile'] = features.get('mac_profile', 0)
                complexity_indicators['total_alleles'] = features.get('total_distinct_alleles', 0)
                complexity_indicators['avg_alleles_per_locus'] = features.get('avg_alleles_per_locus', 0)
                
                # 计算处理难度
                difficulty_score = (complexity_indicators['mac_profile'] * 0.4 + 
                                  complexity_indicators['total_alleles'] / 50 * 0.3 +
                                  complexity_indicators['avg_alleles_per_locus'] / 5 * 0.3)
                
                if difficulty_score < 2:
                    complexity_indicators['processing_difficulty'] = '简单'
                elif difficulty_score < 4:
                    complexity_indicators['processing_difficulty'] = '中等'
                else:
                    complexity_indicators['processing_difficulty'] = '复杂'
                
                complexity_data.append(complexity_indicators)
            
            if not complexity_data:
                logger.warning("没有复杂度分析数据")
                return 0
            
            df_complexity = pd.DataFrame(complexity_data)
            
            # 创建复杂度分析图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. NoC与复杂度的关系
            noc_complexity = df_complexity.groupby('noc').agg({
                'mac_profile': 'mean',
                'total_alleles': 'mean',
                'avg_alleles_per_locus': 'mean'
            }).reset_index()
            
            ax1.scatter(noc_complexity['noc'], noc_complexity['mac_profile'], 
                       s=100, alpha=0.7, color=self.color_palette['Q1_NoC'])
            ax1.set_xlabel('贡献者人数 (NoC)')
            ax1.set_ylabel('平均最大等位基因数')
            ax1.set_title('NoC与图谱复杂度关系')
            ax1.grid(True, alpha=0.3)
            
            # 2. 处理难度分布
            difficulty_counts = df_complexity['processing_difficulty'].value_counts()
            colors = ['#98FB98', '#FFD700', '#FF6347']  # 绿、黄、红
            ax2.pie(difficulty_counts.values, labels=difficulty_counts.index, 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('样本处理难度分布')
            
            # 3. 复杂度散点图
            scatter = ax3.scatter(df_complexity['total_alleles'], 
                                df_complexity['avg_alleles_per_locus'],
                                c=df_complexity['noc'], 
                                cmap='viridis', s=60, alpha=0.7)
            ax3.set_xlabel('总等位基因数')
            ax3.set_ylabel('平均每位点等位基因数')
            ax3.set_title('样本复杂度二维分布')
            plt.colorbar(scatter, ax=ax3, label='NoC')
            
            # 4. 不同NoC的复杂度分布
            noc_values = sorted(df_complexity['noc'].unique())
            for noc in noc_values:
                subset = df_complexity[df_complexity['noc'] == noc]
                ax4.hist(subset['mac_profile'], alpha=0.6, 
                        label=f'{noc}人混合', bins=10)
            
            ax4.set_xlabel('最大等位基因数')
            ax4.set_ylabel('样本数量')
            ax4.set_title('不同NoC的复杂度分布')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sample_complexity_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("样本复杂度分析图已生成")
            return 1
            
        except Exception as e:
            logger.error(f"生成样本复杂度分析图失败: {e}")
            return 0
    
    def create_summary_report(self, all_results: Dict, output_dir: str) -> str:
        """创建综合分析报告"""
        try:
            report_lines = []
            report_lines.append("="*80)
            report_lines.append("          法医DNA分析综合结果报告")
            report_lines.append("="*80)
            
            # 统计各方法的样本数
            report_lines.append("\n📊 方法覆盖率统计:")
            all_samples = set()
            for method_results in all_results.values():
                all_samples.update(method_results.keys())
            
            for method, method_results in all_results.items():
                sample_count = len(method_results)
                coverage_rate = sample_count / len(all_samples) if all_samples else 0
                method_name = self.method_names.get(method, method)
                report_lines.append(f"   {method_name}: {sample_count}个样本 ({coverage_rate:.1%}覆盖率)")
            
            # NoC预测一致性分析
            report_lines.append(f"\n🎯 NoC预测一致性分析:")
            noc_predictions = defaultdict(dict)
            
            for method, method_results in all_results.items():
                for sample_id, result in method_results.items():
                    data = result['data']
                    noc = None
                    if method == 'Q1':
                        noc = data.get('贡献者人数') or data.get('predicted_noc')
                    else:
                        noc = data.get('predicted_noc') or data.get('noc_prediction')
                    
                    if noc is not None:
                        noc_predictions[sample_id][method] = int(noc)
            
            if noc_predictions:
                # 计算各方法的NoC分布
                for method in all_results.keys():
                    noc_values = [pred.get(method) for pred in noc_predictions.values() if method in pred]
                    if noc_values:
                        noc_dist = pd.Series(noc_values).value_counts().sort_index()
                        method_name = self.method_names.get(method, method)
                        report_lines.append(f"   {method_name} NoC分布: {dict(noc_dist)}")
            
            # 处理时间统计
            report_lines.append(f"\n⏱️  处理时间统计:")
            for method, method_results in all_results.items():
                times = []
                for result in method_results.values():
                    comp_time = result['data'].get('computation_time')
                    if comp_time is not None:
                        times.append(float(comp_time))
                
                if times:
                    avg_time = np.mean(times)
                    max_time = np.max(times)
                    min_time = np.min(times)
                    method_name = self.method_names.get(method, method)
                    report_lines.append(f"   {method_name}: 平均{avg_time:.1f}s (范围: {min_time:.1f}s - {max_time:.1f}s)")
            
            # 成功率统计
            report_lines.append(f"\n✅ 分析成功率:")
            for method, method_results in all_results.items():
                successful = 0
                total = len(method_results)
                
                for result in method_results.values():
                    data = result['data']
                    if data.get('success', True):  # 默认为成功
                        successful += 1
                
                success_rate = successful / total if total > 0 else 0
                method_name = self.method_names.get(method, method)
                report_lines.append(f"   {method_name}: {successful}/{total} ({success_rate:.1%})")
            
            # 质量评估
            report_lines.append(f"\n🏆 质量评估:")
            for method, method_results in all_results.items():
                confidences = []
                convergences = []
                
                for result in method_results.values():
                    data = result['data']
                    
                    # 收集置信度
                    confidence = data.get('noc_confidence') or data.get('confidence')
                    if confidence is not None:
                        confidences.append(float(confidence))
                    
                    # 收集收敛性
                    converged = data.get('converged')
                    if converged is not None:
                        convergences.append(bool(converged))
                
                method_name = self.method_names.get(method, method)
                
                if confidences:
                    avg_confidence = np.mean(confidences)
                    report_lines.append(f"   {method_name} 平均置信度: {avg_confidence:.3f}")
                
                if convergences:
                    convergence_rate = np.mean(convergences)
                    report_lines.append(f"   {method_name} 收敛率: {convergence_rate:.1%}")
            
            # 关键发现
            report_lines.append(f"\n💡 关键发现:")
            
            # 找出处理最快的方法
            fastest_method = None
            fastest_time = float('inf')
            for method, method_results in all_results.items():
                times = [float(result['data'].get('computation_time', float('inf'))) 
                        for result in method_results.values() 
                        if result['data'].get('computation_time') is not None]
                if times:
                    avg_time = np.mean(times)
                    if avg_time < fastest_time:
                        fastest_time = avg_time
                        fastest_method = method
            
            if fastest_method:
                report_lines.append(f"   • 处理速度最快: {self.method_names.get(fastest_method, fastest_method)} (平均{fastest_time:.1f}秒)")
            
            # 找出覆盖率最高的方法
            highest_coverage_method = None
            highest_coverage = 0
            for method, method_results in all_results.items():
                coverage = len(method_results) / len(all_samples) if all_samples else 0
                if coverage > highest_coverage:
                    highest_coverage = coverage
                    highest_coverage_method = method
            
            if highest_coverage_method:
                report_lines.append(f"   • 样本覆盖率最高: {self.method_names.get(highest_coverage_method, highest_coverage_method)} ({highest_coverage:.1%})")
            
            # 推荐的分析流程
            report_lines.append(f"\n🚀 推荐分析流程:")
            report_lines.append(f"   1. 使用Q1方法进行NoC识别和特征提取")
            report_lines.append(f"   2. 基于Q1结果，使用Q2方法推断混合比例")
            report_lines.append(f"   3. 结合Q1和Q2结果，使用Q3方法进行基因型推断")
            report_lines.append(f"   4. 对于低质量样本，使用Q4方法进行降噪处理")
            
            report_lines.append(f"\n📈 改进建议:")
            
            # 基于结果给出改进建议
            for method, method_results in all_results.items():
                method_name = self.method_names.get(method, method)
                
                # 计算该方法的问题
                issues = []
                
                # 检查处理时间
                times = [float(result['data'].get('computation_time', 0)) 
                        for result in method_results.values() 
                        if result['data'].get('computation_time') is not None]
                if times and np.mean(times) > 60:  # 超过1分钟
                    issues.append("处理时间较长，建议优化算法或参数")
                
                # 检查成功率
                successful = sum(1 for result in method_results.values() 
                               if result['data'].get('success', True))
                success_rate = successful / len(method_results) if method_results else 0
                if success_rate < 0.8:
                    issues.append("成功率偏低，建议检查数据质量或调整参数")
                
                # 检查收敛性
                convergences = [bool(result['data'].get('converged', True)) 
                              for result in method_results.values() 
                              if 'converged' in result['data']]
                if convergences and np.mean(convergences) < 0.7:
                    issues.append("MCMC收敛率偏低，建议增加迭代次数或调整提议分布")
                
                if issues:
                    report_lines.append(f"   • {method_name}:")
                    for issue in issues:
                        report_lines.append(f"     - {issue}")
            
            report_lines.append(f"\n" + "="*80)
            report_lines.append(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"="*80)
            
            # 保存报告
            report_content = '\n'.join(report_lines)
            report_path = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"综合分析报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return ""

def integrate_all_methods(q1_dir: str = None, q2_dir: str = None, 
                         q3_dir: str = None, q4_dir: str = None,
                         output_dir: str = "./integrated_results") -> Dict:
    """
    集成所有方法的可视化结果
    
    Args:
        q1_dir: Q1结果目录路径
        q2_dir: Q2结果目录路径  
        q3_dir: Q3结果目录路径
        q4_dir: Q4结果目录路径
        output_dir: 输出目录路径
        
    Returns:
        Dict: 包含处理统计信息的字典
    """
    integrator = VisualizationIntegrator()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 开始加载各方法的分析结果...")
    
    # 加载各方法的结果
    all_results = {}
    
    # 定义方法目录映射
    method_dirs = {
        'Q1': q1_dir,
        'Q2': q2_dir, 
        'Q3': q3_dir,
        'Q4': q4_dir
    }
    
    # 加载结果
    for method, dir_path in method_dirs.items():
        if dir_path and os.path.exists(dir_path):
            print(f"   正在加载{method}结果...")
            method_results = integrator.load_results_from_directory(dir_path, method)
            if method_results:
                all_results[method] = method_results
                print(f"   ✅ {method}: 加载了{len(method_results)}个样本结果")
            else:
                print(f"   ⚠️  {method}: 未找到有效结果")
        else:
            print(f"   ❌ {method}: 目录不存在或未指定 - {dir_path}")
    
    if not all_results:
        print("❌ 没有找到任何有效的分析结果")
        return {
            'total_samples': 0,
            'total_plots': 0,
            'methods_processed': 0,
            'success': False
        }
    
    print(f"\n📊 开始生成集成可视化图表...")
    
    # 生成统一比较图表
    total_plots = integrator.create_unified_comparison_plots(all_results, output_dir)
    
    # 生成综合报告
    print("📝 生成综合分析报告...")
    report_path = integrator.create_summary_report(all_results, output_dir)
    
    # 统计信息
    total_samples = len(set().union(*[results.keys() for results in all_results.values()]))
    
    result_summary = {
        'total_samples': total_samples,
        'total_plots': total_plots,
        'methods_processed': len(all_results),
        'methods_found': list(all_results.keys()),
        'output_directory': output_dir,
        'report_path': report_path,
        'success': True
    }
    
    print(f"\n✅ 可视化集成完成!")
    print(f"   📈 生成图表: {total_plots}个")
    print(f"   📊 处理样本: {total_samples}个")
    print(f"   🔬 集成方法: {', '.join(all_results.keys())}")
    print(f"   📁 输出目录: {output_dir}")
    if report_path:
        print(f"   📄 综合报告: {report_path}")
    
    return result_summary

def create_method_comparison_dashboard(all_results: Dict, output_dir: str) -> str:
    """创建方法对比仪表板"""
    try:
        # 这里可以扩展为交互式仪表板
        # 目前创建静态的HTML报告
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>法医DNA分析方法对比仪表板</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .method {{ background-color: #f9f9f9; margin: 10px 0; padding: 10px; }}
                .metric {{ display: inline-block; margin: 5px 10px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>法医DNA分析方法对比仪表板</h1>
                <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>方法概览</h2>
        """
        
        for method, results in all_results.items():
            method_name = VisualizationIntegrator().method_names.get(method, method)
            sample_count = len(results)
            
            html_content += f"""
                <div class="method">
                    <h3>{method_name}</h3>
                    <div class="metric">样本数: <strong>{sample_count}</strong></div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>生成的图表</h2>
                <ul>
                    <li><a href="method_coverage_comparison.png">方法覆盖率对比</a></li>
                    <li><a href="noc_consistency_analysis.png">NoC预测一致性分析</a></li>
                    <li><a href="mixture_ratio_comparison.png">混合比例预测对比</a></li>
                    <li><a href="processing_time_comparison.png">处理时间对比</a></li>
                    <li><a href="performance_radar_chart.png">整体性能雷达图</a></li>
                    <li><a href="sample_complexity_analysis.png">样本复杂度分析</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>详细报告</h2>
                <p><a href="comprehensive_analysis_report.txt">查看完整文本报告</a></p>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = os.path.join(output_dir, 'dashboard.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"仪表板已生成: {dashboard_path}")
        return dashboard_path
        
    except Exception as e:
        logger.error(f"生成仪表板失败: {e}")
        return ""

# 主函数示例
if __name__ == "__main__":
    # 示例使用
    results = integrate_all_methods(
        q1_dir="./q1_results",
        q2_dir="./q2_mgm_rf_results",
        q3_dir="./q3_enhanced_results", 
        q4_dir="./q4_upg_denoising_results",
        output_dir="./integrated_visualization_results"
    )
    
    print(f"集成完成: {results}")