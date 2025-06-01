# -*- coding: utf-8 -*-
"""
STR数据综合分析系统
整合问题1(NoC识别)、问题2(MCMC混合比例)和问题4(峰分类)的结果分析

作者: 数学建模团队
日期: 2025-06-01
版本: V1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os

# 配置设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = './analysis_results'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

for directory in [OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

print("=== STR数据综合分析系统 ===")
print("整合分析问题1(NoC识别)、问题2(MCMC混合比例)和相关数据")

# ==================== 第一部分：数据加载 ====================
print("\n--- 第一部分：数据加载 ---")

class STRDataLoader:
    """STR数据加载器"""
    
    def __init__(self):
        self.p1_data = None  # 问题1特征数据
        self.p2_data = None  # 问题2 MCMC结果
        self.combined_data = {}
    
    def load_problem1_data(self, filepath='prob1_features_enhanced.csv'):
        """加载问题1的特征数据"""
        try:
            self.p1_data = pd.read_csv(filepath, encoding='utf-8')
            print(f"✓ 成功加载问题1数据: {filepath}")
            print(f"  - 样本数: {len(self.p1_data)}")
            print(f"  - 特征数: {len(self.p1_data.columns) - 2}")  # 排除Sample File和NoC_True
            
            # 检查数据质量
            missing_ratio = self.p1_data.isnull().sum().sum() / (len(self.p1_data) * len(self.p1_data.columns))
            print(f"  - 缺失值比例: {missing_ratio:.2%}")
            
            return True
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return False
        except Exception as e:
            print(f"✗ 加载问题1数据时出错: {e}")
            return False
    
    def load_problem2_data(self, filepath='problem2_mcmc_results.json'):
        """加载问题2的MCMC结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.p2_data = json.load(f)
            print(f"✓ 成功加载问题2数据: {filepath}")
            
            # 提取关键信息
            if 'posterior_summary' in self.p2_data:
                mx1_mean = self.p2_data['posterior_summary']['Mx_1']['mean']
                mx2_mean = self.p2_data['posterior_summary']['Mx_2']['mean']
                print(f"  - Mx_1平均值: {mx1_mean:.4f}")
                print(f"  - Mx_2平均值: {mx2_mean:.4f}")
                print(f"  - 收敛状态: {self.p2_data.get('convergence_diagnostics', {}).get('convergence_status', 'Unknown')}")
            
            return True
        except FileNotFoundError:
            print(f"✗ 文件未找到: {filepath}")
            return False
        except Exception as e:
            print(f"✗ 加载问题2数据时出错: {e}")
            return False
    
    def get_noc_distribution(self):
        """获取NoC分布"""
        if self.p1_data is not None:
            return self.p1_data['NoC_True'].value_counts().sort_index()
        return None
    
    def get_performance_metrics(self):
        """获取模型性能指标"""
        if self.p1_data is not None and 'baseline_pred' in self.p1_data.columns:
            accuracy = accuracy_score(self.p1_data['NoC_True'], self.p1_data['baseline_pred'])
            return {
                'accuracy': accuracy,
                'total_samples': len(self.p1_data),
                'correct_predictions': (self.p1_data['NoC_True'] == self.p1_data['baseline_pred']).sum()
            }
        return None

# ==================== 第二部分：问题1数据分析 ====================
print("\n--- 第二部分：问题1数据分析 (NoC识别) ---")

class Problem1Analyzer:
    """问题1分析器"""
    
    def __init__(self, data):
        self.data = data
        self.feature_cols = [col for col in data.columns 
                           if col not in ['Sample File', 'NoC_True', 'baseline_pred', '预测正确', 'corrected_pred']]
    
    def analyze_noc_distribution(self):
        """分析NoC分布"""
        print("\nNoC分布分析:")
        noc_dist = self.data['NoC_True'].value_counts().sort_index()
        for noc, count in noc_dist.items():
            percentage = count / len(self.data) * 100
            print(f"  {noc}人: {count}个样本 ({percentage:.1f}%)")
        
        # 可视化NoC分布
        plt.figure(figsize=(10, 6))
        noc_dist.plot(kind='bar', color='steelblue', alpha=0.8)
        plt.title('STR样本中贡献者人数(NoC)分布')
        plt.xlabel('贡献者人数')
        plt.ylabel('样本数量')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(noc_dist.values):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'noc_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return noc_dist
    
    def analyze_model_performance(self):
        """分析模型性能"""
        if 'baseline_pred' not in self.data.columns:
            print("警告: 没有找到预测结果列")
            return None
        
        print("\n模型性能分析:")
        
        # 整体准确率
        overall_accuracy = accuracy_score(self.data['NoC_True'], self.data['baseline_pred'])
        print(f"整体准确率: {overall_accuracy:.4f}")
        
        # 各NoC类别准确率
        print("\n各NoC类别准确率:")
        for noc in sorted(self.data['NoC_True'].unique()):
            mask = self.data['NoC_True'] == noc
            if mask.sum() > 0:
                noc_accuracy = (self.data.loc[mask, 'NoC_True'] == 
                              self.data.loc[mask, 'baseline_pred']).mean()
                print(f"  {noc}人: {noc_accuracy:.4f} ({mask.sum()}个样本)")
        
        # 混淆矩阵
        cm = confusion_matrix(self.data['NoC_True'], self.data['baseline_pred'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(self.data['NoC_True'].unique()),
                   yticklabels=sorted(self.data['NoC_True'].unique()))
        plt.title('NoC预测混淆矩阵')
        plt.xlabel('预测NoC')
        plt.ylabel('真实NoC')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 分类报告
        class_names = [f"{i}人" for i in sorted(self.data['NoC_True'].unique())]
        report = classification_report(self.data['NoC_True'], self.data['baseline_pred'],
                                     target_names=class_names, output_dict=True)
        
        return {
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性（基于相关性）"""
        print("\n特征重要性分析:")
        
        # 计算与NoC的相关性
        correlations = []
        for feature in self.feature_cols:
            if self.data[feature].dtype in ['int64', 'float64']:
                try:
                    corr = abs(self.data[feature].corr(self.data['NoC_True']))
                    if not np.isnan(corr):
                        correlations.append((feature, corr))
                except:
                    continue
        
        # 排序并显示
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("与NoC相关性最高的前10个特征:")
        for i, (feature, corr) in enumerate(correlations[:10]):
            print(f"  {i+1:2d}. {feature}: {corr:.4f}")
        
        # 可视化前15个重要特征
        if len(correlations) >= 15:
            top_features = correlations[:15]
            features, corr_values = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            plt.barh(y_pos, corr_values, color='lightcoral', alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('与NoC的绝对相关系数')
            plt.title('特征重要性排名 (前15位)')
            plt.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(corr_values):
                plt.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return correlations
    
    def analyze_feature_distributions(self):
        """分析不同NoC下的特征分布"""
        print("\n特征分布分析:")
        
        # 选择几个重要特征进行分布分析
        important_features = ['mac_profile', 'avg_alleles_per_locus', 'avg_peak_height', 
                            'std_peak_height', 'avg_phr', 'num_loci_with_phr']
        
        available_features = [f for f in important_features if f in self.data.columns]
        
        if len(available_features) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(available_features[:4]):
                ax = axes[i]
                
                # 为每个NoC绘制分布
                noc_values = sorted(self.data['NoC_True'].unique())
                for noc in noc_values:
                    data_subset = self.data[self.data['NoC_True'] == noc][feature]
                    if len(data_subset) > 0:
                        ax.hist(data_subset, alpha=0.6, label=f'{noc}人', bins=15)
                
                ax.set_xlabel(feature)
                ax.set_ylabel('频次')
                ax.set_title(f'{feature}在不同NoC下的分布')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()

# ==================== 第三部分：问题2数据分析 ====================
print("\n--- 第三部分：问题2数据分析 (MCMC混合比例) ---")

class Problem2Analyzer:
    """问题2分析器"""
    
    def __init__(self, data):
        self.data = data
    
    def analyze_posterior_summary(self):
        """分析后验分布摘要"""
        if 'posterior_summary' not in self.data:
            print("警告: 没有找到posterior_summary数据")
            return None
        
        posterior = self.data['posterior_summary']
        print("\n后验分布分析:")
        
        # 混合比例分析
        mx1_stats = posterior['Mx_1']
        mx2_stats = posterior['Mx_2']
        
        print(f"Mx_1 (主要贡献者):")
        print(f"  均值: {mx1_stats['mean']:.4f}")
        print(f"  标准差: {mx1_stats['std']:.4f}")
        print(f"  95%置信区间: [{mx1_stats['credible_interval_95'][0]:.4f}, {mx1_stats['credible_interval_95'][1]:.4f}]")
        
        print(f"\nMx_2 (次要贡献者):")
        print(f"  均值: {mx2_stats['mean']:.4f}")
        print(f"  标准差: {mx2_stats['std']:.4f}")
        print(f"  95%置信区间: [{mx2_stats['credible_interval_95'][0]:.4f}, {mx2_stats['credible_interval_95'][1]:.4f}]")
        
        # 模型质量
        model_quality = posterior['model_quality']
        print(f"\n模型质量:")
        print(f"  接受率: {model_quality['acceptance_rate']:.4f}")
        print(f"  有效样本数: {model_quality['n_effective_samples']}")
        print(f"  收敛状态: {model_quality['converged']}")
        
        return {
            'mx1_mean': mx1_stats['mean'],
            'mx2_mean': mx2_stats['mean'],
            'mx1_ci': mx1_stats['credible_interval_95'],
            'mx2_ci': mx2_stats['credible_interval_95'],
            'acceptance_rate': model_quality['acceptance_rate'],
            'converged': model_quality['converged']
        }
    
    def analyze_convergence(self):
        """分析收敛诊断"""
        if 'convergence_diagnostics' not in self.data:
            print("警告: 没有找到convergence_diagnostics数据")
            return None
        
        conv = self.data['convergence_diagnostics']
        print("\n收敛诊断:")
        
        ess = conv['effective_sample_size']
        print(f"有效样本大小:")
        print(f"  Mx_1: {ess[0]:.2f}")
        print(f"  Mx_2: {ess[1]:.2f}")
        print(f"  最小ESS: {conv['min_ess']:.2f}")
        print(f"收敛状态: {conv['convergence_status']}")
        
        return conv
    
    def plot_mixture_ratios_trace(self):
        """绘制混合比例的迹图"""
        if 'sample_mixture_ratios' not in self.data:
            print("警告: 没有找到样本数据")
            return
        
        samples = np.array(self.data['sample_mixture_ratios'])
        mx1_samples = samples[:, 0]
        mx2_samples = samples[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mx_1迹图
        axes[0, 0].plot(mx1_samples, alpha=0.7, color='blue')
        axes[0, 0].set_title('Mx_1 迹图')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('Mx_1')
        axes[0, 0].grid(alpha=0.3)
        
        # Mx_2迹图
        axes[0, 1].plot(mx2_samples, alpha=0.7, color='red')
        axes[0, 1].set_title('Mx_2 迹图')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('Mx_2')
        axes[0, 1].grid(alpha=0.3)
        
        # Mx_1分布
        axes[1, 0].hist(mx1_samples, bins=50, alpha=0.7, color='blue', density=True)
        axes[1, 0].axvline(np.mean(mx1_samples), color='darkblue', linestyle='--', 
                          label=f'均值: {np.mean(mx1_samples):.3f}')
        axes[1, 0].set_title('Mx_1 后验分布')
        axes[1, 0].set_xlabel('Mx_1')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Mx_2分布
        axes[1, 1].hist(mx2_samples, bins=50, alpha=0.7, color='red', density=True)
        axes[1, 1].axvline(np.mean(mx2_samples), color='darkred', linestyle='--',
                          label=f'均值: {np.mean(mx2_samples):.3f}')
        axes[1, 1].set_title('Mx_2 后验分布')
        axes[1, 1].set_xlabel('Mx_2')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'mcmc_trace_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(mx1_samples, mx2_samples, alpha=0.5, s=1)
        plt.xlabel('Mx_1')
        plt.ylabel('Mx_2')
        plt.title('混合比例联合分布')
        
        # 添加约束线 Mx_1 + Mx_2 = 1
        x_line = np.linspace(0, 1, 100)
        y_line = 1 - x_line
        plt.plot(x_line, y_line, 'r--', label='Mx_1 + Mx_2 = 1')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'mixture_ratios_joint.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ==================== 第四部分：综合分析 ====================
print("\n--- 第四部分：综合分析 ---")

class ComprehensiveAnalyzer:
    """综合分析器"""
    
    def __init__(self, p1_data, p2_data):
        self.p1_data = p1_data
        self.p2_data = p2_data
    
    def analyze_model_consistency(self):
        """分析模型一致性"""
        print("\n模型一致性分析:")
        
        if self.p1_data is None or self.p2_data is None:
            print("数据不完整，无法进行一致性分析")
            return
        
        # 分析2人混合物的预测结果
        two_person_samples = self.p1_data[self.p1_data['NoC_True'] == 2]
        if len(two_person_samples) > 0:
            accuracy_2p = (two_person_samples['NoC_True'] == 
                          two_person_samples['baseline_pred']).mean()
            print(f"问题1中2人混合物预测准确率: {accuracy_2p:.4f}")
        
        # 问题2的混合比例
        if 'posterior_summary' in self.p2_data:
            mx1_mean = self.p2_data['posterior_summary']['Mx_1']['mean']
            mx2_mean = self.p2_data['posterior_summary']['Mx_2']['mean']
            
            print(f"问题2中混合比例估计:")
            print(f"  主要贡献者 (Mx_1): {mx1_mean:.4f}")
            print(f"  次要贡献者 (Mx_2): {mx2_mean:.4f}")
            print(f"  比例差异: {abs(mx1_mean - mx2_mean):.4f}")
            
            # 判断混合物类型
            if abs(mx1_mean - mx2_mean) < 0.2:
                mixture_type = "平衡混合物"
            elif mx1_mean > 0.7:
                mixture_type = "主要贡献者占优"
            else:
                mixture_type = "中等不平衡"
            
            print(f"  混合物类型: {mixture_type}")
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n生成综合分析报告...")
        
        report = {
            "analysis_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_summary": {},
            "problem1_results": {},
            "problem2_results": {},
            "recommendations": []
        }
        
        # 问题1结果摘要
        if self.p1_data is not None:
            if 'baseline_pred' in self.p1_data.columns:
                accuracy = accuracy_score(self.p1_data['NoC_True'], self.p1_data['baseline_pred'])
                report["problem1_results"] = {
                    "overall_accuracy": float(accuracy),
                    "total_samples": len(self.p1_data),
                    "noc_distribution": self.p1_data['NoC_True'].value_counts().to_dict()
                }
            
            report["data_summary"]["problem1_samples"] = len(self.p1_data)
            report["data_summary"]["problem1_features"] = len([col for col in self.p1_data.columns 
                                                              if col not in ['Sample File', 'NoC_True']])
        
        # 问题2结果摘要
        if self.p2_data is not None and 'posterior_summary' in self.p2_data:
            posterior = self.p2_data['posterior_summary']
            report["problem2_results"] = {
                "mx1_mean": posterior['Mx_1']['mean'],
                "mx2_mean": posterior['Mx_2']['mean'],
                "mx1_credible_interval": posterior['Mx_1']['credible_interval_95'],
                "mx2_credible_interval": posterior['Mx_2']['credible_interval_95'],
                "model_converged": posterior['model_quality']['converged'],
                "acceptance_rate": posterior['model_quality']['acceptance_rate']
            }
        
        # 生成建议
        if self.p1_data is not None and 'baseline_pred' in self.p1_data.columns:
            accuracy = accuracy_score(self.p1_data['NoC_True'], self.p1_data['baseline_pred'])
            if accuracy > 0.9:
                report["recommendations"].append("问题1的NoC识别模型表现优秀，建议投入实际应用")
            elif accuracy > 0.8:
                report["recommendations"].append("问题1的NoC识别模型表现良好，建议进一步优化特征工程")
            else:
                report["recommendations"].append("问题1的NoC识别模型需要改进，建议重新审视特征选择和模型选择")
        
        if (self.p2_data is not None and 'posterior_summary' in self.p2_data and 
            self.p2_data['posterior_summary']['model_quality']['converged']):
            report["recommendations"].append("问题2的MCMC模型收敛良好，混合比例估计可信")
        
        # 保存报告
        report_file = os.path.join(REPORTS_DIR, 'comprehensive_analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 综合分析报告已保存: {report_file}")
        return report

# ==================== 主执行函数 ====================
def main():
    """主执行函数"""
    
    # 1. 数据加载
    loader = STRDataLoader()
    
    # 加载问题1数据
    p1_loaded = loader.load_problem1_data('prob1_features_enhanced.csv')
    
    # 加载问题2数据
    p2_loaded = loader.load_problem2_data('problem2_mcmc_results.json')
    
    if not p1_loaded and not p2_loaded:
        print("错误: 没有成功加载任何数据文件")
        return
    
    # 2. 问题1分析
    if p1_loaded:
        print("\n" + "="*50)
        print("开始问题1分析...")
        p1_analyzer = Problem1Analyzer(loader.p1_data)
        
        # NoC分布分析
        noc_dist = p1_analyzer.analyze_noc_distribution()
        
        # 模型性能分析
        performance = p1_analyzer.analyze_model_performance()
        
        # 特征重要性分析
        feature_importance = p1_analyzer.analyze_feature_importance()
        
        # 特征分布分析
        p1_analyzer.analyze_feature_distributions()
    
    # 3. 问题2分析
    if p2_loaded:
        print("\n" + "="*50)
        print("开始问题2分析...")
        p2_analyzer = Problem2Analyzer(loader.p2_data)
        
        # 后验分布分析
        posterior_analysis = p2_analyzer.analyze_posterior_summary()
        
        # 收敛分析
        convergence_analysis = p2_analyzer.analyze_convergence()
        
        # 绘制迹图
        p2_analyzer.plot_mixture_ratios_trace()
    
    # 4. 综合分析
    if p1_loaded or p2_loaded:
        print("\n" + "="*50)
        print("开始综合分析...")
        comp_analyzer = ComprehensiveAnalyzer(loader.p1_data, loader.p2_data)
        
        # 模型一致性分析
        comp_analyzer.analyze_model_consistency()
        
        # 生成综合报告
        report = comp_analyzer.generate_comprehensive_report()
    
    # 5. 生成总结
    print("\n" + "="*60)
    print("🎉 STR数据综合分析完成!")
    print("="*60)
    
    print(f"\n📊 分析结果总结:")
    if p1_loaded and loader.p1_data is not None:
        if 'baseline_pred' in loader.p1_data.columns:
            accuracy = accuracy_score(loader.p1_data['NoC_True'], loader.p1_data['baseline_pred'])
            print(f"   问题1 NoC识别准确率: {accuracy:.4f}")
        print(f"   问题1 样本总数: {len(loader.p1_data)}")
    
    if p2_loaded and loader.p2_data is not None:
        if 'posterior_summary' in loader.p2_data:
            mx1_mean = loader.p2_data['posterior_summary']['Mx_1']['mean']
            mx2_mean = loader.p2_data['posterior_summary']['Mx_2']['mean']
            print(f"   问题2 主要贡献者比例: {mx1_mean:.4f}")
            print(f"   问题2 次要贡献者比例: {mx2_mean:.4f}")
            converged = loader.p2_data['posterior_summary']['model_quality']['converged']
            print(f"   问题2 MCMC收敛状态: {'✓' if converged else '✗'}")
    
    print(f"\n📁 输出文件:")
    print(f"   图表目录: {PLOTS_DIR}")
    print(f"   报告目录: {REPORTS_DIR}")
    print(f"   - NoC分布图: noc_distribution.png")
    print(f"   - 混淆矩阵: confusion_matrix.png")
    print(f"   - 特征重要性: feature_importance.png")
    print(f"   - 特征分布: feature_distributions.png")
    if p2_loaded:
        print(f"   - MCMC迹图: mcmc_trace_plots.png")
        print(f"   - 混合比例联合分布: mixture_ratios_joint.png")
    print(f"   - 综合分析报告: comprehensive_analysis_report.json")
    
    print(f"\n💡 使用建议:")
    print(f"   1. 查看图表了解数据分布和模型性能")
    print(f"   2. 阅读JSON报告获取详细分析结果")
    print(f"   3. 根据特征重要性优化模型")
    if p2_loaded:
        print(f"   4. 检查MCMC收敛性确保结果可信")

# ==================== 第五部分：高级分析功能 ====================
print("\n--- 第五部分：高级分析功能 ---")

class AdvancedAnalyzer:
    """高级分析器"""
    
    def __init__(self, p1_data, p2_data):
        self.p1_data = p1_data
        self.p2_data = p2_data
    
    def correlation_analysis(self):
        """相关性分析"""
        if self.p1_data is None:
            return
        
        print("\n高级相关性分析:")
        
        # 选择数值型特征
        numeric_features = self.p1_data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'NoC_True']
        
        if len(numeric_features) > 10:
            # 计算相关性矩阵
            corr_matrix = self.p1_data[numeric_features[:20]].corr()  # 限制前20个特征
            
            # 绘制相关性热图
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('特征相关性热图')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'feature_correlation_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 找出高相关性特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:  # 高相关性阈值
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            if high_corr_pairs:
                print("发现高相关性特征对 (|r| > 0.8):")
                for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
                    print(f"  {feat1} <-> {feat2}: {corr:.3f}")
            else:
                print("未发现高相关性特征对")
    
    def outlier_analysis(self):
        """异常值分析"""
        if self.p1_data is None:
            return
        
        print("\n异常值分析:")
        
        # 选择几个重要的数值特征
        key_features = ['mac_profile', 'avg_alleles_per_locus', 'avg_peak_height', 'std_peak_height']
        available_features = [f for f in key_features if f in self.p1_data.columns]
        
        if len(available_features) >= 2:
            fig, axes = plt.subplots(1, len(available_features), figsize=(4*len(available_features), 6))
            if len(available_features) == 1:
                axes = [axes]
            
            outlier_summary = {}
            
            for i, feature in enumerate(available_features):
                # 使用IQR方法检测异常值
                Q1 = self.p1_data[feature].quantile(0.25)
                Q3 = self.p1_data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.p1_data[(self.p1_data[feature] < lower_bound) | 
                                       (self.p1_data[feature] > upper_bound)]
                
                outlier_summary[feature] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.p1_data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # 绘制箱线图
                axes[i].boxplot(self.p1_data[feature], vert=True)
                axes[i].set_title(f'{feature}\n异常值: {len(outliers)}个')
                axes[i].set_ylabel(feature)
                axes[i].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'outlier_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 输出异常值统计
            for feature, stats in outlier_summary.items():
                print(f"  {feature}: {stats['count']}个异常值 ({stats['percentage']:.1f}%)")
    
    def model_stability_analysis(self):
        """模型稳定性分析"""
        if self.p1_data is None or 'baseline_pred' not in self.p1_data.columns:
            return
        
        print("\n模型稳定性分析:")
        
        # 按NoC分析预测准确率的稳定性
        stability_results = {}
        
        for noc in sorted(self.p1_data['NoC_True'].unique()):
            noc_data = self.p1_data[self.p1_data['NoC_True'] == noc]
            if len(noc_data) > 5:  # 至少5个样本才分析
                # 计算准确率
                accuracy = (noc_data['NoC_True'] == noc_data['baseline_pred']).mean()
                
                # 计算预测置信度（如果有概率预测的话）
                # 这里简化为预测正确性的二项分布置信区间
                n = len(noc_data)
                p = accuracy
                se = np.sqrt(p * (1 - p) / n)
                ci_lower = max(0, p - 1.96 * se)
                ci_upper = min(1, p + 1.96 * se)
                
                stability_results[noc] = {
                    'accuracy': accuracy,
                    'samples': n,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower
                }
                
                print(f"  {noc}人混合物: 准确率 {accuracy:.3f} "
                      f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}], 样本数: {n})")
        
        # 可视化稳定性
        if len(stability_results) > 1:
            nocs = list(stability_results.keys())
            accuracies = [stability_results[noc]['accuracy'] for noc in nocs]
            ci_lowers = [stability_results[noc]['ci_lower'] for noc in nocs]
            ci_uppers = [stability_results[noc]['ci_upper'] for noc in nocs]
            
            plt.figure(figsize=(10, 6))
            x_pos = np.arange(len(nocs))
            
            # 绘制准确率及置信区间
            plt.errorbar(x_pos, accuracies, 
                        yerr=[np.array(accuracies) - np.array(ci_lowers),
                              np.array(ci_uppers) - np.array(accuracies)],
                        fmt='o', capsize=5, capthick=2, markersize=8)
            
            plt.xlabel('贡献者人数 (NoC)')
            plt.ylabel('预测准确率')
            plt.title('模型在不同NoC下的稳定性')
            plt.xticks(x_pos, [f'{noc}人' for noc in nocs])
            plt.grid(alpha=0.3)
            plt.ylim(0, 1.1)
            
            # 添加总体准确率线
            overall_acc = (self.p1_data['NoC_True'] == self.p1_data['baseline_pred']).mean()
            plt.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7,
                       label=f'总体准确率: {overall_acc:.3f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'model_stability.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def mcmc_diagnostics(self):
        """MCMC诊断"""
        if self.p2_data is None or 'sample_mixture_ratios' not in self.p2_data:
            return
        
        print("\nMCMC高级诊断:")
        
        samples = np.array(self.p2_data['sample_mixture_ratios'])
        mx1_samples = samples[:, 0]
        mx2_samples = samples[:, 1]
        
        # 自相关函数分析
        def autocorr(x, max_lag=100):
            """计算自相关函数"""
            n = len(x)
            x = x - np.mean(x)
            autocorrs = np.correlate(x, x, mode='full')
            autocorrs = autocorrs[n-1:]
            autocorrs = autocorrs / autocorrs[0]
            return autocorrs[:max_lag+1]
        
        # 计算自相关
        mx1_autocorr = autocorr(mx1_samples)
        mx2_autocorr = autocorr(mx2_samples)
        
        # 估计有效样本大小
        def effective_sample_size(autocorr_func):
            """根据自相关函数估计有效样本大小"""
            # 找到第一个负值或小于0.05的位置
            tau_int = 1
            for i in range(1, len(autocorr_func)):
                if autocorr_func[i] <= 0.05:
                    break
                tau_int += 2 * autocorr_func[i]
            
            return len(mx1_samples) / (2 * tau_int + 1)
        
        ess_mx1 = effective_sample_size(mx1_autocorr)
        ess_mx2 = effective_sample_size(mx2_autocorr)
        
        print(f"  Mx_1 有效样本大小: {ess_mx1:.0f}")
        print(f"  Mx_2 有效样本大小: {ess_mx2:.0f}")
        
        # 绘制自相关图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        lags = np.arange(len(mx1_autocorr))
        axes[0].plot(lags, mx1_autocorr, 'b-', alpha=0.8)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='阈值 (0.05)')
        axes[0].set_xlabel('滞后 (Lag)')
        axes[0].set_ylabel('自相关')
        axes[0].set_title(f'Mx_1 自相关函数 (ESS≈{ess_mx1:.0f})')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(lags, mx2_autocorr, 'r-', alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='阈值 (0.05)')
        axes[1].set_xlabel('滞后 (Lag)')
        axes[1].set_ylabel('自相关')
        axes[1].set_title(f'Mx_2 自相关函数 (ESS≈{ess_mx2:.0f})')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'mcmc_autocorr.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Geweke诊断
        def geweke_diagnostic(x, first=0.1, last=0.5):
            """Geweke收敛诊断"""
            n = len(x)
            first_part = x[:int(first * n)]
            last_part = x[int((1-last) * n):]
            
            mean1, var1 = np.mean(first_part), np.var(first_part, ddof=1)
            mean2, var2 = np.mean(last_part), np.var(last_part, ddof=1)
            
            se1 = var1 / len(first_part)
            se2 = var2 / len(last_part)
            
            z_score = (mean1 - mean2) / np.sqrt(se1 + se2)
            return z_score
        
        geweke_mx1 = geweke_diagnostic(mx1_samples)
        geweke_mx2 = geweke_diagnostic(mx2_samples)
        
        print(f"  Geweke诊断:")
        print(f"    Mx_1 Z分数: {geweke_mx1:.3f} {'✓' if abs(geweke_mx1) < 2 else '✗'}")
        print(f"    Mx_2 Z分数: {geweke_mx2:.3f} {'✓' if abs(geweke_mx2) < 2 else '✗'}")

# 扩展main函数以包含高级分析
def enhanced_main():
    """增强版主函数"""
    # 执行基本分析
    main()
    
    # 执行高级分析
    print("\n" + "="*50)
    print("开始高级分析...")
    
    loader = STRDataLoader()
    loader.load_problem1_data('prob1_features_enhanced.csv')
    loader.load_problem2_data('problem2_mcmc_results.json')
    
    if loader.p1_data is not None or loader.p2_data is not None:
        advanced_analyzer = AdvancedAnalyzer(loader.p1_data, loader.p2_data)
        
        # 相关性分析
        advanced_analyzer.correlation_analysis()
        
        # 异常值分析
        advanced_analyzer.outlier_analysis()
        
        # 模型稳定性分析
        advanced_analyzer.model_stability_analysis()
        
        # MCMC诊断
        advanced_analyzer.mcmc_diagnostics()
        
        print("\n✓ 高级分析完成")

# ==================== 执行主程序 ====================
if __name__ == "__main__":
    try:
        enhanced_main()
        print("\n🎉 所有分析任务完成!")
        print("请查看生成的图表和报告文件。")
    except Exception as e:
        print(f"\n❌ 程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()