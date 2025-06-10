# -*- coding: utf-8 -*-
"""
Q4增强STR图谱可视化模块
生成降噪前后的专业级STR电泳图谱对比图

版本: V2.0 - Enhanced STR Electropherogram Visualization
日期: 2025-06-10
功能: 
1. 生成类似实验室设备的STR电泳图谱
2. 降噪前后的并排对比显示
3. 支持多色彩标记系统
4. 峰识别和注释功能
5. 专业级图表输出
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy import signal
from scipy.interpolate import interp1d
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

class STRElectropherogramVisualizer:
    """STR电泳图谱可视化器"""
    
    def __init__(self):
        # STR位点的标准颜色映射（模拟ABI 3500设备）
        self.marker_colors = {
            'D3S1358': '#FF0000',   # 红色
            'vWA': '#FF0000',       # 红色
            'D16S539': '#FF0000',   # 红色
            'CSF1PO': '#FF0000',    # 红色
            'TPOX': '#FF0000',      # 红色
            'D8S1179': '#00FF00',   # 绿色
            'D21S11': '#00FF00',    # 绿色
            'D18S51': '#00FF00',    # 绿色
            'D2S441': '#00FF00',    # 绿色
            'D19S433': '#0000FF',   # 蓝色
            'TH01': '#0000FF',      # 蓝色
            'FGA': '#0000FF',       # 蓝色
            'D22S1045': '#0000FF',  # 蓝色
            'D5S818': '#FFFF00',    # 黄色
            'D13S317': '#FFFF00',   # 黄色
            'D7S820': '#FFFF00',    # 黄色
            'D6S1043': '#FFFF00',   # 黄色
            'D10S1248': '#FFFF00',  # 黄色
            'D1S1656': '#FF8000',   # 橙色
            'D12S391': '#FF8000',   # 橙色
            'D2S1338': '#FF8000',   # 橙色
            'AMEL': '#800080',      # 紫色
            'DYS391': '#800080',    # 紫色 (Y染色体)
        }
        
        # 默认颜色循环
        self.default_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF8000', '#800080']
        
        # 图表样式设置
        self.style_config = {
            'figure_size': (16, 10),
            'dpi': 300,
            'line_width': 1.0,
            'peak_line_width': 2.0,
            'marker_size': 6,
            'font_size_title': 14,
            'font_size_label': 12,
            'font_size_annotation': 10,
            'background_color': '#000000',  # 黑色背景（模拟设备）
            'grid_color': '#333333',
            'text_color': '#FFFFFF'
        }
    
    def create_electropherogram_comparison(self, 
                                         original_peaks: pd.DataFrame,
                                         denoised_peaks: pd.DataFrame,
                                         sample_name: str,
                                         output_path: str = None,
                                         show_annotations: bool = True,
                                         show_allele_calls: bool = True) -> None:
        """
        创建降噪前后的STR电泳图谱对比图
        
        Args:
            original_peaks: 原始峰数据
            denoised_peaks: 降噪后峰数据
            sample_name: 样本名称
            output_path: 输出路径
            show_annotations: 是否显示注释
            show_allele_calls: 是否显示等位基因调用
        """
        
        # 创建图形
        fig = plt.figure(figsize=self.style_config['figure_size'], 
                        facecolor=self.style_config['background_color'])
        
        # 创建子图布局
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # 上方：降噪前
        ax1 = fig.add_subplot(gs[0])
        self._plot_single_electropherogram(ax1, original_peaks, 
                                         f"{sample_name} - 降噪前", 
                                         show_annotations, show_allele_calls)
        
        # 下方：降噪后
        ax2 = fig.add_subplot(gs[1])
        self._plot_single_electropherogram(ax2, denoised_peaks, 
                                         f"{sample_name} - 降噪后", 
                                         show_annotations, show_allele_calls)
        
        # 设置x轴对齐
        if not original_peaks.empty and not denoised_peaks.empty:
            all_sizes = pd.concat([original_peaks['Size'], denoised_peaks['Size']])
            x_min, x_max = all_sizes.min() - 10, all_sizes.max() + 10
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
        
        # 添加总标题
        fig.suptitle(f'STR电泳图谱降噪效果对比 - {sample_name}', 
                    fontsize=self.style_config['font_size_title'],
                    color=self.style_config['text_color'],
                    y=0.95)
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, 
                       dpi=self.style_config['dpi'],
                       facecolor=self.style_config['background_color'],
                       bbox_inches='tight')
            print(f"STR电泳图谱已保存到: {output_path}")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_single_electropherogram(self, 
                                    ax: plt.Axes,
                                    peaks_data: pd.DataFrame,
                                    title: str,
                                    show_annotations: bool,
                                    show_allele_calls: bool) -> None:
        """绘制单个电泳图谱"""
        
        # 设置背景
        ax.set_facecolor(self.style_config['background_color'])
        
        if peaks_data.empty:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes,
                   ha='center', va='center', 
                   color=self.style_config['text_color'],
                   fontsize=self.style_config['font_size_label'])
            ax.set_title(title, color=self.style_config['text_color'],
                        fontsize=self.style_config['font_size_label'])
            return
        
        # 按位点分组绘制
        markers = peaks_data['Marker'].unique()
        
        # 创建连续的电泳图谱背景
        if not peaks_data.empty:
            size_range = np.arange(peaks_data['Size'].min() - 20, 
                                 peaks_data['Size'].max() + 20, 0.5)
            baseline = np.zeros_like(size_range)
            
            # 绘制基线
            ax.plot(size_range, baseline, 
                   color='#333333', linewidth=0.5, alpha=0.7)
        
        # 为每个位点绘制峰
        for marker in markers:
            marker_data = peaks_data[peaks_data['Marker'] == marker]
            color = self._get_marker_color(marker)
            
            # 为每个峰创建高斯形状的电泳峰
            for _, peak in marker_data.iterrows():
                size = peak['Size']
                height = peak['Height']
                allele = peak.get('Allele', '')
                
                # 创建高斯峰形状
                peak_x, peak_y = self._create_gaussian_peak(size, height)
                
                # 绘制峰
                ax.plot(peak_x, peak_y, color=color, 
                       linewidth=self.style_config['peak_line_width'],
                       alpha=0.8)
                
                # 填充峰下方区域
                ax.fill_between(peak_x, 0, peak_y, 
                              color=color, alpha=0.3)
                
                # 添加等位基因标注
                if show_allele_calls and allele:
                    ax.annotate(str(allele), 
                              xy=(size, height),
                              xytext=(size, height + height * 0.1),
                              ha='center', va='bottom',
                              color=color,
                              fontsize=self.style_config['font_size_annotation'],
                              fontweight='bold')
                
                # 添加峰高数值
                if show_annotations:
                    ax.annotate(f'{int(height)}', 
                              xy=(size, height),
                              xytext=(size + 2, height + height * 0.05),
                              ha='left', va='bottom',
                              color='white',
                              fontsize=self.style_config['font_size_annotation'] - 2,
                              alpha=0.8)
        
        # 设置坐标轴
        ax.set_xlabel('Size (bp)', color=self.style_config['text_color'],
                     fontsize=self.style_config['font_size_label'])
        ax.set_ylabel('Height (RFU)', color=self.style_config['text_color'],
                     fontsize=self.style_config['font_size_label'])
        ax.set_title(title, color=self.style_config['text_color'],
                    fontsize=self.style_config['font_size_label'])
        
        # 设置网格
        ax.grid(True, color=self.style_config['grid_color'], 
               linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 设置坐标轴颜色
        ax.tick_params(colors=self.style_config['text_color'], 
                      labelsize=self.style_config['font_size_annotation'])
        for spine in ax.spines.values():
            spine.set_color(self.style_config['text_color'])
        
        # 设置y轴范围
        if not peaks_data.empty:
            max_height = peaks_data['Height'].max()
            ax.set_ylim(-max_height * 0.05, max_height * 1.2)
    
    def _create_gaussian_peak(self, center: float, height: float, 
                            width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """创建高斯形状的电泳峰"""
        sigma = width
        x = np.linspace(center - 4*sigma, center + 4*sigma, 100)
        y = height * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return x, y
    
    def _get_marker_color(self, marker: str) -> str:
        """获取位点对应的颜色"""
        if marker in self.marker_colors:
            return self.marker_colors[marker]
        
        # 如果位点不在预定义列表中，使用哈希来分配颜色
        color_index = hash(marker) % len(self.default_colors)
        return self.default_colors[color_index]
    
    def create_detailed_locus_view(self, 
                                 original_peaks: pd.DataFrame,
                                 denoised_peaks: pd.DataFrame,
                                 target_marker: str,
                                 sample_name: str,
                                 output_path: str = None) -> None:
        """
        创建特定位点的详细视图
        
        Args:
            original_peaks: 原始峰数据
            denoised_peaks: 降噪后峰数据
            target_marker: 目标位点
            sample_name: 样本名称
            output_path: 输出路径
        """
        
        # 筛选目标位点数据
        orig_marker = original_peaks[original_peaks['Marker'] == target_marker]
        denoised_marker = denoised_peaks[denoised_peaks['Marker'] == target_marker]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                      facecolor=self.style_config['background_color'])
        
        # 绘制原始数据
        self._plot_locus_detail(ax1, orig_marker, f"{target_marker} - 降噪前")
        
        # 绘制降噪后数据
        self._plot_locus_detail(ax2, denoised_marker, f"{target_marker} - 降噪后")
        
        # 设置相同的x轴范围
        if not orig_marker.empty and not denoised_marker.empty:
            all_sizes = pd.concat([orig_marker['Size'], denoised_marker['Size']])
            x_min, x_max = all_sizes.min() - 5, all_sizes.max() + 5
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
        
        # 添加总标题
        fig.suptitle(f'{target_marker} 位点降噪详细对比 - {sample_name}',
                    color=self.style_config['text_color'],
                    fontsize=self.style_config['font_size_title'])
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, 
                       dpi=self.style_config['dpi'],
                       facecolor=self.style_config['background_color'],
                       bbox_inches='tight')
            print(f"位点详细图已保存到: {output_path}")
        
        plt.tight_layout()
        plt.show()
    
    def _plot_locus_detail(self, ax: plt.Axes, marker_data: pd.DataFrame, title: str) -> None:
        """绘制位点详细视图"""
        
        ax.set_facecolor(self.style_config['background_color'])
        
        if marker_data.empty:
            ax.text(0.5, 0.5, '无数据', transform=ax.transAxes,
                   ha='center', va='center', 
                   color=self.style_config['text_color'])
            ax.set_title(title, color=self.style_config['text_color'])
            return
        
        marker = marker_data.iloc[0]['Marker']
        color = self._get_marker_color(marker)
        
        # 绘制每个峰
        for _, peak in marker_data.iterrows():
            size = peak['Size']
            height = peak['Height']
            allele = peak.get('Allele', '')
            
            # 创建更详细的峰形状
            peak_x, peak_y = self._create_gaussian_peak(size, height, width=0.5)
            
            # 绘制峰
            ax.plot(peak_x, peak_y, color=color, 
                   linewidth=self.style_config['peak_line_width'] + 1,
                   alpha=0.9)
            
            # 填充
            ax.fill_between(peak_x, 0, peak_y, color=color, alpha=0.4)
            
            # 详细标注
            ax.annotate(f'{allele}\n({size:.1f}bp)\n{int(height)}RFU',
                       xy=(size, height),
                       xytext=(size, height + height * 0.2),
                       ha='center', va='bottom',
                       color=color,
                       fontsize=self.style_config['font_size_annotation'],
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='black', alpha=0.7))
        
        # 设置坐标轴
        ax.set_xlabel('Size (bp)', color=self.style_config['text_color'])
        ax.set_ylabel('Height (RFU)', color=self.style_config['text_color'])
        ax.set_title(title, color=self.style_config['text_color'])
        
        # 设置网格
        ax.grid(True, color=self.style_config['grid_color'], 
               linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 设置坐标轴颜色
        ax.tick_params(colors=self.style_config['text_color'])
        for spine in ax.spines.values():
            spine.set_color(self.style_config['text_color'])
    
    def create_multi_sample_comparison(self, 
                                     samples_data: Dict[str, Dict],
                                     output_path: str = None) -> None:
        """
        创建多样本降噪效果对比图
        
        Args:
            samples_data: {sample_name: {'original': df, 'denoised': df}}
            output_path: 输出路径
        """
        
        n_samples = len(samples_data)
        if n_samples == 0:
            return
        
        # 创建子图网格
        fig, axes = plt.subplots(n_samples, 2, 
                               figsize=(16, 6*n_samples),
                               facecolor=self.style_config['background_color'])
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        # 为每个样本绘制对比图
        for i, (sample_name, data) in enumerate(samples_data.items()):
            # 降噪前
            self._plot_single_electropherogram(
                axes[i, 0], data['original'], 
                f"{sample_name} - 降噪前", True, True)
            
            # 降噪后
            self._plot_single_electropherogram(
                axes[i, 1], data['denoised'], 
                f"{sample_name} - 降噪后", True, True)
        
        # 添加总标题
        fig.suptitle('多样本STR图谱降噪效果对比',
                    color=self.style_config['text_color'],
                    fontsize=self.style_config['font_size_title'],
                    y=0.98)
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, 
                       dpi=self.style_config['dpi'],
                       facecolor=self.style_config['background_color'],
                       bbox_inches='tight')
            print(f"多样本对比图已保存到: {output_path}")
        
        plt.tight_layout()
        plt.show()
    
    def create_noise_analysis_plot(self, 
                                 original_peaks: pd.DataFrame,
                                 denoised_peaks: pd.DataFrame,
                                 noise_threshold: float,
                                 sample_name: str,
                                 output_path: str = None) -> None:
        """
        创建噪声分析图，突出显示被过滤的噪声峰
        
        Args:
            original_peaks: 原始峰数据
            denoised_peaks: 降噪后峰数据
            noise_threshold: 噪声阈值
            sample_name: 样本名称
            output_path: 输出路径
        """
        
        fig, ax = plt.subplots(figsize=(16, 8),
                              facecolor=self.style_config['background_color'])
        
        ax.set_facecolor(self.style_config['background_color'])
        
        # 识别被过滤的峰
        if not original_peaks.empty and not denoised_peaks.empty:
            # 创建峰的唯一标识
            def create_peak_id(df):
                return df.apply(lambda x: f"{x['Marker']}_{x['Allele']}_{x['Size']:.1f}", axis=1)
            
            orig_ids = set(create_peak_id(original_peaks))
            denoised_ids = set(create_peak_id(denoised_peaks))
            filtered_ids = orig_ids - denoised_ids
            
            # 绘制保留的峰（绿色）
            for _, peak in denoised_peaks.iterrows():
                size = peak['Size']
                height = peak['Height']
                marker = peak['Marker']
                
                peak_x, peak_y = self._create_gaussian_peak(size, height)
                ax.plot(peak_x, peak_y, color='#00FF00', 
                       linewidth=self.style_config['peak_line_width'],
                       alpha=0.8, label='保留峰' if _ == 0 else "")
                ax.fill_between(peak_x, 0, peak_y, color='#00FF00', alpha=0.3)
            
            # 绘制被过滤的峰（红色）
            for _, peak in original_peaks.iterrows():
                peak_id = f"{peak['Marker']}_{peak['Allele']}_{peak['Size']:.1f}"
                if peak_id in filtered_ids:
                    size = peak['Size']
                    height = peak['Height']
                    
                    peak_x, peak_y = self._create_gaussian_peak(size, height)
                    ax.plot(peak_x, peak_y, color='#FF0000', 
                           linewidth=self.style_config['peak_line_width'],
                           alpha=0.7, linestyle='--',
                           label='过滤峰' if _ == 0 else "")
                    ax.fill_between(peak_x, 0, peak_y, color='#FF0000', alpha=0.2)
        
        # 添加噪声阈值线
        if not original_peaks.empty:
            x_range = [original_peaks['Size'].min() - 10, 
                      original_peaks['Size'].max() + 10]
            ax.plot(x_range, [noise_threshold, noise_threshold], 
                   color='#FFFF00', linestyle='-', linewidth=2,
                   alpha=0.8, label=f'噪声阈值 ({noise_threshold} RFU)')
        
        # 设置图表
        ax.set_xlabel('Size (bp)', color=self.style_config['text_color'],
                     fontsize=self.style_config['font_size_label'])
        ax.set_ylabel('Height (RFU)', color=self.style_config['text_color'],
                     fontsize=self.style_config['font_size_label'])
        ax.set_title(f'噪声过滤分析 - {sample_name}',
                    color=self.style_config['text_color'],
                    fontsize=self.style_config['font_size_title'])
        
        # 设置网格和坐标轴
        ax.grid(True, color=self.style_config['grid_color'], 
               linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(colors=self.style_config['text_color'])
        for spine in ax.spines.values():
            spine.set_color(self.style_config['text_color'])
        
        # 添加图例
        ax.legend(loc='upper right', framealpha=0.8, 
                 facecolor='black', edgecolor='white',
                 labelcolor=self.style_config['text_color'])
        
        # 保存图片
        if output_path:
            plt.savefig(output_path, 
                       dpi=self.style_config['dpi'],
                       facecolor=self.style_config['background_color'],
                       bbox_inches='tight')
            print(f"噪声分析图已保存到: {output_path}")
        
        plt.tight_layout()
        plt.show()


def create_demo_str_data():
    """创建演示用的STR数据"""
    np.random.seed(42)
    
    # STR位点列表
    markers = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11', 
               'D18S51', 'D5S818', 'D13S317', 'D7S820', 'TH01']
    
    # 生成原始峰数据
    original_data = []
    for marker in markers:
        # 每个位点生成2-4个主峰
        n_main_peaks = np.random.randint(2, 5)
        base_size = 100 + hash(marker) % 300
        
        for i in range(n_main_peaks):
            size = base_size + i * 4 + np.random.normal(0, 0.5)
            height = np.random.lognormal(np.log(1000), 0.6)
            allele = str(int((size - 100) / 4) + 10 + i)
            
            original_data.append({
                'Sample File': 'Demo_Sample',
                'Marker': marker,
                'Allele': allele,
                'Size': size,
                'Height': height,
                'Original_Height': height
            })
        
        # 添加一些噪声峰
        n_noise_peaks = np.random.randint(2, 6)
        for i in range(n_noise_peaks):
            size = base_size + np.random.uniform(-20, 20)
            height = np.random.uniform(30, 200)  # 低高度噪声
            allele = f'N{i+1}'
            
            original_data.append({
                'Sample File': 'Demo_Sample',
                'Marker': marker,
                'Allele': allele,
                'Size': size,
                'Height': height,
                'Original_Height': height
            })
    
    original_df = pd.DataFrame(original_data)
    
    # 生成降噪后数据（移除低峰）
    denoised_df = original_df[original_df['Height'] > 150].copy()
    
    return original_df, denoised_df


def demo_str_visualization():
    """演示STR电泳图谱可视化功能"""
    print("=" * 60)
    print("STR电泳图谱可视化演示")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = STRElectropherogramVisualizer()
    
    # 生成演示数据
    print("生成演示数据...")
    original_peaks, denoised_peaks = create_demo_str_data()
    
    print(f"原始峰数: {len(original_peaks)}")
    print(f"降噪后峰数: {len(denoised_peaks)}")
    print(f"过滤峰数: {len(original_peaks) - len(denoised_peaks)}")
    
    # 创建输出目录
    output_dir = './str_visualization_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成整体对比图
    print("\n生成整体电泳图谱对比...")
    visualizer.create_electropherogram_comparison(
        original_peaks, denoised_peaks, 
        "Demo_Sample",
        os.path.join(output_dir, 'str_comparison_overall.png')
    )
    
    # 2. 生成位点详细视图
    print("\n生成位点详细视图...")
    target_marker = 'D3S1358'
    visualizer.create_detailed_locus_view(
        original_peaks, denoised_peaks,
        target_marker, "Demo_Sample",
        os.path.join(output_dir, f'str_detail_{target_marker}.png')
    )
    
    # 3. 生成噪声分析图
    print("\n生成噪声分析图...")
    visualizer.create_noise_analysis_plot(
        original_peaks, denoised_peaks,
        150, "Demo_Sample",
        os.path.join(output_dir, 'str_noise_analysis.png')
    )
    
    # 4. 多样本对比演示
    print("\n生成多样本对比图...")
    
    # 创建第二个演示样本
    original_peaks2, denoised_peaks2 = create_demo_str_data()
    original_peaks2['Sample File'] = 'Demo_Sample_2'
    denoised_peaks2['Sample File'] = 'Demo_Sample_2'
    
    # 多样本数据
    multi_sample_data = {
        'Sample_1': {'original': original_peaks, 'denoised': denoised_peaks},
        'Sample_2': {'original': original_peaks2, 'denoised': denoised_peaks2}
    }
    
    visualizer.create_multi_sample_comparison(
        multi_sample_data,
        os.path.join(output_dir, 'str_multi_sample_comparison.png')
    )
    
    print(f"\n所有图表已保存到目录: {output_dir}")
    print("演示完成！")


# Q4增强可视化集成类
class Q4EnhancedVisualization:
    """Q4增强可视化集成器，与Q4主模块无缝集成"""
    
    def __init__(self, upg_pipeline=None):
        self.visualizer = STRElectropherogramVisualizer()
        self.upg_pipeline = upg_pipeline
        
    def integrate_with_q4_results(self, q4_results: Dict, output_dir: str):
        """与Q4分析结果集成，生成完整的可视化报告"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        sample_id = q4_results.get('sample_file', 'Unknown_Sample')
        original_peaks = q4_results.get('original_peaks', pd.DataFrame())
        denoised_peaks = q4_results.get('denoised_peaks', pd.DataFrame())
        
        print(f"为样本 {sample_id} 生成增强可视化报告...")
        
        # 1. 主要对比图
        if not original_peaks.empty and not denoised_peaks.empty:
            self.visualizer.create_electropherogram_comparison(
                original_peaks, denoised_peaks, sample_id,
                os.path.join(output_dir, f'{sample_id}_electropherogram_comparison.png')
            )
        
        # 2. 为每个主要位点生成详细视图
        if not denoised_peaks.empty:
            major_markers = denoised_peaks['Marker'].value_counts().head(3).index
            for marker in major_markers:
                self.visualizer.create_detailed_locus_view(
                    original_peaks, denoised_peaks, marker, sample_id,
                    os.path.join(output_dir, f'{sample_id}_{marker}_detail.png')
                )
        
        # 3. 噪声分析图
        noise_threshold = 100  # 可以从Q4配置中获取
        self.visualizer.create_noise_analysis_plot(
            original_peaks, denoised_peaks, noise_threshold, sample_id,
            os.path.join(output_dir, f'{sample_id}_noise_analysis.png')
        )
        
        # 4. 生成HTML报告
        self._generate_html_report(q4_results, output_dir)
        
        print(f"增强可视化报告已生成: {output_dir}")
    
    def _generate_html_report(self, q4_results: Dict, output_dir: str):
        """生成HTML格式的综合报告"""
        
        sample_id = q4_results.get('sample_file', 'Unknown_Sample')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q4 STR图谱降噪分析报告 - {sample_id}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>STR图谱智能降噪分析报告</h1>
            <h2>样本: {sample_id}</h2>
            <p>基于UPG-M算法的混合STR图谱降噪与基因型推断</p>
        </div>
        
        <div class="section">
            <h2>📊 分析概要</h2>
            <div class="metric-grid">
"""
        
        # 添加关键指标
        denoising_metrics = q4_results.get('denoising_metrics', {})
        
        original_count = denoising_metrics.get('original_peak_count', 0)
        denoised_count = denoising_metrics.get('denoised_peak_count', 0)
        filtered_count = original_count - denoised_count
        filtering_ratio = denoising_metrics.get('filtering_ratio', 0)
        snr_improvement = denoising_metrics.get('snr_improvement', 1)
        
        html_content += f"""
                <div class="metric-card">
                    <h3>峰过滤统计</h3>
                    <p><strong>原始峰数:</strong> {original_count}</p>
                    <p><strong>降噪后峰数:</strong> {denoised_count}</p>
                    <p><strong>过滤峰数:</strong> {filtered_count}</p>
                    <p><strong>过滤率:</strong> {filtering_ratio:.1%}</p>
                </div>
                
                <div class="metric-card">
                    <h3>信号质量提升</h3>
                    <p><strong>信噪比改善:</strong> {snr_improvement:.2f}x</p>
                    <p><strong>动态范围:</strong> {denoising_metrics.get('dynamic_range', 0):.2f}</p>
                    <p><strong>信号一致性:</strong> {denoising_metrics.get('height_consistency', 0):.3f}</p>
                </div>
"""
        
        # 添加MCMC结果（如果有）
        if q4_results.get('mcmc_results'):
            mcmc_results = q4_results['mcmc_results']
            html_content += f"""
                <div class="metric-card">
                    <h3>MCMC推断结果</h3>
                    <p><strong>接受率:</strong> {mcmc_results['acceptance_rate']:.3f}</p>
                    <p><strong>有效样本数:</strong> {mcmc_results['n_samples']}</p>
                    <p><strong>收敛状态:</strong> {'✅ 已收敛' if mcmc_results.get('converged', False) else '⚠️ 未收敛'}</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 电泳图谱可视化</h2>
            <div class="image-gallery">
"""
        
        # 添加图片
        images = [
            (f'{sample_id}_electropherogram_comparison.png', 'STR电泳图谱降噪前后对比'),
            (f'{sample_id}_noise_analysis.png', '噪声过滤分析')
        ]
        
        # 检查位点详细图
        if not q4_results.get('denoised_peaks', pd.DataFrame()).empty:
            major_markers = q4_results['denoised_peaks']['Marker'].value_counts().head(2).index
            for marker in major_markers:
                images.append((f'{sample_id}_{marker}_detail.png', f'{marker} 位点详细视图'))
        
        for img_file, caption in images:
            html_content += f"""
                <div class="image-container">
                    <img src="{img_file}" alt="{caption}">
                    <h4>{caption}</h4>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 技术细节</h2>
            <div class="highlight">
                <h3>UPG-M算法特点</h3>
                <ul>
                    <li><strong>统一概率框架:</strong> 同时推断NoC、混合比例、基因型和降噪</li>
                    <li><strong>智能峰分类:</strong> 基于梯度提升的峰分类器区分信号、Stutter和噪声</li>
                    <li><strong>自适应CTA:</strong> 基于V5特征动态调整峰质量评估</li>
                    <li><strong>RJMCMC采样:</strong> 跨维MCMC处理变维问题</li>
                </ul>
            </div>
            
            <table class="summary-table">
                <tr>
                    <th>分析步骤</th>
                    <th>算法/方法</th>
                    <th>主要功能</th>
                </tr>
                <tr>
                    <td>1. 特征提取</td>
                    <td>V5特征工程</td>
                    <td>提取90+个STR图谱特征</td>
                </tr>
                <tr>
                    <td>2. 峰分类</td>
                    <td>梯度提升分类器</td>
                    <td>区分真实信号、Stutter和噪声</td>
                </tr>
                <tr>
                    <td>3. 自适应CTA</td>
                    <td>多因子CTA计算</td>
                    <td>动态调整峰质量评估阈值</td>
                </tr>
                <tr>
                    <td>4. 降噪处理</td>
                    <td>概率阈值过滤</td>
                    <td>移除低质量和噪声峰</td>
                </tr>
                <tr>
                    <td>5. 后验推断</td>
                    <td>RJMCMC采样</td>
                    <td>推断基因型和参数后验分布</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>⚡ 分析性能</h2>
"""
        
        computation_time = q4_results.get('computation_time', 0)
        success = q4_results.get('success', False)
        
        html_content += f"""
            <p><strong>计算时间:</strong> {computation_time:.1f} 秒</p>
            <p><strong>分析状态:</strong> {'✅ 成功' if success else '❌ 失败'}</p>
            <p><strong>生成时间:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📝 结论与建议</h2>
"""
        
        # 基于结果生成结论
        if filtering_ratio > 0.5:
            html_content += """
            <div class="highlight">
                <h3>⚠️ 高噪声样本</h3>
                <p>该样本包含较多噪声峰，建议检查实验条件和PCR扩增参数。</p>
            </div>
"""
        elif filtering_ratio > 0.3:
            html_content += """
            <div class="highlight">
                <h3>⚠️ 中等噪声样本</h3>
                <p>样本质量中等，降噪效果良好，可用于后续分析。</p>
            </div>
"""
        else:
            html_content += """
            <div class="highlight">
                <h3>✅ 高质量样本</h3>
                <p>样本质量优秀，噪声水平低，适合进行精确的基因型分析。</p>
            </div>
"""
        
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML报告
        html_file = os.path.join(output_dir, f'{sample_id}_analysis_report.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {html_file}")


# Q4主模块集成示例
def integrate_with_q4_main():
    """展示如何与Q4主模块集成"""
    
    print("=" * 60)
    print("Q4增强可视化集成示例")
    print("=" * 60)
    
    # 模拟Q4分析结果
    original_peaks, denoised_peaks = create_demo_str_data()
    
    # 模拟Q4结果结构
    q4_mock_results = {
        'sample_file': 'A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa',
        'original_peaks': original_peaks,
        'denoised_peaks': denoised_peaks,
        'v5_features': {
            'avg_peak_height': 1200.0,
            'signal_to_noise_ratio': 15.2,
            'weak_signal_ratio': 0.15
        },
        'denoising_metrics': {
            'original_peak_count': len(original_peaks),
            'denoised_peak_count': len(denoised_peaks),
            'filtering_ratio': (len(original_peaks) - len(denoised_peaks)) / len(original_peaks),
            'snr_improvement': 2.3,
            'dynamic_range': 25.5,
            'height_consistency': 0.85
        },
        'mcmc_results': {
            'acceptance_rate': 0.35,
            'n_samples': 2400,
            'converged': True
        },
        'computation_time': 45.2,
        'success': True
    }
    
    # 创建增强可视化器
    enhanced_viz = Q4EnhancedVisualization()
    
    # 生成完整报告
    output_dir = './q4_enhanced_report_demo'
    enhanced_viz.integrate_with_q4_results(q4_mock_results, output_dir)
    
    print(f"\n完整的Q4增强可视化报告已生成！")
    print(f"报告目录: {output_dir}")
    print("包含内容:")
    print("  • STR电泳图谱对比图")
    print("  • 位点详细视图")
    print("  • 噪声分析图")
    print("  • HTML综合报告")


if __name__ == "__main__":
    # 运行演示
    print("选择演示模式:")
    print("1. 基础STR电泳图谱可视化演示")
    print("2. Q4集成可视化演示")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        demo_str_visualization()
    elif choice == "2":
        integrate_with_q4_main()
    else:
        print("运行默认演示...")
        demo_str_visualization()
        integrate_with_q4_main()