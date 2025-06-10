# -*- coding: utf-8 -*-
"""
问题二使用示例脚本
展示如何使用MGM-RF系统进行混合STR图谱贡献者比例推断
"""

import sys
import os
import numpy as np
import pandas as pd
from Q2_MGM_RF_Solution import analyze_single_sample_from_att2, analyze_all_samples_from_att2, MGM_RF_Pipeline

def example_1_single_sample_analysis():
    """示例1：单样本分析"""
    print("\n" + "="*60)
    print("示例1：单样本分析")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"  # 可选
    
    # 检查文件是否存在
    if not os.path.exists(att2_path):
        print(f"错误：找不到附件2文件 - {att2_path}")
        print("请确保文件在当前目录下")
        return
    
    # 首先查看附件2中有哪些样本
    try:
        df_att2 = pd.read_csv(att2_path, encoding='utf-8')
        available_samples = df_att2['Sample File'].unique()
        print(f"附件2中可用的样本ID ({len(available_samples)}个):")
        for i, sample_id in enumerate(available_samples[:10], 1):  # 显示前10个
            print(f"  {i}. {sample_id}")
        if len(available_samples) > 10:
            print(f"  ... 还有{len(available_samples)-10}个样本")
        
        # 选择第一个样本进行分析
        if len(available_samples) > 0:
            target_sample = available_samples[0]
            print(f"\n选择样本 '{target_sample}' 进行分析...")
            
            # 执行分析
            result = analyze_single_sample_from_att2(
                sample_id=target_sample,
                att2_path=att2_path,
                q1_model_path=q1_model_path if os.path.exists(q1_model_path) else None
            )
            
            # 显示结果
            print_analysis_result(result)
            
        else:
            print("附件2中没有找到有效样本")
            
    except Exception as e:
        print(f"分析过程出错: {e}")

def example_2_batch_analysis():
    """示例2：批量分析（前3个样本）"""
    print("\n" + "="*60)
    print("示例2：批量分析前3个样本")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    if not os.path.exists(att2_path):
        print(f"错误：找不到附件2文件 - {att2_path}")
        return
    
    try:
        # 批量分析前3个样本
        all_results = analyze_all_samples_from_att2(
            att2_path=att2_path,
            q1_model_path=q1_model_path if os.path.exists(q1_model_path) else None,
            max_samples=3  # 只分析前3个样本
        )
        
        print(f"\n批量分析完成，共分析了 {len(all_results)} 个样本")
        
        # 显示汇总结果
        print("\n汇总结果:")
        for sample_id, result in all_results.items():
            print(f"\n样本: {sample_id}")
            print_analysis_result(result, brief=True)
            
    except Exception as e:
        print(f"批量分析过程出错: {e}")

def example_3_programmatic_usage():
    """示例3：程序化使用接口"""
    print("\n" + "="*60)
    print("示例3：程序化使用接口")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    if not os.path.exists(att2_path):
        print(f"错误：找不到附件2文件 - {att2_path}")
        return
    
    try:
        # 初始化MGM-RF流水线
        pipeline = MGM_RF_Pipeline(
            q1_model_path if os.path.exists(q1_model_path) else None
        )
        
        # 加载附件2数据
        print("加载附件2数据...")
        att2_data = pipeline.load_attachment2_data(att2_path)
        print(f"成功加载 {len(att2_data)} 个样本")
        
        # 准备频率数据
        print("准备等位基因频率数据...")
        att2_freq_data = pipeline.prepare_att2_frequency_data(att2_data)
        print(f"频率数据涵盖 {len(att2_freq_data)} 个位点")
        
        # 选择一个样本进行详细分析
        if att2_data:
            sample_id = list(att2_data.keys())[0]
            sample_data = att2_data[sample_id]
            
            print(f"\n分析样本: {sample_id}")
            
            # 执行分析
            result = pipeline.analyze_sample(sample_data, att2_freq_data)
            
            # 保存结果
            output_file = f"./example_result_{sample_id}.json"
            pipeline.save_results(result, output_file)
            print(f"结果已保存到: {output_file}")
            
            # 绘制图表
            pipeline.plot_results(result, "./example_plots")
            print("图表已保存到: ./example_plots/")
            
            # 显示结果
            print_analysis_result(result)
            
    except Exception as e:
        print(f"程序化使用过程出错: {e}")

def example_4_custom_parameters():
    """示例4：自定义参数配置"""
    print("\n" + "="*60)
    print("示例4：自定义参数配置")
    print("="*60)
    
    # 修改全局配置
    from Q2_MGM_RF_Solution import config
    
    print("原始MCMC参数:")
    print(f"  迭代次数: {config.N_ITERATIONS}")
    print(f"  预热次数: {config.N_WARMUP}")
    print(f"  K-top采样: {config.K_TOP}")
    
    # 自定义参数（快速测试模式）
    config.N_ITERATIONS = 5000  # 减少迭代次数以加快速度
    config.N_WARMUP = 1000      # 减少预热次数
    config.K_TOP = 200          # 减少K-top采样数量
    
    print("\n修改后的MCMC参数:")
    print(f"  迭代次数: {config.N_ITERATIONS}")
    print(f"  预热次数: {config.N_WARMUP}")
    print(f"  K-top采样: {config.K_TOP}")
    
    # 使用修改后的参数进行分析
    att2_path = "附件2：混合STR图谱数据.csv"
    
    if os.path.exists(att2_path):
        try:
            # 读取第一个样本
            df_att2 = pd.read_csv(att2_path, encoding='utf-8')
            available_samples = df_att2['Sample File'].unique()
            
            if len(available_samples) > 0:
                target_sample = available_samples[0]
                print(f"\n使用快速模式分析样本: {target_sample}")
                
                result = analyze_single_sample_from_att2(
                    sample_id=target_sample,
                    att2_path=att2_path
                )
                
                print(f"快速分析完成，计算时间: {result['computation_time']:.1f}秒")
                print_analysis_result(result, brief=True)
                
        except Exception as e:
            print(f"自定义参数分析出错: {e}")
    else:
        print(f"找不到附件2文件: {att2_path}")

def print_analysis_result(result, brief=False):
    """打印分析结果"""
    print(f"\n--- 分析结果 ---")
    print(f"样本ID: {result['sample_file']}")
    print(f"预测NoC: {result['predicted_noc']} (置信度: {result['noc_confidence']:.3f})")
    print(f"计算时间: {result['computation_time']:.1f}秒")
    
    if result['mcmc_results'] is not None:
        mcmc_results = result['mcmc_results']
        print(f"MCMC接受率: {mcmc_results['acceptance_rate']:.3f}")
        print(f"MCMC收敛: {'是' if mcmc_results['converged'] else '否'}")
        print(f"有效样本数: {mcmc_results['n_samples']}")
        
        if not brief:
            # 详细显示混合比例
            posterior_summary = result['posterior_summary']
            contributor_ranking = posterior_summary['contributor_ranking']
            
            print(f"\n混合比例估计:")
            for rank, (contributor_id, mean_ratio) in enumerate(contributor_ranking, 1):
                mx_stats = posterior_summary[f'Mx_{contributor_id}']
                ci_95 = mx_stats['credible_interval_95']
                print(f"  贡献者{contributor_id}: {mean_ratio:.4f} "
                      f"(95%CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}])")
                
            # 收敛性诊断
            conv_diag = result['convergence_diagnostics']
            print(f"\n收敛性诊断:")
            print(f"  状态: {conv_diag['convergence_status']}")
            print(f"  最小ESS: {conv_diag['min_ess']:.1f}")
            print(f"  最大Geweke: {conv_diag['max_geweke']:.3f}")
            
            if conv_diag['convergence_issues']:
                print(f"  收敛问题: {', '.join(conv_diag['convergence_issues'])}")
    else:
        print("⚠️  无有效峰数据，MCMC未执行")

def create_synthetic_test_data():
    """创建合成测试数据（如果没有真实数据的话）"""
    print("\n" + "="*60)
    print("创建合成测试数据")
    print("="*60)
    
    np.random.seed(42)
    
    # 生成合成的STR数据
    sample_data = []
    markers = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11']
    
    for i, sample_id in enumerate([f"Test_Sample_{j+1}" for j in range(3)], 1):
        for marker in markers:
            # 模拟2人混合样本
            n_alleles = np.random.randint(2, 5)  # 每个位点2-4个等位基因
            
            for allele_idx in range(n_alleles):
                allele_num = np.random.randint(10, 25)
                allele = str(allele_num) if np.random.random() > 0.3 else f"{allele_num}.3"
                
                size = 100 + allele_num * 4 + np.random.normal(0, 2)
                height = np.random.lognormal(np.log(1000), 0.5)
                
                sample_data.append({
                    'Sample File': sample_id,
                    'Marker': marker,
                    f'Allele {allele_idx+1}': allele,
                    f'Size {allele_idx+1}': size,
                    f'Height {allele_idx+1}': height
                })
    
    # 填充空列
    df_synthetic = pd.DataFrame(sample_data)
    
    # 添加空的Allele/Size/Height列
    for i in range(1, 101):
        if f'Allele {i}' not in df_synthetic.columns:
            df_synthetic[f'Allele {i}'] = np.nan
        if f'Size {i}' not in df_synthetic.columns:
            df_synthetic[f'Size {i}'] = np.nan
        if f'Height {i}' not in df_synthetic.columns:
            df_synthetic[f'Height {i}'] = np.nan
    
    # 重新排列列顺序
    cols = ['Sample File', 'Marker']
    for i in range(1, 101):
        cols.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
    
    df_synthetic = df_synthetic.reindex(columns=cols)
    
    # 保存合成数据
    output_path = "合成测试数据.csv"
    df_synthetic.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"合成测试数据已保存到: {output_path}")
    print(f"包含 {len(df_synthetic['Sample File'].unique())} 个样本")
    print(f"涵盖 {len(markers)} 个STR位点")
    
    return output_path

def main():
    """主函数：运行所有示例"""
    print("MGM-RF系统使用示例")
    print("基于Q1随机森林特征工程 + MGM-M基因型边缘化MCMC")
    print("="*80)
    
    # 检查是否有真实数据
    att2_path = "附件2：混合STR图谱数据.csv"
    
    if not os.path.exists(att2_path):
        print(f"没有找到附件2文件: {att2_path}")
        print("是否要创建合成测试数据？(y/n): ", end="")
        
        choice = input().strip().lower()
        if choice == 'y':
            synthetic_path = create_synthetic_test_data()
            print(f"\n请将合成数据重命名为: {att2_path}")
            print("然后重新运行此脚本")
            return
        else:
            print("程序退出")
            return
    
    # 运行示例
    try:
        # 示例1：单样本分析
        example_1_single_sample_analysis()
        
        # 询问是否继续
        print(f"\n是否继续运行其他示例？(y/n): ", end="")
        if input().strip().lower() != 'y':
            return
        
        # 示例2：批量分析
        example_2_batch_analysis()
        
        # 询问是否继续
        print(f"\n是否继续运行程序化示例？(y/n): ", end="")
        if input().strip().lower() != 'y':
            return
        
        # 示例3：程序化使用
        example_3_programmatic_usage()
        
        # 示例4：自定义参数
        print(f"\n是否演示自定义参数配置？(y/n): ", end="")
        if input().strip().lower() == 'y':
            example_4_custom_parameters()
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行")
    except Exception as e:
        print(f"\n示例运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("所有示例运行完成！")
    print("更多详细用法请参考 Q2_MGM_RF_Solution.py 和使用指南")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()