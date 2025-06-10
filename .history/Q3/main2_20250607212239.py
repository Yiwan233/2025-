# -*- coding: utf-8 -*-
"""
问题三使用示例脚本
展示如何使用Q3增强基因型推断系统
"""

import sys
import os
import numpy as np
import pandas as pd
from Q3_Enhanced_Genotype_Inference import analyze_single_sample_q3, analyze_all_samples_q3, Q3EnhancedPipeline

def example_1_single_sample_analysis():
    """示例1：单样本基因型推断"""
    print("\n" + "="*60)
    print("示例1：单样本基因型推断分析")
    print("="*60)
    
    # 配置文件路径
    att1_path = "附件1：不同人数的STR图谱数据.csv"
    att2_path = "附件2：混合STR图谱数据.csv"
    att3_path = "附件3：各个贡献者的基因型.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(att1_path) and not os.path.exists(att2_path):
        print(f"错误：找不到附件1或附件2文件")
        print("请确保数据文件在当前目录下")
        return
    
    # 首先查看可用的样本
    data_file = att2_path if os.path.exists(att2_path) else att1_path
    try:
        df_samples = pd.read_csv(data_file, encoding='utf-8')
        available_samples = df_samples['Sample File'].unique()
        print(f"可用的样本ID ({len(available_samples)}个):")
        for i, sample_id in enumerate(available_samples[:10], 1):
            print(f"  {i}. {sample_id}")
        if len(available_samples) > 10:
            print(f"  ... 还有{len(available_samples)-10}个样本")
        
        # 选择第一个样本进行分析
        if len(available_samples) > 0:
            target_sample = available_samples[0]
            print(f"\n选择样本 '{target_sample}' 进行基因型推断分析...")
            
            # 执行分析
            result = analyze_single_sample_q3(
                sample_id=target_sample,
                att1_or_att2_path=data_file,
                q1_model_path=q1_model_path if os.path.exists(q1_model_path) else None,
                att3_path=att3_path if os.path.exists(att3_path) else None
            )
            
            # 显示结果
            print_analysis_result_q3(result)
            
        else:
            print("没有找到有效样本")
            
    except Exception as e:
        print(f"分析过程出错: {e}")

def example_2_batch_analysis():
    """示例2：批量基因型推断（前3个样本）"""
    print("\n" + "="*60)
    print("示例2：批量基因型推断分析")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    att3_path = "附件3：各个贡献者的基因型.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    if not os.path.exists(att2_path):
        print(f"错误：找不到附件2文件 - {att2_path}")
        return
    
    try:
        # 批量分析前3个样本
        all_results = analyze_all_samples_q3(
            att1_or_att2_path=att2_path,
            q1_model_path=q1_model_path if os.path.exists(q1_model_path) else None,
            att3_path=att3_path if os.path.exists(att3_path) else None,
            max_samples=3  # 只分析前3个样本
        )
        
        print(f"\n批量分析完成，共分析了 {len(all_results)} 个样本")
        
        # 显示汇总结果
        print("\n汇总结果:")
        for sample_id, result in all_results.items():
            print(f"\n样本: {sample_id}")
            print_analysis_result_q3(result, brief=True)
            
    except Exception as e:
        print(f"批量分析过程出错: {e}")

def example_3_programmatic_usage():
    """示例3：程序化使用接口"""
    print("\n" + "="*60)
    print("示例3：程序化使用接口")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    att3_path = "附件3：各个贡献者的基因型.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    if not os.path.exists(att2_path):
        print(f"错误：找不到附件2文件 - {att2_path}")
        return
    
    try:
        # 初始化Q3增强流水线
        pipeline = Q3EnhancedPipeline(
            q1_model_path if os.path.exists(q1_model_path) else None,
            att3_path if os.path.exists(att3_path) else None
        )
        
        # 加载数据
        print("加载附件2数据...")
        pipeline.load_data(att2_path=att2_path)
        print("数据加载完成")
        
        # 读取样本数据
        df_samples = pd.read_csv(att2_path, encoding='utf-8')
        available_samples = df_samples['Sample File'].unique()
        
        if len(available_samples) > 0:
            sample_id = available_samples[0]
            sample_data = df_samples[df_samples['Sample File'] == sample_id]
            
            print(f"\n分析样本: {sample_id}")
            
            # 执行分析
            result = pipeline.analyze_single_sample(sample_id, sample_data)
            
            # 保存结果
            output_file = f"./example_q3_result_{sample_id}.json"
            pipeline.save_results(result, output_file)
            print(f"结果已保存到: {output_file}")
            
            # 绘制图表
            pipeline.plot_results(result, "./example_q3_plots")
            print("图表已保存到: ./example_q3_plots/")
            
            # 显示结果
            print_analysis_result_q3(result)
            
    except Exception as e:
        print(f"程序化使用过程出错: {e}")

def example_4_performance_analysis():
    """示例4：基因型推断性能分析"""
    print("\n" + "="*60)
    print("示例4：基因型推断性能分析")
    print("="*60)
    
    # 配置文件路径
    att2_path = "附件2：混合STR图谱数据.csv"
    att3_path = "附件3：各个贡献者的基因型.csv"
    q1_model_path = "noc_optimized_random_forest_model.pkl"
    
    # 检查必要文件
    if not all([os.path.exists(f) for f in [att2_path, att3_path]]):
        print("错误：缺少必要的数据文件")
        print("性能分析需要附件2和附件3")
        return
    
    try:
        # 初始化流水线
        pipeline = Q3EnhancedPipeline(
            q1_model_path if os.path.exists(q1_model_path) else None,
            att3_path
        )
        
        # 加载数据
        pipeline.load_data(att2_path=att2_path)
        
        # 获取有真实基因型的样本
        available_samples = pipeline.att3_processor.get_available_samples()
        print(f"有真实基因型的样本数: {len(available_samples)}")
        
        if len(available_samples) > 0:
            # 分析第一个有真实基因型的样本
            target_sample = available_samples[0]
            print(f"\n分析样本: {target_sample}")
            
            # 读取样本数据
            df_samples = pd.read_csv(att2_path, encoding='utf-8')
            sample_data = df_samples[df_samples['Sample File'] == target_sample]
            
            if not sample_data.empty:
                # 执行完整分析
                result = pipeline.analyze_single_sample(target_sample, sample_data)
                
                # 详细显示性能评估结果
                if result['evaluation_results']:
                    print_detailed_performance_analysis(result['evaluation_results'])
                
                # 显示基因型推断详情
                print_genotype_inference_details(result)
                
            else:
                print(f"样本 {target_sample} 在附件2中未找到对应数据")
        
    except Exception as e:
        print(f"性能分析过程出错: {e}")

def print_analysis_result_q3(result: Dict, brief=False):
    """打印Q3分析结果"""
    print(f"--- 基因型推断分析结果 ---")
    print(f"样本ID: {result['sample_id']}")
    print(f"预测NoC: {result['predicted_noc']} (置信度: {result['noc_confidence']:.3f})")
    print(f"计算时间: {result['computation_time']:.1f}秒")
    print(f"观测位点数: {len(result['observed_loci'])}")
    
    if result['mcmc_results'] is not None:
        mcmc_results = result['mcmc_results']
        print(f"MCMC混合比例接受率: {mcmc_results['acceptance_rate_mx']:.3f}")
        print(f"MCMC基因型接受率: {mcmc_results['acceptance_rate_gt']:.3f}")
        print(f"MCMC收敛: {'是' if mcmc_results['converged'] else '否'}")
        print(f"有效样本数: {mcmc_results['n_samples']}")
        
        if not brief:
            # 显示基因型推断摘要
            posterior_summary = result['posterior_summary']
            if posterior_summary and result['observed_loci']:
                print(f"\n基因型推断结果摘要:")
                
                # 显示前两个位点的结果
                for locus in result['observed_loci'][:2]:
                    if locus in posterior_summary:
                        print(f"\n位点 {locus}:")
                        locus_summary = posterior_summary[locus]
                        
                        for contributor, contrib_data in locus_summary.items():
                            if 'mode_genotype' in contrib_data:
                                mode_gt = contrib_data['mode_genotype']
                                mode_prob = contrib_data['mode_probability']
                                print(f"  {contributor}: {mode_gt[0]},{mode_gt[1]} "
                                      f"(概率: {mode_prob:.3f})")
            
            # 显示性能评估结果
            if result['evaluation_results'] and 'overall_summary' in result['evaluation_results']:
                eval_summary = result['evaluation_results']['overall_summary']
                print(f"\n性能评估结果:")
                
                if 'genotype_concordance_rate' in eval_summary:
                    gcr = eval_summary['genotype_concordance_rate']
                    grade = eval_summary.get('performance_grade', 'N/A')
                    print(f"  基因型一致性率: {gcr:.3f}")
                    print(f"  性能等级: {grade}")
                
                if 'allele_concordance_rate' in eval_summary:
                    acr = eval_summary['allele_concordance_rate']
                    print(f"  等位基因一致性率: {acr:.3f}")
    else:
        print("⚠️  无有效峰数据，MCMC未执行")

def print_detailed_performance_analysis(evaluation_results: Dict):
    """打印详细的性能分析结果"""
    print(f"\n{'='*50}")
    print("详细性能分析结果")
    print(f"{'='*50}")
    
    # 总体性能摘要
    if 'overall_summary' in evaluation_results:
        summary = evaluation_results['overall_summary']
        print(f"\n📊 总体性能指标:")
        
        if 'genotype_concordance_rate' in summary:
            gcr = summary['genotype_concordance_rate']
            grade = summary.get('performance_grade', 'N/A')
            print(f"   基因型一致性率: {gcr:.4f} ({gcr*100:.2f}%)")
            print(f"   性能等级: {grade}")
        
        if 'allele_concordance_rate' in summary:
            acr = summary['allele_concordance_rate']
            print(f"   等位基因一致性率: {acr:.4f} ({acr*100:.2f}%)")
        
        if 'avg_locus_exact_match_rate' in summary:
            avg_locus = summary['avg_locus_exact_match_rate']
            print(f"   平均位点精确匹配率: {avg_locus:.4f}")
        
        if 'avg_contributor_exact_match_rate' in summary:
            avg_contrib = summary['avg_contributor_exact_match_rate']
            print(f"   平均贡献者精确匹配率: {avg_contrib:.4f}")
    
    # 位点特异性性能
    if 'locus_specific' in evaluation_results:
        locus_perf = evaluation_results['locus_specific']
        print(f"\n📈 位点特异性性能:")
        print(f"{'位点':<12} {'精确匹配率':<12} {'部分匹配率':<12} {'错误匹配率':<12}")
        print("-" * 50)
        
        for locus, perf in locus_perf.items():
            if 'exact_match_rate' in perf:
                exact_rate = perf['exact_match_rate']
                partial_rate = perf.get('partial_match_rate', 0)
                mismatch_rate = perf.get('mismatch_rate', 0)
                print(f"{locus:<12} {exact_rate:<12.3f} {partial_rate:<12.3f} {mismatch_rate:<12.3f}")
    
    # 贡献者特异性性能
    if 'contributor_specific' in evaluation_results:
        contrib_perf = evaluation_results['contributor_specific']
        print(f"\n👥 贡献者特异性性能:")
        print(f"{'贡献者':<12} {'精确匹配率':<12} {'部分匹配率':<12} {'错误匹配率':<12}")
        print("-" * 50)
        
        for contributor, perf in contrib_perf.items():
            if 'exact_match_rate' in perf:
                exact_rate = perf['exact_match_rate']
                partial_rate = perf.get('partial_match_rate', 0)
                mismatch_rate = perf.get('mismatch_rate', 0)
                print(f"{contributor:<12} {exact_rate:<12.3f} {partial_rate:<12.3f} {mismatch_rate:<12.3f}")

def print_genotype_inference_details(result: Dict):
    """打印基因型推断详细信息"""
    print(f"\n{'='*50}")
    print("基因型推断详细信息")
    print(f"{'='*50}")
    
    posterior_summary = result['posterior_summary']
    observed_loci = result['observed_loci']
    
    if not posterior_summary or not observed_loci:
        print("无基因型推断结果")
        return
    
    print(f"\n推断的基因型分布:")
    
    for locus in observed_loci:
        if locus in posterior_summary and locus != 'mcmc_quality':
            print(f"\n🧬 位点 {locus}:")
            locus_summary = posterior_summary[locus]
            
            for contributor, contrib_data in locus_summary.items():
                if 'mode_genotype' not in contrib_data:
                    continue
                
                mode_gt = contrib_data['mode_genotype']
                mode_prob = contrib_data['mode_probability']
                total_samples = contrib_data.get('total_samples', 0)
                
                print(f"  {contributor}:")
                print(f"    最可能基因型: {mode_gt[0]},{mode_gt[1]} (概率: {mode_prob:.3f})")
                print(f"    样本数: {total_samples}")
                
                # 显示95%可信集合
                if 'credible_set_95' in contrib_data:
                    credible_set = contrib_data['credible_set_95']
                    print(f"    95%可信集合:")
                    for gt, prob in credible_set[:3]:  # 只显示前3个
                        print(f"      {gt[0]},{gt[1]}: {prob:.3f}")
                    if len(credible_set) > 3:
                        print(f"      ... 还有{len(credible_set)-3}个基因型")

def create_synthetic_test_data_q3():
    """创建Q3的合成测试数据"""
    print("\n" + "="*60)
    print("创建Q3合成测试数据")
    print("="*60)
    
    np.random.seed(42)
    
    # 创建附件2样式的混合STR数据
    sample_data = []
    markers = ['D3S1358', 'vWA', 'FGA', 'D8S1179', 'D21S11']
    
    for i, sample_id in enumerate([f"Mix_Sample_{j+1}" for j in range(3)], 1):
        for marker in markers:
            # 模拟2-3人混合样本
            n_alleles = np.random.randint(3, 6)  # 每个位点3-5个等位基因
            
            for allele_idx in range(n_alleles):
                allele_num = np.random.randint(12, 20)
                allele = str(allele_num) if np.random.random() > 0.2 else f"{allele_num}.3"
                
                size = 120 + allele_num * 4 + np.random.normal(0, 1)
                height = np.random.lognormal(np.log(800), 0.6)
                
                sample_data.append({
                    'Sample File': sample_id,
                    'Marker': marker,
                    f'Allele {allele_idx+1}': allele,
                    f'Size {allele_idx+1}': size,
                    f'Height {allele_idx+1}': height
                })
    
    # 填充空列
    df_synthetic_att2 = pd.DataFrame(sample_data)
    
    # 添加空的Allele/Size/Height列
    for i in range(1, 101):
        if f'Allele {i}' not in df_synthetic_att2.columns:
            df_synthetic_att2[f'Allele {i}'] = np.nan
        if f'Size {i}' not in df_synthetic_att2.columns:
            df_synthetic_att2[f'Size {i}'] = np.nan
        if f'Height {i}' not in df_synthetic_att2.columns:
            df_synthetic_att2[f'Height {i}'] = np.nan
    
    # 重新排列列顺序
    cols = ['Sample File', 'Marker']
    for i in range(1, 101):
        cols.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
    
    df_synthetic_att2 = df_synthetic_att2.reindex(columns=cols)
    
    # 保存合成附件2
    att2_path = "合成附件2.csv"
    df_synthetic_att2.to_csv(att2_path, index=False, encoding='utf-8-sig')
    
    # 创建对应的附件3（真实基因型）
    genotype_data = []
    for sample_id in [f"Mix_Sample_{j+1}" for j in range(3)]:
        n_contributors = np.random.randint(2, 4)  # 2-3个贡献者
        
        for contributor_id in range(1, n_contributors + 1):
            for marker in markers:
                # 生成真实基因型
                allele1 = str(np.random.randint(12, 20))
                allele2 = str(np.random.randint(12, 20))
                
                genotype_data.append({
                    'Sample': sample_id,
                    'Contributor': f'P{contributor_id}',
                    'Marker': marker,
                    'Allele 1': allele1,
                    'Allele 2': allele2
                })
    
    df_synthetic_att3 = pd.DataFrame(genotype_data)
    att3_path = "合成附件3.csv"
    df_synthetic_att3.to_csv(att3_path, index=False, encoding='utf-8-sig')
    
    print(f"合成附件2已保存到: {att2_path}")
    print(f"合成附件3已保存到: {att3_path}")
    print(f"包含 {len(df_synthetic_att2['Sample File'].unique())} 个混合样本")
    print(f"涵盖 {len(markers)} 个STR位点")
    print(f"包含 {len(df_synthetic_att3)} 个真实基因型记录")
    
    return att2_path, att3_path

def main():
    """主函数：运行所有示例"""
    print("Q3增强基因型推断系统使用示例")
    print("基于Q1随机森林特征工程 + Q2 MGM-RF方法 + 基因型匹配评估")
    print("="*80)
    
    # 检查是否有真实数据
    att2_path = "附件2：混合STR图谱数据.csv"
    att3_path = "附件3：各个贡献者的基因型.csv"
    
    if not os.path.exists(att2_path) or not os.path.exists(att3_path):
        print(f"没有找到必要文件:")
        if not os.path.exists(att2_path):
            print(f"  缺少: {att2_path}")
        if not os.path.exists(att3_path):
            print(f"  缺少: {att3_path}")
        
        print("是否要创建合成测试数据？(y/n): ", end="")
        
        choice = input().strip().lower()
        if choice == 'y':
            synthetic_att2, synthetic_att3 = create_synthetic_test_data_q3()
            print(f"\n请将合成数据重命名为:")
            print(f"  {synthetic_att2} -> {att2_path}")
            print(f"  {synthetic_att3} -> {att3_path}")
            print("然后重新运行此脚本")
            return
        else:
            print("程序退出")
            return
    
    # 运行示例
    try:
        # 示例1：单样本基因型推断
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
        
        # 示例4：性能分析
        print(f"\n是否演示性能分析？(y/n): ", end="")
        if input().strip().lower() == 'y':
            example_4_performance_analysis()
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行")
    except Exception as e:
        print(f"\n示例运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("所有示例运行完成！")
    print("更多详细用法请参考 Q3_Enhanced_Genotype_Inference.py")
    print("主要特点:")
    print("• 集成Q1的RFECV特征选择和随机森林NoC预测")
    print("• 采用Q2的MGM-M基因型边缘化MCMC方法")
    print("• 结合附件3的已知基因型信息进行推断")
    print("• 提供详细的基因型匹配准确性评估")
    print("• 支持多链MCMC收敛性诊断")
    print("• 生成可视化图表和性能报告")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()