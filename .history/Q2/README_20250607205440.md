# 问题二：混合STR图谱贡献者比例推断系统使用指南

## 系统概述

本系统（MGM-RF）专门针对D题问题二设计，实现了基于随机森林特征工程和MGM-M基因型边缘化MCMC方法的混合STR图谱贡献者比例推断。

### 核心创新
1. **继承Q1算法优势**：采用Q1c2.py中的RFECV特征选择和随机森林优化
2. **MGM-M基因型边缘化**：避免高维基因型空间直接采样，提升计算效率
3. **V5特征驱动**：基于V5特征集自适应调整模型参数
4. **一体化流水线**：从NoC预测到混合比例推断的完整解决方案

## 系统架构

```
输入数据 → Q1特征工程 → NoC预测 → MGM-M推断 → 结果分析
    ↓           ↓          ↓         ↓          ↓
附件2数据   RFECV特征选择  随机森林   基因型边缘化  后验分布
           V5特征提取     分类器     MCMC采样    可视化
```

## 安装与配置

### 环境要求
```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib
pip install logging warnings time collections itertools
```

### 文件结构
```
项目目录/
├── Q2_MGM_RF_Solution.py          # 主程序
├── 附件2：混合STR图谱数据.csv       # 输入数据
├── noc_optimized_random_forest_model.pkl  # Q1训练的模型（可选）
└── q2_mgm_rf_results/             # 输出目录（自动创建）
    ├── 样本ID_result.json          # 单样本结果
    ├── 样本ID_mcmc_trace.png       # MCMC轨迹图
    ├── 样本ID_posterior_dist.png   # 后验分布图
    ├── 样本ID_joint_posterior.png  # 联合分布图（2人情况）
    └── batch_analysis_summary.csv   # 批量分析摘要
```

## 使用方法

### 1. 基本使用
```python
python Q2_MGM_RF_Solution.py
```

运行后按提示选择：
- **选项1**：分析单个样本
- **选项2**：批量分析所有样本
- **选项3**：批量分析前N个样本

### 2. 单样本分析
```python
from Q2_MGM_RF_Solution import analyze_single_sample_from_att2

# 分析指定样本
result = analyze_single_sample_from_att2(
    sample_id="your_sample_id",
    att2_path="附件2：混合STR图谱数据.csv",
    q1_model_path="noc_optimized_random_forest_model.pkl"  # 可选
)

# 查看结果
print(f"预测NoC: {result['predicted_noc']}")
print(f"混合比例: {result['posterior_summary']}")
```

### 3. 批量分析
```python
from Q2_MGM_RF_Solution import analyze_all_samples_from_att2

# 批量分析
all_results = analyze_all_samples_from_att2(
    att2_path="附件2：混合STR图谱数据.csv",
    q1_model_path="noc_optimized_random_forest_model.pkl",  # 可选
    max_samples=10  # 可选，限制分析样本数
)
```

### 4. 程序化接口
```python
from Q2_MGM_RF_Solution import MGM_RF_Pipeline

# 初始化流水线
pipeline = MGM_RF_Pipeline("noc_optimized_random_forest_model.pkl")

# 加载数据
att2_data = pipeline.load_attachment2_data("附件2：混合STR图谱数据.csv")
att2_freq_data = pipeline.prepare_att2_frequency_data(att2_data)

# 分析单个样本
sample_data = att2_data["sample_id"]
result = pipeline.analyze_sample(sample_data, att2_freq_data)

# 绘制图表
pipeline.plot_results(result, "./output_dir")

# 保存结果
pipeline.save_results(result, "./sample_result.json")
```

## 输出结果解读

### 1. 基本信息
```json
{
  "sample_file": "样本ID",
  "predicted_noc": 2,              // 预测的贡献者人数
  "noc_confidence": 0.856,         // NoC预测置信度
  "computation_time": 45.3         // 计算耗时（秒）
}
```

### 2. 混合比例后验统计
```json
{
  "Mx_1": {
    "mean": 0.673,                 // 后验均值
    "std": 0.082,                  // 后验标准差
    "median": 0.681,               // 后验中位数
    "mode": 0.695,                 // 后验众数
    "credible_interval_95": [0.518, 0.828],  // 95%置信区间
    "hpdi_95": [0.525, 0.835]      // 95%最高后验密度区间
  },
  "Mx_2": {
    // 类似结构
  }
}
```

### 3. MCMC质量评估
```json
{
  "mcmc_quality": {
    "acceptance_rate": 0.423,       // MCMC接受率（0.2-0.6为佳）
    "n_samples": 2000,             // 有效样本数
    "converged": true              // 是否收敛
  }
}
```

### 4. 收敛性诊断
```json
{
  "convergence_diagnostics": {
    "convergence_status": "Good",   // 收敛状态
    "min_ess": 127.5,              // 最小有效样本量
    "max_geweke": 1.23,            // 最大Geweke得分
    "convergence_issues": []        // 收敛问题列表
  }
}
```

## 参数配置

### 核心参数设置
```python
# 在代码顶部的Config类中修改参数
class Config:
    # STR分析参数
    HEIGHT_THRESHOLD = 50           # 峰高阈值
    SATURATION_THRESHOLD = 30000    # 饱和阈值
    CTA_THRESHOLD = 0.5            # CTA阈值
    
    # MCMC参数
    N_ITERATIONS = 15000           # MCMC迭代次数
    N_WARMUP = 5000               # 预热次数
    N_CHAINS = 4                  # 链数
    THINNING = 5                  # 抽样间隔
    K_TOP = 800                   # K-top采样数量
```

### 算法选择策略
- **N ≤ 3**：完全枚举所有基因型组合
- **N ≥ 4**：使用K-top采样策略，平衡精度与效率
- **自适应步长**：MCMC过程中动态调整提议分布

## 结果可视化

系统自动生成以下图表：

### 1. MCMC轨迹图
- **文件名**：`{样本ID}_mcmc_trace.png`
- **内容**：每个混合比例参数的MCMC采样轨迹
- **用途**：检查MCMC收敛性和混合性

### 2. 后验分布图
- **文件名**：`{样本ID}_posterior_dist.png`
- **内容**：各混合比例的后验概率密度分布
- **用途**：查看参数估计的不确定性

### 3. 联合后验分布图（2人情况）
- **文件名**：`{样本ID}_joint_posterior.png`
- **内容**：两个混合比例的联合分布散点图
- **用途**：观察参数间的相关性

## 性能评估与优化

### 计算复杂度
- **特征提取**：O(P×L) - P为峰数，L为位点数
- **NoC预测**：O(1) - 基于预训练模型
- **基因型枚举**：
  - N ≤ 3：O(|A|^(2N)) - A为等位基因集合
  - N ≥ 4：O(K_TOP) - 固定采样数量
- **MCMC采样**：O(I×K×L) - I为迭代次数，K为有效基因型数，L为位点数

### 性能优化建议
1. **调整K_TOP参数**：N≥4时平衡精度与速度
2. **减少MCMC迭代**：对于快速预估可减少到5000次
3. **并行处理**：批量分析时可考虑多进程
4. **内存优化**：大数据集时可启用样本缓存

## 常见问题与解决方案

### Q1: NoC预测不准确怎么办？
**A1**: 
- 检查Q1模型文件是否正确加载
- 验证输入数据格式是否与训练数据一致
- 可手动指定NoC进行混合比例推断

### Q2: MCMC不收敛怎么办？
**A2**:
- 增加迭代次数（N_ITERATIONS）
- 调整步长（初始值0.05）
- 检查数据质量，确保有足够的信息量
- 对于复杂样本，考虑增加K_TOP值

### Q3: 计算时间过长怎么办？
**A3**:
- 减少MCMC迭代次数和预热次数
- 对于N≥4的情况，降低K_TOP值
- 使用更少的位点进行分析
- 启用多核并行计算

### Q4: 内存不足怎么办？
**A4**:
- 减小批量分析的样本数量
- 降低MCMC样本保存频率（增大THINNING）
- 清理不必要的中间结果

### Q5: 结果可信度评估？
**A5**:
- 检查MCMC收敛性指标（ESS > 100, Geweke < 2）
- 观察95%置信区间宽度
- 对比多次运行结果的一致性
- 验证生物学合理性

## 方法学优势

### 1. 理论突破
- **完全边缘化**：避免传统方法的维度爆炸问题
- **数值稳定**：logsumexp技巧确保计算稳定性
- **现实先验**：基于群体遗传学的伪频率先验

### 2. 工程实现
- **特征驱动**：V5特征自适应调整模型参数
- **多策略融合**：枚举+采样的混合策略
- **质量控制**：完整的收敛性诊断体系

### 3. 实用价值
- **高精度**：特别适合3人以上复杂混合样本
- **高效率**：相比传统方法显著降低计算需求
- **高可靠**：完整的不确定性量化和质量评估

## 引用与参考

如果在学术工作中使用本系统，请引用：

```
MGM-RF: 基于随机森林特征工程和基因型边缘化MCMC的混合STR图谱贡献者比例推断方法
- 特征工程：基于Q1c2.py的RFECV特征选择和随机森林优化
- 推断算法：MGM-M基因型边缘化马尔可夫链蒙特卡罗方法
- 应用领域：法医DNA分析、混合样本解析、身份鉴定
```

## 技术支持

### 日志调试
系统使用Python logging模块，设置日志级别查看详细信息：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 错误排查
1. **数据格式错误**：检查CSV文件编码和列名
2. **模型加载失败**：验证Q1模型文件完整性
3. **内存溢出**：减少批量处理规模
4. **计算超时**：调整MCMC参数或使用更快硬件

### 性能监控
```python
# 启用性能分析
import cProfile
cProfile.run('analyze_single_sample_from_att2(sample_id, att2_path)')
```

## 版本信息

- **当前版本**：V1.0
- **发布日期**：2025-06-07
- **兼容性**：Python 3.8+
- **依赖库版本**：见requirements.txt

## 更新日志

### V1.0 (2025-06-07)
- 初始版本发布
- 集成Q1的RFECV特征选择和随机森林算法
- 实现MGM-M基因型边缘化MCMC推断
- 支持单样本和批量分析模式
- 完整的结果可视化和质量评估体系

---

**注意**：本系统专为D题问题二设计，充分利用了Q1的算法优势，实现了从NoC预测到混合比例推断的完整解决方案。使用时请确保数据格式正确，并根据实际需求调整参数配置。