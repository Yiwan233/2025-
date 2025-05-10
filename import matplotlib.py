import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("INFO: Matplotlib 中文字体尝试设置为 'SimHei'.")
except Exception as e_font:
    print(f"警告: 设置 Matplotlib 中文字体 'SimHei' 失败: {e_font}. 图表中文可能无法显示。")
# 团队成员与颜色设置
members = ["A（建模手）", "B（编程手）", "C（论文手）", "D（辅助手）"]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# 时间段定义（任务、开始时间、持续天数、成员）
tasks = [
    # A 建模手任务
    ("拆题+模型规划", "2025-05-07", 6, "A（建模手）"),
    ("模型1、2设计与分析", "2025-05-13", 7, "A（建模手）"),
    ("模型3、4搭建与整合", "2025-05-20", 7, "A（建模手）"),
    ("模型评估与统一结构", "2025-05-27", 7, "A（建模手）"),
    ("统稿与论文校审", "2025-06-03", 5, "A（建模手）"),
    ("最终检查与提交", "2025-06-08", 3, "A（建模手）"),

    # B 编程手任务
    ("数据处理+图谱识别", "2025-05-07", 6, "B（编程手）"),
    ("模型1、2实现", "2025-05-13", 7, "B（编程手）"),
    ("模型3+去噪实现", "2025-05-20", 7, "B（编程手）"),
    ("调试+图表生成", "2025-05-27", 7, "B（编程手）"),
    ("输出整合与备份", "2025-06-03", 7, "B（编程手）"),

    # C 论文手任务
    ("搭建论文框架", "2025-05-07", 6, "C（论文手）"),
    ("撰写前两节", "2025-05-13", 7, "C（论文手）"),
    ("撰写中段内容", "2025-05-20", 7, "C（论文手）"),
    ("撰写实验与分析", "2025-05-27", 7, "C（论文手）"),
    ("定稿润色与格式调整", "2025-06-03", 7, "C（论文手）"),

    # D 辅助手任务
    ("查STR文献+背景学习", "2025-05-07", 6, "D（辅助手）"),
    ("查找案例与整理数据", "2025-05-13", 14, "D（辅助手）"),
    ("论文协助与图表支持", "2025-05-27", 14, "D（辅助手）"),
    ("格式检查与文档核对", "2025-06-08", 3, "D（辅助手）")
]

# 绘图准备
fig, ax = plt.subplots(figsize=(12, 6))

# 日期处理
base_date = datetime.strptime("2025-05-07", "%Y-%m-%d")
for task, start_str, duration, member in tasks:
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=duration)
    ax.barh(member, (end_date - start_date).days, left=start_date, color=colors[members.index(member)], edgecolor="black")
    ax.text(start_date + timedelta(days=0.5), members.index(member), task, va='center', ha='left', fontsize=8, color='white')

# 美化
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_xlim(datetime(2025, 5, 6), datetime(2025, 6, 11))
ax.set_title("全国大学生建模竞赛甘特图（2025）", fontsize=14)
ax.set_xlabel("日期")
ax.set_ylabel("成员")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()
