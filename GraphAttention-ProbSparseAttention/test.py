import numpy as np
import matplotlib.pyplot as plt

# 字体设置调整为24
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 24,  # 基础字体大小
    'axes.labelsize': 24,  # 坐标轴标签字体大小
    'axes.titlesize': 26,  # 标题字体稍大于大一点
    'legend.fontsize': 24,  # 图例字体大小
    'xtick.labelsize': 24,  # x轴刻度字体大小
    'ytick.labelsize': 24,  # y轴刻度字体大小
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.color': 'gray',
    'grid.alpha': 0.7,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.labelspacing': 0.5
})

# 数据准备
spans = [3, 4, 5, 6, 7, 9, 10, 11]
gat_layers = [1, 2, 3]

mae_data = [
    [0.47232, 0.17825, 0.17657, 0.28244, 0.24806, 0.21494, 0.36101, 0.4029],
    [0.45965, 0.17841, 0.1477, 0.28113, 0.25716, 0.21428, 0.3237, 0.33889],
    [0.4636, 0.19059, 0.14435, 0.23843, 0.24362, 0.20633, 0.30732, 0.31925]
]

rmse_data = [
    [0.63975, 0.27204, 0.25346, 0.39906, 0.41124, 0.34842, 0.62445, 0.72949],
    [0.60625, 0.27397, 0.2229, 0.40685, 0.39374, 0.34466, 0.53335, 0.58096],
    [0.63806, 0.2833, 0.20684, 0.34278, 0.38567, 0.34491, 0.49158, 0.53118]
]

# 绘图配置（适应更大字体）
bar_width = 0.20  # 保持窄柱子
x = np.arange(len(spans)) * 0.7  # 适当调整组间距离
colors = ['#3498db', '#e74c3c', '#2ecc71']
hatches = ['/', '\\', '|']
fig_size = (16, 8)  # 增大画布以适应更大字体

# 1. MAE 图
fig, ax = plt.subplots(figsize=fig_size)

for i, layers in enumerate(gat_layers):
    ax.bar(x + i * bar_width, mae_data[i], width=bar_width,
           label=f'GAT Layers = {layers}',
           color=colors[i], edgecolor='black', linewidth=0.8,
           hatch=hatches[i], alpha=0.8)

ax.set_xticks(x + bar_width * (len(gat_layers) - 1) / 2)
ax.set_xticklabels(spans, fontsize=24)
ax.set_xlabel('Bridge Span', fontsize=24)
ax.set_ylabel('MAE', fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_title('Impact of GAT Layers on MAE Across Different Bridge Spans', pad=20)
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max([max(vals) for vals in mae_data]) * 1.1)
plt.tight_layout()
plt.savefig("MAE.png", dpi=300)

# 2. RMSE 图
fig, ax = plt.subplots(figsize=fig_size)

for i, layers in enumerate(gat_layers):
    ax.bar(x + i * bar_width, rmse_data[i], width=bar_width,
           label=f'GAT Layers = {layers}',
           color=colors[i], edgecolor='black', linewidth=0.8,
           hatch=hatches[i], alpha=0.8)

ax.set_xticks(x + bar_width * (len(gat_layers) - 1) / 2)
ax.set_xticklabels(spans, fontsize=24)
ax.set_xlabel('Bridge Span', fontsize=24)
ax.set_ylabel('RMSE', fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_title('Impact of GAT Layers on RMSE Across Different Bridge Spans', pad=20)
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max([max(vals) for vals in rmse_data]) * 1.1)
plt.tight_layout()
plt.savefig("RMSE.png", dpi=300)
plt.show()
