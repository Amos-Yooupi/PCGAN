import torch
import matplotlib.pyplot as plt


def show_bar(data):
    # 将数据从 GPU 移到 CPU，并转换为 numpy 数组
    data = data.cpu().numpy().flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')  # 设置30个区间
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tensor Values")
    plt.grid(axis='y', alpha=0.75)  # 增加网格线，提升可读性
    plt.show()
