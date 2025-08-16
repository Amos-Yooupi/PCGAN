import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
import torch.nn as nn


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


def show_tensor_data(tensor_data, y_labels=None, time_interval=1., freq=1, point=False, compare=True, title=None):
    tensor_data = tensor_data.detach().cpu().numpy()
    assert tensor_data.shape.__len__() == 2, "tensor data should be (N, time_steps)"
    N, L = tensor_data.shape

    a = tensor_data.copy()
    tensor_data = tensor_data[:, ::freq]
    print(f"Freq: {freq}, Total Length: {tensor_data.shape[-1]}")
    fig, axes = plt.subplots(N, 1)
    if N == 1:
        axes = [axes]
    if y_labels is None:
        y_labels = ["Response Value"]
    if title is not None and isinstance(title, list):
        title_list = title
    else:
        title_list = ["Acc y", "Acc z", "Dis y", "Dis z"]

    t = np.arange(0, L, dtype=np.float32)[::freq] * time_interval
    t_a = np.arange(0, L, dtype=np.float32) * time_interval

    for i in range(N):
        if i == N-1:
            axes[i].set_xlabel("Time(s)", fontsize=26)
        if i == 0:
            axes[i].set_title("Earthquake acceleration", fontsize=30)
        axes[i].set_ylabel(y_labels[i % len(y_labels)], fontsize=26)
        if compare:
            axes[i].plot(t_a, a[i, :], label=f'True-1-{title_list[i % len(title_list)]}')
        if point:
            axes[i].scatter(t, tensor_data[i, :], label=f'label-{i + 1}', c='r')
        axes[i].plot(t, tensor_data[i, :], label=f'True-{freq}-{title_list[i % len(title_list)]}', c='r', linewidth=2)
        axes[i].grid(False)
        axes[i].legend(fontsize=16)

        # Setting the tick parameters
        axes[i].tick_params(direction='in', length=6, width=2, labelsize=20)
        # Increasing tick density
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(10))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def show_predict(true, predict, time_interval=0.01, show_point=False, n_span="default", model_name="default_name"):
    assert true.shape.__len__() == predict.shape.__len__() == 3, "The size should be (N, D, L)"
    assert true.shape == predict.shape, "The shape of predict should be the same as true"
    true = true.detach().cpu().numpy()
    predict = predict.detach().cpu().numpy()
    N, D, L = true.shape
    t = np.arange(L) * time_interval
    title_list = ["Acc y (m²/s)", "Acc z (m²/s)", "Dis y (m)", "Dis z (m)"]

    # Increase subplot size and adjust layout for better visibility
    fig, axes = plt.subplots(N, D, figsize=(18, 10), squeeze=False)  # Adjusted size here

    for r in range(N):
        for c in range(D):
            ax = axes[r, c]
            ax.plot(t, true[r, c, :], label=f"True-{c}")
            ax.plot(t, predict[r, c, :], label=f"Predict-{c}", color='red')
            ax.set_ylabel(title_list[c % len(title_list)], fontsize=20)  # Adjusted font size

            # Set y-axis to use scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Set scientific limits

            if show_point:
                ax.scatter(t, predict[r, c, :], label=f"Predict Point-{c}", color='yellow', s=5)

            ax.set_xlabel("Time (s)", fontsize=20)  # Adjusted font size
            ax.legend(fontsize=12, framealpha=1)  # Smaller font size for legend
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(8))
            ax.tick_params(axis='both', which='major', labelsize=20)  # Adjusted tick params
            ax.grid(True)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.6, hspace=0.3)
    fig.suptitle("The response of high-speed train with three carriages", fontsize=32, fontweight="bold")
    # fig.suptitle(f"{model_name}", fontsize=32, fontweight="bold")
    plt.savefig(fr"Figures\{model_name}_{n_span}.eps")
    print(f"Figures_{n_span}.eps", "save successfully!")
    plt.show()


def show_sample_predict(true, predict_sample, time_interval=0.01, show_point=False, n_span="None"):
    assert predict_sample.shape.__len__() == 4, "The size should be (N, D, n_sample, L)"
    max_predict = torch.max(predict_sample, dim=-2)[0].detach().cpu().numpy()
    min_predict = torch.min(predict_sample, dim=-2)[0].detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    predict = predict_sample.detach().cpu().numpy()
    N, D, L = true.shape
    t = np.arange(L) * time_interval
    title_list = ["Acc y (m²/s)", "Acc z (m²/s)", "Dis y (m)", "Dis z (m)"]

    # Increase subplot size and adjust layout for better visibility
    fig, axes = plt.subplots(N, D, figsize=(16, 10), squeeze=False)  # Adjusted size here

    n_sample = predict_sample.shape[-2]

    for r in range(N):
        for c in range(D):
            ax = axes[r, c]
            ax.plot(t, true[r, c, :], label=f"True-{c}")
            for n in range(n_sample):
                if n == n_sample - 1:
                    # ax.plot(t, predict[r, c, n, :], label=f"Predict-{c}", color='red')
                    ax.fill_between(t, max_predict[r, c, :], min_predict[r, c, :], color='lightgreen', alpha=0.2, label=f"Predict-{c}")

                else:
                    # ax.plot(t, predict[r, c, n, :])
                    ax.fill_between(t, max_predict[r, c, :], min_predict[r, c, :], color='lightgreen', alpha=0.2)

            ax.set_ylabel(title_list[c % len(title_list)], fontsize=20)  # Adjusted font size
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Set scientific limits

            if show_point:
                ax.scatter(t, predict[r, c, :], label=f"Predict Point-{c}", color='yellow', s=5)
            ax.set_xlabel("Time (s)", fontsize=20)  # Adjusted font size
            ax.legend(fontsize=12, framealpha=1)  # Smaller font size for legend
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(8))
            ax.tick_params(axis='both', which='major', labelsize=20)  # Adjusted tick params
            # ax.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.6, hspace=0.3)
    # fig.tight_layout(pad=0.2)  # Adjusted padding
    fig.suptitle("Randomness of the Latent Layer", fontsize=32, fontweight="bold")
    plt.savefig(fr"Figures\latent_{n_span}.eps")
    plt.show()


def show_compare(span, time_interval=0.01, show_point=False, show=True):
    plt.rcParams['font.size'] = 24

    true = torch.load(os.path.join('predict_data', f'true-{span}.pt'))
    N, D, L = true.shape
    t = np.arange(L) * time_interval

    title_list = ["Acc y (m²/s)", "Acc z (m²/s)", "Dis y (m)", "Dis z (m)"]
    model_name = ["LSTM", "GNN", "GAT", "GAT_LSTM", "GATtention"]
    mae = torch.zeros(len(model_name))
    predict_list = []
    temp = []
    for i, name in enumerate(model_name):
        path = os.path.join('predict_data', f'{name}-{span}.pt')
        if os.path.exists(path):
            print(f'{name}', "load successfully")
            predict_list.append(torch.load(path))
            temp.append(name)
            mae[i] = torch.mean(torch.abs(true - predict_list[-1]), dim=-1).sum()
        else:
            print(name, "is not exist")
    model_name = temp

    mae, sorted_idx = torch.sort(mae, descending=False)
    for i, mae_loss in enumerate(mae):
        print(f"{i + 1}、", model_name[sorted_idx[i]], "--", mae_loss.item())

    fig, axes = plt.subplots(N, D, figsize=(30, 20), squeeze=False)

    plt.rcParams['axes.formatter.limits'] = (-3, 3)  # Adjust the range for using scientific notation

    if show:
        for r in range(N):
            for c in range(D):
                ax = axes[r, c]
                ax.plot(t, true[r, c, :], label="True")
                ax.set_ylabel(title_list[c % len(title_list)], fontsize=24)  # Adjusted font size
                for i, predict in enumerate(predict_list):
                    ax.plot(t, predict[r, c, :], label=f"{model_name[i]}")
                if show_point:
                    for i, predict in enumerate(predict_list):
                        ax.scatter(t, predict[r, c, :], label=f"Predict Point-{model_name[i]}", s=5)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                if r == N-1:
                    ax.set_xlabel("Time (s)", fontsize=30, fontweight="bold")
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_major_locator(plt.MaxNLocator(8))
                ax.tick_params(axis='both', which='major', labelsize=30)  # Increase label size
                ax.grid(True)

        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.14, wspace=0.3, hspace=0.3)
        handles, labels = ax.get_legend_handles_labels()
        plt.rcParams['legend.handleheight'] = 4.0  # 调整行间距
        fig.legend(handles, labels, loc='lower center', fontsize=30, ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("The comparison of responses predicted by different models", fontsize=50, fontweight="bold")
        plt.savefig("predict.eps")
        plt.show()