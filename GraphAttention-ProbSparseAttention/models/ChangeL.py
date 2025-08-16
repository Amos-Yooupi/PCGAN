"""这个类是用来将(B, T, D) -> (B, L, D)"""
import torch
import torch.nn as nn


class ChangeL(nn.Module):
    def __init__(self, input_L, hidden_L, output_L, num_layer, dropout=0.1):
        super().__init__()
        # 初始化第一个卷积层
        layers = [nn.Conv1d(input_L, hidden_L, kernel_size=3, padding=1)]

        # 添加多个中间卷积层
        for i in range(num_layer - 1):
            layers.append(nn.Conv1d(hidden_L, hidden_L, kernel_size=3, padding=1))

        # 添加最后的输出卷积层
        layers.append(nn.Conv1d(hidden_L, output_L, kernel_size=3, padding=1))

        # 将所有层组合成一个nn.Sequential
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


if __name__ == '__main__':
    x = torch.zeros((64, 36, 512))
    changel = ChangeL(36, 72, 100, 1, 0.2)
    print(changel(x).shape)