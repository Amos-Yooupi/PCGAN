import torch
import torch.nn as nn


class Distill(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1), nn.ELU(),
                                 nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
                                 nn.MaxPool1d(stride=2, kernel_size=3, padding=1))

    def forward(self, x):
        assert x.shape.__len__() == 3, print("x.shape should be 3")
        return self.net(x.permute(0, 2, 1)).transpose(1, 2)


if __name__ == '__main__':
    distill = Distill(256, 64, 256)
    x = torch.rand((64, 48, 256))
    print(distill(x).shape)