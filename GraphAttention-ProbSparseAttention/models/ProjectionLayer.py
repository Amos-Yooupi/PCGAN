import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layer, dropout, alpha=0.1):
        super().__init__()
        pro_layers = [nn.Linear(in_dim, hidden_dim), nn.ELU(alpha=alpha)]
        for _ in range(num_hidden_layer):
            pro_layers.append(nn.Linear(hidden_dim, hidden_dim))
            pro_layers.append(nn.ELU(alpha=alpha))
        pro_layers.append(nn.Linear(hidden_dim, out_dim))

        self.pro_layers = nn.Sequential(*pro_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.pro_layers(x))


if __name__ == '__main__':
    x = torch.rand((64, 20, 3, 256))
    net = ProjectionLayer(256, 128, 4, 1, 0.2)
    print(net(x).shape)