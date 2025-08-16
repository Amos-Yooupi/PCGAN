import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, embedding_dim, train_input_dim=4, bridge_input_dim=6, earth_input_dim=3):
        super().__init__()
        self.Wt = nn.Linear(train_input_dim, embedding_dim)
        self.Wb = nn.Linear(bridge_input_dim, embedding_dim)
        self.We = nn.Linear(earth_input_dim, embedding_dim)

    def forward(self, train_data, bridge_data, earth_data):
        t = self.Wt(train_data)
        e = self.We(earth_data)
        B, L, N, D = t.shape
        # 通过学习, 如果全部为零就通过学习
        # if torch.max(bridge_data).item() == 0:
        #     g_ = torch.cat([t, e], dim=-2)
        #     assert connect_matrix.shape[-1] == g_.shape[-2], print("Shape is not corresponded")
        #     b = self.Wb1(torch.bmm(connect_matrix.unsqueeze(dim=0).repeat(B*L, 1, 1), g_.contiguous().view(B*L, -1, D)))
        #     b = b.view(B, L, -1, D)
        # 直接使用数据
        b = self.Wb(bridge_data)
        return t, b, e