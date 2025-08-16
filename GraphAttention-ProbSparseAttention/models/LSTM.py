import torch
import torch.nn as nn
from ChangeL import ChangeL


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim)
        self.projects = nn.Sequential(*[nn.Linear(config.hidden_dim, config.obs_dim) for _ in range(3)])
        self.dropout = nn.Dropout(p=config.dropout)
        self.dec_length = config.dec_length
        self.change_l = ChangeL(config.enc_length, config.hidden_L, config.dec_length, 1, config.dropout)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        assert train_data.shape.__len__() == 4, print("x train should be (B, L, N, D)")
        assert earth_data.shape.__len__() == 4, print("x train should be (B, L, N, D)")
        B, L, N, D = train_data.shape
        # x_earth = x_earth.view(B, L, N*D).unsqueeze(dim=-2).repeat(1, 1, N, 1)
        # x = torch.cat([x_train, x_earth], dim=-1)
        # B, L, N, D = x.shape
        bridge_span = bridge_data.shape[-2] // 3
        assert N == 3, print("Train num should be 3")
        x = train_data.transpose(0, 1).contiguous().view(L, B*N, D)
        # span_feature = torch.zeros((L, B*N, 1)).to(train_data.device)
        # span_feature[:] = bridge_span
        # x = torch.cat([x, span_feature], dim=-1)
        result, _ = self.lstm(x)
        result = self.dropout(result)
        # 变成(B*N, L, D)
        result = self.change_l(result.view(L, B, N, -1).permute(1, 2, 0, 3).view(B*N, L, -1))
        # 变回(B*N, L, D) -> (B, L, N, D)
        result = result.view(B, N, self.dec_length, -1).transpose(1, 2)
        result = torch.stack([self.projects[i](result[:, :, i, :]) for i in range(N)], dim=-2)
        return self.dropout(result)

    def __repr__(self):
        return "LSTM"

    def name(self):
        return "LSTM"

