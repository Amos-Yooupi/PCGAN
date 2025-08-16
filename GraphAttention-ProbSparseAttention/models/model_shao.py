import torch
import torch.nn as nn
from GraphAttentionLayer import MultiGraphAttentionLayer
from ProjectionLayer import ProjectionLayer
from ChangeL import ChangeL
from TCN import TemporalConvNet


class GAT_Shao(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = ProjectionLayer(6, config.hidden_dim, config.hidden_dim, 1, 0)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head, dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = ProjectionLayer(config.hidden_dim, config.hidden_dim, 4, 1, 0)
        self.change_l = ChangeL(config.enc_length, config.hidden_L, config.dec_length, 1, config.dropout)
        self.num_layer = config.num_layer

    def forward(self, graph_features, adj, *args):
        g_feature = self.embedding(graph_features)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 将B和L结合起来
        g_feature = g_feature.reshape(-1, N, D)
        for i in range(self.num_layer):
            g_feature, adj = self.gat(g_feature, adj)
        g_feature = g_feature.contiguous().view(B, L, N, D).permute(0, 2, 1, 3).contiguous().view(B*N, L, D)
        g_feature = self.change_l(g_feature).contiguous().view(B, N, -1, D).transpose(1, 2)
        output = self.pro_layer(g_feature.mean(dim=-2))
        return output

    def __repr__(self):
        return "GAT"


class GAT_LSTM_Shao(TemporalConvNet):
    def __init__(self, config):
        super().__init__()
        self.embedding = ProjectionLayer(2, config.hidden_dim, config.hidden_dim, 1, 0)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                            dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = ProjectionLayer(config.hidden_dim, config.hidden_dim, 4, 1, 0)
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim)
        self.num_layer = config.num_layer

    def forward(self, graph_features, adj, *args):
        g_feature = self.embedding(graph_features)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 将B和L结合起来
        g_feature = g_feature.reshape(-1, N, D)
        for i in range(self.num_layer):
            g_feature, adj = self.gat(g_feature, adj)
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(0, 1).contiguous().view(L, B * N, D)
        g_feature, _ = self.lstm(g_feature)
        g_feature = g_feature[:1, :, :]
        g_feature = g_feature.contiguous().view(1, B, N, D).transpose(0, 1)
        output = self.pro_layer(g_feature.sum(dim=-2))
        return output

    def __repr__(self):
        return "GAT_LSTM_SHAO"


class TCN_Shao(TemporalConvNet):
    def __init__(self, config):
        super().__init__(config.hidden_dim, config.num_channels, kernel_size=3, dropout=0.2)
        self.embedding = ProjectionLayer(6, config.hidden_dim, config.hidden_dim, 1, 0)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head, dropout=config.dropout, alpha=config.alpha)
        self.projects = ProjectionLayer(config.num_channels[-1], config.num_channels[-1] // 2, 8, 1, 0)
        self.change_l = ChangeL(config.enc_length, config.hidden_L, config.dec_length, 1, config.dropout)
        self.num_layer = config.num_layer

    def forward(self, g_feature, dis, adj, *args):
        g_feature = torch.cat([g_feature, dis.unsqueeze(dim=-2).repeat(1, 1, g_feature.shape[-2], 1)], dim=-1)
        g_feature = self.embedding(g_feature)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        g_feature = g_feature.contiguous().view(B*L, N, D)
        for i in range(self.num_layer):
            g_feature, adj = self.gat(g_feature, adj)
        g_feature = g_feature.contiguous().view(B, L, N, D).permute(0, 2, 3, 1).contiguous().view(B*N, D, L)
        g_feature = super().forward(g_feature).transpose(1, 2).contiguous().view(B*N, D, -1).transpose(1, 2)  # [B*N, L, D]
        g_feature = self.change_l(g_feature).contiguous().view(B, N, -1, D)
        output = self.projects(g_feature.sum(dim=1))
        return output

    def __repr__(self):
        return "TCN_Shao"