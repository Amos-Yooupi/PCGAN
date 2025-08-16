import torch.nn as nn
import torch
from PositionEmbedding import PositionEmbedding
from ProbSparseAttention import EncoderBlock
from ProjectionLayer import ProjectionLayer
from GraphAttentionLayer import MultiGraphAttentionLayer
from Embedding import Embedding


class PredictHead(nn.Module):
    def __init__(self, nvars, input_dim, forecast_len, individual, dropout):
        super().__init__()
        self.individual = individual
        self.nvars = nvars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.nvars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(input_dim, forecast_len))
                self.dropouts.append(nn.Dropout(p=dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(input_dim, forecast_len)
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Input [bs, nvars, n_patch, patch_len]
        # Output [bs, forcast_len, nvars]
        if self.individual:
            output_x = []
            for i in range(self.nvars):
                x_i = self.flattens[i](x[:, i, :, :])
                x_i = self.linears[i](x_i)
                x_i = self.dropouts[i](x_i)  # [bs, forcast_len]
                output_x.append(x_i)
            output_x = torch.stack(output_x, dim=1)
        else:
            output_x = self.dropout(self.linear(self.flatten(x)))
        return output_x.transpose(1, 2)  # [bs, forcast_len, nvars]


class PatchTST(nn.Module):
    def __init__(self, nvars, patch_len, patch_stride, config):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patch = (config.enc_length - self.patch_len) // self.patch_stride + 1
        self.D = config.hidden_dim
        self.nvars = nvars
        self.res_att = config.res_att
        self.share_embedding = config.share_embedding
        # 每个通道是否要独立进行embedding
        if not self.share_embedding:
            self.channel_embedding = nn.ModuleList([nn.Linear(self.patch_len, self.D) for _ in range(self.nvars)])
        else:
            self.channel_embedding = nn.Linear(self.patch_len, self.D)

        # 位置编码
        self.pe = PositionEmbedding(self.D, config)

        # Encoder 编码
        self.enc_blks = nn.Sequential(*[
            EncoderBlock(config.num_head, config.hidden_dim, config.hidden_dim, config.hidden_dim, config.hidden_dim,
                         config.att_dropout, config.mode, config.res_att, config.gat, config.learn_gat, n_keys=self.n_patch) for _ in range(config.num_layer)])
        # predict Head [bs, D, n_patch, hidden_dim] -> [bs, L, D]
        self.predict_head = PredictHead(self.nvars, self.n_patch * self.D, config.dec_length, config.individual,
                                        config.att_dropout)

    def forward(self, x):
        # Input: [bs, seq_T, D]
        # Output: [bs, seq_L, D]
        # 第一步将Seq_T分为patch
        B, T, D = x.shape
        x = x.transpose(1, 2)
        n_patch = (T - self.patch_len) // self.patch_stride + 1
        assert n_patch == self.n_patch, print("n patch is wrong")
        x = torch.stack(
            [x[:, :, i * self.patch_stride: i * self.patch_stride + self.patch_len] for i in range(n_patch)],
            dim=-2)  # [bs,D,n_patch,patch_len]
        # 进行embedding
        if not self.share_embedding:
            x = torch.stack([self.channel_embedding[i](x[:, i, :, :]) for i in range(D)], dim=1)
        else:
            x = self.channel_embedding(x)  # [bs, D, n_patch, hidden_dim]

        x = x.contiguous().view(B * D, n_patch, -1)  # [bs*D, n_patch, hidden_dim]
        x = self.pe(x)
        # Encoder
        prev = None
        for blk in self.enc_blks:
            if self.res_att:
                x, prev = blk(x, prev=prev)
            else:
                x = blk(x)  # [bs*lower_channel_dim, n_p+1, hidden_dim]
        x = x.contiguous().view(B, D, n_patch, -1)  # [bs, D, n_patch, hidden_dim]
        forcast_x = self.predict_head(x)  # [bs, forcast, D]
        # 预测任务
        return forcast_x


# 车桥输入的x需要处理，自定义一个类
class PatchTST_Customer(PatchTST):
    def __init__(self,config):
        super().__init__(config.input_dim, config.patch_len, config.patch_stride, config)
        self.projects = nn.Sequential(
            *[ProjectionLayer(config.input_dim, config.hidden_dim, config.obs_dim, 1, 0) for _ in range(3)])

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        # Input [bs, T, 3, D]
        # Output [bs, L, 3, D]
        assert train_data.shape.__len__() == 4, print("x train should be (B, L, N, D)")
        B, L, N, D = train_data.shape
        assert N == 3, print("Train num should be 3")
        x = train_data.contiguous().view(B * N, L, D)
        forcast_x = super().forward(x)
        forcast_x = forcast_x.contiguous().view(B, N, -1, D).transpose(1, 2)  # [bs, L, N, D]
        output_x = torch.stack([self.projects[i](forcast_x[:, :, i, :]) for i in range(N)], dim=-2)
        return output_x

    def __repr__(self):
        return "PatchTST"

    def name(self):
        return "Graph Wavenet"


class PatchTST_GAT(PatchTST):
    def __init__(self, config):
        self.lower_dim = config.lower_channel_dim
        super().__init__(self.lower_dim, config.patch_len, config.patch_stride, config)
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim, config.earth_input_dim)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head, dropout=config.att_dropout, alpha=config.alpha)
        self.num_layer = config.num_layer
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(self.lower_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])
        self.lower_dim_layer = nn.Linear(config.hidden_dim, self.lower_dim)

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        g_feature = g_feature.contiguous().view(B*L, N, D)

        for i in range(self.num_layer):
            g_feature, adj = self.gat(g_feature, adj)
        # 过PatchTST
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(1, 2).contiguous().view(B*N, L, D)
        # 降低通道数
        g_feature = self.lower_dim_layer(g_feature)
        g_feature = super().forward(g_feature).contiguous().view(B, N, -1, self.lower_dim).transpose(1, 2)

        output = []
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output

    def __repr__(self):
        return "PatchTST_GAT"


if __name__ == '__main__':
    from model_utils.Config import Config

    config = Config(r"E:\DeskTop\深度学习\GraphAttention-ProbSparseAttention\config.json")
    tst = PatchTST(128, config)
    x = torch.rand((64, 36, 128))
    print(tst(x).shape)
