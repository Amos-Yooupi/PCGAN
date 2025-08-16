import torch
from GraphAttentionLayer import MultiGraphAttentionLayer
from ProjectionLayer import ProjectionLayer
from PositionEmbedding import PositionEmbedding
import torch.nn as nn
from Embedding import Embedding
from TCN import TemporalConvNet
from ProbSparseAttention import EncoderBlock, MultiHeadAttentionLayer
import torch.nn.functional as F
from model_utils.timer import timer


# 这一部分对应通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_chanel, hidden_channel, num_layer=1, dropout=0.1):
        super().__init__()
        # self.mlp = ProjectionLayer(in_chanel, hidden_channel, 2*in_chanel, num_layer, dropout)
        self.mlp = ProjectionLayer(in_chanel, hidden_channel, in_chanel, num_layer, dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.in_channel = in_chanel

    def forward(self, x):
        # Input [bs, D, n_p+1, hidden_dim]
        # Output [bs, D, n_p+1, hidden_dim]
        assert x.shape.__len__() == 4, print("x.shape -> [bs, n_channels, n_p+1, hidden_dim]")
        _, _, n_p, _ = x.shape
        avg_part = self.mlp(self.avg_pool(x).squeeze())
        max_part = self.mlp(self.max_pool(x).squeeze())
        channel_att = F.sigmoid(avg_part + max_part).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # return torch.cat([channel_att[:, :self.in_channel] * x[:, :, :n_p-1, :], channel_att[:, self.in_channel:] * x[:, :, -2:-1, :]], dim=-2) + x
        return channel_att * x + x


class LatentLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layer, dropout):
        super().__init__()
        self.latent_layer = ProjectionLayer(in_dim, hidden_dim, 2 * hidden_dim, num_layer, dropout)
        self.logvar = None
        self.mean = None
        self.D = hidden_dim

    def forward(self, x):
        # Input [B, L, D]
        # Output [B, L, D]
        latent = self.latent_layer(x)
        logvar, mean = latent[:, :, :self.D], latent[:, :, self.D:]
        self.logvar, self.mean = logvar, mean
        eps = torch.rand([1, 1, logvar.shape[-1]]).to(x.device)
        latent_x = eps * (0.5 * logvar).exp() + mean
        # latent_x = eps * 10 + mean
        return latent_x


# 这一部分对应patch的self-attention的一部分
class PatchBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 需要要L分为(n_p, p_len)
        self.n_p = (config.enc_length - config.patch_len) // config.patch_stride + 1
        # 此时已经通过图神经网络，所以nvars就是hidden_dim
        self.nvars = config.lower_channel_dim
        self.lower_channel_dim = config.lower_channel_dim
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride
        self.hidden_dim = config.hidden_dim
        # 进行p_len投影, 每个通道是否共享投影, 对于整个长度也需要有一个投影
        self.share_embedding = config.share_embedding
        if config.share_embedding:
            self.channel_embedding_part = nn.Linear(config.patch_len, config.hidden_dim)
            self.channel_embedding_all = nn.Linear(config.enc_length, config.hidden_dim)
        else:
            self.channel_embedding_part = nn.ModuleList(
                [nn.Linear(config.patch_len, config.hidden_dim) for _ in range(self.lower_channel_dim)])
            self.channel_embedding_all = nn.ModuleList(
                [nn.Linear(config.enc_length, config.hidden_dim) for _ in range(self.lower_channel_dim)])
        # 进行注意力机制
        # 第一步进行位置编码加降低通道数,最后增加维度回来
        self.lower_channel_layer = nn.Linear(config.hidden_dim, config.lower_channel_dim)
        self.up_channel_layer = nn.Linear(config.lower_channel_dim, config.hidden_dim)
        self.pe = PositionEmbedding(config.hidden_dim, config)
        # 第二步进行多头自注意力 -> 包含残差注意力
        self.att_blks = nn.Sequential(
            *[EncoderBlock(config.num_head, config.hidden_dim, config.hidden_dim, config.hidden_dim, config.hidden_dim,
                           config.att_dropout, config.mode, config.res_att, config.gat, config.learn_gat, self.n_p + 1)
              for _ in range(config.num_layer)])

        # self.att_blks = nn.Sequential(
        #     *[MultiHeadAttentionLayer(config.num_head, config.hidden_dim, config.hidden_dim, config.hidden_dim, config.hidden_dim,
        #                    config.att_dropout, config.res_att, config.gat, config.learn_gat, self.n_p + 1)
        #       for _ in range(config.num_layer)])
        # 第三步通道注意力，不同特征维度交接
        self.channel_att = ChannelAttention(config.hidden_dim, config.hidden_dim // 8, 1, config.dropout)
        # 第四步投影成需要的长度
        self.pro_layers = ProjectionLayer((self.n_p + 1) * config.hidden_dim, config.hidden_L, config.dec_length, 1, 0)

        self.res_att = config.res_att

    # @timer()
    def forward(self, x):
        # Input [bs, L, D] -> [bs, D, n_p+1, hidden_dim]
        # Output [bs, D, n_p+1, hidden_dim] -> [bs, T, D]
        assert x.shape.__len__() == 3, print("x.shape should be [bs, L， D]")
        B, _, D = x.shape
        "Step 1 split patch"
        # 降低通道数
        x = self.lower_channel_layer(x).transpose(1, 2)  # [bs, lower_channel, L]
        patch_x = torch.stack(
            [x[:, :, i * self.patch_stride: i * self.patch_stride + self.patch_len] for i in range(self.n_p)],
            dim=-2)  # [bs,D,n_patch,patch_len]
        "Step 2 Channel Embedding"
        if not self.share_embedding:
            patch_x = torch.stack([self.channel_embedding_part[i](patch_x[:, i, :, :]) for i in range(self.nvars)],
                                  dim=1)
            x = torch.stack([self.channel_embedding_all[i](x[:, i, :]) for i in range(self.nvars)], dim=-2).unsqueeze(
                dim=-2)
        else:
            patch_x = self.channel_embedding_part(patch_x)  # [bs, lower_channel_dim, n_p, hidden_dim]
            x = self.channel_embedding_all(x).unsqueeze(dim=-2)  # [bs, lower_channel_dim, 1, hidden_dim]
        x = torch.cat([patch_x, x], dim=-2)  # [bs, lower_channel_dim, n_p+1, hidden_dim]
        x = x.contiguous().view(-1, self.n_p + 1, self.hidden_dim)  # [bs*D, n_p+1, hidden_dim]
        "Step 3 self-attention - res_att"
        x = self.pe(x)  # [bs*lower_channel_dim, n_p+1, hidden_dim]
        prev = None
        for blk in self.att_blks:
            if self.res_att:
                x, prev = blk(x, prev=prev)
            else:
                x = blk(x)  # [bs*lower_channel_dim, n_p+1, hidden_dim]

        # prev = None
        # for blk in self.att_blks:
        #     if self.res_att:
        #         x, prev = blk(x, x, x, prev=prev)
        #     else:
        #         x = blk(x, x, x)  # [bs*lower_channel_dim, n_p+1, hidden_dim]
        "Step 4 channel attention"
        x = x.contiguous().view(B, self.lower_channel_dim, -1,
                                self.hidden_dim)  # [bs, lower_channel_dim, n_p+1, hidden_dim]
        # 增加通道数
        x = self.up_channel_layer(x.transpose(1, 3)).transpose(1, 3)
        x = self.channel_att(x)  # [bs, lower_channel_dim, n_p+1, hidden_dim]
        "Step 5 output length"
        x = x.contiguous().view(B, self.hidden_dim, -1)  # [bs, lower_channel_dim, n_p+1*hidden_dim]
        output = self.pro_layers(x).transpose(1, 2)  # [bs, lower_channel_dim, T] -> [bs, T, lower_channel_dim]
        return output


class PCGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding -> 对每种不同节点进行维度映射，形成图结构的特征
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        # 图注意网络
        self.gats = nn.ModuleList([MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                                            config.att_dropout, config.alpha) for _ in range(1)])
        # patchblock
        self.patch_block = PatchBlock(config)
        # latentLayer
        self.latent_layer = LatentLayer(config.hidden_dim, config.hidden_dim, 1, config.dropout)
        # TCN
        self.tcn = TemporalConvNet(config.hidden_dim, config.num_channels, dropout=config.dropout)
        # pro_layer
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.obs_dim, 1, 0) for _ in range(3)])

    # @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        "Step 1 通过图注意力网络"
        g_feature = g_feature.contiguous().view(B * L, N, D)
        for module in self.gats:
            g_feature = module(g_feature, g_feature, g_feature, adj)  # [B*L, N, D]
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(1, 2)
        N = 3
        g_feature = g_feature[:, :3, :, :].contiguous().view(B * N, L, D)
        "Step 2 通过PatchBlock"
        g_feature = self.patch_block(g_feature)  # [B*N, T, lower_channel_dim]
        "Step 3 通过TCN layer投影到输出维度"
        g_feature = g_feature.transpose(1, 2)  # [B*N, lower_channel_dim, T]
        g_feature = self.tcn(g_feature).transpose(1, 2)  # [B*N, T, hidden_dim]
        "Step 4 通过latent层"
        g_feature = self.latent_layer(g_feature)  # [B*N, T, hidden_dim]
        "Step 5 通过Project layer投影到输出维度"
        g_feature = g_feature.contiguous().view(B, N, g_feature.shape[1], -1)
        output = torch.stack([self.pro_layer[i](g_feature[:, i, :, :]) for i in range(3)], dim=-2)  # [B, L, 3, obs_dim]
        return output

    def __repr__(self):
        return "PCGAN"

    def name(self):
        return "PCGAN"


class PCGAN_no_exaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding -> 对每种不同节点进行维度映射，形成图结构的特征
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        # 图注意网络
        self.gats = nn.ModuleList([MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                                            config.att_dropout, config.alpha) for _ in range(1)])
        # patchblock
        self.patch_block = PatchBlock(config)
        # latentLayer
        self.latent_layer = LatentLayer(config.hidden_dim, config.hidden_dim, 1, config.dropout)
        # TCN
        self.tcn = TemporalConvNet(config.hidden_dim, config.num_channels, dropout=config.dropout)
        # pro_layer
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.obs_dim, 1, 0) for _ in range(3)])

    @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        "Step 1 通过图注意力网络"
        g_feature = g_feature.contiguous().view(B * L, N, D)
        for module in self.gats:
            g_feature = module(g_feature, g_feature, g_feature, adj)  # [B*L, N, D]
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(1, 2)
        g_feature = g_feature.contiguous().view(B * N, L, D)
        "Step 2 通过PatchBlock"
        g_feature = self.patch_block(g_feature)  # [B*N, T, lower_channel_dim]
        "Step 3 通过TCN layer投影到输出维度"
        g_feature = g_feature.transpose(1, 2)  # [B*N, lower_channel_dim, T]
        g_feature = self.tcn(g_feature).transpose(1, 2)  # [B*N, T, hidden_dim]
        "Step 4 通过latent层"
        g_feature = self.latent_layer(g_feature)  # [B*N, T, hidden_dim]
        "Step 5 通过Project layer投影到输出维度"
        g_feature = g_feature.contiguous().view(B, N, g_feature.shape[1], -1)
        output = torch.stack([self.pro_layer[i](g_feature[:, i, :, :]) for i in range(3)], dim=-2)  # [B, L, 3, obs_dim]
        return output

    def __repr__(self):
        return "PCGAN"

    def name(self):
        return "PCGAN"


class PCGAN_no_exaction_attention_point(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding -> 对每种不同节点进行维度映射，形成图结构的特征
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        # 图注意网络
        self.gats = nn.ModuleList([MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                                            config.att_dropout, config.alpha) for _ in range(1)])
        # patchblock
        self.att_block = nn.ModuleList([Att_layer(config.hidden_dim, config.num_head) for _ in range(3)])
        self.patch_block = PatchBlock(config)
        # latentLayer
        self.latent_layer = LatentLayer(config.hidden_dim, config.hidden_dim, 1, config.dropout)
        # TCN
        self.tcn = TemporalConvNet(config.hidden_dim, config.num_channels, dropout=config.dropout)
        # pro_layer
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.obs_dim, 1, 0) for _ in range(3)])

        self.length_linear = nn.Sequential(*[nn.Linear(config.enc_length, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, config.dec_length)])

    @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")

        g_feature = self.length_linear(g_feature.transpose(1, 3)).transpose(1, 3)
        B, L, N, D = g_feature.shape

        "Step 1 通过图注意力网络"
        g_feature = g_feature.contiguous().view(B * L, N, D)
        for module in self.gats:
            g_feature = module(g_feature, g_feature, g_feature, adj)  # [B*L, N, D]
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(1, 2)

        g_feature = g_feature.contiguous().view(B * N, L, D)
        "Step 2 通过PatchBlock"
        for model in self.att_block:
            g_feature, _ = model(g_feature, g_feature, g_feature)
        # g_feature = self.patch_block(g_feature)  # [B*N, T, lower_channel_dim]
        "Step 3 通过TCN layer投影到输出维度"
        g_feature = g_feature.transpose(1, 2)  # [B*N, lower_channel_dim, T]
        g_feature = self.tcn(g_feature).transpose(1, 2)  # [B*N, T, hidden_dim]
        "Step 4 通过latent层"
        g_feature = self.latent_layer(g_feature)  # [B*N, T, hidden_dim]
        "Step 5 通过Project layer投影到输出维度"
        g_feature = g_feature.contiguous().view(B, N, g_feature.shape[1], -1)
        output = torch.stack([self.pro_layer[i](g_feature[:, i, :, :]) for i in range(3)], dim=-2)  # [B, L, 3, obs_dim]
        return output

    def __repr__(self):
        return "PCGAN"

    def name(self):
        return "PCGAN"


class Att_layer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Att_layer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, query, key, value):
        # 执行注意力计算
        attn_output, attn_weights = self.attention(query, key, value)

        # 返回包含两个元素的元组
        return attn_output, attn_weights  # 确保返回的是一个元组


if __name__ == '__main__':
    from model_utils.Config import Config

    config = Config(r"E:\DeskTop\深度学习\GraphAttention-ProbSparseAttention\config.json")
    # 测试通道注意力
    x = torch.rand((64, 128, 16, 256))
    ca = ChannelAttention(128, 16, 1, 0.2)
    print("通道注意力")
    print(ca(x).shape)

    # 测试PatchBlock
    x = torch.rand((64, 36, 256))
    pb = PatchBlock(config)
    print("PatchBlock部分")
    print(pb(x).shape)

    # 测试LatentLayer
    x = torch.rand((64, 36, 256))
    ll = LatentLayer(256, 128, 128, 0)
    print("Latent Layer")
    print(ll(x).shape)
