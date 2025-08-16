from Embedding import Embedding
from GraphAttentionLayer import MultiGraphAttentionLayer
from ProjectionLayer import ProjectionLayer
from ProbSparseAttention import EncoderBlock, DecoderBlock, EncoderDecoder
from PositionEmbedding import PositionEmbedding
import torch.nn as nn
import torch
from ChangeL import ChangeL
from Distill import Distill
import torch.nn.functional as F
from model_utils.timer import timer


class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.linear = nn.Sequential(*[nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layer)])
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])
        self.dropout = nn.Dropout(p=config.dropout)
        self.change_l = ChangeL(config.enc_length, 64, config.dec_length, 1, config.dropout)

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        adj = adj.unsqueeze(dim=0).repeat(B * L, 1, 1)
        # 将B和L结合起来
        g_feature = g_feature.reshape(-1, N, D)
        for blk in self.linear:
            g_feature = F.relu(blk(torch.bmm(adj, g_feature)))
        g_feature = self.dropout(
            g_feature.contiguous().view(B, L, N, D).permute(0, 2, 1, 3).contiguous().view(B * N, L, D))
        g_feature = self.change_l(g_feature).contiguous().view(B, N, -1, D).transpose(1, 2)
        output = []
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output

    def __repr__(self):
        return "GNN"

    def name(self):
        return "GNN"


class GAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                            dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])
        self.change_l = ChangeL(config.enc_length, 64, config.dec_length, 1, config.dropout)
        self.num_layer = config.num_layer
    # @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 将B和L结合起来
        g_feature = g_feature.reshape(-1, N, D)
        for i in range(self.num_layer):
            g_feature = self.gat(g_feature, g_feature, g_feature, adj)
        g_feature = g_feature.contiguous().view(B, L, N, D).permute(0, 2, 1, 3).contiguous().view(B * N, L, D)
        g_feature = self.change_l(g_feature).contiguous().view(B, N, -1, D).transpose(1, 2)
        output = []
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output

    def __repr__(self):
        return "GAT"

    def name(self):
        return "GAT"


class GAT_LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                            dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])
        self.lstm = nn.LSTM(config.hidden_dim, config.hidden_dim, num_layers=1)
        self.num_layer = config.num_layer
        self.change_l = ChangeL(config.enc_length, 64, config.dec_length, 1, config.dropout)

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 图神经网络需要把B和L合并一个维度
        # (B, L, N, D) > (L, B, N, D) -> (L*B, N, D)
        g_feature = g_feature.transpose(0, 1).contiguous().view(-1, N, D)
        for i in range(self.num_layer):
            # 过lstm需要把维度变为(L, B*N, D)
            lstm_in = g_feature.contiguous().view(L, B * N, D)
            lstm_out, _, = self.lstm(lstm_in)
            g_feature = lstm_out.contiguous().view(L * B, N, D)
            g_feature = self.gat(g_feature, g_feature, g_feature, adj)
        # 还原成(B, L, N, D)
        g_feature = g_feature.contiguous().view(L, B, N, D).permute(1, 2, 0, 3).contiguous().view(B * N, L, D)
        g_feature = self.change_l(g_feature).contiguous().view(B, N, -1, D).transpose(1, 2)
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        output = []
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output

    def __repr__(self):
        return "GAT_LSTM"


class GAT_ProbSparseAttention1(nn.Module):
    def __init__(self, config, mode='prob'):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                            dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])
        self.enc_block = EncoderBlock(config.num_head, config.hidden_dim, config.hidden_dim, config.hidden_dim,
                                      config.hidden_dim, dropout=config.dropout, mode=mode)
        self.dec_block = DecoderBlock(config.num_head, config.hidden_dim, config.hidden_dim, config.hidden_dim,
                                      config.hidden_dim, dropout=config.dropout, mode=mode)
        self.pe = PositionEmbedding(config.hidden_dim, config)
        self.dec_length = config.dec_length
        self.num_layer = config.num_layer

    @timer()
    def forward(self, train_data, bridge_data, earth_data, adj):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 需要维度变化加上位置编码
        # (B, L, N, D) > (B, N, L, D) -> (B*N, L, D)
        g_feature = g_feature.transpose(1, 2).contiguous().view(-1, L, D)
        # 解码器的输入，至少要一半是真实的输入，一半是全零
        # 默认是一半的是真实值
        if self.dec_length is None:
            dec_x = torch.zeros((B * N, L, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]
        else:
            assert self.dec_length > L // 2, print("dec length should bigger than L / 2")
            dec_x = torch.zeros((B * N, self.dec_length, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]

        # 加上位置编码
        g_feature = self.pe(g_feature)
        dec_x = self.pe(dec_x.to(g_feature.device))
        # 保持和图神经网络输出一致的维度 (B*L, N, D)
        g_feature = g_feature.contiguous().view(B, N, L, D).transpose(1, 2).contiguous().view(-1, N, D)
        for i in range(self.num_layer):
            # 过稀疏注意力 输出(B*N, L, D)
            att_in = g_feature.contiguous().view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)
            att_out = self.enc_block(att_in)
            # 将att_out从(B*N, L, D) -> (B, N, L, D) -> (B, L, M, D) -> (B*L, N, D)
            g_feature = att_out.contiguous().view(B, N, -1, D).transpose(1, 2).contiguous().view(-1, N, D)
            g_feature, adj = self.gat(g_feature, adj)
        # 还原成(B*N, L, D)
        g_feature = g_feature.contiguous().view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)
        # 进行解码器解码
        for i in range(self.num_layer):
            dec_x = self.dec_block(dec_x, g_feature, g_feature, torch.arange(self.dec_length))
        # 进行还原成(B*N, L, D) -> (B, N, L, D) -> (B, L, N, D)
        g_feature = dec_x.contiguous().view(B, N, -1, D).transpose(1, 2)
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        # 单步预测模型
        output = []
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output


class GAT_ProbSparseAttention2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.gat = nn.Sequential(
            *[MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head, dropout=config.dropout,
                                       alpha=config.alpha) for _ in
              range(config.num_layer)])
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])

        self.encoder_decoder = EncoderDecoder(config.num_layer, config.num_head, config.hidden_dim, config.hidden_dim,
                                              config.hidden_dim, config.hidden_dim, dropout=config.dropout,
                                              mode=config.mode)
        self.pe = PositionEmbedding(config.hidden_dim, )
        self.dec_length = config.dec_length

    def forward(self, train_data, bridge_data, earth_data, adj):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 需要维度变化加上位置编码
        # (B, L, N, D) > (B, N, L, D) -> (B*N, L, D)
        g_feature = g_feature.transpose(1, 2).contiguous().view(-1, L, D)
        # 解码器的输入，至少要一半是真实的输入，一半是全零
        # 默认是一半的是真实值
        if self.dec_length is None:
            dec_x = torch.zeros((B * N, L, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]
        else:
            assert self.dec_length > L // 2, print("dec length should bigger than L / 2")
            dec_x = torch.zeros((B * N, self.dec_length, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]

        # 加上位置编码
        g_feature = self.pe(g_feature)
        dec_x = self.pe(dec_x.to(g_feature.device))
        # 保持和图神经网络输出一致的维度 (B*L, N, D)
        g_feature = g_feature.contiguous().view(B, N, L, D).transpose(1, 2).contiguous().view(-1, N, D)
        for blk in self.gat:
            # 将att_out从(B*N, L, D) -> (B, N, L, D) -> (B, L, M, D) -> (B*L, N, D)
            g_feature, adj = blk(g_feature, adj)
        # 还原成(B*N, L, D)
        g_feature = g_feature.contiguous().view(B, L, N, D).transpose(1, 2).contiguous().view(-1, L, D)
        # 进行解码器解码
        output = self.encoder_decoder(g_feature, dec_x, torch.arange(self.dec_length))
        # 进行还原成(B*N, L, D) -> (B, N, L, D) -> (B, L, N, D)
        g_feature = output.contiguous().view(B, N, -1, D).transpose(1, 2)
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        # 单步预测模型
        output = []
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output


class GATtenion(nn.Module):
    def __init__(self, config, mode='prob'):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        self.gat1 = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                             dropout=config.att_dropout, alpha=config.alpha)
        self.gat2 = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head,
                                             dropout=config.att_dropout, alpha=config.alpha)

        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])

        self.enc_block1 = nn.ModuleList([EncoderBlock(config.num_head, config.hidden_dim, config.hidden_dim,
                                                      config.hidden_dim, config.hidden_dim,
                                                      dropout=config.att_dropout, mode=config.mode) for _ in
                                         range(config.num_layer)])

        self.enc_block2 = nn.ModuleList([EncoderBlock(config.num_head, config.hidden_dim, config.hidden_dim,
                                                      config.hidden_dim, config.hidden_dim,
                                                      dropout=config.att_dropout, mode=config.mode) for _ in
                                         range(config.num_layer)])
        if config.whether_distill:
            self.enc_block1.append(Distill(config.hidden_dim, config.hidden_dim, config.hidden_dim))
            self.enc_block2.append(Distill(config.hidden_dim, config.hidden_dim, config.hidden_dim))

        self.whether_distill = config.whether_distill

        self.dec_block = nn.ModuleList([DecoderBlock(config.num_head, config.hidden_dim, config.hidden_dim,
                                                     config.hidden_dim, config.hidden_dim,
                                                     dropout=config.att_dropout, mode=config.mode) for _ in
                                        range(config.num_att_layer)])
        self.pe = PositionEmbedding(config.hidden_dim, config)
        self.dec_length = config.dec_length
        self.num_layer = config.num_layer


    # @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 需要维度变化加上位置编码
        # (B, L, N, D) > (B, N, L, D) -> (B*N, L, D)
        g_feature = g_feature.transpose(1, 2).contiguous().view(-1, L, D)
        # 解码器的输入，至少要一半是真实的输入，一半是全零
        # 默认是一半的是真实值
        if self.dec_length is None:
            dec_x = torch.zeros((B * N, L, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]
        else:
            assert self.dec_length > L // 2, print("dec length should bigger than L / 2")
            dec_x = torch.zeros((B * N, self.dec_length, D))
            dec_x[:, :L // 2, :] = g_feature[:, L // 2:, :]

        # 加上位置编码
        g_feature = self.pe(g_feature)
        dec_x = self.pe(dec_x.to(g_feature.device))

        "下采样数据"
        g_feature_dow_sample = g_feature[:, ::2, :]

        # 保持和图神经网络输出一致的维度 (B*L, N, D)
        g_feature = g_feature.contiguous().view(B, N, L, D).transpose(1, 2).contiguous().view(-1, N, D)

        for i in range(self.num_layer):
            # 过稀疏注意力 输出(B*N, L, D)
            att_in = g_feature.contiguous().view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)
            blk = self.enc_block1[i]
            att_in = blk(att_in)
            # 将att_out从(B*N, L, D) -> (B, N, L, D) -> (B, L, M, D) -> (B*L, N, D)
            g_feature = att_in.contiguous().view(B, N, -1, D).transpose(1, 2).contiguous().view(-1, N, D)
            g_feature = self.gat1(g_feature, g_feature, g_feature, adj)

        for i in range(self.num_layer - 1):
            # 过稀疏注意力 输出(B*N, L, D)
            att_in = g_feature_dow_sample.contiguous().view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)
            blk = self.enc_block2[i]
            att_in = blk(att_in)
            # 将att_out从(B*N, L, D) -> (B, N, L, D) -> (B, L, M, D) -> (B*L, N, D)
            g_feature_dow_sample = att_in.contiguous().view(B, N, -1, D).transpose(1, 2).contiguous().view(-1, N, D)
            g_feature_dow_sample = self.gat2(g_feature_dow_sample, g_feature_dow_sample, g_feature_dow_sample, adj)
        # 还原成(B*N, L, D)
        g_feature = g_feature.contiguous().view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)
        g_feature_dow_sample = g_feature_dow_sample.view(B, -1, N, D).transpose(1, 2).contiguous().view(B * N, -1, D)

        "进行feature map"
        g_feature = torch.cat([g_feature, g_feature_dow_sample], dim=1)

        # 进行解码器解码
        for i in range(self.num_layer):
            for dec_blk in self.dec_block:
                dec_x = dec_blk(dec_x, g_feature, g_feature,
                                       torch.arange(1, self.dec_length + 1).unsqueeze(0).repeat(B, 1))
        # 进行还原成(B*N, L, D) -> (B, N, L, D) -> (B, L, N, D)
        g_feature = dec_x.contiguous().view(B, N, -1, D).transpose(1, 2)
        assert train_data.shape[-2] == 3, print("Train num should be equal to 3")
        output = []
        for i in range(3):
            output.append(self.pro_layer[i](g_feature[:, :, i, :]))
        output = torch.stack(output, dim=-2)
        return output

    def __repr__(self):
        return "GAttention"

    def name(self):
        return "GTN"
