import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from ProjectionLayer import ProjectionLayer
from Embedding import Embedding
from GraphAttentionLayer import MultiGraphAttentionLayer
from ChangeL import ChangeL


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x, *args):
        return self.network(x)


class TCN(TemporalConvNet):
    def __init__(self, config):
        super().__init__(config.input_dim, config.num_channels, kernel_size=3, dropout=0.2)
        self.projects = nn.Sequential(*[ProjectionLayer(config.num_channels[-1], config.num_channels[-1]//2, config.obs_dim, 1, 0) for _ in range(3)])
        self.change_l = ChangeL(config.enc_length, config.hidden_L, config.dec_length, 1, config.dropout)

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        # Input [bs, T, 3, D]
        # Output [bs, L, 3, D]
        assert train_data.shape.__len__() == 4, print("x train should be (B, L, N, D)")
        B, L, N, D = train_data.shape
        bridge_span = bridge_data.shape[-2] // 3
        assert N == 3, print("Train num should be 3")
        x = train_data.permute(0, 2, 3, 1).contiguous().view(B*N, D, L)
        # 输出(B * N, D, L) -> (B, L, N, D)
        output_x = super().forward(x).transpose(1, 2).contiguous().view(B*N, L, -1)
        output_x = self.change_l(output_x)
        # 进入proj_layer
        output_x = output_x.contiguous().view(B, N, -1, output_x.shape[-1]).transpose(1, 2)
        output_x = torch.stack([self.projects[i](output_x[:, :, i, :]) for i in range(N)], dim=-2)
        return output_x

    def __repr__(self):
        return "TCN"

    def name(self):
        return "STGCN"


class GAT_TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim, config.earth_input_dim)
        self.gat = MultiGraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.num_head, dropout=config.dropout, alpha=config.alpha)
        self.pro_layer = nn.Sequential(
            *[ProjectionLayer(config.hidden_dim, config.hidden_dim, config.train_input_dim, 1, 0) for _ in range(3)])

        self.tcn = TemporalConvNet(config.hidden_dim, config.num_channels, dropout=config.dropout)
        self.change_l = ChangeL(config.enc_length, config.hidden_L, config.dec_length, 1, config.dropout)
        self.num_layer = config.num_layer

    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        # Input [bs, T, N, D]
        # Output [bs, L, 3, D]
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)

        g_feature = torch.cat([train_data, bridge_data, earth_data], dim=-2)
        assert g_feature.shape.__len__() == 4, print("(B, L, N, D)")
        B, L, N, D = g_feature.shape
        assert adj.shape[0] == adj.shape[1] == N, print("adjacent matrix should be equal to node num")
        # 图神经网络需要把B和L合并一个维度
        # (B, L, N, D) > (L, B, N, D) -> (L*B, N, D)
        g_feature = g_feature.transpose(0, 1).contiguous().view(-1, N, D)
        for i in range(self.num_layer):
            # 过TCN需要把维度变为(B*N, D, L)
            tcn_in = g_feature.contiguous().view(-1, B*N, D).permute(1, 2, 0)
            tcn_out = self.tcn(tcn_in)
            # 由(B*N,D, L) -> (B, N, D, L) -> (L*B, N, D)
            g_feature = tcn_out.contiguous().view(B, N, D, -1).permute(3, 0, 1, 2).contiguous().view(-1, N, D)
            g_feature, adj = self.gat(g_feature, adj)
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
        return "GAT_TCN"


if __name__ == '__main__':
    # Example usage
    num_channels = [25, 50, 100]  # Define the number of channels in each layer
    # 假定输入数据维度
    batch_size = 1  # 批量大小
    seq_length = 100  # 序列长度
    num_features = 10  # 特征数量

    # 随机生成输入数据
    x = torch.randn(batch_size, num_features, seq_length)

    # 实例化模型
    num_channels = [25, 50, 100]  # 定义每层的通道数
    model = TemporalConvNet(num_inputs=num_features, num_channels=num_channels, kernel_size=3)

    # 将输入数据通过模型进行前向传播
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
