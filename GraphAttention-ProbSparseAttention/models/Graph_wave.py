import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils.timer import timer
from Embedding import Embedding

# This is implementation for graph wavenet

class GraphWaveBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dilation):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation))
        self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 3), padding=(0, dilation), dilation=(1, dilation))
        self.skip_conv = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)

    def forward(self, x, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # 时序提取
        x = x.permute(0, 3, 1, 2)  # (batch_size, num_features, num_nodes, num_timesteps)
        gated = F.tanh(self.conv1(x))
        conv_x = F.sigmoid(self.conv2(x))
        conv_x = gated * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1)  # (batch_size, num_nodes, num_timesteps, num_features)
        # 空间提取
        lfs = torch.einsum("ij,jklm->kilm", [A, conv_x.permute(1, 0, 2, 3)])
        return lfs + self.skip_conv(x).permute(0, 2, 3, 1)


class GraphWaveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config.hidden_dim, config.train_input_dim, config.bridge_input_dim,
                                   config.earth_input_dim)
        num_layers = 2
        dilation = [2**(i+1) for i in range(num_layers)]
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(GraphWaveBlock(config.hidden_dim, config.hidden_dim, dilation[i]))
        self.fully = nn.Linear(config.hidden_dim, config.obs_dim)
        self.length_linear = nn.Sequential(nn.Linear(config.enc_length, config.enc_length),
                                           nn.ReLU(), nn.Linear(config.enc_length, config.dec_length))

    # @timer()
    def forward(self, train_data, bridge_data, earth_data, adj, *args):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        train_data, bridge_data, earth_data = self.embedding(train_data, bridge_data, earth_data)
        X = torch.cat([train_data, bridge_data, earth_data], dim=-2).transpose(1, 2)
        A = adj
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        result = []
        for block in self.blocks:
            X = block(X, A)
            result.append(X)
        return self.fully(self.length_linear(sum(result)[:, :3].transpose(2, 3)).transpose(2, 3)).permute(0, 2, 1, 3)

    def __repr__(self):
        return "GraphWaveNet"

    def name(self):
        return "GraphWaveNet"

