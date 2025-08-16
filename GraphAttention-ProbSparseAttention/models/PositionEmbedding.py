import torch.nn as nn
import torch
import math


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_size, config, max_len=1000):
        super().__init__()
        # 编码方式
        if config.pos_mode == "sincos":
            self.pe = torch.zeros(max_len, embedding_size)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)
        elif config.pos_mode == "learnable":
            W_pos = torch.empty((max_len, embedding_size))  # pe = None and learn_pe = False can be used to measure impact of pe
            self.pe = nn.init.uniform_(W_pos, -0.02, 0.02)
            self.pe = nn.Parameter(self.pe)
        else:
            raise Exception("pe only can be sincos or learnable")

        self.dropout = nn.Dropout(p=config.dropout)
        self.D = torch.tensor(embedding_size)

        if config.scale_learn:
            self.scale = nn.Parameter(torch.sqrt(self.D))
        else:
            self.scale = torch.sqrt(self.D)

    def forward(self, x):
        assert x.shape.__len__() == 3, print("the shape of input x should be 3")
        B, L, D = x.shape
        assert D == self.pe.shape[-1], print("The dim of x should be equal to embedding size")
        return self.dropout(x * self.scale) + self.pe[:L, :][None, :, :].to(x.device)


if __name__ == '__main__':
    x = torch.rand((64, 20, 512))
    PE = PositionEmbedding(512)
    print(PE(x).shape)
