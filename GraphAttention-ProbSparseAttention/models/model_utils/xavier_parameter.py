import torch.nn as nn


def xavier_parameters(net):
    def initialize_weights(layer):
        if isinstance(layer, nn.Linear):
            # 计算Xavier初始化参数
            nn.init.xavier_uniform_(layer.weight, gain=1.414)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)  # 小偏差初始化
        elif isinstance(layer, nn.Module):
            for m in list(layer.children()):
                initialize_weights(m)
        else:
            return

    for layer in net.children():
        initialize_weights(layer)