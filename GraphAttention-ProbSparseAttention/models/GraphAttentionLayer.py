import torch
import torch.nn as nn
import torch.nn.functional as F
# 相对路径导入失败，只能用绝对路径
try:
    from models.utils.show_attention import show_attention_maps
except:
    from model_utils.show_attention import show_attention_maps


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=F.leaky_relu):
        super().__init__()
        self.Wo = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, x, adj, *args):
        """
        args:
            x: (B, N, in_features)
            adj: (B, N, N)
        returns:
            output: (B, N, out_features)
        """
        return self.activation(self.Wo(torch.bmm(adj, x)))


class GraphAttentionLayer(GraphConvolution):
    def __init__(self, in_features, out_features, dropout=0, alpha=0.1):
        super(GraphAttentionLayer, self).__init__(out_features, out_features, bias=False, activation=nn.LeakyReLU(alpha))
        self.dropout = dropout

        self.Wq = nn.Linear(in_features, out_features, bias=True)
        self.Wk = nn.Linear(in_features, out_features, bias=True)
        self.Wv = nn.Linear(in_features, out_features, bias=True)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.out_features = out_features

        # record the attention result and plot it
        self.mask_attention = None
        self.attention = None

    def forward(self, q, k, v, adj):
        """
        args: q: (B, N, in_features), k: (B, N, in_features), v: (B, N, in_features), adj: (B, N, N)
        returns: output: (B, N, out_features)
        """
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)  # (B, N, out_features)
        un_mask_attention = self._prepare_attentional_mechanism_input(q, k)
        self.mask_attention = mask_attention = self._mask_attention(un_mask_attention, adj)
        # 得到注意力机制
        self.attention = attention = F.softmax(mask_attention, dim=-1)

        h_prime = super().forward(v, attention)  # (B, N, out_features)

        return h_prime

    def _prepare_attentional_mechanism_input(self, q, k):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(q, self.a[:self.out_features, :])
        Wh2 = torch.matmul(k, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)
        # 抑制小数值，让注意力更加集中
        return self.leakyrelu(e)

    def _mask_attention(self, attention, adj):
        """
        args: attention: (B, N, N), adj: (B, N, N)
        returns: masked_attention: (B, N, N)
        """
        fill_element = torch.full_like(attention, -1e9)
        attention = torch.where(adj > 0, attention, fill_element)
        return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiGraphAttentionLayer(GraphAttentionLayer):
    def __init__(self, in_features, out_features, num_head, dropout=0, alpha=0.1):
        super().__init__(in_features, out_features // num_head, dropout, alpha)
        assert out_features % num_head == 0, print("out feature should be divided by num head")

        self.Wq = nn.Linear(in_features, out_features, bias=True)
        self.Wk = nn.Linear(in_features, out_features, bias=True)
        self.Wv = nn.Linear(in_features, out_features, bias=True)

        self.a = nn.Parameter(torch.empty(size=(2*out_features // num_head, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.num_head = num_head

    def forward(self, q, k, v, adj):
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)  # (B, N, out_features)
        q, k, v = self.transpose_qkv(q), self.transpose_qkv(k), self.transpose_qkv(v)
        un_mask_attention = self._prepare_attentional_mechanism_input(q, k)
        self.mask_attention = mask_attention = self._mask_attention(un_mask_attention, adj)
        # 得到注意力机制
        self.attention = attention = F.softmax(mask_attention, dim=-1)

        h_prime = self.activation(self.Wo(torch.bmm(attention, v)))  # (B, N, out_features)
        return self.transpose_output(h_prime)

    def transpose_qkv(self, q):
        """
        args: q: (B, N, in_features)
        returns: q: (B*num_head, N, out_features/num_head)
        """
        B, N, d = q.shape
        q = q.contiguous().view(B, N, self.num_head, d // self.num_head).transpose(1, 2)
        return q.contiguous().view(B * self.num_head, N, d // self.num_head)

    def transpose_output(self, output):
        """
        args: output: (B*num_head, N, out_features/num_head)
        returns: output: (B, N, out_features)
        """
        B, N, d = output.shape
        output = output.contiguous().view(B // self.num_head, self.num_head, N, d).transpose(1, 2)
        return output.contiguous().view(-1, N, d * self.num_head)

    def show_attention(self, idx):
        attention_matrix = self.attention[idx]
        show_attention_maps(attention_matrix, name='graph_attention', camp='Reds')

    def show_mask_attention(self, idx):
        attention_matrix = self.mask_attention[idx]
        show_attention_maps(attention_matrix, name='mask_graph_attention', camp='Reds')


if __name__ == '__main__':
    gat = MultiGraphAttentionLayer(3, 128, 8)
    x = torch.rand((64, 10, 3))
    adjacent_matrix = torch.eye(10)
    print(gat(x, x, x, adjacent_matrix).shape)
