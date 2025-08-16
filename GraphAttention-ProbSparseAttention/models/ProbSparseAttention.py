import math
import torch.nn as nn
import torch
try:
    from models.utils.show_attention import show_attention_maps
except:
    from model_utils.show_attention import show_attention_maps
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, hidden_size, dropout=0, res_att=False):
        super().__init__()
        self.Wq = nn.Linear(query_size, hidden_size)
        self.Wk = nn.Linear(key_size, hidden_size)
        self.Wv = nn.Linear(value_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attention = None
        self.D = hidden_size
        self.res_att = res_att
        if res_att:
            # self.W_res = nn.Linear(2*key_size, 1)
            self.gat = 0.1

    def forward(self, queries, keys, values, mask=None, prev=None):
        assert keys.shape[-2] == values.shape[-2], print("key size should be equal to value size")
        queries, keys, values = self.Wq(queries), self.Wk(keys), self.Wv(values)
        scores = self.get_att_scores(queries, keys)
        if self.res_att and prev:
            # 门控残差注意力
            assert scores.shape == prev, print("prev shape should be same to scores")
            # gat = nn.Sigmoid(self.W_res(torch.cat([prev, scores], dim=-1)))
            scores += prev * self.gat
        self.attention = mask_prob = self.apply_mask(scores, mask)
        att_output = torch.matmul(mask_prob, values)
        if self.res_att:
            return self.out_proj(att_output), scores
        return self.out_proj(att_output)

    def get_att_scores(self, queries, keys, mask=None):
        """
        args: queries: [batch_size, query_len, hidden_size]
              keys: [batch_size, key_len, hidden_size]
              mask: [batch_size, query_len]
        return: scores: [batch_size, num_head, query_len, key_len]
        """
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        return scores

    def apply_mask(self, scores, mask):
        """
        args: scores: [batch_size, query_len, key_len]
              mask: [batch_size, query_len]
        return: mask_scores: [batch_size, query_len, key_len]
        """

        B, query_len, key_len = scores.shape
        if mask is not None:
            mask = mask.repeat(B // mask.shape[0], 1)
            mask = torch.arange(0, key_len)[None, None, :] < mask[:, :, None]
        else:
            mask = torch.ones_like(scores)
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask == 0, -1e9) / math.sqrt(self.D)
        mask_scores = self.dropout((F.softmax(scores, dim=-1)))
        return mask_scores


class MultiHeadAttentionLayer(AttentionLayer):
    def __init__(self, num_head, query_size, key_size, value_size, hidden_size, dropout=0, res_att=False, gat=1, learn_gat=False, n_keys=None):
        super().__init__(query_size, key_size, value_size, hidden_size, dropout)
        assert hidden_size % num_head == 0, print("hidden size should be divided by num_head")
        self.num_head = num_head
        self.mask_attention = None
        self.attention = None
        self.res_att = res_att
        # 这里需要变化一下
        self.D = hidden_size // num_head
        if self.res_att:
            self.learn_gat = learn_gat
            if learn_gat:
                self.gat = nn.Parameter(torch.tensor(gat))
            else:
                assert n_keys is not None, print("If you wanna calculate gat, you should know the n_keys [n_q, 2*n_keys] -> [n_q, 1]")
                self.score_layer = nn.Linear(2 * n_keys, 1)

    def forward(self, queries, keys, values, mask=None, prev=None):
        assert keys.shape[-2] == values.shape[-2], print("key size should be equal to value size")
        B, query_len, hidden_dim = queries.shape
        queries, keys, values = self.Wq(queries), self.Wk(keys), self.Wv(values)

        queries, keys, values = self.transpose_qkv(queries), self.transpose_qkv(keys), self.transpose_qkv(values)
        scores = self.get_att_scores(queries, keys)
        self.attention = scores.detach().contiguous().view(B, self.num_head, query_len, -1)
        if prev is not None and self.res_att:
            # 门控残差注意力
            assert prev.shape == scores.shape, print("prev shape should be same to scores")
            if self.learn_gat:
                gat = self.gat
            else:
                gat = self.score_layer(torch.cat([scores, prev.detach()], dim=-1))
            gat = F.sigmoid(gat)
            scores = prev * gat + (1-gat)*scores
        mask_prob = self.apply_mask(scores, mask)
        self.mask_attention = mask_prob.detach().contiguous().view(B, self.num_head, query_len, -1)

        att_output = torch.matmul(mask_prob, values)
        if self.res_att:
            return self.out_proj(self.restore_qkv(att_output)), scores
        return self.out_proj(self.restore_qkv(att_output))

    def transpose_qkv(self, item):
        B, L, D = item.shape
        transpose_item = item.reshape(B, L, self.num_head, D // self.num_head).transpose(2, 1)
        return transpose_item.contiguous().view(B * self.num_head, L, -1)

    def restore_qkv(self, item):
        B, L, D = item.shape
        restore_item = item.contiguous().view(-1, self.num_head, L, D).transpose(1, 2)
        return restore_item.contiguous().view(-1, L, D * self.num_head)

    def show_attention(self, idx):
        show_attention_maps(F.softmax(self.attention[idx] / math.sqrt(self.D), dim=-1))

    def show_mask_attention(self, idx, name="encoder_attention"):
        show_attention_maps(self.mask_attention[idx], name=name)


class ProbSparseAttention(MultiHeadAttentionLayer):
    def __init__(self, num_head, query_size, key_size, value_size, hidden_size, dropout=0):
        super().__init__(num_head, query_size, key_size, value_size, hidden_size, dropout)
        self.num_head = num_head

    def forward(self, queries, keys, values, mask=None):
        assert keys.shape[-2] == values.shape[-2], print("key size should be equal to value size")
        _, query_len, hidden_dim = queries.shape
        queries, keys, values = self.Wq(queries), self.Wk(keys), self.Wv(values)
        queries, keys, values = self.transpose_qkv(queries), self.transpose_qkv(keys), self.transpose_qkv(values)
        B, L, hidden_dim = keys.shape
        # 随机挑选 num_key // 4，计算活跃度
        choose_idx = torch.randperm(L)[:L//4]
        choose_keys = keys[torch.arange(0, B)[:, None],  choose_idx[None, :]]
        # (B, num_query, num_choose_keys)
        scores = self.get_att_scores(queries, choose_keys)
        # 挑选活跃的query出来
        active_score = torch.abs(torch.max(scores, dim=-1)[0] - torch.mean(scores, dim=-1))
        # 取出num_query // 4 个活跃的query出来
        _, top_k_active_query_idx = torch.topk(active_score, queries.shape[-2] // 4)
        choose_queries = queries[torch.arange(0, B)[:, None], top_k_active_query_idx]
        att_prob = torch.zeros((B, query_len, L)).to(queries.device)
        # encoder是求mean(v)
        att_prob[:] = 1 / L

        # 活跃部分的query进行注意力计算
        att_prob[torch.arange(0, B)[:, None], top_k_active_query_idx] = self.get_att_scores(choose_queries, keys)
        self.attention = att_prob.detach().contiguous().view(-1, self.num_head, queries.shape[-2], self.D)
        att_prob = self.apply_mask(att_prob, mask)
        self.mask_attention = att_prob.detach().contiguous().view(-1, self.num_head, queries.shape[-2], self.D)

        att_output = torch.matmul(att_prob, values)
        return self.out_proj(self.restore_qkv(att_output))


class FeedForwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=0.1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class ResidualLayer(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, y):
        return self.dropout(x + y)


class EncoderBlock(nn.Module):
    def __init__(self, num_head, query_size, key_size, value_size, hidden_size, dropout=0, mode="prob", res_att=False, gat=1, learn_gat=False, n_keys=None):
        super().__init__()
        if mode == "prob":
            self.self_att = ProbSparseAttention(num_head, query_size, key_size, value_size, hidden_size, dropout=dropout)
        elif mode == "norm":
            self.self_att = MultiHeadAttentionLayer(num_head, query_size, key_size, value_size, hidden_size, dropout=dropout, res_att=res_att, gat=gat, learn_gat=learn_gat, n_keys=n_keys)
        else:
            print("Choose should be prob or norm")
            exit()

        self.fc = FeedForwardLayer(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.residual1 = ResidualLayer(dropout)
        self.residual2 = ResidualLayer(dropout)
        self.res_att = res_att

    def forward(self, x, mask=None, prev=None):
        """
        args: x: [batch_size, query_len, hidden_size]
              mask: [batch_size, query_len]
              prev: [batch_size, query_len, key_len]
        return: x: [batch_size, query_len, hidden_size]
        """
        assert x.shape.__len__() == 3, print("the shape of input x should be 3")
        # ProbSparseAttention, MultiHeadAttention
        if self.res_att:
            att_x, prev = self.self_att(x, x, x, mask, prev)
        else:
            att_x = self.self_att(x, x, x, mask, prev)
        # residual
        res_x = self.layer_norm1(self.residual1(x, att_x))
        # FC
        fc_x = self.fc(res_x)
        # residual
        res_x = self.layer_norm2(self.residual2(res_x, fc_x))
        if self.res_att:
            return res_x, prev
        return res_x


# 将decoder分为两部分
class DecoderBlock(nn.Module):
    def __init__(self, num_head, query_size, key_size, value_size, hidden_size, dropout=0, mode="prob"):
        super().__init__()
        if mode == "prob":
            self.self_attention = ProbSparseAttention(num_head, query_size, key_size, value_size, hidden_size, dropout)
        elif mode == "norm":
            self.self_attention = MultiHeadAttentionLayer(num_head, query_size, key_size, value_size, hidden_size, dropout)
        else:
            print("Choose should be prob or norm")
            exit()

        self.cross_attention = MultiHeadAttentionLayer(num_head, query_size, key_size, value_size, hidden_size, dropout)
        self.fc = FeedForwardLayer(hidden_size, hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.res1 = ResidualLayer()
        self.res2 = ResidualLayer()
        self.res3 = ResidualLayer()

    def forward(self, x, keys, values, mask=None):
        # 先进行自注意力 - 需要掩码
        att_x = self.self_attention(x, x, x, mask)
        # 残差连接
        res1_x = self.layer_norm1(self.res1(att_x, x))
        # 交叉注意力 - 不需要掩码
        att_x = self.cross_attention(res1_x, keys, values)
        # 残差连接
        res2_x = self.layer_norm2(self.res2(att_x, res1_x))
        # fc前向传播
        fc_x = self.fc(res2_x)
        # 残差连接
        res3_x = self.layer_norm3(self.res3(fc_x, res2_x))
        return res3_x


class EncoderDecoder(nn.Module):
    def __init__(self, num_layer, num_head, query_size, key_size, value_size, hidden_size, dropout=0, mode='norm'):
        super().__init__()
        self.enc_blks = nn.Sequential(*[EncoderBlock(num_head, query_size, key_size, value_size, hidden_size, dropout, mode=mode) for _ in range(num_layer)])
        self.dec_blks = nn.Sequential(*[DecoderBlock(num_head, query_size, key_size, value_size, hidden_size, dropout, mode=mode) for _ in range(num_layer)])

    def forward(self, enc_x, dec_x, mask=None):
        for blk in self.enc_blks:
            enc_x = blk(enc_x)
        for blk in self.dec_blks:
            dec_x = blk(dec_x, enc_x, enc_x, mask)
        return dec_x


if __name__ == '__main__':
    q = torch.rand((64, 32, 10))
    k, v = q, q
    mask = torch.arange(1, 33)
    # 注意力机制
    att_layer = AttentionLayer(10, 10, 10, 512, 0)
    print("Attention:", att_layer(q, k, v).shape)
    show_attention_maps(att_layer.attention[:4, :, :])
    # 多头案例
    multi_att_layer = MultiHeadAttentionLayer(8, 10, 10, 10, 512, 0)
    print("MultiHeadAttention:", multi_att_layer(q, k, v, torch.arange(32).unsqueeze(0).expand(64, 32)).shape)
    show_attention_maps(multi_att_layer.mask_attention[0, :, :, :])
    # 稀疏注意力
    prob_att_layer = ProbSparseAttention(8, 10, 10, 10, 512, 0)
    print("ProbSparseAttention:", prob_att_layer(q, k, v).shape)
    show_attention_maps(prob_att_layer.attention[0, :, :, :])
    # FC前向传播
    x = torch.zeros((64, 32, 512))
    fc_layer = FeedForwardLayer(512, 512)
    print("FC layer output:", fc_layer(x).shape)
    # 测试EncoderDecoder
    encoder_decoder = EncoderDecoder(2, 8, 512, 512, 512, 512, 0)
    print("EncoderDecoder output:", encoder_decoder(x, x, torch.arange(32).unsqueeze(0).expand(64, 32)).shape)


