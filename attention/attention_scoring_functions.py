import math
import torch
from torch import nn
from d2l import torch as d2l

from utils.Functions import masked_softmax
from utils.Plot import show_heatmaps


class AdditiveAttention(nn.Module):
    """加性注意力"""
    """
    a(q, k) = W_v.T @ tanh(W_q @ q + W_k @ k)
    W_q.shape = (num_hiddens, query_size), W_k.shape = (num_hiddens, key_size)
    W_v.shape = (num_hiddens, 1)
    a(q, k).shape = (batch_size, num_queries, num_keys)
    value.shape = (batch_size, num_keys, value_size)
    output.shape = (batch_size, num_queries, value_size)
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和，得到features (batch_size，查询的个数，“键－值”对的个数，num_hidden)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores, attention_weights的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # value.shape = (batch_size, key-value pairs, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    """
    a(q, k) = (q.T @ k) / d
    q.shape = (batch_size, num_queries, d)
    k.shape = (batch_size, num_keys, d)
    a(q, k).shape = (batch_size, num_queries, num_keys)
    v.shape = (batch_size, num_keys, value_size)
    output = a(q, k) @ v, output.shape = (batch_size, num_queries, value_size)
    """
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == "__main__":
    # test masked_softmax
    # print(masked_softmax(torch.rand(2, 3, 4), torch.tensor([2, 3])))
    # print(masked_softmax(torch.rand(2, 3, 4), torch.tensor([[1, 3, 2], [2, 4, 1]])))

    # AdditiveAttention
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval()
    print("AdditiveAttention: ", attention(queries, keys, values, valid_lens), "\n\n")
    show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

    # DotProductAttention
    queries = torch.normal(0, 1, (2, 1, 2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print("DotProductAttention: ", attention(queries, keys, values, valid_lens))
    show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')











