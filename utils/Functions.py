import collections
import math

import torch
from torch import nn
import torch.nn.functional as F


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def print_net(net, X):
    for layer in net:
        print(layer.__class__.__name__, "input shape:\t", X.shape)
        X = layer(X)
        print(layer.__class__.__name__, "output shape:\t", X.shape)
    print("\n")


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def bleu(pred_seq, label_seq, k, simple_version=False):
    """计算BLEU"""
    """ 
    BLEU = exp(min(0, 1 - len_label / len_pred)) * PI(pn^(1/2^n))
    BLEU = exp(min(0, 1 - len_label / len_pred)) * PI(pn^(1/4)) for simple version
    pn = num_matches in pred / len_ pred - n + 1
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))  # (0, 1]
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        power = 0.25 if simple_version else math.pow(0.5, n)
        score *= math.pow(num_matches / (len_pred - n + 1), power)
    return score

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量,
    # if valid_len 1D, valid_len.shape[0] == X.shape[0],
    # if valid_len 2D, valid_len.shape[0,1] == X.shape[0,1]
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=1e-6)
    return F.softmax(X.reshape(shape), dim=-1)

def to_device(data, device):
    data = [item.to(device) for item in data]
    return data








