import collections
import math

import torch
from torch import nn


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

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    """ 
    BLEU = exp(min(0, 1 - len_label / len_pred)) * PI(pn^(1/2^n))
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
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score












