import torch
import torch.nn as nn


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

class SoftmaxNet:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, X):
        return softmax(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


