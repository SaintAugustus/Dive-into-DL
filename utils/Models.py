import torch
import torch.nn as nn

from utils.Functions import relu


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


class MLP:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def forward(self, X):
        X = X.reshape((-1, self.W1.shape[0]))
        X = relu(X@self.W1 + self.b1)
        return X@self.W2 + self.b2