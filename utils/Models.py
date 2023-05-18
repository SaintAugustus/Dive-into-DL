import torch


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def softmax_net(X, W, b):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)