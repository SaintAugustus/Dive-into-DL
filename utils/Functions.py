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


def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
