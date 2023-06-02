import torch


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def print_net(net, X):
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "output shape:\t", X.shape)
    print("\n")