import torch

from utils.Accumulator import Accumulator


def squared_loss(y_hat, y):
    """均方损失"""
    # 0.5(y-y_hat)^2
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                y_hat = net(X)
            else:
                y_hat = net.forward(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

