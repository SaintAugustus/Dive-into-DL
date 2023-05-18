import torch
from IPython import display
from d2l import torch as d2l

from utils.Accumulator import Accumulator
from utils.Animator import Animator
from utils.ImageClassificationDataset import load_data_fashion_mnist
from utils.Models import SoftmaxNet
from utils.Optim import evaluate_accuracy, accuracy, sgd, cross_entropy


def train_epoch_ch3(net, train_iter, loss, updater, lr):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        if isinstance(net, torch.nn.Module):
            y_hat = net(X)
        else:
            y_hat = net.forward(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater([net.W, net.b], lr, X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, lr):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, lr)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f"epoch {epoch + 1}: train_acc {train_metrics[1]}, test_acc {test_acc}")
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def main():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    net = SoftmaxNet(W, b)

    lr = 0.1
    num_epochs = 10

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, sgd, lr)
    #predict_ch3(net, test_iter)

if __name__ == "__main__":
    main()










