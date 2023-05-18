import random
import torch

from utils.DataUtils import data_iter
from utils.Models import linreg
from utils.Optim import squared_loss, sgd
from utils.SyntheticData import synthetic_data
from utils.Timer import Timer
from d2l import torch as d2l

def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    batch_size = 10
    lr = 0.03
    num_epochs = 50
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            if (epoch + 1) % 5 == 0:
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {abs((true_w - w.reshape(true_w.shape)) / true_w)}')
    print(f'b的估计误差: {abs((true_b - b) / true_b)}')

if __name__ == "__main__":
    main()