import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

from utils.DataUtils import load_array
from utils.SyntheticData import synthetic_data


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 30
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        if epoch % 5 == 0:
            print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    b = net[0].bias.data
    print('w的估计误差：', torch.abs((true_w - w.reshape(true_w.shape)) / true_w))
    print('b的估计误差：', torch.abs((true_b - b) / true_b))

if __name__ == "__main__":
    main()


