import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from datasets.movie_lens_dataset import (load_data_ml100k,
                        read_data_ml100k, split_data_ml100k)


class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Linear(num_users, num_hidden),
                        nn.Sigmoid())
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if self.training:  # Mask the gradient during training
            return pred * torch.sign(input)
        return pred


def evaluator(net, test_data, inter_matrix, device):
    net = net.to(device)
    net.eval()
    mse_loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        # certain item to all users, row of inter_matrix
        for value in inter_matrix:
            metric = d2l.Accumulator(2)
            value = torch.tensor(value, dtype=torch.float32,
                                 device=device)
            pred = net(value)
            loss = mse_loss(pred, value.float())
            metric.add(loss, value.shape[0])
    rmse = math.sqrt(metric[0] / metric[1])
    return rmse

class MovieLensDatasetAutorec(Dataset):
    def __init__(self, datamat):
        # certain item to all users, row of inter_matrix
        self.datamat = datamat

    def __getitem__(self, idx):
        return self.datamat[idx]

    def __len__(self):
        return len(self.datamat)


def train_recsys_rating(net, train_iter, test_iter, loss_fun, optimizer, num_epochs,
                        device=d2l.try_gpu(), evaluator=None, **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 10],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for value in train_iter:
            timer.start()
            value = value.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = net(value)
            loss = loss_fun(pred, value.float()) # reduction = 'sum'
            loss.backward()
            optimizer.step()
            metric.add(loss, value.shape[0])
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'], device)
        else:
            test_rmse = evaluator(net, test_iter, device)
        train_l = metric[0] / metric[1]
        animator.add(epoch + 1, (train_l, test_rmse))

    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == "__main__":
    # test AutoRec
    # net = AutoRec(20, 50)
    # input = torch.normal(0, 1, (64, 50))
    # output = net(input)
    # print(output.shape)

    # load data
    # read dataset
    d2l.DATA_HUB['ml-100k'] = (
        'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

    # Load the MovieLens 100K dataset
    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(df, num_users, num_items)
    _, _, _, train_inter_mat = load_data_ml100k(train_data, num_users,
                                                    num_items)
    _, _, _, test_inter_mat = load_data_ml100k(test_data, num_users,
                                                   num_items)
    train_iter = DataLoader(MovieLensDatasetAutorec(train_inter_mat),
                            batch_size=256, shuffle=True)
    test_iter = DataLoader(MovieLensDatasetAutorec(test_inter_mat),
                           batch_size=256, shuffle=False)

    # Model initialization, training, and evaluation
    device = d2l.try_gpu()
    net = AutoRec(500, num_users)
    lr, num_epochs, wd = 0.002, 30, 1e-5
    trainer = Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.MSELoss(reduction='sum')
    train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                            device, evaluator, inter_mat=test_inter_mat)
    plt.show()









