import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.optim import Adam

from datasets.movie_lens_dataset import load_dataset_ml100k
from utils.train_recsys import train_recsys_rating


class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super().__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = torch.sum(P_u * Q_i, dim=1, keepdim=True)
        outputs = outputs + b_u + b_i
        return outputs.reshape(-1)

def evaluator(net, test_iter, device):
    net = net.to(device)
    net.eval()
    mse_loss = nn.MSELoss(reduction='sum')
    with torch.no_grad():
        metric = d2l.Accumulator(2) # total_loss, total_items
        for users, items, ratings in test_iter:
            users, items, ratings = users.to(device), \
                items.to(device), ratings.to(device)
            output = net(users, items)
            loss = mse_loss(output, ratings.float())
            metric.add(loss, users.shape[0])
    rmse = math.sqrt(metric[0] / metric[1])
    return rmse


if __name__ == "__main__":
    # test MF
    # net = MF(5, 100, 200)
    # user_id = torch.randint(1, 5, (64,))
    # item_id = torch.randint(1, 5, (64,))
    # output = net(user_id, item_id)
    # print(output.shape)

    # read dataset
    d2l.DATA_HUB['ml-100k'] = (
        'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

    # train MF
    device = d2l.try_gpu()
    num_users, num_items, train_iter, test_iter = load_dataset_ml100k(
                            test_ratio=0.1, batch_size=256)
    net = MF(30, num_users, num_items)
    lr, num_epochs, wd = 0.002, 30, 1e-5
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.MSELoss(reduction='sum')
    train_recsys_rating(net, train_iter, test_iter, loss, optimizer, num_epochs,
                        device, evaluator)
    plt.show()












