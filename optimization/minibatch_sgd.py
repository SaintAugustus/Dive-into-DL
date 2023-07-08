import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

from utils.Train_ch11 import train_ch11, get_data_ch11, train_concise_ch11


def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd_ch11, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

def sgd_ch11(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()



if __name__ == "__main__":
    d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                               '76e5be1548fd8222e5074cf0faae75edff8cf93f')

    # scratch train
    gd_res = train_sgd(1, 1500, 10)
    sgd_res = train_sgd(0.005, 1)
    mini1_res = train_sgd(.4, 100)
    mini2_res = train_sgd(.05, 10)

    # concise train
    data_iter, _ = get_data_ch11(10)
    trainer = torch.optim.SGD
    train_concise_ch11(trainer, {'lr': 0.01}, data_iter)










