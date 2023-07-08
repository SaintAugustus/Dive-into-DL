import torch
from d2l import torch as d2l

from utils.Train_ch11 import get_data_ch11, train_ch11, train_concise_ch11


def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    """
    v_t = beta * v_t-1 + grad_t
    p_t = p_t-1 - lr * v_t
    """
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    train_ch11(sgd_momentum, init_momentum_states(feature_dim),
               {'lr': lr, 'momentum': momentum}, data_iter,
               feature_dim, num_epochs)


if __name__ == "__main__":
    # scratch
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    train_momentum(0.02, 0.5)

    # concise
    trainer = torch.optim.SGD
    train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)






















