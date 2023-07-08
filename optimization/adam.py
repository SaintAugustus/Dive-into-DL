import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

from utils.Train_ch11 import get_data_ch11, train_ch11, train_concise_ch11


def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    """
    v_t = beta1 * v_t-1 + (1 - beta1) * g_t
    s_t = beta2 * s_t-1 + (1 - beta2) * g_t^2
    v_t_hat = v_t / (1 - beta1^t), s_t_hat = s_t / (1 - beta2^t)
    g'_t = lr * v_t_hat / (sqrt(s_t_hat) + eps)
    """
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-6, hyperparams['lr']
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= lr * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

def yogi(params, states, hyperparams):
    """
    v_t = beta1 * v_t-1 + (1 - beta1) * g_t
    s_t = s_t-1 + (1 - beta2) * g_t^2 * sign(g_t^2 - s_t-1)
    v_t_hat = v_t / (1 - beta1^t), s_t_hat = s_t / (1 - beta2^t)
    g'_t = lr * v_t_hat / (sqrt(s_t_hat) + eps)
    """
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1



if __name__ == "__main__":
    # scratch
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    train_ch11(adam, init_adam_states(feature_dim),
                {'lr': 0.01, 't': 1}, data_iter, feature_dim)
    plt.show()

    # concise
    trainer = torch.optim.Adam
    train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
    plt.show()

    # yogi
    train_ch11(yogi, init_adam_states(feature_dim),
                {'lr': 0.01, 't': 1}, data_iter, feature_dim)
    plt.show()













