import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

from utils.Train_ch11 import train_ch11, get_data_ch11, train_concise_ch11


def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    """
    s_t = rho * s_t-1 + (1 - rho) * g_t^2
    g'_t = sqrt(delta_x_t-1 + eps) / sqrt(s_t + eps) * g_t
    delta_x_t = rho * delta_x_t-1 + (1 - rho) * g'_t^2
    p_t = p_t-1 - g'_t
    """
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-placeupdatesvia[:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            delta[:] = rho * delta + (1 - rho) * torch.square(g)
            p[:] -= g
        p.grad.data.zero_()


if __name__ == "__main__":
    # scratch
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    train_ch11(adadelta, init_adadelta_states(feature_dim),
                {'rho': 0.9}, data_iter, feature_dim)
    plt.show()

    # concise
    trainer = torch.optim.Adadelta
    train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
    plt.show()

















