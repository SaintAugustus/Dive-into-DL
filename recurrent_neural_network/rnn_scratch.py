import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

from datasets.language_models_dataset import load_data_time_machine
from utils.Train_ch8 import predict_ch8, train_ch8


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    """
    H_t = fi(X_t @ W_xh + H_t-1 @ W_hh + b_t)
    X_t.shape = (batch_size, vocab_size)
    H_t.shape = (batch_size, num_hiddens)
    """
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    """
    O_t = H_t @ W_hq + b_q
    X_t.shape = (batch_size, vocab_size)
    O_t.shape = (batch_size, vocab_size)
    """
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """
    initial H_0
    H_0.shape = (batch_size, num_hiddens)
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    # inputs.shape = (time_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                        get_parms, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_parms(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        print("model scratch: " + self.forward_fn.__name__)

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # onehot
    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)

    # test RNNModelScratch
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                          init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    # test predict_ch8
    preds = predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
    print(preds)

    # test train_ch8
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())












