import torch
from torch import nn
from d2l import torch as d2l

from recurrent_neural_network.rnn_concise import RNNModel
from recurrent_neural_network.rnn_scratch import RNNModelScratch
from utils.Train_ch8 import predict_ch8, train_ch8


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 输入门参数
    """
    I_t = fi(X_t @ W_xi + H_t-1 @ W_hi + b_i)
    X_t.shape = (batch_size, vocab_size)
    I_t.shape = (batch_size, num_hiddens)
    """
    W_xi, W_hi, b_i = three()

    # 遗忘门参数
    """
    F_t = fi(X_t @ W_xf + H_t-1 @ W_hf + b_f)
    F_t.shape = (batch_size, num_hiddens)
    """
    W_xf, W_hf, b_f = three()

    # 输出门参数
    """
    O_t = fi(X_t @ W_xo + H_t-1 @ W_ho + b_o)
    O_t.shape = (batch_size, num_hiddens)
    """
    W_xo, W_ho, b_o = three()

    # 候选记忆元参数
    """
    C_tilda_t = tanh(X_t @ W_xc + H_t-1 @ W_hc + b_c)
    C_tilda_t.shape = (batch_size, num_hiddens)
    """
    W_xc, W_hc, b_c = three()

    # 输出层参数
    """
    C_t = F_t * C_t-1 + I_t * C_tilda_t
    H_t = O_t * tanh(C_t)
    Y_t = H_t @ W_hq + b_q
    C_t.shape = H_t.shape = (batch_size, num_hiddens)
    Y_t.shape = (batch_size, vocab_size)
    """
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho,
        b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # lstm scratch
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                init_lstm_state, lstm)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    # lstm concise
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)








