import torch
from torch import nn
from d2l import torch as d2l

from recurrent_neural_network.rnn_concise import RNNModel
from recurrent_neural_network.rnn_scratch import RNNModelScratch
from utils.Train_ch8 import predict_ch8, train_ch8


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 更新门参数
    """
    Z_t = fi(X_t @ W_xz + H_t-1 @ W_hz + b_z)
    X_t.shape = (batch_size, vocab_size)
    Z_t.shape = (batch_size, num_hiddens)
    """
    W_xz, W_hz, b_z = three()

    # 重置门参数
    """
    R_t = fi(X_t @ W_xr + H_t-1 @ W_hr + b_r)
    R_t.shape = (batch_size, num_hiddens)
    """
    W_xr, W_hr, b_r = three()

    # 候选隐状态参数
    """
    H_tilda_t = tanh(X_t @ W_xh + (R_t * H_t-1) @ W_hh + b_h)
    H_tilda_t.shape = (batch_size, num_hiddens)
    """
    W_xh, W_hh, b_h = three()

    # 输出层参数
    """
    H_t = Z_t * H_t-1 + (1 - Z_t) * H_tilda_t
    Y_t = H_t @ W_hq + b_q
    H_t.shape = (batch_size, num_hiddens)
    Y_t.shape = (batch_size, vocab_size)
    """
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)



if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # train and test gru scratch
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                                init_gru_state, gru)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    # gru concise
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)









