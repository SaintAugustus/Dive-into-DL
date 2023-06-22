import torch
from torch import nn
from d2l import torch as d2l

from recurrent_neural_network.rnn_concise import RNNModel
from utils.Train_ch8 import predict_ch8, train_ch8

"""
O_t = H_L_t @ W_hq + b_q
H_l_t = fi(H_l-1_t @ W_l_xh + H_l_t-1 @ W_l_hh + b_l_h)
H_1_t = fi(H_0_t @ W_1_xh + + H_1_t-1 @ W_1_hh + b_1_h)
H_0_t = X_t
H_l_t.shape = (batch_size, num_hiddens)
H_0_t.shape = X_t.shape = (batch_size, vocab_size)
"""


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # deep lstm concise
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    num_epochs, lr = 500, 2
    train_ch8(model, train_iter, vocab, lr * 1.0, num_epochs, device)




