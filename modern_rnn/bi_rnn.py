import torch
from torch import nn
from d2l import torch as d2l

from recurrent_neural_network.rnn_concise import RNNModel
from recurrent_neural_network.rnn_scratch import train_ch8

"""
H_f_t = fi(X_t @ W_f_xh + H_f_t-1 @ W_f_hh + b_f_h)
H_b_t = fi(X_t @ W_b_xh + H_b_t-1 @ W_b_hh + b_b_h)
O_t = H_t @ W_hq + b_q
"""



if __name__ == "__main__":
    # deep bi-lstm concise
    # 加载数据
    batch_size, num_steps, device = 32, 35, d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # 通过设置“bidirective=True”来定义双向LSTM模型
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    # 训练模型
    num_epochs, lr = 500, 1
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)








