import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

from datasets.machine_translation_dataset import load_data_nmt
from modern_rnn.encoder_decode import Encoder, Decoder, EncoderDecoder
from utils.Functions import sequence_mask, bleu
from utils.Optim import MaskedSoftmaxCELoss
from utils.Train_seq2seq import train_seq2seq, predict_seq2seq


class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional=False, dropout=0, **kwargs):
        super().__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          bidirectional=bidirectional, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # encoder and decoder may be different
        state = enc_outputs[1]
        if state.shape[0] == self.rnn.num_layers * 2:   # encoder is bidirectional 4,4,16
            self.begin_state = torch.zeros((self.rnn.num_layers,
                                           state.shape[1], self.rnn.hidden_size),
                                           device=state.device)
            for i in range(self.rnn.num_layers):
                self.begin_state[i, :, :] = state[2 * i: 2 * i + 2, :, :].mean(dim=0)
        else:  # unidirectional
            self.begin_state = state
        return state

    def forward(self, X, state):
        # 输入'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps，context 取state dim=0的平均
        state_mean = state.mean(dim=0, keepdim=True)
        context = state_mean.repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)

        # 设置state = begin_state
        state = self.begin_state
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


if __name__ == "__main__":
    # test encoder
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2, bidirectional=True)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    """
    # output.shape = (7, 4, 32) = (time_steps, batch_size, num_hiddens * 2),
    # 2 is for bidirectional,
    # state.shape = (4, 4, 16) = (num_layers * 2, batch_size, num_hiddens),
    # 2 is for bidirectional
    """
    output, state = encoder(X)
    print(output.shape, state.shape)


    # test decoder
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
    decoder.eval()
    # decoder init_state is torch.zeros, return latest state of encoder
    state = decoder.init_state(encoder(X))
    output, state = decoder(X, state)
    print(output.shape, state.shape)

    # test sequence_mask
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(sequence_mask(X, torch.tensor([1, 2])))
    X = torch.ones(2, 3, 4)
    print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    # test MaskedSoftmaxCELoss
    loss = MaskedSoftmaxCELoss()
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    loss = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
                torch.tensor([4, 2, 0]))
    print(loss)

    # train seq2seq
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                             bidirectional=True, dropout=dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                             dropout=dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # test seq2seq
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
















