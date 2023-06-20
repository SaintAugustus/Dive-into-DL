import random

import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l

from utils.TextPreprocessing import Vocab, tokenize, read_time_machine, load_corpus_time_machine


def seq_data_iter_random(corpus, batch_size, num_steps): 
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    offset = random.randint(0, num_steps - 1)  # [0, num_steps - 1]
    corpus = corpus[offset:]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    init_idx = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(init_idx)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        init_idx_per_batch = init_idx[i: i + batch_size]
        X = [data(j) for j in init_idx_per_batch]
        Y = [data(j + 1) for j in init_idx_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps - 1)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + num_tokens + 1])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_subseqs = Xs.shape[1] // num_steps  # num of subseqs in Xs.shape[1]
    for i in range(0, num_steps * num_subseqs, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


if __name__ == "__main__":
    tokens = tokenize(read_time_machine())
    # 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    print(vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
             xscale='log', yscale='log')
    plt.show()

    # bigram
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    # trigram
    trigram_tokens = [triple for triple in zip(corpus[: -2], corpus[1: -1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
             ylabel='frequency: n(x)', xscale='log', yscale='log',
             legend=['unigram', 'bigram', 'trigram'])
    plt.show()

    # test seq_data_iter_random
    my_seq = list(range(35))
    print("random iter")
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)
    # test seq_data_iter_sequential
    print("sequential iter")
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)







