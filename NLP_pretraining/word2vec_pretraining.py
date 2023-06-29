import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

from datasets.word_embedding_dataset import load_data_ptb
from utils.Functions import SigmoidBCELoss


def skip_gram(center, contexts_negatives, embed_v, embed_u):
    """
    p(w_o|w_c) = exp(u_o @ v_c) / sum(exp(u_i @ v_c))
    """
    v = embed_v(center)
    u = embed_u(contexts_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)

    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = loss(pred.reshape(label.shape).float(), label.float(), mask)
            l = (l / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')



if __name__ == "__main__":
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size,
                                         num_noise_words)

    # test skip_gram
    embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
    print(f'Parameter embedding_weight ({embed.weight.shape}, '
          f'dtype={embed.weight.dtype})')
    print(skip_gram(torch.ones((2, 1), dtype=torch.long),
              torch.ones((2, 4), dtype=torch.long), embed, embed).shape)

    # test SigmoidBCELoss
    pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
    label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    loss = SigmoidBCELoss()
    print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))

    # train
    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    lr, num_epochs = 0.002, 5
    train(net, data_iter, lr, num_epochs)

    # test get_similar_tokens
    get_similar_tokens('chip', 3, net[0])











