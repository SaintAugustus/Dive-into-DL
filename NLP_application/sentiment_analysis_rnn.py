import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

from NLP_pretraining.similarity_analogy import TokenEmbedding
from datasets.sentiment_analysis_dataset import load_data_imdb
from utils.train_ch13 import train_ch13, predict_sentiment


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size,
                 num_hiddens, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens,
                               num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs




if __name__ == "__main__":
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)

    embed_size, num_hiddens, num_layers = 100, 100, 2
    device = d2l.try_gpu()
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(init_weights)

    # load pretrained wordvec
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print(embeds.shape)

    # fix the pretrained weight of embedding
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    # train
    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, device)
    plt.show()

    # test
    print(predict_sentiment(net, vocab, 'this movie is so great'))
    print(predict_sentiment(net, vocab, 'this movie is so bad'))













