import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def train_recsys_rating(net, train_iter, test_iter, loss_fun, optimizer, num_epochs,
                        device=d2l.try_gpu(), evaluator=None, **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 20],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for users, items, ratings in train_iter:
            timer.start()
            users, items, ratings = users.to(device), \
                items.to(device), ratings.to(device)
            optimizer.zero_grad()
            output = net(users, items)
            loss = loss_fun(output, ratings.float()) # reduction = 'sum'
            loss.backward()
            optimizer.step()
            metric.add(loss, users.shape[0])
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'], device)
        else:
            test_rmse = evaluator(net, test_iter, device)
        train_l = metric[0] / metric[1]
        animator.add(epoch + 1, (train_l, test_rmse))

    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

















