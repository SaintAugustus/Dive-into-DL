import torch
import torchvision
from torch import nn
from d2l import torch as d2l

from PIL import Image

d2l.set_figsize()
img = d2l.Image.open('../../d2l-zh/pytorch/img/cat1.jpg')
d2l.plt.imshow(img)

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size=256)
    test_iter = load_cifar10(False, test_augs, batch_size=256)
    loss = nn.CrossEntropyLoss(reduction="none")



























