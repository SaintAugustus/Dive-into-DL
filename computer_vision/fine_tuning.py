import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def get_iter(train_augs, test_augs, batch_size=128):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, "train"), transform=train_augs),
                                    batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, "test"), transform=train_augs),
                                    batch_size=batch_size, shuffle=True)
    return train_iter, test_iter

def train_fine_tuning(net, train_iter, test_iter, lr, num_epoch=5, param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
     params_1x = [param for name, param in net.named_parameters()
                  if name not in ["fc.weight", "fc.bias"]]
     trainer = torch.optim.SGD([{'params': params_1x},
                                {'params': net.fc.parameters(), 'lr': lr * 10}],
                               lr=lr, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epoch, devices)


if __name__ == "__main__":
    d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                              'fba480ffa8aa7e0febbb511d181409f899b9baa5')
    data_dir = d2l.download_extract('hotdog')
    train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

    # 使用RGB通道的均值和标准差，以标准化每个通道
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    train_iter, test_iter = get_iter(train_augs, test_augs)
    train_fine_tuning(finetune_net, train_iter, test_iter, 5e-5)











