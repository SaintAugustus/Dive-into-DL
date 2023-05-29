import torch
from torch import nn

print(torch.device('cpu'))
print(torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def try_all_gpus():
    devices = [torch.device(f"cuda{i}")
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]

X = torch.ones(2, 3)
X = X.to(torch.device("mps"))
print("X device:", X.device)

# copy net to device
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net[0].weight.data.device)