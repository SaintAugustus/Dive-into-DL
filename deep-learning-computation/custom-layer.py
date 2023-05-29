import torch
import torch.nn.functional as F
from torch import nn

# non params layers
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
print(net)

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_units, units))
        self.bias = nn.Parameter(torch.rand(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net)



