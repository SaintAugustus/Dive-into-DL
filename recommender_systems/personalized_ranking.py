import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class BPRLoss(nn.MSELoss):
    def forward(self, positive, negative):
        distances = positive - negative
        loss = -torch.sum(torch.log(F.sigmoid(distances)),
                          dim=0, keepdim=True)
        return loss

class HingeLossbRec(nn.MSELoss):
    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = torch.sum(torch.maximum(- distances + margin, 0),
                         dim=0, keepdim=True)
        return loss









