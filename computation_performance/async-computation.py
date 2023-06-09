import os
import subprocess
import numpy
import torch
from torch import nn
from d2l import torch as d2l

from utils.Benchmark import Benchmark

device = torch.device("mps")
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)



if __name__ == "__main__":
    with d2l.Benchmark('numpy'):
        for _ in range(10):
            a = numpy.random.normal(size=(1000, 1000))
            b = numpy.dot(a, a)

    with d2l.Benchmark('torch'):
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)

    with d2l.Benchmark():
        for _ in range(10):
            a = torch.randn(size=(1000, 1000), device=device)
            b = torch.mm(a, a)
        #torch.cuda.synchronize(device)

