import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net)
print(net[2].state_dict())

print("\n" + "net 2 bias: ")
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None)

print("\n" + "all parameters: ")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block {i}", block1())
    return net

print("\n" + "rgnet: ")
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print(rgnet[0][1][0])
print(rgnet[0][1][0].weight.shape)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)

net[0].apply(init_constant)
net[1].apply(init_normal)
net[2].apply(init_xavier)

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


# share params
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))


