import torch

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.FloatTensor(num_examples, len(w)).uniform_(0, 1)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features.shape, '\nlabel:', labels.shape)