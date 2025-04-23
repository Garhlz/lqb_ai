import torch
from d2l import torch as d2l
from torch import nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def test_conv2d():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    # 前两个参数表示通道数, 使得操作很简单
    # 输出的shape为 (h0 - k + 1 + p + s) // s 包括了卷积核, 填充, 步长
    X = torch.rand(size=(8, 8))
    res = comp_conv2d(conv2d, X)
    print(res.shape)
    print(X)
    print(res)


def pool2d(X, pool_size, mode="max"):
    ph, pw = pool_size
    Y = torch.zeros(X.shape[0] - ph + 1, X.shape[1] - pw + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i + i + ph, j : j + pw].max()
            elif mode == "avg":
                Y[i, j] = X[i : i + ph, j : j + pw].mean()
    return Y
