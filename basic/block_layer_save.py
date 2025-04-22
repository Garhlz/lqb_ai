import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(
            torch.randn(
                units,
            )
        )

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


def test_load_and_save():
    def save(X):
        net = MLP()
        Y = net(X)  # 调用了forward函数
        print(Y)
        torch.save(net.state_dict(), "mlp.params")
        return Y

    def load(X):
        clone = MLP()
        clone.load_state_dict(torch.load("mlp.params"))
        print(clone)
        Y_clone = clone(X)
        return Y_clone

    X = torch.randn(size=(2, 20))
    a = save(X)
    b = load(X)

    print(a == b)
