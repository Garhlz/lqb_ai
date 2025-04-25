# 优化SGD

from typing import Tuple

import torch
from torch.optim import SGD, Adam


def rosenbrock_function(
    x: torch.Tensor, y: torch.Tensor, a: float = 1.0, b: float = 100.0
) -> torch.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2


def find_rosenbrock_minimum(max_iter: int) -> Tuple[float, float, float, SGD]:
    # 初始化参数 x 和 y，设置 requires_grad=True 以计算梯度
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)

    # 创建 SGD 优化器，设置学习率和动量
    optimizer = SGD([x, y], lr=0.005, momentum=0.9)

    # 迭代优化
    for _ in range(max_iter):
        # 清空之前的梯度
        optimizer.zero_grad()

        # 计算 Rosenbrock 函数值（损失）
        loss = rosenbrock_function(x, y)

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数 x 和 y
        optimizer.step()

    # 计算最终的函数值
    final_loss = rosenbrock_function(x, y).item()

    # 返回结果：x 值、y 值、函数值、优化器实例
    return x.item(), y.item(), final_loss, optimizer


if __name__ == "__main__":
    x_min, y_min, f_min, sgd = find_rosenbrock_minimum(max_iter=1000)
    print(
        f"局部最小值点: x={x_min}, y={y_min}, 对应的函数值: f(x,y)={f_min}, SGD 参数: {sgd.state_dict()}"
    )
