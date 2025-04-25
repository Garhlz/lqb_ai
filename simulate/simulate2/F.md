在机器学习和数值优化中，梯度下降法（Gradient Descent）是一种核心算法，通过沿目标函数负梯度方向迭代更新变量，逐步逼近函数的最优解（最小值或最大值）。随机梯度下降（Stochastic Gradient Descent, SGD）是其常用变种，支持批量更新和动量机制，能够加速收敛并提高稳定性。PyTorch 提供的 SGD 优化器是一个通用的优化工具，支持学习率和动量等参数调节，适用于多种优化场景。

本任务要求使用 SGD 优化器对 Rosenbrock 函数进行优化，通过调整优化器参数，在指定迭代次数内逼近函数的局部最优解。Rosenbrock 函数是一个经典的非凸优化测试函数，其全局最小值位于 (x,y)=(1,1)，函数值为 0。

# 目标
1. 实现 find_rosenbrock_minimum 函数：
参数：
max_iter：最大迭代次数，控制优化步数。

功能：
使用 SGD 优化器迭代更新 Rosenbrock 函数的变量x和y。
在指定迭代次数内寻找局部最优值。

返回值：
一个包含四个值的元组：最终的x值、𝑦值、在最终点的 Rosenbrock 函数值、使用的 SGD 优化器实例。

2. 寻找 SGD 最优参数：
在 find_rosenbrock_minimum 函数中，通过调整 SGD 优化器的超参数（如学习率、动量），在最大迭代次数后尽可能逼近 Rosenbrock 函数的局部最优值（即f(x,y)≈0）

# 代码框架
```python
import torch
from torch.optim import Adam, SGD
from typing import Tuple

def rosenbrock_function(x: torch.Tensor, y: torch.Tensor, a: float=1., b: float=100.) -> torch.Tensor:
    return (a - x)**2 + b * (y - x**2)**2

def find_rosenbrock_minimum(max_iter: int) -> Tuple[float, float, float, SGD]:
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)
    # TODO: Implement SGD optimization
    pass

if __name__ == "__main__":
    x_min, y_min, f_min, sgd = find_rosenbrock_minimum(max_iter=1000)
    print(f"局部最小值点: x={x_min}, y={y_min}, 对应的函数值: f(x,y)={f_min}, SGD 参数: {sgd.state_dict()}")
```
# 约束:
在 #TODO 范围内编写代码，删除 pass。
不得修改文件名、函数名、初始位置 (x=0,y=0)、Rosenbrock 函数参数 (a=1,b=100)、最大迭代次数 (max iter=1000)。
仅使用 torch 和 torch.optim.SGD。

# 判分标准
满分：20 分。
目标 1（实现函数）：5 分。
目标 2（优化效果）：根据最终函数值与局部最优值（0）的绝对差值评分
