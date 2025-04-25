**点积注意力（Dot-Product Attention**是一种广泛应用于自然语言处理和计算机视觉的注意力机制，模仿人类选择性关注信息的能力。它通过动态加权输入数据的不同部分，增强模型对关键信息的处理能力，从而提升性能。点积注意力是 Transformer 模型中常用的注意力计算方式，通过计算查询（Query）与键（Key）的点积来衡量相似度，进而为值（Value）分配权重。

现要求在 task.py 文件中实现点积注意力机制，完成 ScaledDotProductAttention 类的 forward 函数。根据点积注意力的示意图（图-1）和题目提供的逻辑，实现以下功能：

# 目标
## 类：
ScaledDotProductAttention（继承 nn.Module）。

## 函数：
forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tuple[Tensor, Tensor]。

## 功能：
计算点积注意力，返回：
1. Softmax 处理后的注意力分数（对应图-1 中的 output1）。
2. 点积注意力的最终输出（对应图-1 中的 output2）。
3. 
## 参数：
Q：查询张量，形状为 (batch_size, seq_len_q, d_model)。
K：键张量，形状为 (batch_size, seq_len_k, d_model)。
V：值张量，形状为 (batch_size, seq_len_k, d_model)。

## 返回值：
output1：Softmax 后的注意力分数，形状为 (batch_size, seq_len_q, seq_len_k)。
output2：点积注意力的最终输出，形状为 (batch_size, seq_len_q, d_model)。

## 注意：
无需实现掩码（Mask）操作。
仅使用 torch.nn 提供的 self.softmax（nn.Softmax(dim=-1)）。
禁止使用其他库或类对象。

# 输入输出示例
## 输入：
Q：(2, 10, 768)（批量大小 2，查询序列长度 10，模型维度 768）。
K：(2, 20, 768)（键序列长度 20）。
V：(2, 20, 768)。
## 输出：
output1：(2, 10, 20)（注意力分数矩阵）。
output2：(2, 10, 768)（加权后的值张量）。

# 约束
在 #TODO 范围内编写代码，删除 pass。
不得修改文件名、函数名或其他默认代码。
仅使用 self.softmax 和 PyTorch 基本张量操作。
满分 10 分，未满足要求得 0 分。

# 示例代码框架:
```python
import torch
from torch import nn, Tensor
import numpy as np
from typing import Tuple

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement scaled dot-product attention
        pass

if __name__ == "__main__":
    scaled_dot_product_attn = ScaledDotProductAttention()
    d_model = 768
    Q = torch.randn(2, 10, d_model)
    K = torch.randn(2, 20, d_model)
    V = torch.randn(2, 20, d_model)
    output1, output2 = scaled_dot_product_attn(Q, K, V)
    print(output1.shape)  # Expected: torch.Size([2, 10, 20])
    print(output2.shape)  # Expected: torch.Size([2, 10, 768])
```