LeNet-5 是一种经典的卷积神经网络（CNN）架构，由 Yann LeCun 等人于 1998 年在论文《Gradient-Based Learning Applied to Document Recognition》中提出，广泛应用于手写数字识别任务。作为卷积神经网络的奠基之作，LeNet-5 通过多层结构提取图像特征，能够自动学习图像中的关键信息，标志着深度学习在图像识别领域的开端。LeNet 系列包括从 LeNet-1 到 LeNet-5 的多个版本，其中 LeNet-5 以其优化的架构和性能成为最知名的代表。

现要求从头构建 LeNet-5 模型，并将其应用于一个分类任务。根据 LeNet-5 原论文中的模型架构图（图-1）以及小蓝老师手绘的流程图（图-2），在 task.py 文件中完成 LeNet5 类的实现。LeNet5 类需要实现以下两个函数：

__init__(self, num_classes: int = 10, in_channels: int = 1, H: int = 32, W: int = 32) -> None：
功能：初始化 LeNet-5 模型，定义网络的各层结构。
参数：
num_classes：分类任务的类别数量，默认为 10（如 MNIST 手写数字的 0-9）。
in_channels：输入图像的通道数，默认为 1（如灰度图像）。
H：输入图像的高度，默认为 32。
W：输入图像的宽度，默认为 32。
要求：根据流程图（图-2）定义卷积层、池化层和全连接层。

forward(self, x: Tensor) -> Tuple[Tensor, Tensor]：
功能：定义前向传播过程，处理输入图像并返回中间特征和最终输出。
参数：
x：输入图像张量，形状为 (batch_size, channels, H, W)，其中：
batch_size：批量大小，表示一次输入的样本数量。
channels：输入图像的通道数（如 1 表示灰度图像）。
H：图像高度。
W：图像宽度。
返回值：
第一个全连接层的输入张量（形状为 (batch_size, 16, 5, 5)）。
模型最终输出张量（形状为 (batch_size, num_classes)，如 (batch_size, 10)）。
要求：按照流程图实现前向传播，确保返回正确的中间特征和输出。

流程图（图-2）说明：
Convolution (in_channels, out_channels, k, s)：
二维卷积操作，输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为 k×k，步长为 s。
Subsampling (k, s)：
平均池化操作，窗口大小为 k×k，步长为 s。
Full Connection (in_features, out_features)：
全连接层，输入特征维度为 in_features，输出特征维度为 out_features。
Gaussian Connections (in_features, out_features)：
全连接层，功能与普通全连接层相同，输入和输出特征维度分别为 in_features 和 out_features。

注意事项：
每个卷积操作后必须接一个 Sigmoid 激活函数。
仅使用 torch.nn 模块（nn）中的函数或类（如 nn.Conv2d、nn.AvgPool2d），禁止使用 torch.nn.functional（F）或其他库。
输入示例：
输入图像形状：(8, 1, 32, 32)（批量大小 8，单通道，32×32 像素）。
输出：
第一个全连接层的输入：(8, 16, 5, 5)。
模型最终输出：(8, 10)（假设 num_classes=10）。
代码必须在 #TODO 范围内编写，删除 pass，不得修改文件名、函数名或其他默认代码。
示例代码框架：

```python
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, H: int = 32, W: int = 32) -> None:
        super(LeNet5, self).__init__()
        # TODO: Define layers
        pass

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement forward pass
        pass

if __name__ == "__main__":
    img = torch.Tensor(8, 1, 32, 32)
    lenet5 = LeNet5(10, img.size(1), img.size(2), img.size(3))
    x, y = lenet5(img)
    print(x.shape)  # Expected: torch.Size([8, 16, 5, 5])
    print(y.shape)  # Expected: torch.Size([8, 10])
```