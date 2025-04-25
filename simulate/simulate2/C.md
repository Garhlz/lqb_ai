图像加载是计算机视觉和深度学习中的基础步骤，涉及将磁盘上的图像文件读取到内存，并转换为深度学习模型能够处理的格式。在实际应用中，图像加载不仅限于文件读取，还包括格式转换、预处理和标准化等操作。图像通常以文件形式存储（如 .jpg、.png），需要通过工具或库将其转换为数组或张量，以支持后续的模型训练和处理。

# 目标
在基于 PyTorch 框架的计算机视觉任务中，加载后的图像需要满足以下要求：

图像被读取并转换为 PyTorch 张量（Tensor）。
彩色图像采用 RGB 格式。
张量形状为 (C, H, W)，其中：
C：通道数，通常为 3（对应 RGB）。
H：图像高度。
W：图像宽度。
每个像素值归一化到 [0, 1] 范围。
现要求在 task.py 文件中实现以下函数：

**def load_image(file_path: str) -> Tensor:**

## 函数功能：
根据给定的图像文件路径加载图像，并返回符合上述要求的 PyTorch 张量。

## 参数说明：
file_path（str）：图像文件的绝对路径。

## 返回值：
PyTorch 张量（torch.Tensor），表示加载的图像，形状为 (C, H, W)，像素值在 [0, 1] 范围内。

## 提示：
可以使用熟悉的图像加载库（如 PIL、OpenCV）读取图像。
需要结合 PyTorch 的工具（如 torchvision.transforms）进行格式转换和归一化。
最终返回的张量必须满足题目要求的格式。

# 代码框架
```python
from PIL import Image
from torch import Tensor
import numpy as np
from torchvision import transforms

def load_image(file_path: str) -> Tensor:
    # TODO: 实现图像加载
    pass

if __name__ == "__main__":
    file_path = 'img.jpg'
    img = load_image(file_path)
    print(img.shape)  # 示例输出：torch.Size([3, 334, 500])
    print(type(img))  # 输出：<class 'torch.Tensor'>
```