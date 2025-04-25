本任务的目标是实现高效的图像超分辨率推理流程，通过将图像按尺寸分组并进行批量推理，提升 ONNX 超分辨率模型（srcnn.onnx）的处理效率。任务分为两个子目标：首先实现按图像尺寸分组的函数，然后基于分组结果对每组图像进行批量推理，最终返回推理后的图像数据。

# 目标:
1. 实现 group_images_by_size 函数：
参数：
images_dict：字典，键为图像名称（字符串），值为图像数据（np.ndarray，形状为 (height, width, 3)，表示高度、宽度和 RGB 通道）。
功能：
遍历 images_dict，根据图像的高度和宽度（作为元组 (height, width)）分组。
将具有相同尺寸的图像归入同一组。
返回值：
字典，键为尺寸元组 (height, width)，值为列表，列表元素为 (图像名称, 图像数据) 的元组。

2. 实现 grouped_inference 函数：
参数：
images_dict：同上。
filename：ONNX 模型文件路径（如 srcnn.onnx）。
功能：
调用 group_images_by_size 函数对 images_dict 分组。
加载指定的 ONNX 模型。
遍历每个尺寸分组，批量推理该组图像。
模型推理次数等于分组数（优化效率）。
将推理结果存储为输出字典。
返回值：
字典，键为图像名称，值为推理后的图像数据（np.ndarray，形状为 (new_height, new_width, 3)，通常为输入尺寸的 2 倍）。

# 提供代码框架
```python
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict

def group_images_by_size(images_dict: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]]:
    # TODO: Group images by size
    pass

def grouped_inference(images_dict: Dict[str, np.ndarray], filename: str) -> Dict[str, np.ndarray]:
    grouped_images = group_images_by_size(images_dict)
    print([(size, len(images), images[0][1].shape) for size, images in grouped_images.items()])
    # TODO: Perform grouped inference
    pass

def main() -> None:
    filename = 'srcnn.onnx'
    images = {f'image{i}': np.random.random([128*(i%3+1), 128*(i%2+1), 3]).astype(np.float32) for i in range(16)}
    print([(file, image.shape) for file, image in images.items()])
    images = grouped_inference(images, filename)
    print([(file, image.shape) for file, image in images.items()])

if __name__ == '__main__':
    main()
```

# 示例输入输出:
## 输入：
images_dict：包含 16 张图像，名称为 image0 到 image15，尺寸为：
(128, 128, 3), (256, 256, 3), (384, 128, 3), (128, 256, 3), (256, 128, 3), (384, 256, 3) 等。

示例（部分）：
[('image0', (128, 128, 3)), ('image1', (256, 256, 3)), ('image2', (384, 128, 3)), ...]

## 分组输出（group_images_by_size）：

按尺寸分组，示例：
[((128, 128), 3, (128, 128, 3)), ((256, 256), 3, (256, 256, 3)), ((384, 128), 3, (384, 128, 3)),
 ((128, 256), 3, (128, 256, 3)), ((256, 128), 2, (256, 128, 3)), ((384, 256), 2, (384, 256, 3))]
表示 6 个尺寸组，每组包含 2-3 张图像。

## 推理输出（grouped_inference）：
推理后图像尺寸为输入的 2 倍（如 (128, 128, 3) 变为 (256, 256, 3)）。
示例（部分）：
[('image0', (256, 256, 3)), ('image1', (512, 512, 3)), ('image2', (768, 256, 3)), ...]


# 约束:
在 #TODO 范围内编写代码，删除 pass。
不得修改文件名、函数名或其他默认代码。
输入图像数据直接用于推理，无需预处理。
推理次数等于分组数。

# 判分标准:
满分：10 分。
目标 1（group_images_by_size）：4 分。
目标 2（grouped_inference）：6 分。
未遵守约束：0 分。