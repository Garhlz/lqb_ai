from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def group_images_by_size(
    images_dict: Dict[str, np.ndarray],
) -> Dict[Tuple[int, int], List[Tuple[str, np.ndarray]]]:
    # 初始化 defaultdict，键为尺寸元组，值为图像列表
    grouped = defaultdict(list)

    # 遍历图像字典
    for name, image in images_dict.items():
        # 获取图像尺寸 (height, width)
        size = tuple(image.shape[:2])
        # 添加 (名称, 图像) 到对应尺寸组
        grouped[size].append((name, image))

    # 转换为普通字典返回
    return dict(grouped)


def grouped_inference(
    images_dict: Dict[str, np.ndarray], filename: str
) -> Dict[str, np.ndarray]:
    # 调用 group_images_by_size 分组
    grouped_images = group_images_by_size(images_dict)
    print(
        [
            (size, len(images), images[0][1].shape)
            for size, images in grouped_images.items()
        ]
    )

    # 加载 ONNX 模型
    session = ort.InferenceSession(filename)

    # 初始化结果字典
    results = {}

    # 遍历每个尺寸分组
    for size, image_list in grouped_images.items():
        # 提取图像数据，转换为批量 (N, 3, H, W)
        images = np.stack([img for _, img in image_list])  # Shape: (N, H, W, 3)
        images = images.transpose(0, 3, 1, 2)  # Shape: (N, 3, H, W)

        # 执行 ONNX 推理
        input_name = session.get_inputs()[0].name  # 通常为 "input"
        outputs = session.run(None, {input_name: images})[0]  # Shape: (N, 3, 2*H, 2*W)

        # 将输出转换回 (2*H, 2*W, 3) 并存入结果字典
        outputs = outputs.transpose(0, 2, 3, 1)  # Shape: (N, 2*H, 2*W, 3)
        for (name, _), output in zip(image_list, outputs):
            results[name] = output

    return results


def main() -> None:
    filename = "srcnn.onnx"
    images = {
        f"image{i}": np.random.random([128 * (i % 3 + 1), 128 * (i % 2 + 1), 3]).astype(
            np.float32
        )
        for i in range(16)
    }
    print([(file, image.shape) for file, image in images.items()])
    images = grouped_inference(images, filename)
    print([(file, image.shape) for file, image in images.items()])


if __name__ == "__main__":
    main()
