from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# 计算公式: H_out = (H_in - kernel_size + 2 * padding) // stride + 1
class LeNet5(nn.Module):
    def __init__(
        self, num_classes: int = 10, in_channels: int = 1, H: int = 32, W: int = 32
    ) -> None:
        super(LeNet5, self).__init__()
        # Feature extraction layers (C1, S2, C3, S4)
        self.features = nn.Sequential(
            # C1: Convolution (in_channels -> 6, 5x5, stride 1) + Sigmoid
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
            nn.Sigmoid(),
            # S2: Average pooling (2x2, stride 2)
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: Convolution (6 -> 16, 5x5, stride 1) + Sigmoid
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Sigmoid(),
            # S4: Average pooling (2x2, stride 2)
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Classification layers (C5, F6, Output)
        self.classifier = nn.Sequential(
            # C5: Convolution (16 -> 120, 5x5, stride 1) + Sigmoid
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Sigmoid(),
            # Flatten for fully connected layers
            nn.Flatten(),
            # F6: Fully connected (120 -> 84) + Sigmoid
            nn.Linear(120, 84),
            nn.Sigmoid(),
            # Output: Fully connected (84 -> num_classes)
            nn.Linear(84, num_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Pass through feature extraction layers (up to S4)
        intermediate = self.features(x)

        # Pass through classification layers
        output = self.classifier(intermediate)

        return intermediate, output


if __name__ == "__main__":
    img = torch.Tensor(8, 1, 32, 32)
    lenet5 = LeNet5(10, img.size(1), img.size(2), img.size(3))
    x, y = lenet5(img)
    print(x.shape)  # Expected: torch.Size([8, 16, 5, 5])
    print(y.shape)  # Expected: torch.Size([8, 10])
