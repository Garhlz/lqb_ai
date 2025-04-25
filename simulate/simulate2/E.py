# 实现点积注意力
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tuple[Tensor, Tensor]:
        # Get dimensions
        d_k = Q.size(-1)  # Model dimension (e.g., 768)

        # Compute scaled dot-product: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float)
        )

        # Apply Softmax to get attention weights (output1)
        attention_weights = self.softmax(scores)

        # Compute weighted sum: attention_weights * V (output2)
        output = torch.matmul(attention_weights, V)

        return attention_weights, output


if __name__ == "__main__":
    scaled_dot_product_attn = ScaledDotProductAttention()
    d_model = 768
    Q = torch.randn(2, 10, d_model)
    K = torch.randn(2, 20, d_model)
    V = torch.randn(2, 20, d_model)
    output1, output2 = scaled_dot_product_attn(Q, K, V)
    print(output1.shape)  # Expected: torch.Size([2, 10, 20])
    print(output2.shape)  # Expected: torch.Size([2, 10, 768])
