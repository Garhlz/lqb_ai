# 层归一化

import numpy as np


def layer_norm(X: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # 计算均值
    mean = np.mean(X, axis=-1, keepdims=True)
    # 计算方差
    var = np.var(X, axis=-1, keepdims=True)
    # 进行层归一化
    Y = (X - mean) / np.sqrt(var + epsilon)
    return Y
