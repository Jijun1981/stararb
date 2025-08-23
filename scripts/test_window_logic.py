#!/usr/bin/env python3
"""
测试滚动窗口的正确逻辑
"""

import numpy as np

# 模拟数据
residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 10个点
window = 5

print("测试滚动窗口逻辑")
print("=" * 50)
print(f"残差序列: {residuals}")
print(f"窗口大小: {window}")

# 方法1: 原始analyze_kalman_beta.py的方法
print(f"\n方法1 (原始): window_residuals = residuals[i-window:i]")
for i in range(len(residuals)):
    if i >= window:
        window_residuals = residuals[i-window:i]  # 不包含当前点
        mean = np.mean(window_residuals)
        std = np.std(window_residuals)
        z_score = (residuals[i] - mean) / std
        print(f"i={i}: 窗口{list(window_residuals)} -> 当前点{residuals[i]} -> Z={z_score:.3f}")

# 方法2: 包含当前点的60个窗口
print(f"\n方法2 (包含当前点): window_residuals = residuals[i-window+1:i+1]")
for i in range(len(residuals)):
    if i >= window - 1:  # 需要至少window个点
        window_residuals = residuals[i-window+1:i+1]  # 包含当前点
        mean = np.mean(window_residuals)
        std = np.std(window_residuals)
        current_residual = residuals[i]
        z_score = (current_residual - mean) / std
        print(f"i={i}: 窗口{list(window_residuals)} -> 当前点{residuals[i]} -> Z={z_score:.3f}")

print(f"\n哪种方法更合理？")
print("方法1: 用历史数据预测当前点的异常程度 (更常见的异常检测方法)")
print("方法2: 用包含当前点的窗口标准化当前点 (当前点会影响自己的Z-score)")