#!/usr/bin/env python3
"""
验证Z-score计算逻辑的正确性
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.signal_generation import SignalGenerator

print("=" * 80)
print("验证Z-score计算逻辑")
print("=" * 80)

# 读取原始数据
pair_file = "/mnt/e/Star-arb/output/kalman_analysis/RB0_I0_kalman_analysis.csv"
df = pd.read_csv(pair_file)

# 获取残差序列
residuals = df['residual'].values
original_z_scores = df['z_score'].values

print(f"数据长度: {len(residuals)}")

# 手动实现原始逻辑
def original_zscore_calculation(residuals, window=60):
    """原始的Z-score计算逻辑"""
    z_scores = []
    
    for i in range(len(residuals)):
        if i < window:
            z_scores.append(0)
        else:
            window_residuals = residuals[i-window:i]  # 前window个点
            mean = np.mean(window_residuals)
            std = np.std(window_residuals)  # ddof=0
            if std > 0:
                z_score = (residuals[i] - mean) / std  # 当前点
            else:
                z_score = 0
            z_scores.append(z_score)
    
    return np.array(z_scores)

# 重新计算
manual_z_scores = original_zscore_calculation(residuals, 60)

print(f"\n对比结果:")
print(f"原始非零Z-score数量: {np.sum(original_z_scores != 0)}")
print(f"手动计算非零Z-score数量: {np.sum(manual_z_scores != 0)}")

# 检查差异
diff = abs(original_z_scores - manual_z_scores)
max_diff = np.max(diff)
print(f"最大差异: {max_diff:.10f}")

if max_diff < 1e-10:
    print("✅ 手动计算与原始结果完全一致!")
else:
    print("❌ 存在差异")
    # 显示差异点
    diff_indices = np.where(diff > 1e-6)[0]
    print(f"差异点数量: {len(diff_indices)}")
    if len(diff_indices) > 0:
        for idx in diff_indices[:5]:
            print(f"  索引{idx}: 原始={original_z_scores[idx]:.6f}, 手动={manual_z_scores[idx]:.6f}")

# 测试修正后的SignalGenerator
print(f"\n测试修正后的SignalGenerator:")
sg = SignalGenerator(window=60)

# 选择几个测试点
test_indices = [136, 137, 138, 139, 140]  # 2023-07-27等高Z-score点

for idx in test_indices:
    if idx >= 60:
        # 提供从开始到当前点的残差序列
        residuals_up_to_idx = residuals[:idx+1]
        
        # 使用SignalGenerator计算
        sg_z_score = sg.calculate_zscore(residuals_up_to_idx, 60)
        
        # 原始结果
        orig_z_score = original_z_scores[idx]
        manual_z_score = manual_z_scores[idx]
        
        print(f"索引{idx}:")
        print(f"  原始: {orig_z_score:.6f}")
        print(f"  手动: {manual_z_score:.6f}")  
        print(f"  SG: {sg_z_score:.6f}")
        
        # 验证计算细节
        window_res = residuals[idx-60:idx]  # 前60个点
        current_res = residuals[idx]         # 当前点
        
        manual_mean = np.mean(window_res)
        manual_std = np.std(window_res)  # ddof=0
        manual_z = (current_res - manual_mean) / manual_std if manual_std > 0 else 0
        
        print(f"  验证计算: mean={manual_mean:.6f}, std={manual_std:.6f}, z={manual_z:.6f}")
        print(f"  匹配: {'✅' if abs(sg_z_score - manual_z) < 1e-10 else '❌'}")
        print()

print("=" * 80)
print("结论")
print("=" * 80)
print("Z-score计算逻辑:")
print("1. 使用前60个点计算历史统计量（均值、标准差）")
print("2. 用当前点的残差减去历史均值，再除以历史标准差")
print("3. 这样可以判断当前点相对于历史分布的偏离程度")