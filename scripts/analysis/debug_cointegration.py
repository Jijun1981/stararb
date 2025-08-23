#!/usr/bin/env python3
"""
调试协整检验问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer, engle_granger_test

# 期货品种列表
SYMBOLS = ['AG0', 'AU0', 'CU0', 'HC0', 'I0', 'RB0']  # 简化测试

print("=" * 80)
print("协整检验调试")
print("=" * 80)

# 加载数据
data = load_data(
    symbols=SYMBOLS,
    start_date='2020-01-01',
    end_date='2025-08-20',
    columns=['close'],
    log_price=True
)

print(f"数据形状: {data.shape}")
print(f"数据范围: {data.index[0]} 至 {data.index[-1]}")

# 测试单个配对
print("\n测试配对: HC0-I0")
print("-" * 40)

hc_data = data['HC0_close'].values
i_data = data['I0_close'].values

print(f"HC0数据: 长度={len(hc_data)}, 非空={np.isfinite(hc_data).sum()}")
print(f"I0数据: 长度={len(i_data)}, 非空={np.isfinite(i_data).sum()}")

# 手动调用engle_granger_test
print("\n直接调用engle_granger_test:")
try:
    result = engle_granger_test(hc_data, i_data)
    print(f"  p值: {result['pvalue']:.6f}")
    print(f"  Beta: {result['beta']:.4f}")
    print(f"  残差std: {result['residual_std']:.4f}")
except Exception as e:
    print(f"  错误: {e}")

# 使用CointegrationAnalyzer
print("\n使用CointegrationAnalyzer:")
analyzer = CointegrationAnalyzer(data)

# 测试multi_window_test
print("\n测试multi_window_test:")
result = analyzer.multi_window_test(hc_data, i_data)
if result:
    print(f"  结果keys: {result.keys()}")
    if 'windows' in result:
        for window, res in result['windows'].items():
            print(f"  {window}年窗口: p={res.get('pvalue', 'N/A')}, beta={res.get('beta', 'N/A')}")
else:
    print("  结果为None")

# 测试不同时间窗口
print("\n手动测试不同窗口:")
for years in [5, 3, 1]:
    days = years * 252
    if days <= len(hc_data):
        end_idx = len(hc_data)
        start_idx = max(0, end_idx - days)
        
        hc_window = hc_data[start_idx:end_idx]
        i_window = i_data[start_idx:end_idx]
        
        try:
            result = engle_granger_test(hc_window, i_window)
            print(f"  {years}年窗口: p={result['pvalue']:.6f}, beta={result['beta']:.4f}")
        except Exception as e:
            print(f"  {years}年窗口: 错误 - {e}")

# 使用screen_all_pairs看整体结果
print("\n使用screen_all_pairs筛选:")
filtered = analyzer.screen_all_pairs(p_threshold=0.05)
print(f"筛选结果: {len(filtered)} 个配对")
if len(filtered) > 0:
    print("\n前5个配对:")
    for i, row in filtered.head(5).iterrows():
        print(f"  {row['pair']}: p1y={row.get('pvalue_1y', 'N/A')}, p5y={row.get('pvalue_5y', 'N/A')}")