#!/usr/bin/env python3
"""
验证波动率方向判定的问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import calculate_volatility

print("=" * 80)
print("验证波动率方向判定的问题")
print("=" * 80)

# 加载数据
symbols = ['CU0', 'SM0', 'I0', 'RB0', 'HC0']
data = load_data(
    symbols=symbols,
    start_date='2024-01-01',
    end_date='2024-08-20',
    columns=['close'],
    log_price=True  # 对数价格用于波动率计算
)

print("1. 检查价格和波动率:")
for symbol in symbols:
    col_name = f"{symbol}_close"
    if col_name in data.columns:
        # 原始价格（最新）
        raw_prices = load_data([symbol], '2024-08-15', '2024-08-20', ['close'], log_price=False)
        latest_price = raw_prices.iloc[-1, 0] if not raw_prices.empty else 0
        
        # 波动率
        log_prices = data[col_name].values
        dates = data.index
        vol = calculate_volatility(log_prices, dates, '2024-01-01')
        
        print(f"  {symbol}: 价格={latest_price:.0f}, 波动率={vol:.3f}")

print(f"\n2. 检查问题配对:")

# CU0-SM0
print("CU0-SM0配对:")
cu0_col = 'CU0_close'
sm0_col = 'SM0_close'

if cu0_col in data.columns and sm0_col in data.columns:
    cu0_vol = calculate_volatility(data[cu0_col].values, data.index, '2024-01-01')
    sm0_vol = calculate_volatility(data[sm0_col].values, data.index, '2024-01-01')
    
    print(f"  CU0波动率: {cu0_vol:.3f}")
    print(f"  SM0波动率: {sm0_vol:.3f}")
    
    if cu0_vol < sm0_vol:
        print(f"  → 按波动率: X=CU0(低波动), Y=SM0(高波动)")
        print(f"  → 回归: SM0 = α + β×CU0")
    else:
        print(f"  → 按波动率: X=SM0(低波动), Y=CU0(高波动)")
        print(f"  → 回归: CU0 = α + β×SM0")
        
    print(f"  但是！CU0价格(73770) >> SM0价格(6334)")
    print(f"  如果 SM0 = α + β×CU0，则β应该很小(~0.086)")
    print(f"  如果 CU0 = α + β×SM0，则β应该很大(~11.6)")

# RB0-I0  
print(f"\nRB0-I0配对:")
rb0_col = 'RB0_close'
i0_col = 'I0_close'

if rb0_col in data.columns and i0_col in data.columns:
    rb0_vol = calculate_volatility(data[rb0_col].values, data.index, '2024-01-01')
    i0_vol = calculate_volatility(data[i0_col].values, data.index, '2024-01-01')
    
    print(f"  RB0波动率: {rb0_vol:.3f}")
    print(f"  I0波动率: {i0_vol:.3f}")
    
    if rb0_vol < i0_vol:
        print(f"  → 按波动率: X=RB0(低波动), Y=I0(高波动)")
        print(f"  → 回归: I0 = α + β×RB0")
    else:
        print(f"  → 按波动率: X=I0(低波动), Y=RB0(高波动)")
        print(f"  → 回归: RB0 = α + β×I0")
        
    print(f"  但是！RB0价格(3172) >> I0价格(839)")
    print(f"  如果 I0 = α + β×RB0，价差会很奇怪")
    print(f"  更合理的是 RB0 = α + β×I0，β~3.8")

print(f"\n💡 问题总结:")
print("波动率方向判定忽略了价格量级差异！")
print("应该考虑:")
print("1. 价格量级：高价格做Y，低价格做X")
print("2. 或者使用相对波动率（变异系数 = std/mean）")
print("3. 或者价格标准化后再判断波动率")

print(f"\n🔧 可能的解决方案:")
print("1. 修改方向判定逻辑，优先考虑价格量级")
print("2. 使用价格标准化（Z-score）后的数据")
print("3. 手动指定合理的配对方向")
print("4. 使用相对价格而非绝对价格")