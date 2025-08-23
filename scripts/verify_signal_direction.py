#!/usr/bin/env python3
"""
验证信号方向的正确性
特别关注负Beta的处理
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("配对交易信号方向验证")
print("=" * 80)

# =========================
# 理论基础
# =========================
print("\n【理论基础】")
print("-" * 60)
print("价差定义: spread = Y - β*X")
print("Z-score定义: z = (spread - mean) / std")
print("")
print("关键逻辑:")
print("1. 当Z > 0时，spread偏高")
print("2. 当Z < 0时，spread偏低")
print("3. 我们期待spread回归均值")

# =========================
# 案例1: 正Beta情况
# =========================
print("\n" + "=" * 80)
print("案例1: 正Beta (β = 0.8)")
print("-" * 60)

# 模拟数据
beta_pos = 0.8
X_price = 100
Y_price_normal = 80  # 正常情况: Y = 0.8 * X
Y_price_high = 85    # Y偏高
Y_price_low = 75     # Y偏低

spread_normal = Y_price_normal - beta_pos * X_price
spread_high = Y_price_high - beta_pos * X_price  
spread_low = Y_price_low - beta_pos * X_price

print(f"X价格: {X_price}")
print(f"Beta: {beta_pos}")
print(f"")
print(f"情况1 - 正常: Y={Y_price_normal}, spread={spread_normal:.2f}")
print(f"情况2 - Y偏高: Y={Y_price_high}, spread={spread_high:.2f}")
print(f"情况3 - Y偏低: Y={Y_price_low}, spread={spread_low:.2f}")

print(f"\n交易逻辑:")
print(f"情况2 (spread={spread_high:.2f} > 0):")
print(f"  → Z-score > 0 (假设超过阈值)")
print(f"  → spread偏高，预期回归")
print(f"  → 做空价差 = 卖Y + 买{beta_pos}份X")
print(f"  → 信号: open_short")

print(f"\n情况3 (spread={spread_low:.2f} < 0):")
print(f"  → Z-score < 0 (假设超过阈值)")
print(f"  → spread偏低，预期回归")
print(f"  → 做多价差 = 买Y + 卖{beta_pos}份X")
print(f"  → 信号: open_long")

# =========================
# 案例2: 负Beta情况（关键！）
# =========================
print("\n" + "=" * 80)
print("案例2: 负Beta (β = -0.5) ⚠️ 关键案例")
print("-" * 60)

beta_neg = -0.5
X_price = 100
Y_price_normal = -50  # 正常情况: Y = -0.5 * X (注意是负相关)
Y_price_high = -45    # Y偏高（相对其正常负值）
Y_price_low = -55     # Y偏低

# 注意：实际价格应该是正的，这里只是为了演示
# 实际情况：假设Y和X负相关
X_real = 100
Y_real_normal = 50  # X=100时，Y通常是50
Y_real_high = 55    # Y偏高
Y_real_low = 45     # Y偏低

spread_normal_neg = Y_real_normal - beta_neg * X_real  # 50 - (-0.5)*100 = 50 + 50 = 100
spread_high_neg = Y_real_high - beta_neg * X_real      # 55 - (-0.5)*100 = 55 + 50 = 105
spread_low_neg = Y_real_low - beta_neg * X_real        # 45 - (-0.5)*100 = 45 + 50 = 95

print(f"X价格: {X_real}")
print(f"Beta: {beta_neg} (负相关)")
print(f"")
print(f"情况1 - 正常: Y={Y_real_normal}, spread={spread_normal_neg:.2f}")
print(f"情况2 - Y偏高: Y={Y_real_high}, spread={spread_high_neg:.2f}")
print(f"情况3 - Y偏低: Y={Y_real_low}, spread={spread_low_neg:.2f}")

print(f"\n交易逻辑（β为负时）:")
print(f"情况2 (spread={spread_high_neg:.2f} > {spread_normal_neg}):")
print(f"  → Z-score > 0")
print(f"  → spread偏高，预期回归")
print(f"  → 做空价差 = 卖Y + 买({beta_neg})份X")
print(f"  → 由于β<0，买负份相当于卖")
print(f"  → 实际操作: 卖Y + 卖{abs(beta_neg)}份X")
print(f"  → 信号: open_short")

print(f"\n情况3 (spread={spread_low_neg:.2f} < {spread_normal_neg}):")
print(f"  → Z-score < 0")
print(f"  → spread偏低，预期回归")
print(f"  → 做多价差 = 买Y + 卖({beta_neg})份X")
print(f"  → 由于β<0，卖负份相当于买")
print(f"  → 实际操作: 买Y + 买{abs(beta_neg)}份X")
print(f"  → 信号: open_long")

# =========================
# 实际数据验证
# =========================
print("\n" + "=" * 80)
print("实际数据验证")
print("-" * 60)

# 读取协整结果
coint_df = pd.read_csv('/mnt/e/Star-arb/output/pipeline_shifted/cointegration_results.csv')

# 找出负Beta的配对
neg_beta_pairs = coint_df[coint_df['beta_4y'] < 0]
print(f"\n负Beta配对 ({len(neg_beta_pairs)}个):")
for idx, row in neg_beta_pairs.iterrows():
    print(f"  {row['pair']}: β4y={row['beta_4y']:.3f}, β1y={row['beta_1y']:.3f}")

# 详细分析一个负Beta配对
if len(neg_beta_pairs) > 0:
    example = neg_beta_pairs.iloc[0]
    print(f"\n详细分析: {example['pair']}")
    print(f"  4年Beta: {example['beta_4y']:.3f}")
    print(f"  1年Beta: {example['beta_1y']:.3f}")
    
    # 读取该配对的Kalman分析数据（如果存在）
    pair_file = f"/mnt/e/Star-arb/output/kalman_analysis/{example['symbol_x']}_{example['symbol_y']}_kalman_analysis.csv"
    if Path(pair_file).exists():
        df = pd.read_csv(pair_file)
        
        # 找Z-score超阈值的点
        high_z = df[df['z_score'] > 2.0].head(3)
        low_z = df[df['z_score'] < -2.0].head(3)
        
        if len(high_z) > 0:
            print(f"\n  Z-score > 2.0的情况:")
            for _, row in high_z.iterrows():
                print(f"    {row['date']}: Z={row['z_score']:.3f}")
                print(f"      → 做空价差: 卖{example['symbol_y']} + 卖{abs(example['beta_4y']):.2f}份{example['symbol_x']}")
        
        if len(low_z) > 0:
            print(f"\n  Z-score < -2.0的情况:")
            for _, row in low_z.iterrows():
                print(f"    {row['date']}: Z={row['z_score']:.3f}")
                print(f"      → 做多价差: 买{example['symbol_y']} + 买{abs(example['beta_4y']):.2f}份{example['symbol_x']}")

# =========================
# 代码验证
# =========================
print("\n" + "=" * 80)
print("代码逻辑验证")
print("-" * 60)

def correct_signal_logic(z_score, beta):
    """正确的信号逻辑"""
    if abs(z_score) < 2.0:
        return 'hold'
    
    if z_score > 2.0:  # spread偏高
        # 做空价差
        if beta > 0:
            action = f"卖Y + 买{beta:.2f}份X"
        else:
            action = f"卖Y + 卖{abs(beta):.2f}份X"
        return 'open_short', action
    
    if z_score < -2.0:  # spread偏低
        # 做多价差
        if beta > 0:
            action = f"买Y + 卖{beta:.2f}份X"
        else:
            action = f"买Y + 买{abs(beta):.2f}份X"
        return 'open_long', action
    
    return 'hold', ''

# 测试案例
test_cases = [
    (2.5, 0.8, "正Beta, Z>2"),
    (-2.5, 0.8, "正Beta, Z<-2"),
    (2.5, -0.5, "负Beta, Z>2"),
    (-2.5, -0.5, "负Beta, Z<-2"),
]

print("\n测试结果:")
for z, beta, desc in test_cases:
    signal, action = correct_signal_logic(z, beta)
    print(f"  {desc}: β={beta}, Z={z}")
    print(f"    → 信号: {signal}")
    if action:
        print(f"    → 操作: {action}")

# =========================
# 结论
# =========================
print("\n" + "=" * 80)
print("结论")
print("-" * 60)
print("✓ 信号方向的正确逻辑:")
print("  1. Z > 阈值 → spread偏高 → 做空价差 (open_short)")
print("  2. Z < -阈值 → spread偏低 → 做多价差 (open_long)")
print("")
print("✓ 负Beta的处理:")
print("  - 做多价差时: 买Y + 买|β|份X (同向)")
print("  - 做空价差时: 卖Y + 卖|β|份X (同向)")
print("")
print("⚠️ 需要检查的代码:")
print("  1. signal_generation.py中generate_signal方法")
print("  2. 确保Z-score和信号的对应关系正确")
print("  3. 回测时对负Beta配对的处理")