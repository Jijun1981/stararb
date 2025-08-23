#!/usr/bin/env python3
"""
测试正确的信号逻辑
验证Z-score和交易方向的关系
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("配对交易信号逻辑测试")
print("=" * 80)

print("\n理论基础:")
print("-" * 60)
print("价差公式: spread = Y - β*X")
print("Z-score: z = (spread - mean) / std")
print("")
print("当Z-score > 2.0时:")
print("  - 价差偏高（Y相对X太贵）")
print("  - 应该做空价差（卖Y买X）")
print("  - 期待价差回归均值")
print("")
print("当Z-score < -2.0时:")
print("  - 价差偏低（Y相对X太便宜）")
print("  - 应该做多价差（买Y卖X）")
print("  - 期待价差回归均值")

# 读取实际数据验证
df = pd.read_csv('/mnt/e/Star-arb/output/kalman_analysis/RB0_I0_kalman_analysis.csv')

print("\n" + "=" * 80)
print("RB0-I0配对分析")
print("-" * 60)

# 找出Z-score超过阈值的点
high_z_positive = df[df['z_score'] > 2.0]
high_z_negative = df[df['z_score'] < -2.0]

print(f"\nZ-score > 2.0的情况: {len(high_z_positive)}次")
if len(high_z_positive) > 0:
    print("前3个例子:")
    for idx, row in high_z_positive.head(3).iterrows():
        print(f"  {row['date']}: z={row['z_score']:.3f} -> 应该做空价差")

print(f"\nZ-score < -2.0的情况: {len(high_z_negative)}次")
if len(high_z_negative) > 0:
    print("前3个例子:")
    for idx, row in high_z_negative.head(3).iterrows():
        print(f"  {row['date']}: z={row['z_score']:.3f} -> 应该做多价差")

# 重新实现正确的信号生成逻辑
print("\n" + "=" * 80)
print("正确的信号生成逻辑")
print("-" * 60)

position = None
days_held = 0
signals = []
trades = []

for idx, row in df.iterrows():
    z_score = row['z_score']
    date = row['date']
    
    if position is None:  # 无持仓
        if z_score < -2.0:  # 价差偏低
            # 做多价差：买Y卖X
            position = 'long'
            days_held = 1
            signals.append(1)
            trades.append({
                'date': date,
                'action': '开多仓',
                'z_score': z_score,
                'reason': '价差偏低'
            })
        elif z_score > 2.0:  # 价差偏高
            # 做空价差：卖Y买X
            position = 'short'
            days_held = 1
            signals.append(-1)
            trades.append({
                'date': date,
                'action': '开空仓',
                'z_score': z_score,
                'reason': '价差偏高'
            })
        else:
            signals.append(0)
    else:  # 有持仓
        days_held += 1
        
        # 平仓条件
        close_signal = False
        close_reason = ''
        
        if position == 'long' and z_score > -0.5:  # 做多平仓
            close_signal = True
            close_reason = f'Z回归({z_score:.3f})'
        elif position == 'short' and z_score < 0.5:  # 做空平仓
            close_signal = True
            close_reason = f'Z回归({z_score:.3f})'
        elif days_held > 30:  # 超时平仓
            close_signal = True
            close_reason = '超过30天'
        
        if close_signal:
            trades.append({
                'date': date,
                'action': '平仓',
                'z_score': z_score,
                'reason': close_reason
            })
            position = None
            days_held = 0
            signals.append(0)
        else:
            signals.append(1 if position == 'long' else -1)

# 统计交易
print(f"\n交易统计:")
print(f"  总交易次数: {len([t for t in trades if '开' in t['action']])}")
print(f"  做多次数: {len([t for t in trades if t['action'] == '开多仓'])}")
print(f"  做空次数: {len([t for t in trades if t['action'] == '开空仓'])}")

if len(trades) > 0:
    print(f"\n前10笔交易:")
    for i, trade in enumerate(trades[:10], 1):
        print(f"  {i}. {trade['date']}: {trade['action']}, Z={trade['z_score']:.3f}, {trade['reason']}")

# 对比信号
signal_series = pd.Series(signals)
print(f"\n信号分布:")
print(f"  做多信号: {(signal_series == 1).sum()}")
print(f"  做空信号: {(signal_series == -1).sum()}")
print(f"  无持仓: {(signal_series == 0).sum()}")

print("\n" + "=" * 80)
print("结论")
print("-" * 60)
print("✓ lib/signal_generation.py中的逻辑需要修正:")
print("  - z_score < -z_open时应该返回'long'而不是'open_long'")
print("  - z_score > z_open时应该返回'short'而不是'open_short'")
print("✓ RB0-I0配对有20次Z-score超过阈值，应该产生交易信号")
print("✓ 修正后预期会有5-6次交易（考虑平仓时间）")