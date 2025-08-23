#!/usr/bin/env python3
"""
对比不同时间窗口的回测结果
"""

import pandas as pd
import json
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent.parent

print("=" * 80)
print("配对交易策略 - 时间窗口对比分析")
print("=" * 80)

# 原始时间窗口结果（2023-2025）
original_dir = project_root / "output/pipeline_v21"
original_report = list(original_dir.glob("pipeline_report_*.json"))[-1]
original_trades = list(original_dir.glob("trades_*.csv"))[-1]

# 平移时间窗口结果（2022-2024）
shifted_dir = project_root / "output/pipeline_shifted_1year"
shifted_report = list(shifted_dir.glob("pipeline_report_*.json"))[-1]

# 加载报告
with open(original_report) as f:
    orig_data = json.load(f)

with open(shifted_report) as f:
    shift_data = json.load(f)

# 加载交易记录
orig_trades = pd.read_csv(original_trades)

print("\n1. 时间窗口配置对比")
print("-" * 40)
print("\n原始窗口（2024-2025年策略）:")
print(f"  数据起始: {orig_data['time_config']['data_start']}")
print(f"  Beta训练: {orig_data['time_config']['beta_training_start']} 至 {orig_data['time_config']['beta_training_end']}")
print(f"  Kalman收敛: {orig_data['time_config']['convergence_start']} 至 {orig_data['time_config']['convergence_end']}")
print(f"  信号生成: {orig_data['time_config']['signal_start']} 至 {orig_data['time_config']['backtest_end']}")

print("\n平移窗口（2023-2024年策略）:")
print(f"  数据起始: {shift_data['time_config']['data_start']}")
print(f"  Beta训练: {shift_data['time_config']['beta_training_start']} 至 {shift_data['time_config']['beta_training_end']}")
print(f"  Kalman收敛: {shift_data['time_config']['convergence_start']} 至 {shift_data['time_config']['convergence_end']}")
print(f"  信号生成: {shift_data['time_config']['signal_start']} 至 {shift_data['time_config']['backtest_end']}")

print("\n2. 配对筛选结果对比")
print("-" * 40)
print(f"                 原始窗口    平移窗口")
print(f"筛选配对数:         {orig_data['pairs_screened']:3d}        {shift_data['pairs_screened']:3d}")
print(f"收敛配对数:         {orig_data['pairs_converged']:3d}        {shift_data['pairs_converged']:3d}")

print("\n3. 交易执行结果对比")
print("-" * 40)
print(f"                 原始窗口    平移窗口")
print(f"总交易数:          {orig_data.get('total_trades', 0):4d}       {shift_data.get('total_trades', 0):4d}")

# 加载协整配对进行详细对比
orig_coint = list(original_dir.glob("cointegrated_pairs_*.csv"))[-1]
shift_coint = list(shifted_dir.glob("cointegrated_pairs_*.csv"))[-1]

orig_pairs = pd.read_csv(orig_coint)
shift_pairs = pd.read_csv(shift_coint)

# 找出共同的配对
orig_set = set(orig_pairs['pair'].tolist())
shift_set = set(shift_pairs['pair'].tolist())

common_pairs = orig_set & shift_set
only_orig = orig_set - shift_set
only_shift = shift_set - orig_set

print("\n4. 协整配对分析")
print("-" * 40)
print(f"共同配对数: {len(common_pairs)}")
print(f"仅原始窗口: {len(only_orig)}")
print(f"仅平移窗口: {len(only_shift)}")

if len(common_pairs) > 0:
    print(f"\n共同配对列表:")
    for i, pair in enumerate(sorted(common_pairs)[:10], 1):
        # 获取两个时期的p值
        orig_row = orig_pairs[orig_pairs['pair'] == pair].iloc[0]
        shift_row = shift_pairs[shift_pairs['pair'] == pair].iloc[0]
        print(f"  {i:2d}. {pair}")
        print(f"      原始: p1y={orig_row['pvalue_1y']:.4f}, p5y={orig_row['pvalue_5y']:.4f}, β={orig_row['beta_5y']:.4f}")
        print(f"      平移: p1y={shift_row['pvalue_1y']:.4f}, p5y={shift_row['pvalue_5y']:.4f}, β={shift_row['beta_5y']:.4f}")

print("\n5. 原始窗口交易绩效（2024年7月-2025年8月）")
print("-" * 40)
if len(orig_trades) > 0:
    # 按配对统计
    pair_stats = orig_trades.groupby('pair').agg({
        'net_pnl': ['count', 'sum', 'mean'],
        'holding_days': 'mean'
    }).round(2)
    
    pair_stats.columns = ['交易数', '总盈亏', '平均盈亏', '平均持仓天数']
    pair_stats = pair_stats.sort_values('总盈亏', ascending=False)
    
    print("前5个盈利配对:")
    print(pair_stats.head(5))
    
    print(f"\n总计:")
    print(f"  总盈亏: ¥{orig_trades['net_pnl'].sum():,.0f}")
    print(f"  平均盈亏: ¥{orig_trades['net_pnl'].mean():,.0f}")
    print(f"  胜率: {(orig_trades['net_pnl'] > 0).mean() * 100:.1f}%")
    print(f"  平均持仓天数: {orig_trades['holding_days'].mean():.1f}")

print("\n6. 分析总结")
print("-" * 40)
print("关键发现:")
print(f"1. 协整配对稳定性: {len(common_pairs)/max(len(orig_set), len(shift_set))*100:.1f}% 的配对在两个时期都通过协整检验")
print(f"2. 原始窗口(2024-2025)实际执行了 {orig_data.get('total_trades', 0)} 笔交易")
print(f"3. 平移窗口(2023-2024)由于配对格式问题未能成功执行交易")

print("\n注：平移窗口存在技术问题需要修复（配对名称格式不匹配）")
print("=" * 80)