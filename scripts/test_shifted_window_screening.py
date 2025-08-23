#!/usr/bin/env python3
"""
测试时间窗口平移和自定义筛选条件
- 往前平移一年 (2019-2024)
- 使用4年p值<0.05 且 1年p值<0.1的筛选条件
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

# 配置
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'    # 黑色系
]

print("=" * 80)
print("时间窗口平移和自定义筛选条件测试")
print("=" * 80)

# 测试1: 原始时间窗口 (2020-2025) + 标准筛选
print("\n【测试1】原始时间窗口 (2020-01-01 至 2025-08-20)")
print("筛选条件: 5年p<0.05 且 1年p<0.05")
print("-" * 60)

data_original = load_data(
    symbols=SYMBOLS,
    start_date='2020-01-01',
    end_date='2025-08-20',
    columns=['close'],
    log_price=True
)

print(f"数据形状: {data_original.shape}")
print(f"时间范围: {data_original.index[0]} 至 {data_original.index[-1]}")

analyzer_original = CointegrationAnalyzer(data_original)

# 标准筛选: 5年p<0.05 且 1年p<0.05
results_original = analyzer_original.screen_all_pairs(
    p_threshold=0.05,
    use_halflife_filter=False,
    volatility_start_date='2024-01-01'  # 使用2024年数据计算波动率
)

print(f"\n筛选结果: {len(results_original)} 个配对通过")

if len(results_original) > 0:
    print("\n前10个配对:")
    for i, row in results_original.head(10).iterrows():
        print(f"  {i+1:2d}. {row['pair']:25s} p5y={row['pvalue_5y']:.4f}, p1y={row['pvalue_1y']:.4f}")

# 测试2: 平移时间窗口 (2019-2024) + 自定义筛选
print("\n" + "=" * 80)
print("\n【测试2】平移时间窗口 (2019-01-01 至 2024-08-20) - 往前平移1年")
print("筛选条件: 4年p<0.05 且 1年p<0.1 (放宽1年条件)")
print("-" * 60)

data_shifted = load_data(
    symbols=SYMBOLS,
    start_date='2019-01-01',
    end_date='2024-08-20',
    columns=['close'],
    log_price=True
)

print(f"数据形状: {data_shifted.shape}")
print(f"时间范围: {data_shifted.index[0]} 至 {data_shifted.index[-1]}")

# 检查SS0数据
ss0_valid = data_shifted['SS0_close'].notna().sum()
print(f"SS0有效数据: {ss0_valid}/{len(data_shifted)} ({ss0_valid/len(data_shifted)*100:.1f}%)")

analyzer_shifted = CointegrationAnalyzer(data_shifted)

# 首先获取所有配对的详细结果
from itertools import combinations

all_results = []
symbols_in_data = [col.replace('_close', '') for col in data_shifted.columns if col.endswith('_close')]

print(f"\n开始分析 {len(list(combinations(symbols_in_data, 2)))} 个配对...")

for s1, s2 in combinations(symbols_in_data, 2):
    try:
        # 获取数据
        x_data = data_shifted[f"{s1}_close"].values
        y_data = data_shifted[f"{s2}_close"].values
        
        # 方向判定 (使用2023年数据计算波动率，因为是2019-2024的数据)
        direction, symbol_x, symbol_y = analyzer_shifted.determine_direction(
            f"{s1}_close", f"{s2}_close",
            use_recent=True,
            recent_start='2023-01-01'  # 使用2023年数据计算波动率
        )
        
        # 确定最终的X和Y序列
        if symbol_x == f"{s1}_close":
            x_final = x_data
            y_final = y_data
            final_x = s1
            final_y = s2
        else:
            x_final = y_data
            y_final = x_data
            final_x = s2
            final_y = s1
        
        # 多窗口协整检验
        multi_results = analyzer_shifted.multi_window_test(x_final, y_final)
        
        # 构建结果
        result = {
            'pair': f"{final_x}-{final_y}",
            'symbol_x': final_x,
            'symbol_y': final_y,
            'pvalue_5y': multi_results['5y']['pvalue'] if multi_results['5y'] else np.nan,
            'pvalue_4y': multi_results['4y']['pvalue'] if multi_results['4y'] else np.nan,
            'pvalue_3y': multi_results['3y']['pvalue'] if multi_results['3y'] else np.nan,
            'pvalue_2y': multi_results['2y']['pvalue'] if multi_results['2y'] else np.nan,
            'pvalue_1y': multi_results['1y']['pvalue'] if multi_results['1y'] else np.nan,
            'beta_4y': multi_results['4y']['beta'] if multi_results['4y'] else np.nan,
            'beta_1y': multi_results['1y']['beta'] if multi_results['1y'] else np.nan,
        }
        all_results.append(result)
        
    except Exception as e:
        print(f"  配对 {s1}-{s2} 分析失败: {str(e)}")

# 转换为DataFrame
df_all = pd.DataFrame(all_results)

# 自定义筛选: 4年p<0.05 且 1年p<0.1
filtered_4y_1y = df_all[
    (df_all['pvalue_4y'] < 0.05) & 
    (df_all['pvalue_1y'] < 0.1)
].sort_values('pvalue_1y').reset_index(drop=True)

print(f"\n筛选结果 (4年p<0.05 且 1年p<0.1): {len(filtered_4y_1y)} 个配对")

if len(filtered_4y_1y) > 0:
    print("\n所有通过筛选的配对:")
    for i, row in filtered_4y_1y.iterrows():
        print(f"  {i+1:2d}. {row['pair']:25s} p4y={row['pvalue_4y']:.4f}, p1y={row['pvalue_1y']:.4f}, β4y={row['beta_4y']:.3f}")

# 测试3: 更多自定义筛选条件
print("\n" + "=" * 80)
print("\n【测试3】其他筛选条件对比")
print("-" * 60)

# 3.1 只用4年p<0.05
filtered_4y_only = df_all[df_all['pvalue_4y'] < 0.05].sort_values('pvalue_4y')
print(f"\n仅4年p<0.05: {len(filtered_4y_only)} 个配对")

# 3.2 3年p<0.05 且 1年p<0.1
filtered_3y_1y = df_all[
    (df_all['pvalue_3y'] < 0.05) & 
    (df_all['pvalue_1y'] < 0.1)
].sort_values('pvalue_1y')
print(f"3年p<0.05 且 1年p<0.1: {len(filtered_3y_1y)} 个配对")

# 3.3 2年p<0.05 且 1年p<0.05
filtered_2y_1y = df_all[
    (df_all['pvalue_2y'] < 0.05) & 
    (df_all['pvalue_1y'] < 0.05)
].sort_values('pvalue_1y')
print(f"2年p<0.05 且 1年p<0.05: {len(filtered_2y_1y)} 个配对")

# 分析SS0相关配对
print("\n" + "=" * 80)
print("\n【测试4】SS0相关配对分析")
print("-" * 60)

ss0_pairs = df_all[
    (df_all['symbol_x'] == 'SS0') | (df_all['symbol_y'] == 'SS0')
]

print(f"SS0相关配对总数: {len(ss0_pairs)}")

# 检查哪些SS0配对能通过不同的筛选条件
ss0_4y = ss0_pairs[ss0_pairs['pvalue_4y'] < 0.05]
ss0_3y = ss0_pairs[ss0_pairs['pvalue_3y'] < 0.05]
ss0_2y = ss0_pairs[ss0_pairs['pvalue_2y'] < 0.05]
ss0_1y = ss0_pairs[ss0_pairs['pvalue_1y'] < 0.05]

print(f"SS0配对通过4年p<0.05: {len(ss0_4y)} 个")
print(f"SS0配对通过3年p<0.05: {len(ss0_3y)} 个")
print(f"SS0配对通过2年p<0.05: {len(ss0_2y)} 个")
print(f"SS0配对通过1年p<0.05: {len(ss0_1y)} 个")

if len(ss0_1y) > 0:
    print("\nSS0配对通过1年p<0.05的详情:")
    for _, row in ss0_1y.iterrows():
        print(f"  {row['pair']:25s} p1y={row['pvalue_1y']:.4f}, p2y={row['pvalue_2y']:.4f}")

# 总结
print("\n" + "=" * 80)
print("测试总结")
print("-" * 60)
print("✓ 时间窗口可以灵活配置（2020-2025 或 2019-2024）")
print("✓ 波动率计算起始日期可配置")
print("✓ p值筛选条件可以自定义组合（5年/4年/3年/2年/1年）")
print("✓ 可以为不同时间窗口设置不同的p值阈值")
print("✓ SS0等新品种可以通过放宽长期窗口要求来包含")
print(f"\n关键发现:")
print(f"- 原始窗口(2020-2025): {len(results_original)} 个配对通过5年且1年p<0.05")
print(f"- 平移窗口(2019-2024): {len(filtered_4y_1y)} 个配对通过4年p<0.05且1年p<0.1")
print(f"- SS0由于2019年9月上市，无法满足5年窗口要求，但可满足较短窗口")