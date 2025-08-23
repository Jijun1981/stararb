#!/usr/bin/env python3
"""
快速协整检查脚本 - 使用原子服务
"""

import pandas as pd
import sys
from pathlib import Path

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
print("协整分析快速检查")
print("=" * 80)

# 测试1: 原始时间窗口 (2020-2025)
print("\n【1】原始时间窗口: 2020-01-01 至 2025-08-20")
print("-" * 60)

data_2020 = load_data(
    symbols=SYMBOLS,
    start_date='2020-01-01',
    end_date='2025-08-20',
    columns=['close'],
    log_price=True
)

print(f"数据形状: {data_2020.shape}")

# 初始化分析器
analyzer_2020 = CointegrationAnalyzer(data_2020)

# 执行筛选
results_2020 = analyzer_2020.screen_all_pairs(
    p_threshold=0.05,
    use_halflife_filter=False
)

print(f"\n筛选结果 (5年且1年 p<0.05): {len(results_2020)} 个配对")

if len(results_2020) > 0:
    print("\n前10个配对:")
    for i, row in results_2020.head(10).iterrows():
        print(f"  {i+1:2d}. {row['pair']:25s} p5y={row['pvalue_5y']:.4f}, p1y={row['pvalue_1y']:.4f}, β={row['beta_5y']:.3f}")

# 测试2: 平移时间窗口 (2019-2024)
print("\n" + "=" * 80)
print("\n【2】平移时间窗口: 2019-01-01 至 2024-08-20")
print("-" * 60)

data_2019 = load_data(
    symbols=SYMBOLS,
    start_date='2019-01-01',
    end_date='2024-08-20',
    columns=['close'],
    log_price=True
)

print(f"数据形状: {data_2019.shape}")

# 检查SS0数据
ss0_valid = data_2019['SS0_close'].notna().sum()
print(f"SS0有效数据: {ss0_valid}/{len(data_2019)} ({ss0_valid/len(data_2019)*100:.1f}%)")

# 初始化分析器
analyzer_2019 = CointegrationAnalyzer(data_2019)

# 执行筛选
results_2019 = analyzer_2019.screen_all_pairs(
    p_threshold=0.05,
    use_halflife_filter=False
)

print(f"\n筛选结果 (5年且1年 p<0.05): {len(results_2019)} 个配对")

if len(results_2019) > 0:
    print("\n前10个配对:")
    for i, row in results_2019.head(10).iterrows():
        print(f"  {i+1:2d}. {row['pair']:25s} p5y={row['pvalue_5y']:.4f}, p1y={row['pvalue_1y']:.4f}, β={row['beta_5y']:.3f}")

# 对比分析
print("\n" + "=" * 80)
print("\n【3】对比分析")
print("-" * 60)

if len(results_2020) > 0 and len(results_2019) > 0:
    pairs_2020 = set(results_2020['pair'].tolist())
    pairs_2019 = set(results_2019['pair'].tolist())
    
    common = pairs_2020 & pairs_2019
    only_2020 = pairs_2020 - pairs_2019
    only_2019 = pairs_2019 - pairs_2020
    
    print(f"2020-2025期间通过: {len(pairs_2020)} 个")
    print(f"2019-2024期间通过: {len(pairs_2019)} 个")
    print(f"两期都通过: {len(common)} 个")
    print(f"仅2020-2025通过: {len(only_2020)} 个")
    print(f"仅2019-2024通过: {len(only_2019)} 个")
    
    if len(common) > 0:
        print(f"\n两期都通过的配对:")
        for pair in sorted(common)[:10]:
            print(f"  - {pair}")

# 测试3: 放宽条件 (只要求5年p<0.05)
print("\n" + "=" * 80)
print("\n【4】放宽筛选条件测试")
print("-" * 60)

# 在2019-2024数据上只用5年p值筛选
from itertools import combinations

all_results = []
for s1, s2 in combinations(SYMBOLS, 2):
    try:
        x = data_2019[f"{s1}_close"].values
        y = data_2019[f"{s2}_close"].values
        
        # 多窗口测试
        result = analyzer_2019.multi_window_test(x, y)
        if result and '5y' in result:
            p5y = result['5y'].get('pvalue', 1.0) if result['5y'] else 1.0
            if p5y < 0.05:
                all_results.append({
                    'pair': f"{s1}-{s2}",
                    'pvalue_5y': p5y,
                    'beta_5y': result['5y'].get('beta', 0) if result['5y'] else 0
                })
    except:
        pass

print(f"仅5年p<0.05条件: {len(all_results)} 个配对")

if len(all_results) > 0:
    df_relaxed = pd.DataFrame(all_results).sort_values('pvalue_5y')
    print("\n前10个配对 (按5年p值排序):")
    for i, row in df_relaxed.head(10).iterrows():
        print(f"  {i+1:2d}. {row['pair']:25s} p5y={row['pvalue_5y']:.4f}, β={row['beta_5y']:.3f}")

print("\n" + "=" * 80)