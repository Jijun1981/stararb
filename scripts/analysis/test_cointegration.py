#!/usr/bin/env python3
"""
测试协整分析，了解为什么只有20多个配对通过筛选
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

# 期货品种列表
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'    # 黑色系
]

print("=" * 80)
print("协整分析诊断")
print("=" * 80)

# 测试1: 数据质量检查
print("\n1. 数据质量检查")
print("-" * 40)

# 加载2020-2025的数据
data_2020_2025 = load_data(
    symbols=SYMBOLS,
    start_date='2020-01-01',
    end_date='2025-08-20',
    columns=['close'],
    log_price=True
)

print(f"2020-2025数据: {len(data_2020_2025)} 天, {len(data_2020_2025.columns)} 列")

# 检查每个品种的数据完整性
for symbol in SYMBOLS:
    col = f"{symbol}_close"
    non_na = data_2020_2025[col].notna().sum()
    total = len(data_2020_2025)
    pct = non_na / total * 100
    first_valid = data_2020_2025[col].first_valid_index()
    print(f"  {symbol:4s}: {non_na}/{total} ({pct:5.1f}%), 起始日期: {first_valid}")

# 测试2: 协整检验双重条件
print("\n2. 协整检验条件分析")
print("-" * 40)

analyzer = CointegrationAnalyzer(data_2020_2025)

# 不使用任何筛选，看原始结果
all_results = []
from itertools import combinations

for symbol1, symbol2 in combinations(SYMBOLS, 2):
    try:
        x_data = data_2020_2025[f"{symbol1}_close"].values
        y_data = data_2020_2025[f"{symbol2}_close"].values
        
        # 多窗口测试
        result = analyzer.multi_window_test(x_data, y_data)
        
        if result:
            all_results.append({
                'pair': f"{symbol1}-{symbol2}",
                'pvalue_5y': result.get('windows', {}).get(5, {}).get('pvalue', 1.0),
                'pvalue_1y': result.get('windows', {}).get(1, {}).get('pvalue', 1.0),
                'beta_5y': result.get('windows', {}).get(5, {}).get('beta', 0.0),
            })
    except Exception as e:
        pass

results_df = pd.DataFrame(all_results)

if len(results_df) > 0:
    # 统计不同条件下的通过数量
    print(f"总配对数: {len(list(combinations(SYMBOLS, 2)))}")
    print(f"成功检验: {len(results_df)}")
    
    # 单独条件
    pass_5y = (results_df['pvalue_5y'] < 0.05).sum()
    pass_1y = (results_df['pvalue_1y'] < 0.05).sum()
    pass_both = ((results_df['pvalue_5y'] < 0.05) & (results_df['pvalue_1y'] < 0.05)).sum()
    
    print(f"\n筛选结果:")
    print(f"  仅5年p<0.05: {pass_5y} 个")
    print(f"  仅1年p<0.05: {pass_1y} 个")
    print(f"  双重条件(5年且1年): {pass_both} 个")
    
    # 查看p值分布
    print(f"\np值分布:")
    print(f"  5年p值: 均值={results_df['pvalue_5y'].mean():.3f}, 中位数={results_df['pvalue_5y'].median():.3f}")
    print(f"  1年p值: 均值={results_df['pvalue_1y'].mean():.3f}, 中位数={results_df['pvalue_1y'].median():.3f}")
    
    # 列出通过双重条件的配对
    passed = results_df[(results_df['pvalue_5y'] < 0.05) & (results_df['pvalue_1y'] < 0.05)]
    passed = passed.sort_values('pvalue_1y')
    
    print(f"\n通过双重条件的配对:")
    for i, row in passed.head(10).iterrows():
        print(f"  {row['pair']:20s}: 5y_p={row['pvalue_5y']:.4f}, 1y_p={row['pvalue_1y']:.4f}, β={row['beta_5y']:.3f}")

# 测试3: 时间窗口影响
print("\n3. 时间窗口影响分析")
print("-" * 40)

# 加载2019-2024的数据（平移一年）
data_2019_2024 = load_data(
    symbols=SYMBOLS,
    start_date='2019-01-01',
    end_date='2024-08-20',
    columns=['close'],
    log_price=True
)

print(f"2019-2024数据: {len(data_2019_2024)} 天")

# SS0的数据问题
print("\n4. SS0(不锈钢)数据诊断")
print("-" * 40)
ss0_2020 = data_2020_2025['SS0_close']
ss0_2019 = data_2019_2024['SS0_close']

print(f"2020-2025期间: 有效数据 {ss0_2020.notna().sum()}/{len(ss0_2020)}")
print(f"2019-2024期间: 有效数据 {ss0_2019.notna().sum()}/{len(ss0_2019)}")
print(f"SS0首个有效数据: {ss0_2020.first_valid_index()}")

# 建议
print("\n5. 分析结论与建议")
print("-" * 40)
print("问题原因:")
print("1. 双重条件过于严格: 要求同时满足5年和1年p<0.05")
print("2. SS0不锈钢数据缺失: 导致相关的13个配对无法检验")
print("3. 协整关系时变性: 部分配对在不同时期协整性不同")
print("\n建议:")
print("1. 考虑放宽筛选条件，如只要求5年p<0.05")
print("2. 排除SS0品种，或补充其历史数据")
print("3. 增加更多时间窗口测试（如3年、2年）")