#!/usr/bin/env python3
"""
检查SS0数据缺失问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("SS0（不锈钢）数据问题诊断")
print("=" * 80)

# 1. 检查原始数据文件
data_file = project_root / "data" / "SS0.parquet"
if data_file.exists():
    df = pd.read_parquet(data_file)
    print(f"\n1. 原始数据文件检查:")
    print(f"   文件路径: {data_file}")
    print(f"   数据形状: {df.shape}")
    print(f"   时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"   总天数: {len(df)}")
    
    # 检查数据缺失情况
    print(f"\n2. 数据完整性:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"   {col}: {non_null}/{len(df)} 有效 ({null_count} 缺失)")
    
    # 检查时间连续性
    print(f"\n3. 时间连续性检查:")
    first_valid = df['close'].first_valid_index()
    last_valid = df['close'].last_valid_index()
    print(f"   首个有效数据: {first_valid}")
    print(f"   最后有效数据: {last_valid}")
    
    # 查看2019年数据
    df_2019 = df[(df.index >= '2019-01-01') & (df.index < '2020-01-01')]
    print(f"\n4. 2019年数据情况:")
    print(f"   2019年记录数: {len(df_2019)}")
    print(f"   2019年有效close: {df_2019['close'].notna().sum()}")
    
    if len(df_2019) > 0:
        first_2019 = df_2019['close'].first_valid_index()
        print(f"   2019年首个有效数据: {first_2019}")
        
        # 显示前10条2019年数据
        print(f"\n   2019年前10条数据:")
        print(df_2019.head(10))
    
    # 查看数据断点
    print(f"\n5. 数据断点分析:")
    # 找出所有NaN
    nan_mask = df['close'].isna()
    if nan_mask.any():
        # 找连续的NaN段
        nan_changes = nan_mask.ne(nan_mask.shift())
        nan_starts = df.index[nan_mask & nan_changes]
        nan_ends = df.index[(~nan_mask) & nan_changes]
        
        print(f"   发现 {len(nan_starts)} 个数据缺失段:")
        for i, start in enumerate(nan_starts[:5]):  # 只显示前5个
            if i < len(nan_ends):
                end = nan_ends[i]
                days = (end - start).days
                print(f"   - {start.date()} 至 {end.date()} ({days}天)")
    
    # SS0上市时间
    print(f"\n6. SS0品种信息:")
    print(f"   SS0是不锈钢期货，上海期货交易所品种")
    print(f"   上市时间: 2019年9月25日")
    print(f"   这解释了为什么2019年前9个月没有数据！")
    
else:
    print(f"数据文件不存在: {data_file}")

# 对比其他品种
print("\n" + "=" * 80)
print("其他品种数据对比")
print("-" * 60)

symbols = ['AG0', 'AU0', 'CU0', 'SS0']
for symbol in symbols:
    file = project_root / "data" / f"{symbol}.parquet"
    if file.exists():
        df = pd.read_parquet(file)
        first_valid = df['close'].first_valid_index()
        valid_count = df['close'].notna().sum()
        print(f"{symbol}: 首个数据 {first_valid}, 有效数据 {valid_count}/{len(df)}")