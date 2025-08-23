#!/usr/bin/env python3
"""
检查SS0数据问题 - 修复版
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent

print("=" * 80)
print("SS0（不锈钢）数据诊断")
print("=" * 80)

# 检查原始数据文件
data_file = project_root / "data" / "SS0.parquet"
if data_file.exists():
    df = pd.read_parquet(data_file)
    print(f"\n1. 原始Parquet文件:")
    print(f"   形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    print(f"   索引类型: {type(df.index)}")
    
    # 查看前几行
    print(f"\n   前5行数据:")
    print(df.head())
    
    # 检查date列
    if 'date' in df.columns:
        print(f"\n2. 日期列分析:")
        print(f"   最早日期: {df['date'].min()}")
        print(f"   最晚日期: {df['date'].max()}")
        print(f"   日期类型: {df['date'].dtype}")
        
        # 转换为日期索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 检查2019年数据
        df_2019 = df[(df.index >= '2019-01-01') & (df.index < '2020-01-01')]
        print(f"\n3. 2019年数据:")
        print(f"   记录数: {len(df_2019)}")
        if len(df_2019) > 0:
            print(f"   2019年首个数据: {df_2019.index[0]}")
            print(f"   2019年最后数据: {df_2019.index[-1]}")
        else:
            print(f"   2019年无数据")
            
        # 检查2020年数据
        df_2020 = df[(df.index >= '2020-01-01') & (df.index < '2021-01-01')]
        print(f"\n4. 2020年数据:")
        print(f"   记录数: {len(df_2020)}")
        if len(df_2020) > 0:
            print(f"   2020年首个数据: {df_2020.index[0]}")

print("\n" + "=" * 80)
print("品种上市时间对比")
print("-" * 60)

# 检查所有品种的数据起始时间
symbols = ['AG0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0', 'AL0']

data_info = []
for symbol in symbols:
    file = project_root / "data" / f"{symbol}.parquet"
    if file.exists():
        df = pd.read_parquet(file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        first_date = df.index[0] if len(df) > 0 else None
        last_date = df.index[-1] if len(df) > 0 else None
        
        # 统计2019年之前的数据
        pre_2019 = df[df.index < '2019-01-01']
        
        data_info.append({
            '品种': symbol,
            '首个数据': first_date,
            '最后数据': last_date,
            '总记录数': len(df),
            '2019前记录': len(pre_2019)
        })

info_df = pd.DataFrame(data_info)
print(info_df.to_string(index=False))

print("\n结论:")
print("SS0（不锈钢期货）是2019年9月25日才上市的新品种")
print("这就是为什么2019-2024年5年窗口协整检验会失败的原因")
print("SS0没有足够的历史数据来做5年窗口的协整检验")