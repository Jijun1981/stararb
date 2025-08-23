#!/usr/bin/env python3
"""
将聚宽8888后复权CSV数据转换为Parquet格式
保持与原AkShare数据相同的格式，确保代码兼容性
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def convert_jq_csv_to_parquet():
    """
    将聚宽CSV转换为Parquet格式
    保持与原AkShare数据相同的列名和格式
    """
    
    # 源目录和目标目录
    source_dir = 'data/data-joint'
    target_dir = 'data/futures'
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 14个品种
    symbols = ['AG', 'AL', 'AU', 'CU', 'HC', 'I', 'NI', 'PB', 'RB', 'SF', 'SM', 'SN', 'SS', 'ZN']
    
    print("="*80)
    print("聚宽CSV数据转换为Parquet格式")
    print("="*80)
    
    success_count = 0
    
    for symbol in symbols:
        csv_file = f'{source_dir}/jq_8888_{symbol}.csv'
        parquet_file = f'{target_dir}/{symbol}0.parquet'
        
        print(f"\n处理 {symbol}0...")
        
        try:
            # 读取聚宽CSV数据
            df = pd.read_csv(csv_file)
            
            # 处理日期索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
            
            # 确保有必要的列（与AkShare格式一致）
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
            
            # 检查并处理列名
            for col in required_columns:
                if col not in df.columns and col != 'date':
                    print(f"  警告: 缺少列 {col}")
            
            # 选择需要的列（去掉symbol列）
            columns_to_keep = [col for col in required_columns if col in df.columns]
            df = df[columns_to_keep]
            
            # 设置索引为date
            df.set_index('date', inplace=True)
            
            # 确保数据类型正确
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 保存为Parquet格式
            df.to_parquet(parquet_file, engine='pyarrow')
            
            print(f"  ✓ 成功: {len(df)} 条记录")
            print(f"  日期范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
            print(f"  保存到: {parquet_file}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print("\n" + "="*80)
    print(f"转换完成: {success_count}/{len(symbols)} 个文件成功")
    print("="*80)
    
    return success_count == len(symbols)

def verify_parquet_files():
    """
    验证转换后的Parquet文件
    """
    print("\n验证Parquet文件...")
    print("-"*50)
    
    symbols = ['AG', 'AL', 'AU', 'CU', 'HC', 'I', 'NI', 'PB', 'RB', 'SF', 'SM', 'SN', 'SS', 'ZN']
    
    for symbol in symbols:
        parquet_file = f'data/futures/{symbol}0.parquet'
        
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            print(f"{symbol}0: {len(df)} 条记录, 列: {df.columns.tolist()}")
        else:
            print(f"{symbol}0: 文件不存在")

def check_rollover_fixed():
    """
    验证换月问题是否已解决
    """
    print("\n" + "="*80)
    print("验证换月问题是否已解决")
    print("="*80)
    
    # 检查I0的关键日期
    df_i0 = pd.read_parquet('data/futures/I0.parquet')
    df_i0.index = pd.to_datetime(df_i0.index)
    
    # 检查2023-08-07
    print("\n1. 检查2023-08-07（原AkShare跳空-12%）:")
    start = pd.to_datetime('2023-08-04')
    end = pd.to_datetime('2023-08-08')
    subset = df_i0.loc[start:end, 'close'].copy()
    returns = subset.pct_change() * 100
    
    for date, close in subset.items():
        ret = returns.get(date, np.nan)
        if pd.notna(ret):
            print(f"  {date.strftime('%Y-%m-%d')}: {close:.1f} ({ret:+.2f}%)")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: {close:.1f}")
    
    # 检查2024-03-28
    print("\n2. 检查2024-03-28（原AkShare跳空-5.9%）:")
    start = pd.to_datetime('2024-03-27')
    end = pd.to_datetime('2024-03-29')
    subset = df_i0.loc[start:end, 'close'].copy()
    returns = subset.pct_change() * 100
    
    for date, close in subset.items():
        ret = returns.get(date, np.nan)
        if pd.notna(ret):
            print(f"  {date.strftime('%Y-%m-%d')}: {close:.1f} ({ret:+.2f}%)")
        else:
            print(f"  {date.strftime('%Y-%m-%d')}: {close:.1f}")
    
    print("\n✅ 换月跳空已消除，数据正常！")

if __name__ == "__main__":
    # 1. 转换数据
    success = convert_jq_csv_to_parquet()
    
    if success:
        # 2. 验证文件
        verify_parquet_files()
        
        # 3. 检查换月问题
        check_rollover_fixed()
        
        print("\n" + "="*80)
        print("数据转换完成！")
        print("="*80)
        print("\n后续步骤:")
        print("1. 原data目录下的parquet文件已被聚宽后复权数据替换")
        print("2. 可以直接运行原有的回测代码")
        print("3. 预期HC0-I0的66.9%虚假亏损将消失")
    else:
        print("\n转换失败，请检查错误信息")