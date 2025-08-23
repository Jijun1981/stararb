#!/usr/bin/env python3
"""
检测所有品种的换月情况
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def detect_rollovers_in_symbol(symbol, data_dir='data/futures'):
    """
    检测单个品种的所有换月点
    换月特征：
    1. 价格变化 > 5%
    2. 持仓量变化 > 20%
    """
    filepath = os.path.join(data_dir, f'{symbol}.parquet')
    
    try:
        df = pd.read_parquet(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 计算日收益率和持仓量变化
        df['return'] = df['close'].pct_change()
        df['oi_change'] = df['open_interest'].pct_change()
        
        # 识别换月点
        rollover_mask = (abs(df['return']) > 0.05) & (abs(df['oi_change']) > 0.20)
        rollovers = df[rollover_mask].copy()
        
        # 添加前一天的价格信息
        if len(rollovers) > 0:
            rollovers['prev_close'] = df['close'].shift(1)
            rollovers['prev_oi'] = df['open_interest'].shift(1)
            
        return rollovers[['close', 'prev_close', 'return', 'open_interest', 'prev_oi', 'oi_change']]
        
    except Exception as e:
        print(f"读取{symbol}失败: {e}")
        return pd.DataFrame()

def analyze_all_symbols():
    """分析所有品种的换月情况"""
    
    # 14个金属期货品种
    symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 
               'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']
    
    print("=" * 100)
    print("所有品种换月检测报告")
    print("=" * 100)
    
    all_rollovers = {}
    rollover_summary = []
    
    for symbol in symbols:
        rollovers = detect_rollovers_in_symbol(symbol)
        
        if len(rollovers) > 0:
            all_rollovers[symbol] = rollovers
            rollover_summary.append({
                'symbol': symbol,
                'count': len(rollovers),
                'dates': rollovers.index.tolist()
            })
            
            print(f"\n{symbol}: 发现 {len(rollovers)} 个换月点")
            print("-" * 50)
            
            for idx, row in rollovers.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                print(f"  {date_str}: 价格 {row['prev_close']:.1f} → {row['close']:.1f} "
                      f"({row['return']*100:+.1f}%), "
                      f"持仓 {row['prev_oi']:.0f} → {row['open_interest']:.0f} "
                      f"({row['oi_change']*100:+.1f}%)")
        else:
            print(f"\n{symbol}: 未发现换月点")
    
    # 统计汇总
    print("\n" + "=" * 100)
    print("换月统计汇总")
    print("=" * 100)
    
    # 按换月次数排序
    rollover_summary.sort(key=lambda x: x['count'], reverse=True)
    
    print("\n品种换月次数排名:")
    for item in rollover_summary:
        print(f"  {item['symbol']}: {item['count']}次")
    
    # 分析换月时间分布
    print("\n换月时间分布分析:")
    all_dates = []
    for item in rollover_summary:
        all_dates.extend(item['dates'])
    
    if all_dates:
        all_dates = pd.to_datetime(all_dates)
        
        # 按年份统计
        years = pd.DatetimeIndex(all_dates).year
        year_counts = pd.Series(years).value_counts().sort_index()
        
        print("\n按年份统计:")
        for year, count in year_counts.items():
            print(f"  {year}年: {count}次")
        
        # 按月份统计
        months = pd.DatetimeIndex(all_dates).month
        month_counts = pd.Series(months).value_counts().sort_index()
        
        print("\n按月份统计(查看季节性):")
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                      '7月', '8月', '9月', '10月', '11月', '12月']
        for month, count in month_counts.items():
            print(f"  {month_names[month-1]}: {count}次")
    
    # 检查回测期间(2023-07至2024-08)的换月
    print("\n" + "=" * 100)
    print("回测期间(2023-07至2024-08)的换月事件")
    print("=" * 100)
    
    backtest_start = pd.to_datetime('2023-07-01')
    backtest_end = pd.to_datetime('2024-08-31')
    
    backtest_rollovers = []
    for symbol, rollovers in all_rollovers.items():
        mask = (rollovers.index >= backtest_start) & (rollovers.index <= backtest_end)
        backtest_period = rollovers[mask]
        
        if len(backtest_period) > 0:
            for idx, row in backtest_period.iterrows():
                backtest_rollovers.append({
                    'date': idx,
                    'symbol': symbol,
                    'price_change': row['return'],
                    'oi_change': row['oi_change']
                })
    
    # 按日期排序
    backtest_rollovers.sort(key=lambda x: x['date'])
    
    print(f"\n回测期间共发现 {len(backtest_rollovers)} 个换月事件:")
    for item in backtest_rollovers:
        print(f"  {item['date'].strftime('%Y-%m-%d')} - {item['symbol']}: "
              f"价格变化 {item['price_change']*100:+.1f}%, "
              f"持仓变化 {item['oi_change']*100:+.1f}%")
    
    return all_rollovers

if __name__ == "__main__":
    all_rollovers = analyze_all_symbols()