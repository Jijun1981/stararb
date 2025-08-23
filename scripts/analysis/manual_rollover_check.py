#!/usr/bin/env python3
"""
手动检查特定交易的换月情况
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_specific_trades():
    """手动检查几个关键交易"""
    
    # 关键交易列表
    key_trades = [
        {'id': 1, 'pair': 'HC0-I0', 'open': '2023-07-28', 'close': '2023-08-07', 'return': -66.9},
        {'id': 45, 'pair': 'HC0-I0', 'open': '2024-03-28', 'close': '2024-04-19', 'return': 66.3},
        {'id': 37, 'pair': 'PB0-I0', 'open': '2024-03-28', 'close': '2024-03-29', 'return': -21.0},
        {'id': 47, 'pair': 'SN0-ZN0', 'open': '2024-04-19', 'close': '2024-04-23', 'return': -30.2},
        {'id': 63, 'pair': 'AL0-CU0', 'open': '2024-05-20', 'close': '2024-06-06', 'return': 50.8}
    ]
    
    for trade in key_trades:
        print(f"\n{'='*80}")
        print(f"交易 {trade['id']}: {trade['pair']} (收益率: {trade['return']:.1f}%)")
        print(f"开仓: {trade['open']}, 平仓: {trade['close']}")
        print('='*80)
        
        symbol_y, symbol_x = trade['pair'].split('-')
        
        # 检查每个品种
        for symbol in [symbol_y, symbol_x]:
            print(f"\n{symbol}:")
            try:
                df = pd.read_parquet(f'/mnt/e/Star-arb/data/{symbol}.parquet')
                df.index = pd.to_datetime(df.index)
                
                # 获取交易期间的数据
                start = pd.to_datetime(trade['open']) - timedelta(days=2)
                end = pd.to_datetime(trade['close']) + timedelta(days=2)
                subset = df.loc[start:end, ['close', 'open_interest']].copy()
                
                if len(subset) > 0:
                    # 计算每日变化
                    subset['price_change'] = subset['close'].pct_change()
                    subset['oi_change'] = subset['open_interest'].pct_change()
                    
                    # 打印关键日期的数据
                    print(f"  {'日期':<12} {'收盘价':>10} {'价格变化%':>10} {'持仓量':>12} {'持仓变化%':>10}")
                    print(f"  {'-'*60}")
                    
                    for idx, row in subset.iterrows():
                        date_str = idx.strftime('%Y-%m-%d')
                        price_chg = row['price_change'] * 100 if pd.notna(row['price_change']) else 0
                        oi_chg = row['oi_change'] * 100 if pd.notna(row['oi_change']) else 0
                        
                        # 标记可能的换月（价格变化>5%且持仓量变化>20%）
                        marker = ""
                        if abs(price_chg) > 5 and abs(oi_chg) > 20:
                            marker = " ⚠️ 换月"
                        elif abs(price_chg) > 8:
                            marker = " ⚠️ 大幅跳空"
                        
                        print(f"  {date_str:<12} {row['close']:>10.1f} {price_chg:>10.2f} {row['open_interest']:>12.0f} {oi_chg:>10.2f}{marker}")
                    
            except Exception as e:
                print(f"  读取{symbol}数据失败: {e}")

if __name__ == "__main__":
    check_specific_trades()