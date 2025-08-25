#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查交易细节，验证手数、方向、PnL计算
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import load_all_symbols_data

def check_trade_logic():
    """检查交易逻辑"""
    
    # 1. 加载信号文件
    signals = pd.read_csv('signals_residual_timeline_20250824_192040.csv')
    signals['date'] = pd.to_datetime(signals['date'])
    
    # 2. 加载价格数据
    log_prices = load_all_symbols_data()
    prices = np.exp(log_prices)  # 转换为原始价格
    
    # 3. 查看一个具体的交易案例
    # 找第一个开仓信号
    open_signals = signals[signals['signal'].str.contains('open')]
    first_open = open_signals.iloc[0]
    
    print("="*60)
    print("第一个开仓信号分析:")
    print("="*60)
    print(f"日期: {first_open['date']}")
    print(f"配对: {first_open['pair']}")
    print(f"信号: {first_open['signal']}")
    print(f"Z-score: {first_open['z_score']:.4f}")
    print(f"Beta: {first_open['beta']:.6f}")
    print(f"Symbol X: {first_open['symbol_x']}")
    print(f"Symbol Y: {first_open['symbol_y']}")
    
    # 获取当日价格
    date = first_open['date']
    symbol_x = first_open['symbol_x']
    symbol_y = first_open['symbol_y']
    
    price_x = prices.loc[date, symbol_x]
    price_y = prices.loc[date, symbol_y]
    
    print(f"\n价格:")
    print(f"  {symbol_x}: {price_x:.2f}")
    print(f"  {symbol_y}: {price_y:.2f}")
    
    # 4. 分析配对交易逻辑
    beta = first_open['beta']
    signal_type = first_open['signal']
    
    print(f"\n配对交易逻辑:")
    print(f"  Beta: {beta:.6f}")
    print(f"  关系: Y = Beta * X + 残差")
    
    if 'short' in signal_type:
        print(f"  信号: 做空配对 (Z > 2.2，价差偏高)")
        print(f"  操作: 卖出Y，买入{abs(beta):.2f}份X")
        print(f"  逻辑: 预期价差会收敛")
    else:
        print(f"  信号: 做多配对 (Z < -2.2，价差偏低)")
        print(f"  操作: 买入Y，卖出{abs(beta):.2f}份X")
        print(f"  逻辑: 预期价差会扩大")
    
    # 5. 验证残差计算
    residual = first_open['residual']
    calc_residual = np.log(price_y) - beta * np.log(price_x)
    
    print(f"\n残差验证:")
    print(f"  记录的残差: {residual:.6f}")
    print(f"  计算的残差: {calc_residual:.6f}")
    print(f"  差异: {abs(residual - calc_residual):.8f}")
    
    # 6. 合约规格
    contract_specs = {
        'AG': {'multiplier': 15, 'tick_size': 1},
        'AU': {'multiplier': 1000, 'tick_size': 0.01},
        'AL': {'multiplier': 5, 'tick_size': 5},
        'CU': {'multiplier': 5, 'tick_size': 10},
        'NI': {'multiplier': 1, 'tick_size': 10},
        'PB': {'multiplier': 5, 'tick_size': 5},
        'SN': {'multiplier': 1, 'tick_size': 10},
        'ZN': {'multiplier': 5, 'tick_size': 5},
        'HC': {'multiplier': 10, 'tick_size': 1},
        'I': {'multiplier': 100, 'tick_size': 0.5},
        'RB': {'multiplier': 10, 'tick_size': 1},
        'SF': {'multiplier': 5, 'tick_size': 2},
        'SM': {'multiplier': 5, 'tick_size': 2},
        'SS': {'multiplier': 5, 'tick_size': 5}
    }
    
    spec_x = contract_specs.get(symbol_x, {'multiplier': 1, 'tick_size': 1})
    spec_y = contract_specs.get(symbol_y, {'multiplier': 1, 'tick_size': 1})
    
    print(f"\n合约规格:")
    print(f"  {symbol_x}: 乘数={spec_x['multiplier']}, tick={spec_x['tick_size']}")
    print(f"  {symbol_y}: 乘数={spec_y['multiplier']}, tick={spec_y['tick_size']}")
    
    # 7. 计算理论手数比
    # 目标：beta * lots_x * price_x * multiplier_x = lots_y * price_y * multiplier_y
    theoretical_ratio = abs(beta) * price_x * spec_x['multiplier'] / (price_y * spec_y['multiplier'])
    
    print(f"\n手数计算:")
    print(f"  理论Y/X比例: {theoretical_ratio:.4f}")
    print(f"  如果X=1手，Y应该={theoretical_ratio:.2f}手")
    
    # 8. 分析所有开仓信号的Z-score分布
    print("\n" + "="*60)
    print("所有开仓信号统计:")
    print("="*60)
    
    open_long = signals[signals['signal'] == 'open_long']
    open_short = signals[signals['signal'] == 'open_short']
    
    print(f"做多信号: {len(open_long)}个")
    if len(open_long) > 0:
        print(f"  Z-score范围: [{open_long['z_score'].min():.2f}, {open_long['z_score'].max():.2f}]")
        print(f"  平均Z-score: {open_long['z_score'].mean():.4f}")
    
    print(f"\n做空信号: {len(open_short)}个")  
    if len(open_short) > 0:
        print(f"  Z-score范围: [{open_short['z_score'].min():.2f}, {open_short['z_score'].max():.2f}]")
        print(f"  平均Z-score: {open_short['z_score'].mean():.4f}")
    
    # 9. 检查配对的beta符号
    print("\n" + "="*60)
    print("配对Beta分析:")
    print("="*60)
    
    pairs_beta = signals[signals['signal'].str.contains('open')][['pair', 'beta', 'symbol_x', 'symbol_y']].drop_duplicates()
    
    print(f"总配对数: {len(pairs_beta)}")
    print(f"正Beta配对: {(pairs_beta['beta'] > 0).sum()}个")
    print(f"负Beta配对: {(pairs_beta['beta'] < 0).sum()}个")
    
    # 显示前10个配对的beta
    print("\n前10个配对的Beta:")
    for idx, row in pairs_beta.head(10).iterrows():
        print(f"  {row['pair']}: beta={row['beta']:.4f} ({row['symbol_x']}-{row['symbol_y']})")

if __name__ == "__main__":
    check_trade_logic()