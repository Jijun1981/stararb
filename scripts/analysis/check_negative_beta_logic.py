#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查负Beta配对的交易逻辑
"""

import pandas as pd
import numpy as np

def analyze_negative_beta_logic():
    """分析负Beta配对的交易逻辑"""
    
    print("="*60)
    print("负Beta配对交易逻辑分析")
    print("="*60)
    
    # 1. 理论分析
    print("\n1. 理论分析:")
    print("-"*40)
    print("配对关系: log(Y) = β * log(X) + 残差")
    print("\n当β为负时（如β = -0.5）:")
    print("  - X和Y负相关")
    print("  - X上涨时，Y倾向下跌")
    print("  - X下跌时，Y倾向上涨")
    
    print("\n残差 = log(Y) - β * log(X)")
    print("当β < 0时:")
    print("  残差 = log(Y) - (-0.5) * log(X)")
    print("  残差 = log(Y) + 0.5 * log(X)")
    
    print("\n2. 交易信号逻辑:")
    print("-"*40)
    print("当Z > 2.2 (残差偏高，做空配对):")
    print("  意味着: log(Y) + 0.5*log(X) 偏高")
    print("  预期: 残差会回归均值（下降）")
    print("  操作: ")
    print("    - 卖Y (因为Y相对偏高)")
    print("    - 卖|β|份X (因为β<0，X也偏高)")
    print("    - 即：同时做空Y和X")
    
    print("\n当Z < -2.2 (残差偏低，做多配对):")
    print("  意味着: log(Y) + 0.5*log(X) 偏低")
    print("  预期: 残差会回归均值（上升）")
    print("  操作: ")
    print("    - 买Y (因为Y相对偏低)")
    print("    - 买|β|份X (因为β<0，X也偏低)")
    print("    - 即：同时做多Y和X")
    
    # 2. 检查实际信号
    print("\n3. 检查实际交易信号:")
    print("-"*40)
    
    # 加载信号文件
    signals = pd.read_csv('signals_rolling_ols_20250824_192926.csv')
    
    # 找出负Beta的开仓信号
    open_signals = signals[signals['signal'].str.contains('open')]
    neg_beta_signals = open_signals[open_signals['beta'] < 0]
    
    print(f"负Beta开仓信号数: {len(neg_beta_signals)}")
    
    # 显示几个例子
    print("\n负Beta信号示例:")
    for idx, row in neg_beta_signals.head(5).iterrows():
        print(f"\n配对: {row['pair']} ({row['symbol_x']}-{row['symbol_y']})")
        print(f"  Beta: {row['beta']:.4f}")
        print(f"  Z-score: {row['z_score']:.4f}")
        print(f"  信号: {row['signal']}")
        print(f"  残差: {row['residual']:.6f}")
        
        if 'long' in row['signal']:
            print(f"  → 做多配对 (Z < -2.2)")
            print(f"     买Y({row['symbol_y']}) + 买{abs(row['beta']):.2f}份X({row['symbol_x']})")
        else:
            print(f"  → 做空配对 (Z > 2.2)")
            print(f"     卖Y({row['symbol_y']}) + 卖{abs(row['beta']):.2f}份X({row['symbol_x']})")
    
    # 3. 检查回测引擎的处理
    print("\n4. 回测引擎的交易执行逻辑:")
    print("-"*40)
    print("当前实现（lib/backtest/trade_executor.py）:")
    print("  if direction == 'long':")
    print("    # 做多配对：买X卖Y")
    print("    open_price_x = price_x + slippage  # 买入加滑点")
    print("    open_price_y = price_y - slippage  # 卖出减滑点")
    print("  else:")
    print("    # 做空配对：卖X买Y")
    print("    open_price_x = price_x - slippage  # 卖出减滑点")
    print("    open_price_y = price_y + slippage  # 买入加滑点")
    
    print("\n问题分析:")
    print("  ✗ 当前逻辑假设做多配对总是'买X卖Y'")
    print("  ✗ 没有考虑β的符号")
    print("  ✗ 对于负β配对，方向完全错误！")
    
    print("\n正确的逻辑应该是:")
    print("  if direction == 'long':")
    print("    if beta > 0:")
    print("      # 正β：买Y卖X")
    print("    else:")
    print("      # 负β：买Y买X")
    print("  else: # short")
    print("    if beta > 0:")
    print("      # 正β：卖Y买X")
    print("    else:")
    print("      # 负β：卖Y卖X")
    
    # 4. 手数计算
    print("\n5. 手数计算逻辑:")
    print("-"*40)
    print("目标：保持价值对冲")
    print("  |β| * lots_x * price_x * multiplier_x = lots_y * price_y * multiplier_y")
    print("\n无论β正负，都使用|β|计算手数比例")
    print("但交易方向要根据β符号调整！")
    
    return neg_beta_signals

if __name__ == "__main__":
    neg_beta_signals = analyze_negative_beta_logic()