#!/usr/bin/env python3
"""
排查交易逻辑错误 - 多空方向、价格、手数
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.signal_generation import SignalGenerator

print("=" * 80)
print("排查交易逻辑错误")
print("=" * 80)

# 1. 检查信号生成的逻辑
print("\n1. 检查信号生成逻辑:")
print("理论：")
print("  价差 = Y - β*X")
print("  Z > 2.0 → 价差过高 → 做空价差 (卖Y买X)")
print("  Z < -2.0 → 价差过低 → 做多价差 (买Y卖X)")

# 2. 检查配对方向
print("\n2. 检查配对方向:")
print("我们用的配对格式: 'symbol_x-symbol_y'")
print("比如 'CU0-I0' 意思是:")
print("  X = CU0, Y = I0")
print("  价差 = I0 - β*CU0")
print("  如果β > 0，正常配对")
print("  如果β < 0，反向配对")

# 3. 检查具体例子
print("\n3. 检查具体交易例子:")

# 从最近的交易记录检查
trades_file = project_root / "output" / "backtest_results" / "trades_20250822_193321.csv"
if trades_file.exists():
    trades = pd.read_csv(trades_file)
    print(f"✓ 找到交易记录: {len(trades)}笔")
    
    # 分析前几笔交易
    print(f"\n前5笔交易分析:")
    for i in range(min(5, len(trades))):
        trade = trades.iloc[i]
        print(f"\n交易 {i+1}:")
        print(f"  配对: {trade['pair']}")
        print(f"  方向: {trade['direction']}")
        print(f"  Beta: {trade['beta']:.3f}")
        print(f"  手数: Y={trade['contracts_y']}, X={trade['contracts_x']}")
        print(f"  开仓价: Y={trade['open_price_y']}, X={trade['open_price_x']}")
        print(f"  平仓价: Y={trade['close_price_y']}, X={trade['close_price_x']}")
        print(f"  净PnL: {trade['net_pnl']:.0f}")
        
        # 验证PnL计算
        symbol_x, symbol_y = trade['pair'].split('-')
        
        # 价格变动
        delta_y = trade['close_price_y'] - trade['open_price_y'] 
        delta_x = trade['close_price_x'] - trade['open_price_x']
        
        print(f"  验证计算:")
        print(f"    Y价格变动: {delta_y:.2f}")
        print(f"    X价格变动: {delta_x:.2f}")
        
        # 理论PnL计算 (不含手续费)
        if trade['direction'] == 'long_spread':
            # 做多价差：买Y卖X
            theoretical_pnl = delta_y * trade['contracts_y'] - delta_x * trade['contracts_x']
            print(f"    理论PnL(多价差): {delta_y}*{trade['contracts_y']} - {delta_x}*{trade['contracts_x']} = {theoretical_pnl:.0f}")
        else:
            # 做空价差：卖Y买X  
            theoretical_pnl = -delta_y * trade['contracts_y'] + delta_x * trade['contracts_x']
            print(f"    理论PnL(空价差): -{delta_y}*{trade['contracts_y']} + {delta_x}*{trade['contracts_x']} = {theoretical_pnl:.0f}")
        
        print(f"    实际净PnL: {trade['net_pnl']:.0f}")
        print(f"    差异: {abs(theoretical_pnl - trade['net_pnl']):.0f} (应该约等于手续费)")
        
        # 检查这笔交易是否合理
        if trade['direction'] == 'long_spread' and theoretical_pnl < 0:
            print("    ⚠️  做多价差但亏损 - 可能信号方向错误")
        elif trade['direction'] == 'short_spread' and theoretical_pnl < 0:
            print("    ⚠️  做空价差但亏损 - 可能信号方向错误")
        else:
            print("    ✓ 方向看起来正确")

    # 统计盈亏情况
    print(f"\n整体统计:")
    total_trades = len(trades)
    profitable = (trades['net_pnl'] > 0).sum()
    losing = (trades['net_pnl'] < 0).sum()
    
    print(f"  总交易: {total_trades}")
    print(f"  盈利: {profitable} ({profitable/total_trades*100:.1f}%)")
    print(f"  亏损: {losing} ({losing/total_trades*100:.1f}%)")
    print(f"  平均PnL: {trades['net_pnl'].mean():.0f}")
    
    # 按方向分析
    long_trades = trades[trades['direction'] == 'long_spread']
    short_trades = trades[trades['direction'] == 'short_spread']
    
    if len(long_trades) > 0:
        print(f"  做多价差: {len(long_trades)}笔, 平均PnL: {long_trades['net_pnl'].mean():.0f}")
    if len(short_trades) > 0:
        print(f"  做空价差: {len(short_trades)}笔, 平均PnL: {short_trades['net_pnl'].mean():.0f}")
        
else:
    print("❌ 未找到交易记录文件")

# 4. 检查信号生成的方向逻辑
print(f"\n4. 检查信号生成的理论逻辑:")

# 模拟一个简单的配对和信号
print("模拟例子: CU0-I0配对")
print("  假设 β = 0.1 (正数)")
print("  假设当前 CU0=66000, I0=800")
print("  价差 = I0 - β*CU0 = 800 - 0.1*66000 = 800 - 6600 = -5800")
print("  如果历史均值是-5000，标准差是200")
print("  Z-score = (-5800 - (-5000))/200 = -800/200 = -4.0")
print("  Z < -2.0 → 价差过低 → 应该做多价差 (买I0卖CU0)")
print("  这意味着我们认为I0会涨或CU0会跌")

print(f"\n5. 可能的错误点:")
print("❓ 1. 配对方向是否一致 (X-Y vs Y-X)")
print("❓ 2. Beta符号是否正确")  
print("❓ 3. 信号方向映射是否正确")
print("❓ 4. 手数计算中的X/Y是否对应")
print("❓ 5. PnL计算中的多空方向是否正确")

print(f"\n💡 重点检查:")
print("1. 配对格式要统一")
print("2. 信号方向要对应价差的高低")
print("3. 手数分配要对应X和Y")
print("4. PnL计算要考虑实际持仓方向")