#!/usr/bin/env python3
"""
验证合约乘数问题
"""

import pandas as pd
import json
from pathlib import Path

project_root = Path(__file__).parent.parent

print("=" * 80)
print("验证合约乘数问题")
print("=" * 80)

# 读取合约规格
contract_specs_file = project_root / "configs" / "contract_specs.json"
with open(contract_specs_file, 'r') as f:
    specs = json.load(f)

# 读取交易记录
trades_file = project_root / "output" / "backtest_results" / "trades_20250822_193321.csv"
trades = pd.read_csv(trades_file)

print("验证前5笔交易的PnL计算:")

for i in range(min(5, len(trades))):
    trade = trades.iloc[i]
    symbol_x, symbol_y = trade['pair'].split('-')
    
    print(f"\n交易 {i+1}: {trade['pair']}")
    print(f"  方向: {trade['direction']}")
    print(f"  手数: Y={trade['contracts_y']}, X={trade['contracts_x']}")
    
    # 获取合约乘数
    mult_y = specs[symbol_y]['multiplier']
    mult_x = specs[symbol_x]['multiplier']
    print(f"  乘数: Y({symbol_y})={mult_y}, X({symbol_x})={mult_x}")
    
    # 价格变动
    delta_y = trade['close_price_y'] - trade['open_price_y']
    delta_x = trade['close_price_x'] - trade['open_price_x']
    
    # 计算理论PnL (含乘数)
    if trade['direction'] == 'long_spread':
        # 做多价差：买Y卖X
        theoretical_pnl = (delta_y * trade['contracts_y'] * mult_y) - (delta_x * trade['contracts_x'] * mult_x)
        print(f"  理论PnL(含乘数): ({delta_y}*{trade['contracts_y']}*{mult_y}) - ({delta_x}*{trade['contracts_x']}*{mult_x}) = {theoretical_pnl:.0f}")
    else:
        # 做空价差：卖Y买X
        theoretical_pnl = -(delta_y * trade['contracts_y'] * mult_y) + (delta_x * trade['contracts_x'] * mult_x)
        print(f"  理论PnL(含乘数): -({delta_y}*{trade['contracts_y']}*{mult_y}) + ({delta_x}*{trade['contracts_x']}*{mult_x}) = {theoretical_pnl:.0f}")
    
    print(f"  实际净PnL: {trade['net_pnl']:.0f}")
    print(f"  差异: {abs(theoretical_pnl - trade['net_pnl']):.0f}")
    
    # 估算手续费
    y_value = trade['open_price_y'] * trade['contracts_y'] * mult_y
    x_value = trade['open_price_x'] * trade['contracts_x'] * mult_x
    estimated_commission = (y_value + x_value) * 0.0002 * 2  # 开仓+平仓
    print(f"  估算手续费: {estimated_commission:.0f}")
    
    if abs(theoretical_pnl - trade['net_pnl']) <= estimated_commission * 1.5:
        print("  ✓ PnL计算正确")
    else:
        print("  ❌ PnL计算有误")

print(f"\n🔍 检查结论:")
print("如果上面的计算都正确，说明PnL计算本身没问题")
print("问题可能在于:")
print("1. 信号方向错误")
print("2. 手数分配错误") 
print("3. 价格数据错误")
print("4. Beta计算错误")