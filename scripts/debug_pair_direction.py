#!/usr/bin/env python3
"""
检查配对方向问题 - 这可能是核心问题
"""

import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("检查配对方向问题")  
print("=" * 80)

# 读取协整结果
coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
coint_results = pd.read_csv(coint_file)

# 读取交易记录
trades_file = project_root / "output" / "backtest_results" / "trades_20250822_193321.csv" 
trades = pd.read_csv(trades_file)

print("🔍 关键问题发现:")
print("\n1. 协整结果的配对格式:")
print("前5个配对:")
for i in range(min(5, len(coint_results))):
    row = coint_results.iloc[i]
    print(f"  {row['pair']}: {row['symbol_x']} - {row['symbol_y']}")

print("\n2. 交易记录的配对格式:")
unique_pairs = trades['pair'].unique()[:5]
for pair in unique_pairs:
    print(f"  {pair}")

print(f"\n3. 检查格式一致性:")
print("协整结果格式: symbol_x-symbol_y")
print("交易记录格式: symbol_?-symbol_?")

# 检查具体配对的方向
print(f"\n4. 检查HC0-RB0配对:")
hc_rb_coint = coint_results[coint_results['pair'] == 'HC0-RB0']
if not hc_rb_coint.empty:
    row = hc_rb_coint.iloc[0]
    print(f"  协整结果: X={row['symbol_x']}, Y={row['symbol_y']}")
    print(f"  Beta: {row.get('beta_4y', 'N/A')}")
else:
    print("  ❌ 协整结果中未找到HC0-RB0")

hc_rb_trades = trades[trades['pair'] == 'HC0-RB0']
if not hc_rb_trades.empty:
    trade = hc_rb_trades.iloc[0]
    print(f"  交易记录: 手数Y={trade['contracts_y']}, X={trade['contracts_x']}")
    print(f"  交易Beta: {trade['beta']}")
    
    # 推断X和Y
    # 从BacktestEngine的逻辑看，pair.split('-')的结果应该是[symbol_x, symbol_y]
    # 所以HC0-RB0应该是X=HC0, Y=RB0
    print(f"  推断: X=HC0, Y=RB0")
    print(f"  价差公式: RB0 - β*HC0")
    
    # 检查这是否合理
    print(f"  检查合理性:")
    print(f"    如果β=0.994 (接近1)")
    print(f"    价差 ≈ RB0 - HC0")
    print(f"    这两个都是钢材品种，价差接近0是合理的")

print(f"\n5. 检查CU0-SM0配对:")
cu_sm_coint = coint_results[coint_results['pair'] == 'CU0-SM0']
if not cu_sm_coint.empty:
    row = cu_sm_coint.iloc[0]
    print(f"  协整结果: X={row['symbol_x']}, Y={row['symbol_y']}")
    print(f"  Beta: {row.get('beta_4y', 'N/A')}")
    
    # 这里可能有问题！
    print(f"  ⚠️ 注意: CU0是铜(价格~66000), SM0是锰硅(价格~6800)")
    print(f"  如果X=CU0, Y=SM0, 价差 = SM0 - β*CU0")
    print(f"  但是CU0价格比SM0高10倍！β应该很小才对")

cu_sm_trades = trades[trades['pair'] == 'CU0-SM0']
if not cu_sm_trades.empty:
    trade = cu_sm_trades.iloc[0]
    print(f"  交易记录: Beta={trade['beta']:.3f}")
    if trade['beta'] > 0.5:
        print(f"  ❌ Beta太大！CU0价格是SM0的10倍，β应该<0.1")

print(f"\n💡 可能的问题:")
print("1. 配对方向可能颠倒了")
print("2. 协整分析时X/Y的定义与交易时不一致") 
print("3. 需要检查价格量级差异巨大的配对")
print("4. Beta值是否合理 (考虑价格量级)")

# 检查价格量级
print(f"\n6. 检查价格量级:")
from lib.data import load_data

price_data = load_data(
    symbols=['HC0', 'RB0', 'CU0', 'SM0'],
    start_date='2024-08-01',
    end_date='2024-08-20', 
    columns=['close'],
    log_price=False
)

if not price_data.empty:
    latest = price_data.iloc[-1]
    print(f"  HC0: {latest.get('HC0_close', 'N/A')}")
    print(f"  RB0: {latest.get('RB0_close', 'N/A')}")
    print(f"  CU0: {latest.get('CU0_close', 'N/A')}")
    print(f"  SM0: {latest.get('SM0_close', 'N/A')}")
    print(f"  CU0/SM0比值: {latest.get('CU0_close', 0)/latest.get('SM0_close', 1):.1f}")