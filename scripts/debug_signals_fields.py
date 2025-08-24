#!/usr/bin/env python3
"""
调试signals字段内容
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator

# 快速生成signals
symbols = ['AG', 'AU']
data = {}
for symbol in symbols:
    data[symbol] = load_symbol_data(symbol)

prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
prices = prices.dropna()
log_prices = np.log(prices)

analyzer = CointegrationAnalyzer(log_prices)
pairs_df = analyzer.screen_all_pairs(p_threshold=0.05)

price_data = prices.reset_index()
price_data.rename(columns={'index': 'date'}, inplace=True)

pairs_params = {}
for _, row in pairs_df.iterrows():
    pair_name = row['pair']
    pairs_params[pair_name] = {
        'x': row['symbol_x'],
        'y': row['symbol_y'],
        'beta': row['beta_1y'],
        'beta_initial': row['beta_1y']
    }

generator = SignalGenerator(window=60, z_open=2.0, z_close=0.5, max_holding_days=30)
signals = generator.generate_all_signals(
    pairs_params=pairs_params,
    price_data=price_data,
    convergence_end='2024-07-01',
    signal_start='2024-07-01'
)

print("signals字段:")
print(signals.columns.tolist())

print("\n前5行signals数据:")
print(signals.head())

print("\nsymbol_x和symbol_y字段内容:")
open_signals = signals[signals['signal'].isin(['open_long', 'open_short'])]
if len(open_signals) > 0:
    first_signal = open_signals.iloc[0]
    print(f"第一个开仓信号的symbol_x: '{first_signal['symbol_x']}'")
    print(f"第一个开仓信号的symbol_y: '{first_signal['symbol_y']}'")
    print(f"第一个开仓信号的pair: '{first_signal['pair']}'")