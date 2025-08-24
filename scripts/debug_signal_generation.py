#!/usr/bin/env python3
"""
调试信号生成问题
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.signal_generation import SignalGenerator

# 加载数据
print("加载数据...")
ag_data = load_symbol_data('AG')
au_data = load_symbol_data('AU')

# 准备价格数据
price_data = pd.DataFrame({
    'AG0': ag_data['close'],
    'AU0': au_data['close']
})

# 确保时间范围
price_data = price_data.loc['2024-01-01':'2025-08-20']
price_data = price_data.fillna(method='ffill')

print(f"价格数据形状: {price_data.shape}")
print(f"价格数据索引: {price_data.index.name}")
print(f"价格数据列: {price_data.columns.tolist()}")

# 将索引转换为列（signal_generation需要'date'列）
price_data = price_data.reset_index()
price_data.rename(columns={'index': 'date'}, inplace=True)

# 准备配对参数
pairs_params = {
    'AG0-AU0': {
        'x': 'AG0',
        'y': 'AU0',
        'beta': 0.0865,  # 从之前的结果
        'beta_initial': 0.0865  # 也加上这个以防万一
    }
}

# 创建信号生成器
generator = SignalGenerator(
    window=60,
    z_open=2.0,
    z_close=0.5
)

print("\n生成信号...")

# 尝试生成信号
try:
    signals = generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end='2024-07-01',
        signal_start='2024-07-01',
        hist_start='2024-01-01',
        hist_end='2025-08-20'
    )
    
    print(f"生成信号数量: {len(signals)}")
    if len(signals) > 0:
        print(f"信号列: {signals.columns.tolist()}")
        print(f"信号类型统计: {signals['signal'].value_counts().to_dict()}")
        print(f"前5条信号:\n{signals.head()}")
    else:
        print("没有生成信号")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

# 调试：直接检查数据处理
print("\n调试数据处理...")

# 模拟generate_all_signals内部逻辑
pair_name = 'AG0-AU0'
symbol_x, symbol_y = pair_name.split('-')

print(f"symbol_x: {symbol_x}, symbol_y: {symbol_y}")
print(f"检查列是否存在: {symbol_x} in columns: {symbol_x in price_data.columns}")
print(f"检查列是否存在: {symbol_y} in columns: {symbol_y in price_data.columns}")

# 检查是否需要重置索引
if price_data.index.name == 'date':
    print("索引名称是'date'，尝试重置索引...")
    price_data_reset = price_data.reset_index()
    print(f"重置后的列: {price_data_reset.columns.tolist()}")
    print(f"重置后前3行:\n{price_data_reset.head(3)}")