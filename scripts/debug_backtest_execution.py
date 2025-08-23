#!/usr/bin/env python3
"""
调试回测执行问题 - 为什么大部分信号无法执行
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入原子服务
from lib.data import load_data
from lib.signal_generation import SignalGenerator
from lib.backtest import BacktestEngine
import yaml

print("=" * 80)
print("调试回测执行问题")
print("=" * 80)

# 配置
TIME_CONFIG = {
    'data_start': '2019-01-01',
    'data_end': '2024-08-20',
    'convergence_end': '2023-06-30',
    'signal_start': '2023-07-01',
    'backtest_start': '2023-07-01',
    'hist_start': '2022-01-01',
    'hist_end': '2022-12-31'
}

BACKTEST_CONFIG = {
    'initial_capital': 5000000,
    'margin_rate': 0.12,
    'commission_rate': 0.0002,
    'slippage_ticks': 3,
    'position_weight': 0.05
}

# 1. 加载价格数据
print("1. 加载价格数据...")
symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']

price_data = load_data(
    symbols=symbols,
    start_date=TIME_CONFIG['data_start'],
    end_date=TIME_CONFIG['data_end'],
    columns=['close'],
    log_price=False,
    fill_method='ffill'
)

if 'date' not in price_data.columns:
    price_data = price_data.reset_index()

rename_dict = {col: col.replace('_close', '') for col in price_data.columns if col.endswith('_close')}
if rename_dict:
    price_data = price_data.rename(columns=rename_dict)

print(f"✓ 价格数据: {price_data.shape}")
print(f"  列名: {list(price_data.columns)}")

# 2. 检查特定日期的价格数据
test_date = '2023-10-20'
test_prices_data = price_data[price_data['date'] == test_date]

if not test_prices_data.empty:
    print(f"\n2. 检查 {test_date} 的价格数据:")
    current_prices = {}
    for col in test_prices_data.columns:
        if col != 'date':
            current_prices[col] = test_prices_data.iloc[0][col]
    
    print(f"  可用价格数据: {len(current_prices)} 个品种")
    for symbol, price in current_prices.items():
        print(f"    {symbol}: {price}")
else:
    print(f"\n❌ {test_date} 无价格数据")

# 3. 测试BacktestEngine手数计算
print(f"\n3. 测试BacktestEngine手数计算:")

# 初始化BacktestEngine并加载合约规格
backtest_engine = BacktestEngine(
    initial_capital=BACKTEST_CONFIG['initial_capital'],
    margin_rate=BACKTEST_CONFIG['margin_rate'],
    commission_rate=BACKTEST_CONFIG['commission_rate'],
    slippage_ticks=BACKTEST_CONFIG['slippage_ticks']
)

# 从YAML加载合约规格
business_config_file = project_root / "configs" / "business.yaml"
with open(business_config_file, 'r', encoding='utf-8') as f:
    business_config = yaml.safe_load(f)

contract_specs = business_config.get('contract_specs', {})
backtest_engine.contract_specs = contract_specs

print(f"✓ 合约规格加载: {len(backtest_engine.contract_specs)} 个")

# 4. 测试具体信号的手数计算
print(f"\n4. 测试具体信号的手数计算:")

# 模拟几个典型信号
test_signals = [
    {
        'pair': 'AU0-AG0',
        'theoretical_ratio': 1.4,
        'signal': 'open_long'
    },
    {
        'pair': 'AG0-I0', 
        'theoretical_ratio': 0.32,
        'signal': 'open_long'
    },
    {
        'pair': 'CU0-I0',
        'theoretical_ratio': 0.10,
        'signal': 'open_long'
    }
]

for signal in test_signals:
    print(f"\n  测试信号: {signal['pair']}")
    
    # 调用手数计算
    lots_result = backtest_engine.calculate_lots(
        signal=signal,
        position_weight=0.05,
        current_prices=current_prices
    )
    
    if lots_result:
        print(f"    ✓ 手数计算成功:")
        print(f"      Y合约手数: {lots_result['contracts_y']}")
        print(f"      X合约手数: {lots_result['contracts_x']}")
        print(f"      保证金需求: {lots_result['margin_required']:,.0f}")
    else:
        print(f"    ❌ 手数计算失败")
        
        # 详细诊断
        pair = signal['pair']
        symbol_x, symbol_y = pair.split('-')
        
        print(f"      配对解析: {symbol_x} - {symbol_y}")
        print(f"      {symbol_x} in specs: {symbol_x in contract_specs}")
        print(f"      {symbol_y} in specs: {symbol_y in contract_specs}")
        print(f"      {symbol_x} in prices: {symbol_x in current_prices}")
        print(f"      {symbol_y} in prices: {symbol_y in current_prices}")
        
        if symbol_x in current_prices and symbol_y in current_prices:
            print(f"      价格: {symbol_x}={current_prices[symbol_x]}, {symbol_y}={current_prices[symbol_y]}")

# 5. 检查实际的信号数据格式
print(f"\n5. 检查实际信号数据格式:")

# 加载协整结果
coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
coint_results = pd.read_csv(coint_file)

# 筛选有效配对（简化）
valid_pairs = coint_results.head(3)  # 只取前3个配对测试

pairs_params = {}
for _, row in valid_pairs.iterrows():
    pair_name = f"{row['symbol_x']}-{row['symbol_y']}"
    beta_initial = row.get('beta_4y', row.get('beta_1y', 1.0))
    
    pairs_params[pair_name] = {
        'symbol_x': row['symbol_x'],
        'symbol_y': row['symbol_y'],
        'beta_initial': beta_initial,
        'direction': row.get('direction', 'y_on_x')
    }

print(f"✓ 测试配对参数: {len(pairs_params)} 个")

# 为信号生成准备对数价格数据
log_price_data = load_data(
    symbols=symbols,
    start_date=TIME_CONFIG['data_start'],
    end_date=TIME_CONFIG['data_end'],
    columns=['close'],
    log_price=True,
    fill_method='ffill'
)

if 'date' not in log_price_data.columns:
    log_price_data = log_price_data.reset_index()

rename_dict = {col: col.replace('_close', '') for col in log_price_data.columns if col.endswith('_close')}
if rename_dict:
    log_price_data = log_price_data.rename(columns=rename_dict)

# 生成测试信号
signal_generator = SignalGenerator(
    window=60,
    z_open=2.0,
    z_close=0.5,
    convergence_days=20,
    convergence_threshold=0.01
)

test_signals = signal_generator.generate_all_signals(
    pairs_params=pairs_params,
    price_data=log_price_data,
    convergence_end=TIME_CONFIG['convergence_end'],
    signal_start=TIME_CONFIG['signal_start'],
    hist_start=TIME_CONFIG['hist_start'],
    hist_end=TIME_CONFIG['hist_end']
)

print(f"✓ 生成测试信号: {len(test_signals)} 条")

# 检查信号格式
if len(test_signals) > 0:
    sample_signal = test_signals.iloc[0]
    print(f"  信号样例:")
    for key, value in sample_signal.items():
        print(f"    {key}: {value}")
        
    # 测试该信号的手数计算
    print(f"\n  测试信号手数计算:")
    lots_result = backtest_engine.calculate_lots(
        signal=sample_signal.to_dict(),
        position_weight=0.05,
        current_prices=current_prices
    )
    
    if lots_result:
        print(f"    ✓ 成功")
    else:
        print(f"    ❌ 失败")

print(f"\n" + "=" * 80)
print("调试完成")
print("=" * 80)