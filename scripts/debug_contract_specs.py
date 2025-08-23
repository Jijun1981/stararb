#!/usr/bin/env python3
"""
调试合约规格加载问题
"""

import pandas as pd
import yaml
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("调试合约规格加载")
print("=" * 60)

# 1. 检查YAML合约规格
print("\n1. 检查YAML合约规格:")
business_config_file = project_root / "configs" / "business.yaml"
with open(business_config_file, 'r', encoding='utf-8') as f:
    business_config = yaml.safe_load(f)

contract_specs = business_config.get('contract_specs', {})
print(f"✓ 加载了 {len(contract_specs)} 个合约规格")

for symbol, spec in contract_specs.items():
    print(f"  {symbol}: multiplier={spec.get('multiplier')}, tick_size={spec.get('tick_size')}")

# 2. 测试I0配对为什么失败
print("\n2. 测试I0配对问题:")
sample_pairs = ['AU0-AG0', 'AG0-I0', 'CU0-I0', 'HC0-I0']

for pair in sample_pairs:
    symbol_x, symbol_y = pair.split('-')
    
    print(f"\n  配对: {pair}")
    print(f"    {symbol_x} in specs: {symbol_x in contract_specs}")
    print(f"    {symbol_y} in specs: {symbol_y in contract_specs}")
    
    if symbol_x in contract_specs and symbol_y in contract_specs:
        spec_x = contract_specs[symbol_x]
        spec_y = contract_specs[symbol_y]
        print(f"    {symbol_x}: {spec_x}")
        print(f"    {symbol_y}: {spec_y}")
    else:
        print(f"    ❌ 缺少合约规格")

# 3. 检查协整结果中的配对
print("\n3. 检查协整结果中的配对:")
coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
coint_results = pd.read_csv(coint_file)

print(f"✓ 协整结果: {len(coint_results)} 个配对")

# 分析哪些配对缺少合约规格
missing_specs = []
valid_specs = []

for _, row in coint_results.iterrows():
    symbol_x = row['symbol_x']
    symbol_y = row['symbol_y']
    pair = f"{symbol_x}-{symbol_y}"
    
    if symbol_x in contract_specs and symbol_y in contract_specs:
        valid_specs.append(pair)
    else:
        missing_specs.append(pair)
        missing_symbols = []
        if symbol_x not in contract_specs:
            missing_symbols.append(symbol_x)
        if symbol_y not in contract_specs:
            missing_symbols.append(symbol_y)
        print(f"  ❌ {pair}: 缺少 {missing_symbols}")

print(f"\n总结:")
print(f"  有效配对: {len(valid_specs)} 个")
print(f"  缺少规格: {len(missing_specs)} 个")

if len(valid_specs) > 0:
    print(f"  有效配对列表:")
    for pair in valid_specs[:10]:  # 只显示前10个
        print(f"    {pair}")