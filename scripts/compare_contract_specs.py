#!/usr/bin/env python3
"""
对比两个合约规格文件的差异
"""

import json
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent

print("=" * 80)
print("对比合约规格文件")
print("=" * 80)

# 读取JSON格式
json_file = project_root / "configs" / "contract_specs.json"
with open(json_file, 'r', encoding='utf-8') as f:
    json_specs = json.load(f)

# 读取YAML格式
yaml_file = project_root / "configs" / "business.yaml"
with open(yaml_file, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)
    yaml_specs = yaml_config.get('contract_specs', {})

print(f"JSON文件: {len(json_specs)} 个合约")
print(f"YAML文件: {len(yaml_specs)} 个合约")

# 对比每个合约的乘数
print(f"\n合约乘数对比:")
print(f"{'品种':<8} {'JSON':<12} {'YAML':<12} {'差异'}")
print("-" * 50)

all_symbols = set(json_specs.keys()) | set(yaml_specs.keys())

for symbol in sorted(all_symbols):
    json_mult = json_specs.get(symbol, {}).get('multiplier', 'N/A')
    yaml_mult = yaml_specs.get(symbol, {}).get('multiplier', 'N/A')
    
    if json_mult != 'N/A' and yaml_mult != 'N/A':
        diff = "✓" if json_mult == yaml_mult else "✗"
    else:
        diff = "缺失"
    
    print(f"{symbol:<8} {str(json_mult):<12} {str(yaml_mult):<12} {diff}")

# 详细检查主要品种
print(f"\n详细对比主要品种:")
key_symbols = ['I0', 'CU0', 'AL0', 'AG0', 'AU0']

for symbol in key_symbols:
    print(f"\n{symbol}:")
    if symbol in json_specs:
        print(f"  JSON: {json_specs[symbol]}")
    if symbol in yaml_specs:
        print(f"  YAML: {yaml_specs[symbol]}")

# 重新计算I0的保证金（使用正确的乘数）
print(f"\n重新计算I0保证金:")
i0_price = 839.0
print(f"I0价格: {i0_price}元/吨")

if 'I0' in json_specs:
    json_margin = i0_price * json_specs['I0']['multiplier'] * 0.12
    print(f"JSON规格保证金: {i0_price} × {json_specs['I0']['multiplier']} × 0.12 = {json_margin:,.0f}元")

if 'I0' in yaml_specs:
    yaml_margin = i0_price * yaml_specs['I0']['multiplier'] * 0.12
    print(f"YAML规格保证金: {i0_price} × {yaml_specs['I0']['multiplier']} × 0.12 = {yaml_margin:,.0f}元")

print(f"\n配对预算: 250,000元")
print(f"JSON格式下可行性: {'✓' if json_margin < 250000 else '✗'}")