#!/usr/bin/env python3
"""
精确调试手数计算问题
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

from lib.backtest import BacktestEngine
import yaml

print("=" * 60)
print("精确调试手数计算")
print("=" * 60)

# 初始化BacktestEngine
backtest_engine = BacktestEngine(
    initial_capital=5000000,
    margin_rate=0.12,
    commission_rate=0.0002,
    slippage_ticks=3
)

# 加载合约规格
business_config_file = project_root / "configs" / "business.yaml"
with open(business_config_file, 'r', encoding='utf-8') as f:
    business_config = yaml.safe_load(f)

contract_specs = business_config.get('contract_specs', {})
backtest_engine.contract_specs = contract_specs

# 测试价格数据
current_prices = {
    'AG0': 5831.0,
    'I0': 839.0,
    'AU0': 476.68,
    'CU0': 66160.0
}

# 测试信号1: AG0-I0
print("\n1. 测试 AG0-I0 配对:")
signal1 = {
    'pair': 'AG0-I0',
    'theoretical_ratio': 0.32,
    'signal': 'open_long'
}

print(f"  理论比率: {signal1['theoretical_ratio']}")
print(f"  AG0价格: {current_prices['AG0']}, 合约乘数: {contract_specs['AG0']['multiplier']}")
print(f"  I0价格: {current_prices['I0']}, 合约乘数: {contract_specs['I0']['multiplier']}")

# 手动计算保证金
ag0_margin_per_lot = current_prices['AG0'] * contract_specs['AG0']['multiplier'] * 0.12
i0_margin_per_lot = current_prices['I0'] * contract_specs['I0']['multiplier'] * 0.12

print(f"  AG0单手保证金: {ag0_margin_per_lot:,.0f}")
print(f"  I0单手保证金: {i0_margin_per_lot:,.0f}")

# 预算
position_budget = 5000000 * 0.05  # 5%仓位
print(f"  配对预算: {position_budget:,.0f}")

# 手动验证比率计算
print(f"\n  手动验证:")
print(f"    理论比率 0.32 → 32:100 → 8:25")
print(f"    8手AG0 + 25手I0保证金: {8*ag0_margin_per_lot + 25*i0_margin_per_lot:,.0f}")
print(f"    实际比率: {8/25:.3f}")

# 尝试更简单的比率
print(f"    简化比率 3:10")
print(f"    3手AG0 + 10手I0保证金: {3*ag0_margin_per_lot + 10*i0_margin_per_lot:,.0f}")
print(f"    实际比率: {3/10:.3f}")

# 调用BacktestEngine的手数计算
print(f"\n  BacktestEngine计算结果:")
lots_result = backtest_engine.calculate_lots(
    signal=signal1,
    position_weight=0.05,
    current_prices=current_prices
)

if lots_result:
    print(f"    ✓ 成功:")
    print(f"      AG0手数: {lots_result['contracts_y']}")
    print(f"      I0手数: {lots_result['contracts_x']}")
    print(f"      保证金: {lots_result['margin_required']:,.0f}")
    print(f"      实际比率: {lots_result['contracts_y']/lots_result['contracts_x']:.3f}")
else:
    print(f"    ❌ 失败")
    
    # 逐步调试
    print(f"\n  逐步调试:")
    
    # 检查网格搜索算法
    result1 = backtest_engine._grid_search_lots(0.32, position_budget, ag0_margin_per_lot, i0_margin_per_lot)
    print(f"    网格搜索: {'✓' if result1 else '✗'}")
    if result1:
        print(f"      {result1}")
    
    # 检查比率约简算法
    result2 = backtest_engine._ratio_reduction_lots(0.32, position_budget, ag0_margin_per_lot, i0_margin_per_lot)
    print(f"    比率约简: {'✓' if result2 else '✗'}")
    if result2:
        print(f"      {result2}")
        
    # 检查线性规划算法
    result3 = backtest_engine._lp_approximation_lots(0.32, position_budget, ag0_margin_per_lot, i0_margin_per_lot)
    print(f"    线性规划: {'✓' if result3 else '✗'}")
    if result3:
        print(f"      {result3}")

# 测试信号2: CU0-I0  
print(f"\n" + "=" * 60)
print("2. 测试 CU0-I0 配对:")
signal2 = {
    'pair': 'CU0-I0',
    'theoretical_ratio': 0.10,
    'signal': 'open_long'
}

print(f"  理论比率: {signal2['theoretical_ratio']}")

cu0_margin_per_lot = current_prices['CU0'] * contract_specs['CU0']['multiplier'] * 0.12
print(f"  CU0单手保证金: {cu0_margin_per_lot:,.0f}")
print(f"  I0单手保证金: {i0_margin_per_lot:,.0f}")

lots_result2 = backtest_engine.calculate_lots(
    signal=signal2,
    position_weight=0.05,
    current_prices=current_prices
)

if lots_result2:
    print(f"  ✓ 成功:")
    print(f"    CU0手数: {lots_result2['contracts_y']}")
    print(f"    I0手数: {lots_result2['contracts_x']}")
else:
    print(f"  ❌ 失败")

print(f"\n" + "=" * 60)