#!/usr/bin/env python3
"""
测试run_complete_pipeline_v2_1.py中的所有计算逻辑
重点验证方向是否正确
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator, calculate_ols_beta
from lib.backtest import BacktestEngine
from statsmodels.api import OLS, add_constant

print("=" * 80)
print("Pipeline v2.1 计算验证测试")
print("=" * 80)

# ============================================================================
# 测试1: 验证协整筛选的方向
# ============================================================================
print("\n【测试1】验证协整筛选方向")
print("-" * 60)

# 加载数据
print("加载数据...")
data = load_data(
    symbols=['AL0', 'SN0', 'HC0', 'I0', 'SF0'],
    start_date='2020-01-01',
    columns=['close'],
    log_price=True
)

# 创建协整分析器
analyzer = CointegrationAnalyzer(data)

# 筛选配对
filtered_pairs = analyzer.screen_all_pairs(p_threshold=0.05)

print(f"\n筛选出的配对数: {len(filtered_pairs)}")
if len(filtered_pairs) > 0:
    print("\n前5个配对的方向信息:")
    for idx, row in filtered_pairs.head(5).iterrows():
        pair = row['pair']
        symbol_x = row['symbol_x'].replace('_close', '')
        symbol_y = row['symbol_y'].replace('_close', '')
        direction = row['direction']
        beta = row['beta_5y']
        
        print(f"\n{idx+1}. {pair}")
        print(f"   symbol_x: {symbol_x}")
        print(f"   symbol_y: {symbol_y}")
        print(f"   direction: {direction}")
        print(f"   beta_5y: {beta:.6f}")
        
        # 验证方向逻辑
        if direction == 'y_on_x':
            print(f"   含义: {symbol_y} = α + {beta:.3f} × {symbol_x}")
            print(f"   回归: Y={symbol_y}, X={symbol_x}")
        else:
            print(f"   含义: {symbol_x} = α + {beta:.3f} × {symbol_y}")
            print(f"   回归: Y={symbol_x}, X={symbol_y}")

# ============================================================================
# 测试2: 验证初始Beta计算
# ============================================================================
print("\n\n【测试2】验证初始Beta计算")
print("-" * 60)

# 使用2023年数据
data_2023 = load_data(
    symbols=['AL0', 'SN0', 'HC0', 'I0'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    columns=['close'],
    log_price=True
)

# 测试AL0-SN0配对
if 'AL0_close' in data_2023.columns and 'SN0_close' in data_2023.columns:
    log_al = data_2023['AL0_close'].values
    log_sn = data_2023['SN0_close'].values
    
    # 假设从协整分析得到direction='y_on_x'，即SN0是Y，AL0是X
    print("\n测试配对: AL0-SN0 (假设direction=y_on_x)")
    
    # 方法1: 使用calculate_ols_beta
    beta1 = calculate_ols_beta(log_sn, log_al, window=len(log_sn))
    print(f"calculate_ols_beta结果: {beta1:.6f}")
    
    # 方法2: 手动计算验证
    covariance = np.cov(log_sn, log_al, ddof=1)[0, 1]
    variance_al = np.var(log_al, ddof=1)
    beta2 = covariance / variance_al
    print(f"手动计算结果: {beta2:.6f}")
    
    # 方法3: statsmodels验证
    X = add_constant(log_al)
    model = OLS(log_sn, X).fit()
    beta3 = model.params[1]
    print(f"statsmodels结果: {beta3:.6f}")
    
    # 验证是否一致
    if abs(beta1 - beta2) < 1e-6 and abs(beta2 - beta3) < 1e-6:
        print("✓ 三种方法计算结果一致")
    else:
        print("✗ 计算结果不一致!")

# ============================================================================
# 测试3: 验证信号生成的配对方向
# ============================================================================
print("\n\n【测试3】验证信号生成的配对方向")
print("-" * 60)

# 创建测试用的pairs_params
test_params = {
    'AL0-SN0': {
        'beta_initial': 7.837609,  # 从之前的分析得到
        'symbol_x': 'AL0_close',
        'symbol_y': 'SN0_close',
        'R': 0.001
    },
    'HC0-I0': {
        'beta_initial': 5.435185,
        'symbol_x': 'HC0_close',
        'symbol_y': 'I0_close',
        'R': 0.001
    }
}

# 加载价格数据（注意：信号生成使用原始价格，不是对数价格）
price_data = load_data(
    symbols=['AL0', 'SN0', 'HC0', 'I0'],
    start_date='2023-01-01',
    columns=['close'],
    log_price=False  # 原始价格
)

print("\n验证信号生成中的配对格式:")
for pair, params in test_params.items():
    print(f"\n配对: {pair}")
    print(f"  symbol_x: {params['symbol_x']}")
    print(f"  symbol_y: {params['symbol_y']}")
    print(f"  beta_initial: {params['beta_initial']:.4f}")
    
    # 从配对名称解析
    parts = pair.split('-')
    if len(parts) == 2:
        pair_symbol_x = parts[0]
        pair_symbol_y = parts[1]
        print(f"  从配对名称解析: X={pair_symbol_x}, Y={pair_symbol_y}")
        
        # 验证是否与params一致
        if f"{pair_symbol_x}_close" == params['symbol_x'] and \
           f"{pair_symbol_y}_close" == params['symbol_y']:
            print("  ✓ 配对名称与参数一致")
        else:
            print("  ✗ 配对名称与参数不一致!")
            print(f"    期望: X={pair_symbol_x}_close, Y={pair_symbol_y}_close")
            print(f"    实际: X={params['symbol_x']}, Y={params['symbol_y']}")

# ============================================================================
# 测试4: 验证回测中的配对拆分
# ============================================================================
print("\n\n【测试4】验证回测中的配对拆分")
print("-" * 60)

# 测试不同的配对格式
test_pairs = [
    'AL0-SN0',
    'HC0-I0',
    'SF0-SN0',
    'RB0-I0'
]

print("测试配对拆分逻辑:")
for pair in test_pairs:
    print(f"\n配对: {pair}")
    
    # 回测中的拆分逻辑（来自backtest.py第255行和342行）
    # symbol_y, symbol_x = pair.split('-')  # 配对格式是Y-X
    parts = pair.split('-')
    if len(parts) == 2:
        # 按照backtest.py的逻辑
        symbol_y, symbol_x = parts
        print(f"  backtest.py拆分: Y={symbol_y}, X={symbol_x}")
        
        # 但是配对名称是X-Y格式！
        # 正确的应该是：
        symbol_x_correct, symbol_y_correct = parts
        print(f"  正确的拆分应该是: X={symbol_x_correct}, Y={symbol_y_correct}")
        
        if symbol_x == symbol_x_correct and symbol_y == symbol_y_correct:
            print("  ✓ 拆分逻辑正确")
        else:
            print("  ✗ 拆分逻辑有问题!")
            print("    配对名称格式是X-Y，但代码当成了Y-X")

# ============================================================================
# 测试5: 端到端验证
# ============================================================================
print("\n\n【测试5】端到端验证")
print("-" * 60)

# 选择一个具体配对进行完整验证
test_pair = 'AL0-SN0'
print(f"测试配对: {test_pair}")

# 1. 协整分析阶段
print("\n1. 协整分析阶段:")
data_coint = load_data(
    symbols=['AL0', 'SN0'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    columns=['close'],
    log_price=True
)

# 手动计算方向
log_al = data_coint['AL0_close'].values
log_sn = data_coint['SN0_close'].values

# 计算两个方向的Beta
# 方向1: SN0 = α + β × AL0 (y_on_x)
cov_sn_al = np.cov(log_sn, log_al, ddof=1)[0, 1]
var_al = np.var(log_al, ddof=1)
beta_y_on_x = cov_sn_al / var_al

# 方向2: AL0 = α + β × SN0 (x_on_y)
var_sn = np.var(log_sn, ddof=1)
beta_x_on_y = cov_sn_al / var_sn

print(f"  方向1 (SN0 on AL0): β = {beta_y_on_x:.6f}")
print(f"  方向2 (AL0 on SN0): β = {beta_x_on_y:.6f}")

# 根据Beta大小选择方向（通常选择|β|更接近1的方向）
if abs(beta_y_on_x - 1) < abs(beta_x_on_y - 1):
    selected_direction = 'y_on_x'
    selected_beta = beta_y_on_x
    print(f"  选择方向: y_on_x (SN0是Y，AL0是X)")
else:
    selected_direction = 'x_on_y'
    selected_beta = beta_x_on_y
    print(f"  选择方向: x_on_y (AL0是Y，SN0是X)")

# 2. 信号生成阶段
print("\n2. 信号生成阶段:")
print(f"  配对名称: {test_pair}")
print(f"  Beta: {selected_beta:.6f}")
print(f"  方向: {selected_direction}")

# 3. 回测阶段
print("\n3. 回测阶段:")
print(f"  配对: {test_pair}")
# 当前backtest.py的逻辑
symbol_y_wrong, symbol_x_wrong = test_pair.split('-')
print(f"  当前代码拆分: Y={symbol_y_wrong}, X={symbol_x_wrong}")

# 正确的逻辑
symbol_x_right, symbol_y_right = test_pair.split('-')
print(f"  正确的拆分: X={symbol_x_right}, Y={symbol_y_right}")

if selected_direction == 'y_on_x':
    if symbol_x_right == 'AL0' and symbol_y_right == 'SN0':
        print("  ✓ 方向一致性验证通过")
    else:
        print("  ✗ 方向不一致!")

# ============================================================================
# 测试6: 验证实际价格计算
# ============================================================================
print("\n\n【测试6】验证实际价格计算")
print("-" * 60)

# 加载实际价格
price_data = load_data(
    symbols=['SF0', 'SN0'],
    start_date='2024-08-01',
    end_date='2024-08-20',
    columns=['close'],
    log_price=False
)

print("\nSF0-SN0配对的实际价格:")
print(f"SF0最新价格: {price_data['SF0_close'].iloc[-1]:.2f}")
print(f"SN0最新价格: {price_data['SN0_close'].iloc[-1]:.2f}")

# 根据之前的分析，SF0价格应该在6000-8000范围
if 6000 <= price_data['SF0_close'].iloc[-1] <= 8000:
    print("✓ SF0价格在合理范围内")
else:
    print("✗ SF0价格异常!")

# SN0价格应该在140000-180000范围
if 140000 <= price_data['SN0_close'].iloc[-1] <= 180000:
    print("✓ SN0价格在合理范围内")
else:
    print("✗ SN0价格异常!")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

# ============================================================================
# 总结问题
# ============================================================================
print("\n【问题总结】")
print("-" * 60)
print("1. 配对命名约定:")
print("   - 协整分析输出: X-Y格式（如AL0-SN0表示AL0是X，SN0是Y）")
print("   - 信号生成期望: X-Y格式")
print("   - 回测拆分逻辑: 错误地当成Y-X格式")
print("")
print("2. 影响:")
print("   - backtest.py第255行和342行的拆分逻辑颠倒了X和Y")
print("   - 导致计算PnL时X和Y的价格对调")
print("   - 这就是为什么SF0-SN0显示SF0价格26万的原因")
print("")
print("3. 修复建议:")
print("   - 修改backtest.py中的拆分逻辑")
print("   - 从 symbol_y, symbol_x = pair.split('-')")
print("   - 改为 symbol_x, symbol_y = pair.split('-')")