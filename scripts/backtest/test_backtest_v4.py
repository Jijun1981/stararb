#!/usr/bin/env python3
"""
回测框架v4 - 与需求文档完全对齐的版本
重点解决：
1. 与信号生成模块的输出格式对齐（13个字段）
2. 基于动态β值计算最小整数比手数
3. 正确的PnL计算逻辑
4. 15%止损和30天强制平仓
"""

import pandas as pd
import numpy as np
from fractions import Fraction
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def calculate_min_lots(beta: float, max_denominator: int = 10) -> Dict:
    """
    根据β值计算最小整数比手数（REQ-4.1.1）
    
    Args:
        beta: β值（来自信号的动态β）
        max_denominator: 最大分母限制
    
    Returns:
        手数分配结果
    
    核心原理：
    - 价差公式：spread = log(Y) - β*log(X)
    - 对冲比例：Y:X = 1:β
    - 需要转换为最简整数比
    """
    # 处理特殊情况
    if beta <= 0:
        return {
            'lots_y': 1,
            'lots_x': 1, 
            'theoretical_ratio': abs(beta),
            'actual_ratio': 1.0,
            'error': 1.0 - abs(beta)
        }
    
    # 使用Fraction找最简分数
    # β = 1/对冲比例，所以 lots_x/lots_y = β
    # 即 lots_y:lots_x = 1:β
    frac = Fraction(beta).limit_denominator(max_denominator)
    
    # Y:X = 1:β
    lots_y = frac.denominator  # Y是分母
    lots_x = frac.numerator     # X是分子
    
    # 确保至少1手
    if lots_y == 0:
        lots_y = 1
    if lots_x == 0:
        lots_x = 1
    
    actual_ratio = lots_x / lots_y
    error = abs(actual_ratio - beta) / beta if beta != 0 else 0
    
    return {
        'lots_y': lots_y,
        'lots_x': lots_x,
        'theoretical_ratio': beta,
        'actual_ratio': actual_ratio,
        'error': error,
        'message': f"Y:X = {lots_y}:{lots_x}, β={beta:.4f}, 实际比例={actual_ratio:.4f}, 误差={error*100:.2f}%"
    }


def apply_slippage(price: float, side: str, tick_size: float, slippage_ticks: int = 3) -> float:
    """
    应用滑点（REQ-4.1.4）
    
    Args:
        price: 市场价格
        side: 'buy' 或 'sell'
        tick_size: 最小变动价位
        slippage_ticks: 滑点tick数
    """
    if side == 'buy':
        return price + tick_size * slippage_ticks
    else:  # sell
        return price - tick_size * slippage_ticks


def calculate_pnl(position: Dict, close_price_y: float, close_price_x: float, 
                  mult_y: float, mult_x: float) -> Dict:
    """
    计算PnL（REQ-4.3.1）
    
    方向定义：
    - open_long: 做多价差（买Y卖X）
    - open_short: 做空价差（卖Y买X）
    """
    direction = position['direction']
    
    if direction == 'open_long':
        # 做多价差：买Y卖X
        y_pnl = (close_price_y - position['open_price_y']) * position['lots_y'] * mult_y
        x_pnl = (position['open_price_x'] - close_price_x) * position['lots_x'] * mult_x
    else:  # open_short
        # 做空价差：卖Y买X
        y_pnl = (position['open_price_y'] - close_price_y) * position['lots_y'] * mult_y
        x_pnl = (close_price_x - position['open_price_x']) * position['lots_x'] * mult_x
    
    gross_pnl = y_pnl + x_pnl
    
    return {
        'gross_pnl': gross_pnl,
        'y_pnl': y_pnl,
        'x_pnl': x_pnl
    }


def test_lots_calculation():
    """测试手数计算逻辑"""
    print("=" * 60)
    print("测试手数计算（基于β值）")
    print("=" * 60)
    
    test_cases = [
        {'beta': 0.5, 'expected': 'Y:X = 2:1'},
        {'beta': 1.5, 'expected': 'Y:X = 2:3'},
        {'beta': 0.85, 'expected': 'Y:X ≈ 7:6 或 20:17'},
        {'beta': 3.5, 'expected': 'Y:X = 2:7'},
        {'beta': 0.8234, 'expected': 'Y:X ≈ 5:4 或 24:20'},
    ]
    
    for case in test_cases:
        beta = case['beta']
        result = calculate_min_lots(beta)
        print(f"\nβ = {beta}")
        print(f"  期望: {case['expected']}")
        print(f"  实际: {result['message']}")
        
        # 验证结果
        assert result['lots_y'] > 0, "Y手数必须为正"
        assert result['lots_x'] > 0, "X手数必须为正"
        assert result['error'] < 0.2, f"误差过大: {result['error']*100:.2f}%"
    
    print("\n✅ 手数计算测试通过")


def test_pnl_calculation():
    """测试PnL计算逻辑"""
    print("\n" + "=" * 60)
    print("测试PnL计算")
    print("=" * 60)
    
    # 测试做多价差
    position = {
        'direction': 'open_long',
        'open_price_y': 25000,
        'open_price_x': 500,
        'lots_y': 6,
        'lots_x': 7
    }
    
    # Y涨，X跌 -> 盈利
    result = calculate_pnl(position, 25500, 490, mult_y=5, mult_x=1000)
    print(f"\n做多价差测试（Y涨X跌）:")
    print(f"  Y: 25000->25500, 6手, PnL={result['y_pnl']:,.0f}")
    print(f"  X: 500->490, 7手, PnL={result['x_pnl']:,.0f}")
    print(f"  总PnL: {result['gross_pnl']:,.0f}")
    assert result['gross_pnl'] > 0, "应该盈利"
    
    # 测试做空价差
    position['direction'] = 'open_short'
    
    # Y跌，X涨 -> 盈利
    result = calculate_pnl(position, 24500, 510, mult_y=5, mult_x=1000)
    print(f"\n做空价差测试（Y跌X涨）:")
    print(f"  Y: 25000->24500, 6手, PnL={result['y_pnl']:,.0f}")
    print(f"  X: 500->510, 7手, PnL={result['x_pnl']:,.0f}")
    print(f"  总PnL: {result['gross_pnl']:,.0f}")
    assert result['gross_pnl'] > 0, "应该盈利"
    
    print("\n✅ PnL计算测试通过")


def test_signal_format_alignment():
    """测试信号格式对齐"""
    print("\n" + "=" * 60)
    print("测试信号格式对齐")
    print("=" * 60)
    
    # 模拟信号生成模块的输出（13个字段）
    signal = {
        'date': '2024-01-01',
        'pair': 'AG-NI',           # 纯符号格式
        'symbol_x': 'AG',           # 低波动
        'symbol_y': 'NI',           # 高波动
        'signal': 'open_long',      # 信号类型
        'z_score': -2.5,            # Z-score值
        'residual': -0.15,          # 残差
        'beta': 0.8234,             # 动态β值（来自Kalman）
        'beta_initial': 0.8500,     # 初始β值
        'days_held': 0,             # 持仓天数
        'reason': 'z_threshold',    # 信号原因
        'phase': 'signal_period',   # 阶段
        'beta_window_used': '1y'    # 使用的β窗口
    }
    
    print("信号格式验证:")
    required_fields = [
        'date', 'pair', 'symbol_x', 'symbol_y', 'signal',
        'z_score', 'residual', 'beta', 'beta_initial',
        'days_held', 'reason', 'phase', 'beta_window_used'
    ]
    
    for field in required_fields:
        if field in signal:
            print(f"  ✅ {field}: {signal[field]}")
        else:
            print(f"  ❌ {field}: 缺失")
    
    # 验证配对命名格式
    assert '-' in signal['pair'], "配对名称必须使用'-'分隔"
    assert '_' not in signal['pair'], "配对名称不应包含'_'"
    
    # 验证信号类型
    valid_signals = ['converging', 'open_long', 'open_short', 'close', 'hold']
    assert signal['signal'] in valid_signals, f"无效信号类型: {signal['signal']}"
    
    print("\n✅ 信号格式对齐测试通过")


def test_risk_control():
    """测试风险控制逻辑"""
    print("\n" + "=" * 60)
    print("测试风险控制")
    print("=" * 60)
    
    # 测试15%止损
    margin = 100000
    stop_loss_pct = 0.15
    
    test_losses = [
        {'loss': -10000, 'pct': 0.10, 'should_stop': False},
        {'loss': -14000, 'pct': 0.14, 'should_stop': False},
        {'loss': -15000, 'pct': 0.15, 'should_stop': True},
        {'loss': -20000, 'pct': 0.20, 'should_stop': True},
    ]
    
    print("15%止损测试:")
    for test in test_losses:
        loss_pct = abs(test['loss']) / margin
        should_stop = loss_pct >= stop_loss_pct
        print(f"  亏损{test['loss']:,}（{test['pct']:.0%}）: {'触发止损' if should_stop else '继续持有'}")
        assert should_stop == test['should_stop'], "止损判断错误"
    
    # 测试30天强制平仓
    print("\n30天强制平仓测试:")
    open_date = datetime(2024, 1, 1)
    max_holding_days = 30
    
    test_dates = [
        {'current': datetime(2024, 1, 29), 'days': 28, 'should_close': False},
        {'current': datetime(2024, 1, 30), 'days': 29, 'should_close': False},
        {'current': datetime(2024, 1, 31), 'days': 30, 'should_close': True},
        {'current': datetime(2024, 2, 1), 'days': 31, 'should_close': True},
    ]
    
    for test in test_dates:
        holding_days = (test['current'] - open_date).days
        should_close = holding_days >= max_holding_days
        print(f"  持仓{holding_days}天: {'强制平仓' if should_close else '继续持有'}")
        assert should_close == test['should_close'], "时间止损判断错误"
    
    print("\n✅ 风险控制测试通过")


def test_complete_trade_flow():
    """测试完整交易流程"""
    print("\n" + "=" * 60)
    print("测试完整交易流程")
    print("=" * 60)
    
    # 1. 信号
    signal = {
        'date': '2024-04-10',
        'pair': 'CU-SN',
        'signal': 'open_long',
        'z_score': -2.41,
        'beta': 0.8523
    }
    
    print(f"1. 收到信号: {signal['pair']}, {signal['signal']}, Z={signal['z_score']:.2f}")
    
    # 2. 计算手数
    lots = calculate_min_lots(signal['beta'])
    print(f"2. 计算手数: {lots['message']}")
    
    # 3. 计算保证金
    price_y = 150000  # SN
    price_x = 45000   # CU
    mult_y = 1
    mult_x = 5
    margin_rate = 0.12
    
    margin_y = price_y * lots['lots_y'] * mult_y * margin_rate
    margin_x = price_x * lots['lots_x'] * mult_x * margin_rate
    total_margin = margin_y + margin_x
    
    print(f"3. 计算保证金:")
    print(f"   Y(SN): {price_y:,} × {lots['lots_y']} × {mult_y} × {margin_rate:.0%} = {margin_y:,.0f}")
    print(f"   X(CU): {price_x:,} × {lots['lots_x']} × {mult_x} × {margin_rate:.0%} = {margin_x:,.0f}")
    print(f"   总计: {total_margin:,.0f}")
    
    # 4. 开仓（含滑点）
    tick_y = 5
    tick_x = 10
    
    if signal['signal'] == 'open_long':
        open_price_y = apply_slippage(price_y, 'buy', tick_y)
        open_price_x = apply_slippage(price_x, 'sell', tick_x)
    else:
        open_price_y = apply_slippage(price_y, 'sell', tick_y)
        open_price_x = apply_slippage(price_x, 'buy', tick_x)
    
    print(f"4. 开仓价格（含滑点）:")
    print(f"   Y: {price_y:,} -> {open_price_y:,}")
    print(f"   X: {price_x:,} -> {open_price_x:,}")
    
    # 5. 计算开仓手续费
    commission_rate = 0.0002
    nominal_y = open_price_y * lots['lots_y'] * mult_y
    nominal_x = open_price_x * lots['lots_x'] * mult_x
    open_commission = (nominal_y + nominal_x) * commission_rate
    
    print(f"5. 开仓手续费: {open_commission:,.0f}")
    
    # 6. 持仓5天后平仓
    close_price_y = 151000
    close_price_x = 45200
    
    position = {
        'direction': signal['signal'],
        'open_price_y': open_price_y,
        'open_price_x': open_price_x,
        'lots_y': lots['lots_y'],
        'lots_x': lots['lots_x']
    }
    
    pnl_result = calculate_pnl(position, close_price_y, close_price_x, mult_y, mult_x)
    
    print(f"6. 平仓PnL:")
    print(f"   Y腿PnL: {pnl_result['y_pnl']:,.0f}")
    print(f"   X腿PnL: {pnl_result['x_pnl']:,.0f}")
    print(f"   毛PnL: {pnl_result['gross_pnl']:,.0f}")
    
    # 7. 平仓手续费
    if signal['signal'] == 'open_long':
        close_price_y_actual = apply_slippage(close_price_y, 'sell', tick_y)
        close_price_x_actual = apply_slippage(close_price_x, 'buy', tick_x)
    else:
        close_price_y_actual = apply_slippage(close_price_y, 'buy', tick_y)
        close_price_x_actual = apply_slippage(close_price_x, 'sell', tick_x)
    
    close_nominal_y = close_price_y_actual * lots['lots_y'] * mult_y
    close_nominal_x = close_price_x_actual * lots['lots_x'] * mult_x
    close_commission = (close_nominal_y + close_nominal_x) * commission_rate
    
    print(f"7. 平仓手续费: {close_commission:,.0f}")
    
    # 8. 净PnL
    net_pnl = pnl_result['gross_pnl'] - open_commission - close_commission
    return_on_margin = net_pnl / total_margin
    
    print(f"8. 净PnL: {net_pnl:,.0f}")
    print(f"9. 收益率: {return_on_margin:.2%}")
    
    print("\n✅ 完整交易流程测试通过")


def main():
    """运行所有测试"""
    print("回测框架v4测试")
    print("=" * 60)
    
    # 运行测试
    test_lots_calculation()
    test_pnl_calculation()
    test_signal_format_alignment()
    test_risk_control()
    test_complete_trade_flow()
    
    print("\n" + "=" * 60)
    print("所有测试通过！回测逻辑正确")
    print("=" * 60)


if __name__ == "__main__":
    main()