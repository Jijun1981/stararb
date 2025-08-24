#!/usr/bin/env python3
"""
手动验证所有计算的正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.backtest_core import *

def verify_pnl_calculation():
    """验证PnL计算逻辑"""
    print("=" * 60)
    print("验证PnL计算")
    print("=" * 60)
    
    # 场景1：做多价差（买Y卖X）
    print("\n场景1：做多价差盈利")
    position = {
        'direction': 'long',
        'lots_y': 10,
        'lots_x': 8,
        'entry_price_y': 1000,
        'entry_price_x': 2000,
        'margin': 50000,
        'open_commission': 100
    }
    
    exit_price_y = 1100  # Y涨100
    exit_price_x = 2050  # X涨50
    mult_y = 10
    mult_x = 5
    
    # 手动计算
    print(f"Y腿：买入@{position['entry_price_y']}，卖出@{exit_price_y}")
    pnl_y = (exit_price_y - position['entry_price_y']) * position['lots_y'] * mult_y
    print(f"Y腿PnL = ({exit_price_y} - {position['entry_price_y']}) × {position['lots_y']} × {mult_y} = {pnl_y}")
    
    print(f"X腿：卖出@{position['entry_price_x']}，买入@{exit_price_x}")
    pnl_x = (position['entry_price_x'] - exit_price_x) * position['lots_x'] * mult_x
    print(f"X腿PnL = ({position['entry_price_x']} - {exit_price_x}) × {position['lots_x']} × {mult_x} = {pnl_x}")
    
    gross_pnl = pnl_y + pnl_x
    print(f"毛利润 = {pnl_y} + {pnl_x} = {gross_pnl}")
    
    # 调用函数计算
    result = calculate_pnl(position, exit_price_y, exit_price_x, mult_y, mult_x)
    print(f"函数计算结果：毛利润 = {result['gross_pnl']}")
    assert result['gross_pnl'] == gross_pnl, "毛利润计算错误！"
    print("✓ 毛利润计算正确")
    
    # 场景2：做空价差（卖Y买X）
    print("\n场景2：做空价差亏损")
    position = {
        'direction': 'short',
        'lots_y': 5,
        'lots_x': 4,
        'entry_price_y': 1000,
        'entry_price_x': 2000,
        'margin': 30000,
        'open_commission': 60
    }
    
    exit_price_y = 1050  # Y涨50（不利）
    exit_price_x = 1950  # X跌50（有利）
    
    # 手动计算
    print(f"Y腿：卖出@{position['entry_price_y']}，买入@{exit_price_y}")
    pnl_y = (position['entry_price_y'] - exit_price_y) * position['lots_y'] * mult_y
    print(f"Y腿PnL = ({position['entry_price_y']} - {exit_price_y}) × {position['lots_y']} × {mult_y} = {pnl_y}")
    
    print(f"X腿：买入@{position['entry_price_x']}，卖出@{exit_price_x}")
    pnl_x = (exit_price_x - position['entry_price_x']) * position['lots_x'] * mult_x
    print(f"X腿PnL = ({exit_price_x} - {position['entry_price_x']}) × {position['lots_x']} × {mult_x} = {pnl_x}")
    
    gross_pnl = pnl_y + pnl_x
    print(f"毛利润 = {pnl_y} + {pnl_x} = {gross_pnl}")
    
    # 调用函数计算
    result = calculate_pnl(position, exit_price_y, exit_price_x, mult_y, mult_x)
    print(f"函数计算结果：毛利润 = {result['gross_pnl']}")
    assert result['gross_pnl'] == gross_pnl, "毛利润计算错误！"
    print("✓ 毛利润计算正确")


def verify_time_weighted_return():
    """验证时间加权收益率计算"""
    print("\n" + "=" * 60)
    print("验证时间加权收益率")
    print("=" * 60)
    
    trades = [
        {'net_pnl': 10000, 'margin': 100000, 'holding_days': 5},
        {'net_pnl': -5000, 'margin': 50000, 'holding_days': 10},
        {'net_pnl': 8000, 'margin': 80000, 'holding_days': 3}
    ]
    
    print("\n交易数据：")
    for i, t in enumerate(trades, 1):
        print(f"交易{i}: PnL={t['net_pnl']}, 保证金={t['margin']}, 持仓天数={t['holding_days']}")
    
    # 手动计算
    total_pnl = sum(t['net_pnl'] for t in trades)
    print(f"\n总PnL = {trades[0]['net_pnl']} + ({trades[1]['net_pnl']}) + {trades[2]['net_pnl']} = {total_pnl}")
    
    margin_days = sum(t['margin'] * t['holding_days'] for t in trades)
    print(f"保证金·天 = {trades[0]['margin']}×{trades[0]['holding_days']} + {trades[1]['margin']}×{trades[1]['holding_days']} + {trades[2]['margin']}×{trades[2]['holding_days']}")
    print(f"         = {trades[0]['margin'] * trades[0]['holding_days']} + {trades[1]['margin'] * trades[1]['holding_days']} + {trades[2]['margin'] * trades[2]['holding_days']}")
    print(f"         = {margin_days}")
    
    tw_return = (total_pnl / margin_days) * 100
    print(f"时间加权收益率 = ({total_pnl} / {margin_days}) × 100 = {tw_return:.4f}%")
    
    annual_return = tw_return * 252
    print(f"年化收益率 = {tw_return:.4f}% × 252 = {annual_return:.2f}%")
    
    # 调用函数计算
    result = calculate_time_weighted_return(trades)
    print(f"\n函数计算结果：")
    print(f"时间加权收益率 = {result['tw_return']:.4f}%")
    print(f"年化收益率 = {result['annual_return']:.2f}%")
    
    assert abs(result['tw_return'] - tw_return) < 0.0001, "时间加权收益率计算错误！"
    print("✓ 时间加权收益率计算正确")


def verify_lots_calculation():
    """验证手数计算逻辑"""
    print("\n" + "=" * 60)
    print("验证手数计算")
    print("=" * 60)
    
    beta = 0.85
    price_y = 1000
    price_x = 2000
    mult_y = 10
    mult_x = 5
    margin_rate = 0.12
    available_capital = 100000
    
    print(f"\n输入参数：")
    print(f"β = {beta}")
    print(f"Y价格 = {price_y}, X价格 = {price_x}")
    print(f"Y乘数 = {mult_y}, X乘数 = {mult_x}")
    print(f"保证金率 = {margin_rate}")
    print(f"可用资金 = {available_capital}")
    
    # 手动计算
    margin_per_y = price_y * mult_y * margin_rate
    margin_per_x = price_x * mult_x * margin_rate
    print(f"\n单手保证金：")
    print(f"Y单手保证金 = {price_y} × {mult_y} × {margin_rate} = {margin_per_y}")
    print(f"X单手保证金 = {price_x} × {mult_x} × {margin_rate} = {margin_per_x}")
    
    # 搜索最优手数
    print(f"\n搜索最优手数（Y:X = 1:{beta}）：")
    for lots_y in range(1, 10):
        lots_x = round(lots_y * beta)
        total_margin = lots_y * margin_per_y + lots_x * margin_per_x
        
        if total_margin > available_capital:
            print(f"Y={lots_y}手, X={lots_x}手: 保证金={total_margin:.0f} > {available_capital} (超出)")
            break
        else:
            print(f"Y={lots_y}手, X={lots_x}手: 保证金={total_margin:.0f} ✓")
            max_lots_y = lots_y
            max_lots_x = lots_x
            max_margin = total_margin
    
    print(f"\n最优解：Y={max_lots_y}手, X={max_lots_x}手, 占用保证金={max_margin:.0f}")
    
    # 调用函数计算
    result = calculate_lots(beta, available_capital, price_y, price_x, mult_y, mult_x, margin_rate)
    print(f"\n函数计算结果：")
    print(f"Y={result['lots_y']}手, X={result['lots_x']}手, 占用保证金={result['margin']:.0f}")
    
    assert result['lots_y'] == max_lots_y, "Y手数计算错误！"
    assert result['lots_x'] == max_lots_x, "X手数计算错误！"
    print("✓ 手数计算正确")


def verify_max_drawdown():
    """验证最大回撤计算"""
    print("\n" + "=" * 60)
    print("验证最大回撤计算")
    print("=" * 60)
    
    cumulative_returns = [0, 0.05, 0.1, 0.08, 0.03, 0.12, 0.09, 0.15]
    
    print("\n累计收益率序列：")
    for i, ret in enumerate(cumulative_returns):
        print(f"第{i}期: {ret:.2%}")
    
    print("\n手动计算最大回撤：")
    print("最高点在第2期：0.1 (10%)")
    print("最低点在第4期：0.03 (3%)")
    print("回撤计算公式：(当前值 - 最高值) / (1 + 最高值)")
    drawdown = (0.03 - 0.1) / (1 + 0.1)
    print(f"最大回撤 = (0.03 - 0.1) / (1 + 0.1) = -0.07 / 1.1 = {drawdown:.4f}")
    
    # 调用函数计算
    max_dd, dd_duration = calculate_max_drawdown(cumulative_returns)
    print(f"\n函数计算结果：")
    print(f"最大回撤 = {max_dd:.4f}")
    print(f"回撤持续期 = {dd_duration}")
    
    assert abs(max_dd - drawdown) < 0.0001, "最大回撤计算错误！"
    print("✓ 最大回撤计算正确")


if __name__ == '__main__':
    verify_pnl_calculation()
    verify_time_weighted_return()
    verify_lots_calculation()
    verify_max_drawdown()
    
    print("\n" + "=" * 60)
    print("所有计算验证通过！✓")
    print("=" * 60)