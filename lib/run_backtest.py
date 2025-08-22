#!/usr/bin/env python
"""
完整回测脚本 - 使用原子服务
根据需求文档04_backtest_framework.md实现
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fractions import Fraction
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_from_parquet

# 回测参数
INITIAL_CAPITAL = 5000000  # 初始资金500万
MARGIN_RATE = 0.12  # 保证金率12%
COMMISSION_RATE = 0.0002  # 手续费率（单边）
SLIPPAGE_TICKS = 3  # 滑点tick数
STOP_LOSS_RATE = 0.15  # 止损线：保证金的15%
MAX_HOLDING_DAYS = 30  # 最大持仓天数

# 合约规格（简化版，实际应从数据源获取）
CONTRACT_SPECS = {
    'AG0': {'multiplier': 15, 'tick_size': 1},
    'AU0': {'multiplier': 1000, 'tick_size': 0.02},
    'AL0': {'multiplier': 5, 'tick_size': 5},
    'CU0': {'multiplier': 5, 'tick_size': 10},
    'NI0': {'multiplier': 1, 'tick_size': 10},
    'PB0': {'multiplier': 5, 'tick_size': 5},
    'SN0': {'multiplier': 1, 'tick_size': 10},
    'ZN0': {'multiplier': 5, 'tick_size': 5},
    'HC0': {'multiplier': 10, 'tick_size': 1},
    'I0': {'multiplier': 100, 'tick_size': 0.5},
    'RB0': {'multiplier': 10, 'tick_size': 1},
    'SF0': {'multiplier': 5, 'tick_size': 2},
    'SM0': {'multiplier': 5, 'tick_size': 2},
    'SS0': {'multiplier': 5, 'tick_size': 5},
}

def calculate_min_lots(beta, max_total_lots=10):
    """根据β值计算最小整数比手数
    
    控制总手数不超过max_total_lots（默认10手）
    对于负Beta：
    - 表示Y和X负相关
    - 仍然使用绝对值计算手数比例
    - 在交易方向上体现负相关（同向开仓而非对冲）
    """
    if abs(beta) < 0.01:
        return {'lots_y': 1, 'lots_x': 1, 'actual_ratio': 1.0, 'is_negative': False}
    
    is_negative = beta < 0
    beta_abs = abs(beta)
    
    # 先尝试找最简分数
    frac = Fraction(beta_abs).limit_denominator(max_total_lots)
    
    lots_y = frac.numerator
    lots_x = frac.denominator
    
    # 如果总手数超过限制，按比例缩小
    total_lots = lots_y + lots_x
    if total_lots > max_total_lots:
        # 使用不同的分母限制重试
        for max_denom in range(max_total_lots-1, 0, -1):
            frac = Fraction(beta_abs).limit_denominator(max_denom)
            temp_y = frac.numerator
            temp_x = frac.denominator
            if temp_y + temp_x <= max_total_lots:
                lots_y = temp_y
                lots_x = temp_x
                break
        else:
            # 如果还是找不到，使用简单近似
            if beta_abs > 1:
                # beta > 1, Y的手数多
                lots_x = 1
                lots_y = min(int(beta_abs + 0.5), max_total_lots - 1)
            else:
                # beta < 1, X的手数多
                lots_y = 1
                lots_x = min(int(1/beta_abs + 0.5), max_total_lots - 1)
    
    # 确保至少1手
    if lots_y == 0:
        lots_y = 1
    if lots_x == 0:
        lots_x = 1
    
    # 最终检查总手数
    total_lots = lots_y + lots_x
    if total_lots > max_total_lots:
        # 强制限制在10手以内
        if lots_y > lots_x:
            lots_y = max_total_lots - 1
            lots_x = 1
        else:
            lots_x = max_total_lots - 1
            lots_y = 1
    
    return {
        'lots_y': lots_y,
        'lots_x': lots_x,
        'actual_ratio': lots_y / lots_x,
        'theoretical_ratio': beta,  # 保留原始beta（含符号）
        'is_negative': is_negative,
        'total_lots': lots_y + lots_x
    }

def calculate_pnl(position, exit_prices):
    """计算单笔交易PnL"""
    
    # 获取合约规格
    spec_x = CONTRACT_SPECS.get(position['symbol_x'], {'multiplier': 10, 'tick_size': 1})
    spec_y = CONTRACT_SPECS.get(position['symbol_y'], {'multiplier': 10, 'tick_size': 1})
    
    # 计算价格变动
    price_change_x = exit_prices['x'] - position['entry_price_x']
    price_change_y = exit_prices['y'] - position['entry_price_y']
    
    # 根据方向和Beta符号计算PnL
    # 对于负Beta，X和Y是负相关，交易时需要同向操作
    if position.get('is_negative', False):  # 负Beta
        if position['direction'] == 'long':  # 做多价差
            # 负Beta时：做多Y，同时做多X（而非做空）
            pnl_y = price_change_y * position['lots_y'] * spec_y['multiplier']
            pnl_x = price_change_x * position['lots_x'] * spec_x['multiplier']
        else:  # 做空价差
            # 负Beta时：做空Y，同时做空X（而非做多）
            pnl_y = -price_change_y * position['lots_y'] * spec_y['multiplier']
            pnl_x = -price_change_x * position['lots_x'] * spec_x['multiplier']
    else:  # 正Beta（正常对冲）
        if position['direction'] == 'long':  # 做多价差
            # 做多Y，做空X
            pnl_y = price_change_y * position['lots_y'] * spec_y['multiplier']
            pnl_x = -price_change_x * position['lots_x'] * spec_x['multiplier']
        else:  # 做空价差
            # 做空Y，做多X
            pnl_y = -price_change_y * position['lots_y'] * spec_y['multiplier']
            pnl_x = price_change_x * position['lots_x'] * spec_x['multiplier']
    
    gross_pnl = pnl_x + pnl_y
    
    # 计算手续费（双边）
    nominal_x_entry = position['entry_price_x'] * position['lots_x'] * spec_x['multiplier']
    nominal_y_entry = position['entry_price_y'] * position['lots_y'] * spec_y['multiplier']
    nominal_x_exit = exit_prices['x'] * position['lots_x'] * spec_x['multiplier']
    nominal_y_exit = exit_prices['y'] * position['lots_y'] * spec_y['multiplier']
    
    commission = (nominal_x_entry + nominal_y_entry + nominal_x_exit + nominal_y_exit) * COMMISSION_RATE
    
    # 净PnL
    net_pnl = gross_pnl - commission
    
    # 收益率（基于保证金）
    return_on_margin = net_pnl / position['margin'] * 100 if position['margin'] > 0 else 0
    
    return {
        'gross_pnl': gross_pnl,
        'commission': commission,
        'net_pnl': net_pnl,
        'return_on_margin': return_on_margin,
        'pnl_x': pnl_x,
        'pnl_y': pnl_y
    }

def run_backtest(trades_file='/mnt/e/Star-arb/data/kalman_trades.csv'):
    """运行回测"""
    
    print("="*80)
    print("期货配对交易回测")
    print("="*80)
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"保证金率: {MARGIN_RATE:.1%}")
    print(f"手续费率: {COMMISSION_RATE:.2%} (单边)")
    print(f"滑点: {SLIPPAGE_TICKS} ticks")
    print(f"止损线: {STOP_LOSS_RATE:.1%} of margin")
    print()
    
    # 加载交易信号
    trades = pd.read_csv(trades_file)
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['exit_date'] = pd.to_datetime(trades['exit_date'])
    
    # 加载价格数据
    print("加载价格数据...")
    price_data = {}
    symbols = set()
    
    for pair in trades['pair'].unique():
        sym1, sym2 = pair.split('-')
        symbols.add(sym1)
        symbols.add(sym2)
    
    for symbol in symbols:
        try:
            df = load_from_parquet(symbol)
            if df is not None:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                price_data[symbol] = df
                print(f"  {symbol}: 已加载")
        except:
            print(f"  {symbol}: 加载失败")
    
    print()
    
    # 回测每笔交易
    print("执行回测...")
    backtest_results = []
    
    for idx, trade in trades.iterrows():
        pair = trade['pair']
        sym1, sym2 = pair.split('-')
        
        if sym1 not in price_data or sym2 not in price_data:
            continue
        
        # 获取入场和出场价格
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        
        try:
            entry_price_x = price_data[sym1].loc[entry_date, 'close']
            entry_price_y = price_data[sym2].loc[entry_date, 'close']
            exit_price_x = price_data[sym1].loc[exit_date, 'close']
            exit_price_y = price_data[sym2].loc[exit_date, 'close']
        except:
            continue
        
        # 计算手数（基于beta）
        lots_info = calculate_min_lots(trade['entry_beta'])
        
        # 加滑点
        spec_x = CONTRACT_SPECS.get(sym1, {'tick_size': 1})
        spec_y = CONTRACT_SPECS.get(sym2, {'tick_size': 1})
        
        if trade['entry_action'] == 'open_long':
            # 做多价差：买Y卖X
            entry_price_y += SLIPPAGE_TICKS * spec_y['tick_size']  # 买入加滑点
            entry_price_x -= SLIPPAGE_TICKS * spec_x['tick_size']  # 卖出减滑点
            exit_price_y -= SLIPPAGE_TICKS * spec_y['tick_size']   # 卖出减滑点
            exit_price_x += SLIPPAGE_TICKS * spec_x['tick_size']   # 买入加滑点
            direction = 'long'
        else:
            # 做空价差：卖Y买X
            entry_price_y -= SLIPPAGE_TICKS * spec_y['tick_size']  # 卖出减滑点
            entry_price_x += SLIPPAGE_TICKS * spec_x['tick_size']  # 买入加滑点
            exit_price_y += SLIPPAGE_TICKS * spec_y['tick_size']   # 买入加滑点
            exit_price_x -= SLIPPAGE_TICKS * spec_x['tick_size']   # 卖出减滑点
            direction = 'short'
        
        # 计算保证金
        spec_x_mult = CONTRACT_SPECS.get(sym1, {'multiplier': 10})['multiplier']
        spec_y_mult = CONTRACT_SPECS.get(sym2, {'multiplier': 10})['multiplier']
        
        margin_x = entry_price_x * lots_info['lots_x'] * spec_x_mult * MARGIN_RATE
        margin_y = entry_price_y * lots_info['lots_y'] * spec_y_mult * MARGIN_RATE
        total_margin = margin_x + margin_y
        
        # 创建持仓记录
        position = {
            'symbol_x': sym1,
            'symbol_y': sym2,
            'direction': direction,
            'entry_price_x': entry_price_x,
            'entry_price_y': entry_price_y,
            'lots_x': lots_info['lots_x'],
            'lots_y': lots_info['lots_y'],
            'margin': total_margin,
            'beta': trade['entry_beta'],
            'is_negative': lots_info.get('is_negative', False)
        }
        
        # 计算PnL
        exit_prices = {'x': exit_price_x, 'y': exit_price_y}
        pnl_result = calculate_pnl(position, exit_prices)
        
        # 记录结果
        result = {
            'trade_id': idx + 1,
            'pair': pair,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'holding_days': trade['holding_days'],
            'signal_type': trade['entry_action'],
            'exit_reason': trade['exit_action'],
            'beta': trade['entry_beta'],
            'entry_z_score': trade['entry_z_score'],
            'exit_z_score': trade['exit_z_score'],
            'lots_x': lots_info['lots_x'],
            'lots_y': lots_info['lots_y'],
            'entry_price_x': entry_price_x,
            'entry_price_y': entry_price_y,
            'exit_price_x': exit_price_x,
            'exit_price_y': exit_price_y,
            'margin': total_margin,
            'gross_pnl': pnl_result['gross_pnl'],
            'commission': pnl_result['commission'],
            'net_pnl': pnl_result['net_pnl'],
            'return_pct': pnl_result['return_on_margin']
        }
        
        backtest_results.append(result)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(backtest_results)
    
    if len(df_results) == 0:
        print("没有可回测的交易")
        return None
    
    # 生成完整报告
    print("\n" + "="*80)
    print("回测报告")
    print("="*80)
    
    generate_full_report(df_results)
    
    return df_results

def generate_full_report(df_results):
    """生成完整的回测报告（按需求文档要求）"""
    
    # ========== 1. 总体绩效统计 ==========
    print("\n【1. 总体绩效统计】")
    print("-"*50)
    
    total_trades = len(df_results)
    total_pnl = df_results['net_pnl'].sum()
    avg_pnl = df_results['net_pnl'].mean()
    std_pnl = df_results['net_pnl'].std()
    
    winning_trades = len(df_results[df_results['net_pnl'] > 0])
    losing_trades = len(df_results[df_results['net_pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    print(f"总交易数: {total_trades}")
    print(f"总净盈亏: {total_pnl:,.2f}")
    print(f"平均每笔盈亏: {avg_pnl:,.2f}")
    print(f"盈亏标准差: {std_pnl:,.2f}")
    print(f"盈利交易: {winning_trades} ({win_rate:.1f}%)")
    print(f"亏损交易: {losing_trades} ({100-win_rate:.1f}%)")
    
    # 盈亏比
    if winning_trades > 0 and losing_trades > 0:
        avg_win = df_results[df_results['net_pnl'] > 0]['net_pnl'].mean()
        avg_loss = abs(df_results[df_results['net_pnl'] < 0]['net_pnl'].mean())
        profit_factor = avg_win / avg_loss
        print(f"平均盈利: {avg_win:,.2f}")
        print(f"平均亏损: {avg_loss:,.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
    
    # ========== 2. 收益率分析 ==========
    print("\n【2. 收益率分析】")
    print("-"*50)
    
    total_return = total_pnl / INITIAL_CAPITAL * 100
    total_margin_used = df_results['margin'].sum()
    return_on_margin = total_pnl / total_margin_used * 100 if total_margin_used > 0 else 0
    
    # 计算时间跨度
    date_range = (df_results['exit_date'].max() - df_results['entry_date'].min()).days
    years = date_range / 365
    annual_return = (1 + total_return/100) ** (1/years) - 1 if years > 0 else 0
    
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益率: {annual_return*100:.2f}%")
    print(f"保证金收益率: {return_on_margin:.2f}%")
    print(f"总保证金占用: {total_margin_used:,.0f}")
    
    # ========== 3. 风险指标 ==========
    print("\n【3. 风险指标】")
    print("-"*50)
    
    # 计算累计PnL用于最大回撤
    df_results_sorted = df_results.sort_values('exit_date')
    df_results_sorted['cum_pnl'] = df_results_sorted['net_pnl'].cumsum()
    df_results_sorted['cum_max'] = df_results_sorted['cum_pnl'].cummax()
    df_results_sorted['drawdown'] = df_results_sorted['cum_pnl'] - df_results_sorted['cum_max']
    
    max_drawdown = df_results_sorted['drawdown'].min()
    max_drawdown_pct = max_drawdown / INITIAL_CAPITAL * 100
    
    # 夏普比率（简化计算）
    if len(df_results) > 1 and std_pnl > 0:
        sharpe_ratio = avg_pnl / std_pnl * np.sqrt(252 / (date_range/total_trades))
    else:
        sharpe_ratio = 0
    
    print(f"最大回撤: {max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"最大单笔盈利: {df_results['net_pnl'].max():,.2f}")
    print(f"最大单笔亏损: {df_results['net_pnl'].min():,.2f}")
    
    # ========== 4. 持仓分析 ==========
    print("\n【4. 持仓分析】")
    print("-"*50)
    
    avg_holding = df_results['holding_days'].mean()
    max_holding = df_results['holding_days'].max()
    min_holding = df_results['holding_days'].min()
    
    print(f"平均持仓天数: {avg_holding:.1f}")
    print(f"最长持仓: {max_holding} 天")
    print(f"最短持仓: {min_holding} 天")
    
    # 退出原因统计
    exit_stats = df_results['exit_reason'].value_counts()
    print("\n退出原因分布:")
    for reason, count in exit_stats.items():
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
    
    # ========== 5. 配对分析（TOP 10） ==========
    print("\n【5. 配对收益排名 (TOP 10)】")
    print("-"*50)
    
    pair_stats = df_results.groupby('pair').agg({
        'net_pnl': ['sum', 'mean', 'std', 'count'],
        'holding_days': 'mean'
    }).round(2)
    
    pair_stats.columns = ['总PnL', '平均PnL', 'PnL标准差', '交易次数', '平均持仓天数']
    pair_stats = pair_stats.sort_values('总PnL', ascending=False)
    
    print(f"\n{'配对':<12} {'总PnL':>12} {'平均PnL':>10} {'交易次数':>8} {'平均持仓':>10}")
    print("-"*60)
    
    for pair, row in pair_stats.head(10).iterrows():
        print(f"{pair:<12} {row['总PnL']:>12,.0f} {row['平均PnL']:>10,.0f} "
              f"{row['交易次数']:>8.0f} {row['平均持仓天数']:>10.1f}")
    
    # ========== 6. Beta分布分析 ==========
    print("\n【6. Beta系数分析】")
    print("-"*50)
    
    print(f"Beta均值: {df_results['beta'].mean():.4f}")
    print(f"Beta标准差: {df_results['beta'].std():.4f}")
    print(f"Beta最小值: {df_results['beta'].min():.4f}")
    print(f"Beta最大值: {df_results['beta'].max():.4f}")
    
    # Beta分组统计
    beta_bins = pd.cut(df_results['beta'].abs(), bins=[0, 0.5, 1, 1.5, 2, 3], 
                       labels=['0.3-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3'])
    beta_group = df_results.groupby(beta_bins)['net_pnl'].agg(['sum', 'count', 'mean'])
    
    print("\nBeta分组收益:")
    print(f"{'Beta范围':<10} {'总PnL':>12} {'交易数':>8} {'平均PnL':>10}")
    for idx, row in beta_group.iterrows():
        if row['count'] > 0:
            print(f"{idx:<10} {row['sum']:>12,.0f} {row['count']:>8.0f} {row['mean']:>10,.0f}")
    
    # ========== 7. 月度收益分析 ==========
    print("\n【7. 月度收益分析】")
    print("-"*50)
    
    df_results['exit_month'] = df_results['exit_date'].dt.to_period('M')
    monthly_pnl = df_results.groupby('exit_month')['net_pnl'].sum()
    
    print(f"{'月份':<10} {'净盈亏':>12} {'累计盈亏':>12}")
    print("-"*35)
    
    cum_pnl = 0
    for month, pnl in monthly_pnl.head(12).items():
        cum_pnl += pnl
        print(f"{str(month):<10} {pnl:>12,.0f} {cum_pnl:>12,.0f}")
    
    # ========== 8. 交易成本分析 ==========
    print("\n【8. 交易成本分析】")
    print("-"*50)
    
    total_commission = df_results['commission'].sum()
    total_gross_pnl = df_results['gross_pnl'].sum()
    cost_ratio = total_commission / abs(total_gross_pnl) * 100 if total_gross_pnl != 0 else 0
    
    print(f"总手续费: {total_commission:,.2f}")
    print(f"平均每笔手续费: {total_commission/total_trades:,.2f}")
    print(f"手续费占毛盈亏比例: {cost_ratio:.2f}%")
    print(f"毛盈亏: {total_gross_pnl:,.2f}")
    print(f"净盈亏: {total_pnl:,.2f}")
    
    # ========== 9. 保存详细结果 ==========
    output_file = '/mnt/e/Star-arb/data/backtest_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n详细回测结果已保存至: {output_file}")
    
    # 保存统计报告
    report = {
        '总体绩效': {
            '总交易数': total_trades,
            '总净盈亏': round(total_pnl, 2),
            '胜率': round(win_rate, 2),
            '盈亏比': round(profit_factor, 2) if 'profit_factor' in locals() else 0
        },
        '收益率': {
            '总收益率': round(total_return, 2),
            '年化收益率': round(annual_return*100, 2),
            '保证金收益率': round(return_on_margin, 2)
        },
        '风险指标': {
            '最大回撤': round(max_drawdown, 2),
            '夏普比率': round(sharpe_ratio, 2)
        },
        '持仓统计': {
            '平均持仓天数': round(avg_holding, 1),
            '正常平仓比例': round(exit_stats.get('close_normal', 0)/total_trades*100, 1) if 'exit_stats' in locals() else 0,
            '超时平仓比例': round(exit_stats.get('close_timeout', 0)/total_trades*100, 1) if 'exit_stats' in locals() else 0
        }
    }
    
    report_file = '/mnt/e/Star-arb/data/backtest_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"统计报告已保存至: {report_file}")

if __name__ == "__main__":
    results = run_backtest()