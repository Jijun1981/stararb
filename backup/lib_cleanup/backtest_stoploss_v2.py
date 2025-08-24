#!/usr/bin/env python
"""
带止损的回测引擎 V2
- 15%单笔保证金止损
- 简化计算逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def run_backtest_with_stoploss():
    """运行带止损的回测"""
    
    print("="*80)
    print("带15%止损的回测分析 V2")
    print("="*80)
    
    # 加载原始回测结果
    df_original = pd.read_csv('/mnt/e/Star-arb/data/backtest_results.csv')
    print(f"加载 {len(df_original)} 笔交易")
    
    # 复制数据
    df_results = df_original.copy()
    
    # 添加止损标记列
    df_results['stop_triggered'] = False
    df_results['stoploss_pnl'] = df_results['net_pnl']
    
    # 处理每笔交易
    stop_count = 0
    saved_loss = 0
    
    for idx, trade in df_results.iterrows():
        # 计算止损线（亏损15%保证金）
        stop_loss_amount = -trade['margin'] * 0.15
        
        # 如果原始PnL低于止损线，触发止损
        if trade['net_pnl'] < stop_loss_amount:
            df_results.at[idx, 'stop_triggered'] = True
            df_results.at[idx, 'stoploss_pnl'] = stop_loss_amount
            saved_loss += trade['net_pnl'] - stop_loss_amount  # 止损避免的额外亏损
            stop_count += 1
    
    # 统计分析
    print("\n止损统计")
    print("="*60)
    
    total_trades = len(df_results)
    stop_rate = stop_count / total_trades * 100
    
    print(f"总交易数: {total_trades}")
    print(f"触发止损数: {stop_count}")
    print(f"止损触发率: {stop_rate:.1f}%")
    
    # 收益对比
    original_pnl = df_results['net_pnl'].sum()
    stoploss_pnl = df_results['stoploss_pnl'].sum()
    pnl_improvement = stoploss_pnl - original_pnl
    
    print(f"\n收益对比:")
    print(f"  原始总PnL: {original_pnl:,.2f} 元")
    print(f"  止损后总PnL: {stoploss_pnl:,.2f} 元")
    print(f"  止损保护金额: {pnl_improvement:,.2f} 元")
    print(f"  改善比例: {pnl_improvement/abs(original_pnl)*100:.1f}%")
    
    # 止损交易详情
    stopped_trades = df_results[df_results['stop_triggered']]
    if len(stopped_trades) > 0:
        print(f"\n止损交易分析:")
        print(f"  止损交易数: {len(stopped_trades)}")
        print(f"  原始平均亏损: {stopped_trades['net_pnl'].mean():,.2f} 元")
        print(f"  止损后平均亏损: {stopped_trades['stoploss_pnl'].mean():,.2f} 元")
        print(f"  每笔平均减少亏损: {(stopped_trades['stoploss_pnl'] - stopped_trades['net_pnl']).mean():,.2f} 元")
        print(f"  总共减少亏损: {saved_loss:,.2f} 元")
    
    # 新的绩效指标
    print(f"\n绩效指标（止损后）:")
    
    # 胜率
    win_rate = (df_results['stoploss_pnl'] > 0).sum() / total_trades * 100
    print(f"  胜率: {win_rate:.1f}%")
    
    # 基于保证金的收益率
    df_results['stoploss_return'] = df_results['stoploss_pnl'] / df_results['margin'] * 100
    avg_return = df_results['stoploss_return'].mean()
    print(f"  平均收益率: {avg_return:.2f}%")
    
    # 最大回撤
    df_sorted = df_results.sort_values('exit_date')
    df_sorted['cum_pnl'] = df_sorted['stoploss_pnl'].cumsum()
    peak = df_sorted['cum_pnl'].expanding().max()
    drawdown = df_sorted['cum_pnl'] - peak
    max_drawdown = drawdown.min()
    
    print(f"  最大回撤: {max_drawdown:,.2f} 元")
    
    # 基于最大保证金占用的收益率
    max_margin = 1613984.40
    total_return = stoploss_pnl / max_margin * 100
    annual_return = total_return / 1.1
    
    print(f"  总收益率（基于最大占用）: {total_return:.2f}%")
    print(f"  年化收益率: {annual_return:.2f}%")
    
    # 夏普比率
    returns = df_results['stoploss_return']
    sharpe = returns.mean() / returns.std() * np.sqrt(252/26) if returns.std() > 0 else 0
    print(f"  年化夏普比率: {sharpe:.2f}")
    
    # 收益分布
    print(f"\n收益率分布（止损后）:")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(df_results['stoploss_return'], p)
        print(f"  {p}%分位数: {value:.2f}%")
    
    # 对比原始和止损后的关键指标
    print(f"\n关键指标对比:")
    print(f"{'指标':<20} {'原始':<15} {'止损后':<15} {'变化':<15}")
    print("-"*65)
    
    # 原始指标
    orig_win_rate = (df_results['net_pnl'] > 0).sum() / total_trades * 100
    orig_avg_return = (df_results['net_pnl'] / df_results['margin'] * 100).mean()
    orig_total_return = original_pnl / max_margin * 100
    
    print(f"{'总PnL':<20} {original_pnl:>12,.0f}元  {stoploss_pnl:>12,.0f}元  {pnl_improvement:>+12,.0f}元")
    print(f"{'胜率':<20} {orig_win_rate:>12.1f}%  {win_rate:>12.1f}%  {win_rate-orig_win_rate:>+12.1f}%")
    print(f"{'平均收益率':<20} {orig_avg_return:>12.2f}%  {avg_return:>12.2f}%  {avg_return-orig_avg_return:>+12.2f}%")
    print(f"{'总收益率':<20} {orig_total_return:>12.2f}%  {total_return:>12.2f}%  {total_return-orig_total_return:>+12.2f}%")
    
    # 保存结果
    output_file = '/mnt/e/Star-arb/data/backtest_stoploss_v2.csv'
    df_results[['trade_id', 'pair', 'entry_date', 'exit_date', 'margin', 
                'net_pnl', 'stoploss_pnl', 'stop_triggered', 'stoploss_return']].to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")
    
    return df_results

if __name__ == "__main__":
    df_results = run_backtest_with_stoploss()