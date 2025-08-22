#!/usr/bin/env python
"""
绘制按天的PnL曲线和收益曲线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_pnl_curves():
    """绘制PnL曲线"""
    
    print("生成收益曲线...")
    
    # 加载数据
    df_backtest = pd.read_csv('/mnt/e/Star-arb/data/backtest_results.csv')
    df_stoploss = pd.read_csv('/mnt/e/Star-arb/data/backtest_stoploss_v2.csv')
    
    # 转换日期
    df_backtest['exit_date'] = pd.to_datetime(df_backtest['exit_date'])
    df_stoploss['exit_date'] = pd.to_datetime(df_stoploss['exit_date'])
    
    # 按退出日期排序
    df_backtest = df_backtest.sort_values('exit_date')
    df_stoploss = df_stoploss.sort_values('exit_date')
    
    # 计算累计PnL
    df_backtest['cum_pnl'] = df_backtest['net_pnl'].cumsum()
    df_stoploss['cum_pnl_stoploss'] = df_stoploss['stoploss_pnl'].cumsum()
    
    # 创建每日数据（填充非交易日）
    date_range = pd.date_range(
        start=df_backtest['exit_date'].min(),
        end=df_backtest['exit_date'].max(),
        freq='D'
    )
    
    # 原始策略每日PnL
    daily_original = pd.DataFrame(index=date_range)
    daily_original['pnl'] = 0
    daily_original['cum_pnl'] = 0
    
    for date in df_backtest['exit_date'].unique():
        day_pnl = df_backtest[df_backtest['exit_date'] == date]['net_pnl'].sum()
        daily_original.loc[date, 'pnl'] = day_pnl
    
    daily_original['cum_pnl'] = daily_original['pnl'].cumsum()
    
    # 止损策略每日PnL
    daily_stoploss = pd.DataFrame(index=date_range)
    daily_stoploss['pnl'] = 0
    daily_stoploss['cum_pnl'] = 0
    
    for date in df_stoploss['exit_date'].unique():
        day_pnl = df_stoploss[df_stoploss['exit_date'] == date]['stoploss_pnl'].sum()
        daily_stoploss.loc[date, 'pnl'] = day_pnl
    
    daily_stoploss['cum_pnl'] = daily_stoploss['pnl'].cumsum()
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 累计PnL曲线
    ax1 = axes[0]
    ax1.plot(daily_original.index, daily_original['cum_pnl']/10000, 
             label='原始策略', linewidth=2, color='blue')
    ax1.plot(daily_stoploss.index, daily_stoploss['cum_pnl']/10000, 
             label='15%止损策略', linewidth=2, color='green')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(daily_original.index, 0, daily_original['cum_pnl']/10000, 
                      where=(daily_original['cum_pnl'] > 0), alpha=0.2, color='blue')
    ax1.fill_between(daily_stoploss.index, 0, daily_stoploss['cum_pnl']/10000, 
                      where=(daily_stoploss['cum_pnl'] > 0), alpha=0.2, color='green')
    ax1.set_title('累计PnL曲线（万元）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('累计盈亏（万元）', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 标注关键点
    max_pnl_orig = daily_original['cum_pnl'].max()
    max_pnl_stop = daily_stoploss['cum_pnl'].max()
    ax1.annotate(f'最终: {max_pnl_orig/10000:.1f}万', 
                 xy=(daily_original.index[-1], max_pnl_orig/10000),
                 xytext=(-60, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3))
    ax1.annotate(f'最终: {max_pnl_stop/10000:.1f}万', 
                 xy=(daily_stoploss.index[-1], max_pnl_stop/10000),
                 xytext=(-60, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.3))
    
    # 2. 每日PnL柱状图
    ax2 = axes[1]
    positive_days = daily_stoploss['pnl'] > 0
    negative_days = daily_stoploss['pnl'] < 0
    ax2.bar(daily_stoploss.index[positive_days], daily_stoploss['pnl'][positive_days]/10000, 
            color='green', alpha=0.6, label='盈利日')
    ax2.bar(daily_stoploss.index[negative_days], daily_stoploss['pnl'][negative_days]/10000, 
            color='red', alpha=0.6, label='亏损日')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_title('每日PnL（止损策略）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('日盈亏（万元）', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[2]
    
    # 计算回撤
    running_max_orig = daily_original['cum_pnl'].cummax()
    drawdown_orig = (daily_original['cum_pnl'] - running_max_orig) / 10000
    
    running_max_stop = daily_stoploss['cum_pnl'].cummax()
    drawdown_stop = (daily_stoploss['cum_pnl'] - running_max_stop) / 10000
    
    ax3.fill_between(daily_original.index, 0, drawdown_orig, 
                      color='red', alpha=0.3, label='原始策略回撤')
    ax3.fill_between(daily_stoploss.index, 0, drawdown_stop, 
                      color='orange', alpha=0.5, label='止损策略回撤')
    ax3.set_title('回撤曲线（万元）', fontsize=14, fontweight='bold')
    ax3.set_ylabel('回撤（万元）', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='lower left', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 标注最大回撤
    max_dd_orig = drawdown_orig.min()
    max_dd_stop = drawdown_stop.min()
    max_dd_date_orig = drawdown_orig.idxmin()
    max_dd_date_stop = drawdown_stop.idxmin()
    
    ax3.annotate(f'最大回撤: {max_dd_orig:.1f}万', 
                 xy=(max_dd_date_orig, max_dd_orig),
                 xytext=(30, -10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3))
    ax3.annotate(f'最大回撤: {max_dd_stop:.1f}万', 
                 xy=(max_dd_date_stop, max_dd_stop),
                 xytext=(30, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='orange'),
                 bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.3))
    
    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = '/mnt/e/Star-arb/data/pnl_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"收益曲线已保存至: {output_file}")
    
    # 显示统计信息
    print("\n收益曲线统计:")
    print("="*60)
    print(f"原始策略:")
    print(f"  最终累计PnL: {daily_original['cum_pnl'].iloc[-1]:,.2f} 元")
    print(f"  最大回撤: {(drawdown_orig.min()*10000):,.2f} 元")
    print(f"  盈利天数: {(daily_original['pnl'] > 0).sum()} 天")
    print(f"  亏损天数: {(daily_original['pnl'] < 0).sum()} 天")
    
    print(f"\n止损策略:")
    print(f"  最终累计PnL: {daily_stoploss['cum_pnl'].iloc[-1]:,.2f} 元")
    print(f"  最大回撤: {(drawdown_stop.min()*10000):,.2f} 元")
    print(f"  盈利天数: {(daily_stoploss['pnl'] > 0).sum()} 天")
    print(f"  亏损天数: {(daily_stoploss['pnl'] < 0).sum()} 天")
    
    print(f"\n改善效果:")
    pnl_improvement = daily_stoploss['cum_pnl'].iloc[-1] - daily_original['cum_pnl'].iloc[-1]
    dd_improvement = (drawdown_orig.min() - drawdown_stop.min()) * 10000
    print(f"  PnL改善: {pnl_improvement:,.2f} 元 (+{pnl_improvement/daily_original['cum_pnl'].iloc[-1]*100:.1f}%)")
    print(f"  回撤改善: {dd_improvement:,.2f} 元 ({dd_improvement/(drawdown_orig.min()*10000)*100:.1f}%)")
    
    plt.show()
    
    return daily_original, daily_stoploss

if __name__ == "__main__":
    daily_orig, daily_stop = plot_pnl_curves()