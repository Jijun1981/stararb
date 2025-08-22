#!/usr/bin/env python
"""
绘制按时间的PnL曲线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_pnl_curves():
    """绘制PnL曲线"""
    
    print("生成PnL曲线...")
    
    # 加载最终报告
    df = pd.read_csv('/mnt/e/Star-arb/data/FINAL_REPORT_CORRECTED.csv')
    
    # 转换日期
    df['出场日期'] = pd.to_datetime(df['出场日期'])
    
    # 按出场日期排序
    df = df.sort_values('出场日期')
    
    # 创建每日PnL数据
    date_range = pd.date_range(
        start=df['出场日期'].min(),
        end=df['出场日期'].max(),
        freq='D'
    )
    
    # 原始策略每日PnL
    daily_original = pd.DataFrame(index=date_range)
    daily_original['pnl'] = 0
    daily_original['cum_pnl'] = 0
    
    for date in df['出场日期'].unique():
        day_trades = df[df['出场日期'] == date]
        day_pnl = day_trades['原始盈亏'].sum()
        daily_original.loc[date, 'pnl'] = day_pnl
    
    daily_original['cum_pnl'] = daily_original['pnl'].cumsum()
    
    # 止损策略每日PnL
    daily_stoploss = pd.DataFrame(index=date_range)
    daily_stoploss['pnl'] = 0
    daily_stoploss['cum_pnl'] = 0
    
    for date in df['出场日期'].unique():
        day_trades = df[df['出场日期'] == date]
        day_pnl = day_trades['止损后盈亏'].sum()
        daily_stoploss.loc[date, 'pnl'] = day_pnl
    
    daily_stoploss['cum_pnl'] = daily_stoploss['pnl'].cumsum()
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 累计PnL曲线
    ax1 = axes[0]
    ax1.plot(daily_original.index, daily_original['cum_pnl']/10000, 
             label='Original Strategy', linewidth=2, color='blue', alpha=0.7)
    ax1.plot(daily_stoploss.index, daily_stoploss['cum_pnl']/10000, 
             label='15% Stop-Loss Strategy', linewidth=2, color='green')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(daily_stoploss.index, 0, daily_stoploss['cum_pnl']/10000, 
                      where=(daily_stoploss['cum_pnl'] > 0), alpha=0.3, color='green')
    ax1.set_title('Cumulative PnL Curve (10k CNY)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative PnL (10k CNY)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 标注最终值
    final_orig = daily_original['cum_pnl'].iloc[-1]
    final_stop = daily_stoploss['cum_pnl'].iloc[-1]
    ax1.text(daily_original.index[-1], final_orig/10000, 
             f'{final_orig/10000:.1f}', 
             fontsize=10, ha='right', va='bottom', color='blue')
    ax1.text(daily_stoploss.index[-1], final_stop/10000, 
             f'{final_stop/10000:.1f}', 
             fontsize=10, ha='right', va='top', color='green')
    
    # 2. 每日PnL柱状图
    ax2 = axes[1]
    positive_days = daily_stoploss['pnl'] > 0
    negative_days = daily_stoploss['pnl'] < 0
    ax2.bar(daily_stoploss.index[positive_days], daily_stoploss['pnl'][positive_days]/10000, 
            color='green', alpha=0.6, label='Profit Days', width=1)
    ax2.bar(daily_stoploss.index[negative_days], daily_stoploss['pnl'][negative_days]/10000, 
            color='red', alpha=0.6, label='Loss Days', width=1)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_title('Daily PnL (Stop-Loss Strategy)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Daily PnL (10k CNY)', fontsize=12)
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
                      color='red', alpha=0.3, label='Original Drawdown')
    ax3.fill_between(daily_stoploss.index, 0, drawdown_stop, 
                      color='orange', alpha=0.5, label='Stop-Loss Drawdown')
    ax3.set_title('Drawdown Curve (10k CNY)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (10k CNY)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend(loc='lower left', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 标注最大回撤
    max_dd_orig = drawdown_orig.min()
    max_dd_stop = drawdown_stop.min()
    max_dd_date_orig = drawdown_orig.idxmin()
    max_dd_date_stop = drawdown_stop.idxmin()
    
    ax3.text(max_dd_date_orig, max_dd_orig, 
             f'Max: {max_dd_orig:.1f}', 
             fontsize=9, ha='center', va='top', color='red')
    ax3.text(max_dd_date_stop, max_dd_stop, 
             f'Max: {max_dd_stop:.1f}', 
             fontsize=9, ha='center', va='bottom', color='orange')
    
    # 格式化x轴日期
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = '/mnt/e/Star-arb/data/pnl_curves_final.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"PnL曲线已保存至: {output_file}")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("PnL曲线统计")
    print("="*60)
    
    print(f"\n原始策略:")
    print(f"  最终累计PnL: {final_orig:,.2f} 元")
    print(f"  最大回撤: {(max_dd_orig*10000):,.2f} 元")
    print(f"  盈利天数: {(daily_original['pnl'] > 0).sum()} 天")
    print(f"  亏损天数: {(daily_original['pnl'] < 0).sum()} 天")
    print(f"  最大单日盈利: {daily_original['pnl'].max():,.2f} 元")
    print(f"  最大单日亏损: {daily_original['pnl'].min():,.2f} 元")
    
    print(f"\n止损策略:")
    print(f"  最终累计PnL: {final_stop:,.2f} 元")
    print(f"  最大回撤: {(max_dd_stop*10000):,.2f} 元")
    print(f"  盈利天数: {(daily_stoploss['pnl'] > 0).sum()} 天")
    print(f"  亏损天数: {(daily_stoploss['pnl'] < 0).sum()} 天")
    print(f"  最大单日盈利: {daily_stoploss['pnl'].max():,.2f} 元")
    print(f"  最大单日亏损: {daily_stoploss['pnl'].min():,.2f} 元")
    
    print(f"\n改善效果:")
    pnl_improvement = final_stop - final_orig
    dd_improvement = (max_dd_orig - max_dd_stop) * 10000
    print(f"  PnL改善: {pnl_improvement:,.2f} 元 (+{pnl_improvement/final_orig*100:.1f}%)")
    print(f"  回撤改善: {dd_improvement:,.2f} 元 ({dd_improvement/(abs(max_dd_orig)*10000)*100:.1f}% 减少)")
    
    # 月度统计
    df['月份'] = df['出场日期'].dt.to_period('M')
    monthly_stats = df.groupby('月份').agg({
        '原始盈亏': 'sum',
        '止损后盈亏': 'sum'
    })
    
    print(f"\n月度PnL对比:")
    print(f"{'月份':<12} {'原始PnL':<15} {'止损PnL':<15} {'改善':<15}")
    print("-"*60)
    for month, row in monthly_stats.iterrows():
        orig = row['原始盈亏']
        stop = row['止损后盈亏']
        diff = stop - orig
        print(f"{str(month):<12} {orig:>12,.0f}  {stop:>12,.0f}  {diff:>+12,.0f}")
    
    plt.close()
    
    return daily_original, daily_stoploss

if __name__ == "__main__":
    daily_orig, daily_stop = plot_pnl_curves()