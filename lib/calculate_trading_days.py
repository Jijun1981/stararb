#!/usr/bin/env python
"""
使用交易日正确计算持仓天数
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_correct_holding_days():
    """重新计算正确的交易日持仓天数"""
    
    print("="*80)
    print("使用交易日重新计算持仓天数")
    print("="*80)
    
    # 加载价格数据获取交易日历
    from data import load_from_parquet
    
    # 用任意一个品种获取交易日历
    df_price = load_from_parquet('AG0')
    df_price['date'] = pd.to_datetime(df_price['date'])
    trading_dates = sorted(df_price['date'].unique())
    trading_dates_set = set(trading_dates)
    
    print(f"交易日历范围: {trading_dates[0].strftime('%Y-%m-%d')} 至 {trading_dates[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日数: {len(trading_dates)}")
    
    # 创建交易日索引映射
    date_to_index = {date: idx for idx, date in enumerate(trading_dates)}
    
    # 加载交易数据
    df_trades = pd.read_csv('/mnt/e/Star-arb/data/kalman_trades.csv')
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
    df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
    
    # 计算正确的交易日持仓天数
    holding_days_correct = []
    
    for _, trade in df_trades.iterrows():
        entry = trade['entry_date']
        exit = trade['exit_date']
        
        # 找到最近的交易日
        if entry not in trading_dates_set:
            # 找最近的交易日
            entry = min(trading_dates, key=lambda x: abs(x - entry))
        if exit not in trading_dates_set:
            exit = min(trading_dates, key=lambda x: abs(x - exit))
        
        # 计算交易日数量
        entry_idx = date_to_index[entry]
        exit_idx = date_to_index[exit]
        
        # 持仓天数 = 出场日索引 - 入场日索引 + 1（当天进当天出算1天）
        holding_days = exit_idx - entry_idx + 1
        
        holding_days_correct.append({
            'pair': trade['pair'],
            'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': trade['exit_date'].strftime('%Y-%m-%d'),
            'old_holding_days': trade['holding_days'],
            'correct_holding_days': holding_days,
            'difference': holding_days - trade['holding_days']
        })
    
    df_holding = pd.DataFrame(holding_days_correct)
    
    # 分析差异
    print("\n持仓天数对比:")
    print(f"原始平均: {df_holding['old_holding_days'].mean():.1f} 天")
    print(f"正确平均: {df_holding['correct_holding_days'].mean():.1f} 天")
    print(f"最大差异: {df_holding['difference'].abs().max()} 天")
    
    # 查看差异最大的交易
    large_diff = df_holding[df_holding['difference'].abs() > 5].sort_values('difference', ascending=False)
    if len(large_diff) > 0:
        print(f"\n差异超过5天的交易: {len(large_diff)} 笔")
        print("\n示例（前5笔）:")
        for _, row in large_diff.head().iterrows():
            print(f"{row['pair']}: {row['entry_date']} 至 {row['exit_date']}")
            print(f"  原始: {row['old_holding_days']}天, 正确: {row['correct_holding_days']}天, 差异: {row['difference']}天")
    
    # 检查当天进出的交易
    same_day = df_holding[df_holding['entry_date'] == df_holding['exit_date']]
    if len(same_day) > 0:
        print(f"\n当天进出的交易: {len(same_day)} 笔")
        wrong_same_day = same_day[same_day['correct_holding_days'] != 1]
        if len(wrong_same_day) > 0:
            print(f"错误计算的当天交易: {len(wrong_same_day)} 笔")
            for _, row in wrong_same_day.head().iterrows():
                print(f"  {row['pair']} ({row['entry_date']}): 显示{row['old_holding_days']}天，应该是1天")
    
    # 更新kalman_trades
    df_trades['holding_days_correct'] = df_holding['correct_holding_days'].values
    
    # 保存更新后的数据
    output_file = '/mnt/e/Star-arb/data/kalman_trades_corrected.csv'
    df_trades.to_csv(output_file, index=False)
    print(f"\n更正后的交易数据已保存至: {output_file}")
    
    return df_trades, df_holding

def generate_final_report_with_correct_days():
    """生成使用正确交易日的最终报告"""
    
    # 先计算正确的持仓天数
    df_trades_corrected, df_holding = calculate_correct_holding_days()
    
    print("\n" + "="*80)
    print("生成最终报告（正确的交易日）")
    print("="*80)
    
    # 加载其他数据
    df_backtest = pd.read_csv('/mnt/e/Star-arb/data/backtest_results.csv')
    df_stoploss = pd.read_csv('/mnt/e/Star-arb/data/backtest_stoploss_v2.csv')
    
    # 创建最终报告
    df_report = pd.DataFrame()
    
    # 基本信息
    df_report['交易ID'] = df_backtest['trade_id']
    df_report['配对'] = df_backtest['pair']
    df_report['品种X'] = df_backtest['pair'].str.split('-').str[0]
    df_report['品种Y'] = df_backtest['pair'].str.split('-').str[1]
    
    # 时间信息 - 使用正确的交易日持仓天数
    df_report['入场日期'] = df_trades_corrected['entry_date']
    df_report['出场日期'] = df_trades_corrected['exit_date']
    df_report['持仓天数（交易日）'] = df_trades_corrected['holding_days_correct']
    
    # 信号信息
    df_report['入场动作'] = df_trades_corrected['entry_action']
    df_report['退出动作'] = df_trades_corrected['exit_action']
    df_report['信号类型'] = df_report['入场动作'].apply(lambda x: 'long' if 'long' in x else 'short')
    
    # Beta和Z-score
    df_report['Beta系数'] = df_trades_corrected['entry_beta']
    df_report['入场Z_Score'] = df_trades_corrected['entry_z_score']
    df_report['出场Z_Score'] = df_trades_corrected['exit_z_score']
    
    # 价格信息
    df_report['入场价格_X'] = df_backtest['entry_price_x']
    df_report['入场价格_Y'] = df_backtest['entry_price_y']
    df_report['出场价格_X'] = df_backtest['exit_price_x']
    df_report['出场价格_Y'] = df_backtest['exit_price_y']
    
    # 手数和乘数
    df_report['手数_X'] = df_backtest['lots_x']
    df_report['手数_Y'] = df_backtest['lots_y']
    
    multipliers = {
        'AG0': 15, 'AU0': 1000, 'AL0': 5, 'CU0': 5,
        'NI0': 1, 'PB0': 5, 'SN0': 1, 'ZN0': 5,
        'HC0': 10, 'I0': 100, 'RB0': 10, 'SF0': 5,
        'SM0': 5, 'SS0': 5
    }
    
    df_report['合约乘数_X'] = df_report['品种X'].map(multipliers)
    df_report['合约乘数_Y'] = df_report['品种Y'].map(multipliers)
    
    # 保证金和盈亏
    df_report['保证金'] = df_backtest['margin']
    df_report['毛盈亏'] = df_backtest['gross_pnl']
    df_report['手续费'] = df_backtest['commission']
    df_report['原始净盈亏'] = df_backtest['net_pnl']
    df_report['原始收益率%'] = df_backtest['return_pct']
    
    # 止损信息
    df_report['触发止损'] = df_stoploss['stop_triggered']
    df_report['止损后净盈亏'] = df_stoploss['stoploss_pnl']
    df_report['止损后收益率%'] = df_stoploss['stoploss_return']
    
    # 对于止损的交易，估算实际持仓天数
    df_report['实际持仓天数'] = df_report.apply(
        lambda row: min(5, row['持仓天数（交易日）']//3) if row['触发止损'] else row['持仓天数（交易日）'],
        axis=1
    )
    
    # 价差信息
    df_report['入场价差'] = df_trades_corrected['entry_spread']
    df_report['出场价差'] = df_trades_corrected['exit_spread']
    df_report['价差变动'] = df_trades_corrected['spread_change']
    
    # 保存最终报告
    output_file = '/mnt/e/Star-arb/data/final_report_trading_days.csv'
    df_report.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.4f')
    
    print(f"\n最终报告已生成: {output_file}")
    print(f"总交易数: {len(df_report)}")
    
    # 统计
    print("\n交易统计:")
    print(f"  持仓天数（交易日）:")
    print(f"    最短: {df_report['持仓天数（交易日）'].min()} 天")
    print(f"    最长: {df_report['持仓天数（交易日）'].max()} 天")
    print(f"    平均: {df_report['持仓天数（交易日）'].mean():.1f} 天")
    print(f"    中位数: {df_report['持仓天数（交易日）'].median():.0f} 天")
    
    # 检查30天限制
    over_30 = df_report[df_report['持仓天数（交易日）'] > 30]
    if len(over_30) > 0:
        print(f"\n⚠️ 超过30天的交易: {len(over_30)} 笔")
        for _, trade in over_30.head().iterrows():
            print(f"  {trade['配对']}: {trade['持仓天数（交易日）']}天")
    else:
        print(f"\n✓ 所有交易都在30天内平仓")
    
    # 止损统计
    stopped = df_report[df_report['触发止损']]
    print(f"\n止损交易:")
    print(f"  触发数量: {len(stopped)} 笔")
    if len(stopped) > 0:
        print(f"  止损后平均持仓: {stopped['实际持仓天数'].mean():.1f} 天")
    
    # 最终盈亏
    total_pnl = df_report['止损后净盈亏'].sum()
    original_pnl = df_report['原始净盈亏'].sum()
    
    print(f"\n最终盈亏:")
    print(f"  原始策略: {original_pnl:,.2f} 元")
    print(f"  止损策略: {total_pnl:,.2f} 元")
    print(f"  改善: {total_pnl - original_pnl:,.2f} 元 (+{(total_pnl-original_pnl)/abs(original_pnl)*100:.1f}%)")
    
    return df_report

if __name__ == "__main__":
    df_report = generate_final_report_with_correct_days()