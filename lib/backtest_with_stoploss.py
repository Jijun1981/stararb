#!/usr/bin/env python
"""
带止损的回测引擎
- 15%单笔保证金止损
- 逐日计算浮动盈亏
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def load_price_data():
    """加载价格数据"""
    from data import load_from_parquet
    
    # 加载所有品种的价格数据
    symbols = ['AG0', 'AU0', 'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',
               'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0']
    
    price_data = {}
    for symbol in symbols:
        try:
            df = load_from_parquet(symbol)
            if df is not None:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                price_data[symbol] = df
                print(f"  加载 {symbol}: {len(df)} 条数据")
        except Exception as e:
            print(f"  {symbol}: 加载失败 - {e}")
    
    return price_data

def calculate_daily_pnl(entry_date, exit_date, entry_px, entry_py, beta, 
                       lots_x, lots_y, sym1, sym2, price_data, 
                       stop_loss_pct=0.15, margin=None):
    """
    计算每日PnL并检查止损
    
    Parameters:
    - stop_loss_pct: 止损比例（相对于保证金）
    """
    
    # 获取价格序列
    df1 = price_data[sym1]
    df2 = price_data[sym2]
    
    # 获取交易期间的日期
    trade_dates = pd.date_range(start=entry_date, end=exit_date, freq='D')
    trade_dates = trade_dates[trade_dates.isin(df1.index) & trade_dates.isin(df2.index)]
    
    daily_pnls = []
    cumulative_pnl = 0
    stop_triggered = False
    actual_exit_date = exit_date
    actual_exit_px = None
    actual_exit_py = None
    
    for date in trade_dates:
        px = df1.loc[date, 'close']
        py = df2.loc[date, 'close']
        
        # 计算当日PnL
        if beta > 0:  # 正beta：对冲交易
            # Y是X的beta倍
            # 做多spread = 做多Y + 做空X*beta
            # 做空spread = 做空Y + 做多X*beta
            if lots_y > 0:  # 做多Y（做多spread）
                daily_pnl = (py - entry_py) * lots_y * 10 - (px - entry_px) * lots_x * 5
            else:  # 做空Y（做空spread）
                daily_pnl = (entry_py - py) * abs(lots_y) * 10 + (px - entry_px) * lots_x * 5
        else:  # 负beta：同向交易
            # Y = alpha + beta*X，beta<0意味着反向关系
            # 做多spread = 做多Y + 做多X*|beta|
            # 做空spread = 做空Y + 做空X*|beta|
            if lots_y > 0:  # 做多（同时做多X和Y）
                daily_pnl = (py - entry_py) * lots_y * 10 + (px - entry_px) * lots_x * 5
            else:  # 做空（同时做空X和Y）
                daily_pnl = (entry_py - py) * abs(lots_y) * 10 + (entry_px - px) * lots_x * 5
        
        cumulative_pnl = daily_pnl
        
        # 检查止损
        if margin and cumulative_pnl < -margin * stop_loss_pct:
            stop_triggered = True
            actual_exit_date = date
            actual_exit_px = px
            actual_exit_py = py
            break
        
        daily_pnls.append({
            'date': date,
            'px': px,
            'py': py,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': cumulative_pnl
        })
        
        actual_exit_px = px
        actual_exit_py = py
    
    return {
        'stop_triggered': stop_triggered,
        'actual_exit_date': actual_exit_date,
        'actual_exit_px': actual_exit_px,
        'actual_exit_py': actual_exit_py,
        'final_pnl': cumulative_pnl,
        'daily_pnls': daily_pnls
    }

def run_backtest_with_stoploss():
    """运行带止损的回测"""
    
    print("="*80)
    print("带15%止损的回测分析")
    print("="*80)
    
    # 加载原始交易
    df_trades = pd.read_csv('/mnt/e/Star-arb/data/kalman_trades.csv')
    print(f"加载 {len(df_trades)} 笔交易")
    
    # 加载价格数据
    print("\n加载价格数据...")
    price_data = load_price_data()
    
    # 读取原始回测结果获取手数和保证金信息
    df_original = pd.read_csv('/mnt/e/Star-arb/data/backtest_results.csv')
    
    # 处理每笔交易
    results = []
    stop_loss_count = 0
    
    for idx, trade in df_trades.iterrows():
        # 从原始回测中获取对应交易的详细信息
        orig_trade = df_original[
            (df_original['pair'] == trade['pair']) & 
            (df_original['entry_date'] == trade['entry_date'])
        ]
        
        if len(orig_trade) == 0:
            continue
            
        orig_trade = orig_trade.iloc[0]
        
        # 解析品种
        sym1, sym2 = trade['pair'].split('-')
        
        # 计算带止损的PnL
        result = calculate_daily_pnl(
            entry_date=pd.to_datetime(trade['entry_date']),
            exit_date=pd.to_datetime(trade['exit_date']),
            entry_px=orig_trade['entry_price_x'],
            entry_py=orig_trade['entry_price_y'],
            beta=orig_trade['beta'],
            lots_x=orig_trade['lots_x'],
            lots_y=orig_trade['lots_y'],
            sym1=sym1,
            sym2=sym2,
            price_data=price_data,
            stop_loss_pct=0.15,
            margin=orig_trade['margin']
        )
        
        # 计算手续费
        if sym1 in ['AU0']:
            commission = (orig_trade['entry_price_x'] + result['actual_exit_px']) * orig_trade['lots_x'] * 10 * 0.00002
        else:
            commission = (orig_trade['entry_price_x'] + result['actual_exit_px']) * orig_trade['lots_x'] * 5 * 0.00002
        
        if sym2 in ['AU0']:
            commission += (orig_trade['entry_price_y'] + result['actual_exit_py']) * abs(orig_trade['lots_y']) * 10 * 0.00002
        else:
            commission += (orig_trade['entry_price_y'] + result['actual_exit_py']) * abs(orig_trade['lots_y']) * 5 * 0.00002
        
        # 净PnL
        net_pnl = result['final_pnl'] - commission
        
        # 记录结果
        trade_result = {
            'trade_id': idx + 1,
            'pair': trade['pair'],
            'entry_date': trade['entry_date'],
            'original_exit_date': trade['exit_date'],
            'actual_exit_date': result['actual_exit_date'].strftime('%Y-%m-%d'),
            'stop_triggered': result['stop_triggered'],
            'margin': orig_trade['margin'],
            'gross_pnl': result['final_pnl'],
            'commission': commission,
            'net_pnl': net_pnl,
            'return_pct': net_pnl / orig_trade['margin'] * 100,
            'original_pnl': orig_trade['net_pnl'],
            'pnl_difference': net_pnl - orig_trade['net_pnl']
        }
        
        results.append(trade_result)
        
        if result['stop_triggered']:
            stop_loss_count += 1
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 统计分析
    print("\n" + "="*80)
    print("止损统计")
    print("="*80)
    
    total_trades = len(df_results)
    stopped_trades = df_results['stop_triggered'].sum()
    stop_rate = stopped_trades / total_trades * 100
    
    print(f"总交易数: {total_trades}")
    print(f"触发止损数: {stopped_trades}")
    print(f"止损触发率: {stop_rate:.1f}%")
    
    # 收益对比
    original_pnl = df_results['original_pnl'].sum()
    new_pnl = df_results['net_pnl'].sum()
    pnl_impact = new_pnl - original_pnl
    
    print(f"\n原始总PnL: {original_pnl:,.2f} 元")
    print(f"止损后总PnL: {new_pnl:,.2f} 元")
    print(f"止损影响: {pnl_impact:,.2f} 元 ({pnl_impact/original_pnl*100:.1f}%)")
    
    # 收益率统计
    win_rate = (df_results['net_pnl'] > 0).sum() / total_trades * 100
    avg_return = df_results['return_pct'].mean()
    
    print(f"\n胜率: {win_rate:.1f}%")
    print(f"平均收益率: {avg_return:.2f}%")
    
    # 止损交易分析
    if stopped_trades > 0:
        stopped_df = df_results[df_results['stop_triggered']]
        print(f"\n止损交易详情:")
        print(f"  平均亏损: {stopped_df['net_pnl'].mean():,.2f} 元")
        print(f"  最大亏损: {stopped_df['net_pnl'].min():,.2f} 元")
        print(f"  止损避免的额外亏损: {stopped_df['pnl_difference'].sum():,.2f} 元")
    
    # 保存结果
    output_file = '/mnt/e/Star-arb/data/backtest_with_stoploss.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")
    
    return df_results

def calculate_metrics_with_stoploss(df_results):
    """计算带止损的绩效指标"""
    
    print("\n" + "="*80)
    print("带止损的绩效指标")
    print("="*80)
    
    # 基础统计
    total_pnl = df_results['net_pnl'].sum()
    total_trades = len(df_results)
    win_rate = (df_results['net_pnl'] > 0).sum() / total_trades * 100
    
    # 计算累计PnL曲线
    df_sorted = df_results.sort_values('actual_exit_date')
    df_sorted['cum_pnl'] = df_sorted['net_pnl'].cumsum()
    
    # 最大回撤
    peak = df_sorted['cum_pnl'].expanding().max()
    drawdown = df_sorted['cum_pnl'] - peak
    max_drawdown = drawdown.min()
    max_dd_pct = (max_drawdown / peak[drawdown.idxmin()]) * 100 if peak[drawdown.idxmin()] > 0 else 0
    
    # 基于保证金的收益率
    avg_margin = df_results['margin'].mean()
    max_margin = 1613984.40  # 之前计算的最大同时占用
    
    total_return = total_pnl / max_margin * 100
    annual_return = total_return / 1.1  # 约1.1年
    
    # 夏普比率（简化计算）
    returns = df_results['return_pct']
    sharpe = returns.mean() / returns.std() * np.sqrt(252/26) if returns.std() > 0 else 0
    
    print(f"\n【收益指标】")
    print(f"  总净盈亏: {total_pnl:,.2f} 元")
    print(f"  总收益率（基于最大占用）: {total_return:.2f}%")
    print(f"  年化收益率: {annual_return:.2f}%")
    
    print(f"\n【风险指标】")
    print(f"  最大回撤: {max_drawdown:,.2f} 元 ({max_dd_pct:.2f}%)")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  年化夏普比率: {sharpe:.2f}")
    
    # 与原始对比
    original_pnl = 466569.40
    improvement = total_pnl - original_pnl
    
    print(f"\n【止损效果】")
    print(f"  原始收益: {original_pnl:,.2f} 元")
    print(f"  止损后收益: {total_pnl:,.2f} 元")
    print(f"  改善金额: {improvement:,.2f} 元")
    print(f"  改善比例: {improvement/original_pnl*100:.1f}%")
    
    return {
        'total_pnl': total_pnl,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe
    }

if __name__ == "__main__":
    # 运行带止损的回测
    df_results = run_backtest_with_stoploss()
    
    # 计算绩效指标
    metrics = calculate_metrics_with_stoploss(df_results)