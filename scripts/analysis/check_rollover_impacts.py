#!/usr/bin/env python3
"""
检查回测结果中超过20%盈亏的交易是否由换月造成
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_price_data(symbol):
    """加载期货价格数据"""
    try:
        df = pd.read_parquet(f'/mnt/e/Star-arb/data/{symbol}.parquet')
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return None

def detect_rollover(df, date, window=5):
    """
    检测指定日期附近是否存在换月
    换月特征：
    1. 价格跳变 > 5%
    2. 持仓量变化 > 20%
    """
    try:
        # 获取日期附近的数据
        start_date = date - timedelta(days=window)
        end_date = date + timedelta(days=window)
        
        subset = df.loc[start_date:end_date].copy()
        if len(subset) < 2:
            return False, {}
        
        # 计算日收益率和持仓量变化
        subset['return'] = subset['close'].pct_change()
        subset['oi_change'] = subset['open_interest'].pct_change()
        
        # 检查是否有换月特征
        for idx in range(1, len(subset)):
            ret = abs(subset['return'].iloc[idx])
            oi_chg = abs(subset['oi_change'].iloc[idx])
            
            if ret > 0.05 and oi_chg > 0.20:
                return True, {
                    'date': subset.index[idx],
                    'price_change': subset['return'].iloc[idx],
                    'oi_change': subset['oi_change'].iloc[idx],
                    'price_before': subset['close'].iloc[idx-1],
                    'price_after': subset['close'].iloc[idx]
                }
        
        return False, {}
    except:
        return False, {}

def analyze_trades():
    """分析交易结果中的换月影响"""
    
    # 读取交易数据
    trades_df = pd.read_csv('/mnt/e/Star-arb/output/pipeline_shifted/trades_20250823_092300.csv')
    
    # 筛选超过20%盈亏的交易
    high_impact_trades = trades_df[abs(trades_df['return_on_margin']) > 0.20].copy()
    
    print(f"总交易数: {len(trades_df)}")
    print(f"超过20%盈亏的交易数: {len(high_impact_trades)}")
    print(f"占比: {len(high_impact_trades)/len(trades_df)*100:.1f}%\n")
    
    print("=" * 100)
    print("超过20%盈亏的交易分析")
    print("=" * 100)
    
    rollover_affected = []
    
    for idx, trade in high_impact_trades.iterrows():
        pair = trade['pair']
        symbol_y, symbol_x = pair.split('-')
        open_date = pd.to_datetime(trade['open_date'])
        close_date = pd.to_datetime(trade['close_date'])
        ret = trade['return_on_margin']
        
        print(f"\n交易 {trade['trade_id']}: {pair}")
        print(f"  开仓: {open_date.strftime('%Y-%m-%d')}, 平仓: {close_date.strftime('%Y-%m-%d')}")
        print(f"  收益率: {ret*100:.1f}%")
        
        # 检查Y品种换月
        df_y = load_price_data(symbol_y)
        if df_y is not None:
            # 检查开仓和平仓期间的换月
            has_rollover_open, info_open = detect_rollover(df_y, open_date)
            has_rollover_close, info_close = detect_rollover(df_y, close_date)
            
            if has_rollover_open:
                print(f"  ⚠️ {symbol_y}在开仓附近换月: {info_open['date'].strftime('%Y-%m-%d')}, "
                      f"价格变化{info_open['price_change']*100:.1f}%, "
                      f"持仓量变化{info_open['oi_change']*100:.1f}%")
                rollover_affected.append((trade['trade_id'], symbol_y, 'open'))
            
            if has_rollover_close:
                print(f"  ⚠️ {symbol_y}在平仓附近换月: {info_close['date'].strftime('%Y-%m-%d')}, "
                      f"价格变化{info_close['price_change']*100:.1f}%, "
                      f"持仓量变化{info_close['oi_change']*100:.1f}%")
                rollover_affected.append((trade['trade_id'], symbol_y, 'close'))
        
        # 检查X品种换月
        df_x = load_price_data(symbol_x)
        if df_x is not None:
            has_rollover_open, info_open = detect_rollover(df_x, open_date)
            has_rollover_close, info_close = detect_rollover(df_x, close_date)
            
            if has_rollover_open:
                print(f"  ⚠️ {symbol_x}在开仓附近换月: {info_open['date'].strftime('%Y-%m-%d')}, "
                      f"价格变化{info_open['price_change']*100:.1f}%, "
                      f"持仓量变化{info_open['oi_change']*100:.1f}%")
                rollover_affected.append((trade['trade_id'], symbol_x, 'open'))
            
            if has_rollover_close:
                print(f"  ⚠️ {symbol_x}在平仓附近换月: {info_close['date'].strftime('%Y-%m-%d')}, "
                      f"价格变化{info_close['price_change']*100:.1f}%, "
                      f"持仓量变化{info_close['oi_change']*100:.1f}%")
                rollover_affected.append((trade['trade_id'], symbol_x, 'close'))
    
    print("\n" + "=" * 100)
    print("统计汇总")
    print("=" * 100)
    
    print(f"\n受换月影响的交易: {len(set([t[0] for t in rollover_affected]))} / {len(high_impact_trades)}")
    
    # 按盈亏分类统计
    high_profit = high_impact_trades[high_impact_trades['return_on_margin'] > 0.20]
    high_loss = high_impact_trades[high_impact_trades['return_on_margin'] < -0.20]
    
    print(f"\n高盈利交易(>20%): {len(high_profit)}个")
    for idx, trade in high_profit.iterrows():
        affected = any(t[0] == trade['trade_id'] for t in rollover_affected)
        status = "⚠️ 可能受换月影响" if affected else "✓ 正常"
        print(f"  {trade['trade_id']}. {trade['pair']}: {trade['return_on_margin']*100:.1f}% - {status}")
    
    print(f"\n高亏损交易(<-20%): {len(high_loss)}个")
    for idx, trade in high_loss.iterrows():
        affected = any(t[0] == trade['trade_id'] for t in rollover_affected)
        status = "⚠️ 可能受换月影响" if affected else "✓ 正常"
        print(f"  {trade['trade_id']}. {trade['pair']}: {trade['return_on_margin']*100:.1f}% - {status}")
    
    # 分析换月品种频率
    print("\n换月品种频率统计:")
    from collections import Counter
    symbols = [t[1] for t in rollover_affected]
    symbol_counts = Counter(symbols)
    for symbol, count in symbol_counts.most_common():
        print(f"  {symbol}: {count}次")
    
    return rollover_affected

if __name__ == "__main__":
    rollover_affected = analyze_trades()