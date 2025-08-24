#!/usr/bin/env python3
"""
60天OLS滚动信号生成脚本
使用60天滚动窗口OLS回归估计Beta，生成Z-score信号
确保2023年7月1日开始就有信号

与Kalman滤波方法进行对比分析
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加lib路径
sys.path.insert(0, '/mnt/e/Star-arb/lib')
sys.path.insert(0, '/mnt/e/Star-arb')

from lib.data import load_symbol_data
from lib.backtest import run_backtest

def calculate_rolling_ols_beta(y_data, x_data, window=60):
    """
    计算60天滚动OLS Beta
    
    Args:
        y_data: Y序列 (被回归变量)
        x_data: X序列 (回归变量) 
        window: 滚动窗口大小
    
    Returns:
        pandas.Series: 滚动Beta序列
    """
    aligned_data = pd.DataFrame({'y': y_data, 'x': x_data}).dropna()
    
    beta_series = pd.Series(index=aligned_data.index, dtype=float)
    
    for i in range(window-1, len(aligned_data)):
        y_window = aligned_data['y'].iloc[i-window+1:i+1]
        x_window = aligned_data['x'].iloc[i-window+1:i+1]
        
        if len(y_window) == window and y_window.std() > 0 and x_window.std() > 0:
            # OLS回归: y = alpha + beta * x
            covariance = np.cov(y_window, x_window)[0, 1]
            variance_x = np.var(x_window, ddof=1)
            
            if variance_x > 0:
                beta = covariance / variance_x
                beta_series.iloc[i] = beta
    
    return beta_series

def generate_ols_rolling_signals(pair_info, start_date='2023-07-01', end_date='2024-12-31'):
    """
    生成60天OLS滚动信号
    
    Args:
        pair_info: dict, 包含配对信息
        start_date: 信号开始日期
        end_date: 信号结束日期
    
    Returns:
        DataFrame: 信号数据
    """
    symbol_x = pair_info['symbol_x']
    symbol_y = pair_info['symbol_y']
    direction = pair_info['direction']
    
    print(f"处理配对: {symbol_x}-{symbol_y}, 方向: {direction}")
    
    # 加载数据，需要更早的数据用于60天窗口
    data_start = '2023-03-01'  # 提前4个月确保7月1日有足够历史数据
    
    try:
        df_x = load_symbol_data(symbol_x, start_date=data_start, end_date=end_date)
        df_y = load_symbol_data(symbol_y, start_date=data_start, end_date=end_date)
    except Exception as e:
        print(f"加载数据失败 {symbol_x}-{symbol_y}: {e}")
        return pd.DataFrame()
    
    if df_x.empty or df_y.empty:
        print(f"数据为空 {symbol_x}-{symbol_y}")
        return pd.DataFrame()
    
    # 合并数据并对齐
    data = pd.merge(df_x[['close']], df_y[['close']], 
                   left_index=True, right_index=True, 
                   how='inner', suffixes=('_x', '_y'))
    
    if len(data) < 60:
        print(f"数据不足60天: {symbol_x}-{symbol_y}, 只有{len(data)}天")
        return pd.DataFrame()
    
    # 计算对数价格
    data['log_x'] = np.log(data['close_x'])
    data['log_y'] = np.log(data['close_y'])
    
    # 根据方向确定回归关系
    if direction == 'x_on_y':
        # X对Y回归: X = alpha + beta * Y
        y_var = data['log_y']  # 自变量
        x_var = data['log_x']  # 因变量
    else:  # y_on_x
        # Y对X回归: Y = alpha + beta * X  
        y_var = data['log_x']  # 自变量
        x_var = data['log_y']  # 因变量
    
    # 计算60天滚动OLS Beta
    print(f"计算60天滚动OLS Beta...")
    rolling_beta = calculate_rolling_ols_beta(x_var, y_var, window=60)
    
    # 计算残差和Z-score
    signals = []
    
    for date in rolling_beta.dropna().index:
        if date < pd.Timestamp(start_date):
            continue
            
        beta = rolling_beta[date]
        if pd.isna(beta):
            continue
        
        # 计算残差
        if direction == 'x_on_y':
            residual = data.loc[date, 'log_x'] - beta * data.loc[date, 'log_y']
            # 计算60天滚动Z-score
            end_idx = data.index.get_loc(date)
            start_idx = max(0, end_idx - 59)
            window_data = data.iloc[start_idx:end_idx+1]
            
            residuals_window = window_data['log_x'] - beta * window_data['log_y']
        else:
            residual = data.loc[date, 'log_y'] - beta * data.loc[date, 'log_x']
            # 计算60天滚动Z-score
            end_idx = data.index.get_loc(date)
            start_idx = max(0, end_idx - 59)
            window_data = data.iloc[start_idx:end_idx+1]
            
            residuals_window = window_data['log_y'] - beta * window_data['log_x']
        
        if len(residuals_window) > 1:
            z_score = (residual - residuals_window.mean()) / residuals_window.std()
        else:
            continue
            
        signals.append({
            'date': date,
            'pair': f"{symbol_x}-{symbol_y}",
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'direction': direction,
            'price_x': data.loc[date, 'close_x'],
            'price_y': data.loc[date, 'close_y'],
            'ols_beta': beta,
            'residual': residual,
            'z_score': z_score
        })
    
    if not signals:
        print(f"未生成信号: {symbol_x}-{symbol_y}")
        return pd.DataFrame()
        
    signals_df = pd.DataFrame(signals)
    signals_df.set_index('date', inplace=True)
    
    print(f"生成信号数: {len(signals_df)}, 日期范围: {signals_df.index.min()} 到 {signals_df.index.max()}")
    
    return signals_df

def main():
    """主函数"""
    print("=" * 80)
    print("60天OLS滚动信号生成")
    print("=" * 80)
    
    # 读取协整结果
    coint_file = "/mnt/e/Star-arb/output/pipeline_shifted/cointegration_results.csv"
    if not os.path.exists(coint_file):
        print(f"协整结果文件不存在: {coint_file}")
        return
        
    coint_results = pd.read_csv(coint_file)
    print(f"加载{len(coint_results)}个协整配对")
    
    # 生成所有配对的信号
    all_signals = []
    
    for _, row in coint_results.iterrows():
        pair_info = {
            'symbol_x': row['symbol_x'],
            'symbol_y': row['symbol_y'], 
            'direction': row['direction'],
            'pvalue_4y': row['pvalue_4y'],
            'beta_1y': row['beta_1y']
        }
        
        signals_df = generate_ols_rolling_signals(pair_info)
        if not signals_df.empty:
            all_signals.append(signals_df)
    
    if not all_signals:
        print("未生成任何信号")
        return
        
    # 合并所有信号
    combined_signals = pd.concat(all_signals, ignore_index=False)
    combined_signals.sort_index(inplace=True)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/mnt/e/Star-arb/output/ols_rolling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存信号
    signals_file = f"{output_dir}/signals_ols_rolling_{timestamp}.csv"
    combined_signals.to_csv(signals_file)
    print(f"\n信号已保存: {signals_file}")
    print(f"总信号数: {len(combined_signals)}")
    print(f"配对数: {combined_signals['pair'].nunique()}")
    print(f"日期范围: {combined_signals.index.min()} 到 {combined_signals.index.max()}")
    
    # 检查是否从2023-07-01开始有信号
    first_signal_date = combined_signals.index.min()
    target_date = pd.Timestamp('2023-07-01')
    
    print(f"\n信号时间检查:")
    print(f"目标开始日期: {target_date}")
    print(f"实际首个信号: {first_signal_date}")
    
    if first_signal_date <= target_date + timedelta(days=7):  # 允许一周内的差异
        print("✅ 信号时间符合要求")
    else:
        print("❌ 信号开始时间偏晚")
    
    # 运行回测对比
    print("\n" + "=" * 80)
    print("运行OLS滚动策略回测")
    print("=" * 80)
    
    try:
        # 转换信号格式用于回测
        backtest_signals = combined_signals.reset_index()
        backtest_signals = backtest_signals.rename(columns={'date': 'datetime'})
        
        # 运行回测
        backtest_result = run_backtest(
            signals_df=backtest_signals,
            initial_capital=5000000,
            z_open_threshold=2.0,
            z_close_threshold=0.5,
            z_open_max=3.2,
            stop_loss_pct=0.10,
            max_hold_days=30,
            output_dir=output_dir
        )
        
        print(f"\n回测完成，结果保存在: {output_dir}")
        
        # 输出关键指标
        if backtest_result and 'summary' in backtest_result:
            summary = backtest_result['summary']
            print(f"\n=== OLS滚动策略回测结果 ===")
            print(f"总收益率: {summary.get('total_return', 'N/A'):.2%}")
            print(f"年化收益率: {summary.get('annualized_return', 'N/A'):.2%}")
            print(f"夏普比率: {summary.get('sharpe_ratio', 'N/A'):.3f}")
            print(f"最大回撤: {summary.get('max_drawdown', 'N/A'):.2%}")
            print(f"交易次数: {summary.get('total_trades', 'N/A')}")
            print(f"胜率: {summary.get('win_rate', 'N/A'):.1%}")
            
    except Exception as e:
        print(f"回测运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()