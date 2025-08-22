#!/usr/bin/env python
"""
使用卡尔曼滤波生成交易信号
时间配置：
- Beta训练期：2023-01-01 至 2023-12-31
- Kalman收敛期：2024-01-01 至 2024-06-30
- 信号生成期：2024-07-01 至 2025-08-20
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_from_parquet
from signal_generation import KalmanFilter1D
import json

# 时间配置
BETA_TRAIN_START = '2023-01-01'
BETA_TRAIN_END = '2023-12-31'
KALMAN_CONVERGE_START = '2024-01-01'
KALMAN_CONVERGE_END = '2024-06-30'
SIGNAL_START = '2024-07-01'
SIGNAL_END = '2025-08-20'

# 信号参数
Z_SCORE_OPEN = 2.2     # 开仓阈值
Z_SCORE_CLOSE = 0.3    # 平仓阈值
ROLLING_WINDOW = 60    # 滚动窗口
MAX_HOLDING_DAYS = 30  # 最大持仓天数

def calculate_initial_beta(x_train, y_train):
    """计算初始beta (OLS)"""
    x_with_const = np.column_stack([np.ones(len(x_train)), x_train])
    params = np.linalg.lstsq(x_with_const, y_train, rcond=None)[0]
    return params[1], params[0]  # beta, alpha

def generate_signals_for_pair(sym1, sym2, data1, data2):
    """为单个配对生成交易信号"""
    
    # 对齐数据
    common_dates = data1.index.intersection(data2.index)
    df1 = data1.loc[common_dates].copy()
    df2 = data2.loc[common_dates].copy()
    
    # 划分时间段
    beta_train_mask = (common_dates >= pd.Timestamp(BETA_TRAIN_START)) & \
                      (common_dates <= pd.Timestamp(BETA_TRAIN_END))
    kalman_converge_mask = (common_dates >= pd.Timestamp(KALMAN_CONVERGE_START)) & \
                          (common_dates <= pd.Timestamp(KALMAN_CONVERGE_END))
    signal_mask = (common_dates >= pd.Timestamp(SIGNAL_START)) & \
                  (common_dates <= pd.Timestamp(SIGNAL_END))
    
    # 获取各阶段数据
    train_dates = common_dates[beta_train_mask]
    converge_dates = common_dates[kalman_converge_mask]
    signal_dates = common_dates[signal_mask]
    
    if len(train_dates) < 100 or len(converge_dates) < 50 or len(signal_dates) < 20:
        return None
    
    # 训练期数据
    x_train = df1.loc[train_dates, 'close'].values
    y_train = df2.loc[train_dates, 'close'].values
    
    # 计算初始beta
    initial_beta, initial_alpha = calculate_initial_beta(x_train, y_train)
    
    # Beta过滤：绝对值必须在0.3到3之间
    if abs(initial_beta) < 0.3 or abs(initial_beta) > 3:
        return None
    
    # 初始化卡尔曼滤波器
    kf = KalmanFilter1D(
        initial_beta=initial_beta,
        Q=1e-5,  # 过程噪声方差
        R=1e-3,  # 观测噪声方差
        P0=0.1   # 初始不确定性
    )
    
    # 收敛期：让卡尔曼滤波器收敛
    for date in converge_dates:
        x_t = df1.loc[date, 'close']
        y_t = df2.loc[date, 'close']
        kf.update(y_t, x_t)  # 注意参数顺序: y_t, x_t
    
    # 信号生成期
    results = []
    positions = []
    holding_days = 0
    current_position = 0
    entry_date = None
    entry_index = None  # 记录入场的索引位置
    
    # 滚动窗口数据
    spread_history = []
    
    for i, date in enumerate(signal_dates):
        x_t = df1.loc[date, 'close']
        y_t = df2.loc[date, 'close']
        
        # 更新卡尔曼滤波器
        update_result = kf.update(y_t, x_t)
        beta_t = kf.beta
        
        # 计算价差 (没有alpha项，因为KalmanFilter1D只估计beta)
        spread = y_t - beta_t * x_t
        spread_history.append(spread)
        
        # 保持滚动窗口
        if len(spread_history) > ROLLING_WINDOW:
            spread_history.pop(0)
        
        # 计算z-score
        if len(spread_history) >= 20:  # 至少20个数据点
            spread_mean = np.mean(spread_history)
            spread_std = np.std(spread_history)
            
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
            else:
                z_score = 0
        else:
            z_score = 0
            spread_mean = 0
            spread_std = 0
        
        # 生成交易信号
        signal = 0
        action = "hold"
        
        if current_position == 0:  # 无仓位
            if abs(z_score) > Z_SCORE_OPEN:
                signal = -1 if z_score > 0 else 1  # 均值回归
                current_position = signal
                entry_date = date
                entry_index = i  # 记录入场索引
                holding_days = 0
                action = "open_short" if signal == -1 else "open_long"
        else:  # 有仓位
            holding_days += 1
            
            # 平仓条件
            if abs(z_score) < Z_SCORE_CLOSE:  # 回归均值
                action = "close_normal"
                signal = 0
                current_position = 0
                entry_date = None
                entry_index = None
                holding_days = 0
            elif holding_days >= MAX_HOLDING_DAYS:  # 强制平仓
                action = "close_timeout"
                signal = 0
                current_position = 0
                entry_date = None
                entry_index = None
                holding_days = 0
            else:
                signal = current_position  # 保持仓位
        
        # 记录结果
        result = {
            'date': date,
            'x_price': x_t,
            'y_price': y_t,
            'beta': beta_t,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'z_score': z_score,
            'signal': signal,
            'position': current_position,
            'action': action,
            'holding_days': holding_days
        }
        
        results.append(result)
        positions.append(current_position)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 统计交易
    trades = []
    for i, row in df_results.iterrows():
        if row['action'].startswith('open'):
            trade = {
                'entry_date': row['date'],
                'entry_action': row['action'],
                'entry_z_score': row['z_score'],
                'entry_spread': row['spread'],
                'entry_beta': row['beta']
            }
            
            # 查找对应的平仓
            for j in range(i+1, len(df_results)):
                if df_results.iloc[j]['action'].startswith('close'):
                    exit_row = df_results.iloc[j]
                    trade['exit_date'] = exit_row['date']
                    trade['exit_action'] = exit_row['action']
                    trade['exit_z_score'] = exit_row['z_score']
                    trade['exit_spread'] = exit_row['spread']
                    # 使用交易日计数，而不是自然日
                    trade['holding_days'] = j - i  # 交易日数量
                    trade['spread_change'] = exit_row['spread'] - row['spread']
                    trades.append(trade)
                    break
    
    return {
        'pair': f"{sym1}-{sym2}",
        'sym1': sym1,
        'sym2': sym2,
        'initial_beta': initial_beta,
        'initial_alpha': initial_alpha,
        'signals': df_results,
        'trades': trades,
        'num_trades': len(trades),
        'days_in_position': sum(1 for p in positions if p != 0),
        'position_ratio': sum(1 for p in positions if p != 0) / len(positions) if positions else 0
    }

def main():
    """主函数"""
    
    print("="*80)
    print("卡尔曼滤波交易信号生成")
    print("="*80)
    print(f"Beta训练期: {BETA_TRAIN_START} 至 {BETA_TRAIN_END}")
    print(f"Kalman收敛期: {KALMAN_CONVERGE_START} 至 {KALMAN_CONVERGE_END}")
    print(f"信号生成期: {SIGNAL_START} 至 {SIGNAL_END}")
    print(f"Z-score开仓阈值: {Z_SCORE_OPEN}")
    print(f"Z-score平仓阈值: {Z_SCORE_CLOSE}")
    print(f"滚动窗口: {ROLLING_WINDOW}天")
    print(f"最大持仓: {MAX_HOLDING_DAYS}天")
    print()
    
    # 加载验证结果
    verification_file = '/mnt/e/Star-arb/data/pairs_verification_results.csv'
    df_pairs = pd.read_csv(verification_file)
    
    # 筛选满足条件的配对（5年和1年p值都<0.05）
    qualified_pairs = df_pairs[(df_pairs['pass_5y']) & (df_pairs['pass_1y'])]
    print(f"符合协整条件的配对数: {len(qualified_pairs)}")
    print()
    
    # 加载数据
    print("加载数据...")
    data_dict = {}
    symbols = set()
    
    for _, row in qualified_pairs.iterrows():
        symbols.add(row['sym1'])
        symbols.add(row['sym2'])
    
    for symbol in symbols:
        try:
            df = load_from_parquet(symbol)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                data_dict[symbol] = df
        except Exception as e:
            print(f"  {symbol}: 加载失败 - {e}")
    
    print(f"成功加载 {len(data_dict)} 个品种数据")
    print()
    
    # 生成信号
    print("生成交易信号...")
    all_results = []
    filtered_by_beta = 0
    
    for idx, row in qualified_pairs.iterrows():
        sym1 = row['sym1']
        sym2 = row['sym2']
        
        if sym1 not in data_dict or sym2 not in data_dict:
            continue
        
        try:
            result = generate_signals_for_pair(
                sym1, sym2,
                data_dict[sym1], 
                data_dict[sym2]
            )
            
            if result is None:
                filtered_by_beta += 1
            elif result['num_trades'] > 0:
                all_results.append(result)
                print(f"  {sym1}-{sym2}: {result['num_trades']} 笔交易, "
                      f"持仓率 {result['position_ratio']:.1%}, Beta={result['initial_beta']:.4f}")
        except Exception as e:
            print(f"  {sym1}-{sym2}: 生成失败 - {e}")
    
    print()
    print("="*80)
    print("交易统计")
    print("="*80)
    
    print(f"\nBeta过滤统计:")
    print(f"  被Beta过滤的配对数: {filtered_by_beta}")
    print(f"  通过Beta过滤的配对数: {len(all_results)}")
    
    # 汇总所有交易
    all_trades = []
    for result in all_results:
        for trade in result['trades']:
            trade['pair'] = result['pair']
            all_trades.append(trade)
    
    df_trades = pd.DataFrame(all_trades)
    
    if len(df_trades) > 0:
        print(f"总配对数: {len(all_results)}")
        print(f"总交易数: {len(df_trades)}")
        print(f"平均每配对交易数: {len(df_trades)/len(all_results):.1f}")
        print()
        
        # 按交易数量排序
        print("交易最活跃的配对 (Top 10):")
        print(f"{'配对':<12} {'交易数':<8} {'持仓率':<10} {'初始Beta':<10}")
        print("-"*50)
        
        sorted_results = sorted(all_results, key=lambda x: x['num_trades'], reverse=True)[:10]
        for result in sorted_results:
            print(f"{result['pair']:<12} {result['num_trades']:<8} "
                  f"{result['position_ratio']:<10.1%} {result['initial_beta']:<10.4f}")
        
        print()
        print("交易退出原因统计:")
        exit_reasons = df_trades['exit_action'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/len(df_trades)*100:.1f}%)")
        
        print()
        print("持仓天数统计:")
        print(f"  平均: {df_trades['holding_days'].mean():.1f} 天")
        print(f"  中位数: {df_trades['holding_days'].median():.0f} 天")
        print(f"  最短: {df_trades['holding_days'].min()} 天")
        print(f"  最长: {df_trades['holding_days'].max()} 天")
        
        # 保存交易记录
        output_file = '/mnt/e/Star-arb/data/kalman_trades.csv'
        df_trades.to_csv(output_file, index=False)
        print(f"\n交易记录已保存至: {output_file}")
        
        # 保存信号详情
        signals_file = '/mnt/e/Star-arb/data/kalman_signals.json'
        signals_data = []
        for result in all_results:
            signals_data.append({
                'pair': result['pair'],
                'num_trades': result['num_trades'],
                'position_ratio': result['position_ratio'],
                'initial_beta': result['initial_beta'],
                'trades_count': len(result['trades'])
            })
        
        with open(signals_file, 'w') as f:
            json.dump(signals_data, f, indent=2)
        print(f"信号统计已保存至: {signals_file}")
        
    else:
        print("没有生成任何交易信号")
    
    return all_results, df_trades

if __name__ == "__main__":
    results, trades = main()