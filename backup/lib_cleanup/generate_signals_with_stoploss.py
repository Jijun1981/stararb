#!/usr/bin/env python
"""
生成带Z-score止损的交易信号
参数调整：
- 开仓Z-score: 2.0
- 平仓Z-score: 0.5  
- 止损Z-score: 3.5
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
sys.path.append('/mnt/e/Star-arb/lib')

from data import load_from_parquet

# ==============================================================================
# 参数配置
# ==============================================================================

# Z-score参数（调整版）
Z_SCORE_OPEN = 2.0    # 开仓阈值（从2.2调整到2.0）
Z_SCORE_CLOSE = 0.5   # 平仓阈值（从0.3调整到0.5）
Z_SCORE_STOP = 3.5    # 止损阈值（新增）
WINDOW = 60           # 滚动窗口
MAX_HOLDING_DAYS = 30 # 最大持仓天数

# Beta约束
BETA_MIN = 0.3
BETA_MAX = 3.0

# 时间配置
TRAINING_START = '2023-01-01'
TRAINING_END = '2023-12-31'
CONVERGENCE_START = '2024-01-01'
CONVERGENCE_END = '2024-06-30'
SIGNAL_START = '2024-07-01'
SIGNAL_END = '2025-08-20'

# ==============================================================================
# Kalman滤波器
# ==============================================================================

class KalmanFilter1D:
    """一维Kalman滤波器（用于动态Beta估计）"""
    
    def __init__(self, initial_beta, Q=1e-5, R=0.01):
        self.beta = initial_beta
        self.P = 1.0  # 初始协方差
        self.Q = Q    # 过程噪声
        self.R = R    # 观测噪声
        
    def update(self, x_t, y_t):
        """更新Beta估计"""
        # 预测步
        beta_pred = self.beta
        P_pred = self.P + self.Q
        
        # 更新步
        innovation = y_t - beta_pred * x_t
        S = x_t * P_pred * x_t + self.R
        K = P_pred * x_t / S
        
        self.beta = beta_pred + K * innovation
        self.P = (1 - K * x_t) * P_pred
        
        return self.beta

# ==============================================================================
# 信号生成函数
# ==============================================================================

def calculate_initial_beta(x_train, y_train):
    """使用OLS计算初始Beta"""
    X = np.column_stack([x_train, np.ones_like(x_train)])
    beta, alpha = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return beta, alpha

def generate_signals_for_pair(sym1, sym2, data1, data2):
    """为一对品种生成交易信号（带Z-score止损）"""
    
    # 对齐数据（只使用共同日期）
    common_dates = data1.index.intersection(data2.index)
    data1 = data1.loc[common_dates]
    data2 = data2.loc[common_dates]
    
    # 对数价格
    log_price1 = np.log(data1['close'])
    log_price2 = np.log(data2['close'])
    
    # 使用2023年数据计算初始Beta
    mask_train = (data1.index >= TRAINING_START) & (data1.index <= TRAINING_END)
    x_train = log_price1[mask_train].values
    y_train = log_price2[mask_train].values
    
    if len(x_train) < 100:
        return None
    
    initial_beta, initial_alpha = calculate_initial_beta(x_train, y_train)
    
    # Beta过滤
    if abs(initial_beta) < BETA_MIN or abs(initial_beta) > BETA_MAX:
        return None
    
    # 初始化Kalman滤波器
    kf = KalmanFilter1D(
        initial_beta=initial_beta,
        Q=1e-5,  # 过程噪声方差
        R=0.01   # 观测噪声方差
    )
    
    # Kalman收敛阶段
    mask_conv = (data1.index >= CONVERGENCE_START) & (data1.index <= CONVERGENCE_END)
    for i in range(mask_conv.sum()):
        idx = np.where(mask_conv)[0][i]
        kf.update(log_price1.iloc[idx], log_price2.iloc[idx])
    
    # 信号生成阶段
    mask_signal = (data1.index >= SIGNAL_START) & (data1.index <= SIGNAL_END)
    signal_indices = np.where(mask_signal)[0]
    
    trades = []
    current_position = 0
    entry_date = None
    entry_index = None
    entry_spread = None
    entry_z_score = None
    entry_beta = None
    holding_days = 0
    
    for idx in signal_indices:
        # 更新Beta
        current_beta = kf.update(log_price1.iloc[idx], log_price2.iloc[idx])
        
        # 计算价差
        spread = log_price2.iloc[idx] - current_beta * log_price1.iloc[idx]
        
        # 计算Z-score（滚动窗口）
        if idx >= WINDOW:
            window_spreads = []
            for j in range(idx - WINDOW + 1, idx + 1):
                window_spread = log_price2.iloc[j] - current_beta * log_price1.iloc[j]
                window_spreads.append(window_spread)
            
            mean_spread = np.mean(window_spreads)
            std_spread = np.std(window_spreads)
            
            if std_spread > 0:
                z_score = (spread - mean_spread) / std_spread
            else:
                z_score = 0
        else:
            continue
        
        # 交易逻辑
        if current_position == 0:  # 无仓位
            if abs(z_score) > Z_SCORE_OPEN:  # 开仓信号
                current_position = -1 if z_score > 0 else 1
                entry_date = data1.index[idx]
                entry_index = idx
                entry_spread = spread
                entry_z_score = z_score
                entry_beta = current_beta
                holding_days = 0
                action = "open_short" if z_score > 0 else "open_long"
        else:  # 有仓位
            holding_days += 1
            
            # 平仓条件
            close_signal = False
            exit_reason = ""
            
            # 1. 正常平仓（回归均值）
            if abs(z_score) < Z_SCORE_CLOSE:
                close_signal = True
                exit_reason = "close_normal"
            
            # 2. 止损平仓（Z-score超过3.5）- 已禁用
            # elif abs(z_score) > Z_SCORE_STOP:
            #     close_signal = True
            #     exit_reason = "close_stoploss"
            
            # 3. 超时平仓（30天）
            elif holding_days >= MAX_HOLDING_DAYS:
                close_signal = True
                exit_reason = "close_timeout"
            
            if close_signal:
                # 记录交易
                trade = {
                    'pair': f"{sym1}-{sym2}",
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': data1.index[idx].strftime('%Y-%m-%d'),
                    'holding_days': holding_days,
                    'entry_action': "open_short" if current_position == -1 else "open_long",
                    'exit_action': exit_reason,
                    'entry_beta': entry_beta,
                    'exit_beta': current_beta,
                    'entry_z_score': entry_z_score,
                    'exit_z_score': z_score,
                    'entry_spread': entry_spread,
                    'exit_spread': spread,
                    'spread_change': spread - entry_spread
                }
                trades.append(trade)
                
                # 重置状态
                current_position = 0
                entry_date = None
                entry_index = None
                holding_days = 0
    
    # 返回结果
    if trades:
        return {
            'pair': f"{sym1}-{sym2}",
            'num_trades': len(trades),
            'position_ratio': len(trades) * 15 / len(signal_indices) if len(signal_indices) > 0 else 0,
            'initial_beta': initial_beta,
            'trades': trades,
            'trades_count': len(trades)
        }
    
    return None

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    
    print("="*80)
    print("生成带Z-score止损的交易信号")
    print("="*80)
    print(f"参数配置:")
    print(f"  开仓Z-score: {Z_SCORE_OPEN}")
    print(f"  平仓Z-score: {Z_SCORE_CLOSE}")
    print(f"  止损Z-score: {Z_SCORE_STOP}")
    print(f"  滚动窗口: {WINDOW}天")
    print(f"  最大持仓: {MAX_HOLDING_DAYS}天")
    print(f"  Beta范围: [{BETA_MIN}, {BETA_MAX}]")
    print()
    
    # 加载协整配对
    qualified_pairs = pd.read_csv('/mnt/e/Star-arb/data/pairs_verification_results.csv')
    print(f"加载 {len(qualified_pairs)} 个协整配对")
    
    # 加载价格数据
    print("\n加载价格数据...")
    symbols = list(set(qualified_pairs['sym1'].tolist() + qualified_pairs['sym2'].tolist()))
    data_dict = {}
    
    for symbol in symbols:
        try:
            df = load_from_parquet(symbol)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            data_dict[symbol] = df
            print(f"  {symbol}: 已加载")
        except Exception as e:
            print(f"  {symbol}: 加载失败 - {e}")
    
    # 生成交易信号
    print("\n生成交易信号...")
    all_results = []
    all_trades = []
    filtered_by_beta = 0
    stoploss_count = 0
    
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
                
                # 统计止损交易
                for trade in result['trades']:
                    all_trades.append(trade)
                    if trade['exit_action'] == 'close_stoploss':
                        stoploss_count += 1
                
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
    
    if all_trades:
        total_trades = len(all_trades)
        df_trades = pd.DataFrame(all_trades)
        
        print(f"\n交易总数: {total_trades}")
        
        # 退出原因统计
        exit_reasons = df_trades['exit_action'].value_counts()
        print(f"\n退出原因分布:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
        
        # 持仓天数统计
        print(f"\n持仓天数统计:")
        print(f"  平均: {df_trades['holding_days'].mean():.1f}天")
        print(f"  最短: {df_trades['holding_days'].min()}天")
        print(f"  最长: {df_trades['holding_days'].max()}天")
        
        # 保存结果
        output_file = '/mnt/e/Star-arb/data/kalman_trades_with_stoploss.csv'
        df_trades.to_csv(output_file, index=False)
        print(f"\n交易记录已保存至: {output_file}")
        
        # 保存信号摘要
        signals_file = '/mnt/e/Star-arb/data/kalman_signals_with_stoploss.json'
        with open(signals_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"信号摘要已保存至: {signals_file}")
    
    print("\n完成!")
    return all_trades

if __name__ == "__main__":
    trades = main()