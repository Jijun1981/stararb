#!/usr/bin/env python3
"""
调试信号处理流程
专门分析RB0-I0配对为什么没有产生交易信号
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.signal_generation import SignalGenerator, KalmanFilter1D

print("=" * 80)
print("信号处理流程调试")
print("=" * 80)

# 读取RB0-I0的Kalman分析数据
pair_file = "/mnt/e/Star-arb/output/kalman_analysis/RB0_I0_kalman_analysis.csv"
if not Path(pair_file).exists():
    print(f"❌ 文件不存在: {pair_file}")
    sys.exit(1)

df = pd.read_csv(pair_file)
print(f"✓ 读取数据: {len(df)}条记录")
print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")

# 检查Z-score超阈值的情况
high_z_pos = df[df['z_score'] > 2.0]
high_z_neg = df[df['z_score'] < -2.0]

print(f"\n超阈值情况统计:")
print(f"  Z > 2.0: {len(high_z_pos)}次")
print(f"  Z < -2.0: {len(high_z_neg)}次")

if len(high_z_neg) > 0:
    print(f"\nZ < -2.0的前5个例子:")
    for _, row in high_z_neg.head(5).iterrows():
        print(f"  {row['date']}: Z={row['z_score']:.3f}")

# 重新实现正确的信号生成逻辑
print(f"\n" + "=" * 60)
print("重新生成信号 - 完整流程")
print("=" * 60)

# 准备数据 (模拟从数据管理模块获取的格式)
price_data = df[['date', 'x_price', 'y_price']].copy()
price_data = price_data.rename(columns={'x_price': 'x', 'y_price': 'y'})
price_data['date'] = pd.to_datetime(price_data['date'])

# 时间配置 (模拟pipeline配置)
TIME_CONFIG = {
    'convergence_end': '2023-06-30',  # 收敛期结束
    'signal_start': '2023-07-01',     # 信号期开始
    'hist_start': '2022-01-01',       # 历史数据开始(用于R估计)
    'hist_end': '2022-12-31'          # 历史数据结束
}

# 配对参数 (模拟从协整模块获取)
initial_beta = 0.585  # 从之前的分析得出

print(f"配置参数:")
print(f"  初始Beta: {initial_beta:.6f}")
print(f"  收敛期结束: {TIME_CONFIG['convergence_end']}")
print(f"  信号期开始: {TIME_CONFIG['signal_start']}")

# 创建信号生成器
sg = SignalGenerator(
    window=60,
    z_open=2.0,
    z_close=0.5,
    convergence_days=20,
    convergence_threshold=0.01
)

print(f"\n开始处理信号...")

# 生成信号
try:
    signals_df = sg.process_pair_signals(
        pair_data=price_data,
        initial_beta=initial_beta,
        convergence_end=TIME_CONFIG['convergence_end'],
        signal_start=TIME_CONFIG['signal_start'],
        hist_start=TIME_CONFIG['hist_start'],
        hist_end=TIME_CONFIG['hist_end']
    )
    
    if signals_df.empty:
        print("❌ 没有生成任何信号!")
    else:
        print(f"✓ 生成信号: {len(signals_df)}条")
        
        # 分析信号分布
        signal_counts = signals_df['signal'].value_counts()
        print(f"\n信号分布:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # 查找实际的交易信号 (非converging, hold)
        trading_signals = signals_df[~signals_df['signal'].isin(['converging', 'hold'])]
        
        if len(trading_signals) > 0:
            print(f"\n✓ 找到交易信号: {len(trading_signals)}个")
            print(f"前10个交易信号:")
            for _, row in trading_signals.head(10).iterrows():
                print(f"  {row['date']}: {row['signal']}, Z={row['z_score']:.3f}, β={row['beta']:.6f}")
        else:
            print(f"\n❌ 没有找到任何交易信号!")
            
            # 分析为什么没有交易信号
            print(f"\n原因分析:")
            
            # 检查信号期的数据
            signal_period = signals_df[signals_df['phase'] == 'signal_period']
            print(f"  信号期数据: {len(signal_period)}条")
            
            if len(signal_period) > 0:
                # 检查Z-score超阈值的情况
                high_z_signal = signal_period[abs(signal_period['z_score']) > 2.0]
                print(f"  信号期内|Z|>2.0: {len(high_z_signal)}次")
                
                if len(high_z_signal) > 0:
                    print(f"  前5个高Z-score例子:")
                    for _, row in high_z_signal.head(5).iterrows():
                        print(f"    {row['date']}: signal={row['signal']}, Z={row['z_score']:.3f}, reason={row['reason']}")
                
                # 检查数据量是否足够
                insufficient_data = signal_period[signal_period['reason'] == 'insufficient_data']
                print(f"  数据不足: {len(insufficient_data)}条")
                
            else:
                print(f"  ❌ 没有信号期数据!")
                
                # 检查收敛期
                conv_period = signals_df[signals_df['phase'] == 'convergence_period']
                print(f"  收敛期数据: {len(conv_period)}条")
                
                if len(conv_period) > 0:
                    converged_data = conv_period[conv_period['converged'] == True]
                    print(f"  收敛成功: {len(converged_data)}条")

except Exception as e:
    print(f"❌ 信号生成失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 60)
print("对比原始Kalman分析结果")
print("=" * 60)

# 对比分析：原始Kalman分析 vs 新信号生成
print(f"原始Kalman分析:")
print(f"  总记录数: {len(df)}")
print(f"  |Z|>2.0次数: {len(df[abs(df['z_score']) > 2.0])}")

# 查看2023年7月1日之后的数据
signal_start_date = pd.to_datetime('2023-07-01')
df['date'] = pd.to_datetime(df['date'])
df_signal_period = df[df['date'] >= signal_start_date]

print(f"信号期(2023-07-01后)原始数据:")
print(f"  记录数: {len(df_signal_period)}")
if len(df_signal_period) > 0:
    high_z_original = df_signal_period[abs(df_signal_period['z_score']) > 2.0]
    print(f"  |Z|>2.0次数: {len(high_z_original)}")
    if len(high_z_original) > 0:
        print(f"  前3个例子:")
        for _, row in high_z_original.head(3).iterrows():
            print(f"    {row['date']}: Z={row['z_score']:.3f}")

print(f"\n" + "=" * 80)
print("结论")
print("=" * 80)
print("1. 检查信号生成器是否正确处理时间配置")
print("2. 检查是否有足够的数据用于Z-score计算")
print("3. 检查收敛期评估是否正确")
print("4. 验证信号生成逻辑的正确性")