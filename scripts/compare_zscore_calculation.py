#!/usr/bin/env python3
"""
对比Z-score计算方法
分析原始Kalman分析和signal_generation.py的差异
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.signal_generation import SignalGenerator

print("=" * 80)
print("Z-score计算方法对比")
print("=" * 80)

# 读取原始Kalman分析数据
pair_file = "/mnt/e/Star-arb/output/kalman_analysis/RB0_I0_kalman_analysis.csv"
df = pd.read_csv(pair_file)

print(f"✓ 读取数据: {len(df)}条记录")
print(f"  原始数据Z-score范围: {df['z_score'].min():.3f} ~ {df['z_score'].max():.3f}")

# 准备数据
x_prices = df['x_price'].values  # log价格
y_prices = df['y_price'].values  # log价格
kalman_betas = df['kalman_beta'].values
dates = pd.to_datetime(df['date'])

# 信号期开始时间
signal_start = pd.to_datetime('2023-07-01')
signal_start_idx = dates.searchsorted(signal_start)

print(f"  信号期开始索引: {signal_start_idx} (日期: {dates[signal_start_idx]})")

# 方法1: 原始Kalman分析的残差和Z-score
print(f"\n方法1: 原始Kalman分析")
original_residuals = df['residual'].values
original_z_scores = df['z_score'].values

# 方法2: 重新计算残差 (使用相同的Kalman beta)
print(f"\n方法2: 重新计算残差")
recalc_residuals = y_prices - kalman_betas * x_prices
print(f"  残差差异 (原始 vs 重计算): 最大={max(abs(original_residuals - recalc_residuals)):.6f}")

# 方法3: SignalGenerator的Z-score计算
print(f"\n方法3: SignalGenerator Z-score计算")
sg = SignalGenerator(window=60)

# 信号期的Z-score重新计算
new_z_scores = []
window = 60

for i in range(len(recalc_residuals)):
    if i >= signal_start_idx and i > window:  # 信号期且有足够数据 (修正：需要>window)
        # 修正：传入从开始到当前点的所有残差，让calculate_zscore自己选择窗口
        residuals_up_to_i = recalc_residuals[:i+1]
        z_score = sg.calculate_zscore(residuals_up_to_i, window)
        new_z_scores.append(z_score)
    else:
        new_z_scores.append(0.0)

new_z_scores = np.array(new_z_scores)

# 对比信号期的Z-score
signal_period_mask = np.arange(len(dates)) >= signal_start_idx
signal_original = original_z_scores[signal_period_mask]
signal_new = new_z_scores[signal_period_mask]
signal_dates = dates[signal_period_mask]

print(f"信号期Z-score对比:")
print(f"  原始方法非零Z-score: {np.sum(signal_original != 0)}")
print(f"  新方法非零Z-score: {np.sum(signal_new != 0)}")

# 找出差异较大的点
diff_mask = abs(signal_original - signal_new) > 0.1
if np.any(diff_mask):
    print(f"  差异>0.1的点数: {np.sum(diff_mask)}")
    print(f"  前10个差异点:")
    for i in np.where(diff_mask)[0][:10]:
        date_str = signal_dates.iloc[i].strftime('%Y-%m-%d')
        orig = signal_original[i]
        new = signal_new[i]
        print(f"    {date_str}: 原始={orig:.3f}, 新={new:.3f}, 差异={orig-new:.3f}")

# 方法4: 手动验证几个点的Z-score计算
print(f"\n方法4: 手动验证Z-score计算")

# 选择几个原始方法中|Z|>2的点进行验证
high_z_indices = np.where(abs(signal_original) > 2.0)[0]
if len(high_z_indices) > 0:
    print(f"  验证前5个|Z|>2的点:")
    for idx in high_z_indices[:5]:
        global_idx = signal_start_idx + idx
        date_str = dates[global_idx].strftime('%Y-%m-%d')
        
        # 手动计算Z-score
        if global_idx >= window:
            window_res = recalc_residuals[global_idx-window+1:global_idx+1]
            mean_manual = np.mean(window_res)
            std_manual = np.std(window_res, ddof=1)
            z_manual = (recalc_residuals[global_idx] - mean_manual) / std_manual if std_manual > 0 else 0
            
            print(f"    {date_str} (idx={global_idx}):")
            print(f"      原始Z: {signal_original[idx]:.3f}")
            print(f"      新方法Z: {signal_new[idx]:.3f}")
            print(f"      手动Z: {z_manual:.3f}")
            print(f"      残差: {recalc_residuals[global_idx]:.6f}")
            print(f"      窗口均值: {mean_manual:.6f}, 标准差: {std_manual:.6f}")

# 方法5: 分析残差序列和滚动统计
print(f"\n方法5: 分析滚动统计")

# 选择2023-07-27这个原始数据显示Z=-2.088的点
target_date = '2023-07-27'
target_idx = None
for i, date in enumerate(dates):
    if date.strftime('%Y-%m-%d') == target_date:
        target_idx = i
        break

if target_idx and target_idx >= window:
    print(f"分析 {target_date} (索引 {target_idx}):")
    
    # 原始数据的值
    orig_z = df.loc[target_idx, 'z_score']
    orig_residual = df.loc[target_idx, 'residual']
    
    # 窗口数据分析
    window_res = recalc_residuals[target_idx-window+1:target_idx+1]
    
    print(f"  原始Z-score: {orig_z:.3f}")
    print(f"  原始残差: {orig_residual:.6f}")
    print(f"  重计算残差: {recalc_residuals[target_idx]:.6f}")
    print(f"  窗口大小: {len(window_res)}")
    print(f"  窗口残差范围: {window_res.min():.6f} ~ {window_res.max():.6f}")
    print(f"  窗口均值: {np.mean(window_res):.6f}")
    print(f"  窗口标准差: {np.std(window_res, ddof=1):.6f}")
    
    # 新方法Z-score
    new_z = sg.calculate_zscore(window_res, window)
    print(f"  新方法Z-score: {new_z:.3f}")

print(f"\n" + "=" * 80)
print("结论分析")
print("=" * 80)
print("1. 检查残差计算是否一致")
print("2. 检查滚动窗口的起始位置")
print("3. 检查标准差计算 (ddof=0 vs ddof=1)")
print("4. 检查窗口边界处理")