#!/usr/bin/env python3
"""
分析Kalman滤波的动态Beta和信号生成
检查为什么没有产生交易信号
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.signal_generation import KalmanFilter1D, SignalGenerator
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS

# 配置
TOP_PAIRS = [
    ('RB0', 'I0', 0.432353),   # p1y=0.000005
    ('HC0', 'I0', 0.439301),   # p1y=0.000006
    ('PB0', 'ZN0', 0.191787),  # p1y=0.000282
    ('AL0', 'CU0', 0.869543),  # p1y=0.000614
    ('SS0', 'NI0', 0.400092),  # p1y=0.003225
]

# 时间配置
DATA_START = '2019-01-01'
DATA_END = '2024-08-20'
SIGNAL_START = '2023-01-01'

print("=" * 80)
print("Kalman滤波Beta分析")
print("=" * 80)

# 加载数据
symbols = list(set([s for pair in TOP_PAIRS for s in pair[:2]]))
data = load_data(
    symbols=symbols,
    start_date=DATA_START,
    end_date=DATA_END,
    columns=['close'],
    log_price=True
)

# 提取信号期数据
signal_data = data[data.index >= SIGNAL_START]
print(f"\n信号期数据: {len(signal_data)} 天")

# 分析每个配对
for symbol_x, symbol_y, beta_4y in TOP_PAIRS:
    print("\n" + "=" * 80)
    print(f"配对分析: {symbol_x}-{symbol_y}")
    print("-" * 60)
    
    # 获取价格数据
    x_full = data[f"{symbol_x}_close"].values
    y_full = data[f"{symbol_y}_close"].values
    x_signal = signal_data[f"{symbol_x}_close"].values
    y_signal = signal_data[f"{symbol_y}_close"].values
    
    # 1. 使用2022年数据估计初始Beta
    train_data = data[(data.index >= '2022-01-01') & (data.index <= '2022-12-31')]
    x_train = train_data[f"{symbol_x}_close"].values
    y_train = train_data[f"{symbol_y}_close"].values
    
    X = add_constant(x_train)
    model = OLS(y_train, X).fit()
    beta_initial = model.params[1]
    
    print(f"\n初始Beta (2022年OLS): {beta_initial:.6f}")
    print(f"4年Beta (协整分析): {beta_4y:.6f}")
    
    # 2. 初始化Kalman滤波器
    kf = KalmanFilter1D(
        initial_beta=beta_initial,
        Q=1e-5,  # 过程噪声（Beta变化速度）
        R=0.001  # 测量噪声
    )
    
    # 3. 运行Kalman滤波
    kalman_betas = []
    residuals = []
    
    for i in range(len(x_signal)):
        if i == 0:
            beta_t = beta_initial
        else:
            result = kf.update(y_signal[i], x_signal[i])
            beta_t = result['beta']
        
        kalman_betas.append(beta_t)
        residual = y_signal[i] - beta_t * x_signal[i]
        residuals.append(residual)
    
    # 4. 计算Z-score
    z_scores = []
    window = 60
    
    for i in range(len(residuals)):
        if i < window:
            z_scores.append(0)
        else:
            window_residuals = residuals[i-window:i]
            mean = np.mean(window_residuals)
            std = np.std(window_residuals)
            if std > 0:
                z_score = (residuals[i] - mean) / std
            else:
                z_score = 0
            z_scores.append(z_score)
    
    # 5. 生成交易信号
    generator = SignalGenerator(
        window=60,
        z_open=2.0,
        z_close=0.5
    )
    
    signals = []
    position = None
    days_held = 0
    
    for i, z_score in enumerate(z_scores):
        signal_str = generator.generate_signal(
            z_score=z_score,
            position=position,
            days_held=days_held,
            max_days=30
        )
        
        if signal_str == 'long':
            signals.append(1)
            position = 'long'
            days_held = 1
        elif signal_str == 'short':
            signals.append(-1)
            position = 'short'
            days_held = 1
        elif signal_str == 'close':
            signals.append(0)
            position = None
            days_held = 0
        else:  # hold
            signals.append(1 if position == 'long' else (-1 if position == 'short' else 0))
            if position:
                days_held += 1
    
    # 6. 统计分析
    print(f"\nKalman Beta统计:")
    print(f"  最小值: {min(kalman_betas):.6f}")
    print(f"  最大值: {max(kalman_betas):.6f}")
    print(f"  均值: {np.mean(kalman_betas):.6f}")
    print(f"  标准差: {np.std(kalman_betas):.6f}")
    print(f"  变化幅度: {(max(kalman_betas) - min(kalman_betas))/beta_initial*100:.2f}%")
    
    print(f"\nZ-score统计:")
    z_scores_valid = [z for z in z_scores if z != 0]
    if z_scores_valid:
        print(f"  最小值: {min(z_scores_valid):.3f}")
        print(f"  最大值: {max(z_scores_valid):.3f}")
        print(f"  均值: {np.mean(z_scores_valid):.3f}")
        print(f"  标准差: {np.std(z_scores_valid):.3f}")
        print(f"  超过2.0的次数: {sum(1 for z in z_scores_valid if abs(z) > 2.0)}")
        print(f"  超过1.5的次数: {sum(1 for z in z_scores_valid if abs(z) > 1.5)}")
    
    print(f"\n信号统计:")
    signal_counts = pd.Series(signals).value_counts()
    print(f"  做多信号: {signal_counts.get(1, 0)}")
    print(f"  做空信号: {signal_counts.get(-1, 0)}")
    print(f"  无持仓: {signal_counts.get(0, 0)}")
    
    # 7. 保存详细数据到CSV
    output_dir = project_root / 'output' / 'kalman_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_analysis = pd.DataFrame({
        'date': signal_data.index,
        'x_price': x_signal,
        'y_price': y_signal,
        'kalman_beta': kalman_betas,
        'residual': residuals,
        'z_score': z_scores,
        'signal': signals
    })
    
    output_file = output_dir / f'{symbol_x}_{symbol_y}_kalman_analysis.csv'
    df_analysis.to_csv(output_file, index=False)
    print(f"\n详细数据已保存至: {output_file}")
    
    # 8. 创建可视化
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Beta变化
    axes[0].plot(signal_data.index, kalman_betas, label='Kalman Beta', linewidth=1)
    axes[0].axhline(y=beta_initial, color='r', linestyle='--', label=f'初始Beta={beta_initial:.3f}')
    axes[0].axhline(y=beta_4y, color='g', linestyle='--', label=f'4年Beta={beta_4y:.3f}')
    axes[0].set_ylabel('Beta')
    axes[0].set_title(f'{symbol_x}-{symbol_y} Kalman滤波Beta动态变化')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 残差
    axes[1].plot(signal_data.index, residuals, label='残差', linewidth=1, color='orange')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('残差')
    axes[1].set_title('价差残差')
    axes[1].grid(True, alpha=0.3)
    
    # Z-score
    axes[2].plot(signal_data.index, z_scores, label='Z-score', linewidth=1, color='blue')
    axes[2].axhline(y=2, color='r', linestyle='--', label='开仓阈值')
    axes[2].axhline(y=-2, color='r', linestyle='--')
    axes[2].axhline(y=0.5, color='g', linestyle='--', label='平仓阈值')
    axes[2].axhline(y=-0.5, color='g', linestyle='--')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Z-score')
    axes[2].set_title('标准化Z-score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 信号
    axes[3].plot(signal_data.index, signals, label='交易信号', linewidth=1, color='purple')
    axes[3].set_ylabel('信号')
    axes[3].set_title('交易信号 (1=做多, -1=做空, 0=无持仓)')
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_file = output_dir / f'{symbol_x}_{symbol_y}_kalman_chart.png'
    plt.savefig(chart_file, dpi=100)
    plt.close()
    print(f"图表已保存至: {chart_file}")

print("\n" + "=" * 80)
print("问题诊断")
print("-" * 60)

print("\n可能的原因:")
print("1. Z-score阈值设置过高（2.0）")
print("2. Kalman滤波参数Q太小，Beta变化太慢")
print("3. 滚动窗口(60天)可能太长")
print("4. 市场处于趋势行情，配对关系暂时失效")
print("\n建议调整:")
print("- 降低开仓阈值到1.5")
print("- 增大Q值到1e-4，让Beta更灵活")
print("- 缩短滚动窗口到30天")

print("\n分析完成！")