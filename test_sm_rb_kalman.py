#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from lib.signal_generation import KalmanFilter2D
from lib.coint import CointegrationAnalyzer
import matplotlib.pyplot as plt

def main():
    print('加载数据...')
    df = load_all_symbols_data()
    
    # 使用SM-RB配对
    x_symbol, y_symbol = 'SM', 'RB'
    print(f'分析配对: {x_symbol}-{y_symbol}')
    
    # 获取协整结果确认方向
    analyzer = CointegrationAnalyzer(df)
    x_data = df[x_symbol].dropna()
    y_data = df[y_symbol].dropna()
    
    # 对齐数据
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates].values
    y_aligned = y_data.loc[common_dates].values
    
    # 运行协整检验
    coint_result = analyzer.test_pair_cointegration(x_aligned, y_aligned)
    print(f'协整p值: {coint_result["pvalue"]:.6f}')
    print(f'Beta (Y~X): {coint_result["beta"]:.6f}')
    print(f'R²: {coint_result["r_squared"]:.6f}')
    
    dates = common_dates
    
    print(f'数据点数量: {len(x_aligned)}')
    print(f'时间范围: {dates[0].strftime("%Y-%m-%d")} 到 {dates[-1].strftime("%Y-%m-%d")}')
    
    # 计算初始OLS参数用于Kalman初始化
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])  # 用前252天初始化
    initial_beta = float(reg.coef_[0])
    initial_c = float(reg.intercept_)
    initial_residuals = y_aligned[:252] - (initial_beta * x_aligned[:252] + initial_c)
    
    print(f'初始化参数: beta={initial_beta:.6f}, c={initial_c:.6f}')
    
    # 初始化Kalman滤波器
    kf = KalmanFilter2D(initial_beta, initial_c, initial_residuals, x_aligned[:252])
    
    # 运行对比测试
    window = 60
    min_periods = 252 + window  # 确保有足够数据
    
    kalman_betas = []
    ols_betas = []
    valid_dates = []
    
    print('开始对比测试...')
    for i in range(min_periods, len(x_aligned)):
        # Kalman更新
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        kf.update(x_t, y_t)
        kalman_betas.append(kf.beta)
        
        # 60天滚动OLS
        x_window = x_aligned[i-window+1:i+1]
        y_window = y_aligned[i-window+1:i+1]
        reg_window = LinearRegression()
        reg_window.fit(x_window.reshape(-1, 1), y_window)
        ols_betas.append(float(reg_window.coef_[0]))
        
        valid_dates.append(dates[i])
        
        if i % 200 == 0:
            print(f'处理进度: {i-min_periods+1}/{len(x_aligned)-min_periods} 完成')
    
    print('测试完成，分析结果...')
    
    # 转换为numpy数组便于分析
    kalman_betas = np.array(kalman_betas)
    ols_betas = np.array(ols_betas)
    
    # 计算统计指标
    kalman_std = np.std(kalman_betas)
    ols_std = np.std(ols_betas)
    stability_ratio = ols_std / kalman_std
    
    correlation = np.corrcoef(kalman_betas, ols_betas)[0, 1]
    
    # 计算创新序列的标准差（白化诊断）
    innovations = []
    for i, (x_val, y_val) in enumerate(zip(x_aligned[min_periods:], y_aligned[min_periods:])):
        predicted_y = kalman_betas[i] * x_val + kf.c  # 使用最后的c值近似
        innovation = y_val - predicted_y
        innovations.append(innovation)
    
    innovation_std = np.std(innovations)
    
    print(f'\n=== SM-RB配对 Kalman vs OLS 对比结果 ===')
    print(f'Kalman Beta - 均值: {np.mean(kalman_betas):.6f}, 标准差: {kalman_std:.6f}')
    print(f'OLS Beta - 均值: {np.mean(ols_betas):.6f}, 标准差: {ols_std:.6f}')
    print(f'稳定性改善倍数: {stability_ratio:.2f}x')
    print(f'相关性: {correlation:.4f}')
    print(f'创新序列标准差: {innovation_std:.6f}')
    whitening_status = "良好" if 0.9 <= innovation_std <= 1.1 else "需调整"
    print(f'创新白化状态: {whitening_status}')
    
    # 保存结果
    results_df = pd.DataFrame({
        'date': valid_dates,
        'kalman_beta': kalman_betas,
        'ols_beta': ols_betas,
        'beta_diff': np.abs(kalman_betas - ols_betas)
    })
    
    results_df.to_csv('sm_rb_kalman_ols_comparison.csv', index=False)
    print(f'\n结果已保存到: sm_rb_kalman_ols_comparison.csv')
    
    # 生成对比图
    plt.figure(figsize=(15, 10))
    
    # 子图1: Beta时间序列对比
    plt.subplot(2, 2, 1)
    plt.plot(valid_dates, kalman_betas, label='Kalman Beta', alpha=0.8, linewidth=1.5)
    plt.plot(valid_dates, ols_betas, label='OLS Beta (60天)', alpha=0.7, linewidth=1)
    plt.title('SM-RB配对: Kalman vs OLS Beta对比')
    plt.xlabel('日期')
    plt.ylabel('Beta值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: Beta差异
    plt.subplot(2, 2, 2)
    plt.plot(valid_dates, np.abs(kalman_betas - ols_betas), color='red', alpha=0.7)
    plt.title('Kalman与OLS的Beta绝对差异')
    plt.xlabel('日期')
    plt.ylabel('|Beta_Kalman - Beta_OLS|')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 相关性散点图
    plt.subplot(2, 2, 3)
    plt.scatter(ols_betas, kalman_betas, alpha=0.5)
    plt.plot([min(ols_betas), max(ols_betas)], [min(ols_betas), max(ols_betas)], 'r--', alpha=0.8)
    plt.xlabel('OLS Beta (60天)')
    plt.ylabel('Kalman Beta')
    plt.title(f'相关性散点图 (r={correlation:.4f})')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 稳定性对比（标准差）
    plt.subplot(2, 2, 4)
    methods = ['Kalman', 'OLS (60天)']
    stds = [kalman_std, ols_std]
    colors = ['blue', 'orange']
    bars = plt.bar(methods, stds, color=colors, alpha=0.7)
    plt.ylabel('Beta标准差')
    plt.title(f'稳定性对比 (改善: {stability_ratio:.2f}x)')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, std in zip(bars, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std*0.01, 
                f'{std:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sm_rb_kalman_ols_comparison.png', dpi=150, bbox_inches='tight')
    print('对比图已保存到: sm_rb_kalman_ols_comparison.png')
    
    return results_df, {
        'kalman_std': kalman_std,
        'ols_std': ols_std,
        'stability_ratio': stability_ratio,
        'correlation': correlation,
        'innovation_std': innovation_std,
        'whitening_status': whitening_status
    }

if __name__ == '__main__':
    results_df, stats = main()