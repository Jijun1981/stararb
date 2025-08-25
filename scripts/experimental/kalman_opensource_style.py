#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照开源资源的思路重新实现Kalman Filter
参考: china-futures-pairs-trading/Kalman Filter.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def KalmanFilterRegression_OpenSource(x, y, delta=1e-3, obs_cov=1.0):
    """
    按开源代码的思路实现Kalman滤波回归
    
    Args:
        x, y: 价格序列（不是对数价格！）
        delta: 过程噪声参数
        obs_cov: 观测噪声
    """
    # 转换数据格式
    x_vals = x.values if hasattr(x, 'values') else x
    y_vals = y.values if hasattr(y, 'values') else y
    
    n = len(x_vals)
    
    # 参数设置（按开源代码）
    trans_cov = delta / (1 - delta) * np.eye(2)  # Q矩阵
    
    # 手动实现Kalman滤波（避免依赖pykalman）
    # 状态: [α, β]
    state = np.zeros(2)  # [α, β]
    P = np.ones((2, 2))  # 初始协方差
    
    # 记录结果
    state_means = np.zeros((n, 2))
    
    for i in range(n):
        # 观测矩阵 H = [1, x_t]
        H = np.array([1.0, x_vals[i]]).reshape(1, 2)
        
        # 预测步
        P_pred = P + trans_cov
        y_pred = H @ state  # α + β*x_t
        
        # 创新
        v = y_vals[i] - y_pred
        S = H @ P_pred @ H.T + obs_cov
        S = max(S.item(), 1e-12)
        
        # 更新步
        K = P_pred @ H.T / S
        state = state + K.ravel() * v
        P = P_pred - K.reshape(-1, 1) @ H @ P_pred
        
        # 记录
        state_means[i] = state.copy()
    
    return state_means

def calculate_noise_ratio_opensource(x, y, state_means):
    """
    按开源代码计算noise ratio
    """
    # 计算预测值
    y_hat = state_means[:, 0] + state_means[:, 1] * x  # α + β*x
    
    # R: 残差方差
    R_error = y_hat - y
    R = np.var(R_error)
    
    # Q: β变化的方差
    Beta = pd.Series(state_means[:, 1])  # β序列
    Beta_1 = Beta.shift(1)
    Q_error = Beta - Beta_1
    Q = np.var(Q_error.dropna())
    
    # Noise ratio
    noise_ratio = Q / R if R > 0 else np.inf
    
    return noise_ratio, R, Q, R_error, Q_error.dropna()

def test_opensource_approach():
    """
    测试开源方法的效果
    """
    print("=== 按开源资源方法测试AL-ZN配对 ===")
    
    # 加载原始价格数据（不取对数）
    df = load_all_symbols_data()
    
    # 使用原始价格而不是对数价格
    x_data = df['AL'].dropna()  # 原始价格
    y_data = df['ZN'].dropna()  # 原始价格
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates]
    y_aligned = y_data.loc[common_dates]
    
    print(f'数据: {len(x_aligned)}个点')
    print(f'X (AL): 均值={np.mean(x_aligned):.2f}, 标准差={np.std(x_aligned):.2f}')
    print(f'Y (ZN): 均值={np.mean(y_aligned):.2f}, 标准差={np.std(y_aligned):.2f}')
    
    # 测试不同的delta参数
    deltas_to_test = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    obs_covs_to_test = [0.1, 1.0, 10.0]
    
    print(f'\\n=== 参数搜索 ===')
    print('delta     obs_cov   noise_ratio   R        Q        相关性')
    print('-' * 65)
    
    best_result = None
    best_noise_ratio = np.inf
    
    for delta in deltas_to_test:
        for obs_cov in obs_covs_to_test:
            # 运行Kalman滤波
            state_means = KalmanFilterRegression_OpenSource(x_aligned, y_aligned, delta, obs_cov)
            
            # 计算noise ratio
            noise_ratio, R, Q, R_error, Q_error = calculate_noise_ratio_opensource(x_aligned, y_aligned, state_means)
            
            # 与60天OLS对比
            window = 60
            ols_betas = []
            kalman_betas = state_means[252:, 1]  # β序列
            
            for i in range(252 + window, len(x_aligned)):
                x_window = x_aligned.iloc[i-window+1:i+1]
                y_window = y_aligned.iloc[i-window+1:i+1]
                reg = LinearRegression()
                reg.fit(x_window.values.reshape(-1, 1), y_window.values)
                ols_betas.append(reg.coef_[0])
            
            # 对齐长度计算相关性
            min_len = min(len(kalman_betas), len(ols_betas))
            if min_len > 10:
                correlation = np.corrcoef(kalman_betas[-min_len:], ols_betas[-min_len:])[0, 1]
            else:
                correlation = 0
            
            print(f'{delta:.0e}     {obs_cov:5.1f}     {noise_ratio:8.6f}  {R:8.2f}  {Q:8.6f}  {correlation:8.4f}')
            
            # 记录最佳结果（基于合理的noise ratio）
            if 0.001 <= noise_ratio <= 0.1 and abs(correlation) > 0.3:  # 寻找合理的noise ratio
                if noise_ratio < best_noise_ratio or best_result is None:
                    best_noise_ratio = noise_ratio
                    best_result = {
                        'delta': delta,
                        'obs_cov': obs_cov,
                        'noise_ratio': noise_ratio,
                        'R': R,
                        'Q': Q,
                        'correlation': correlation,
                        'state_means': state_means,
                        'kalman_betas': kalman_betas,
                        'ols_betas': ols_betas
                    }
    
    if best_result:
        print(f'\\n=== 最佳参数结果 ===')
        print(f'delta: {best_result["delta"]:.0e}')
        print(f'obs_cov: {best_result["obs_cov"]:.1f}')
        print(f'noise_ratio: {best_result["noise_ratio"]:.6f}')
        print(f'相关性: {best_result["correlation"]:.4f}')
        
        # 计算创新白化效果
        state_means = best_result['state_means']
        innovations = []
        
        # 重新计算标准化创新
        for i in range(1, len(x_aligned)):
            alpha, beta = state_means[i-1]  # 使用前一步的状态
            y_pred = alpha + beta * x_aligned.iloc[i]
            v = y_aligned.iloc[i] - y_pred
            
            # 简化的创新标准化（用R作为方差估计）
            z = v / np.sqrt(best_result['R'])
            innovations.append(z)
        
        innovations = np.array(innovations)
        z_mean = np.mean(innovations)
        z_std = np.std(innovations)
        
        print(f'\\n=== 创新白化检查 ===')
        print(f'mean(z): {z_mean:.4f}')
        print(f'std(z): {z_std:.4f}')
        whitening_ok = abs(z_mean) <= 0.1 and 0.8 <= z_std <= 1.2
        print(f'白化状态: {"✅" if whitening_ok else "❌"}')
        
        # 生成对比图
        plot_opensource_results(x_aligned, y_aligned, best_result, innovations)
        
        return best_result
    else:
        print('\\n❌ 未找到合适的参数组合')
        return None

def plot_opensource_results(x_data, y_data, result, innovations):
    """
    绘制开源方法的结果
    """
    state_means = result['state_means']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 子图1: α和β的时间序列
    axes[0,0].plot(state_means[:, 0], label='α (截距)', color='blue')
    axes[0,0].set_title('Kalman α (截距) 时间序列')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(state_means[:, 1], label='β (斜率)', color='red')
    axes[0,1].set_title('Kalman β (斜率) 时间序列')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 子图3: 原始价格对比
    axes[0,2].plot(x_data.values, label='AL (原始价格)', alpha=0.7)
    axes[0,2].plot(y_data.values, label='ZN (原始价格)', alpha=0.7)
    axes[0,2].set_title('原始价格序列')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 子图4: 创新分布
    axes[1,0].hist(innovations, bins=50, alpha=0.7, density=True, color='green')
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[1,0].set_title(f'创新分布 (std={np.std(innovations):.3f})')
    axes[1,0].set_xlabel('标准化创新')
    axes[1,0].grid(True, alpha=0.3)
    
    # 子图5: 残差时间序列
    residuals = y_data.values - (state_means[:, 0] + state_means[:, 1] * x_data.values)
    axes[1,1].plot(residuals, alpha=0.7, linewidth=0.5)
    axes[1,1].axhline(0, color='red', linestyle='--', alpha=0.8)
    axes[1,1].set_title('残差时间序列')
    axes[1,1].set_ylabel('残差')
    axes[1,1].grid(True, alpha=0.3)
    
    # 子图6: Noise ratio信息
    axes[1,2].text(0.1, 0.8, f'参数设置:', fontsize=12, weight='bold', transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.7, f'delta = {result["delta"]:.0e}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.6, f'obs_cov = {result["obs_cov"]:.1f}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.5, f'noise_ratio = {result["noise_ratio"]:.6f}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.4, f'相关性 = {result["correlation"]:.4f}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.3, f'R = {result["R"]:.2f}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].text(0.1, 0.2, f'Q = {result["Q"]:.6f}', fontsize=10, transform=axes[1,2].transAxes)
    axes[1,2].axis('off')
    axes[1,2].set_title('开源方法参数总结')
    
    plt.tight_layout()
    plt.savefig('opensource_kalman_results.png', dpi=150, bbox_inches='tight')
    print('开源方法结果图已保存: opensource_kalman_results.png')

if __name__ == '__main__':
    result = test_opensource_approach()