#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用最终优化参数进行完整的Kalman滤波测试
与OLS滚动对比，并进行ADF平稳性检验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def run_kalman_with_optimal_params():
    """
    使用优化参数运行完整的Kalman滤波
    """
    print("=== 使用最终优化参数运行Kalman滤波 ===")
    
    # 加载数据
    df = load_all_symbols_data()
    x_data = np.log(df['AL'].dropna())
    y_data = np.log(df['ZN'].dropna())
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates]
    y_aligned = y_data.loc[common_dates]
    
    print(f'数据范围: {x_aligned.index[0].date()} 到 {x_aligned.index[-1].date()}')
    print(f'总样本数: {len(x_aligned)}')
    
    # 最终优化参数（来自kalman_final_perfect.py的结果）
    # 用前500天估计更稳定的β
    reg = LinearRegression()
    reg.fit(x_aligned[:500].values.reshape(-1, 1), y_aligned[:500].values)
    beta0 = reg.coef_[0]
    c0 = reg.intercept_
    
    # 调整截距以消除系统性偏移
    innovations_test = []
    for i in range(252, 352):
        if i < len(x_aligned):
            v = y_aligned.iloc[i] - (beta0 * x_aligned.iloc[i] + c0)
            innovations_test.append(v)
    
    v_mean = np.mean(innovations_test)
    c0_adjusted = c0 + v_mean
    
    # 最终参数设定
    v_var_adj = np.var([y_aligned.iloc[i] - (beta0 * x_aligned.iloc[i] + c0_adjusted) 
                       for i in range(252, 352) if i < len(x_aligned)])
    
    target_S = v_var_adj * 1.1
    R = target_S * 0.85
    avg_x = np.mean(x_aligned.iloc[252:352])
    P_target = target_S * 0.15 / (avg_x ** 2)
    Q_beta = P_target * 0.005
    Q_c = R * 1e-6
    
    print(f'Kalman参数设定:')
    print(f'  初始β: {beta0:.6f}')
    print(f'  初始c: {c0_adjusted:.6f}')
    print(f'  R: {R:.6f}')
    print(f'  Q_β: {Q_beta:.8f}')
    print(f'  Q_c: {Q_c:.8f}')
    
    # 运行Kalman滤波
    beta_kf = beta0
    c_kf = c0_adjusted
    P = np.diag([P_target, P_target * 0.5])
    Q = np.diag([Q_beta, Q_c])
    
    # 存储结果
    kalman_results = []
    start_idx = 500  # 从第500天开始，前面用于参数估计
    
    for i in range(start_idx, len(x_aligned)):
        x_t = x_aligned.iloc[i]
        y_t = y_aligned.iloc[i]
        date = x_aligned.index[i]
        
        # 预测
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        y_pred = beta_kf * x_t + c_kf
        
        # 创新和残差
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        z = v / np.sqrt(S)
        
        # 存储结果
        kalman_results.append({
            'date': date,
            'x': x_t,
            'y': y_t,
            'beta': beta_kf,
            'c': c_kf,
            'y_pred': y_pred,
            'residual': v,  # Kalman残差
            'z_score': z,
            'S': S
        })
        
        # 更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta_kf += update_vec[0]
        c_kf += update_vec[1]
        
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
    
    kalman_df = pd.DataFrame(kalman_results).set_index('date')
    
    print(f'Kalman滤波完成，处理{len(kalman_df)}个样本')
    print(f'β变化: {kalman_df.beta.iloc[0]:.6f} → {kalman_df.beta.iloc[-1]:.6f}')
    print(f'残差统计: mean={kalman_df.residual.mean():.6f}, std={kalman_df.residual.std():.6f}')
    print(f'Z-score统计: mean={kalman_df.z_score.mean():.4f}, std={kalman_df.z_score.std():.4f}')
    
    return kalman_df, x_aligned, y_aligned

def run_ols_rolling_comparison(kalman_df, x_aligned, y_aligned, window=60):
    """
    运行OLS滚动窗口对比
    """
    print(f'\n=== OLS滚动窗口对比 (窗口={window}天) ===')
    
    ols_results = []
    start_idx = 500  # 与Kalman保持一致
    
    for i in range(start_idx + window, len(x_aligned)):
        # 滚动窗口数据
        x_window = x_aligned.iloc[i-window:i]
        y_window = y_aligned.iloc[i-window:i]
        date = x_aligned.index[i]
        
        # OLS回归
        reg = LinearRegression()
        reg.fit(x_window.values.reshape(-1, 1), y_window.values)
        beta_ols = reg.coef_[0]
        c_ols = reg.intercept_
        
        # 当期预测和残差
        if i < len(x_aligned):
            x_t = x_aligned.iloc[i]
            y_t = y_aligned.iloc[i]
            y_pred_ols = beta_ols * x_t + c_ols
            residual_ols = y_t - y_pred_ols
            
            ols_results.append({
                'date': date,
                'beta_ols': beta_ols,
                'c_ols': c_ols,
                'y_pred_ols': y_pred_ols,
                'residual_ols': residual_ols
            })
    
    ols_df = pd.DataFrame(ols_results).set_index('date')
    
    # 对齐数据进行比较
    common_dates = kalman_df.index.intersection(ols_df.index)
    kalman_aligned = kalman_df.loc[common_dates]
    ols_aligned = ols_df.loc[common_dates]
    
    # Beta相关性
    beta_corr = np.corrcoef(kalman_aligned.beta, ols_aligned.beta_ols)[0, 1]
    
    print(f'OLS滚动完成，处理{len(ols_aligned)}个样本')
    print(f'Beta相关性: {beta_corr:.4f}')
    print(f'OLS残差统计: mean={ols_aligned.residual_ols.mean():.6f}, std={ols_aligned.residual_ols.std():.6f}')
    
    return ols_aligned, kalman_aligned, beta_corr

def adf_stationarity_test(kalman_aligned, ols_aligned):
    """
    对Kalman和OLS残差进行ADF平稳性检验
    """
    print(f'\n=== ADF平稳性检验 ===')
    
    # Kalman残差ADF检验
    kalman_residuals = kalman_aligned.residual.dropna()
    adf_kalman = adfuller(kalman_residuals, autolag='AIC')
    
    print(f'Kalman残差ADF检验:')
    print(f'  ADF统计量: {adf_kalman[0]:.6f}')
    print(f'  p值: {adf_kalman[1]:.6f}')
    print(f'  临界值: {dict(adf_kalman[4])}')
    kalman_stationary = adf_kalman[1] < 0.05
    print(f'  平稳性: {"✅ 平稳" if kalman_stationary else "❌ 非平稳"}')
    
    # OLS残差ADF检验
    ols_residuals = ols_aligned.residual_ols.dropna()
    adf_ols = adfuller(ols_residuals, autolag='AIC')
    
    print(f'\nOLS残差ADF检验:')
    print(f'  ADF统计量: {adf_ols[0]:.6f}')
    print(f'  p值: {adf_ols[1]:.6f}')
    print(f'  临界值: {dict(adf_ols[4])}')
    ols_stationary = adf_ols[1] < 0.05
    print(f'  平稳性: {"✅ 平稳" if ols_stationary else "❌ 非平稳"}')
    
    # 对比总结
    print(f'\n📊 平稳性对比总结:')
    print(f'  Kalman滤波: {"✅ 残差平稳" if kalman_stationary else "❌ 残差非平稳"} (p={adf_kalman[1]:.6f})')
    print(f'  OLS滚动: {"✅ 残差平稳" if ols_stationary else "❌ 残差非平稳"} (p={adf_ols[1]:.6f})')
    
    if kalman_stationary and not ols_stationary:
        print('🏆 Kalman滤波在残差平稳性方面优于OLS滚动！')
    elif ols_stationary and not kalman_stationary:
        print('⚠️ OLS滚动在残差平稳性方面优于Kalman滤波')
    elif kalman_stationary and ols_stationary:
        print('✅ 两种方法的残差都平稳')
    else:
        print('❌ 两种方法的残差都不平稳')
    
    return {
        'kalman_adf': adf_kalman,
        'ols_adf': adf_ols,
        'kalman_stationary': kalman_stationary,
        'ols_stationary': ols_stationary
    }

def plot_comprehensive_comparison(kalman_aligned, ols_aligned, beta_corr, adf_results):
    """
    绘制全面对比图表
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 子图1: Beta对比
    axes[0,0].plot(kalman_aligned.index, kalman_aligned.beta, label='Kalman β', color='blue', alpha=0.8)
    axes[0,0].plot(ols_aligned.index, ols_aligned.beta_ols, label='OLS β (60天)', color='red', alpha=0.7)
    axes[0,0].set_title(f'β系数对比 (相关性={beta_corr:.4f})')
    axes[0,0].set_ylabel('Beta')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 子图2: Beta散点图
    axes[0,1].scatter(kalman_aligned.beta, ols_aligned.beta_ols, alpha=0.6, s=20)
    axes[0,1].plot([kalman_aligned.beta.min(), kalman_aligned.beta.max()], 
                   [kalman_aligned.beta.min(), kalman_aligned.beta.max()], 'r--', alpha=0.8)
    axes[0,1].set_xlabel('Kalman β')
    axes[0,1].set_ylabel('OLS β')
    axes[0,1].set_title(f'β相关性散点图 (r={beta_corr:.4f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # 子图3: 残差对比
    axes[1,0].plot(kalman_aligned.index, kalman_aligned.residual, label='Kalman残差', alpha=0.7, linewidth=0.8)
    axes[1,0].plot(ols_aligned.index, ols_aligned.residual_ols, label='OLS残差', alpha=0.7, linewidth=0.8)
    axes[1,0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('残差时间序列对比')
    axes[1,0].set_ylabel('残差')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 子图4: 残差分布对比
    axes[1,1].hist(kalman_aligned.residual, bins=30, alpha=0.7, density=True, label='Kalman残差', color='blue')
    axes[1,1].hist(ols_aligned.residual_ols, bins=30, alpha=0.7, density=True, label='OLS残差', color='red')
    axes[1,1].set_title('残差分布对比')
    axes[1,1].set_xlabel('残差值')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 子图5: Z-score分析
    axes[2,0].plot(kalman_aligned.index, kalman_aligned.z_score, alpha=0.8, linewidth=0.8, color='green')
    axes[2,0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[2,0].axhline(2, color='orange', linestyle=':', alpha=0.7)
    axes[2,0].axhline(-2, color='orange', linestyle=':', alpha=0.7)
    axes[2,0].set_title(f'Kalman Z-score (mean={kalman_aligned.z_score.mean():.3f}, std={kalman_aligned.z_score.std():.3f})')
    axes[2,0].set_ylabel('Z-score')
    axes[2,0].grid(True, alpha=0.3)
    
    # 子图6: ADF检验结果总结
    axes[2,1].axis('off')
    
    # 创建ADF结果表格
    kalman_status = "✅ 平稳" if adf_results['kalman_stationary'] else "❌ 非平稳"
    ols_status = "✅ 平稳" if adf_results['ols_stationary'] else "❌ 非平稳"
    
    axes[2,1].text(0.1, 0.9, 'ADF平稳性检验结果', fontsize=14, weight='bold', transform=axes[2,1].transAxes)
    
    axes[2,1].text(0.1, 0.8, 'Kalman滤波残差:', fontsize=12, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.75, f'  ADF统计量: {adf_results["kalman_adf"][0]:.4f}', fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.70, f'  p值: {adf_results["kalman_adf"][1]:.6f}', fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.65, f'  结果: {kalman_status}', fontsize=11, weight='bold', 
                   color='green' if adf_results['kalman_stationary'] else 'red', transform=axes[2,1].transAxes)
    
    axes[2,1].text(0.1, 0.55, 'OLS滚动残差:', fontsize=12, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.50, f'  ADF统计量: {adf_results["ols_adf"][0]:.4f}', fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.45, f'  p值: {adf_results["ols_adf"][1]:.6f}', fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.40, f'  结果: {ols_status}', fontsize=11, weight='bold',
                   color='green' if adf_results['ols_stationary'] else 'red', transform=axes[2,1].transAxes)
    
    # 添加其他统计信息
    axes[2,1].text(0.1, 0.25, f'Beta相关性: {beta_corr:.4f}', fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.20, f'Kalman β变化: {abs(kalman_aligned.beta.iloc[-1] - kalman_aligned.beta.iloc[0])/kalman_aligned.beta.iloc[0]*100:.1f}%', 
                   fontsize=11, transform=axes[2,1].transAxes)
    axes[2,1].text(0.1, 0.15, f'样本数: {len(kalman_aligned)}', fontsize=11, transform=axes[2,1].transAxes)
    
    axes[2,1].set_title('统计检验总结')
    
    plt.tight_layout()
    plt.savefig('kalman_ols_comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print('📊 全面对比图表已保存: kalman_ols_comprehensive_comparison.png')

if __name__ == '__main__':
    # 1. 运行Kalman滤波
    kalman_df, x_aligned, y_aligned = run_kalman_with_optimal_params()
    
    # 2. OLS滚动对比
    ols_aligned, kalman_aligned, beta_corr = run_ols_rolling_comparison(kalman_df, x_aligned, y_aligned)
    
    # 3. ADF平稳性检验
    adf_results = adf_stationarity_test(kalman_aligned, ols_aligned)
    
    # 4. 绘制全面对比图表
    plot_comprehensive_comparison(kalman_aligned, ols_aligned, beta_corr, adf_results)
    
    print(f'\n🎯 完整分析完成!')
    print(f'✅ Kalman滤波参数工程化成功')
    print(f'📈 与OLS对比完成，Beta相关性: {beta_corr:.4f}')
    print(f'📊 ADF平稳性检验完成')