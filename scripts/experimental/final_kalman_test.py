#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终Kalman测试 - 使用调优后的参数进行完整对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def final_kalman_ols_comparison():
    """
    使用调优后的最佳参数进行Kalman vs OLS完整对比
    """
    print("=== 最终Kalman vs OLS对比测试 ===")
    
    # 加载对数价格数据
    df = load_all_symbols_data()
    x_data = np.log(df['SM'].dropna())
    y_data = np.log(df['RB'].dropna())
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates].values
    y_aligned = y_data.loc[common_dates].values
    dates = common_dates
    
    print(f'数据: {len(x_aligned)}个点 ({dates[0].strftime("%Y-%m-%d")} 至 {dates[-1].strftime("%Y-%m-%d")})')
    
    # 初始OLS参数
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])
    beta0 = reg.coef_[0]
    c0 = reg.intercept_
    residuals = y_aligned[:252] - (beta0 * x_aligned[:252] + c0)
    
    print(f'初始OLS: β={beta0:.6f}, c={c0:.6f}, 残差std={np.std(residuals):.6f}')
    
    # 最佳调优参数（从网格搜索得到）
    target_S = 2.809e-03
    R = target_S * 0.5  # R_scale = 0.5
    eta_beta = 1e-6     # eta_β = 1e-6
    
    x_var = np.var(x_aligned[:252])
    q_beta = eta_beta * R / x_var
    q_c = eta_beta * R * 0.1
    
    print(f'最佳参数: R={R:.2e}, Q_β={q_beta:.2e}, Q_c={q_c:.2e}')
    
    # 运行完整对比测试
    window = 60
    start_idx = 252 + window
    
    # 初始化Kalman
    beta_kf = beta0
    c_kf = c0
    P = np.diag([1e-6, 1e-8])
    Q = np.diag([q_beta, q_c])
    
    # 记录数组
    kalman_betas = []
    ols_betas = []
    valid_dates = []
    z_scores = []
    S_values = []
    
    print('开始对比测试...')
    
    for i in range(start_idx, len(x_aligned)):
        # Kalman更新
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        
        # 预测步
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        state = np.array([beta_kf, c_kf])
        y_pred = float(H @ state)
        
        # 创新和更新
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        z = v / np.sqrt(S)
        
        # Kalman增益和状态更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta_kf += update_vec[0]
        c_kf += update_vec[1]
        
        # 协方差更新
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
        
        # 记录Kalman结果
        kalman_betas.append(beta_kf)
        z_scores.append(z)
        S_values.append(S)
        
        # OLS滚动窗口
        x_window = x_aligned[i-window+1:i+1]
        y_window = y_aligned[i-window+1:i+1]
        reg_window = LinearRegression()
        reg_window.fit(x_window.reshape(-1, 1), y_window)
        ols_betas.append(float(reg_window.coef_[0]))
        
        valid_dates.append(dates[i])
        
        if i % 200 == 0:
            print(f'进度: {i-start_idx+1}/{len(x_aligned)-start_idx}')
    
    # 转换为numpy数组
    kalman_betas = np.array(kalman_betas)
    ols_betas = np.array(ols_betas)
    z_scores = np.array(z_scores)
    
    # 计算统计指标
    kf_mean = np.mean(kalman_betas)
    kf_std = np.std(kalman_betas)
    ols_mean = np.mean(ols_betas)
    ols_std = np.std(ols_betas)
    
    stability_ratio = ols_std / kf_std if kf_std > 0 else 0
    correlation = np.corrcoef(kalman_betas, ols_betas)[0, 1]
    
    # 创新白化检查
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    whitening_ok = abs(z_mean) <= 0.1 and 0.9 <= z_std <= 1.1
    
    print('\\n=== 最终对比结果 ===')
    print(f'Kalman Beta - 均值: {kf_mean:.6f}, 标准差: {kf_std:.6f}')
    print(f'OLS Beta - 均值: {ols_mean:.6f}, 标准差: {ols_std:.6f}')
    print(f'稳定性改善: {stability_ratio:.2f}x')
    print(f'相关性: {correlation:.4f}')
    print(f'创新白化: mean(z)={z_mean:.4f}, std(z)={z_std:.4f}')
    print(f'白化状态: {"✅通过" if whitening_ok else "❌需调整"}')
    
    # 专家目标检查
    targets_met = {
        'innovation_whitening': whitening_ok,
        'correlation': correlation >= 0.6,
        'stability': 2.0 <= stability_ratio <= 5.0
    }
    
    all_targets_met = all(targets_met.values())
    
    print(f'\\n=== 专家目标达成情况 ===')
    print(f'创新白化: {"✅" if targets_met["innovation_whitening"] else "❌"}')
    print(f'相关性≥0.6: {"✅" if targets_met["correlation"] else "❌"}')
    print(f'稳定性2-5x: {"✅" if targets_met["stability"] else "❌"}')
    print(f'总体评估: {"✅全部达标" if all_targets_met else "❌部分未达标"}')
    
    # 保存结果
    results_df = pd.DataFrame({
        'date': valid_dates,
        'kalman_beta': kalman_betas,
        'ols_beta': ols_betas,
        'z_score': z_scores
    })
    results_df.to_csv('final_kalman_ols_comparison.csv', index=False)
    print(f'\\n结果已保存: final_kalman_ols_comparison.csv')
    
    # 生成对比图
    plt.figure(figsize=(16, 12))
    
    # 子图1: Beta时间序列
    plt.subplot(2, 3, 1)
    plt.plot(valid_dates, kalman_betas, label=f'Kalman (std={kf_std:.6f})', alpha=0.8, linewidth=1.5)
    plt.plot(valid_dates, ols_betas, label=f'OLS-60 (std={ols_std:.6f})', alpha=0.7, linewidth=1)
    plt.title('Beta时间序列对比')
    plt.xlabel('日期')
    plt.ylabel('Beta值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 子图2: 相关性散点图
    plt.subplot(2, 3, 2)
    plt.scatter(ols_betas, kalman_betas, alpha=0.5, s=1)
    plt.plot([min(ols_betas), max(ols_betas)], [min(ols_betas), max(ols_betas)], 'r--', alpha=0.8)
    plt.xlabel('OLS Beta')
    plt.ylabel('Kalman Beta')
    plt.title(f'相关性散点图 (r={correlation:.4f})')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 稳定性对比
    plt.subplot(2, 3, 3)
    methods = ['Kalman', 'OLS-60天']
    stds = [kf_std, ols_std]
    colors = ['blue', 'orange']
    bars = plt.bar(methods, stds, color=colors, alpha=0.7)
    plt.ylabel('Beta标准差')
    plt.title(f'稳定性对比 ({stability_ratio:.2f}x改善)')
    plt.grid(True, alpha=0.3)
    
    for bar, std in zip(bars, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std*0.01, 
                f'{std:.6f}', ha='center', va='bottom')
    
    # 子图4: 创新白化诊断
    plt.subplot(2, 3, 4)
    plt.hist(z_scores, bins=50, alpha=0.7, density=True, color='green')
    plt.axvline(0, color='red', linestyle='--', alpha=0.8, label='均值=0')
    plt.axvline(np.mean(z_scores), color='blue', linestyle='-', alpha=0.8, label=f'实际均值={z_mean:.4f}')
    plt.xlabel('标准化创新 z')
    plt.ylabel('密度')
    plt.title(f'创新白化检查 (std={z_std:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5: Z-score时间序列
    plt.subplot(2, 3, 5)
    plt.plot(valid_dates, z_scores, alpha=0.6, linewidth=0.5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5, label='±1标准差')
    plt.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('日期')
    plt.ylabel('z-score')
    plt.title('标准化创新时间序列')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 子图6: Beta差异
    plt.subplot(2, 3, 6)
    beta_diff = np.abs(kalman_betas - ols_betas)
    plt.plot(valid_dates, beta_diff, color='purple', alpha=0.7, linewidth=1)
    plt.xlabel('日期')
    plt.ylabel('|Beta_Kalman - Beta_OLS|')
    plt.title('Beta绝对差异')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('final_kalman_comparison.png', dpi=150, bbox_inches='tight')
    print('对比图已保存: final_kalman_comparison.png')
    
    return {
        'kalman_stats': {'mean': kf_mean, 'std': kf_std},
        'ols_stats': {'mean': ols_mean, 'std': ols_std},
        'stability_ratio': stability_ratio,
        'correlation': correlation,
        'innovation_whitening': {'mean': z_mean, 'std': z_std, 'ok': whitening_ok},
        'targets_met': targets_met,
        'all_targets_met': all_targets_met
    }

if __name__ == '__main__':
    results = final_kalman_ols_comparison()
    
    print(f'\\n=== 总结 ===')
    if results['all_targets_met']:
        print('🎉 成功！所有专家目标均已达成')
    else:
        print('⚠️  部分目标未达成，但已找到可行的参数设置')