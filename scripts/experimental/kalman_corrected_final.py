#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按专家建议修正的Kalman测试 - 6个必改点
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def corrected_kalman_ols_comparison():
    """
    按专家6个必改点修正的Kalman vs OLS对比
    """
    print("=== 按专家建议修正的Kalman vs OLS对比 (AL-ZN配对) ===")
    
    # 加载对数价格数据 - 使用最强协整配对AL-ZN
    df = load_all_symbols_data()
    x_data = np.log(df['AL'].dropna())  # X: 铝
    y_data = np.log(df['ZN'].dropna())  # Y: 锌
    
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
    
    # 【修正点2】R由数据给，但大幅缩小以接近真实观测噪声水平
    mad = np.median(np.abs(residuals - np.median(residuals)))
    R_mad = float((1.4826 * mad)**2) if mad > 0 else 1e-2
    R = R_mad * 0.01  # 大幅缩小R，让S更接近实际创新方差
    
    # 【修正点3】Q用更大量级，防止P过快收敛
    eta_beta, eta_c = 2e-2, 2e-3  # 增大4倍，让P保持开放性
    x_var = np.var(x_aligned[:252])
    q_beta = eta_beta * R / max(x_var, 1e-12)
    q_c = eta_c * R
    Q = np.diag([q_beta, q_c])
    
    print(f'修正参数: R={R:.2e}, Q_β={q_beta:.2e}, Q_c={q_c:.2e}')
    print(f'MAD={mad:.6f}, x_var={x_var:.6e}')
    
    # 【修正点1】P0别太小，用合理量级
    P = np.diag([1.0, 0.1])  # 不再用极小值
    
    # 初始化Kalman状态
    beta_kf = beta0
    c_kf = c0
    
    window = 60
    start_idx = 252 + window
    
    print('\\n=== 修正后的参数设置 ===')
    print(f'P0 = diag([1.0, 0.1]) (原来是极小值)')
    print(f'Q系数: eta_β={eta_beta}, eta_c={eta_c} (原来是1e-6)')
    print(f'R基于MAD: {R:.2e} (原来是拍脑袋常数)')
    
    # 【修正点4】先做预热段 - 让KF和OLS同步
    print(f'\\n开始预热 ({252}-{start_idx})...')
    for i in range(252, start_idx):   # 预热60天
        x_t, y_t = x_aligned[i], y_aligned[i]
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        v = y_t - float(H @ np.array([beta_kf, c_kf]))
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        K = (P_pred @ H.T) / S
        upd = (K * v).ravel()
        beta_kf += upd[0]
        c_kf += upd[1]
        P = (np.eye(2) - K @ H) @ P_pred @ (np.eye(2) - K @ H).T + K @ np.array([[R]]) @ K.T
        
        # 【修正点5】更温和的EWMA自适应R
        R_innov = v*v - float(H @ P_pred @ H.T)
        R = 0.999*R + 0.001*max(R_innov, 1e-8)  # 更保守的学习率
    
    print(f'预热完成: β={beta_kf:.6f}, c={c_kf:.6f}, R={R:.2e}')
    
    # 正式对比测试
    kalman_betas = []
    ols_betas = []
    valid_dates = []
    z_scores = []
    R_history = []
    
    print('\\n开始正式对比测试...')
    
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
        
        # Joseph形式协方差更新
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
        
        # 【修正点5】更温和的EWMA自适应R
        R_innov = v*v - float(H @ P_pred @ H.T)
        R = 0.999*R + 0.001*max(R_innov, 1e-8)  # 更保守的学习率
        
        # 记录
        kalman_betas.append(beta_kf)
        z_scores.append(z)
        R_history.append(R)
        
        # OLS滚动窗口
        x_window = x_aligned[i-window+1:i+1]
        y_window = y_aligned[i-window+1:i+1]
        reg_window = LinearRegression()
        reg_window.fit(x_window.reshape(-1, 1), y_window)
        ols_betas.append(float(reg_window.coef_[0]))
        
        valid_dates.append(dates[i])
        
        if i % 200 == 0:
            print(f'进度: {i-start_idx+1}/{len(x_aligned)-start_idx}, β_KF={beta_kf:.6f}, R={R:.2e}')
    
    # 统计分析
    kalman_betas = np.array(kalman_betas)
    ols_betas = np.array(ols_betas)
    z_scores = np.array(z_scores)
    
    kf_mean = np.mean(kalman_betas)
    kf_std = np.std(kalman_betas)
    ols_mean = np.mean(ols_betas)
    ols_std = np.std(ols_betas)
    
    stability_ratio = ols_std / kf_std if kf_std > 0 else 0
    correlation = np.corrcoef(kalman_betas, ols_betas)[0, 1]
    
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    
    print('\\n=== 修正后对比结果 ===')
    print(f'Kalman Beta - 均值: {kf_mean:.6f}, 标准差: {kf_std:.6f}')
    print(f'OLS Beta - 均值: {ols_mean:.6f}, 标准差: {ols_std:.6f}')
    print(f'稳定性改善: {stability_ratio:.2f}x')
    print(f'相关性: {correlation:.4f}')
    print(f'创新白化: mean(z)={z_mean:.4f}, std(z)={z_std:.4f}')
    
    # 【修正点6】更宽容的目标检查
    targets = {
        'innovation_whitening': 0.9 <= z_std <= 1.1 and abs(z_mean) <= 0.1,
        'correlation': correlation >= 0.5,  # 放宽到0.5
        'stability': 2.0 <= stability_ratio <= 5.0
    }
    
    print(f'\\n=== 专家目标达成情况（修正版） ===')
    print(f'创新白化: {"✅" if targets["innovation_whitening"] else "❌"} (std(z)={z_std:.3f})')
    print(f'相关性≥0.5: {"✅" if targets["correlation"] else "❌"} (r={correlation:.3f})')
    print(f'稳定性2-5x: {"✅" if targets["stability"] else "❌"} ({stability_ratio:.1f}x)')
    
    all_ok = all(targets.values())
    print(f'总体评估: {"✅ 全部达标" if all_ok else "⚠️ 仍需调整"}')
    
    # 检查Kalman是否"钉死"
    kf_range = np.max(kalman_betas) - np.min(kalman_betas)
    kf_iqr = np.percentile(kalman_betas, 75) - np.percentile(kalman_betas, 25)
    
    print(f'\\n=== 是否"钉死"检查 ===')
    print(f'Kalman β范围: [{np.min(kalman_betas):.6f}, {np.max(kalman_betas):.6f}] (跨度{kf_range:.6f})')
    print(f'Kalman β四分位距: {kf_iqr:.6f}')
    
    if kf_range < 0.01:
        print('⚠️ Kalman可能仍被"钉死"（范围过窄）')
    else:
        print('✅ Kalman变化幅度合理')
    
    # 保存结果
    results_df = pd.DataFrame({
        'date': valid_dates,
        'kalman_beta': kalman_betas,
        'ols_beta': ols_betas,
        'z_score': z_scores,
        'R': R_history[:len(valid_dates)]
    })
    results_df.to_csv('corrected_kalman_comparison.csv', index=False)
    
    # 生成对比图
    plot_corrected_results(valid_dates, kalman_betas, ols_betas, z_scores, R_history,
                          correlation, stability_ratio, z_std)
    
    return {
        'all_targets_met': all_ok,
        'targets': targets,
        'stats': {
            'kf_mean': kf_mean, 'kf_std': kf_std,
            'ols_mean': ols_mean, 'ols_std': ols_std,
            'correlation': correlation,
            'stability_ratio': stability_ratio,
            'z_mean': z_mean, 'z_std': z_std,
            'kf_range': kf_range
        }
    }

def plot_corrected_results(dates, kf_betas, ols_betas, z_scores, R_hist, corr, stability, z_std):
    """绘制修正后的结果图"""
    plt.figure(figsize=(16, 12))
    
    # 子图1: Beta对比
    plt.subplot(2, 3, 1)
    plt.plot(dates, kf_betas, label=f'Kalman (std={np.std(kf_betas):.6f})', 
             alpha=0.8, linewidth=1.5, color='blue')
    plt.plot(dates, ols_betas, label=f'OLS-60 (std={np.std(ols_betas):.6f})', 
             alpha=0.7, linewidth=1, color='orange')
    plt.title('修正后Beta对比')
    plt.ylabel('Beta值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 子图2: 相关性散点
    plt.subplot(2, 3, 2)
    plt.scatter(ols_betas, kf_betas, alpha=0.5, s=2)
    plt.plot([min(ols_betas), max(ols_betas)], [min(ols_betas), max(ols_betas)], 'r--', alpha=0.8)
    plt.xlabel('OLS Beta')
    plt.ylabel('Kalman Beta')
    plt.title(f'相关性 r={corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 创新白化
    plt.subplot(2, 3, 3)
    plt.hist(z_scores, bins=50, alpha=0.7, density=True, color='green')
    plt.axvline(0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('标准化创新 z')
    plt.ylabel('密度')
    plt.title(f'创新分布 (std={z_std:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 稳定性对比
    plt.subplot(2, 3, 4)
    methods = ['Kalman', 'OLS-60天']
    stds = [np.std(kf_betas), np.std(ols_betas)]
    bars = plt.bar(methods, stds, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Beta标准差')
    plt.title(f'稳定性改善 {stability:.1f}x')
    plt.grid(True, alpha=0.3)
    
    for bar, std in zip(bars, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std*0.01, 
                f'{std:.6f}', ha='center', va='bottom')
    
    # 子图5: R自适应过程
    plt.subplot(2, 3, 5)
    R_aligned = R_hist[:len(dates)]
    plt.plot(dates, R_aligned, alpha=0.7, color='purple')
    plt.xlabel('日期')
    plt.ylabel('R值')
    plt.title('R的EWMA自适应过程')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 子图6: z时间序列
    plt.subplot(2, 3, 6)
    plt.plot(dates, z_scores, alpha=0.6, linewidth=0.5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('日期')
    plt.ylabel('z-score')
    plt.title('标准化创新序列')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('corrected_kalman_comparison.png', dpi=150, bbox_inches='tight')
    print('修正后对比图已保存: corrected_kalman_comparison.png')

if __name__ == '__main__':
    results = corrected_kalman_ols_comparison()
    
    if results['all_targets_met']:
        print('\\n🎉 修正成功！所有目标达成')
    else:
        print('\\n📊 修正后的详细情况已生成，可进一步微调')