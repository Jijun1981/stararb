#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版Kalman调参器 - 基于S值反推调整参数
"""

import numpy as np
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

def test_corrected_kalman():
    """
    使用反推的方法修正Kalman参数，达到std(z) ≈ 1
    """
    print("=== 修正版Kalman参数测试 ===")
    
    # 加载对数价格数据
    df = load_all_symbols_data()
    x_data = np.log(df['SM'].dropna())
    y_data = np.log(df['RB'].dropna())
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates].values
    y_aligned = y_data.loc[common_dates].values
    
    print(f'数据: {len(x_aligned)}个点, {common_dates[0].strftime("%Y-%m-%d")} 到 {common_dates[-1].strftime("%Y-%m-%d")}')
    
    # 初始OLS
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])
    beta0 = reg.coef_[0]
    c0 = reg.intercept_
    residuals = y_aligned[:252] - (beta0 * x_aligned[:252] + c0)
    
    print(f'初始OLS: β={beta0:.6f}, c={c0:.6f}, 残差std={np.std(residuals):.6f}')
    
    # 根据调试结果，目标S应该约为0.0028
    # 当前问题：S ≈ 58，需要降到0.0028，调整倍数约0.00005
    
    # 方法1：大幅减小Q（更符合专家"KF太敏感"的建议）
    print("\\n=== 方法1：大幅减小Q ===")
    R_original = np.var(residuals)
    x_var = np.var(x_aligned[:252])
    
    # 从eta_beta=0.405大幅降低到合理水平
    eta_beta_corrected = 1e-6  # 比原始5e-3还要小很多
    eta_c_corrected = 1e-7
    
    q_beta = eta_beta_corrected * R_original / x_var
    q_c = eta_c_corrected * R_original
    
    print(f'修正参数: eta_β={eta_beta_corrected:.1e}, eta_c={eta_c_corrected:.1e}')
    print(f'Q_β={q_beta:.2e}, Q_c={q_c:.2e}, R={R_original:.2e}')
    
    result1 = run_kalman_test(x_aligned, y_aligned, beta0, c0, q_beta, q_c, R_original, "方法1-减小Q")
    
    # 方法2：适度增大R
    print("\\n=== 方法2：增大R + 适度Q ===")
    R_increased = R_original * 50  # 增大50倍
    eta_beta = 5e-3  # 回到专家建议的均衡值
    eta_c = 5e-4
    
    q_beta_2 = eta_beta * R_increased / x_var
    q_c_2 = eta_c * R_increased
    
    print(f'修正参数: R增大50倍, eta_β={eta_beta:.1e}')
    print(f'Q_β={q_beta_2:.2e}, Q_c={q_c_2:.2e}, R={R_increased:.2e}')
    
    result2 = run_kalman_test(x_aligned, y_aligned, beta0, c0, q_beta_2, q_c_2, R_increased, "方法2-增大R")
    
    # 方法3：均衡调整（专家推荐）
    print("\\n=== 方法3：均衡调整 ===")
    # 从实际残差反推合理的R
    actual_innovation_std = 0.053  # 从调试得到
    target_S = (actual_innovation_std / 1.0) ** 2  # 目标std(z)=1
    
    # 估算合理的R：让R和H@P@H.T项平衡
    # 典型x_t≈8.9，如果P_ββ保持在合理范围(如0.01)，则H@P@H.T ≈ 79*0.01 = 0.79
    estimated_H_P_HT = 0.001  # 保守估计
    R_balanced = target_S - estimated_H_P_HT
    R_balanced = max(R_balanced, target_S * 0.1)  # 至少占10%
    
    eta_beta_3 = 1e-4  # 保守设定，让P_ββ不会太大
    eta_c_3 = 1e-5
    
    q_beta_3 = eta_beta_3 * R_balanced / x_var
    q_c_3 = eta_c_3 * R_balanced
    
    print(f'均衡参数: 目标S={target_S:.2e}, R={R_balanced:.2e}')
    print(f'Q_β={q_beta_3:.2e}, Q_c={q_c_3:.2e}')
    
    result3 = run_kalman_test(x_aligned, y_aligned, beta0, c0, q_beta_3, q_c_3, R_balanced, "方法3-均衡")
    
    # 汇总结果
    print("\\n=== 三种方法对比 ===")
    results = [result1, result2, result3]
    for r in results:
        status = "✅通过" if (0.9 <= r['z_std'] <= 1.1) and r['correlation'] >= 0.6 else "❌未达标"
        print(f"{r['method']}: std(z)={r['z_std']:.3f}, 相关性={r['correlation']:.3f}, 稳定性={r['stability_ratio']:.1f}x {status}")
    
    return results

def run_kalman_test(x_aligned, y_aligned, beta0, c0, q_beta, q_c, R, method_name, test_steps=300):
    """
    运行Kalman测试，返回关键指标
    """
    # 初始化
    beta = beta0
    c = c0
    P = np.diag([0.01, 0.001])  # 适度的初始不确定性
    Q = np.diag([q_beta, q_c])
    
    # 记录
    betas_kf = []
    z_scores = []
    S_values = []
    
    # 同时计算OLS对比
    window = 60
    betas_ols = []
    
    start_idx = 252
    end_idx = min(start_idx + test_steps, len(x_aligned) - window)
    
    for i in range(start_idx, end_idx):
        # Kalman更新
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        
        # 预测
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        state = np.array([beta, c])
        y_pred = float(H @ state)
        
        # 创新和S
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        z = v / np.sqrt(S)
        
        # 记录
        S_values.append(S)
        z_scores.append(z)
        
        # 状态更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta += update_vec[0]
        c += update_vec[1]
        betas_kf.append(beta)
        
        # 协方差更新
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
        
        # OLS对比（如果有足够数据）
        if i >= start_idx + window:
            x_window = x_aligned[i-window+1:i+1]
            y_window = y_aligned[i-window+1:i+1]
            reg_window = LinearRegression()
            reg_window.fit(x_window.reshape(-1, 1), y_window)
            betas_ols.append(float(reg_window.coef_[0]))
    
    # 统计
    z_scores = np.array(z_scores)
    betas_kf = np.array(betas_kf)
    
    # 只对比有效的OLS数据
    if len(betas_ols) > 0:
        betas_ols = np.array(betas_ols)
        betas_kf_aligned = betas_kf[-len(betas_ols):]  # 对齐长度
        
        kf_std = np.std(betas_kf_aligned)
        ols_std = np.std(betas_ols)
        stability_ratio = ols_std / kf_std if kf_std > 0 else 0
        correlation = np.corrcoef(betas_kf_aligned, betas_ols)[0, 1] if len(betas_ols) > 1 else 0
    else:
        kf_std = np.std(betas_kf)
        stability_ratio = 0
        correlation = 0
    
    return {
        'method': method_name,
        'z_mean': np.mean(z_scores),
        'z_std': np.std(z_scores),
        'S_mean': np.mean(S_values),
        'S_range': [np.min(S_values), np.max(S_values)],
        'beta_std': kf_std,
        'stability_ratio': stability_ratio,
        'correlation': correlation,
        'betas_kf': betas_kf,
        'z_scores': z_scores
    }

if __name__ == '__main__':
    results = test_corrected_kalman()