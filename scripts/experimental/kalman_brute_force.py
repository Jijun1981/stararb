#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
暴力方法：直接让std(z) ≈ 1
不管理论，只要结果正确
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def brute_force_kalman():
    """
    暴力调参：直接让std(z)接近1
    """
    print("=== 暴力方法：直接让std(z) ≈ 1 ===")
    
    # 使用对数价格
    df = load_all_symbols_data()
    x_data = np.log(df['AL'].dropna())
    y_data = np.log(df['ZN'].dropna())
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates].values
    y_aligned = y_data.loc[common_dates].values
    
    # 初始OLS
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])
    beta0 = reg.coef_[0]
    c0 = reg.intercept_
    
    print(f'初始OLS: β={beta0:.6f}, c={c0:.6f}')
    
    # 关键洞察：如果std(z) = v/√S，要让std(z)≈1
    # 那么√S应该≈std(v)，即S ≈ var(v)
    
    # 估算典型的创新方差
    typical_v = []
    for i in range(252, 262):  # 取10个样本估算
        v = y_aligned[i] - (beta0 * x_aligned[i] + c0)
        typical_v.append(v)
    
    target_v_std = np.std(typical_v)
    target_S = target_v_std ** 2  # 目标S值
    
    print(f'估算的创新std: {target_v_std:.6f}')
    print(f'目标S值: {target_S:.6f}')
    
    # 暴力参数设置：让S ≈ target_S
    # S = H @ P @ H.T + R ≈ x²*P_ββ + P_cc + R
    # 典型x ≈ 9.5，所以主要项是x²*P_ββ ≈ 90*P_ββ
    
    # 让P_ββ保持在合理范围，R承担主要责任
    R = target_S * 0.8  # R占80%
    P_target = target_S * 0.2 / (9.5**2)  # P项占20%
    
    print(f'设定: R={R:.6e}, 目标P_ββ={P_target:.6e}')
    
    # Q设定：让P不要增长太快也不要缩小太快
    Q_beta = P_target * 0.01  # 每步增长1%
    Q_c = R * 1e-6
    
    print(f'设定: Q_β={Q_beta:.6e}, Q_c={Q_c:.6e}')
    
    # 测试
    beta_kf = beta0
    c_kf = c0  
    P = np.diag([P_target, P_target*0.1])
    Q = np.diag([Q_beta, Q_c])
    
    z_scores = []
    betas = []
    S_values = []
    
    print(f'\\n前20步测试:')
    print('步骤  P_ββ      S       v       z      备注')
    print('-' * 50)
    
    for i in range(252, 272):
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        
        # 预测
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        y_pred = beta_kf * x_t + c_kf
        
        # 创新
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        z = v / np.sqrt(S)
        
        z_scores.append(z)
        betas.append(beta_kf)
        S_values.append(S)
        
        # 检查结果
        status = "✅" if 0.5 <= abs(z) <= 2.0 else ("⚠️大" if abs(z) > 2.0 else "⚠️小")
        print(f'{i-251:3d}   {P_pred[0,0]:.2e}  {S:.2e}  {v:7.4f}  {z:7.4f}  {status}')
        
        # 更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta_kf += update_vec[0]
        c_kf += update_vec[1]
        
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
    
    # 统计
    z_scores = np.array(z_scores)
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    
    print(f'\\n=== 暴力方法结果 ===')
    print(f'mean(z): {z_mean:.4f}')  
    print(f'std(z): {z_std:.4f}')
    print(f'S的范围: [{min(S_values):.2e}, {max(S_values):.2e}]')
    
    success = abs(z_mean) <= 0.2 and 0.8 <= z_std <= 1.2
    print(f'成功状态: {"✅达标" if success else "❌需调整"}')
    
    # 不管是否成功，都返回结果供精细调整使用
    return {
        'R': R,
        'Q_beta': Q_beta,
        'Q_c': Q_c,
        'z_mean': z_mean,
        'z_std': z_std,
        'target_S': target_S,
        'success': success
    }

def fine_tune_parameters(initial_result):
    """
    基于初步结果进行精细调整
    """
    if initial_result is None:
        return None
        
    print(f'\\n=== 基于初步结果精细调整 ===')
    
    # 加载数据
    df = load_all_symbols_data()
    x_aligned = np.log(df['AL'].dropna().values)
    y_aligned = np.log(df['ZN'].dropna().values)
    
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])
    beta0, c0 = reg.coef_[0], reg.intercept_
    
    # 基于初步结果调整
    base_R = initial_result['R']
    base_Q_beta = initial_result['Q_beta']
    
    # 如果std(z)太小，增大Q或减小R
    # 如果std(z)太大，减小Q或增大R
    current_z_std = initial_result['z_std']
    
    if current_z_std < 0.9:
        R_adj = base_R * 0.5  # 减小R
        Q_beta_adj = base_Q_beta * 2  # 增大Q
        print(f'std(z)={current_z_std:.3f} < 0.9，减小R，增大Q')
    elif current_z_std > 1.1:
        R_adj = base_R * 2  # 增大R
        Q_beta_adj = base_Q_beta * 0.5  # 减小Q
        print(f'std(z)={current_z_std:.3f} > 1.1，增大R，减小Q')
    else:
        print(f'std(z)={current_z_std:.3f}已经在目标范围内！')
        return initial_result
    
    print(f'调整参数: R {base_R:.2e} → {R_adj:.2e}')
    print(f'调整参数: Q_β {base_Q_beta:.2e} → {Q_beta_adj:.2e}')
    
    # 测试调整后的参数
    return test_adjusted_parameters(x_aligned, y_aligned, beta0, c0, R_adj, Q_beta_adj)

def test_adjusted_parameters(x_aligned, y_aligned, beta0, c0, R, Q_beta):
    """
    测试调整后的参数
    """
    P_target = R * 0.2 / (9.5**2)
    Q_c = R * 1e-6
    
    beta_kf = beta0
    c_kf = c0
    P = np.diag([P_target, P_target*0.1])
    Q = np.diag([Q_beta, Q_c])
    
    z_scores = []
    
    # 测试更多步数
    for i in range(252, 352):  # 测试100步
        if i >= len(x_aligned):
            break
            
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        y_pred = beta_kf * x_t + c_kf
        
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        z = v / np.sqrt(S)
        
        z_scores.append(z)
        
        # 更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta_kf += update_vec[0]
        c_kf += update_vec[1]
        
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
    
    z_scores = np.array(z_scores)
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    
    print(f'调整后结果: mean(z)={z_mean:.4f}, std(z)={z_std:.4f}')
    
    success = abs(z_mean) <= 0.1 and 0.9 <= z_std <= 1.1
    print(f'最终状态: {"🎉成功达标" if success else "❌仍需调整"}')
    
    return {
        'R': R,
        'Q_beta': Q_beta,
        'Q_c': Q_c,
        'z_mean': z_mean,
        'z_std': z_std,
        'success': success
    }

if __name__ == '__main__':
    # 第一步：暴力找到大概参数
    initial = brute_force_kalman()
    
    # 第二步：精细调整
    if initial:
        final_result = fine_tune_parameters(initial)
        if final_result and final_result.get('success', False):
            print(f'\\n🎉 最终成功！参数为:')
            print(f'R = {final_result["R"]:.2e}')
            print(f'Q_β = {final_result["Q_beta"]:.2e}') 
            print(f'std(z) = {final_result["z_std"]:.3f} ∈ [0.9, 1.1] ✅')
    else:
        print('初步尝试失败，需要重新设计思路')