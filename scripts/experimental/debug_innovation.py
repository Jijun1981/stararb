#!/usr/bin/env python3
# 调试创新白化问题 - 找出std(z)过小的根因

import numpy as np
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def debug_innovation_calculation():
    # 加载对数价格数据
    df = load_all_symbols_data()
    x_data = np.log(df['SM'].dropna())
    y_data = np.log(df['RB'].dropna())
    
    common_dates = x_data.index.intersection(y_data.index)
    x_aligned = x_data.loc[common_dates].values
    y_aligned = y_data.loc[common_dates].values
    
    # 初始OLS
    reg = LinearRegression()
    reg.fit(x_aligned[:252].reshape(-1, 1), y_aligned[:252])
    beta0 = reg.coef_[0]
    c0 = reg.intercept_
    residuals = y_aligned[:252] - (beta0 * x_aligned[:252] + c0)
    
    print(f'初始参数: β={beta0:.6f}, c={c0:.6f}')
    print(f'残差std: {np.std(residuals):.6f}')
    
    # 按当前方法计算参数
    R = np.var(residuals)
    x_var = np.var(x_aligned[:252])
    eta_beta = 0.405  # 最后一轮的eta_beta
    q_beta = eta_beta * R / x_var
    q_c = 5e-4 * R
    
    print(f'\\n当前参数设置:')
    print(f'R = {R:.6e}')
    print(f'Q_β = {q_beta:.6e}')
    print(f'Q_c = {q_c:.6e}')
    print(f'Q_β/R = {q_beta/R:.6f}')
    
    # 模拟几步Kalman更新，检查S的值
    beta = beta0
    c = c0
    P = np.diag([1.0, 0.1])  # 初始P
    Q = np.diag([q_beta, q_c])
    
    S_values = []
    innovations = []
    z_scores = []
    
    print(f'\\n前10步更新的详细分析:')
    for i in range(252, 262):
        x_t = x_aligned[i]
        y_t = y_aligned[i]
        
        # 预测步
        P_pred = P + Q
        H = np.array([[x_t, 1.0]])
        state = np.array([beta, c])
        y_pred = float(H @ state)
        
        # 创新和S
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + R)
        S = max(S, 1e-12)
        
        # 标准化创新
        z = v / np.sqrt(S)
        
        S_values.append(S)
        innovations.append(v)
        z_scores.append(z)
        
        print(f'步{i-251}: x_t={x_t:.4f}, y_t={y_t:.4f}, v={v:.6f}, S={S:.6e}, z={z:.6f}')
        
        # Kalman更新（简化）
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        beta += update_vec[0]
        c += update_vec[1]
        
        # P更新
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
    
    print(f'\\n统计结果:')
    print(f'S的范围: [{min(S_values):.2e}, {max(S_values):.2e}]')
    print(f'创新v的std: {np.std(innovations):.6f}')
    print(f'z_score的std: {np.std(z_scores):.6f}')
    
    # 问题诊断
    print(f'\\n问题诊断:')
    typical_S = np.mean(S_values)
    expected_v_std = np.sqrt(typical_S)  # 如果白化良好，v的std应该约等于sqrt(S)
    actual_v_std = np.std(innovations)
    
    print(f'典型S值: {typical_S:.6e}')
    print(f'期望创新std: {expected_v_std:.6f}')
    print(f'实际创新std: {actual_v_std:.6f}')
    print(f'比例: {actual_v_std/expected_v_std:.6f}')
    
    if actual_v_std < expected_v_std * 0.5:
        print('⚠️ 创新标准差明显小于期望值，说明R可能设置过大')
        suggested_R = R * (actual_v_std/expected_v_std)**2
        print(f'建议R: {suggested_R:.6e} (当前R的{suggested_R/R:.2f}倍)')
    elif actual_v_std > expected_v_std * 2:
        print('⚠️ 创新标准差明显大于期望值，说明Q可能设置过小')
        
    # 测试专家建议：直接设定目标std(z)=1来反推R
    target_z_std = 1.0
    typical_v_std = np.std(innovations)  # 实际观测到的创新std
    required_S = (typical_v_std / target_z_std) ** 2
    
    print(f'\\n反推分析（专家目标std(z)=1）:')
    print(f'实际创新std: {typical_v_std:.6f}')
    print(f'需要的S: {required_S:.6e}')
    print(f'当前S: {typical_S:.6e}')
    print(f'S需要调整倍数: {required_S/typical_S:.2f}')
    
    return {
        'current_R': R,
        'current_S': typical_S,
        'required_S': required_S,
        'adjustment_factor': required_S/typical_S
    }

if __name__ == '__main__':
    result = debug_innovation_calculation()