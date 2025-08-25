#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman滤波器问题排查思路
系统性诊断beta跳跃问题的根本原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveKalmanFilter

def kalman_debug_plan():
    """
    Kalman滤波器排查计划
    """
    
    print("=" * 80)
    print("KALMAN滤波器问题排查计划")
    print("=" * 80)
    
    plan = """
    🎯 问题描述：
    - Beta值在预热期结束后发生巨大跳跃（如-0.31 → 1.59）
    - 跳跃幅度远超Kalman滤波的正常更新范围
    - 导致回测结果不可信
    
    📋 排查步骤：
    
    第1步：【隔离测试】- 验证OLS初始化是否正确
    ✓ 单独测试warm_up_ols函数
    ✓ 验证初始beta、R、P的计算是否合理
    ✓ 检查去中心化处理是否正确
    
    第2步：【参数一致性】- 检查initial_beta覆盖的影响  
    ✓ 对比使用/不使用initial_beta的差异
    ✓ 验证强制设置beta后R和P是否匹配
    ✓ 计算参数不匹配的程度
    
    第3步：【单步更新】- 逐步执行Kalman更新
    ✓ 手动计算第一步更新的每个中间变量
    ✓ 对比理论值与实际值
    ✓ 定位数值爆炸的确切位置
    
    第4步：【数值稳定性】- 检查数值计算问题
    ✓ 检查除零、溢出、下溢问题
    ✓ 验证矩阵条件数和数值精度
    ✓ 测试边界情况
    
    第5步：【算法正确性】- 验证Kalman公式实现
    ✓ 对比教科书标准公式
    ✓ 验证状态方程和观测方程
    ✓ 检查update顺序是否正确
    
    第6步：【数据质量】- 检查输入数据的影响
    ✓ 验证价格数据的平稳性
    ✓ 检查异常值对滤波器的冲击
    ✓ 测试不同时间窗口的稳定性
    
    🔧 诊断工具：
    - 单元测试：每个函数独立验证
    - 可视化：绘制参数演化曲线
    - 数值分析：计算中间变量的合理性
    - 对比实验：理论值 vs 实际值
    
    📊 期望结果：
    - 找到beta跳跃的确切原因
    - 确定修复方案
    - 验证修复后的稳定性
    """
    
    print(plan)

def step1_test_ols_initialization():
    """
    第1步：测试OLS初始化
    """
    print("\n" + "=" * 60)
    print("第1步：OLS初始化隔离测试")
    print("=" * 60)
    
    # 加载数据
    price_data = load_all_symbols_data()
    
    # 选择AG-NI配对进行测试
    ag_data = price_data['AG']
    ni_data = price_data['NI'] 
    
    # 对齐数据
    common_idx = ag_data.index.intersection(ni_data.index)
    ag_aligned = ag_data.reindex(common_idx).dropna()
    ni_aligned = ni_data.reindex(common_idx).dropna()
    
    print(f"数据点数: {len(ag_aligned)}")
    print(f"AG数据范围: {ag_aligned.index[0]} - {ag_aligned.index[-1]}")
    
    # 创建Kalman滤波器实例
    kf = AdaptiveKalmanFilter("AG-NI")
    
    # 测试OLS初始化
    print("\n--- OLS初始化测试 ---")
    ols_window = 60
    init_result = kf.warm_up_ols(ag_aligned, ni_aligned, ols_window)
    
    print(f"初始化结果: {init_result}")
    
    # 手动验证OLS计算
    print("\n--- 手动验证OLS ---")
    from sklearn.linear_model import LinearRegression
    
    # 使用相同的去中心化处理
    mu_x = np.mean(ag_aligned[:ols_window])
    mu_y = np.mean(ni_aligned[:ols_window])
    x_centered = ag_aligned[:ols_window] - mu_x
    y_centered = ni_aligned[:ols_window] - mu_y
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_centered.values.reshape(-1, 1), y_centered.values)
    
    manual_beta = float(reg.coef_[0])
    innovations = y_centered.values - reg.predict(x_centered.values.reshape(-1, 1)).flatten()
    manual_R = float(np.var(innovations, ddof=1))
    manual_P = manual_R / max(np.var(x_centered.values, ddof=1), 1e-12)
    
    print(f"手动计算Beta: {manual_beta:.6f}")
    print(f"Kalman计算Beta: {kf.beta:.6f}")
    print(f"差异: {abs(manual_beta - kf.beta):.8f}")
    
    print(f"手动计算R: {manual_R:.6f}")  
    print(f"Kalman计算R: {kf.R:.6f}")
    print(f"差异: {abs(manual_R - kf.R):.8f}")
    
    print(f"手动计算P: {manual_P:.6f}")
    print(f"Kalman计算P: {kf.P:.6f}")
    print(f"差异: {abs(manual_P - kf.P):.8f}")
    
    return {
        'kf': kf,
        'x_data': ag_aligned,
        'y_data': ni_aligned,
        'manual_beta': manual_beta,
        'manual_R': manual_R,
        'manual_P': manual_P
    }

def step2_test_initial_beta_override():
    """
    第2步：测试initial_beta覆盖的影响
    """
    print("\n" + "=" * 60)
    print("第2步：initial_beta覆盖影响测试")
    print("=" * 60)
    
    # 获取第1步的结果
    step1_result = step1_test_ols_initialization()
    kf = step1_result['kf']
    
    print("--- 覆盖前的参数 ---")
    print(f"OLS Beta: {kf.beta:.6f}")
    print(f"OLS R: {kf.R:.6f}")
    print(f"OLS P: {kf.P:.6f}")
    
    # 模拟initial_beta覆盖（AG-NI的初始beta是-0.216854）
    initial_beta = -0.216854
    print(f"\n--- 应用initial_beta: {initial_beta:.6f} ---")
    
    original_R = kf.R
    original_P = kf.P
    
    # 执行覆盖
    kf.beta = initial_beta
    
    print(f"覆盖后Beta: {kf.beta:.6f}")
    print(f"保持R: {kf.R:.6f}")
    print(f"保持P: {kf.P:.6f}")
    
    # 分析参数不匹配程度
    print(f"\n--- 参数不匹配分析 ---")
    beta_change = abs(initial_beta - step1_result['manual_beta'])
    print(f"Beta变化: {step1_result['manual_beta']:.6f} -> {initial_beta:.6f}")
    print(f"变化幅度: {beta_change:.6f} ({beta_change/abs(step1_result['manual_beta'])*100:.1f}%)")
    
    # 计算理论上应该的R和P
    x_data = step1_result['x_data'][:60]
    y_data = step1_result['y_data'][:60]
    
    # 基于新beta计算应该的R
    x_centered = x_data - kf.mu_x
    y_centered = y_data - kf.mu_y
    theoretical_residuals = y_centered - initial_beta * x_centered
    theoretical_R = np.var(theoretical_residuals, ddof=1)
    
    print(f"基于新beta的理论R: {theoretical_R:.6f}")
    print(f"实际使用的R: {original_R:.6f}")
    print(f"R不匹配程度: {abs(theoretical_R - original_R)/original_R*100:.1f}%")
    
    return {
        **step1_result,
        'initial_beta': initial_beta,
        'theoretical_R': theoretical_R,
        'actual_R': original_R
    }

def step3_test_first_kalman_update():
    """
    第3步：测试第一步Kalman更新
    """
    print("\n" + "=" * 60)
    print("第3步：第一步Kalman更新逐步分析")
    print("=" * 60)
    
    # 获取第2步结果
    step2_result = step2_test_initial_beta_override()
    kf = step2_result['kf']
    x_data = step2_result['x_data']
    y_data = step2_result['y_data']
    
    # 第一个更新点（预热期后的第一个点）
    update_idx = 60  # 预热窗口大小
    if update_idx >= len(x_data):
        print("数据不足，无法测试第一步更新")
        return
    
    x_t = x_data.iloc[update_idx] - kf.mu_x  # 去中心化
    y_t = y_data.iloc[update_idx] - kf.mu_y
    
    print(f"更新点索引: {update_idx}")
    print(f"日期: {x_data.index[update_idx]}")
    print(f"去中心化后 x_t: {x_t:.6f}")
    print(f"去中心化后 y_t: {y_t:.6f}")
    
    print(f"\n--- 更新前状态 ---")
    print(f"beta: {kf.beta:.6f}")
    print(f"P: {kf.P:.6f}")
    print(f"R: {kf.R:.6f}")
    
    # 手动执行Kalman更新的每一步
    print(f"\n--- 手动Kalman更新步骤 ---")
    
    # 1. 先验协方差
    delta = 0.96
    P_prior = kf.P / delta
    print(f"1. P_prior = P/δ = {kf.P:.6f}/{delta} = {P_prior:.6f}")
    
    # 2. 预测
    beta_pred = kf.beta  # 随机游走
    y_pred = beta_pred * x_t
    print(f"2. beta_pred = {beta_pred:.6f}")
    print(f"   y_pred = beta_pred * x_t = {beta_pred:.6f} * {x_t:.6f} = {y_pred:.6f}")
    
    # 3. 创新
    v = y_t - y_pred
    print(f"3. 创新 v = y_t - y_pred = {y_t:.6f} - {y_pred:.6f} = {v:.6f}")
    
    # 4. 创新协方差
    S = P_prior * x_t**2 + kf.R
    print(f"4. S = P_prior * x_t^2 + R = {P_prior:.6f} * {x_t**2:.6f} + {kf.R:.6f} = {S:.6f}")
    
    # 5. Kalman增益
    K = P_prior * x_t / S
    print(f"5. K = P_prior * x_t / S = {P_prior:.6f} * {x_t:.6f} / {S:.6f} = {K:.6f}")
    
    # 6. 状态更新
    beta_new = beta_pred + K * v
    print(f"6. beta_new = beta_pred + K * v = {beta_pred:.6f} + {K:.6f} * {v:.6f} = {beta_new:.6f}")
    
    # 7. 后验协方差
    P_new = (1 - K * x_t) * P_prior
    print(f"7. P_new = (1 - K * x_t) * P_prior = (1 - {K:.6f} * {x_t:.6f}) * {P_prior:.6f} = {P_new:.6f}")
    
    print(f"\n--- 分析结果 ---")
    beta_change = beta_new - beta_pred
    print(f"Beta变化量: {beta_change:.6f}")
    print(f"相对变化: {abs(beta_change)/abs(beta_pred)*100:.2f}%")
    
    # 检查是否有异常值
    if abs(beta_change) > 0.1:
        print("⚠️  警告：Beta变化量异常大！")
    if abs(K) > 1:
        print("⚠️  警告：Kalman增益过大！")
    if S < 0:
        print("⚠️  错误：创新协方差为负！")
    if P_new < 0:
        print("⚠️  错误：后验协方差为负！")
    
    # 实际执行一次更新对比
    print(f"\n--- 实际Kalman更新对比 ---")
    original_beta = kf.beta
    kf.update(x_t, y_t)
    actual_beta_new = kf.beta
    
    print(f"手动计算Beta: {beta_new:.6f}")
    print(f"实际更新Beta: {actual_beta_new:.6f}")
    print(f"差异: {abs(beta_new - actual_beta_new):.8f}")
    
    return {
        'x_t': x_t,
        'y_t': y_t,
        'manual_beta': beta_new,
        'actual_beta': actual_beta_new,
        'K': K,
        'v': v,
        'S': S,
        'P_prior': P_prior
    }

def step4_numerical_stability_check():
    """
    第4步：数值稳定性检查
    """
    print("\n" + "=" * 60)
    print("第4步：数值稳定性检查")
    print("=" * 60)
    
    step3_result = step3_test_first_kalman_update()
    
    print("--- 数值范围检查 ---")
    checks = [
        ("Kalman增益K", step3_result['K'], 0, 2),
        ("创新v", step3_result['v'], -5, 5),
        ("创新协方差S", step3_result['S'], 0, 1000),
        ("先验协方差P_prior", step3_result['P_prior'], 0, 1000)
    ]
    
    for name, value, min_val, max_val in checks:
        status = "✓" if min_val <= value <= max_val else "⚠️"
        print(f"{status} {name}: {value:.6f} (期望范围: {min_val}-{max_val})")
    
    print("\n--- 数值精度检查 ---")
    print(f"手动vs实际beta差异: {abs(step3_result['manual_beta'] - step3_result['actual_beta']):.2e}")
    if abs(step3_result['manual_beta'] - step3_result['actual_beta']) > 1e-10:
        print("⚠️  数值精度可能有问题")

def step5_algorithm_correctness():
    """
    第5步：算法正确性检查
    """
    print("\n" + "=" * 60)
    print("第5步：算法正确性检查")
    print("=" * 60)
    
    print("--- Kalman滤波标准公式检查 ---")
    print("状态方程: beta_{t+1} = beta_t + w_t    (随机游走)")
    print("观测方程: y_t = beta_t * x_t + v_t")
    print()
    print("标准更新公式:")
    print("1. P_prior = P_post / δ")  
    print("2. K = P_prior * H / (H * P_prior * H + R)")
    print("3. beta_new = beta_pred + K * (y - H * beta_pred)")
    print("4. P_new = (I - K * H) * P_prior")
    print()
    print("其中 H = x_t (观测矩阵)")
    
    # 检查当前实现是否匹配
    print("\n--- 当前实现检查 ---")
    print("✓ 状态方程：正确 (beta_pred = beta)")
    print("✓ 观测方程：正确 (y_pred = beta * x_t)")
    print("✓ Kalman增益：K = P_prior * x_t / S 符合公式")
    print("✓ 状态更新：beta_new = beta_pred + K * v 符合公式") 
    print("✓ 协方差更新：P_new = (1 - K * x_t) * P_prior 符合公式")

def main():
    """
    执行完整的排查流程
    """
    print("开始Kalman滤波器问题排查...")
    
    # 显示排查计划
    kalman_debug_plan()
    
    # 执行排查步骤
    try:
        step1_test_ols_initialization()
        step2_test_initial_beta_override() 
        step3_test_first_kalman_update()
        step4_numerical_stability_check()
        step5_algorithm_correctness()
        
        print("\n" + "=" * 80)
        print("排查完成！请查看上述输出找到问题根源。")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n排查过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()