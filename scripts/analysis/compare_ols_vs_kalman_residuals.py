#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLS vs Kalman残差对比验证脚本
对比固定OLS和Kalman滤波的残差平稳性
验证Kalman滤波实现是否正确
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveKalmanFilter


def adf_test(residuals, name):
    """执行ADF检验"""
    try:
        clean_residuals = residuals.dropna()
        if len(clean_residuals) < 10:
            return {'name': name, 'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False}
        
        result = adfuller(clean_residuals, autolag='AIC')
        is_stationary = result[1] < 0.05 or result[0] < result[4]['5%']
        
        return {
            'name': name,
            'n_obs': len(clean_residuals),
            'adf_stat': result[0],
            'p_value': result[1],
            'critical_5%': result[4]['5%'],
            'is_stationary': is_stationary,
            'residual_mean': clean_residuals.mean(),
            'residual_std': clean_residuals.std()
        }
    except Exception as e:
        return {'name': name, 'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False, 'error': str(e)}


def main():
    print("=" * 80)
    print("OLS vs Kalman残差对比验证")
    print("=" * 80)
    
    # 1. 加载数据
    print("加载期货数据...")
    price_data = load_all_symbols_data()
    print(f"数据范围: {price_data.index[0]} 至 {price_data.index[-1]}")
    
    # 2. 选择测试配对（选择几个已知有问题的配对）
    test_pairs = [
        {'pair': 'AG-NI', 'symbol_x': 'AG', 'symbol_y': 'NI', 'beta_1y': -3.3401},
        {'pair': 'CU-SN', 'symbol_x': 'CU', 'symbol_y': 'SN', 'beta_1y': 3.3449},
        {'pair': 'SF-ZN', 'symbol_x': 'SF', 'symbol_y': 'ZN', 'beta_1y': 2.2640},  # 之前非平稳
        {'pair': 'AU-ZN', 'symbol_x': 'AU', 'symbol_y': 'ZN', 'beta_1y': -10.7913}, # 之前非平稳
    ]
    
    # 使用近期数据测试
    test_period_start = '2023-01-01'
    test_data = price_data[test_period_start:].copy()
    
    results = []
    
    for pair_info in test_pairs:
        pair_name = pair_info['pair']
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        beta_init = pair_info['beta_1y']
        
        print(f"\n" + "=" * 60)
        print(f"分析配对: {pair_name}")
        print("=" * 60)
        
        # 获取价格数据
        x_prices = test_data[symbol_x].dropna()
        y_prices = test_data[symbol_y].dropna()
        
        # 对齐数据
        common_dates = x_prices.index.intersection(y_prices.index)
        if len(common_dates) < 100:
            print(f"数据不足，跳过")
            continue
            
        x_aligned = x_prices[common_dates]
        y_aligned = y_prices[common_dates]
        
        print(f"数据点数: {len(common_dates)}")
        print(f"价格范围: {x_aligned.iloc[0]:.4f}-{x_aligned.iloc[-1]:.4f} (X), {y_aligned.iloc[0]:.4f}-{y_aligned.iloc[-1]:.4f} (Y)")
        
        # 检查数据是否是对数价格（简单检查：对数价格通常是小数）
        x_is_log = x_aligned.max() < 20  # 对数价格通常<20
        y_is_log = y_aligned.max() < 20
        print(f"疑似对数价格: X={x_is_log}, Y={y_is_log} (最大值: X={x_aligned.max():.2f}, Y={y_aligned.max():.2f})")
        
        # ==================== 方法1: 固定OLS ====================
        print("\n1. 固定OLS回归:")
        
        # 使用全部数据的OLS
        reg_full = LinearRegression(fit_intercept=True)
        reg_full.fit(x_aligned.values.reshape(-1, 1), y_aligned.values)
        beta_ols_full = reg_full.coef_[0]
        alpha_ols_full = reg_full.intercept_
        
        # 计算残差
        residuals_ols_full = y_aligned - (alpha_ols_full + beta_ols_full * x_aligned)
        
        print(f"  全数据OLS: α={alpha_ols_full:.4f}, β={beta_ols_full:.4f}")
        print(f"  残差统计: 均值={residuals_ols_full.mean():.4f}, 标准差={residuals_ols_full.std():.4f}")
        
        # ADF检验
        adf_ols_full = adf_test(residuals_ols_full, f"{pair_name}_OLS_Full")
        print(f"  ADF检验: 统计量={adf_ols_full['adf_stat']:.4f}, p值={adf_ols_full['p_value']:.4f}, 平稳={adf_ols_full['is_stationary']}")
        
        # 使用初始60天的OLS（模拟Kalman预热）
        reg_60 = LinearRegression(fit_intercept=True)
        reg_60.fit(x_aligned.iloc[:60].values.reshape(-1, 1), y_aligned.iloc[:60].values)
        beta_ols_60 = reg_60.coef_[0]
        alpha_ols_60 = reg_60.intercept_
        
        # 用60天OLS参数计算全期残差
        residuals_ols_60 = y_aligned - (alpha_ols_60 + beta_ols_60 * x_aligned)
        
        print(f"  60天OLS: α={alpha_ols_60:.4f}, β={beta_ols_60:.4f}")
        print(f"  残差统计: 均值={residuals_ols_60.mean():.4f}, 标准差={residuals_ols_60.std():.4f}")
        
        # ADF检验
        adf_ols_60 = adf_test(residuals_ols_60, f"{pair_name}_OLS_60d")
        print(f"  ADF检验: 统计量={adf_ols_60['adf_stat']:.4f}, p值={adf_ols_60['p_value']:.4f}, 平稳={adf_ols_60['is_stationary']}")
        
        # ==================== 方法2: Kalman滤波 ====================
        print(f"\n2. Kalman滤波:")
        
        # 初始化Kalman滤波器
        kf = AdaptiveKalmanFilter(pair_name)
        kf.warm_up_ols(x_aligned.values, y_aligned.values, window=60)
        
        print(f"  OLS预热: β={kf.beta:.4f}, R={kf.R:.6f}, P={kf.P:.6f}")
        
        # 运行Kalman滤波
        kalman_residuals = []
        kalman_betas = []
        
        for i in range(60, len(x_aligned)):
            result = kf.update(y_aligned.iloc[i], x_aligned.iloc[i])
            kalman_residuals.append(result['v'])  # 创新值就是残差
            kalman_betas.append(result['beta'])
        
        kalman_residuals = np.array(kalman_residuals)
        kalman_betas = np.array(kalman_betas)
        
        print(f"  Kalman结果: 最终β={kalman_betas[-1]:.4f}, β变化范围=[{kalman_betas.min():.4f}, {kalman_betas.max():.4f}]")
        print(f"  残差统计: 均值={kalman_residuals.mean():.4f}, 标准差={kalman_residuals.std():.4f}")
        
        # ADF检验（只检验60天后的残差）
        adf_kalman = adf_test(pd.Series(kalman_residuals), f"{pair_name}_Kalman")
        print(f"  ADF检验: 统计量={adf_kalman['adf_stat']:.4f}, p值={adf_kalman['p_value']:.4f}, 平稳={adf_kalman['is_stationary']}")
        
        # ==================== 方法3: 使用初始β（协整模块给出的） ====================
        print(f"\n3. 使用协整β={beta_init:.4f}:")
        
        # 使用协整模块给出的β计算残差（不估计截距）
        residuals_coint = y_aligned - beta_init * x_aligned
        
        print(f"  残差统计: 均值={residuals_coint.mean():.4f}, 标准差={residuals_coint.std():.4f}")
        
        # ADF检验
        adf_coint = adf_test(residuals_coint, f"{pair_name}_Coint_Beta")
        print(f"  ADF检验: 统计量={adf_coint['adf_stat']:.4f}, p值={adf_coint['p_value']:.4f}, 平稳={adf_coint['is_stationary']}")
        
        # 保存结果
        pair_result = {
            'pair': pair_name,
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'data_points': len(common_dates),
            'x_max': x_aligned.max(),
            'y_max': y_aligned.max(),
            'suspected_log_prices': x_is_log and y_is_log
        }
        
        # 添加各方法结果
        methods = [adf_ols_full, adf_ols_60, adf_kalman, adf_coint]
        for method_result in methods:
            prefix = method_result['name'].split('_', 1)[1]  # 去掉配对名称
            for key, value in method_result.items():
                if key != 'name':
                    pair_result[f"{prefix}_{key}"] = value
        
        results.append(pair_result)
    
    # ==================== 汇总结果 ====================
    print("\n" + "=" * 80)
    print("汇总对比结果")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # 统计各方法的平稳性
        methods = ['OLS_Full', 'OLS_60d', 'Kalman', 'Coint_Beta']
        print("\n平稳性对比:")
        print("-" * 60)
        for method in methods:
            stationary_col = f"{method}_is_stationary"
            if stationary_col in results_df.columns:
                stationary_count = results_df[stationary_col].sum()
                total_count = len(results_df)
                print(f"{method:12s}: {stationary_count}/{total_count} 平稳 ({stationary_count/total_count*100:.1f}%)")
        
        # 详细结果
        print(f"\n详细结果:")
        print("-" * 120)
        for _, row in results_df.iterrows():
            print(f"\n{row['pair']:8s} ({row['data_points']}个点):")
            for method in methods:
                adf_col = f"{method}_adf_stat"
                p_col = f"{method}_p_value"
                stat_col = f"{method}_is_stationary"
                if all(col in row for col in [adf_col, p_col, stat_col]):
                    status = "平稳" if row[stat_col] else "非平稳"
                    print(f"  {method:12s}: ADF={row[adf_col]:7.4f}, p={row[p_col]:7.4f} -> {status}")
        
        # 保存详细结果
        output_file = f"ols_vs_kalman_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("对比完成")
    print("=" * 80)


if __name__ == "__main__":
    main()