#!/usr/bin/env python3
"""
简化的参数测试 - 快速验证几个关键参数组合
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveKalmanFilter
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def test_parameter_combinations():
    """测试几个关键参数组合"""
    
    print("🔧 简化Kalman参数测试")
    print("=" * 60)
    
    # 参数组合
    param_combinations = [
        (0.96, 0.92, "原始参数"),
        (0.93, 0.89, "平衡参数"),  
        (0.90, 0.85, "激进参数"),
        (0.92, 0.88, "优化参数1"),
        (0.94, 0.90, "保守参数")
    ]
    
    # 测试配对
    test_pairs = [
        ('CU-SN', 'CU', 'SN'),  # 优秀配对
        ('AU-ZN', 'AU', 'ZN'),  # 问题配对
        ('ZN-SM', 'ZN', 'SM')   # 中等配对
    ]
    
    # 加载数据
    data = load_all_symbols_data()
    
    results = []
    
    for delta, lambda_r, desc in param_combinations:
        print(f"\\n=== 测试 {desc}: δ={delta}, λ={lambda_r} ===")
        
        combo_results = []
        
        for pair, symbol_x, symbol_y in test_pairs:
            try:
                result = test_single_pair(pair, symbol_x, symbol_y, delta, lambda_r, data)
                if result:
                    combo_results.append(result)
                    print(f"  {pair}: Z>2={result['z_gt2_ratio']*100:.1f}%, IR={result['ir']:.3f}, "
                          f"OLS相关={result['ols_correlation']:.3f}, 平稳={'✅' if result['is_stationary'] else '❌'}")
            except Exception as e:
                print(f"  {pair}: 测试失败 - {e}")
        
        if combo_results:
            # 汇总结果
            avg_z_ratio = np.mean([r['z_gt2_ratio'] for r in combo_results])
            avg_ir = np.mean([r['ir'] for r in combo_results])
            avg_ols_corr = np.mean([r['ols_correlation'] for r in combo_results])
            stationary_ratio = np.mean([r['is_stationary'] for r in combo_results])
            
            results.append({
                'desc': desc,
                'delta': delta,
                'lambda_r': lambda_r,
                'avg_z_ratio': avg_z_ratio,
                'avg_ir': avg_ir,
                'avg_ols_corr': avg_ols_corr,
                'stationary_ratio': stationary_ratio,
                'valid_pairs': len(combo_results)
            })
            
            print(f"  汇总: Z>2比例={avg_z_ratio*100:.1f}%, IR={avg_ir:.3f}, "
                  f"OLS相关={avg_ols_corr:.3f}, 平稳率={stationary_ratio*100:.0f}%")
    
    # 结果分析
    if results:
        print(f"\\n📊 参数对比结果:")
        print("-" * 80)
        print(f"{'参数组合':<12} {'δ':<5} {'λ':<5} {'Z>2%':<6} {'IR':<7} {'OLS相关':<8} {'平稳率':<6} {'配对数':<4}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['desc']:<12} {r['delta']:<5.2f} {r['lambda_r']:<5.2f} "
                  f"{r['avg_z_ratio']*100:<6.1f} {r['avg_ir']:<7.3f} "
                  f"{r['avg_ols_corr']:<8.3f} {r['stationary_ratio']*100:<6.0f} {r['valid_pairs']:<4}")
        
        # 推荐参数
        print(f"\\n🎯 推荐分析:")
        
        # 按不同目标推荐
        z_target = [r for r in results if 0.02 <= r['avg_z_ratio'] <= 0.05]
        ir_best = max(results, key=lambda x: x['avg_ir'])
        corr_best = max(results, key=lambda x: x['avg_ols_corr'])
        stability_best = max(results, key=lambda x: x['stationary_ratio'])
        
        if z_target:
            z_best = max(z_target, key=lambda x: x['avg_ir'])
            print(f"Z>2在目标范围(2-5%)的最佳IR: {z_best['desc']} (IR={z_best['avg_ir']:.3f})")
        
        print(f"最高IR: {ir_best['desc']} (IR={ir_best['avg_ir']:.3f})")
        print(f"最高OLS相关性: {corr_best['desc']} (相关性={corr_best['avg_ols_corr']:.3f})")
        print(f"最高平稳率: {stability_best['desc']} (平稳率={stability_best['stationary_ratio']*100:.0f}%)")
        
        return results
    else:
        print("\\n❌ 所有参数组合都失败了")
        return None

def test_single_pair(pair, symbol_x, symbol_y, delta, lambda_r, data):
    """测试单个配对"""
    
    # 数据准备
    data_start_date = '2024-02-08'
    signal_end_date = '2025-08-20'
    analysis_data = data[data_start_date:signal_end_date]
    
    if symbol_x not in analysis_data.columns or symbol_y not in analysis_data.columns:
        return None
    
    x_prices = analysis_data[symbol_x].dropna()
    y_prices = analysis_data[symbol_y].dropna()
    common_dates = x_prices.index.intersection(y_prices.index)
    
    if len(common_dates) < 150:
        return None
    
    x_data = x_prices[common_dates].values
    y_data = y_prices[common_dates].values
    
    # Kalman滤波
    kf = AdaptiveKalmanFilter(pair_name=pair, delta=delta, lambda_r=lambda_r)
    kf.warm_up_ols(x_data, y_data, 60)
    
    z_scores = []
    innovations = []
    beta_values = []
    
    # 信号期数据
    warmup_end = 90
    for i in range(warmup_end, len(x_data)):
        result = kf.update(y_data[i], x_data[i])
        z_scores.append(result['z'])
        innovations.append(result['v'])
        beta_values.append(result['beta'])
    
    if len(z_scores) < 50:
        return None
    
    # 计算指标
    z_scores = np.array(z_scores)
    
    # 1. Z>2比例
    z_gt2_ratio = np.sum(np.abs(z_scores) > 2.0) / len(z_scores)
    
    # 2. IR (信息比率)
    returns_proxy = -np.diff(z_scores)
    ir = np.mean(returns_proxy) / (np.std(returns_proxy) + 1e-8) if len(returns_proxy) > 0 else 0.0
    
    # 3. OLS相关性
    if len(x_data) >= 150:
        rolling_betas = []
        for i in range(60, len(x_data)):
            x_window = x_data[i-60:i]
            y_window = y_data[i-60:i]
            reg = LinearRegression(fit_intercept=False)
            reg.fit(x_window.reshape(-1, 1), y_window)
            rolling_betas.append(reg.coef_[0])
        
        # 对齐长度
        min_len = min(len(beta_values), len(rolling_betas))
        kalman_betas_aligned = beta_values[:min_len]
        rolling_betas_aligned = rolling_betas[:min_len]
        
        if min_len > 10:
            ols_correlation, _ = pearsonr(kalman_betas_aligned, rolling_betas_aligned)
        else:
            ols_correlation = 0.0
    else:
        ols_correlation = 0.0
    
    # 4. 平稳性
    try:
        adf_result = adfuller(innovations, autolag='AIC')
        adf_pvalue = adf_result[1]
        is_stationary = adf_pvalue < 0.05
    except:
        adf_pvalue = 1.0
        is_stationary = False
    
    return {
        'pair': pair,
        'z_gt2_ratio': z_gt2_ratio,
        'ir': ir,
        'ols_correlation': ols_correlation,
        'adf_pvalue': adf_pvalue,
        'is_stationary': is_stationary,
        'innovation_std': np.std(innovations),
        'beta_stability': np.std(beta_values) / np.mean(beta_values) if np.mean(beta_values) != 0 else np.inf
    }

if __name__ == "__main__":
    results = test_parameter_combinations()