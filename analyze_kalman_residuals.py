#!/usr/bin/env python3
"""
分析Kalman滤波后残差的平稳程度
检查：ADF单位根检验、KPSS检验、残差自相关、方差齐性
"""
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def adf_test(series, name=""):
    """ADF单位根检验"""
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'name': name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}

def kpss_test(series, name=""):
    """KPSS平稳性检验"""
    try:
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return {
            'name': name,
            'kpss_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # KPSS零假设是平稳的
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}

def ljung_box_test(series, lags=10, name=""):
    """Ljung-Box自相关检验"""
    try:
        result = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
        # 检查是否有显著自相关
        significant_lags = result['lb_pvalue'] < 0.05
        return {
            'name': name,
            'has_autocorr': significant_lags.any(),
            'min_p_value': result['lb_pvalue'].min(),
            'significant_lags': significant_lags.sum()
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}

def variance_stability_test(series, window=60, name=""):
    """方差齐性检验(滚动方差稳定性)"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < window * 2:
            return {'name': name, 'error': 'insufficient data'}
            
        # 计算滚动方差
        rolling_var = series_clean.rolling(window=window).var().dropna()
        
        # Levene方差齐性检验 - 分前后两半
        mid = len(rolling_var) // 2
        first_half = rolling_var[:mid]
        second_half = rolling_var[mid:]
        
        levene_stat, levene_p = stats.levene(first_half, second_half)
        
        return {
            'name': name,
            'rolling_var_mean': rolling_var.mean(),
            'rolling_var_std': rolling_var.std(),
            'var_cv': rolling_var.std() / rolling_var.mean(),  # 变异系数
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'variance_stable': levene_p > 0.05
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}

def analyze_residual_stationarity():
    """分析Kalman滤波后残差的平稳程度"""
    
    print("🔍 Kalman滤波后残差平稳性分析")
    print("=" * 60)
    
    # 加载最新信号文件
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if not signal_files:
        print("❌ 未找到信号文件")
        return
        
    latest_signal_file = max(signal_files)
    signals_df = pd.read_csv(latest_signal_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    print(f"分析信号文件: {latest_signal_file}")
    
    # 选择代表性配对进行分析
    analysis_pairs = ['AU-ZN', 'CU-SN', 'ZN-SM', 'RB-SM', 'SS-NI']
    
    results = {}
    
    for pair in analysis_pairs:
        if pair not in signals_df['pair'].unique():
            print(f"⚠️ 跳过{pair}：数据中不存在")
            continue
            
        print(f"\n=== {pair}配对残差分析 ===")
        
        # 获取该配对的信号期残差
        pair_signals = signals_df[
            (signals_df['pair'] == pair) & 
            (signals_df['phase'] == 'signal_period')
        ].copy()
        pair_signals = pair_signals.sort_values('date')
        
        if len(pair_signals) < 60:
            print(f"❌ 数据点不足: {len(pair_signals)}")
            continue
            
        # 提取残差(innovation)和z_score
        residuals = pair_signals['innovation'].values
        z_scores = pair_signals['z_score'].values
        
        print(f"残差数据点数: {len(residuals)}")
        print(f"残差统计: 均值={np.mean(residuals):.6f}, 标准差={np.std(residuals):.6f}")
        print(f"Z-score统计: 均值={np.mean(z_scores):.6f}, 标准差={np.std(z_scores):.6f}")
        
        # 1. ADF单位根检验
        adf_residual = adf_test(pd.Series(residuals), f"{pair}_residual")
        print(f"ADF检验(残差): 统计量={adf_residual.get('adf_statistic', 'N/A'):.4f}, "
              f"p值={adf_residual.get('p_value', 'N/A'):.4f}, "
              f"平稳={'✅' if adf_residual.get('is_stationary', False) else '❌'}")
        
        # 2. KPSS检验
        kpss_residual = kpss_test(pd.Series(residuals), f"{pair}_residual")
        print(f"KPSS检验(残差): 统计量={kpss_residual.get('kpss_statistic', 'N/A'):.4f}, "
              f"p值={kpss_residual.get('p_value', 'N/A'):.4f}, "
              f"平稳={'✅' if kpss_residual.get('is_stationary', False) else '❌'}")
        
        # 3. 对z_score也做检验
        adf_zscore = adf_test(pd.Series(z_scores), f"{pair}_zscore")
        kpss_zscore = kpss_test(pd.Series(z_scores), f"{pair}_zscore")
        
        print(f"ADF检验(Z-score): p值={adf_zscore.get('p_value', 'N/A'):.4f}, "
              f"平稳={'✅' if adf_zscore.get('is_stationary', False) else '❌'}")
        print(f"KPSS检验(Z-score): p值={kpss_zscore.get('p_value', 'N/A'):.4f}, "
              f"平稳={'✅' if kpss_zscore.get('is_stationary', False) else '❌'}")
        
        # 4. Ljung-Box自相关检验
        ljung_residual = ljung_box_test(pd.Series(residuals), lags=min(10, len(residuals)//4), name=f"{pair}_residual")
        print(f"Ljung-Box检验(残差): 最小p值={ljung_residual.get('min_p_value', 'N/A'):.4f}, "
              f"自相关={'❌' if ljung_residual.get('has_autocorr', False) else '✅'}")
        
        # 5. 方差齐性检验
        var_test = variance_stability_test(pd.Series(residuals), window=30, name=f"{pair}_residual")
        print(f"方差稳定性: CV={var_test.get('var_cv', 'N/A'):.4f}, "
              f"Levene p值={var_test.get('levene_p_value', 'N/A'):.4f}, "
              f"方差齐性={'✅' if var_test.get('variance_stable', False) else '❌'}")
        
        # 6. 综合评价
        residual_stationary = (adf_residual.get('is_stationary', False) and 
                             kpss_residual.get('is_stationary', False))
        zscore_stationary = (adf_zscore.get('is_stationary', False) and 
                           kpss_zscore.get('is_stationary', False))
        no_autocorr = not ljung_residual.get('has_autocorr', True)
        var_stable = var_test.get('variance_stable', False)
        
        quality_score = sum([residual_stationary, zscore_stationary, no_autocorr, var_stable])
        
        if quality_score >= 3:
            quality = "优秀"
        elif quality_score >= 2:
            quality = "良好"
        elif quality_score >= 1:
            quality = "一般"
        else:
            quality = "差"
            
        print(f"📊 残差质量: {quality} ({quality_score}/4)")
        
        # 保存结果
        results[pair] = {
            'pair': pair,
            'n_points': len(residuals),
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'zscore_mean': np.mean(z_scores),
            'zscore_std': np.std(z_scores),
            'adf_residual_pvalue': adf_residual.get('p_value'),
            'kpss_residual_pvalue': kpss_residual.get('p_value'),
            'adf_zscore_pvalue': adf_zscore.get('p_value'),
            'kpss_zscore_pvalue': kpss_zscore.get('p_value'),
            'ljung_box_min_pvalue': ljung_residual.get('min_p_value'),
            'variance_cv': var_test.get('var_cv'),
            'levene_pvalue': var_test.get('levene_p_value'),
            'residual_stationary': residual_stationary,
            'zscore_stationary': zscore_stationary,
            'no_autocorr': no_autocorr,
            'variance_stable': var_stable,
            'quality_score': quality_score,
            'quality': quality
        }
    
    # 总体分析
    print(f"\n=== 总体残差平稳性分析 ===")
    
    if results:
        quality_scores = [r['quality_score'] for r in results.values()]
        mean_quality = np.mean(quality_scores)
        
        print(f"平均质量评分: {mean_quality:.2f}/4")
        
        # 质量分布
        from collections import Counter
        quality_dist = Counter([r['quality'] for r in results.values()])
        print(f"质量分布:")
        for quality, count in quality_dist.items():
            print(f"  {quality}: {count}个配对")
        
        # 各项指标通过率
        stationary_rate = sum([r['residual_stationary'] for r in results.values()]) / len(results)
        zscore_stationary_rate = sum([r['zscore_stationary'] for r in results.values()]) / len(results)
        no_autocorr_rate = sum([r['no_autocorr'] for r in results.values()]) / len(results)
        var_stable_rate = sum([r['variance_stable'] for r in results.values()]) / len(results)
        
        print(f"\n各项指标通过率:")
        print(f"  残差平稳性: {stationary_rate*100:.1f}%")
        print(f"  Z-score平稳性: {zscore_stationary_rate*100:.1f}%")
        print(f"  无自相关: {no_autocorr_rate*100:.1f}%")
        print(f"  方差稳定: {var_stable_rate*100:.1f}%")
        
        # 找出问题配对
        problem_pairs = [pair for pair, result in results.items() if result['quality_score'] <= 1]
        if problem_pairs:
            print(f"\n⚠️ 需要关注的配对 (评分≤1):")
            for pair in problem_pairs:
                r = results[pair]
                print(f"  {pair}: 评分={r['quality_score']}/4, 残差std={r['residual_std']:.4f}")
        
        # 保存详细结果
        summary_data = []
        for pair, result in results.items():
            summary_data.append(result)
        
        summary_df = pd.DataFrame(summary_data)
        output_file = f"residual_stationarity_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\n📊 详细残差分析结果已保存: {output_file}")
        
        # 关键结论
        print(f"\n🎯 关键结论:")
        
        if mean_quality >= 3:
            print("✅ Kalman滤波残差整体平稳性优秀")
        elif mean_quality >= 2:
            print("✅ Kalman滤波残差整体平稳性良好")
        elif mean_quality >= 1:
            print("⚠️ Kalman滤波残差平稳性一般，需要优化")
        else:
            print("❌ Kalman滤波残差平稳性差，存在系统性问题")
            
        if stationary_rate < 0.7:
            print("⚠️ 残差平稳性不足，可能需要调整Kalman参数")
        if no_autocorr_rate < 0.7:
            print("⚠️ 残差存在显著自相关，滤波效果不理想")
        if var_stable_rate < 0.7:
            print("⚠️ 残差方差不稳定，可能存在异方差问题")
        
        return results, summary_df
    else:
        print("❌ 无法完成分析，数据不足")
        return None, None

if __name__ == "__main__":
    results, summary = analyze_residual_stationarity()