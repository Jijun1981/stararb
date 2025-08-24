#!/usr/bin/env python3
"""
核心问题分析：预热期β变异和Kalman vs OLS滚动β对比
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def calculate_rolling_ols_beta(x_data, y_data, window=60):
    """计算滚动OLS β值"""
    betas = []
    for i in range(window, len(x_data)):
        x_window = x_data[i-window:i]
        y_window = y_data[i-window:i]
        
        reg = LinearRegression(fit_intercept=False)  # 与Kalman一致：无截距
        reg.fit(x_window.reshape(-1, 1), y_window)
        betas.append(reg.coef_[0])
    
    return np.array(betas)

def analyze_beta_tracking():
    """分析β值跟踪效果"""
    
    print("🔍 核心问题分析：Kalman vs OLS滚动β对比")
    print("=" * 60)
    
    # 1. 加载数据和最新信号
    data = load_all_symbols_data()
    
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if not signal_files:
        print("❌ 未找到信号文件")
        return
        
    latest_signal_file = max(signal_files)
    signals_df = pd.read_csv(latest_signal_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    print(f"分析信号文件: {latest_signal_file}")
    
    # 2. 选择几个代表性配对进行详细分析
    analysis_pairs = ['NI-AG', 'AU-ZN', 'CU-SN', 'RB-SM', 'ZN-SM']
    
    results = {}
    
    for pair in analysis_pairs:
        if pair not in signals_df['pair'].unique():
            print(f"⚠️ 跳过{pair}：数据中不存在")
            continue
            
        print(f"\n=== {pair}配对分析 ===")
        
        # 获取该配对的信号数据
        pair_signals = signals_df[signals_df['pair'] == pair].copy()
        pair_signals = pair_signals.sort_values('date')
        
        # 获取符号
        symbol_x = pair_signals['symbol_x'].iloc[0]
        symbol_y = pair_signals['symbol_y'].iloc[0]
        
        print(f"配对: {symbol_x} -> {symbol_y}")
        
        # 获取原始价格数据（信号生成期）
        signal_start_date = pair_signals['date'].min()
        signal_end_date = pair_signals['date'].max()
        
        price_data_period = data[signal_start_date:signal_end_date]
        
        if symbol_x not in price_data_period.columns or symbol_y not in price_data_period.columns:
            print(f"❌ 价格数据不完整")
            continue
            
        # 对齐价格数据
        x_prices = price_data_period[symbol_x].dropna()
        y_prices = price_data_period[symbol_y].dropna()
        common_dates = x_prices.index.intersection(y_prices.index)
        
        x_aligned = x_prices[common_dates].values
        y_aligned = y_prices[common_dates].values
        dates_aligned = common_dates
        
        print(f"价格数据点数: {len(x_aligned)}")
        
        # 3. 计算滚动60天OLS β
        rolling_betas = calculate_rolling_ols_beta(x_aligned, y_aligned, window=60)
        rolling_dates = dates_aligned[60:]  # 对应滚动β的日期
        
        print(f"滚动OLS β范围: [{rolling_betas.min():.6f}, {rolling_betas.max():.6f}]")
        print(f"滚动OLS β标准差: {np.std(rolling_betas):.6f}")
        
        # 4. 对齐Kalman β值
        # 匹配日期
        kalman_betas = []
        kalman_dates_matched = []
        ols_betas_matched = []
        
        for i, date in enumerate(rolling_dates):
            # 找到最接近的Kalman信号日期
            signal_on_date = pair_signals[pair_signals['date'] == date]
            if len(signal_on_date) > 0:
                kalman_beta = signal_on_date['beta'].iloc[0]
                kalman_betas.append(kalman_beta)
                kalman_dates_matched.append(date)
                ols_betas_matched.append(rolling_betas[i])
        
        if len(kalman_betas) < 10:
            print(f"❌ 匹配的数据点太少: {len(kalman_betas)}")
            continue
            
        kalman_betas = np.array(kalman_betas)
        ols_betas_matched = np.array(ols_betas_matched)
        
        print(f"匹配数据点数: {len(kalman_betas)}")
        
        # 5. 关键分析：相关性
        correlation, p_value = pearsonr(kalman_betas, ols_betas_matched)
        
        print(f"🎯 Kalman vs OLS相关性: {correlation:.4f} (p={p_value:.4e})")
        
        # 6. 偏差分析
        beta_diff = kalman_betas - ols_betas_matched
        mean_diff = np.mean(beta_diff)
        std_diff = np.std(beta_diff)
        max_diff = np.max(np.abs(beta_diff))
        
        print(f"平均偏差: {mean_diff:.6f}")
        print(f"偏差标准差: {std_diff:.6f}")
        print(f"最大绝对偏差: {max_diff:.6f}")
        
        # 7. 预热期β变异分析
        beta_initial = pair_signals['beta_initial'].iloc[0]
        
        # 预热期结束后的第一个β值（应该接近OLS预热值）
        warmup_signals = pair_signals[pair_signals['phase'] == 'warm_up']
        signal_period_signals = pair_signals[pair_signals['phase'] == 'signal_period']
        
        if len(warmup_signals) > 0 and len(signal_period_signals) > 0:
            beta_after_warmup = signal_period_signals['beta'].iloc[0]
            
            print(f"β初始值(协整): {beta_initial:.6f}")
            print(f"预热结束后β值: {beta_after_warmup:.6f}")
            print(f"预热期β变化: {abs(beta_after_warmup - beta_initial):.6f}")
            
            # 理论上预热结束后的β应该接近OLS预热期的β
            # 计算预热期的理论OLS β
            if len(x_aligned) >= 60:
                x_warmup = x_aligned[:60] - np.mean(x_aligned[:60])  # 去中心化
                y_warmup = y_aligned[:60] - np.mean(y_aligned[:60])
                
                reg_warmup = LinearRegression(fit_intercept=False)
                reg_warmup.fit(x_warmup.reshape(-1, 1), y_warmup)
                theoretical_warmup_beta = reg_warmup.coef_[0]
                
                print(f"理论OLS预热β: {theoretical_warmup_beta:.6f}")
                print(f"实际vs理论预热差异: {abs(beta_after_warmup - theoretical_warmup_beta):.6f}")
        
        # 8. 跟踪质量评估
        if correlation > 0.8:
            tracking_quality = "优秀"
        elif correlation > 0.6:
            tracking_quality = "良好"
        elif correlation > 0.4:
            tracking_quality = "一般"
        else:
            tracking_quality = "差"
            
        print(f"📊 跟踪质量: {tracking_quality}")
        
        # 保存结果
        results[pair] = {
            'correlation': correlation,
            'p_value': p_value,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'max_diff': max_diff,
            'tracking_quality': tracking_quality,
            'n_points': len(kalman_betas),
            'beta_initial': beta_initial,
            'beta_range_kalman': [kalman_betas.min(), kalman_betas.max()],
            'beta_range_ols': [ols_betas_matched.min(), ols_betas_matched.max()],
            'kalman_betas': kalman_betas,
            'ols_betas': ols_betas_matched,
            'dates': kalman_dates_matched
        }
    
    # 9. 总体分析
    print(f"\n=== 总体分析结果 ===")
    
    if results:
        correlations = [r['correlation'] for r in results.values()]
        mean_correlations = np.mean(correlations)
        
        print(f"平均相关性: {mean_correlations:.4f}")
        print(f"相关性范围: [{min(correlations):.4f}, {max(correlations):.4f}]")
        
        # 质量分布
        qualities = [r['tracking_quality'] for r in results.values()]
        from collections import Counter
        quality_count = Counter(qualities)
        
        print(f"跟踪质量分布:")
        for quality, count in quality_count.items():
            print(f"  {quality}: {count}个配对")
        
        # 找出问题配对
        problem_pairs = [pair for pair, result in results.items() if result['correlation'] < 0.6]
        if problem_pairs:
            print(f"\n⚠️ 需要关注的配对 (相关性<0.6):")
            for pair in problem_pairs:
                r = results[pair]
                print(f"  {pair}: 相关性={r['correlation']:.4f}, 最大偏差={r['max_diff']:.4f}")
        
        # 保存详细结果
        summary_data = []
        for pair, result in results.items():
            summary_data.append({
                'pair': pair,
                'correlation': result['correlation'],
                'p_value': result['p_value'],
                'mean_diff': result['mean_diff'],
                'std_diff': result['std_diff'],
                'max_diff': result['max_diff'],
                'tracking_quality': result['tracking_quality'],
                'n_points': result['n_points'],
                'beta_initial': result['beta_initial'],
                'kalman_beta_min': result['beta_range_kalman'][0],
                'kalman_beta_max': result['beta_range_kalman'][1],
                'ols_beta_min': result['beta_range_ols'][0],
                'ols_beta_max': result['beta_range_ols'][1]
            })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = f"kalman_ols_beta_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\n📊 详细分析结果已保存: {output_file}")
        
        # 10. 关键结论
        print(f"\n🎯 关键结论:")
        
        if mean_correlations > 0.8:
            print("✅ Kalman滤波器整体跟踪效果优秀，β值变异合理")
        elif mean_correlations > 0.6:
            print("✅ Kalman滤波器整体跟踪效果良好，可接受范围内")
        elif mean_correlations > 0.4:
            print("⚠️ Kalman滤波器跟踪效果一般，需要参数调优")
        else:
            print("❌ Kalman滤波器跟踪效果差，存在系统性问题")
            
        if len(problem_pairs) > 0:
            print(f"⚠️ {len(problem_pairs)}个配对需要特别关注")
        
        return results, summary_df
    else:
        print("❌ 无法完成分析，数据不足")
        return None, None

if __name__ == "__main__":
    results, summary = analyze_beta_tracking()