#!/usr/bin/env python3
"""
对比原始协整残差 vs Kalman滤波残差的平稳性
诊断Kalman滤波后残差为什么不平稳
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')

from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def compare_residual_stationarity():
    """对比原始协整残差vs Kalman滤波残差"""
    
    print("🔍 协整残差 vs Kalman残差平稳性对比")
    print("=" * 60)
    
    # 1. 加载数据
    data = load_all_symbols_data()
    signals_df = pd.read_csv('signals_e2e_20250824_182241.csv')
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # 2. 选择几个代表性配对
    test_pairs = ['AU-ZN', 'CU-SN', 'RB-SM']
    
    for pair in test_pairs:
        if pair not in signals_df['pair'].unique():
            continue
            
        print(f"\n=== {pair} 残差对比分析 ===")
        
        pair_info = signals_df[signals_df['pair'] == pair].iloc[0]
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        
        print(f"配对: {symbol_x} -> {symbol_y}")
        
        # 获取信号期数据
        pair_signals = signals_df[
            (signals_df['pair'] == pair) & 
            (signals_df['phase'] == 'signal_period')
        ].copy()
        pair_signals = pair_signals.sort_values('date')
        
        if len(pair_signals) < 60:
            continue
            
        # 获取价格数据对齐到信号期
        signal_dates = pair_signals['date'].values
        price_data_signal = data.loc[signal_dates[[0, -1]]]
        
        # 确保有足够数据
        extended_data = data[signal_dates[0]:signal_dates[-1]]
        
        if symbol_x not in extended_data.columns or symbol_y not in extended_data.columns:
            print(f"❌ 价格数据不完整")
            continue
            
        # 对齐数据
        x_prices = extended_data[symbol_x].dropna()
        y_prices = extended_data[symbol_y].dropna()
        common_dates = x_prices.index.intersection(y_prices.index)
        
        x_aligned = x_prices[common_dates].values
        y_aligned = y_prices[common_dates].values
        dates_aligned = common_dates
        
        print(f"价格数据点数: {len(x_aligned)}")
        
        # 3. 计算原始协整残差
        # 使用1年窗口的β估计协整残差
        analyzer = CointegrationAnalyzer(data)
        
        # 计算整个信号期的静态协整残差
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(x_aligned.reshape(-1, 1), y_aligned)
        static_beta = reg.coef_[0]
        
        static_residuals = y_aligned - static_beta * x_aligned
        
        print(f"静态协整β: {static_beta:.6f}")
        print(f"静态协整残差统计: 均值={np.mean(static_residuals):.6f}, 标准差={np.std(static_residuals):.6f}")
        
        # 4. 获取Kalman滤波残差
        kalman_innovations = pair_signals['innovation'].values
        kalman_dates = pair_signals['date'].values
        
        print(f"Kalman残差统计: 均值={np.mean(kalman_innovations):.6f}, 标准差={np.std(kalman_innovations):.6f}")
        
        # 5. 平稳性检验对比
        def test_stationarity(series, name):
            try:
                result = adfuller(series, autolag='AIC')
                return {
                    'name': name,
                    'adf_stat': result[0],
                    'p_value': result[1],
                    'is_stationary': result[1] < 0.05
                }
            except:
                return {'name': name, 'error': True}
        
        # 对齐到相同的时间段进行比较
        signal_period_dates = pair_signals['date']
        
        # 从原始数据中提取对应时间段的静态残差
        static_residuals_aligned = []
        for date in signal_period_dates:
            try:
                idx = list(dates_aligned).index(date)
                static_residuals_aligned.append(static_residuals[idx])
            except ValueError:
                # 如果日期不匹配，跳过
                continue
        
        if len(static_residuals_aligned) < 50:
            print("❌ 对齐数据不足")
            continue
            
        static_residuals_aligned = np.array(static_residuals_aligned[:len(kalman_innovations)])
        
        # 平稳性检验
        static_test = test_stationarity(static_residuals_aligned, '静态协整残差')
        kalman_test = test_stationarity(kalman_innovations, 'Kalman滤波残差')
        
        print(f"\\n平稳性检验结果:")
        print(f"  静态协整残差: ADF={static_test.get('adf_stat', 'N/A'):.4f}, p={static_test.get('p_value', 'N/A'):.4f}, 平稳={'✅' if static_test.get('is_stationary', False) else '❌'}")
        print(f"  Kalman滤波残差: ADF={kalman_test.get('adf_stat', 'N/A'):.4f}, p={kalman_test.get('p_value', 'N/A'):.4f}, 平稳={'✅' if kalman_test.get('is_stationary', False) else '❌'}")
        
        # 6. 残差标准差对比
        static_std = np.std(static_residuals_aligned)
        kalman_std = np.std(kalman_innovations)
        
        print(f"\\n残差标准差对比:")
        print(f"  静态协整残差: {static_std:.6f}")
        print(f"  Kalman滤波残差: {kalman_std:.6f}")
        print(f"  标准差比值: {kalman_std/static_std:.4f}")
        
        # 7. 诊断Kalman残差问题
        print(f"\\n🔬 Kalman残差问题诊断:")
        
        # 检查β值变化程度
        beta_values = pair_signals['beta'].values
        beta_range = beta_values.max() - beta_values.min()
        beta_cv = np.std(beta_values) / np.mean(beta_values)
        
        print(f"  β值范围: [{beta_values.min():.4f}, {beta_values.max():.4f}]")
        print(f"  β值变化: {beta_range:.4f}")
        print(f"  β值变异系数: {beta_cv:.4f}")
        
        # 检查innovation的趋势
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(kalman_innovations)), kalman_innovations)
        print(f"  Innovation趋势斜率: {slope:.6f}")
        print(f"  趋势显著性p值: {p_value:.4f}")
        
        if abs(slope) > 1e-4 and p_value < 0.05:
            print("  ⚠️ Kalman残差存在显著趋势，可能是β估计滞后导致")
            
        # 检查R值(测量噪声方差)变化
        R_values = pair_signals['R'].values
        R_cv = np.std(R_values) / np.mean(R_values)
        print(f"  测量噪声R变异系数: {R_cv:.4f}")
        
        if R_cv > 0.5:
            print("  ⚠️ 测量噪声方差变化较大，可能影响滤波效果")
        
        # 8. 结论
        print(f"\\n📊 {pair} 结论:")
        
        if static_test.get('is_stationary', False) and not kalman_test.get('is_stationary', False):
            print("  ❌ 原始协整残差平稳，但Kalman残差不平稳")
            print("  → 可能原因：Kalman滤波参数不当或β估计滞后")
        elif not static_test.get('is_stationary', False) and not kalman_test.get('is_stationary', False):
            print("  ❌ 原始协整和Kalman残差都不平稳")
            print("  → 可能原因：配对本身协整关系不稳定")
        elif kalman_test.get('is_stationary', False):
            print("  ✅ Kalman残差平稳，滤波效果良好")
        
        if kalman_std > static_std * 1.5:
            print("  ⚠️ Kalman残差标准差明显大于静态残差")
            print("  → 建议：检查δ参数是否过小，导致跟踪过于敏感")
        elif kalman_std < static_std * 0.5:
            print("  ⚠️ Kalman残差标准差明显小于静态残差")
            print("  → 建议：检查δ参数是否过大，导致跟踪过于缓慢")

if __name__ == "__main__":
    compare_residual_stationarity()