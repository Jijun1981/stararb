#!/usr/bin/env python3
"""
分析β值一致性问题
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from sklearn.linear_model import LinearRegression

def analyze_beta_consistency():
    """分析β值一致性问题"""
    
    print("🔍 分析NI-AG配对的β值一致性问题")
    print("=" * 60)
    
    # 1. 加载数据
    data = load_all_symbols_data()
    print(f"数据加载: {data.shape}")
    
    # 2. 手动计算协整β值
    print("\n=== 手动重现协整分析的β值 ===")
    
    # 使用2024年数据进行方向判定
    recent_data = data['2024-01-01':]
    ni_vol_2024 = recent_data['NI'].std()
    ag_vol_2024 = recent_data['AG'].std()
    
    print(f"2024年波动率: NI={ni_vol_2024:.6f}, AG={ag_vol_2024:.6f}")
    print(f"方向判定: {'NI(X)->AG(Y)' if ni_vol_2024 < ag_vol_2024 else 'AG(X)->NI(Y)'}")
    
    # 获取NI-AG的完整时间序列数据
    ni_prices = data['NI'].dropna()  # 对数价格
    ag_prices = data['AG'].dropna()  # 对数价格
    
    # 数据对齐
    common_dates = ni_prices.index.intersection(ag_prices.index)
    ni_aligned = ni_prices[common_dates]
    ag_aligned = ag_prices[common_dates]
    
    print(f"对齐后数据长度: {len(common_dates)}")
    print(f"数据范围: {common_dates[0]} 至 {common_dates[-1]}")
    
    # 使用1年窗口计算β值（与协整分析一致）
    one_year_data = data.iloc[-252:]  # 最近1年数据
    ni_1y = one_year_data['NI'].dropna()
    ag_1y = one_year_data['AG'].dropna()
    
    # 对齐1年数据
    common_1y = ni_1y.index.intersection(ag_1y.index)
    ni_1y_aligned = ni_1y[common_1y]
    ag_1y_aligned = ag_1y[common_1y]
    
    print(f"1年数据长度: {len(common_1y)}")
    
    # OLS回归：AG = α + β * NI (因为方向是NI->AG)
    reg = LinearRegression()
    X = ni_1y_aligned.values.reshape(-1, 1)
    y = ag_1y_aligned.values
    reg.fit(X, y)
    
    beta_1y_manual = reg.coef_[0]
    alpha_1y_manual = reg.intercept_
    r_squared = reg.score(X, y)
    
    print(f"手动计算1年β值: {beta_1y_manual:.6f}")
    print(f"手动计算截距α: {alpha_1y_manual:.6f}")
    print(f"R²: {r_squared:.6f}")
    
    # 3. 对比协整分析结果
    print(f"\n=== 对比协整分析结果 ===")
    analyzer = CointegrationAnalyzer(data)
    
    # 获取协整分析的β值
    results = analyzer.screen_all_pairs(
        screening_windows=['1y'],
        p_thresholds={'1y': 0.05},
        vol_start_date='2024-01-01'
    )
    
    ni_ag_result = results[results['pair'] == 'NI-AG']
    if len(ni_ag_result) > 0:
        coint_beta = ni_ag_result['beta_1y'].iloc[0]
        print(f"协整分析β值: {coint_beta:.6f}")
        print(f"手动vs协整差异: {abs(beta_1y_manual - coint_beta):.6f}")
        
        if abs(beta_1y_manual - coint_beta) < 0.001:
            print("✅ 协整分析β值正确")
        else:
            print("❌ 协整分析β值异常")
    else:
        print("❌ 未找到NI-AG协整结果")
    
    # 4. 分析Kalman滤波的初始化
    print(f"\n=== 分析Kalman滤波初始化 ===")
    
    # 使用信号生成期的数据模拟Kalman滤波初始化
    signal_data = data['2024-02-08':'2025-08-20'].copy()  # 信号生成期数据
    
    ni_signal = signal_data['NI'].dropna()
    ag_signal = signal_data['AG'].dropna()
    
    # 对齐信号期数据
    common_signal = ni_signal.index.intersection(ag_signal.index)
    ni_signal_aligned = ni_signal[common_signal]
    ag_signal_aligned = ag_signal[common_signal]
    
    print(f"信号期数据长度: {len(common_signal)}")
    
    # 模拟OLS预热（前60天）
    ols_window = 60
    ni_warmup = ni_signal_aligned[:ols_window]
    ag_warmup = ag_signal_aligned[:ols_window]
    
    # OLS预热回归
    reg_warmup = LinearRegression()
    X_warmup = ni_warmup.values.reshape(-1, 1)
    y_warmup = ag_warmup.values
    reg_warmup.fit(X_warmup, y_warmup)
    
    beta_warmup = reg_warmup.coef_[0]
    print(f"OLS预热β值: {beta_warmup:.6f}")
    
    # 5. 分析β值变化的合理性
    print(f"\n=== β值变化合理性分析 ===")
    
    print(f"协整分析β值: {coint_beta:.6f}")
    print(f"OLS预热β值: {beta_warmup:.6f}")
    print(f"信号中最终β值: 0.762x (从日志看)")
    
    # 计算不同时期的β值
    periods = [
        ('2024年全年', data['2024-01-01':'2024-12-31']),
        ('2024年2-8月', data['2024-02-08':'2024-08-31']),
        ('2024年9月至今', data['2024-09-01':])
    ]
    
    print(f"\n不同时期的β值:")
    for period_name, period_data in periods:
        if len(period_data) < 10:
            continue
            
        ni_period = period_data['NI'].dropna()
        ag_period = period_data['AG'].dropna()
        
        if len(ni_period) < 10 or len(ag_period) < 10:
            continue
            
        common_period = ni_period.index.intersection(ag_period.index)
        if len(common_period) < 10:
            continue
            
        ni_period_aligned = ni_period[common_period]
        ag_period_aligned = ag_period[common_period]
        
        reg_period = LinearRegression()
        X_period = ni_period_aligned.values.reshape(-1, 1)
        y_period = ag_period_aligned.values
        reg_period.fit(X_period, y_period)
        
        beta_period = reg_period.coef_[0]
        print(f"  {period_name}: β={beta_period:.6f} (n={len(common_period)})")
    
    # 6. 检查是否存在结构性变化
    print(f"\n=== 检查结构性变化 ===")
    
    # 计算滚动β值
    window_size = 60
    rolling_betas = []
    dates = []
    
    for i in range(window_size, len(common_signal)):
        start_idx = i - window_size
        end_idx = i
        
        ni_window = ni_signal_aligned.iloc[start_idx:end_idx]
        ag_window = ag_signal_aligned.iloc[start_idx:end_idx]
        
        if len(ni_window) >= 30:  # 确保有足够数据点
            reg_rolling = LinearRegression()
            X_rolling = ni_window.values.reshape(-1, 1)
            y_rolling = ag_window.values
            reg_rolling.fit(X_rolling, y_rolling)
            
            rolling_betas.append(reg_rolling.coef_[0])
            dates.append(common_signal[end_idx-1])
    
    if rolling_betas:
        rolling_df = pd.DataFrame({
            'date': dates,
            'beta': rolling_betas
        })
        
        print(f"滚动β值范围: {min(rolling_betas):.6f} 至 {max(rolling_betas):.6f}")
        print(f"滚动β值标准差: {np.std(rolling_betas):.6f}")
        
        # 显示几个关键时点的β值
        key_dates = ['2024-07-01', '2024-12-01', '2025-01-01']
        for key_date in key_dates:
            nearest_idx = rolling_df['date'].sub(pd.to_datetime(key_date)).abs().idxmin()
            if nearest_idx < len(rolling_df):
                nearest_beta = rolling_df.iloc[nearest_idx]['beta']
                nearest_date = rolling_df.iloc[nearest_idx]['date']
                print(f"  {key_date}附近({nearest_date.strftime('%Y-%m-%d')}): β={nearest_beta:.6f}")

if __name__ == "__main__":
    analyze_beta_consistency()