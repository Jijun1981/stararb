#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成残差验证脚本
对比Kalman滤波与OLS的残差平稳性
验证信号生成模块的残差质量
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.data import load_all_symbols_data


def adf_test(residuals, name):
    """ADF平稳性检验"""
    try:
        clean_residuals = residuals.dropna()
        if len(clean_residuals) < 20:
            return {'name': name, 'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False, 'error': 'insufficient_data'}
        
        result = adfuller(clean_residuals, autolag='AIC')
        return {
            'name': name,
            'n_obs': len(clean_residuals),
            'adf_stat': result[0],
            'p_value': result[1],
            'critical_5%': result[4]['5%'],
            'is_stationary': result[1] < 0.05,
            'residual_mean': clean_residuals.mean(),
            'residual_std': clean_residuals.std()
        }
    except Exception as e:
        return {'name': name, 'adf_stat': np.nan, 'p_value': np.nan, 'is_stationary': False, 'error': str(e)}


def main():
    print("=" * 80)
    print("信号生成残差验证 - Kalman vs OLS")
    print("=" * 80)
    
    # 1. 加载最新的信号数据
    print("加载最新信号数据...")
    try:
        # 找到最新的信号文件
        signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
        if not signal_files:
            print("未找到信号文件")
            return
        
        latest_signal_file = sorted(signal_files)[-1]
        print(f"使用信号文件: {latest_signal_file}")
        
        signals_df = pd.read_csv(latest_signal_file)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        
        # 只分析信号期数据
        signal_period_df = signals_df[signals_df['phase'] == 'signal_period'].copy()
        
        print(f"信号期数据: {len(signal_period_df)}个数据点")
        print(f"包含配对: {signal_period_df['pair'].nunique()}个")
        
    except Exception as e:
        print(f"加载信号数据失败: {e}")
        return
    
    # 2. 加载对数价格数据（与信号生成使用相同数据）
    print("\\n加载对数价格数据...")
    price_data = load_all_symbols_data()  # 对数价格
    print(f"价格数据确认为对数价格: {price_data.max().max() < 20}")
    
    # 3. 选择有交易信号的配对进行验证
    trading_pairs = signal_period_df[signal_period_df['signal'].isin(['open_long', 'open_short', 'close'])]
    active_pairs = trading_pairs['pair'].value_counts().head(10)  # 取前10个最活跃的配对
    
    print(f"\\n选择{len(active_pairs)}个最活跃的配对进行验证:")
    for pair, count in active_pairs.items():
        print(f"  {pair}: {count}个交易信号")
    
    # 4. 对每个配对进行残差分析
    results = []
    
    for pair_name in active_pairs.index:
        print(f"\\n{'='*60}")
        print(f"分析配对: {pair_name}")
        print(f"{'='*60}")
        
        # 获取配对数据
        pair_signals = signal_period_df[signal_period_df['pair'] == pair_name].copy()
        if len(pair_signals) == 0:
            continue
            
        symbol_x = pair_signals['symbol_x'].iloc[0]
        symbol_y = pair_signals['symbol_y'].iloc[0]
        beta_initial = pair_signals['beta_initial'].iloc[0]
        
        print(f"配对: {symbol_x}-{symbol_y}, 协整β: {beta_initial:.4f}")
        
        # 获取相应时间段的价格数据
        start_date = pair_signals['date'].min()
        end_date = pair_signals['date'].max()
        
        # 对齐价格数据
        period_price_data = price_data[start_date:end_date]
        
        if symbol_x not in period_price_data.columns or symbol_y not in period_price_data.columns:
            print(f"  跳过: 缺少价格数据")
            continue
        
        x_prices = period_price_data[symbol_x].dropna()
        y_prices = period_price_data[symbol_y].dropna()
        
        # 对齐数据
        common_dates = x_prices.index.intersection(y_prices.index).intersection(pair_signals['date'])
        if len(common_dates) < 30:
            print(f"  跳过: 有效数据不足 ({len(common_dates)}个点)")
            continue
        
        x_aligned = x_prices[common_dates]
        y_aligned = y_prices[common_dates]
        
        print(f"  分析期间: {common_dates[0].date()} 至 {common_dates[-1].date()} ({len(common_dates)}个点)")
        
        # ========== 方法1: 全数据OLS ==========
        print("\\n1. 全数据OLS回归:")
        reg_full = LinearRegression(fit_intercept=True)
        reg_full.fit(x_aligned.values.reshape(-1, 1), y_aligned.values)
        
        alpha_full = reg_full.intercept_
        beta_full = reg_full.coef_[0]
        
        residuals_ols_full = y_aligned - (alpha_full + beta_full * x_aligned)
        
        print(f"  α={alpha_full:.4f}, β={beta_full:.4f}")
        print(f"  残差: 均值={residuals_ols_full.mean():.6f}, 标准差={residuals_ols_full.std():.4f}")
        
        adf_ols_full = adf_test(residuals_ols_full, f"{pair_name}_OLS_Full")
        print(f"  ADF: 统计量={adf_ols_full['adf_stat']:.4f}, p={adf_ols_full['p_value']:.4f}, 平稳={adf_ols_full['is_stationary']}")
        
        # ========== 方法2: Kalman滤波残差 ==========
        print("\\n2. Kalman滤波残差:")
        
        # 从信号数据中提取Kalman的创新值（残差）
        pair_signals_aligned = pair_signals.set_index('date').reindex(common_dates).fillna(method='ffill')
        kalman_residuals = pair_signals_aligned['innovation'].values
        kalman_betas = pair_signals_aligned['beta'].values
        
        # 去除NaN值
        valid_idx = ~(np.isnan(kalman_residuals) | np.isnan(kalman_betas))
        if valid_idx.sum() < 20:
            print(f"  跳过: Kalman数据不足")
            continue
        
        kalman_residuals_clean = kalman_residuals[valid_idx]
        kalman_betas_clean = kalman_betas[valid_idx]
        
        print(f"  β范围: {kalman_betas_clean.min():.4f} - {kalman_betas_clean.max():.4f}")
        print(f"  残差: 均值={kalman_residuals_clean.mean():.6f}, 标准差={kalman_residuals_clean.std():.4f}")
        
        adf_kalman = adf_test(pd.Series(kalman_residuals_clean), f"{pair_name}_Kalman")
        print(f"  ADF: 统计量={adf_kalman['adf_stat']:.4f}, p={adf_kalman['p_value']:.4f}, 平稳={adf_kalman['is_stationary']}")
        
        # ========== 方法3: 使用协整β ==========
        print("\\n3. 协整β残差:")
        residuals_coint = y_aligned - beta_initial * x_aligned
        
        print(f"  协整β={beta_initial:.4f}")
        print(f"  残差: 均值={residuals_coint.mean():.6f}, 标准差={residuals_coint.std():.4f}")
        
        adf_coint = adf_test(residuals_coint, f"{pair_name}_Coint")
        print(f"  ADF: 统计量={adf_coint['adf_stat']:.4f}, p={adf_coint['p_value']:.4f}, 平稳={adf_coint['is_stationary']}")
        
        # 保存结果
        pair_result = {
            'pair': pair_name,
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'data_points': len(common_dates),
            'trading_signals': len(trading_pairs[trading_pairs['pair'] == pair_name]),
            'beta_initial': beta_initial,
            'beta_ols': beta_full,
            'beta_kalman_min': kalman_betas_clean.min(),
            'beta_kalman_max': kalman_betas_clean.max(),
        }
        
        # 添加ADF结果
        for method_result in [adf_ols_full, adf_kalman, adf_coint]:
            method = method_result['name'].split('_')[-1]  # 提取方法名
            for key, value in method_result.items():
                if key != 'name':
                    pair_result[f"{method.lower()}_{key}"] = value
        
        results.append(pair_result)
    
    # ========== 汇总分析 ==========
    print("\\n" + "=" * 80)
    print("残差平稳性汇总分析")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # 平稳性统计
        methods = ['full', 'kalman', 'coint']
        method_names = ['全数据OLS', 'Kalman滤波', '协整β']
        
        print("\\n平稳性对比:")
        print("-" * 60)
        for i, method in enumerate(methods):
            stationary_col = f"{method}_is_stationary"
            if stationary_col in results_df.columns:
                stationary_count = results_df[stationary_col].sum()
                total_count = len(results_df)
                print(f"{method_names[i]:12s}: {stationary_count}/{total_count} 平稳 ({stationary_count/total_count*100:.1f}%)")
        
        # 详细结果表
        print(f"\\n详细对比 (前5个配对):")
        print("-" * 120)
        display_cols = ['pair', 'trading_signals', 'beta_initial', 'beta_ols', 
                       'full_p_value', 'full_is_stationary',
                       'kalman_p_value', 'kalman_is_stationary',
                       'coint_p_value', 'coint_is_stationary']
        
        display_df = results_df[display_cols].head(5)
        for _, row in display_df.iterrows():
            print(f"\\n{row['pair']:8s} (信号:{row['trading_signals']}个):")
            print(f"  β: 协整={row['beta_initial']:7.4f}, OLS={row['beta_ols']:7.4f}")
            print(f"  OLS:      p={row['full_p_value']:7.4f}, 平稳={row['full_is_stationary']}")
            print(f"  Kalman:   p={row['kalman_p_value']:7.4f}, 平稳={row['kalman_is_stationary']}")
            print(f"  协整β:    p={row['coint_p_value']:7.4f}, 平稳={row['coint_is_stationary']}")
        
        # 保存详细结果
        output_file = f"residual_validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\\n详细结果已保存到: {output_file}")
        
        # 关键发现总结
        print(f"\\n" + "=" * 60)
        print("关键发现:")
        print("=" * 60)
        
        kalman_better = results_df['kalman_is_stationary'].sum()
        ols_better = results_df['full_is_stationary'].sum()
        
        if kalman_better > ols_better:
            print(f"✅ Kalman滤波残差平稳性优于OLS ({kalman_better}>{ols_better})")
            print("   说明自适应β估计有效改善了残差平稳性")
        elif kalman_better == ols_better:
            print(f"🤝 Kalman滤波与OLS残差平稳性相当 ({kalman_better}={ols_better})")
        else:
            print(f"⚠️  OLS残差平稳性优于Kalman滤波 ({ols_better}>{kalman_better})")
            print("   可能需要进一步优化Kalman参数")
        
        # z-score质量分析
        signal_files = [f for f in os.listdir('.') if f.startswith('quality_report_') and f.endswith('.csv')]
        if signal_files:
            latest_quality_file = sorted(signal_files)[-1]
            quality_df = pd.read_csv(latest_quality_file)
            good_quality = len(quality_df[quality_df['quality_status'] == 'good'])
            total_pairs = len(quality_df)
            
            print(f"\\nz-score质量分布:")
            print(f"  Good质量配对: {good_quality}/{total_pairs} ({good_quality/total_pairs*100:.1f}%)")
            print(f"  这些配对的残差平稳性应该更好")
        
    else:
        print("没有获得有效的验证结果")
    
    print("\\n" + "=" * 80)
    print("残差验证完成")
    print("=" * 80)


if __name__ == "__main__":
    main()