#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman滤波参数网格搜索
目标：找到最佳参数组合
- 残差平稳性最大化
- |z|>=2比例在2%-5%之间
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import logging
from statsmodels.tsa.stattools import adfuller

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import AdaptiveSignalGenerator

# 设置日志
logging.basicConfig(level=logging.WARNING)  # 减少日志输出


def adf_test_simple(residuals):
    """简化的ADF检验"""
    try:
        clean_residuals = residuals.dropna()
        if len(clean_residuals) < 20:
            return False
        result = adfuller(clean_residuals, autolag='AIC')
        return result[1] < 0.05  # p值<0.05为平稳
    except:
        return False


def test_mean_reversion(signal_data, price_data):
    """测试|z|>2时的均值回归收益"""
    returns = []
    
    for pair in signal_data['pair'].unique():
        pair_signals = signal_data[signal_data['pair'] == pair].copy()
        
        # 只看|z|>2的点
        extreme_signals = pair_signals[pair_signals['z_score'].abs() > 2].copy()
        
        if len(extreme_signals) < 3:
            continue
            
        symbol_x = pair_signals['symbol_x'].iloc[0]
        symbol_y = pair_signals['symbol_y'].iloc[0]
        
        # 获取价格数据
        if symbol_x not in price_data.columns or symbol_y not in price_data.columns:
            continue
            
        for _, row in extreme_signals.iterrows():
            date = row['date']
            z_score = row['z_score']
            beta = row['beta']
            
            # 找到后续5天的数据
            try:
                date_idx = price_data.index.get_loc(pd.to_datetime(date))
                if date_idx + 5 >= len(price_data):
                    continue
                    
                # T日和T+5日价格
                x_t0 = price_data[symbol_x].iloc[date_idx]
                y_t0 = price_data[symbol_y].iloc[date_idx]  
                x_t5 = price_data[symbol_x].iloc[date_idx + 5]
                y_t5 = price_data[symbol_y].iloc[date_idx + 5]
                
                # 计算spread变化
                spread_t0 = y_t0 - beta * x_t0
                spread_t5 = y_t5 - beta * x_t5
                spread_change = spread_t5 - spread_t0
                
                # 均值回归预期：z_score与spread_change应该反向
                # 如果z>2（spread过高），预期spread_change<0（回归）
                # 如果z<-2（spread过低），预期spread_change>0（回归）
                expected_return = -np.sign(z_score) * spread_change
                returns.append(expected_return)
                
            except:
                continue
    
    if len(returns) < 5:
        return 0, 0
        
    mean_return = np.mean(returns)
    ir = mean_return / (np.std(returns) + 1e-8)  # 信息比率
    
    return mean_return, ir


def evaluate_params(delta_init, p0_scale, delta_min, lambda_r, test_pairs, price_data):
    """评估一组参数的效果"""
    
    # 修改AdaptiveKalmanFilter的参数
    from lib.signal_generation import AdaptiveKalmanFilter
    
    # 临时保存原始参数
    original_warm_up = AdaptiveKalmanFilter.warm_up_ols
    
    def new_warm_up_ols(self, x_data, y_data, window=60):
        # 去中心化处理
        self.mu_x = np.mean(x_data[:window])
        self.mu_y = np.mean(y_data[:window])
        x_use = x_data[:window] - self.mu_x
        y_use = y_data[:window] - self.mu_y
        
        # OLS回归
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(x_use.reshape(-1, 1), y_use)
        
        self.beta = float(reg.coef_[0])
        innovations = y_use - reg.predict(x_use.reshape(-1, 1)).flatten()
        self.R = float(np.var(innovations, ddof=1))
        
        # 调整P0初始化
        x_var = np.var(x_use, ddof=1)
        self.P = p0_scale * self.R / max(x_var, 1e-12)  # 使用参数化的P0
        
        return {
            'beta': self.beta, 'R': self.R, 'P': self.P,
            'mu_x': self.mu_x, 'mu_y': self.mu_y
        }
    
    # 临时替换方法
    AdaptiveKalmanFilter.warm_up_ols = new_warm_up_ols
    
    try:
        # 创建信号生成器 (使用参数化的delta和lambda)
        sg = AdaptiveSignalGenerator(
            z_open=2.0, z_close=0.5, max_holding_days=30, calibration_freq=5,
            ols_window=60, warm_up_days=30
        )
        
        # 临时修改delta参数
        sg._delta_init = delta_init
        sg._delta_min = delta_min  
        sg._lambda_r = lambda_r
        
        # 处理测试配对
        results = sg.process_all_pairs(
            pairs_df=test_pairs,
            price_data=price_data,
            beta_window='1y'
        )
        
        if results.empty:
            return None
            
        # 分析结果
        signal_period = results[results['phase'] == 'signal_period']
        
        # 1. Z-score分布
        z_scores = signal_period['z_score']
        z_over2_pct = (z_scores.abs() >= 2).mean() * 100
        z_std = z_scores.std()
        
        # 2. 残差平稳性测试
        stationarity_scores = []
        
        for pair in test_pairs['pair']:
            pair_data = signal_period[signal_period['pair'] == pair]
            if len(pair_data) > 50:  # 足够的数据点
                innovations = pair_data['innovation'].values
                is_stationary = adf_test_simple(pd.Series(innovations))
                stationarity_scores.append(is_stationary)
        
        stationarity_rate = np.mean(stationarity_scores) * 100 if stationarity_scores else 0
        
        # 3. 均值回归收益测试 (关键新增!)
        mean_reversion_return, mean_reversion_ir = test_mean_reversion(signal_period, price_data)
        
        # 4. 质量评估
        quality_report = sg.get_quality_report()
        good_quality_pct = (quality_report['quality_status'] == 'good').mean() * 100
        
        # 5. 综合评分 (新增均值回归权重)
        z_score_bonus = 15 if 2 <= z_over2_pct <= 5 else 0
        mr_bonus = 10 if mean_reversion_return > 0 else 0  # 均值回归收益>0奖励
        ir_bonus = 5 if mean_reversion_ir > 0.2 else 0     # IR>0.2额外奖励
        
        total_score = stationarity_rate + z_score_bonus + mr_bonus + ir_bonus
        
        return {
            'delta_init': delta_init,
            'p0_scale': p0_scale, 
            'delta_min': delta_min,
            'lambda_r': lambda_r,
            'z_over2_pct': z_over2_pct,
            'z_std': z_std,
            'stationarity_rate': stationarity_rate,
            'mean_reversion_return': mean_reversion_return,
            'mean_reversion_ir': mean_reversion_ir,
            'good_quality_pct': good_quality_pct,
            'n_pairs': len(stationarity_scores),
            'score': total_score  # 综合评分
        }
        
    except Exception as e:
        print(f"参数组合失败: delta_init={delta_init}, p0_scale={p0_scale}, error={e}")
        return None
    finally:
        # 恢复原始方法
        AdaptiveKalmanFilter.warm_up_ols = original_warm_up
        

def main():
    print("=" * 80)
    print("Kalman滤波参数网格搜索")
    print("目标: 残差平稳性 + |z|>=2在2%-5%")
    print("=" * 80)
    
    # 1. 加载数据
    print("加载数据...")
    price_data = load_all_symbols_data()
    analysis_data = price_data['2024-04-01':'2025-08-24'].copy()
    
    # 2. 选择代表性配对（快速测试用）
    test_pairs = pd.DataFrame({
        'pair': ['CU-SN', 'HC-SM', 'ZN-SM', 'RB-SM', 'SS-NI', 'SM-I'],
        'symbol_x': ['CU', 'HC', 'ZN', 'RB', 'SS', 'SM'], 
        'symbol_y': ['SN', 'SM', 'SM', 'SM', 'NI', 'I'],
        'beta_1y': [0.977974, 0.920247, 0.684857, 1.046174, 0.829148, 0.682580]
    })
    
    print(f"测试配对: {list(test_pairs['pair'])}")
    
    # 3. 定义搜索网格 - 简单但关键的参数
    param_grid = {
        'delta_init': [0.96, 0.98, 0.99],           # 初始delta  
        'p0_scale': [1.0, 3.0, 5.0],               # P0缩放因子
        'delta_min': [0.90, 0.93, 0.95],           # delta下界
        'lambda_r': [0.92, 0.96, 0.98]             # R的EWMA参数
    }
    
    print(f"搜索空间: {len(param_grid['delta_init']) * len(param_grid['p0_scale']) * len(param_grid['delta_min']) * len(param_grid['lambda_r'])}个组合")
    
    # 4. 网格搜索
    results = []
    total_combinations = len(param_grid['delta_init']) * len(param_grid['p0_scale']) * len(param_grid['delta_min']) * len(param_grid['lambda_r'])
    current = 0
    
    for delta_init in param_grid['delta_init']:
        for p0_scale in param_grid['p0_scale']:
            for delta_min in param_grid['delta_min']:
                for lambda_r in param_grid['lambda_r']:
                    current += 1
                    print(f"\n进度 {current}/{total_combinations}: delta_init={delta_init}, p0_scale={p0_scale}, delta_min={delta_min}, lambda_r={lambda_r}")
                    
                    result = evaluate_params(
                        delta_init, p0_scale, delta_min, lambda_r,
                        test_pairs, analysis_data
                    )
                    
                    if result:
                        results.append(result)
                        print(f"  |z|>=2: {result['z_over2_pct']:.2f}%, 平稳性: {result['stationarity_rate']:.1f}%, "
                              f"MR收益: {result['mean_reversion_return']:.4f}, IR: {result['mean_reversion_ir']:.2f}, 评分: {result['score']:.1f}")
    
    # 5. 分析结果
    if results:
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print("网格搜索结果")
        print("=" * 80)
        
        # 按综合评分排序
        top_results = results_df.nlargest(10, 'score')
        
        print("\n最佳参数组合 (前10名):")
        print("-" * 140)
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"{i:2d}. delta_init={row['delta_init']:.2f}, p0_scale={row['p0_scale']:.1f}, "
                  f"delta_min={row['delta_min']:.2f}, lambda_r={row['lambda_r']:.2f}")
            print(f"     |z|>=2: {row['z_over2_pct']:.2f}%, 平稳性: {row['stationarity_rate']:.1f}%, "
                  f"MR收益: {row['mean_reversion_return']:.4f}, IR: {row['mean_reversion_ir']:.2f}, 评分: {row['score']:.1f}")
            print()
        
        # 保存结果
        output_file = f"kalman_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"详细结果已保存到: {output_file}")
        
        # 推荐最佳参数
        best = top_results.iloc[0]
        print(f"\n🎯 推荐参数:")
        print(f"   delta_init = {best['delta_init']}")
        print(f"   p0_scale = {best['p0_scale']}")
        print(f"   delta_min = {best['delta_min']}")  
        print(f"   lambda_r = {best['lambda_r']}")
        print(f"   预期效果: |z|>=2约{best['z_over2_pct']:.1f}%, 平稳性约{best['stationarity_rate']:.1f}%")
        print(f"   均值回归: 收益{best['mean_reversion_return']:.4f}, IR={best['mean_reversion_ir']:.2f}")
    else:
        print("未获得有效结果")
    
    print("\n" + "=" * 80)
    print("搜索完成")
    print("=" * 80)


if __name__ == "__main__":
    main()