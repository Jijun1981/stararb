#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
残差ADF平稳性检验脚本
验证配对交易中残差序列的平稳性
独立验证脚本，不影响主代码
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer


def calculate_residuals(x_prices, y_prices, beta):
    """计算残差序列"""
    return y_prices - beta * x_prices


def adf_test(residuals, pair_name):
    """执行ADF检验"""
    try:
        # 去除NaN值
        clean_residuals = residuals.dropna()
        
        if len(clean_residuals) < 10:
            return {
                'pair': pair_name,
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': '数据不足'
            }
        
        # 执行ADF检验
        result = adfuller(clean_residuals, autolag='AIC')
        
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # 判断是否平稳 (p值<0.05 或 统计量 < 5%临界值)
        is_stationary = (p_value < 0.05) or (adf_statistic < critical_values['5%'])
        
        return {
            'pair': pair_name,
            'n_obs': len(clean_residuals),
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_1%': critical_values['1%'],
            'critical_5%': critical_values['5%'],
            'critical_10%': critical_values['10%'],
            'is_stationary_p': p_value < 0.05,
            'is_stationary_5%': adf_statistic < critical_values['5%'],
            'is_stationary': is_stationary,
            'error': None
        }
        
    except Exception as e:
        return {
            'pair': pair_name,
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_1%': np.nan,
            'critical_5%': np.nan,
            'critical_10%': np.nan,
            'is_stationary_p': False,
            'is_stationary_5%': False,
            'is_stationary': False,
            'error': str(e)
        }


def main():
    """主函数"""
    print("=" * 60)
    print("残差ADF平稳性检验")
    print("=" * 60)
    
    # 1. 加载数据
    print("加载期货数据...")
    try:
        price_data = load_all_symbols_data()
        print(f"数据范围: {price_data.index[0]} 至 {price_data.index[-1]}")
        print(f"品种数量: {len(price_data.columns)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 使用已有的配对结果（从信号生成脚本的成功配对）
    print("\n使用已有的协整配对...")
    
    # 手动定义一些已知的协整配对进行验证
    test_pairs = [
        {'pair': 'AG-NI', 'symbol_x': 'AG', 'symbol_y': 'NI', 'beta_1y': -3.3401},
        {'pair': 'SF-ZN', 'symbol_x': 'SF', 'symbol_y': 'ZN', 'beta_1y': 2.2640},
        {'pair': 'AU-ZN', 'symbol_x': 'AU', 'symbol_y': 'ZN', 'beta_1y': -10.7913},
        {'pair': 'CU-SN', 'symbol_x': 'CU', 'symbol_y': 'SN', 'beta_1y': 3.3449},
        {'pair': 'AG-CU', 'symbol_x': 'AG', 'symbol_y': 'CU', 'beta_1y': 3.0440},
        {'pair': 'AG-ZN', 'symbol_x': 'AG', 'symbol_y': 'ZN', 'beta_1y': -1.0089},
        {'pair': 'AG-PB', 'symbol_x': 'AG', 'symbol_y': 'PB', 'beta_1y': -0.0525},
        {'pair': 'SS-ZN', 'symbol_x': 'SS', 'symbol_y': 'ZN', 'beta_1y': 1.6196},
        {'pair': 'RB-SM', 'symbol_x': 'RB', 'symbol_y': 'SM', 'beta_1y': 1.9690},
        {'pair': 'AU-PB', 'symbol_x': 'AU', 'symbol_y': 'PB', 'beta_1y': -0.6714},
    ]
    
    filtered_pairs = pd.DataFrame(test_pairs)
    print(f"测试{len(filtered_pairs)}个已知协整配对")
    
    # 3. 对每个配对进行残差ADF检验
    print("\n开始ADF检验...")
    
    results = []
    test_period_start = '2023-01-01'  # 使用最近2年的数据进行检验
    test_data = price_data[test_period_start:].copy()
    
    for idx, pair_info in filtered_pairs.iterrows():
        pair_name = pair_info['pair']
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        beta_1y = pair_info['beta_1y']
        
        print(f"  检验配对 {pair_name} (β={beta_1y:.4f})")
        
        try:
            # 获取价格数据
            x_prices = test_data[symbol_x].dropna()
            y_prices = test_data[symbol_y].dropna()
            
            # 对齐数据
            common_dates = x_prices.index.intersection(y_prices.index)
            if len(common_dates) < 50:  # 至少需要50个观测
                print(f"    跳过: 数据不足 ({len(common_dates)}个观测)")
                continue
                
            x_aligned = x_prices[common_dates]
            y_aligned = y_prices[common_dates]
            
            # 计算残差
            residuals = calculate_residuals(x_aligned, y_aligned, beta_1y)
            
            # ADF检验
            adf_result = adf_test(residuals, pair_name)
            adf_result.update({
                'symbol_x': symbol_x,
                'symbol_y': symbol_y, 
                'beta_1y': beta_1y,
                'test_period': f"{test_data.index[0].date()} 至 {test_data.index[-1].date()}",
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std()
            })
            
            results.append(adf_result)
            
            # 显示结果
            if adf_result['error']:
                print(f"    错误: {adf_result['error']}")
            else:
                status = "平稳" if adf_result['is_stationary'] else "非平稳"
                print(f"    ADF统计量: {adf_result['adf_statistic']:.4f}, "
                      f"p值: {adf_result['p_value']:.4f}, "
                      f"结果: {status}")
                
        except Exception as e:
            print(f"    处理{pair_name}时出错: {e}")
            continue
    
    # 4. 汇总结果
    print(f"\n" + "=" * 60)
    print("ADF检验结果汇总")
    print("=" * 60)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # 统计汇总
        total_pairs = len(results_df)
        stationary_pairs = len(results_df[results_df['is_stationary']])
        stationary_rate = stationary_pairs / total_pairs * 100
        
        print(f"总检验配对数: {total_pairs}")
        print(f"平稳配对数: {stationary_pairs}")
        print(f"平稳比例: {stationary_rate:.1f}%")
        
        # 按p值统计
        p_stationary = len(results_df[results_df['is_stationary_p']])
        cv_stationary = len(results_df[results_df['is_stationary_5%']])
        
        print(f"\n平稳性判定方式:")
        print(f"  p值<0.05: {p_stationary}个 ({p_stationary/total_pairs*100:.1f}%)")
        print(f"  统计量<5%临界值: {cv_stationary}个 ({cv_stationary/total_pairs*100:.1f}%)")
        
        # p值分布
        valid_p = results_df[results_df['p_value'].notna()]['p_value']
        print(f"\np值分布:")
        print(f"  <0.01: {len(valid_p[valid_p < 0.01])}个")
        print(f"  0.01-0.05: {len(valid_p[(valid_p >= 0.01) & (valid_p < 0.05)])}个")
        print(f"  0.05-0.10: {len(valid_p[(valid_p >= 0.05) & (valid_p < 0.10)])}个")
        print(f"  >0.10: {len(valid_p[valid_p >= 0.10])}个")
        
        # 保存详细结果
        output_file = f"adf_stationarity_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        # 显示前10个最平稳的配对
        stationary_results = results_df[results_df['is_stationary']].copy()
        if len(stationary_results) > 0:
            stationary_results = stationary_results.sort_values('p_value')
            print(f"\n最平稳的10个配对 (按p值排序):")
            print("-" * 80)
            for idx, row in stationary_results.head(10).iterrows():
                print(f"{row['pair']:8s} | ADF: {row['adf_statistic']:8.4f} | "
                      f"p值: {row['p_value']:7.4f} | β: {row['beta_1y']:8.4f}")
        
        # 显示非平稳配对
        non_stationary = results_df[~results_df['is_stationary']].copy()
        if len(non_stationary) > 0:
            print(f"\n非平稳配对 ({len(non_stationary)}个):")
            print("-" * 80)
            for idx, row in non_stationary.iterrows():
                print(f"{row['pair']:8s} | ADF: {row['adf_statistic']:8.4f} | "
                      f"p值: {row['p_value']:7.4f} | β: {row['beta_1y']:8.4f}")
                      
    else:
        print("没有获得有效的检验结果")
    
    print("\n" + "=" * 60)
    print("检验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()