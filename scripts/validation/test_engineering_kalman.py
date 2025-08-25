#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工程版Kalman滤波器
对比60天滚动OLS基准
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_engineering_version import (
    EngineeringKalmanFilter, 
    run_single_test,
    check_mean_reversion,
    compare_with_rolling_ols
)
from lib.data import load_all_symbols_data
from find_stable_pairs import find_stable_cointegrated_pairs

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def grid_search_parameters(x_data: np.ndarray, y_data: np.ndarray, 
                          stable_pair_name: str) -> pd.DataFrame:
    """
    网格搜索最优参数
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"参数网格搜索: {stable_pair_name}")
    logger.info(f"{'='*60}")
    
    # 参数网格
    param_grid = {
        'lambda_r': [0.90, 0.92, 0.95, 0.97],  # R的EWMA参数
        'c': [0.7, 0.8, 0.85, 0.9],            # R的EWMA系数
        'delta': [0.96, 0.97, 0.98, 0.985],    # 折扣因子
    }
    
    # 固定参数
    fixed_params = {
        'warmup': 60,
        'rho': 0.995,
        'z_in': 2.0,
        'z_out': 0.5,
        'beta_bounds': (-10, 10)  # 放宽边界
    }
    
    results = []
    total_tests = len(param_grid['lambda_r']) * len(param_grid['c']) * len(param_grid['delta'])
    test_count = 0
    
    for lambda_r in param_grid['lambda_r']:
        for c in param_grid['c']:
            for delta in param_grid['delta']:
                test_count += 1
                
                params = fixed_params.copy()
                params.update({
                    'lambda_r': lambda_r,
                    'c': c,
                    'delta': delta
                })
                
                logger.info(f"\n测试 {test_count}/{total_tests}: λ={lambda_r}, c={c}, δ={delta}")
                
                # 运行单次测试
                metrics = run_single_test(x_data, y_data, params)
                
                if metrics:
                    result = {
                        'pair': stable_pair_name,
                        'lambda_r': lambda_r,
                        'c': c,
                        'delta': delta,
                        **metrics
                    }
                    results.append(result)
                    
                    # 实时显示关键指标
                    logger.info(f"  Z方差: {metrics.get('z_var', np.nan):.3f}")
                    logger.info(f"  Z>2比例: {metrics.get('z_gt2_ratio', 0)*100:.1f}%")
                    logger.info(f"  均值回归率: {metrics.get('reversion_rate', 0)*100:.1f}%")
                    logger.info(f"  与OLS相关性: {metrics.get('correlation', 0):.3f}")
                    logger.info(f"  综合评分: {metrics.get('score', 0)}/9")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 排序（按综合评分）
    results_df = results_df.sort_values('score', ascending=False)
    
    return results_df


def test_multiple_pairs(num_pairs: int = 5) -> pd.DataFrame:
    """
    测试多个稳定配对
    """
    logger.info("\n" + "="*80)
    logger.info("多配对测试")
    logger.info("="*80)
    
    # 1. 加载数据
    log_prices = load_all_symbols_data()
    
    # 2. 找出稳定配对
    logger.info("\n寻找稳定配对...")
    stable_pairs = find_stable_cointegrated_pairs()
    
    if len(stable_pairs) == 0:
        logger.error("没有找到稳定配对")
        return pd.DataFrame()
    
    # 3. 选择前N个最稳定的配对
    test_pairs = stable_pairs.head(num_pairs)
    logger.info(f"\n选择{len(test_pairs)}个配对进行测试:")
    for _, row in test_pairs.iterrows():
        logger.info(f"  {row['pair']}: 5年p值={row['pvalue_5y']:.6f}")
    
    all_results = []
    
    # 4. 对每个配对进行测试
    for idx, row in test_pairs.iterrows():
        symbol_x = row['symbol_x']
        symbol_y = row['symbol_y']
        pair_name = row['pair']
        
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"测试配对 {idx+1}/{len(test_pairs)}: {pair_name}")
        logger.info(f"{'#'*80}")
        
        # 获取数据
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        
        # 只测试默认参数（快速验证）
        params = {
            'warmup': 60,
            'rho': 0.995,
            'lambda_r': 0.95,
            'c': 0.85,
            'delta': 0.98,
            'z_in': 2.0,
            'z_out': 0.5,
            'beta_bounds': (-10, 10)
        }
        
        metrics = run_single_test(x_data, y_data, params)
        
        if metrics:
            result = {
                'pair': pair_name,
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                **params,
                **metrics
            }
            all_results.append(result)
            
            # 显示结果
            logger.info(f"\n{pair_name} 测试结果:")
            logger.info(f"  Z方差: {metrics.get('z_var', np.nan):.3f}")
            logger.info(f"  Z均值: {metrics.get('z_mean', np.nan):.3f}")
            logger.info(f"  Z>2比例: {metrics.get('z_gt2_ratio', 0)*100:.1f}%")
            logger.info(f"  均值回归率: {metrics.get('reversion_rate', 0)*100:.1f}%")
            logger.info(f"  平均回归时间: {metrics.get('avg_reversion_time', np.nan):.1f}天")
            logger.info(f"  创新平稳性: {'是' if metrics.get('innovation_stationary', False) else '否'}")
            logger.info(f"  与OLS相关性: {metrics.get('correlation', 0):.3f}")
            logger.info(f"  Beta差异均值: {metrics.get('beta_diff_mean', np.nan):.4f}")
            logger.info(f"  综合评分: {metrics.get('score', 0)}/9")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"engineering_kalman_test_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    logger.info(f"\n测试结果已保存: {filename}")
    
    return results_df


def find_best_parameters() -> Dict:
    """
    找出最优参数组合
    """
    logger.info("\n" + "="*80)
    logger.info("寻找最优参数")
    logger.info("="*80)
    
    # 1. 加载数据
    log_prices = load_all_symbols_data()
    
    # 2. 使用最稳定的配对（AU-SS）
    x_data = log_prices['AU'].values
    y_data = log_prices['SS'].values
    
    # 3. 网格搜索
    results_df = grid_search_parameters(x_data, y_data, 'AU-SS')
    
    if len(results_df) == 0:
        logger.error("网格搜索失败")
        return {}
    
    # 4. 找出最优参数
    best_row = results_df.iloc[0]
    
    logger.info("\n" + "="*60)
    logger.info("最优参数组合:")
    logger.info(f"  λ_r = {best_row['lambda_r']}")
    logger.info(f"  c = {best_row['c']}")
    logger.info(f"  δ = {best_row['delta']}")
    logger.info(f"  综合评分: {best_row['score']}/9")
    logger.info("\n性能指标:")
    logger.info(f"  Z方差: {best_row['z_var']:.3f}")
    logger.info(f"  Z>2比例: {best_row['z_gt2_ratio']*100:.1f}%")
    logger.info(f"  均值回归率: {best_row['reversion_rate']*100:.1f}%")
    logger.info(f"  与OLS相关性: {best_row['correlation']:.3f}")
    logger.info("="*60)
    
    # 5. 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"kalman_grid_search_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    logger.info(f"\n详细结果已保存: {filename}")
    
    # 返回最优参数
    best_params = {
        'lambda_r': best_row['lambda_r'],
        'c': best_row['c'],
        'delta': best_row['delta']
    }
    
    return best_params


def run_comprehensive_test():
    """
    运行综合测试
    """
    logger.info("\n" + "="*80)
    logger.info("工程版Kalman滤波器综合测试")
    logger.info("="*80)
    
    # 1. 快速测试多个配对
    logger.info("\n步骤1: 测试前5个最稳定配对")
    multi_results = test_multiple_pairs(5)
    
    if len(multi_results) > 0:
        # 统计分析
        logger.info("\n" + "-"*60)
        logger.info("统计分析:")
        logger.info(f"  平均Z方差: {multi_results['z_var'].mean():.3f}")
        logger.info(f"  平均Z>2比例: {multi_results['z_gt2_ratio'].mean()*100:.1f}%")
        logger.info(f"  平均均值回归率: {multi_results['reversion_rate'].mean()*100:.1f}%")
        logger.info(f"  平均OLS相关性: {multi_results['correlation'].mean():.3f}")
        logger.info(f"  平均综合评分: {multi_results['score'].mean():.1f}/9")
        
        # 找出最佳配对
        best_pair_idx = multi_results['score'].idxmax()
        best_pair = multi_results.loc[best_pair_idx]
        logger.info(f"\n表现最佳的配对: {best_pair['pair']}")
        logger.info(f"  综合评分: {best_pair['score']}/9")
    
    # 2. 参数优化（使用AU-SS）
    logger.info("\n步骤2: 参数网格搜索（使用AU-SS配对）")
    best_params = find_best_parameters()
    
    logger.info("\n" + "="*80)
    logger.info("测试完成！")
    logger.info("="*80)
    
    return best_params


if __name__ == "__main__":
    # 运行综合测试
    best_params = run_comprehensive_test()
    
    if best_params:
        logger.info("\n推荐的参数配置:")
        logger.info(f"  lambda_r: {best_params.get('lambda_r', 0.95)}")
        logger.info(f"  c: {best_params.get('c', 0.85)}")
        logger.info(f"  delta: {best_params.get('delta', 0.98)}")