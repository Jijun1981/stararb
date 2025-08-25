#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对所有44个稳定配对进行滚动年度Kalman评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_original_version import OriginalKalmanFilter
from lib.data import load_all_symbols_data
from find_stable_pairs import find_stable_cointegrated_pairs

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_kalman_one_year(x_data: np.ndarray, y_data: np.ndarray, 
                             start_idx: int, pair_name: str, year: str) -> Dict:
    """
    评估一年的Kalman性能（简化版，加快速度）
    """
    if start_idx < 60 or start_idx + 250 > len(x_data):
        return None
    
    # 1. 用前60天做OLS初始化
    X_init = x_data[start_idx-60:start_idx].reshape(-1, 1)
    Y_init = y_data[start_idx-60:start_idx]
    model = LinearRegression()
    model.fit(X_init, Y_init)
    ols_beta_60d = model.coef_[0]
    
    # 2. 运行Kalman滤波器
    kf = OriginalKalmanFilter(
        warmup=60,
        Q_beta=5e-6,
        Q_alpha=1e-5,
        R_init=0.005,
        R_adapt=True,
        z_in=2.0,
        z_out=0.5
    )
    
    kf.initialize(x_data[start_idx-60:start_idx+250], 
                 y_data[start_idx-60:start_idx+250])
    
    for i in range(60, min(310, len(x_data[start_idx-60:]) - (start_idx-60))):
        kf.update(x_data[start_idx-60+i], y_data[start_idx-60+i])
    
    if len(kf.beta_history) == 0:
        return None
    
    beta_history = np.array(kf.beta_history)
    z_history = np.array(kf.z_history)
    
    # 3. 核心指标计算
    beta_mean = np.mean(beta_history)
    beta_std = np.std(beta_history)
    beta_range = np.max(beta_history) - np.min(beta_history)
    
    # 符号变化
    sign_changes = 0
    for i in range(1, len(beta_history)):
        if np.sign(beta_history[i]) != np.sign(beta_history[i-1]):
            sign_changes += 1
    
    # Z-score指标
    z_var = np.var(z_history)
    z_gt2_ratio = np.sum(np.abs(z_history) > 2.0) / len(z_history)
    
    # 均值回归
    z_gt2 = np.sum(np.abs(z_history) > 2.0)
    reversion_count = 0
    i = 0
    while i < len(z_history) - 20:
        if abs(z_history[i]) > 2.0:
            for j in range(i + 1, min(i + 21, len(z_history))):
                if abs(z_history[j]) < 0.5:
                    reversion_count += 1
                    break
            i += 1
        else:
            i += 1
    
    reversion_rate = reversion_count / z_gt2 if z_gt2 > 0 else 0
    
    # 综合评分
    score = 0
    if 0.8 <= z_var <= 1.3:
        score += 2
    elif 0.6 <= z_var <= 1.5:
        score += 1
    
    if 0.02 <= z_gt2_ratio <= 0.05:
        score += 2
    elif 0.01 <= z_gt2_ratio <= 0.08:
        score += 1
    
    if reversion_rate > 0.7:
        score += 2
    elif reversion_rate > 0.5:
        score += 1
    
    if sign_changes == 0:
        score += 2
    elif sign_changes <= 2:
        score += 1
    
    return {
        'pair': pair_name,
        'year': year,
        'ols_beta_60d': ols_beta_60d,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_range': beta_range,
        'sign_changes': sign_changes,
        'z_var': z_var,
        'z_gt2_ratio': z_gt2_ratio,
        'z_gt2_count': int(z_gt2),
        'reversion_rate': reversion_rate,
        'score': score
    }


def main():
    """
    主函数
    """
    logger.info("="*80)
    logger.info("全部44个稳定配对的Kalman滤波器年度评估")
    logger.info("="*80)
    
    # 1. 加载数据
    log_prices = load_all_symbols_data()
    logger.info(f"数据范围: {log_prices.index[0]} 至 {log_prices.index[-1]}")
    
    # 2. 获取所有稳定配对
    logger.info("\n寻找稳定配对...")
    stable_pairs = find_stable_cointegrated_pairs()
    logger.info(f"找到{len(stable_pairs)}个稳定配对")
    
    # 3. 定义评估时间点
    year_starts = [
        (60, '2020'),   # 第60天开始
        (310, '2021'),  # 2021年
        (560, '2022'),  # 2022年
        (810, '2023'),  # 2023年
        (1060, '2024'), # 2024年
    ]
    
    # 4. 评估所有配对
    all_results = []
    total_pairs = len(stable_pairs)
    
    for idx, row in stable_pairs.iterrows():
        pair_name = row['pair']
        symbol_x = row['symbol_x']
        symbol_y = row['symbol_y']
        
        logger.info(f"\n评估配对 {idx+1}/{total_pairs}: {pair_name}")
        
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        
        for start_idx, year in year_starts:
            result = evaluate_kalman_one_year(
                x_data, y_data, start_idx, pair_name, year
            )
            
            if result:
                all_results.append(result)
                logger.info(f"  {year}: Beta={result['beta_mean']:.3f}±{result['beta_std']:.3f}, "
                          f"Z_var={result['z_var']:.2f}, Score={result['score']}/9")
    
    # 5. 转换为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 6. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'all_stable_pairs_yearly_{timestamp}.csv'
    results_df.to_csv(filename, index=False)
    logger.info(f"\n结果已保存: {filename}")
    
    # 7. 统计分析
    logger.info("\n" + "="*80)
    logger.info("统计分析")
    logger.info("="*80)
    
    # 按年份统计
    for year in results_df['year'].unique():
        year_data = results_df[results_df['year'] == year]
        logger.info(f"\n{year}年统计:")
        logger.info(f"  配对数: {len(year_data)}")
        logger.info(f"  平均Beta标准差: {year_data['beta_std'].mean():.4f}")
        logger.info(f"  平均Z方差: {year_data['z_var'].mean():.3f}")
        logger.info(f"  平均Z>2比例: {year_data['z_gt2_ratio'].mean()*100:.1f}%")
        logger.info(f"  平均均值回归率: {year_data['reversion_rate'].mean()*100:.1f}%")
        logger.info(f"  无符号变化比例: {(year_data['sign_changes']==0).sum()/len(year_data)*100:.1f}%")
        logger.info(f"  平均评分: {year_data['score'].mean():.1f}/9")
    
    # 整体统计
    logger.info("\n整体统计（所有年份）:")
    logger.info(f"  总样本数: {len(results_df)}")
    logger.info(f"  平均Beta标准差: {results_df['beta_std'].mean():.4f}")
    logger.info(f"  平均Z方差: {results_df['z_var'].mean():.3f}")
    logger.info(f"  平均Z>2比例: {results_df['z_gt2_ratio'].mean()*100:.1f}%")
    logger.info(f"  平均均值回归率: {results_df['reversion_rate'].mean()*100:.1f}%")
    logger.info(f"  无符号变化比例: {(results_df['sign_changes']==0).sum()/len(results_df)*100:.1f}%")
    logger.info(f"  平均评分: {results_df['score'].mean():.1f}/9")
    
    # 找出最佳和最差配对
    pair_scores = results_df.groupby('pair').agg({
        'score': 'mean',
        'z_var': 'mean',
        'z_gt2_ratio': 'mean',
        'sign_changes': 'mean',
        'reversion_rate': 'mean'
    }).round(3)
    
    pair_scores = pair_scores.sort_values('score', ascending=False)
    
    logger.info("\n最佳配对（平均评分）:")
    for pair in pair_scores.head(10).index:
        row = pair_scores.loc[pair]
        logger.info(f"  {pair}: 评分={row['score']:.1f}, Z方差={row['z_var']:.2f}, "
                   f"回归率={row['reversion_rate']*100:.0f}%")
    
    logger.info("\n最差配对（平均评分）:")
    for pair in pair_scores.tail(5).index:
        row = pair_scores.loc[pair]
        logger.info(f"  {pair}: 评分={row['score']:.1f}, Z方差={row['z_var']:.2f}, "
                   f"回归率={row['reversion_rate']*100:.0f}%")
    
    # 问题配对（有符号变化的）
    problem_pairs = results_df[results_df['sign_changes'] > 0]
    if len(problem_pairs) > 0:
        logger.info(f"\n有符号变化的配对:")
        problem_summary = problem_pairs.groupby('pair')['sign_changes'].sum()
        for pair, changes in problem_summary.items():
            logger.info(f"  {pair}: 总计{changes}次符号变化")
    
    # 8. 绘制汇总图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Z方差分布
    ax = axes[0, 0]
    results_df['z_var'].hist(bins=30, ax=ax, edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', label='理想值')
    ax.set_xlabel('Z方差')
    ax.set_ylabel('频数')
    ax.set_title('Z方差分布（所有配对-年份）')
    ax.legend()
    
    # Z>2比例分布
    ax = axes[0, 1]
    (results_df['z_gt2_ratio']*100).hist(bins=30, ax=ax, edgecolor='black')
    ax.axvline(x=2, color='r', linestyle='--', label='目标下限')
    ax.axvline(x=5, color='r', linestyle='--', label='目标上限')
    ax.set_xlabel('Z>2比例 (%)')
    ax.set_ylabel('频数')
    ax.set_title('Z>2比例分布')
    ax.legend()
    
    # 评分分布
    ax = axes[1, 0]
    results_df['score'].hist(bins=9, ax=ax, edgecolor='black')
    ax.set_xlabel('评分')
    ax.set_ylabel('频数')
    ax.set_title('综合评分分布（满分9）')
    
    # 年度趋势
    ax = axes[1, 1]
    yearly_stats = results_df.groupby('year')['score'].mean()
    ax.plot(yearly_stats.index, yearly_stats.values, marker='o')
    ax.set_xlabel('年份')
    ax.set_ylabel('平均评分')
    ax.set_title('年度平均评分趋势')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('44个稳定配对Kalman性能分析', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f'all_stable_pairs_analysis_{timestamp}.png', dpi=100)
    logger.info(f"\n分析图已保存: all_stable_pairs_analysis_{timestamp}.png")
    
    logger.info("\n" + "="*80)
    logger.info("评估完成！")
    logger.info("="*80)
    
    return results_df


if __name__ == "__main__":
    results = main()