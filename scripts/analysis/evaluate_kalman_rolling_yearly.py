#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滚动年度评估Kalman滤波器性能
从2020年开始，每年评估一次
评估Beta稳定性、Z-score质量、均值回归等所有指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_original_version import OriginalKalmanFilter
from lib.data import load_all_symbols_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_kalman_one_year(x_data: np.ndarray, y_data: np.ndarray, 
                             start_idx: int, pair_name: str, year: str) -> Dict:
    """
    评估一年的Kalman性能
    
    Args:
        x_data, y_data: 完整数据
        start_idx: 开始位置（60天OLS之后）
        pair_name: 配对名称
        year: 年份标签
    """
    # 确保有足够的数据
    if start_idx < 60:
        return None
    if start_idx + 250 > len(x_data):  # 一年约250个交易日
        return None
    
    # 1. 用前60天做OLS初始化
    X_init = x_data[start_idx-60:start_idx].reshape(-1, 1)
    Y_init = y_data[start_idx-60:start_idx]
    model = LinearRegression()
    model.fit(X_init, Y_init)
    ols_beta_60d = model.coef_[0]
    
    # 2. 运行Kalman滤波器（一年）
    kf = OriginalKalmanFilter(
        warmup=60,
        Q_beta=5e-6,
        Q_alpha=1e-5,
        R_init=0.005,
        R_adapt=True,
        z_in=2.0,
        z_out=0.5
    )
    
    # 使用前60天初始化
    kf.initialize(x_data[start_idx-60:start_idx+250], 
                 y_data[start_idx-60:start_idx+250])
    
    # 运行一年的Kalman更新
    for i in range(60, min(310, len(x_data[start_idx-60:]) - (start_idx-60))):
        kf.update(x_data[start_idx-60+i], y_data[start_idx-60+i])
    
    if len(kf.beta_history) == 0:
        return None
    
    beta_history = np.array(kf.beta_history)
    z_history = np.array(kf.z_history)
    residual_history = np.array(kf.residual_history)
    
    # 3. Beta分析
    beta_mean = np.mean(beta_history)
    beta_std = np.std(beta_history)
    beta_min = np.min(beta_history)
    beta_max = np.max(beta_history)
    beta_range = beta_max - beta_min
    
    # Beta符号变化
    sign_changes = 0
    neg_to_pos = 0
    pos_to_neg = 0
    for i in range(1, len(beta_history)):
        if np.sign(beta_history[i]) != np.sign(beta_history[i-1]):
            sign_changes += 1
            if beta_history[i-1] < 0 and beta_history[i] > 0:
                neg_to_pos += 1
            elif beta_history[i-1] > 0 and beta_history[i] < 0:
                pos_to_neg += 1
    
    # Beta相对于初始值的变化
    beta_vs_initial = abs(beta_mean - ols_beta_60d)
    beta_drift = beta_history[-1] - beta_history[0]
    
    # 4. Z-score分析
    z_mean = np.mean(z_history)
    z_std = np.std(z_history)
    z_var = np.var(z_history)
    z_min = np.min(z_history)
    z_max = np.max(z_history)
    
    # Z>2信号比例
    z_gt2 = np.sum(np.abs(z_history) > 2.0)
    z_gt2_ratio = z_gt2 / len(z_history) if len(z_history) > 0 else 0
    
    # Z>2.5和Z>3的比例
    z_gt25_ratio = np.sum(np.abs(z_history) > 2.5) / len(z_history)
    z_gt3_ratio = np.sum(np.abs(z_history) > 3.0) / len(z_history)
    
    # 5. 均值回归分析
    reversion_count = 0
    reversion_times = []
    i = 0
    while i < len(z_history) - 20:
        if abs(z_history[i]) > 2.0:
            for j in range(i + 1, min(i + 21, len(z_history))):
                if abs(z_history[j]) < 0.5:
                    reversion_count += 1
                    reversion_times.append(j - i)
                    break
            i += 1
        else:
            i += 1
    
    reversion_rate = reversion_count / z_gt2 if z_gt2 > 0 else 0
    avg_reversion_time = np.mean(reversion_times) if reversion_times else np.nan
    
    # 6. 残差平稳性检验
    if len(residual_history) > 30:
        try:
            adf_stat, adf_pvalue = adfuller(residual_history)[:2]
            residual_stationary = adf_pvalue < 0.05
        except:
            adf_pvalue = np.nan
            residual_stationary = False
    else:
        adf_pvalue = np.nan
        residual_stationary = False
    
    # 7. 与滚动OLS对比
    ols_betas = []
    for i in range(60, min(250, len(z_history))):
        X = x_data[start_idx-60+i-60:start_idx-60+i].reshape(-1, 1)
        Y = y_data[start_idx-60+i-60:start_idx-60+i]
        if len(X) >= 30:
            model = LinearRegression()
            model.fit(X, Y)
            ols_betas.append(model.coef_[0])
    
    if len(ols_betas) > 0 and len(beta_history) > 0:
        min_len = min(len(ols_betas), len(beta_history))
        correlation = np.corrcoef(ols_betas[:min_len], beta_history[:min_len])[0, 1]
    else:
        correlation = np.nan
    
    # 8. 综合评分
    score = 0
    # Z方差接近1
    if 0.8 <= z_var <= 1.3:
        score += 2
    elif 0.6 <= z_var <= 1.5:
        score += 1
    
    # Z>2比例在2-5%
    if 0.02 <= z_gt2_ratio <= 0.05:
        score += 2
    elif 0.01 <= z_gt2_ratio <= 0.08:
        score += 1
    
    # 残差平稳
    if residual_stationary:
        score += 1
    
    # 均值回归率>70%
    if reversion_rate > 0.7:
        score += 2
    elif reversion_rate > 0.5:
        score += 1
    
    # 与OLS相关性>0.6
    if correlation > 0.6:
        score += 1
    
    # Beta稳定（无符号变化）
    if sign_changes == 0:
        score += 1
    
    results = {
        'pair': pair_name,
        'year': year,
        'ols_beta_60d': ols_beta_60d,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'beta_range': beta_range,
        'beta_drift': beta_drift,
        'beta_vs_initial': beta_vs_initial,
        'sign_changes': sign_changes,
        'neg_to_pos': neg_to_pos,
        'pos_to_neg': pos_to_neg,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_var': z_var,
        'z_min': z_min,
        'z_max': z_max,
        'z_gt2_ratio': z_gt2_ratio,
        'z_gt25_ratio': z_gt25_ratio,
        'z_gt3_ratio': z_gt3_ratio,
        'z_gt2_count': int(z_gt2),
        'reversion_rate': reversion_rate,
        'avg_reversion_time': avg_reversion_time,
        'adf_pvalue': adf_pvalue,
        'residual_stationary': residual_stationary,
        'ols_correlation': correlation,
        'score': score,
        'data_points': len(beta_history)
    }
    
    return results


def evaluate_all_years(pair_data: List[Tuple[str, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    评估所有年份的Kalman性能
    """
    all_results = []
    
    # 定义评估时间点（每年年初）
    # 数据从2020-01-02开始，每250个交易日约为一年
    year_starts = [
        (60, '2020'),   # 第60天开始（前60天用于初始化）
        (310, '2021'),  # 2021年开始
        (560, '2022'),  # 2022年开始
        (810, '2023'),  # 2023年开始
        (1060, '2024'), # 2024年开始
    ]
    
    for pair_name, x_data, y_data in pair_data:
        logger.info(f"\n评估配对: {pair_name}")
        
        for start_idx, year in year_starts:
            logger.info(f"  评估{year}年...")
            result = evaluate_kalman_one_year(
                x_data, y_data, start_idx, pair_name, year
            )
            
            if result:
                all_results.append(result)
                logger.info(f"    Beta: {result['beta_mean']:.4f}±{result['beta_std']:.4f}")
                logger.info(f"    Z方差: {result['z_var']:.3f}, Z>2: {result['z_gt2_ratio']*100:.1f}%")
                logger.info(f"    评分: {result['score']}/9")
    
    return pd.DataFrame(all_results)


def plot_yearly_analysis(results_df: pd.DataFrame):
    """
    绘制年度分析图表
    """
    # 按年份分组
    years = results_df['year'].unique()
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 1. Beta稳定性随时间变化
    ax = axes[0, 0]
    for pair in results_df['pair'].unique()[:5]:  # 只画前5个配对
        pair_data = results_df[results_df['pair'] == pair]
        ax.plot(pair_data['year'], pair_data['beta_std'], marker='o', label=pair)
    ax.set_title('Beta标准差随时间变化')
    ax.set_xlabel('年份')
    ax.set_ylabel('Beta Std')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Z方差随时间变化
    ax = axes[0, 1]
    for pair in results_df['pair'].unique()[:5]:
        pair_data = results_df[results_df['pair'] == pair]
        ax.plot(pair_data['year'], pair_data['z_var'], marker='o', label=pair)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想值')
    ax.set_title('Z方差随时间变化')
    ax.set_xlabel('年份')
    ax.set_ylabel('Z Variance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Z>2比例分布
    ax = axes[1, 0]
    for year in years:
        year_data = results_df[results_df['year'] == year]['z_gt2_ratio'] * 100
        ax.boxplot(year_data, positions=[int(year)], widths=0.6)
    ax.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Z>2比例分布（目标2-5%）')
    ax.set_xlabel('年份')
    ax.set_ylabel('Z>2 比例 (%)')
    ax.grid(True, alpha=0.3)
    
    # 4. 均值回归率
    ax = axes[1, 1]
    for year in years:
        year_data = results_df[results_df['year'] == year]['reversion_rate'] * 100
        ax.boxplot(year_data, positions=[int(year)], widths=0.6)
    ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='目标>70%')
    ax.set_title('均值回归率')
    ax.set_xlabel('年份')
    ax.set_ylabel('回归率 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 综合评分
    ax = axes[2, 0]
    avg_scores = results_df.groupby('year')['score'].mean()
    ax.bar(avg_scores.index, avg_scores.values)
    ax.axhline(y=6, color='g', linestyle='--', alpha=0.5, label='良好')
    ax.set_title('平均综合评分')
    ax.set_xlabel('年份')
    ax.set_ylabel('评分 (满分9)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Beta符号变化统计
    ax = axes[2, 1]
    sign_change_ratio = results_df.groupby('year').apply(
        lambda x: (x['sign_changes'] > 0).sum() / len(x) * 100
    )
    ax.bar(sign_change_ratio.index, sign_change_ratio.values)
    ax.set_title('有符号变化的配对比例')
    ax.set_xlabel('年份')
    ax.set_ylabel('比例 (%)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman滤波器年度性能分析', fontsize=14)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'kalman_yearly_analysis_{timestamp}.png', dpi=100, bbox_inches='tight')
    logger.info(f"\n图表已保存: kalman_yearly_analysis_{timestamp}.png")


def main():
    """
    主函数
    """
    logger.info("="*80)
    logger.info("Kalman滤波器滚动年度评估")
    logger.info("="*80)
    
    # 加载数据
    log_prices = load_all_symbols_data()
    logger.info(f"数据范围: {log_prices.index[0]} 至 {log_prices.index[-1]}")
    logger.info(f"数据点数: {len(log_prices)}")
    
    # 选择主要配对进行评估
    test_pairs = [
        ('AU', 'SS'),
        ('SS', 'SF'),
        ('AL', 'SN'),
        ('AU', 'CU'),
        ('AU', 'PB'),
        ('CU', 'SN'),
        ('RB', 'SM'),
        ('AU', 'AG'),
        ('AU', 'ZN'),
        ('AG', 'NI')
    ]
    
    # 准备数据
    pair_data = []
    for symbol_x, symbol_y in test_pairs:
        pair_name = f"{symbol_x}-{symbol_y}"
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        pair_data.append((pair_name, x_data, y_data))
    
    # 评估所有年份
    results_df = evaluate_all_years(pair_data)
    
    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'kalman_yearly_evaluation_{timestamp}.csv', index=False)
    logger.info(f"\n详细结果已保存: kalman_yearly_evaluation_{timestamp}.csv")
    
    # 统计分析
    logger.info("\n" + "="*80)
    logger.info("统计分析")
    logger.info("="*80)
    
    # 按年份统计
    for year in results_df['year'].unique():
        year_data = results_df[results_df['year'] == year]
        logger.info(f"\n{year}年统计:")
        logger.info(f"  样本数: {len(year_data)}")
        logger.info(f"  Beta标准差: {year_data['beta_std'].mean():.4f} ± {year_data['beta_std'].std():.4f}")
        logger.info(f"  Z方差: {year_data['z_var'].mean():.3f} ± {year_data['z_var'].std():.3f}")
        logger.info(f"  Z>2比例: {year_data['z_gt2_ratio'].mean()*100:.1f}% ± {year_data['z_gt2_ratio'].std()*100:.1f}%")
        logger.info(f"  均值回归率: {year_data['reversion_rate'].mean()*100:.1f}%")
        logger.info(f"  无符号变化: {(year_data['sign_changes']==0).sum()}/{len(year_data)}")
        logger.info(f"  平均评分: {year_data['score'].mean():.1f}/9")
    
    # 整体统计
    logger.info("\n整体统计:")
    logger.info(f"  总样本数: {len(results_df)}")
    logger.info(f"  平均Beta标准差: {results_df['beta_std'].mean():.4f}")
    logger.info(f"  平均Z方差: {results_df['z_var'].mean():.3f}")
    logger.info(f"  平均Z>2比例: {results_df['z_gt2_ratio'].mean()*100:.1f}%")
    logger.info(f"  平均均值回归率: {results_df['reversion_rate'].mean()*100:.1f}%")
    logger.info(f"  无符号变化比例: {(results_df['sign_changes']==0).sum()/len(results_df)*100:.1f}%")
    logger.info(f"  平均综合评分: {results_df['score'].mean():.1f}/9")
    
    # 找出最稳定和最不稳定的配对
    pair_scores = results_df.groupby('pair')['score'].mean().sort_values(ascending=False)
    logger.info(f"\n最稳定的配对:")
    for pair, score in pair_scores.head(3).items():
        logger.info(f"  {pair}: 平均评分 {score:.1f}")
    
    logger.info(f"\n最不稳定的配对:")
    for pair, score in pair_scores.tail(3).items():
        logger.info(f"  {pair}: 平均评分 {score:.1f}")
    
    # 绘制分析图表
    plot_yearly_analysis(results_df)
    
    logger.info("\n" + "="*80)
    logger.info("评估完成！")
    logger.info("="*80)
    
    return results_df


if __name__ == "__main__":
    results = main()