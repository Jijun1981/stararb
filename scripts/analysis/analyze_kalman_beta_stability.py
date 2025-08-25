#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Kalman滤波的Beta稳定性
重点检查：
1. Beta变化幅度
2. 是否有负转正的情况
3. Beta的稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import os
from typing import Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_original_version import OriginalKalmanFilter
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_beta_stability(x_data: np.ndarray, y_data: np.ndarray, 
                          pair_name: str, initial_beta: float) -> Dict:
    """
    分析Beta稳定性
    """
    # 使用原始版Kalman（最优参数）
    kf = OriginalKalmanFilter(
        warmup=60,
        Q_beta=5e-6,
        Q_alpha=1e-5,
        R_init=0.005,
        R_adapt=True
    )
    
    # 初始化并运行
    kf.initialize(x_data, y_data)
    
    for i in range(60, len(x_data)):
        kf.update(x_data[i], y_data[i])
    
    # 分析Beta变化
    beta_history = np.array(kf.beta_history)
    initial_beta_kf = beta_history[0]  # Kalman初始beta
    
    # 1. Beta符号分析
    sign_changes = 0
    negative_to_positive = 0
    positive_to_negative = 0
    
    for i in range(1, len(beta_history)):
        if np.sign(beta_history[i]) != np.sign(beta_history[i-1]):
            sign_changes += 1
            if beta_history[i-1] < 0 and beta_history[i] > 0:
                negative_to_positive += 1
            elif beta_history[i-1] > 0 and beta_history[i] < 0:
                positive_to_negative += 1
    
    # 2. Beta变化幅度
    beta_mean = np.mean(beta_history)
    beta_std = np.std(beta_history)
    beta_min = np.min(beta_history)
    beta_max = np.max(beta_history)
    beta_range = beta_max - beta_min
    
    # 3. Beta变异系数
    beta_cv = beta_std / abs(beta_mean) if beta_mean != 0 else np.inf
    
    # 4. Beta漂移分析
    beta_drift = beta_history[-1] - beta_history[0]
    beta_drift_rate = beta_drift / len(beta_history)
    
    # 5. 与初始Beta的偏离
    deviation_from_initial = abs(beta_mean - initial_beta)
    max_deviation = np.max(np.abs(beta_history - initial_beta))
    
    # 6. Beta稳定性评分
    stability_score = 0
    if sign_changes == 0:
        stability_score += 3  # 无符号变化
    elif sign_changes <= 2:
        stability_score += 2  # 少量符号变化
    elif sign_changes <= 5:
        stability_score += 1  # 中等符号变化
    
    if beta_cv < 0.1:
        stability_score += 3  # 变异系数很小
    elif beta_cv < 0.3:
        stability_score += 2  # 变异系数中等
    elif beta_cv < 0.5:
        stability_score += 1  # 变异系数较大
    
    if abs(beta_drift_rate) < 0.0001:
        stability_score += 2  # 漂移很小
    elif abs(beta_drift_rate) < 0.001:
        stability_score += 1  # 漂移中等
    
    # 判断是否稳定
    is_stable = (sign_changes == 0 and beta_cv < 0.3)
    
    results = {
        'pair': pair_name,
        'initial_beta_ols': initial_beta,
        'initial_beta_kf': initial_beta_kf,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'beta_range': beta_range,
        'beta_cv': beta_cv,
        'sign_changes': sign_changes,
        'neg_to_pos': negative_to_positive,
        'pos_to_neg': positive_to_negative,
        'beta_drift': beta_drift,
        'beta_drift_rate': beta_drift_rate,
        'deviation_from_initial': deviation_from_initial,
        'max_deviation': max_deviation,
        'stability_score': stability_score,
        'is_stable': is_stable
    }
    
    return results, beta_history


def plot_beta_evolution(beta_histories: dict, initial_betas: dict):
    """
    绘制Beta演化图
    """
    n_pairs = len(beta_histories)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (pair, beta_history) in enumerate(beta_histories.items()):
        # Beta时间序列
        ax = axes[idx, 0]
        ax.plot(beta_history, label='Kalman Beta', color='blue', alpha=0.7)
        ax.axhline(y=initial_betas[pair], color='red', linestyle='--', 
                   label=f'初始Beta: {initial_betas[pair]:.4f}', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 标记符号变化点
        for i in range(1, len(beta_history)):
            if np.sign(beta_history[i]) != np.sign(beta_history[i-1]):
                ax.axvline(x=i, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{pair} - Beta演化')
        ax.set_xlabel('时间')
        ax.set_ylabel('Beta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta分布
        ax = axes[idx, 1]
        ax.hist(beta_history, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=initial_betas[pair], color='red', linestyle='--', 
                   label=f'初始: {initial_betas[pair]:.4f}')
        ax.axvline(x=np.mean(beta_history), color='green', linestyle='--', 
                   label=f'均值: {np.mean(beta_history):.4f}')
        ax.set_title(f'{pair} - Beta分布')
        ax.set_xlabel('Beta值')
        ax.set_ylabel('频数')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"beta_stability_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    logger.info(f"Beta稳定性分析图已保存: {filename}")


def main():
    """
    主函数
    """
    logger.info("="*80)
    logger.info("Kalman滤波Beta稳定性分析")
    logger.info("="*80)
    
    # 1. 加载数据
    log_prices = load_all_symbols_data()
    
    # 2. 获取协整配对信息
    analyzer = CointegrationAnalyzer(log_prices)
    all_pairs = analyzer.screen_all_pairs()
    
    # 筛选稳定配对（5年p值<0.01）
    stable_pairs = all_pairs[all_pairs['pvalue_5y'] < 0.01].head(10)
    
    logger.info(f"\n分析{len(stable_pairs)}个稳定配对的Beta变化")
    logger.info("-"*60)
    
    results_list = []
    beta_histories = {}
    initial_betas = {}
    
    for idx, row in stable_pairs.iterrows():
        pair_name = row['pair']
        symbol_x = row['symbol_x']
        symbol_y = row['symbol_y']
        initial_beta = row.get('beta_1y', row.get('beta', 0))
        
        logger.info(f"\n分析配对: {pair_name}")
        logger.info(f"  初始Beta (1年OLS): {initial_beta:.6f}")
        
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        
        # 分析Beta稳定性
        result, beta_history = analyze_beta_stability(
            x_data, y_data, pair_name, initial_beta
        )
        
        results_list.append(result)
        beta_histories[pair_name] = beta_history
        initial_betas[pair_name] = initial_beta
        
        # 显示关键信息
        logger.info(f"  Kalman初始Beta: {result['initial_beta_kf']:.6f}")
        logger.info(f"  Beta均值: {result['beta_mean']:.6f}")
        logger.info(f"  Beta标准差: {result['beta_std']:.6f}")
        logger.info(f"  Beta范围: [{result['beta_min']:.6f}, {result['beta_max']:.6f}]")
        logger.info(f"  变异系数: {result['beta_cv']:.4f}")
        logger.info(f"  符号变化次数: {result['sign_changes']}")
        if result['neg_to_pos'] > 0:
            logger.info(f"  ⚠️ 负转正次数: {result['neg_to_pos']}")
        if result['pos_to_neg'] > 0:
            logger.info(f"  ⚠️ 正转负次数: {result['pos_to_neg']}")
        logger.info(f"  稳定性评分: {result['stability_score']}/8")
        logger.info(f"  是否稳定: {'是' if result['is_stable'] else '否'}")
    
    # 3. 汇总分析
    results_df = pd.DataFrame(results_list)
    
    logger.info("\n" + "="*80)
    logger.info("汇总分析")
    logger.info("="*80)
    
    # 统计符号变化
    total_pairs = len(results_df)
    no_sign_change = (results_df['sign_changes'] == 0).sum()
    has_sign_change = (results_df['sign_changes'] > 0).sum()
    has_neg_to_pos = (results_df['neg_to_pos'] > 0).sum()
    has_pos_to_neg = (results_df['pos_to_neg'] > 0).sum()
    
    logger.info(f"\n符号变化统计:")
    logger.info(f"  无符号变化: {no_sign_change}/{total_pairs} ({no_sign_change/total_pairs*100:.1f}%)")
    logger.info(f"  有符号变化: {has_sign_change}/{total_pairs} ({has_sign_change/total_pairs*100:.1f}%)")
    logger.info(f"  有负转正: {has_neg_to_pos}/{total_pairs} ({has_neg_to_pos/total_pairs*100:.1f}%)")
    logger.info(f"  有正转负: {has_pos_to_neg}/{total_pairs} ({has_pos_to_neg/total_pairs*100:.1f}%)")
    
    # Beta变异系数统计
    cv_small = (results_df['beta_cv'] < 0.1).sum()
    cv_medium = ((results_df['beta_cv'] >= 0.1) & (results_df['beta_cv'] < 0.3)).sum()
    cv_large = (results_df['beta_cv'] >= 0.3).sum()
    
    logger.info(f"\nBeta变异系数分布:")
    logger.info(f"  CV < 0.1 (很稳定): {cv_small}/{total_pairs}")
    logger.info(f"  0.1 ≤ CV < 0.3 (中等): {cv_medium}/{total_pairs}")
    logger.info(f"  CV ≥ 0.3 (不稳定): {cv_large}/{total_pairs}")
    
    # 稳定性评分统计
    high_stability = (results_df['stability_score'] >= 6).sum()
    medium_stability = ((results_df['stability_score'] >= 4) & (results_df['stability_score'] < 6)).sum()
    low_stability = (results_df['stability_score'] < 4).sum()
    
    logger.info(f"\n稳定性评分分布:")
    logger.info(f"  高稳定性 (≥6/8): {high_stability}/{total_pairs}")
    logger.info(f"  中稳定性 (4-5/8): {medium_stability}/{total_pairs}")
    logger.info(f"  低稳定性 (<4/8): {low_stability}/{total_pairs}")
    
    # 找出问题配对
    problem_pairs = results_df[
        (results_df['sign_changes'] > 0) | 
        (results_df['beta_cv'] > 0.3)
    ]
    
    if len(problem_pairs) > 0:
        logger.info(f"\n⚠️ 问题配对 (有符号变化或CV>0.3):")
        for _, row in problem_pairs.iterrows():
            logger.info(f"  {row['pair']}: 符号变化={row['sign_changes']}, CV={row['beta_cv']:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"beta_stability_results_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    logger.info(f"\n结果已保存: {filename}")
    
    # 绘制Beta演化图
    plot_beta_evolution(beta_histories, initial_betas)
    
    # 最终结论
    logger.info("\n" + "="*80)
    logger.info("结论")
    logger.info("="*80)
    
    avg_cv = results_df['beta_cv'].mean()
    stable_ratio = no_sign_change / total_pairs
    
    if stable_ratio > 0.8 and avg_cv < 0.2:
        logger.info("✅ Kalman滤波的Beta非常稳定，没有大量负转正的问题")
    elif stable_ratio > 0.6 and avg_cv < 0.3:
        logger.info("✅ Kalman滤波的Beta相对稳定，少量配对有符号变化")
    else:
        logger.info("⚠️ Kalman滤波的Beta稳定性需要改进")
    
    logger.info(f"\n关键指标:")
    logger.info(f"  平均变异系数: {avg_cv:.4f}")
    logger.info(f"  无符号变化比例: {stable_ratio*100:.1f}%")
    logger.info(f"  平均稳定性评分: {results_df['stability_score'].mean():.1f}/8")
    
    return results_df


if __name__ == "__main__":
    results = main()