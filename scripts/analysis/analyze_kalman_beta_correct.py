#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确分析Kalman Beta稳定性
比较60天OLS Beta与Kalman运行后的Beta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import os
from typing import Dict
from sklearn.linear_model import LinearRegression

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


def analyze_beta_evolution(x_data: np.ndarray, y_data: np.ndarray, 
                          pair_name: str) -> Dict:
    """
    分析Beta从60天OLS到Kalman的演化
    """
    # 1. 计算60天OLS Beta（这是Kalman的真正初始值）
    X = x_data[:60].reshape(-1, 1)
    Y = y_data[:60]
    model = LinearRegression()
    model.fit(X, Y)
    ols_beta_60d = model.coef_[0]
    
    # 2. 运行Kalman滤波器
    kf = OriginalKalmanFilter(
        warmup=60,
        Q_beta=5e-6,
        Q_alpha=1e-5,
        R_init=0.005,
        R_adapt=True
    )
    
    kf.initialize(x_data, y_data)
    
    for i in range(60, len(x_data)):
        kf.update(x_data[i], y_data[i])
    
    beta_history = np.array(kf.beta_history)
    
    # 3. 分析Beta变化
    # 第一个记录的beta应该非常接近60天OLS
    first_kalman_beta = beta_history[0] if len(beta_history) > 0 else ols_beta_60d
    final_kalman_beta = beta_history[-1] if len(beta_history) > 0 else ols_beta_60d
    
    # 符号变化分析
    sign_changes = 0
    for i in range(1, len(beta_history)):
        if np.sign(beta_history[i]) != np.sign(beta_history[i-1]):
            sign_changes += 1
    
    # 统计指标
    beta_mean = np.mean(beta_history)
    beta_std = np.std(beta_history)
    beta_min = np.min(beta_history)
    beta_max = np.max(beta_history)
    
    # 与初始60天OLS的偏离
    max_deviation_from_ols = np.max(np.abs(beta_history - ols_beta_60d))
    final_deviation_from_ols = abs(final_kalman_beta - ols_beta_60d)
    
    # 检查是否有符号反转（相对于60天OLS）
    sign_flipped = np.sign(ols_beta_60d) != np.sign(final_kalman_beta)
    
    results = {
        'pair': pair_name,
        'ols_beta_60d': ols_beta_60d,
        'first_kalman_beta': first_kalman_beta,
        'final_kalman_beta': final_kalman_beta,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'sign_changes': sign_changes,
        'sign_flipped': sign_flipped,
        'max_deviation': max_deviation_from_ols,
        'final_deviation': final_deviation_from_ols,
        'cv': beta_std / abs(beta_mean) if beta_mean != 0 else np.inf
    }
    
    return results, beta_history, ols_beta_60d


def main():
    """
    主函数
    """
    logger.info("="*80)
    logger.info("Kalman Beta演化分析（60天OLS -> Kalman）")
    logger.info("="*80)
    
    # 加载数据
    log_prices = load_all_symbols_data()
    
    # 测试主要配对
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
    
    results_list = []
    
    # 创建图表
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for idx, (symbol_x, symbol_y) in enumerate(test_pairs[:10]):
        pair_name = f"{symbol_x}-{symbol_y}"
        logger.info(f"\n分析配对: {pair_name}")
        
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        
        # 分析Beta演化
        result, beta_history, ols_beta_60d = analyze_beta_evolution(
            x_data, y_data, pair_name
        )
        
        results_list.append(result)
        
        # 显示结果
        logger.info(f"  60天OLS Beta: {result['ols_beta_60d']:.6f}")
        logger.info(f"  首个Kalman Beta: {result['first_kalman_beta']:.6f}")
        logger.info(f"  最终Kalman Beta: {result['final_kalman_beta']:.6f}")
        logger.info(f"  Beta均值: {result['beta_mean']:.6f}")
        logger.info(f"  Beta标准差: {result['beta_std']:.6f}")
        logger.info(f"  符号变化次数: {result['sign_changes']}")
        if result['sign_flipped']:
            logger.info(f"  ⚠️ Beta符号反转了！")
        logger.info(f"  最大偏离: {result['max_deviation']:.6f}")
        logger.info(f"  最终偏离: {result['final_deviation']:.6f}")
        
        # 绘图
        ax = axes[idx]
        ax.plot(beta_history, label='Kalman Beta', color='blue', alpha=0.7)
        ax.axhline(y=ols_beta_60d, color='red', linestyle='--', 
                   label=f'60d OLS: {ols_beta_60d:.4f}', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{pair_name}')
        ax.set_xlabel('Days after warmup')
        ax.set_ylabel('Beta')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kalman Beta Evolution (60-day OLS → Kalman)', fontsize=14)
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'beta_evolution_{timestamp}.png', dpi=100, bbox_inches='tight')
    logger.info(f"\n图表已保存: beta_evolution_{timestamp}.png")
    
    # 汇总分析
    results_df = pd.DataFrame(results_list)
    
    logger.info("\n" + "="*80)
    logger.info("汇总分析")
    logger.info("="*80)
    
    # 统计
    total = len(results_df)
    no_sign_change = (results_df['sign_changes'] == 0).sum()
    sign_flipped = results_df['sign_flipped'].sum()
    
    logger.info(f"\n稳定性统计:")
    logger.info(f"  无符号变化: {no_sign_change}/{total} ({no_sign_change/total*100:.1f}%)")
    logger.info(f"  Beta符号反转: {sign_flipped}/{total} ({sign_flipped/total*100:.1f}%)")
    
    # 偏离度分析
    avg_max_deviation = results_df['max_deviation'].mean()
    avg_final_deviation = results_df['final_deviation'].mean()
    avg_cv = results_df['cv'].mean()
    
    logger.info(f"\n偏离度分析:")
    logger.info(f"  平均最大偏离: {avg_max_deviation:.6f}")
    logger.info(f"  平均最终偏离: {avg_final_deviation:.6f}")
    logger.info(f"  平均变异系数: {avg_cv:.4f}")
    
    # 找出问题配对
    problem_pairs = results_df[
        (results_df['sign_changes'] > 2) | 
        (results_df['sign_flipped'] == True)
    ]
    
    if len(problem_pairs) > 0:
        logger.info(f"\n⚠️ 问题配对:")
        for _, row in problem_pairs.iterrows():
            logger.info(f"  {row['pair']}: 符号变化={row['sign_changes']}, "
                       f"符号反转={'是' if row['sign_flipped'] else '否'}")
    
    # 保存结果
    results_df.to_csv(f'beta_evolution_results_{timestamp}.csv', index=False)
    logger.info(f"\n结果已保存: beta_evolution_results_{timestamp}.csv")
    
    logger.info("\n" + "="*80)
    logger.info("结论")
    logger.info("="*80)
    
    if no_sign_change/total > 0.7 and avg_cv < 0.2:
        logger.info("✅ Kalman Beta相对60天OLS初始值稳定")
    else:
        logger.info("⚠️ 部分配对的Kalman Beta不够稳定")
    
    return results_df


if __name__ == "__main__":
    results = main()