#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较不同版本的Kalman滤波器
1. 工程版（EWMA均值调整）
2. 原始版（状态空间模型）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalman_engineering_version import EngineeringKalmanFilter
from kalman_original_version import OriginalKalmanFilter
from lib.data import load_all_symbols_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_kalman_versions(x_data: np.ndarray, y_data: np.ndarray, pair_name: str):
    """
    比较两个版本的Kalman滤波器
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"比较Kalman版本: {pair_name}")
    logger.info(f"{'='*60}")
    
    # 1. 工程版Kalman（默认参数）
    eng_kf = EngineeringKalmanFilter(
        warmup=60,
        rho=0.995,
        lambda_r=0.95,
        c=0.85,
        delta=0.98,
        z_in=2.0,
        z_out=0.5
    )
    
    # 2. 原始版Kalman（最优参数）
    orig_kf = OriginalKalmanFilter(
        warmup=60,
        Q_beta=5e-6,
        Q_alpha=1e-5,
        R_init=0.005,
        R_adapt=True,
        z_in=2.0,
        z_out=0.5
    )
    
    # 初始化
    eng_kf.initialize(x_data, y_data)
    orig_kf.initialize(x_data, y_data)
    
    # 运行滤波
    for i in range(60, len(x_data)):
        eng_result = eng_kf.update(x_data[i], y_data[i])
        orig_result = orig_kf.update(x_data[i], y_data[i])
    
    # 获取指标
    eng_metrics = eng_kf.get_metrics()
    orig_metrics = orig_kf.get_metrics()
    
    # 创建对比表
    comparison = pd.DataFrame({
        '工程版': [
            eng_metrics.get('z_var', np.nan),
            eng_metrics.get('z_mean', np.nan),
            eng_metrics.get('z_std', np.nan),
            eng_metrics.get('z_gt2_ratio', 0) * 100,
            eng_metrics.get('beta_std', np.nan),
            eng_metrics.get('innovation_adf_pvalue', np.nan),
            eng_metrics.get('innovation_stationary', False),
        ],
        '原始版': [
            orig_metrics.get('z_var', np.nan),
            orig_metrics.get('z_mean', np.nan),
            orig_metrics.get('z_std', np.nan),
            orig_metrics.get('z_gt2_ratio', 0) * 100,
            orig_metrics.get('beta_std', np.nan),
            orig_metrics.get('residual_adf_pvalue', np.nan),
            orig_metrics.get('residual_stationary', False),
        ]
    }, index=['Z方差', 'Z均值', 'Z标准差', 'Z>2比例(%)', 'Beta标准差', 'ADF p值', '平稳性'])
    
    logger.info("\n性能对比:")
    print(comparison.to_string())
    
    # 绘图比较
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'{pair_name} - Kalman版本对比', fontsize=16)
    
    # Z-score对比
    axes[0, 0].plot(eng_kf.z_history, label='工程版', alpha=0.7)
    axes[0, 0].plot(orig_kf.z_history, label='原始版', alpha=0.7)
    axes[0, 0].axhline(y=2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Z-score时间序列')
    axes[0, 0].set_ylabel('Z-score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Z-score分布
    axes[0, 1].hist(eng_kf.z_history, bins=50, alpha=0.5, label='工程版', density=True)
    axes[0, 1].hist(orig_kf.z_history, bins=50, alpha=0.5, label='原始版', density=True)
    axes[0, 1].set_title('Z-score分布')
    axes[0, 1].set_xlabel('Z-score')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beta对比
    axes[1, 0].plot(eng_kf.beta_history, label='工程版', alpha=0.7)
    axes[1, 0].plot(orig_kf.beta_history, label='原始版', alpha=0.7)
    axes[1, 0].set_title('Beta时间序列')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差/创新对比
    axes[1, 1].plot(eng_kf.innovation_history, label='工程版创新', alpha=0.7)
    axes[1, 1].plot(orig_kf.residual_history, label='原始版残差', alpha=0.7)
    axes[1, 1].set_title('残差/创新时间序列')
    axes[1, 1].set_ylabel('值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 累积信号数
    eng_signals = np.cumsum(np.abs(eng_kf.z_history) > 2)
    orig_signals = np.cumsum(np.abs(orig_kf.z_history) > 2)
    axes[2, 0].plot(eng_signals, label=f'工程版 (总计: {eng_signals[-1]})', alpha=0.7)
    axes[2, 0].plot(orig_signals, label=f'原始版 (总计: {orig_signals[-1]})', alpha=0.7)
    axes[2, 0].set_title('累积信号数')
    axes[2, 0].set_xlabel('时间')
    axes[2, 0].set_ylabel('累积信号数')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Q-Q图比较
    from scipy import stats
    stats.probplot(eng_kf.z_history, dist="norm", plot=axes[2, 1])
    axes[2, 1].get_lines()[0].set_label('工程版')
    stats.probplot(orig_kf.z_history, dist="norm", plot=axes[2, 1])
    axes[2, 1].get_lines()[2].set_label('原始版')
    axes[2, 1].set_title('Q-Q图（正态性检验）')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"kalman_comparison_{pair_name}_{timestamp}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    logger.info(f"\n对比图已保存: {filename}")
    
    return comparison, eng_kf, orig_kf


def test_multiple_pairs():
    """
    测试多个配对
    """
    logger.info("\n" + "="*80)
    logger.info("多配对Kalman版本对比")
    logger.info("="*80)
    
    # 加载数据
    log_prices = load_all_symbols_data()
    
    # 测试配对列表
    test_pairs = [
        ('AU', 'SS'),  # 稳定配对
        ('SS', 'SF'),  # 最稳定配对
        ('AL', 'SN'),  # 中等稳定
        ('AU', 'CU'),  # 正beta配对
        ('AU', 'PB'),  # 负beta配对
    ]
    
    all_results = []
    
    for symbol_x, symbol_y in test_pairs:
        pair_name = f"{symbol_x}-{symbol_y}"
        logger.info(f"\n{'#'*60}")
        logger.info(f"测试配对: {pair_name}")
        logger.info(f"{'#'*60}")
        
        x_data = log_prices[symbol_x].values
        y_data = log_prices[symbol_y].values
        
        comparison, eng_kf, orig_kf = compare_kalman_versions(x_data, y_data, pair_name)
        
        # 记录结果
        result = {
            'pair': pair_name,
            'eng_z_var': comparison.loc['Z方差', '工程版'],
            'orig_z_var': comparison.loc['Z方差', '原始版'],
            'eng_z_ratio': comparison.loc['Z>2比例(%)', '工程版'],
            'orig_z_ratio': comparison.loc['Z>2比例(%)', '原始版'],
            'eng_stationary': comparison.loc['平稳性', '工程版'],
            'orig_stationary': comparison.loc['平稳性', '原始版'],
        }
        all_results.append(result)
    
    # 汇总结果
    results_df = pd.DataFrame(all_results)
    
    logger.info("\n" + "="*60)
    logger.info("汇总对比结果:")
    logger.info("="*60)
    print(results_df.to_string())
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"kalman_version_comparison_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    logger.info(f"\n结果已保存: {filename}")
    
    # 统计分析
    logger.info("\n统计分析:")
    logger.info("-"*40)
    
    # Z方差比较
    eng_z_var_mean = results_df['eng_z_var'].mean()
    orig_z_var_mean = results_df['orig_z_var'].mean()
    logger.info(f"平均Z方差:")
    logger.info(f"  工程版: {eng_z_var_mean:.3f}")
    logger.info(f"  原始版: {orig_z_var_mean:.3f}")
    logger.info(f"  原始版/工程版: {orig_z_var_mean/eng_z_var_mean:.1f}x")
    
    # Z>2比例比较
    eng_ratio_mean = results_df['eng_z_ratio'].mean()
    orig_ratio_mean = results_df['orig_z_ratio'].mean()
    logger.info(f"\n平均Z>2比例:")
    logger.info(f"  工程版: {eng_ratio_mean:.1f}%")
    logger.info(f"  原始版: {orig_ratio_mean:.1f}%")
    
    # 平稳性比较
    eng_stationary = results_df['eng_stationary'].sum()
    orig_stationary = results_df['orig_stationary'].sum()
    logger.info(f"\n平稳配对数:")
    logger.info(f"  工程版: {eng_stationary}/{len(results_df)}")
    logger.info(f"  原始版: {orig_stationary}/{len(results_df)}")
    
    logger.info("\n结论:")
    if orig_z_var_mean > eng_z_var_mean * 1.5 and orig_ratio_mean > eng_ratio_mean * 2:
        logger.info("原始版Kalman滤波器显著优于工程版，建议使用原始版")
    elif orig_z_var_mean > eng_z_var_mean:
        logger.info("原始版Kalman滤波器表现更好，但优势不明显")
    else:
        logger.info("两个版本表现相当，需要进一步测试")


if __name__ == "__main__":
    test_multiple_pairs()