#!/usr/bin/env python3
"""测试P0和R改进的效果"""

import numpy as np
import pandas as pd
from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveKalmanFilter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_pair():
    """测试单个配对的改进效果"""
    
    # 加载数据
    data = load_all_symbols_data()
    
    # 选择一个配对
    pair_name = "SM-RB"
    y_col = "SM"
    x_col = "RB"
    
    # 获取数据
    y_data = data[y_col].values
    x_data = data[x_col].values
    
    # 创建Kalman滤波器
    kf = AdaptiveKalmanFilter(pair_name=pair_name)
    
    # OLS预热
    ols_result = kf.warm_up_ols(y_data[-150:-90], x_data[-150:-90])
    
    logger.info(f"\nOLS初始化结果:")
    logger.info(f"  初始β: {ols_result['beta']:.6f}")
    logger.info(f"  初始R: {ols_result['R']:.6f}")
    logger.info(f"  初始P: {ols_result['P']:.6f}")
    
    # 测试X的方差
    x_var = np.var(x_data[-150:-90], ddof=1)
    logger.info(f"  X方差: {x_var:.6f}")
    logger.info(f"  P/R比例: {ols_result['P']/ols_result['R']:.6f}")
    
    # 运行Kalman更新
    z_scores = []
    for i in range(-90, -30):
        result = kf.update(y_data[i], x_data[i])
        z_scores.append(result['z'])
        
        # 每10步校准一次
        if (i + 90) % 10 == 0 and (i + 90) > 0:
            kf.calibrate_delta()
    
    # 计算z-score统计
    z_scores = np.array(z_scores)
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    z_var = np.var(z_scores)
    
    logger.info(f"\nZ-score统计:")
    logger.info(f"  均值: {z_mean:.6f}")
    logger.info(f"  标准差: {z_std:.6f}")
    logger.info(f"  方差: {z_var:.6f}")
    logger.info(f"  最小值: {np.min(z_scores):.6f}")
    logger.info(f"  最大值: {np.max(z_scores):.6f}")
    logger.info(f"  |z|>2的比例: {np.mean(np.abs(z_scores) > 2)*100:.2f}%")
    logger.info(f"  |z|>1.5的比例: {np.mean(np.abs(z_scores) > 1.5)*100:.2f}%")
    
    logger.info(f"\n最终参数:")
    logger.info(f"  最终β: {kf.beta:.6f}")
    logger.info(f"  最终P: {kf.P:.6f}")
    logger.info(f"  最终R: {kf.R:.6f}")
    logger.info(f"  最终δ: {kf.delta:.6f}")
    
    # 比较改进前后
    logger.info(f"\n改进效果分析:")
    logger.info(f"  P0使用X方差尺度: P0 = R0 / x_var = {ols_result['R']:.2f} / {x_var:.2f} = {ols_result['P']:.2f}")
    logger.info(f"  R的EWMA系数: λ=0.94 (之前0.96)")
    logger.info(f"  R的当期缩放: 0.85 * v^2")
    
    return z_scores

if __name__ == "__main__":
    test_single_pair()