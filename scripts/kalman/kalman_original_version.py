#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始版Kalman滤波实现
不使用EWMA均值调整，直接用原始残差
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class OriginalKalmanFilter:
    """
    原始版Kalman滤波器
    不使用EWMA调整，直接估计beta和截距
    状态向量: [beta, alpha]'
    """
    
    def __init__(self,
                 warmup: int = 60,
                 Q_beta: float = 1e-5,    # beta的过程噪声
                 Q_alpha: float = 1e-4,   # 截距的过程噪声
                 R_init: float = 0.01,    # 初始观测噪声
                 R_adapt: bool = True,    # 是否自适应R
                 z_in: float = 2.0,       # 开仓阈值
                 z_out: float = 0.5):     # 平仓阈值
        """
        初始化参数
        """
        self.warmup = warmup
        self.Q = np.array([[Q_beta, 0], 
                          [0, Q_alpha]])  # 过程噪声协方差
        self.R_init = R_init
        self.R_adapt = R_adapt
        self.z_in = z_in
        self.z_out = z_out
        
        # 状态变量
        self.state = None  # [beta, alpha]'
        self.P = None      # 状态协方差
        self.R = None      # 观测噪声方差
        
        # 历史记录
        self.residual_history = []
        self.z_history = []
        self.beta_history = []
        self.alpha_history = []
        self.innovation_history = []
        
    def initialize(self, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        """
        使用OLS初始化
        """
        if len(x_data) < self.warmup or len(y_data) < self.warmup:
            return False
        
        # 1. OLS回归获取初始参数
        X = np.column_stack([x_data[:self.warmup], np.ones(self.warmup)])
        Y = y_data[:self.warmup]
        
        # 最小二乘法
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta_init = coeffs[0]
        alpha_init = coeffs[1]
        
        # 计算残差方差
        residuals = Y - X @ coeffs
        sigma2 = np.var(residuals)
        
        # 2. 初始化状态
        self.state = np.array([beta_init, alpha_init])
        
        # 3. 初始化协方差
        self.P = np.eye(2) * sigma2 * 0.01  # 初始不确定性
        
        # 4. 初始化观测噪声
        self.R = self.R_init if self.R_init > 0 else sigma2
        
        logger.info(f"初始化完成: beta={beta_init:.6f}, alpha={alpha_init:.6f}, R={self.R:.6f}")
        
        return True
    
    def update(self, x_t: float, y_t: float) -> Dict:
        """
        Kalman滤波更新
        """
        # 1. 预测步骤
        # 状态预测: x_k|k-1 = x_k-1|k-1 (随机游走)
        state_pred = self.state
        
        # 协方差预测: P_k|k-1 = P_k-1|k-1 + Q
        P_pred = self.P + self.Q
        
        # 2. 观测模型
        # H = [x_t, 1] (观测矩阵)
        H = np.array([x_t, 1.0])
        
        # 预测观测值: y_hat = H * state_pred
        y_hat = H @ state_pred
        
        # 3. 创新/残差
        innovation = y_t - y_hat
        
        # 创新协方差: S = H * P_pred * H' + R
        S = H @ P_pred @ H.T + self.R
        S = max(S, 1e-12)  # 数值稳定性
        
        # 4. 计算Z-score
        residual = innovation  # 原始残差
        z = residual / np.sqrt(self.R)  # 用R作为残差标准差
        
        # 5. Kalman增益
        K = P_pred @ H.T / S
        
        # 6. 状态更新
        self.state = state_pred + K * innovation
        
        # 7. 协方差更新
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        # 8. 自适应R（如果启用）
        if self.R_adapt and len(self.innovation_history) > 20:
            # 使用最近的创新来估计R
            recent_innovations = self.innovation_history[-20:]
            self.R = np.var(recent_innovations) * 0.9 + self.R * 0.1
            self.R = max(self.R, 1e-6)  # 保证正定
        
        # 9. 记录历史
        self.residual_history.append(residual)
        self.z_history.append(z)
        self.beta_history.append(self.state[0])
        self.alpha_history.append(self.state[1])
        self.innovation_history.append(innovation)
        
        return {
            'z': z,
            'residual': residual,
            'beta': self.state[0],
            'alpha': self.state[1],
            'innovation': innovation,
            'S': S,
            'P': self.P,
            'R': self.R
        }
    
    def get_metrics(self, window: int = 60) -> Dict:
        """
        获取评估指标
        """
        if len(self.z_history) < window:
            return {}
        
        recent_z = self.z_history[-window:]
        recent_beta = self.beta_history[-window:]
        recent_residual = self.residual_history[-window:]
        
        # Z统计
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        z_std = np.std(recent_z)
        z_gt2_ratio = np.mean(np.abs(recent_z) > 2.0)
        
        # Beta稳定性
        beta_std = np.std(recent_beta)
        beta_change_rate = np.mean(np.abs(np.diff(recent_beta)))
        
        # 残差平稳性
        try:
            adf_stat, adf_pvalue = adfuller(recent_residual)[:2]
            residual_stationary = adf_pvalue < 0.05
        except:
            adf_pvalue = np.nan
            residual_stationary = False
        
        return {
            'z_var': z_var,
            'z_mean': z_mean,
            'z_std': z_std,
            'z_gt2_ratio': z_gt2_ratio,
            'beta_std': beta_std,
            'beta_change_rate': beta_change_rate,
            'residual_adf_pvalue': adf_pvalue,
            'residual_stationary': residual_stationary,
            'z_in_range': 0.8 <= z_var <= 1.3,
            'z_ratio_good': 0.02 <= z_gt2_ratio <= 0.05
        }


def test_original_kalman(x_data: np.ndarray, y_data: np.ndarray, params: Dict) -> Dict:
    """
    测试原始Kalman滤波器
    """
    kf = OriginalKalmanFilter(
        warmup=params.get('warmup', 60),
        Q_beta=params.get('Q_beta', 1e-5),
        Q_alpha=params.get('Q_alpha', 1e-4),
        R_init=params.get('R_init', 0.01),
        R_adapt=params.get('R_adapt', True),
        z_in=params.get('z_in', 2.0),
        z_out=params.get('z_out', 0.5)
    )
    
    # 初始化
    if not kf.initialize(x_data, y_data):
        return {}
    
    # 运行滤波
    for i in range(kf.warmup, len(x_data)):
        kf.update(x_data[i], y_data[i])
    
    # 获取指标
    metrics = kf.get_metrics()
    
    # 检查均值回归
    reversion_metrics = check_mean_reversion(kf.z_history)
    metrics.update(reversion_metrics)
    
    # 与OLS对比
    ols_metrics = compare_with_rolling_ols(x_data, y_data, kf.beta_history, kf.warmup)
    metrics.update(ols_metrics)
    
    # 综合评分
    score = calculate_score(metrics)
    metrics['score'] = score
    
    return metrics


def check_mean_reversion(z_history: List[float], window: int = 20) -> Dict:
    """
    检查均值回归
    """
    reversion_count = 0
    total_signals = 0
    reversion_times = []
    
    i = 0
    while i < len(z_history) - window:
        if abs(z_history[i]) > 2.0:
            total_signals += 1
            for j in range(i + 1, min(i + window + 1, len(z_history))):
                if abs(z_history[j]) < 0.5:
                    reversion_count += 1
                    reversion_times.append(j - i)
                    break
            i += 1
        else:
            i += 1
    
    if total_signals > 0:
        reversion_rate = reversion_count / total_signals
        avg_reversion_time = np.mean(reversion_times) if reversion_times else np.nan
    else:
        reversion_rate = 0
        avg_reversion_time = np.nan
    
    return {
        'total_signals': total_signals,
        'reversion_count': reversion_count,
        'reversion_rate': reversion_rate,
        'avg_reversion_time': avg_reversion_time
    }


def compare_with_rolling_ols(x_data: np.ndarray, y_data: np.ndarray, 
                            kalman_beta: List[float], warmup: int) -> Dict:
    """
    与滚动OLS对比
    """
    window = 60
    ols_betas = []
    
    for i in range(warmup, len(x_data)):
        start_idx = max(0, i - window)
        X = x_data[start_idx:i].reshape(-1, 1)
        Y = y_data[start_idx:i]
        
        if len(X) >= 20:  # 至少20个点
            model = LinearRegression()
            model.fit(X, Y)
            ols_betas.append(model.coef_[0])
    
    # 对齐长度
    min_len = min(len(ols_betas), len(kalman_beta))
    if min_len > 0:
        ols_betas = ols_betas[:min_len]
        kalman_beta = kalman_beta[:min_len]
        
        correlation = np.corrcoef(ols_betas, kalman_beta)[0, 1]
        beta_diff_mean = np.mean(np.abs(np.array(kalman_beta) - np.array(ols_betas)))
        beta_diff_std = np.std(np.array(kalman_beta) - np.array(ols_betas))
    else:
        correlation = np.nan
        beta_diff_mean = np.nan
        beta_diff_std = np.nan
    
    return {
        'correlation': correlation,
        'beta_diff_mean': beta_diff_mean,
        'beta_diff_std': beta_diff_std
    }


def calculate_score(metrics: Dict) -> int:
    """
    计算综合评分
    """
    score = 0
    
    # Z方差接近1（2分）
    z_var = metrics.get('z_var', 0)
    if 0.8 <= z_var <= 1.3:
        score += 2
    elif 0.6 <= z_var <= 1.5:
        score += 1
    
    # Z>2比例在2-5%（2分）
    z_ratio = metrics.get('z_gt2_ratio', 0)
    if 0.02 <= z_ratio <= 0.05:
        score += 2
    elif 0.01 <= z_ratio <= 0.08:
        score += 1
    
    # 残差平稳（1分）
    if metrics.get('residual_stationary', False):
        score += 1
    
    # 均值回归率>70%（2分）
    reversion_rate = metrics.get('reversion_rate', 0)
    if reversion_rate > 0.7:
        score += 2
    elif reversion_rate > 0.5:
        score += 1
    
    # 与OLS相关性>0.6（1分）
    correlation = metrics.get('correlation', 0)
    if correlation > 0.6:
        score += 1
    
    # Beta稳定性（1分）
    beta_change = metrics.get('beta_change_rate', 1)
    if beta_change < 0.02:
        score += 1
    
    return score


def grid_search_original(x_data: np.ndarray, y_data: np.ndarray) -> pd.DataFrame:
    """
    网格搜索原始Kalman参数
    """
    logger.info("原始Kalman参数网格搜索")
    
    param_grid = {
        'Q_beta': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        'Q_alpha': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'R_init': [0.001, 0.005, 0.01, 0.05, 0.1],
        'R_adapt': [True, False]
    }
    
    results = []
    total = len(param_grid['Q_beta']) * len(param_grid['Q_alpha']) * len(param_grid['R_init']) * len(param_grid['R_adapt'])
    count = 0
    
    for Q_beta in param_grid['Q_beta']:
        for Q_alpha in param_grid['Q_alpha']:
            for R_init in param_grid['R_init']:
                for R_adapt in param_grid['R_adapt']:
                    count += 1
                    
                    params = {
                        'warmup': 60,
                        'Q_beta': Q_beta,
                        'Q_alpha': Q_alpha,
                        'R_init': R_init,
                        'R_adapt': R_adapt,
                        'z_in': 2.0,
                        'z_out': 0.5
                    }
                    
                    logger.info(f"测试 {count}/{total}: Q_β={Q_beta:.1e}, Q_α={Q_alpha:.1e}, R={R_init:.3f}, adapt={R_adapt}")
                    
                    metrics = test_original_kalman(x_data, y_data, params)
                    
                    if metrics:
                        result = {**params, **metrics}
                        results.append(result)
                        
                        # 显示关键指标
                        logger.info(f"  Z方差: {metrics.get('z_var', np.nan):.3f}")
                        logger.info(f"  Z>2比例: {metrics.get('z_gt2_ratio', 0)*100:.1f}%")
                        logger.info(f"  评分: {metrics.get('score', 0)}/9")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime
    
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from lib.data import load_all_symbols_data
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据
    log_prices = load_all_symbols_data()
    
    # 使用AU-SS配对测试
    x_data = log_prices['AU'].values
    y_data = log_prices['SS'].values
    
    logger.info("="*60)
    logger.info("原始Kalman滤波器测试")
    logger.info("="*60)
    
    # 1. 测试默认参数
    logger.info("\n1. 默认参数测试")
    default_params = {
        'warmup': 60,
        'Q_beta': 1e-5,
        'Q_alpha': 1e-4,
        'R_init': 0.01,
        'R_adapt': True,
        'z_in': 2.0,
        'z_out': 0.5
    }
    
    metrics = test_original_kalman(x_data, y_data, default_params)
    
    logger.info("\n默认参数结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # 2. 网格搜索
    logger.info("\n2. 参数网格搜索")
    results_df = grid_search_original(x_data, y_data)
    
    if len(results_df) > 0:
        # 排序并显示最佳参数
        results_df = results_df.sort_values('score', ascending=False)
        best = results_df.iloc[0]
        
        logger.info("\n最佳参数:")
        logger.info(f"  Q_beta: {best['Q_beta']:.1e}")
        logger.info(f"  Q_alpha: {best['Q_alpha']:.1e}")
        logger.info(f"  R_init: {best['R_init']:.3f}")
        logger.info(f"  R_adapt: {best['R_adapt']}")
        logger.info(f"  评分: {best['score']}/9")
        logger.info(f"  Z方差: {best['z_var']:.3f}")
        logger.info(f"  Z>2比例: {best['z_gt2_ratio']*100:.1f}%")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"original_kalman_grid_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"\n结果已保存: {filename}")