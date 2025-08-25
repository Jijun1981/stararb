#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程版Kalman滤波实现
最简单、能直接上手的方案
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class EngineeringKalmanFilter:
    """
    工程版Kalman滤波器
    只用一维β，两个旋钮：R的EWMA + δ折扣
    """
    
    def __init__(self,
                 warmup: int = 60,
                 rho: float = 0.995,      # EWMA均值（吃掉截距/慢漂移）
                 lambda_r: float = 0.95,   # R的EWMA参数
                 c: float = 0.85,          # R的EWMA系数
                 delta: float = 0.98,      # 折扣因子
                 z_in: float = 2.0,        # 开仓阈值
                 z_out: float = 0.5,       # 平仓阈值
                 beta_bounds: Tuple[float, float] = (-4, 4)):
        """
        初始化参数
        """
        self.warmup = warmup
        self.rho = rho
        self.lambda_r = lambda_r
        self.c = c
        self.delta = delta
        self.z_in = z_in
        self.z_out = z_out
        self.beta_bounds = beta_bounds
        
        # 状态变量
        self.beta = None
        self.P = None
        self.R = None
        self.mu_x = None
        self.mu_y = None
        self.sigma2_init = None
        
        # 历史记录
        self.z_history = []
        self.beta_history = []
        self.innovation_history = []
        self.S_history = []
        
    def initialize(self, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        """
        初始化（一次性）
        """
        if len(x_data) < self.warmup or len(y_data) < self.warmup:
            return False
        
        # 1. 用前warmup天做OLS
        X = x_data[:self.warmup].reshape(-1, 1)
        Y = y_data[:self.warmup]
        
        model = LinearRegression()
        model.fit(X, Y)
        b_ols = model.coef_[0]
        residuals = Y - model.predict(X)
        sigma2 = np.var(residuals)
        
        # 2. 设运行均值
        self.mu_x = np.mean(x_data[:self.warmup])
        self.mu_y = np.mean(y_data[:self.warmup])
        
        # 3. 设初值
        x_centered = x_data[:self.warmup] - self.mu_x
        self.beta = b_ols
        self.P = sigma2 / np.sum(x_centered**2)
        self.R = sigma2
        self.sigma2_init = sigma2
        
        return True
    
    def update(self, x_t: float, y_t: float) -> Dict:
        """
        每根更新（核心8步）
        """
        # 1) EWMA means（吃掉截距）
        self.mu_x = self.rho * self.mu_x + (1 - self.rho) * x_t
        self.mu_y = self.rho * self.mu_y + (1 - self.rho) * y_t
        x_c = x_t - self.mu_x
        y_c = y_t - self.mu_y
        
        # 2) 预测
        P_pred = self.P / self.delta
        y_hat = self.beta * x_c
        S_pred = x_c**2 * P_pred + self.R
        S_pred = max(S_pred, 1e-12)
        
        # 3) 创新与z
        v = y_c - y_hat
        z = v / np.sqrt(S_pred)
        
        # 4) 交易信号（这里只计算z，信号生成在外部）
        
        # 5) 仓位按风险配比（这里只记录S_pred）
        
        # 6) 滤波更新
        K = (P_pred * x_c) / S_pred
        self.beta = self.beta + K * v
        self.beta = np.clip(self.beta, self.beta_bounds[0], self.beta_bounds[1])
        self.P = (1 - K * x_c) * P_pred
        
        # 7) R的EWMA（用z²·S_pred，带夹板）
        z2 = min(z * z, 9.0)  # clip at 3σ
        self.R = self.lambda_r * self.R + (1 - self.lambda_r) * self.c * z2 * S_pred
        self.R = np.clip(self.R, 0.25 * self.sigma2_init, 20 * self.sigma2_init)
        
        # 8) 记录z用于监控
        self.z_history.append(z)
        self.beta_history.append(self.beta)
        self.innovation_history.append(v)
        self.S_history.append(S_pred)
        
        return {
            'z': z,
            'beta': self.beta,
            'innovation': v,
            'S_pred': S_pred,
            'P': self.P,
            'R': self.R
        }
    
    def calibrate_delta(self, window: int = 60) -> bool:
        """
        每周校准（可选）
        """
        if len(self.z_history) < window:
            return False
        
        z_var = np.var(self.z_history[-window:])
        
        if z_var > 1.3:
            self.delta = max(0.95, self.delta - 0.005)
            return True
        elif z_var < 0.8:
            self.delta = min(0.985, self.delta + 0.005)
            return True
        
        return False
    
    def get_metrics(self, window: int = 60) -> Dict:
        """
        获取评估指标
        """
        if len(self.z_history) < window:
            return {}
        
        recent_z = self.z_history[-window:]
        recent_beta = self.beta_history[-window:]
        recent_innovation = self.innovation_history[-window:]
        
        # 1. Z的统计
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        z_std = np.std(recent_z)
        z_gt2_ratio = np.mean(np.abs(recent_z) > 2.0)
        
        # 2. Beta的稳定性
        beta_std = np.std(recent_beta)
        beta_change_rate = np.mean(np.abs(np.diff(recent_beta)))
        
        # 3. 创新的平稳性（ADF检验）
        try:
            adf_stat, adf_pvalue = adfuller(recent_innovation)[:2]
            innovation_stationary = adf_pvalue < 0.05
        except:
            adf_pvalue = np.nan
            innovation_stationary = False
        
        return {
            'z_var': z_var,
            'z_mean': z_mean,
            'z_std': z_std,
            'z_gt2_ratio': z_gt2_ratio,
            'beta_std': beta_std,
            'beta_change_rate': beta_change_rate,
            'innovation_adf_pvalue': adf_pvalue,
            'innovation_stationary': innovation_stationary,
            'z_in_range': 0.8 <= z_var <= 1.3,
            'z_ratio_good': 0.02 <= z_gt2_ratio <= 0.05
        }


def check_mean_reversion(z_history: List[float], window: int = 20) -> Dict:
    """
    检查Z>2后是否在20天内完成均值回归
    """
    reversion_count = 0
    total_signals = 0
    reversion_times = []
    
    i = 0
    while i < len(z_history) - window:
        if abs(z_history[i]) > 2.0:
            total_signals += 1
            # 检查后续window天内是否回归到0附近
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
                            kalman_beta: List[float], window: int = 60) -> Dict:
    """
    与滚动OLS做对比
    """
    ols_betas = []
    
    for i in range(window, len(x_data)):
        X = x_data[i-window:i].reshape(-1, 1)
        Y = y_data[i-window:i]
        model = LinearRegression()
        model.fit(X, Y)
        ols_betas.append(model.coef_[0])
    
    # 对齐长度
    min_len = min(len(ols_betas), len(kalman_beta))
    ols_betas = ols_betas[:min_len]
    kalman_beta = kalman_beta[:min_len]
    
    # 计算相关性
    if len(ols_betas) > 0:
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


def run_single_test(x_data: np.ndarray, y_data: np.ndarray, params: Dict) -> Dict:
    """
    运行单次测试
    """
    kf = EngineeringKalmanFilter(
        warmup=params.get('warmup', 60),
        rho=params.get('rho', 0.995),
        lambda_r=params.get('lambda_r', 0.95),
        c=params.get('c', 0.85),
        delta=params.get('delta', 0.98),
        z_in=params.get('z_in', 2.0),
        z_out=params.get('z_out', 0.5),
        beta_bounds=params.get('beta_bounds', (-4, 4))
    )
    
    # 初始化
    if not kf.initialize(x_data, y_data):
        return {}
    
    # 运行
    for i in range(kf.warmup, len(x_data)):
        kf.update(x_data[i], y_data[i])
        
        # 定期校准（每60根）
        if i % 60 == 0 and i > kf.warmup + 60:
            kf.calibrate_delta()
    
    # 获取指标
    metrics = kf.get_metrics()
    
    # 检查均值回归
    reversion_metrics = check_mean_reversion(kf.z_history)
    metrics.update(reversion_metrics)
    
    # 与OLS对比
    ols_metrics = compare_with_rolling_ols(x_data, y_data, kf.beta_history)
    metrics.update(ols_metrics)
    
    # 综合评分
    score = 0
    if metrics.get('z_in_range', False):
        score += 2
    if metrics.get('z_ratio_good', False):
        score += 2
    if metrics.get('innovation_stationary', False):
        score += 1
    if metrics.get('reversion_rate', 0) > 0.7:
        score += 2
    if metrics.get('correlation', 0) > 0.6:
        score += 1
    if metrics.get('beta_change_rate', 1) < 0.02:
        score += 1
    
    metrics['score'] = score
    
    return metrics


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from lib.data import load_all_symbols_data
    
    # 加载数据
    log_prices = load_all_symbols_data()
    
    # 选择一个测试配对（AU-SS，最稳定的配对之一）
    x_data = log_prices['AU'].values
    y_data = log_prices['SS'].values
    
    # 测试默认参数
    params = {
        'warmup': 60,
        'rho': 0.995,
        'lambda_r': 0.95,
        'c': 0.85,
        'delta': 0.98,
        'z_in': 2.0,
        'z_out': 0.5,
        'beta_bounds': (-4, 4)
    }
    
    metrics = run_single_test(x_data, y_data, params)
    
    print("测试结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")