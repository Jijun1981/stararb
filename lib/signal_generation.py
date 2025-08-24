#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块 - 自适应双旋钮Kalman滤波实现
严格按照需求文档 docs/Requirements/03_signal_generation.md 实现
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression

# 设置日志
logger = logging.getLogger(__name__)


class AdaptiveKalmanFilter:
    """
    自适应Kalman滤波器 - REQ-3.1
    使用折扣因子和EWMA自适应测量噪声
    """
    
    def __init__(self, 
                 pair_name: str,
                 delta: float = 0.93,      # 优化参数: δ=0.93 (平稳性优先)
                 lambda_r: float = 0.89,    # 优化参数: λ=0.89 (平稳性优先)
                 beta_bounds: Optional[Tuple[float, float]] = None):  # 移除β边界限制
        """
        初始化自适应Kalman滤波器
        
        Args:
            pair_name: 配对名称
            delta: 折扣因子（优化后默认0.96）
            lambda_r: R的EWMA参数（优化后默认0.92）
            beta_bounds: β边界限制（已移除限制）
        """
        self.pair_name = pair_name
        self.delta = delta
        self.lambda_r = lambda_r
        self.beta_bounds = beta_bounds
        
        # 状态变量（通过OLS预热初始化）
        self.beta = None
        self.P = None
        self.R = None
        
        # 去中心化参数
        self.mu_x = None
        self.mu_y = None
        
        # 历史记录
        self.z_history = []
        self.beta_history = []
        self.innovation_history = []
        self.calibration_log = []
        self.last_calibration_step = 0
        
    def warm_up_ols(self, x_data: np.ndarray, y_data: np.ndarray, window: int = 60) -> Dict:
        """
        OLS预热获得初始参数 - REQ-3.1.3
        使用去中心化处理避免截距问题
        
        Args:
            x_data: X价格序列
            y_data: Y价格序列
            window: OLS窗口大小
            
        Returns:
            包含初始参数的字典
        """
        # 去中心化处理（关键：避免截距问题导致R膨胀）
        self.mu_x = np.mean(x_data[:window])
        self.mu_y = np.mean(y_data[:window])
        x_use = x_data[:window] - self.mu_x
        y_use = y_data[:window] - self.mu_y
        
        # OLS回归估计初始β（基于去中心化数据）
        reg = LinearRegression(fit_intercept=False)  # 不需要截距
        reg.fit(x_use.reshape(-1, 1), y_use)
        
        self.beta = float(reg.coef_[0])
        innovations = y_use - reg.predict(x_use.reshape(-1, 1)).flatten()
        self.R = float(np.var(innovations, ddof=1))  # 初始R为创新方差
        
        # P0初始化：使用Var(x)标定，不再乘0.1
        x_var = np.var(x_use, ddof=1)
        self.P = self.R / max(x_var, 1e-12)  # 让x²*P与R同量级
        
        logger.debug(f"{self.pair_name} OLS初始化: β={self.beta:.4f}, R={self.R:.6f}, P={self.P:.6f}")
        
        return {
            'beta': self.beta, 
            'R': self.R, 
            'P': self.P,
            'mu_x': self.mu_x, 
            'mu_y': self.mu_y
        }
    
    def update(self, y_t: float, x_t: float) -> Dict:
        """
        折扣Kalman更新 - REQ-3.1.5
        
        Args:
            y_t: 当前Y观测值
            x_t: 当前X观测值
            
        Returns:
            包含更新结果的字典
        """
        if self.beta is None or self.P is None or self.R is None:
            raise ValueError("必须先调用warm_up_ols初始化")
        
        # 1. 折扣先验协方差（等价于加入过程噪声）- REQ-3.1.5
        P_prior = self.P / self.delta
        
        # 2. 预测
        beta_pred = self.beta  # 随机游走
        y_pred = beta_pred * x_t
        
        # 3. 创新
        v = y_t - y_pred
        S = x_t * P_prior * x_t + self.R
        S = max(S, 1e-12)  # 数值稳定性
        
        # 4. 创新标准化 z = v/√S - REQ-3.3.1
        z = v / np.sqrt(S)
        
        # 5. Kalman增益
        K = P_prior * x_t / S
        
        # 6. 状态更新
        beta_new = beta_pred + K * v
        
        # 7. β边界保护（可选）
        if self.beta_bounds is not None:
            beta_new = np.clip(beta_new, self.beta_bounds[0], self.beta_bounds[1])
        
        # 8. 后验协方差
        self.P = (1 - K * x_t) * P_prior
        
        # 9. R自适应（EWMA） - REQ-3.1.4
        self.R = self.lambda_r * self.R + (1 - self.lambda_r) * (v ** 2)
        
        # 10. 更新状态
        self.beta = beta_new
        self.z_history.append(z)
        self.beta_history.append(self.beta)
        self.innovation_history.append(v)
        
        return {
            'beta': self.beta,
            'v': v,      # 创新
            'S': S,      # 创新方差
            'z': z,      # 创新标准化
            'R': self.R,
            'K': K,
            'P': self.P
        }
    
    def calibrate_delta(self, window: int = 60) -> bool:
        """
        自动校准δ - REQ-3.2
        
        Args:
            window: 用于计算z方差的窗口大小
            
        Returns:
            是否进行了校准
        """
        if len(self.z_history) < window:
            return False
        
        # 计算最近window个z的方差 - REQ-3.2.3
        z_var = np.var(self.z_history[-window:])
        
        old_delta = self.delta
        
        # δ调整规则 - REQ-3.2.4 (优化后下界)
        if z_var > 1.3:
            self.delta = max(self.delta - 0.01, 0.90)  # 优化后: 下界0.90
        elif z_var < 0.8:
            self.delta = min(self.delta + 0.01, 0.995)  # REQ-3.2.5: 上界0.995
        else:
            return False  # 在目标范围内，无需调整
        
        # 记录校准日志 - REQ-3.2.7
        self.calibration_log.append({
            'step': len(self.z_history),
            'z_var': z_var,
            'old_delta': old_delta,
            'new_delta': self.delta,
            'timestamp': datetime.now()
        })
        
        logger.debug(f"{self.pair_name} δ校准: z_var={z_var:.3f}, δ: {old_delta:.3f}->{self.delta:.3f}")
        
        return True
    
    def get_quality_metrics(self, window: int = 60) -> Dict:
        """
        获取质量指标 - REQ-3.5
        
        Args:
            window: 计算指标的窗口大小
            
        Returns:
            质量指标字典
        """
        if len(self.z_history) < window:
            return {
                'z_var': np.nan,
                'z_mean': np.nan,
                'z_std': np.nan,
                'quality_status': 'insufficient_data',
                'current_delta': self.delta,
                'current_R': self.R,
                'current_beta': self.beta
            }
        
        recent_z = self.z_history[-window:]
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        z_std = np.std(recent_z)
        
        # 质量评级 - REQ-3.5.3
        if 0.8 <= z_var <= 1.3:
            quality_status = 'good'
        elif 0.6 <= z_var < 0.8 or 1.3 < z_var <= 1.5:
            quality_status = 'warning'
        else:
            quality_status = 'bad'
        
        return {
            'z_var': z_var,
            'z_mean': z_mean,
            'z_std': z_std,
            'quality_status': quality_status,
            'current_delta': self.delta,
            'current_R': self.R,
            'current_beta': self.beta
        }


def generate_signal(z_score: float, 
                   position: Optional[str], 
                   days_held: int,
                   z_open: float, 
                   z_close: float, 
                   max_days: int) -> str:
    """
    生成交易信号 - REQ-3.3
    
    Args:
        z_score: 标准化创新
        position: 当前持仓状态 ('long'/'short'/None)
        days_held: 持仓天数
        z_open: 开仓阈值
        z_close: 平仓阈值
        max_days: 最大持仓天数
        
    Returns:
        信号字符串
    """
    # 强制平仓（最高优先级）- REQ-3.3.3, REQ-3.3.6
    if position and days_held >= max_days:
        return 'close'
    
    # 平仓条件 - REQ-3.3.2
    if position and abs(z_score) <= z_close:
        return 'close'
    
    # 开仓条件（仅在空仓时）- REQ-3.3.1, REQ-3.3.5
    if not position:
        if z_score <= -z_open:
            return 'open_long'
        elif z_score >= z_open:
            return 'open_short'
        return 'empty'  # 空仓等待
    
    # 持仓状态 - REQ-3.3.4
    if position == 'long':
        return 'holding_long'
    elif position == 'short':
        return 'holding_short'
    
    return 'empty'


class AdaptiveSignalGenerator:
    """
    自适应信号生成器主类 - REQ-3.4
    """
    
    def __init__(self,
                 z_open: float = 2.0,      # REQ-3.3.1
                 z_close: float = 0.5,     # REQ-3.3.2
                 max_holding_days: int = 30,  # REQ-3.3.3
                 calibration_freq: int = 5,   # REQ-3.2.1
                 ols_window: int = 60,        # REQ-3.1.3
                 warm_up_days: int = 60):     # REQ-3.1.10
        """
        初始化信号生成器
        
        Args:
            z_open: 开仓阈值
            z_close: 平仓阈值
            max_holding_days: 最大持仓天数
            calibration_freq: 校准频率（天）
            ols_window: OLS预热窗口
            warm_up_days: Kalman预热天数
        """
        self.z_open = z_open
        self.z_close = z_close
        self.max_holding_days = max_holding_days
        self.calibration_freq = calibration_freq
        self.ols_window = ols_window
        self.warm_up_days = warm_up_days
        
        # 存储各配对的Kalman滤波器
        self.pair_filters: Dict[str, AdaptiveKalmanFilter] = {}
        
        # 存储所有信号
        self.all_signals: List[pd.DataFrame] = []
        
    def process_pair(self,
                    pair_name: str,
                    x_data: pd.Series,
                    y_data: pd.Series,
                    initial_beta: Optional[float] = None) -> pd.DataFrame:
        """
        处理单个配对 - REQ-3.4.3
        
        Args:
            pair_name: 配对名称
            x_data: X价格序列（对数价格）
            y_data: Y价格序列（对数价格）
            initial_beta: 初始β（可选）
            
        Returns:
            信号DataFrame
        """
        # 对齐数据
        common_dates = x_data.index.intersection(y_data.index)
        x_aligned = x_data[common_dates].values
        y_aligned = y_data[common_dates].values
        dates = common_dates
        
        if len(x_aligned) < self.ols_window:
            logger.warning(f"{pair_name}: 数据不足，需要至少{self.ols_window}个数据点")
            return pd.DataFrame()
        
        # 初始化Kalman滤波器
        kf = AdaptiveKalmanFilter(pair_name)
        self.pair_filters[pair_name] = kf
        
        # OLS预热 - REQ-3.1.3: 60日窗口估计初始β
        kf.warm_up_ols(x_aligned, y_aligned, self.ols_window)
        
        # 注释掉β覆盖逻辑：让OLS预热自己估计初始β，符合REQ-3.1.3要求
        # if initial_beta is not None:
        #     kf.beta = initial_beta
        
        signals = []
        position = None
        days_held = 0
        calibration_counter = 0
        
        # 1. OLS预热期（前ols_window天）
        for i in range(self.ols_window):
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': 'warm_up',
                'z_score': 0.0,
                'innovation': 0.0,
                'beta': kf.beta,
                'days_held': 0,
                'reason': 'ols_warmup',
                'phase': 'warm_up',
                'delta': kf.delta,
                'R': kf.R,
                'P': kf.P
            })
        
        # 2. Kalman预热期
        warm_up_end = min(self.ols_window + self.warm_up_days, len(x_aligned))
        
        for i in range(self.ols_window, warm_up_end):
            result = kf.update(y_aligned[i], x_aligned[i])
            
            # 预热期不生成交易信号
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': 'warm_up',
                'z_score': result['z'],
                'innovation': result['v'],
                'beta': result['beta'],
                'days_held': 0,
                'reason': 'kalman_warmup',
                'phase': 'convergence_period',
                'delta': kf.delta,
                'R': result['R'],
                'P': result['P']
            })
            
            # 预热期也进行校准
            calibration_counter += 1
            if calibration_counter % self.calibration_freq == 0:
                kf.calibrate_delta()
        
        # 3. 正式交易期
        for i in range(warm_up_end, len(x_aligned)):
            # Kalman更新
            result = kf.update(y_aligned[i], x_aligned[i])
            
            # 使用创新标准化z生成信号
            z = result['z']
            signal = generate_signal(z, position, days_held,
                                   self.z_open, self.z_close, self.max_holding_days)
            
            # 确定原因
            if signal == 'close':
                if days_held >= self.max_holding_days:
                    reason = 'force_close'
                else:
                    reason = 'z_threshold'
            elif signal.startswith('open'):
                reason = 'z_threshold'
            elif signal.startswith('holding'):
                reason = 'holding'
            else:
                reason = 'no_signal'
            
            # 更新持仓状态
            if signal == 'open_long':
                position = 'long'
                days_held = 1
            elif signal == 'open_short':
                position = 'short'
                days_held = 1
            elif signal == 'close':
                position = None
                days_held = 0
            elif position:
                days_held += 1
            
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': signal,
                'z_score': z,
                'innovation': result['v'],
                'beta': result['beta'],
                'days_held': days_held,
                'reason': reason,
                'phase': 'signal_period',
                'delta': kf.delta,
                'R': result['R'],
                'P': result['P']
            })
            
            # 定期校准
            calibration_counter += 1
            if calibration_counter % self.calibration_freq == 0:
                kf.calibrate_delta()
        
        df = pd.DataFrame(signals)
        self.all_signals.append(df)
        return df
    
    def process_all_pairs(self,
                         pairs_df: pd.DataFrame,
                         price_data: pd.DataFrame,
                         beta_window: str = '1y') -> pd.DataFrame:
        """
        批量处理配对 - REQ-3.4.1, REQ-3.4.2
        
        Args:
            pairs_df: 协整模块输出的配对DataFrame
            price_data: 价格数据DataFrame
            beta_window: 使用的β时间窗口
            
        Returns:
            所有配对的信号DataFrame
        """
        all_results = []
        
        beta_col = f'beta_{beta_window}'
        if beta_col not in pairs_df.columns:
            logger.warning(f"配对数据中没有{beta_col}列，使用OLS自动估计")
            beta_col = None
        
        for idx, pair_info in pairs_df.iterrows():
            pair_name = pair_info['pair']
            symbol_x = pair_info['symbol_x']
            symbol_y = pair_info['symbol_y']
            
            logger.info(f"处理配对: {pair_name}")
            
            # 检查数据
            if symbol_x not in price_data.columns or symbol_y not in price_data.columns:
                logger.warning(f"配对{pair_name}的价格数据不完整，跳过")
                continue
            
            x_data = price_data[symbol_x].dropna()
            y_data = price_data[symbol_y].dropna()
            
            # 获取初始beta
            initial_beta = pair_info[beta_col] if beta_col else None
            
            # 处理配对
            signals_df = self.process_pair(pair_name, x_data, y_data, initial_beta)
            
            if not signals_df.empty:
                # 添加额外信息
                signals_df['symbol_x'] = symbol_x
                signals_df['symbol_y'] = symbol_y
                signals_df['beta_initial'] = initial_beta
                signals_df['beta_window_used'] = beta_window
                
                all_results.append(signals_df)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_quality_report(self) -> pd.DataFrame:
        """
        获取质量报告 - REQ-3.5.4
        
        Returns:
            质量报告DataFrame
        """
        reports = []
        
        for pair_name, kf in self.pair_filters.items():
            metrics = kf.get_quality_metrics()
            
            reports.append({
                'pair': pair_name,
                'z_var': metrics['z_var'],
                'z_mean': metrics['z_mean'],
                'z_std': metrics['z_std'],
                'current_delta': metrics['current_delta'],
                'current_R': metrics['current_R'],
                'current_beta': metrics['current_beta'],
                'quality_status': metrics['quality_status'],
                'calibration_count': len(kf.calibration_log),
                'total_signals': len(kf.z_history)
            })
        
        return pd.DataFrame(reports)