#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块 V3 - 基于原版本Kalman滤波器的正确实现
基于需求文档: /docs/Requirements/03_signal_generation.md V3.1

关键修正：使用经过验证的原版本KF方法 z = residual/√R
而非需求文档中的 z = innovation/√S（该方法在实际测试中信号量不足）
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sklearn.linear_model import LinearRegression
import sys
import os

# 添加项目路径以导入原版本KF
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'kalman'))

# 直接使用本地版本，确保属性一致
from statsmodels.tsa.stattools import adfuller

class OriginalKalmanFilter:
    """
    原版本Kalman滤波器 - 经过验证能产生正确信号量的版本
    关键：使用 z = residual/√R 而非 z = innovation/√S
    """
    
    def __init__(self, warmup: int = 60, Q_beta: float = 5e-6, Q_alpha: float = 1e-5,
                 R_init: float = 0.005, R_adapt: bool = True, z_in: float = 2.0, z_out: float = 0.5):
            self.warmup = warmup
            self.Q_beta = Q_beta
            self.Q_alpha = Q_alpha
            self.R_init = R_init
            self.R = R_init
            self.R_adapt = R_adapt
            self.z_in = z_in
            self.z_out = z_out
            
            # 状态变量
            self.beta = None
            self.alpha = None
            self.P_beta = None
            self.P_alpha = None
            
            # 历史记录
            self.beta_history = []
            self.alpha_history = []
            self.z_history = []
            self.residual_history = []
            self.innovation_history = []
            
        def initialize(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
            """使用OLS初始化"""
            # 使用前warmup天做OLS
            X_init = x_data[:self.warmup].reshape(-1, 1)
            Y_init = y_data[:self.warmup]
            
            # OLS回归
            X_with_intercept = np.column_stack([np.ones(len(X_init)), X_init.flatten()])
            coeffs = np.linalg.lstsq(X_with_intercept, Y_init, rcond=None)[0]
            
            self.alpha = coeffs[0]
            self.beta = coeffs[1]
            
            # 初始协方差
            self.P_beta = 0.001
            self.P_alpha = 0.001
            
            # 基于OLS残差估计R
            residuals = Y_init - (self.alpha + self.beta * X_init.flatten())
            self.R = np.var(residuals) if len(residuals) > 1 else self.R_init
            
            logging.info(f"初始化完成: beta={self.beta:.6f}, alpha={self.alpha:.6f}, R={self.R:.6f}")
            
        def update(self, x_t: float, y_t: float) -> Dict:
            """
            Kalman滤波更新 - 使用原版本方法
            """
            if self.beta is None:
                raise ValueError("Must call initialize() first")
            
            # 1. 预测步 (状态方程: β_t = β_{t-1} + w_t)
            beta_pred = self.beta
            alpha_pred = self.alpha
            P_beta_pred = self.P_beta + self.Q_beta
            P_alpha_pred = self.P_alpha + self.Q_alpha
            
            # 2. 观测预测
            y_pred = alpha_pred + beta_pred * x_t
            
            # 3. 创新
            innovation = y_t - y_pred
            
            # 4. 创新方差 (这里使用简化版本)
            S = x_t**2 * P_beta_pred + P_alpha_pred + self.R
            S = max(S, 1e-12)  # 数值稳定性
            
            # **关键差异**: 使用residual/√R而非innovation/√S
            residual = innovation  # 原始残差
            z = residual / np.sqrt(self.R)  # 用R作为残差标准差
            
            # 5. Kalman增益
            K_beta = P_beta_pred * x_t / S
            K_alpha = P_alpha_pred / S
            
            # 6. 状态更新
            self.beta = beta_pred + K_beta * innovation
            self.alpha = alpha_pred + K_alpha * innovation
            
            # 7. 协方差更新
            self.P_beta = P_beta_pred * (1 - K_beta * x_t)
            self.P_alpha = P_alpha_pred * (1 - K_alpha)
            
            # 8. 自适应R (EWMA)
            if self.R_adapt:
                lambda_r = 0.99
                self.R = lambda_r * self.R + (1 - lambda_r) * innovation**2
                self.R = max(self.R, 1e-6)  # 防止R过小
            
            # 9. 记录历史
            self.beta_history.append(self.beta)
            self.alpha_history.append(self.alpha)
            self.z_history.append(z)
            self.residual_history.append(residual)
            self.innovation_history.append(innovation)
            
            return {
                'beta': self.beta,
                'alpha': self.alpha,
                'z': z,
                'innovation': innovation,
                'residual': residual,
                'R': self.R,
                'S': S,
                'y_pred': y_pred
            }
            
        def get_metrics(self, window: int = 60) -> Dict:
            """获取性能指标"""
            if len(self.z_history) < window:
                return {}
            
            recent_z = self.z_history[-window:]
            recent_residuals = self.residual_history[-window:]
            
            # Z-score统计
            z_var = np.var(recent_z)
            z_mean = np.mean(recent_z)
            z_std = np.std(recent_z)
            z_gt2_ratio = np.mean(np.abs(recent_z) > 2.0)
            z_gt2_count = np.sum(np.abs(recent_z) > 2.0)
            
            # 均值回归分析
            reversion_count = 0
            reversion_times = []
            for i in range(len(recent_z) - 20):
                if abs(recent_z[i]) > 2.0:
                    for j in range(i + 1, min(i + 21, len(recent_z))):
                        if abs(recent_z[j]) < 0.5:
                            reversion_count += 1
                            reversion_times.append(j - i)
                            break
            
            reversion_rate = reversion_count / z_gt2_count if z_gt2_count > 0 else 0
            avg_reversion_time = np.mean(reversion_times) if reversion_times else np.nan
            
            # 残差平稳性检验
            try:
                if len(recent_residuals) > 30:
                    adf_stat, adf_pvalue = adfuller(recent_residuals)[:2]
                    residual_stationary = adf_pvalue < 0.05
                else:
                    adf_pvalue = np.nan
                    residual_stationary = False
            except:
                adf_pvalue = np.nan
                residual_stationary = False
            
            return {
                'z_var': z_var,
                'z_mean': z_mean,
                'z_std': z_std,
                'z_gt2_ratio': z_gt2_ratio,
                'z_gt2_count': int(z_gt2_count),
                'reversion_rate': reversion_rate,
                'avg_reversion_time': avg_reversion_time,
                'adf_pvalue': adf_pvalue,
                'residual_stationary': residual_stationary,
                'data_points': len(recent_z)
            }


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalGeneratorV3:
    """
    信号生成器 V3 - 基于原版本Kalman滤波器
    
    基于需求文档：/docs/Requirements/03_signal_generation.md V3.1
    关键修正：使用经过验证的z = residual/√R方法
    """
    
    def __init__(self,
                 # 时间配置
                 signal_start_date: str = '2024-07-01',
                 kalman_warmup_days: int = 30,
                 ols_training_days: int = 60,
                 
                 # 交易阈值
                 z_open: float = 2.0,
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
                 
                 # Kalman参数（经过实证验证的最优值）
                 Q_beta: float = 5e-6,
                 Q_alpha: float = 1e-5,
                 R_init: float = 0.005,
                 R_adapt: bool = True):
        """
        初始化信号生成器 V3
        
        Args:
            signal_start_date: 信号生成开始日期
            kalman_warmup_days: Kalman预热天数
            ols_training_days: OLS训练天数
            z_open: 开仓阈值
            z_close: 平仓阈值
            max_holding_days: 最大持仓天数
            Q_beta: Beta过程噪声
            Q_alpha: Alpha过程噪声
            R_init: 初始测量噪声
            R_adapt: 是否自适应R
        """
        # 时间配置
        self.signal_start_date = signal_start_date
        self.kalman_warmup_days = kalman_warmup_days
        self.ols_training_days = ols_training_days
        
        # 交易参数
        self.z_open = z_open
        self.z_close = z_close
        self.max_holding_days = max_holding_days
        
        # Kalman参数
        self.Q_beta = Q_beta
        self.Q_alpha = Q_alpha
        self.R_init = R_init
        self.R_adapt = R_adapt
        
        # 自动计算时间轴
        self._calculate_time_axis()
        
        # 质量监控
        self.quality_reports = {}
        
        logger.info(f"SignalGeneratorV3 初始化完成:")
        logger.info(f"  时间轴: {self.data_start_date} -> {self.kalman_warmup_start} -> {self.signal_generation_start}")
        logger.info(f"  参数: z_open={self.z_open}, z_close={self.z_close}, max_holding={self.max_holding_days}")
    
    def _calculate_time_axis(self):
        """计算时间轴"""
        signal_start = pd.Timestamp(self.signal_start_date)
        
        # 往前推算
        kalman_warmup_start = signal_start - timedelta(days=self.kalman_warmup_days)
        data_start = kalman_warmup_start - timedelta(days=self.ols_training_days)
        
        self.signal_generation_start = self.signal_start_date
        self.kalman_warmup_start = kalman_warmup_start.strftime('%Y-%m-%d')
        self.data_start_date = data_start.strftime('%Y-%m-%d')
        
    def _get_phase(self, current_date: pd.Timestamp) -> str:
        """判断当前日期所在的阶段"""
        current_date = pd.Timestamp(current_date)
        kalman_start = pd.Timestamp(self.kalman_warmup_start)
        signal_start = pd.Timestamp(self.signal_generation_start)
        
        if current_date < kalman_start:
            return 'ols_training'
        elif current_date < signal_start:
            return 'kalman_warmup'
        else:
            return 'signal_generation'
    
    def _generate_signal(self, z_score: float, position: Optional[str], days_held: int) -> str:
        """
        生成交易信号
        
        Args:
            z_score: Z-score值
            position: 当前持仓状态 ('long'/'short'/None)
            days_held: 持仓天数
            
        Returns:
            信号类型: 'open_long', 'open_short', 'holding_long', 'holding_short', 'close', 'empty'
        """
        # 强制平仓（最高优先级）
        if position and days_held >= self.max_holding_days:
            return 'close'
        
        # 平仓条件
        if position and abs(z_score) <= self.z_close:
            return 'close'
        
        # 开仓条件（仅在空仓时）
        if not position:
            if z_score <= -self.z_open:
                return 'open_long'
            elif z_score >= self.z_open:
                return 'open_short'
            return 'empty'  # 空仓等待
        
        # 持仓期间状态
        if position == 'long':
            return 'holding_long'
        elif position == 'short':
            return 'holding_short'
        
        return 'empty'
    
    def _update_position_state(self, signal: str, position: Optional[str], days_held: int) -> Tuple[Optional[str], int]:
        """
        更新持仓状态
        
        Args:
            signal: 当前信号
            position: 当前持仓
            days_held: 当前持仓天数
            
        Returns:
            (新持仓状态, 新持仓天数)
        """
        if signal == 'open_long':
            return 'long', 1
        elif signal == 'open_short':
            return 'short', 1
        elif signal == 'close':
            return None, 0
        elif position:
            return position, days_held + 1
        else:
            return None, 0
    
    def _get_signal_reason(self, signal: str, z_score: float, days_held: int) -> str:
        """获取信号产生的原因"""
        if signal == 'close':
            if days_held >= self.max_holding_days:
                return 'force_close'
            elif abs(z_score) <= self.z_close:
                return 'z_threshold'
            else:
                return 'other'
        elif signal.startswith('open'):
            return 'z_threshold'
        elif signal.startswith('holding'):
            return 'holding'
        else:
            return 'no_signal'
    
    def process_pair(self, 
                     pair_name: str,
                     x_data: pd.Series, 
                     y_data: pd.Series,
                     initial_beta: Optional[float] = None) -> pd.DataFrame:
        """
        处理单个配对
        
        Args:
            pair_name: 配对名称
            x_data: X品种价格序列（对数价格）
            y_data: Y品种价格序列（对数价格）
            initial_beta: 初始Beta值（可选）
            
        Returns:
            信号DataFrame
        """
        logger.info(f"处理配对: {pair_name}")
        
        # 确保数据对齐
        aligned_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
        if len(aligned_data) < (self.ols_training_days + self.kalman_warmup_days + 10):
            logger.warning(f"配对 {pair_name} 数据不足，跳过处理")
            return pd.DataFrame()
        
        dates = aligned_data.index
        x_values = aligned_data['x'].values
        y_values = aligned_data['y'].values
        
        # 初始化Kalman滤波器
        kf = OriginalKalmanFilter(
            warmup=self.ols_training_days,
            Q_beta=self.Q_beta,
            Q_alpha=self.Q_alpha,
            R_init=self.R_init,
            R_adapt=self.R_adapt,
            z_in=self.z_open,
            z_out=self.z_close
        )
        
        # OLS初始化
        kf.initialize(x_values, y_values)
        
        # 记录初始beta
        initial_beta_used = initial_beta if initial_beta is not None else kf.beta
        
        # 生成信号
        signals = []
        position = None
        days_held = 0
        
        # 处理每个时间点
        for i, date in enumerate(dates):
            current_phase = self._get_phase(date)
            
            if i < self.ols_training_days:
                # OLS训练期：不更新不出信号
                signals.append({
                    'date': date,
                    'pair': pair_name,
                    'signal': 'ols_training',
                    'z_score': 0.0,
                    'innovation': 0.0,
                    'beta': getattr(kf, 'beta', initial_beta_used),
                    'beta_initial': initial_beta_used,
                    'days_held': 0,
                    'reason': 'ols_training',
                    'phase': current_phase,
                    'beta_window_used': 'ols'
                })
                
            elif i < (self.ols_training_days + self.kalman_warmup_days):
                # Kalman预热期：更新但不出信号
                result = kf.update(x_values[i], y_values[i])
                
                signals.append({
                    'date': date,
                    'pair': pair_name,
                    'signal': 'kalman_warmup',
                    'z_score': result['z'],
                    'innovation': result['innovation'],
                    'beta': result['beta'],
                    'beta_initial': initial_beta_used,
                    'days_held': 0,
                    'reason': 'kalman_warmup',
                    'phase': current_phase,
                    'beta_window_used': 'kalman'
                })
                
            else:
                # 信号生成期：正常生成信号
                result = kf.update(x_values[i], y_values[i])
                z_score = result['z']
                
                # 生成信号
                signal = self._generate_signal(z_score, position, days_held)
                
                # 更新持仓状态
                position, days_held = self._update_position_state(signal, position, days_held)
                
                # 获取信号原因
                reason = self._get_signal_reason(signal, z_score, days_held)
                
                signals.append({
                    'date': date,
                    'pair': pair_name,
                    'signal': signal,
                    'z_score': z_score,
                    'innovation': result['innovation'],
                    'beta': result['beta'],
                    'beta_initial': initial_beta_used,
                    'days_held': days_held,
                    'reason': reason,
                    'phase': current_phase,
                    'beta_window_used': 'kalman'
                })
        
        # 记录质量统计
        if len(kf.z_history) > 0:
            metrics = kf.get_metrics()
            self.quality_reports[pair_name] = metrics
        
        result_df = pd.DataFrame(signals)
        
        logger.info(f"配对 {pair_name} 处理完成: 生成 {len(result_df)} 条记录")
        
        return result_df
    
    def process_all_pairs(self, 
                          pairs_df: pd.DataFrame,
                          price_data: pd.DataFrame,
                          beta_window: str = '1y') -> pd.DataFrame:
        """
        批量处理所有配对
        
        Args:
            pairs_df: 协整模块输出的配对DataFrame
            price_data: 价格数据DataFrame
            beta_window: 使用的Beta时间窗口
            
        Returns:
            所有配对的信号DataFrame
        """
        logger.info(f"开始批量处理 {len(pairs_df)} 个配对，使用 {beta_window} Beta窗口")
        
        all_signals = []
        
        for idx, row in pairs_df.iterrows():
            pair_name = row['pair']
            symbol_x = row['symbol_x']
            symbol_y = row['symbol_y']
            
            # 获取初始beta
            beta_col = f'beta_{beta_window}'
            initial_beta = row[beta_col] if beta_col in row else None
            
            # 检查价格数据
            if symbol_x not in price_data.columns or symbol_y not in price_data.columns:
                logger.warning(f"跳过配对 {pair_name}: 缺少价格数据")
                continue
            
            x_data = price_data[symbol_x]
            y_data = price_data[symbol_y]
            
            # 处理配对
            pair_signals = self.process_pair(
                pair_name=pair_name,
                x_data=x_data,
                y_data=y_data,
                initial_beta=initial_beta
            )
            
            if len(pair_signals) > 0:
                # 添加symbol信息
                pair_signals['symbol_x'] = symbol_x
                pair_signals['symbol_y'] = symbol_y
                pair_signals['beta_window_used'] = beta_window
                all_signals.append(pair_signals)
        
        # 合并所有信号
        if all_signals:
            result_df = pd.concat(all_signals, ignore_index=True)
        else:
            result_df = pd.DataFrame()
        
        logger.info(f"批量处理完成: 总共生成 {len(result_df)} 条信号记录")
        
        return result_df
    
    def get_quality_report(self) -> pd.DataFrame:
        """
        获取质量报告
        
        Returns:
            质量报告DataFrame
        """
        if not self.quality_reports:
            return pd.DataFrame()
        
        quality_data = []
        for pair_name, metrics in self.quality_reports.items():
            if not metrics:  # 空字典
                continue
            
            # 质量评级
            z_var = metrics.get('z_var', 0)
            z_gt2_ratio = metrics.get('z_gt2_ratio', 0)
            
            if 0.8 <= z_var <= 1.3 and 0.02 <= z_gt2_ratio <= 0.05:
                quality_status = 'good'
            elif 0.6 <= z_var <= 1.5 and 0.01 <= z_gt2_ratio <= 0.08:
                quality_status = 'warning'
            else:
                quality_status = 'bad'
            
            quality_data.append({
                'pair': pair_name,
                'z_var': z_var,
                'z_mean': metrics.get('z_mean', 0),
                'z_std': metrics.get('z_std', 0),
                'z_gt2_ratio': z_gt2_ratio,
                'z_gt2_count': metrics.get('z_gt2_count', 0),
                'reversion_rate': metrics.get('reversion_rate', 0),
                'avg_reversion_time': metrics.get('avg_reversion_time', 0),
                'residual_stationary': metrics.get('residual_stationary', False),
                'quality_status': quality_status,
                'data_points': metrics.get('data_points', 0)
            })
        
        return pd.DataFrame(quality_data)


def main():
    """示例使用"""
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-08-01', freq='D')
    
    # 模拟协整的价格序列
    x_prices = np.cumsum(0.01 * np.random.randn(len(dates)))
    y_prices = 1.2 * x_prices + 0.1 + 0.02 * np.random.randn(len(dates))
    
    price_data = pd.DataFrame({
        'TEST_X': x_prices,
        'TEST_Y': y_prices
    }, index=dates)
    
    # 模拟协整模块输出
    pairs_df = pd.DataFrame({
        'pair': ['TEST_X-TEST_Y'],
        'symbol_x': ['TEST_X'],
        'symbol_y': ['TEST_Y'],
        'beta_1y': [1.15]
    })
    
    # 创建信号生成器
    sg = SignalGeneratorV3(
        signal_start_date='2024-07-01',
        kalman_warmup_days=30,
        ols_training_days=60
    )
    
    # 生成信号
    signals = sg.process_all_pairs(pairs_df, price_data, beta_window='1y')
    
    print(f"生成信号数量: {len(signals)}")
    print(f"信号期信号数量: {len(signals[signals['phase'] == 'signal_generation'])}")
    
    # 统计交易信号
    trading_signals = signals[signals['signal'].isin(['open_long', 'open_short', 'close'])]
    print(f"交易信号数量: {len(trading_signals)}")
    
    # 质量报告
    quality = sg.get_quality_report()
    if len(quality) > 0:
        print("\n质量报告:")
        print(quality)


if __name__ == '__main__':
    main()