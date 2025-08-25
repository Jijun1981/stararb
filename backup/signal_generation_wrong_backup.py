#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块 V3.0 - 原始状态空间Kalman滤波实现
严格按照需求文档 docs/Requirements/03_signal_generation.md V3.0 实现
使用二维状态空间模型 [β, α] 和固定的最优参数
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression

# 设置日志
logger = logging.getLogger(__name__)


class OriginalKalmanFilter:
    """
    原始状态空间Kalman滤波器 - REQ-3.1
    使用二维状态向量 [β, α] 和固定的过程噪声协方差
    """
    
    def __init__(self, 
                 warmup: int = 60,
                 Q_beta: float = 5e-6,      # Beta过程噪声（实证最优）
                 Q_alpha: float = 1e-5,     # Alpha过程噪声（实证最优）
                 R_init: float = 0.005,     # 初始测量噪声（实证最优）
                 R_adapt: bool = True,      # 是否自适应R
                 z_in: float = 2.0,         # 开仓阈值
                 z_out: float = 0.5):       # 平仓阈值
        """
        初始化原始状态空间Kalman滤波器
        
        Args:
            warmup: 预热期长度
            Q_beta: Beta过程噪声
            Q_alpha: Alpha过程噪声  
            R_init: 初始测量噪声
            R_adapt: 是否自适应调整R
            z_in: 开仓阈值
            z_out: 平仓阈值
        """
        self.warmup = warmup
        self.Q = np.diag([Q_beta, Q_alpha])  # 过程噪声协方差矩阵
        self.R = R_init                       # 测量噪声
        self.R_init = R_init
        self.R_adapt = R_adapt
        self.z_in = z_in
        self.z_out = z_out
        
        # 状态变量 [beta, alpha]'
        self.state = None
        self.P = None  # 状态协方差矩阵
        
        # 历史记录
        self.beta_history = []
        self.alpha_history = []
        self.z_history = []
        self.innovation_history = []
        
    def initialize(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        使用60天OLS初始化状态 - REQ-3.1.3
        
        Args:
            x_data: X价格序列（完整数据）
            y_data: Y价格序列（完整数据）
        """
        # 使用前warmup天数据做OLS
        X = x_data[:self.warmup].reshape(-1, 1)
        Y = y_data[:self.warmup]
        
        # OLS回归 y = β*x + α
        model = LinearRegression()
        model.fit(X, Y)
        
        # 初始状态 [β, α]'
        self.state = np.array([model.coef_[0], model.intercept_])
        
        # 初始协方差矩阵
        self.P = np.eye(2) * 0.001
        
        # 初始R基于残差方差
        residuals = Y - model.predict(X)
        self.R = np.var(residuals)
        
        logger.debug(f"Kalman初始化: β={self.state[0]:.4f}, α={self.state[1]:.4f}, R={self.R:.6f}")
        
    def update(self, x_t: float, y_t: float) -> None:
        """
        状态空间Kalman更新 - REQ-3.1.1, 3.1.2
        
        Args:
            x_t: 当前X值
            y_t: 当前Y值
        """
        if self.state is None or self.P is None:
            raise ValueError("必须先调用initialize初始化")
        
        # 1. 预测步
        # 状态预测: x_t|t-1 = x_t-1|t-1
        state_pred = self.state
        
        # 协方差预测: P_t|t-1 = P_t-1|t-1 + Q
        P_pred = self.P + self.Q
        
        # 2. 观测预测
        # H = [x_t, 1] 观测矩阵
        H = np.array([x_t, 1.0])
        
        # 预测观测: y_pred = β*x + α
        y_pred = H @ state_pred
        
        # 3. 创新
        v = y_t - y_pred  # 创新
        S = H @ P_pred @ H.T + self.R  # 创新方差
        S = max(S, 1e-12)  # 数值稳定性
        
        # 4. 标准化创新（使用原版本方法：residual/√R）
        z = v / np.sqrt(self.R)  # 修复：使用R而非S进行标准化
        
        # 5. 更新步
        # Kalman增益
        K = P_pred @ H.T / S
        
        # 状态更新
        self.state = state_pred + K * v
        
        # 协方差更新
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        # 6. 自适应R（可选）
        if self.R_adapt:
            lambda_r = 0.99  # EWMA参数
            self.R = lambda_r * self.R + (1 - lambda_r) * v**2
        
        # 7. 记录历史
        self.beta_history.append(self.state[0])
        self.alpha_history.append(self.state[1])
        self.z_history.append(z)
        self.innovation_history.append(v)
        
    def get_state(self) -> Dict:
        """
        获取当前状态
        
        Returns:
            当前状态字典
        """
        if self.state is None:
            return {
                'beta': None,
                'alpha': None,
                'R': self.R,
                'initialized': False
            }
        
        return {
            'beta': self.state[0],
            'alpha': self.state[1],
            'R': self.R,
            'P': self.P,
            'initialized': True
        }
        
    def get_quality_metrics(self, window: int = 60) -> Dict:
        """
        获取质量指标 - REQ-3.2, REQ-3.5
        
        Args:
            window: 计算指标的窗口大小
            
        Returns:
            质量指标字典
        """
        if len(self.z_history) < window:
            return {
                'z_variance': np.nan,
                'z_mean': np.nan,
                'z_std': np.nan,
                'z_gt2_ratio': np.nan,
                'signal_ratio': np.nan,
                'quality_status': 'insufficient_data',
                'current_R': self.R,
                'current_beta': self.state[0] if self.state is not None else None,
                'current_alpha': self.state[1] if self.state is not None else None
            }
        
        recent_z = np.array(self.z_history[-window:])
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        z_std = np.std(recent_z)
        z_gt2_ratio = np.sum(np.abs(recent_z) > 2.0) / len(recent_z)
        
        # 质量评级 - REQ-3.2.1
        if 0.8 <= z_var <= 1.3 and 0.02 <= z_gt2_ratio <= 0.05:
            quality_status = 'good'
        elif 0.6 <= z_var <= 1.5 and 0.01 <= z_gt2_ratio <= 0.08:
            quality_status = 'warning'
        else:
            quality_status = 'bad'
        
        return {
            'z_variance': z_var,
            'z_mean': z_mean,
            'z_std': z_std,
            'z_gt2_ratio': z_gt2_ratio,
            'signal_ratio': z_gt2_ratio,  # 与z_gt2_ratio相同
            'quality_status': quality_status,
            'current_R': self.R,
            'current_beta': self.state[0] if self.state is not None else None,
            'current_alpha': self.state[1] if self.state is not None else None
        }


class SignalGenerator:
    """
    信号生成器 - 使用原始状态空间Kalman滤波
    """
    
    def __init__(self,
                 # 时间配置（新增 - REQ-3.0）
                 signal_start_date: str,                    # 信号生成开始日期
                 kalman_warmup_days: int = 30,              # Kalman预热天数
                 ols_training_days: int = 60,               # OLS训练天数
                 
                 # 交易阈值
                 z_open: float = 2.0,
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
                 
                 # Kalman参数（实证最优）
                 Q_beta: float = 5e-6,
                 Q_alpha: float = 1e-5,
                 R_init: float = 0.005,
                 R_adapt: bool = True):
        """
        初始化信号生成器 V3.1 - 支持灵活时间轴配置
        
        Args:
            signal_start_date: 信号生成开始日期 (如'2024-07-01')
            kalman_warmup_days: Kalman预热天数（从信号期往前推）
            ols_training_days: OLS训练天数（从Kalman预热期往前推）
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
        
        # 验证参数
        if kalman_warmup_days < 0 or ols_training_days < 0:
            raise ValueError("预热天数不能为负数")
        
        # 验证日期格式
        try:
            from datetime import datetime
            datetime.strptime(signal_start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"信号开始日期格式错误: {signal_start_date}，应为YYYY-MM-DD格式")
        
        # 交易参数
        self.z_open = z_open
        self.z_close = z_close
        self.max_holding_days = max_holding_days
        
        # Kalman参数
        self.Q_beta = Q_beta
        self.Q_alpha = Q_alpha
        self.R_init = R_init
        self.R_adapt = R_adapt
        
        # 为了向后兼容，保留warmup属性
        self.warmup = ols_training_days
        
        # 存储每个配对的Kalman滤波器
        self.kalman_filters = {}
    
    def _calculate_data_start_date(self) -> str:
        """
        计算数据开始日期 - REQ-3.0.4
        
        Returns:
            数据开始日期字符串 (YYYY-MM-DD格式)
        """
        from datetime import datetime, timedelta
        
        signal_start = datetime.strptime(self.signal_start_date, '%Y-%m-%d')
        data_start = signal_start - timedelta(days=self.kalman_warmup_days + self.ols_training_days)
        
        return data_start.strftime('%Y-%m-%d')
    
    def _get_phase_boundaries(self) -> Dict[str, datetime]:
        """
        计算各阶段的时间边界 - REQ-3.0.5
        
        Returns:
            包含各阶段开始时间的字典
        """
        from datetime import datetime, timedelta
        
        signal_start = datetime.strptime(self.signal_start_date, '%Y-%m-%d')
        kalman_start = signal_start - timedelta(days=self.kalman_warmup_days)
        ols_start = kalman_start - timedelta(days=self.ols_training_days)
        
        return {
            'ols_start': ols_start,
            'kalman_start': kalman_start,
            'signal_start': signal_start
        }
    
    def _determine_phase(self, current_date: datetime, boundaries: Dict[str, datetime]) -> str:
        """
        根据当前日期确定所处阶段
        
        Args:
            current_date: 当前日期
            boundaries: 阶段边界字典
            
        Returns:
            阶段名称: 'ols_training', 'kalman_warmup', 'signal_generation'
        """
        if current_date < boundaries['kalman_start']:
            return 'ols_training'
        elif current_date < boundaries['signal_start']:
            return 'kalman_warmup'
        else:
            return 'signal_generation'
        
    def _generate_signal(self, 
                        z_score: float,
                        position: Optional[str],
                        days_held: int) -> str:
        """
        生成交易信号 - REQ-3.3
        
        Args:
            z_score: 标准化创新
            position: 当前持仓 ('long'/'short'/None)
            days_held: 持仓天数
            
        Returns:
            交易信号
        """
        # 强制平仓（最高优先级）
        if position and days_held >= self.max_holding_days:
            return 'close'
        
        # 平仓条件
        if position and abs(z_score) < self.z_close:
            return 'close'
        
        # 开仓条件（仅在空仓时）
        if not position:
            if z_score < -self.z_open:
                return 'open_long'
            elif z_score > self.z_open:
                return 'open_short'
            return 'empty'  # 空仓等待
        
        # 持仓状态
        if position == 'long':
            return 'holding_long'
        elif position == 'short':
            return 'holding_short'
        
        return 'empty'
        
    def process_pair(self,
                    pair_name: str,
                    x_data: pd.Series,
                    y_data: pd.Series,
                    initial_beta: Optional[float] = None) -> pd.DataFrame:
        """
        处理单个配对的信号生成
        
        Args:
            pair_name: 配对名称
            x_data: X价格序列
            y_data: Y价格序列
            initial_beta: 初始beta（可选）
            
        Returns:
            信号DataFrame
        """
        # 确保数据对齐
        common_index = x_data.index.intersection(y_data.index)
        x_data = x_data.loc[common_index]
        y_data = y_data.loc[common_index]
        
        # 转换为numpy数组
        x_values = x_data.values
        y_values = y_data.values
        dates = x_data.index
        
        # 创建Kalman滤波器
        kf = OriginalKalmanFilter(
            warmup=self.ols_training_days,  # 使用OLS训练天数
            Q_beta=self.Q_beta,
            Q_alpha=self.Q_alpha,
            R_init=self.R_init,
            R_adapt=self.R_adapt,
            z_in=self.z_open,
            z_out=self.z_close
        )
        
        # 初始化（只使用OLS训练期的数据）
        kf.initialize(x_values, y_values)
        
        # 存储滤波器实例
        self.kalman_filters[pair_name] = kf
        
        # 获取时间边界
        boundaries = self._get_phase_boundaries()
        
        # 生成信号
        signals = []
        position = None
        days_held = 0
        
        for i in range(len(x_values)):
            current_date = dates[i]
            if hasattr(current_date, 'to_pydatetime'):
                current_date = current_date.to_pydatetime()
            
            # 确定当前阶段
            phase = self._determine_phase(current_date, boundaries)
            
            if phase == 'ols_training':
                # OLS训练期：只做初始化，不出信号
                signal = 'warm_up'
                z_score = 0.0
                beta = kf.state[0] if kf.state is not None else initial_beta
                innovation = 0.0
                reason = 'ols_training'
                
            elif phase == 'kalman_warmup':
                # Kalman预热期：更新状态但不出信号
                kf.update(x_values[i], y_values[i])
                signal = 'warm_up'
                z_score = kf.z_history[-1] if kf.z_history else 0.0
                beta = kf.state[0]
                innovation = kf.innovation_history[-1] if kf.innovation_history else 0.0
                reason = 'kalman_warmup'
                
            else:
                # 信号生成期：正常生成交易信号
                kf.update(x_values[i], y_values[i])
                z_score = kf.z_history[-1]
                beta = kf.state[0]
                innovation = kf.innovation_history[-1]
                
                # 生成交易信号
                signal = self._generate_signal(z_score, position, days_held)
                reason = 'signal_generation'
                
                # 更新持仓状态
                if signal.startswith('open'):
                    position = 'long' if signal == 'open_long' else 'short'
                    days_held = 1
                    reason = 'z_threshold'
                elif signal == 'close':
                    if days_held >= self.max_holding_days:
                        reason = 'force_close'
                    else:
                        reason = 'z_converge'
                    position = None
                    days_held = 0
                elif position:
                    days_held += 1
                    reason = 'holding'
                else:
                    reason = 'waiting'
            
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': signal,
                'z_score': z_score,
                'beta': beta,
                'alpha': kf.state[1] if kf.state is not None else 0.0,
                'innovation': innovation,
                'days_held': days_held,
                'phase': phase,
                'reason': reason,
                'R': kf.R
            })
        
        return pd.DataFrame(signals)
    
    def process_all_pairs(self,
                         pairs_df: pd.DataFrame,
                         price_data: pd.DataFrame,
                         beta_window: str = '1y') -> pd.DataFrame:
        """
        批量处理多个配对
        
        Args:
            pairs_df: 配对DataFrame（来自协整模块）
            price_data: 价格数据DataFrame
            beta_window: 使用的beta时间窗口
            
        Returns:
            所有配对的信号DataFrame
        """
        all_signals = []
        
        for idx, row in pairs_df.iterrows():
            pair_name = row['pair']
            symbol_x = row['symbol_x']
            symbol_y = row['symbol_y']
            
            # 获取初始beta（如果有）
            beta_col = f'beta_{beta_window}'
            initial_beta = row[beta_col] if beta_col in row else None
            
            # 获取价格数据
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
            
            # 添加beta来源信息
            pair_signals['beta_window_used'] = beta_window
            
            all_signals.append(pair_signals)
        
        # 合并所有信号
        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_quality_report(self) -> pd.DataFrame:
        """
        获取所有配对的质量报告
        
        Returns:
            质量报告DataFrame
        """
        reports = []
        
        for pair_name, kf in self.kalman_filters.items():
            metrics = kf.get_quality_metrics()
            metrics['pair'] = pair_name
            reports.append(metrics)
        
        if reports:
            return pd.DataFrame(reports).set_index('pair')
        else:
            return pd.DataFrame()


# 向后兼容的别名
AdaptiveKalmanFilter = OriginalKalmanFilter
AdaptiveSignalGenerator = SignalGenerator