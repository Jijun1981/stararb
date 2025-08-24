#!/usr/bin/env python3
"""
信号生成模块 - REQ-3.x.x (重构版)
使用一维Kalman滤波和三阶段处理生成交易信号

主要功能:
1. 一维Kalman滤波动态估计β 
2. 三阶段处理：初始化、收敛、信号生成
3. 残差Z-score信号生成
4. 自适应R更新(EWMA)

作者: Star-arb Team  
日期: 2025-08-21
版本: V2.1 (重构)
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_ols_beta(y_data: np.ndarray, x_data: np.ndarray, window: int = 60) -> float:
    """
    计算60天滚动窗口OLS beta作为Kalman滤波验证基准
    
    Args:
        y_data: Y价格序列
        x_data: X价格序列  
        window: 滚动窗口大小（默认60天）
    
    Returns:
        float: OLS回归系数
    """
    if len(y_data) < window or len(x_data) < window:
        return np.nan
    
    # 取最后window天的数据
    y_window = y_data[-window:]
    x_window = x_data[-window:]
    
    # OLS回归: y = alpha + beta * x
    X = np.column_stack([np.ones(len(x_window)), x_window])
    try:
        coeffs = np.linalg.lstsq(X, y_window, rcond=None)[0]
        return coeffs[1]  # beta系数
    except:
        return np.nan


class KalmanFilter2D:
    """
    二维Kalman滤波器 - 工程实践版
    按照专业建议：均衡档参数，轻量自适应，专注稳定性
    
    状态向量: [β_t, c_t] (hedge ratio + 截距)
    观测方程: y_t = β_t * x_t + c_t + v_t
    """
    
    def __init__(self, initial_beta: float, initial_c: float, ols_residuals: np.ndarray, x_data: np.ndarray):
        """
        初始化二维Kalman - 基于专家建议的保守设计
        
        Args:
            initial_beta: 初始β (来自OLS)
            initial_c: 初始截距 (来自OLS)  
            ols_residuals: OLS残差用于R0计算
            x_data: X数据用于方差计算
        """
        # 状态向量 [beta, c]
        self.beta = float(initial_beta)
        self.c = float(initial_c)
        
        # 计算MAD作为稳健估计
        residual_mad = np.median(np.abs(ols_residuals - np.median(ols_residuals)))
        x_mad = np.median(np.abs(x_data - np.median(x_data)))
        
        # 观测噪声R: 使用MAD的稳健估计
        self.R = (residual_mad * 1.4826) ** 2  # MAD转标准差再平方
        self.R = max(self.R, 1e-6)
        
        # 过程噪声Q: 保守设计，避免过度跳跃
        var_x = (x_mad * 1.4826) ** 2
        var_x = max(var_x, 1e-10)
        
        # 专家建议的保守参数: 让R/Q比例更合理
        eta_beta = 1e-4  # 大幅减小，让beta变化更平滑
        eta_c = 1e-5     # 对应减小c的变化率
        
        q_beta = eta_beta * self.R / var_x
        q_c = eta_c * self.R
        self.Q = np.diag([q_beta, q_c])
        
        # 初始不确定性: 适度设定
        p_beta = q_beta * 10  # β的初始不确定性
        p_c = q_c * 10        # c的初始不确定性  
        self.P = np.diag([p_beta, p_c])
        
        # 历史记录
        self.beta_history = [self.beta]
        self.innovation_z = []
        
        logger.info(f"KF2D初始化(保守): β={self.beta:.6f}, c={self.c:.2f}")
        logger.info(f"  R={self.R:.2e}, Q_β={q_beta:.2e}, Q_c={q_c:.2e}")
    
    def update(self, y_t: float, x_t: float):
        """Kalman更新 - 保守稳定版本"""
        # 观测矩阵 H = [x_t, 1]
        H = np.array([[x_t, 1.0]])
        
        # 1. 预测步
        P_pred = self.P + self.Q
        state = np.array([self.beta, self.c])
        y_pred = float(H @ state)
        
        # 2. 创新和创新协方差
        v = y_t - y_pred  # 创新 (innovation)
        S = float(H @ P_pred @ H.T + self.R)
        S = max(S, 1e-12)  # 数值稳定性护栏
        
        # 3. Kalman增益
        K = (P_pred @ H.T) / S
        
        # 4. 状态更新
        update_vec = (K * v).ravel()
        self.beta += update_vec[0]
        self.c += update_vec[1]
        
        # 5. 协方差更新 (Joseph形式更稳定)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ np.array([[self.R]]) @ K.T
        
        # 6. 诊断记录
        z_score = v / np.sqrt(S)  # 标准化创新
        self.innovation_z.append(z_score)
        self.beta_history.append(self.beta)
        
        return self.beta
    
    def get_residual(self, y_t: float, x_t: float) -> float:
        """计算残差用于Z-score"""
        return y_t - (self.beta * x_t + self.c)
    
    def diagnostics(self, window: int = 250):
        """创新白化诊断"""
        if len(self.innovation_z) < window:
            return None
        z_recent = np.array(self.innovation_z[-window:])
        return {
            'z_mean': np.mean(z_recent),
            'z_std': np.std(z_recent), 
            'whitened': abs(np.mean(z_recent)) <= 0.1 and 0.9 <= np.std(z_recent) <= 1.1
        }


class KalmanFilter1D:
    """
    一维Kalman滤波器 - REQ-3.1.x
    
    状态方程: β_t = β_{t-1} + w_t, w_t ~ N(0, Q)
    观测方程: y_t = β_t * x_t + v_t, v_t ~ N(0, R)
    """
    
    def __init__(self, initial_beta: float):
        """
        初始化一维Kalman滤波器 - 所有参数写死，不可配置
        
        Args:
            initial_beta: 初始β值（来自协整模块）
            
        固定参数（不可配置）：
            Q = 1e-4 (过程噪声方差)
            R = 1.0 (观测噪声方差，初始值)
            P0 = 0.1 (初始不确定性)
        """
        self.beta = float(initial_beta)
        # 所有参数写死，不可配置
        self.P = 0.01     # 固定初始不确定性（降低）
        self.Q = 1e-10    # 固定过程噪声（几乎为0）
        self.R = 1.0      # 固定观测噪声初始值（大观测噪声）
        self.beta_history = [self.beta]
        
        # 验证参数合理性
        self._validate_params()
        
        logger.debug(f"KF1D初始化: β₀={self.beta:.4f}, Q={self.Q:.2e}, R={self.R:.2e}, P₀={self.P:.3f}")
    
    def _validate_params(self):
        """验证Kalman滤波参数的合理性"""
        if self.Q <= 0:
            raise ValueError(f"过程噪声Q必须>0: {self.Q}")
        if self.R <= 0:
            raise ValueError(f"观测噪声R必须>0: {self.R}")
        if self.P <= 0:
            raise ValueError(f"初始不确定性P0必须>0: {self.P}")
        if abs(self.beta) > 10:
            warnings.warn(f"初始β值可能过大: {self.beta}")
    
    def update(self, y_t: float, x_t: float) -> Dict[str, float]:
        """
        一步Kalman滤波更新 - REQ-3.1.1, REQ-3.1.2
        
        Args:
            y_t: 观测值 log(Y_t)
            x_t: 解释变量 log(X_t)
            
        Returns:
            dict: 包含更新后β、残差等信息
        """
        # 输入验证
        if not np.isfinite(y_t) or not np.isfinite(x_t):
            raise ValueError(f"输入包含无效值: y_t={y_t}, x_t={x_t}")
        if abs(x_t) < 1e-10:
            raise ValueError(f"x_t过小可能导致数值不稳定: {x_t}")
        
        # 1. 预测步 (状态预测)
        beta_pred = self.beta  # 随机游走模型: β_t = β_{t-1}
        P_pred = self.P + self.Q
        
        # 2. 计算残差（观测-预测）
        y_pred = beta_pred * x_t
        residual = y_t - y_pred
        
        # 3. 创新协方差
        S = x_t * P_pred * x_t + self.R
        
        # 数值稳定性检查
        if S <= 1e-15:
            logger.warning(f"创新协方差过小: S={S}, 可能导致数值不稳定")
            S = max(S, 1e-15)
        
        # 4. Kalman增益
        K = P_pred * x_t / S
        
        # 5. 状态更新
        beta_new = beta_pred + K * residual
        
        # 6. β变化限制（日变化不超过5%） - REQ-3.1.7
        # 设置最小绝对变化阈值防止死螺旋
        min_abs_change = 0.001  # 最小允许变化0.001
        max_change = max(abs(self.beta) * 0.05, min_abs_change)
        if abs(beta_new - self.beta) > max_change:
            beta_new = self.beta + np.sign(beta_new - self.beta) * max_change
            logger.debug(f"β变化被限制: {self.beta:.6f} -> {beta_new:.6f}")
        
        self.beta = beta_new
        
        # 7. 协方差更新 (Joseph form for numerical stability)
        self.P = (1 - K * x_t) * P_pred
        
        # 8. 自适应观测噪声更新（EWMA） - REQ-3.1.11
        innovation_sq = residual * residual
        self.R = 0.98 * self.R + 0.02 * max(innovation_sq, 1e-6)
        
        # 9. 记录历史
        self.beta_history.append(self.beta)
        
        # 最终验证
        if not np.isfinite(self.beta) or not np.isfinite(self.P):
            raise RuntimeError(f"Kalman滤波数值不稳定: β={self.beta}, P={self.P}")
        
        return {
            'beta': float(self.beta),
            'residual': float(residual),
            'K': float(K),
            'P': float(self.P),
            'innovation_covariance': float(S)
        }
    
    def get_current_beta(self) -> float:
        """获取当前β值"""
        return float(self.beta)
    
    def get_convergence_metrics(self, days: int = 20) -> Dict[str, float]:
        """
        计算收敛性指标 - REQ-3.2.5
        
        Args:
            days: 用于计算收敛性的天数
            
        Returns:
            dict: 包含收敛性指标
        """
        if len(self.beta_history) < days + 1:
            return {'converged': False, 'max_change': np.nan, 'mean_change': np.nan}
        
        recent_betas = self.beta_history[-days-1:]
        changes = []
        
        for i in range(1, len(recent_betas)):
            prev_beta = recent_betas[i-1]
            curr_beta = recent_betas[i]
            if abs(prev_beta) > 1e-10:
                change = abs(curr_beta - prev_beta) / abs(prev_beta)
                changes.append(change)
        
        if not changes:
            return {'converged': False, 'max_change': np.nan, 'mean_change': np.nan}
        
        max_change = max(changes)
        mean_change = np.mean(changes)
        
        return {
            'converged': max_change < 0.01,  # 所有变化都<1%认为收敛
            'max_change': float(max_change),
            'mean_change': float(mean_change),
            'num_changes': len(changes)
        }


class SignalGenerator:
    """
    信号生成器 - REQ-3.2.x, REQ-3.3.x
    
    主要功能：
    1. 管理三阶段处理流程
    2. 计算残差Z-score
    3. 生成交易信号
    4. 批量处理多配对
    """
    
    def __init__(self, 
                 window: int = 60,
                 z_open: float = 2.0, 
                 z_close: float = 0.5,
                 convergence_days: int = 30,
                 convergence_threshold: float = 0.02,
                 max_holding_days: int = 30):
        """
        初始化信号生成器 - 按需求文档参数化
        
        Args:
            window: 滚动窗口大小（默认60）
            z_open: 开仓Z-score阈值（默认2.0）
            z_close: 平仓Z-score阈值（默认0.5）
            convergence_days: 收敛判定天数（默认30）
            convergence_threshold: 收敛阈值（默认2%）
            max_holding_days: 最大持仓天数（默认30）
            
        注：所有Kalman参数在KalmanFilter1D中写死，不在此处配置
        """
        self.window = window
        self.z_open = z_open
        self.z_close = z_close
        self.convergence_days = convergence_days
        self.convergence_threshold = convergence_threshold
        self.max_holding_days = max_holding_days
        
        logger.info(f"信号生成器初始化: window={window}, z_open={z_open}, z_close={z_close}, max_holding_days={max_holding_days}")
    
    def calculate_residual(self, y: float, x: float, beta: float) -> float:
        """
        计算残差 - REQ-3.2.1
        
        Args:
            y: 观测值 log(Y_t)
            x: 解释变量 log(X_t)
            beta: 当前β值
            
        Returns:
            residual: 残差值
        """
        residual = y - beta * x
        
        # 验证计算结果
        if not np.isfinite(residual):
            raise ValueError(f"残差计算结果无效: y={y}, x={x}, beta={beta}")
        
        return float(residual)
    
    def calculate_zscore(self, residuals: np.ndarray, window: int) -> float:
        """
        计算滚动Z-score - REQ-3.2.3
        
        根据用户要求修正：当前点作为60个滚动窗口的最后一个点
        - 使用包含当前点的window个点计算统计量
        - 使用ddof=0计算标准差
        
        Args:
            residuals: 残差序列
            window: 滚动窗口大小
            
        Returns:
            z_score: 当前残差的Z-score
        """
        if len(residuals) < window:  # 需要至少window个点
            return 0.0
        
        # 修正：使用包含当前点的window个点计算统计量
        window_data = residuals[-window:]       # 包含当前点的最后window个点
        current_residual = residuals[-1]       # 当前点的残差
        
        # 验证数据有效性
        valid_data = window_data[np.isfinite(window_data)]
        if len(valid_data) < window * 0.8:  # 至少80%有效数据
            logger.warning(f"滚动窗口中无效数据过多: {len(valid_data)}/{window}")
            return 0.0
        
        mean = np.mean(valid_data)
        std = np.std(valid_data, ddof=0)  # 修正：使用ddof=0与原始方法一致
        
        if std < 1e-10:
            logger.warning(f"滚动窗口标准差过小: {std}")
            return 0.0
        
        z_score = (current_residual - mean) / std
        
        # 验证Z-score合理性
        if not np.isfinite(z_score):
            logger.error(f"Z-score计算无效: residual={current_residual}, mean={mean}, std={std}")
            return 0.0
        
        if abs(z_score) > 10:
            logger.warning(f"Z-score异常大: {z_score}")
        
        return float(z_score)
    
    def generate_signal(self, 
                       z_score: float, 
                       position: Optional[str], 
                       days_held: int,
                       z_open: Optional[float] = None,
                       z_close: Optional[float] = None,
                       max_days: int = 30) -> str:
        """
        生成交易信号 - REQ-3.3.x
        使用清晰的状态机制：empty, holding_long, holding_short, open_*, close
        
        Args:
            z_score: 残差Z-score
            position: 当前持仓状态 ('long', 'short', None)
            days_held: 持仓天数
            z_open: 开仓阈值（可覆盖默认值）
            z_close: 平仓阈值（可覆盖默认值）
            max_days: 最大持仓天数
            
        Returns:
            signal: 信号类型 (empty, open_long, open_short, holding_long, holding_short, close)
        """
        z_open = z_open if z_open is not None else self.z_open
        z_close = z_close if z_close is not None else self.z_close
        
        # 输入验证
        if not np.isfinite(z_score):
            return 'empty' if not position else ('holding_long' if position == 'long' else 'holding_short')
        
        # 强制平仓 - REQ-3.3.4 (最高优先级)
        if position and days_held >= max_days:
            return 'close'
        
        # 平仓条件 - REQ-3.3.3: |z| <= z_close
        if position and abs(z_score) <= z_close:
            return 'close'
        
        # 开仓条件 - REQ-3.3.2: |z| >= z_open (防重复开仓 - REQ-3.3.6)
        if not position:
            abs_z = abs(z_score)
            
            if abs_z >= z_open:
                if z_score <= -z_open:  # 负Z-score开多
                    return 'open_long'
                elif z_score >= z_open:  # 正Z-score开空
                    return 'open_short'
            return 'empty'  # 空仓等待
        
        # 持仓期间状态 - REQ-3.3.8
        if position == 'long':
            return 'holding_long'
        elif position == 'short':
            return 'holding_short'
        
        return 'empty'
    
    def process_pair_signals(self,
                           pair_data: pd.DataFrame,
                           initial_beta: float,
                           convergence_end: str,
                           signal_start: str,
                           hist_start: Optional[str] = None,
                           hist_end: Optional[str] = None,
                           pair_info: Optional[Dict] = None,
                           beta_window: str = '1y') -> pd.DataFrame:
        """
        处理单个配对的信号生成 - REQ-3.1.8, REQ-3.1.9
        
        Args:
            pair_data: 包含x, y价格的DataFrame
            initial_beta: 初始β值
            convergence_end: 收敛期结束日期
            signal_start: 信号生成开始日期  
            hist_start: 历史数据开始日期（用于R估计）
            hist_end: 历史数据结束日期（用于R估计）
            
        Returns:
            signals_df: 信号DataFrame
        """
        # 数据验证
        required_cols = ['date', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in pair_data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")
        
        if len(pair_data) < self.window:
            logger.warning(f"数据量不足: {len(pair_data)} < {self.window}")
            return pd.DataFrame()
        
        # 按日期排序
        data = pair_data.sort_values('date').copy()
        
        # 初始化Kalman滤波器
        kf = KalmanFilter1D(initial_beta)
        
        # 使用历史数据估计R
        if hist_start and hist_end:
            hist_data = data[(data['date'] >= hist_start) & (data['date'] <= hist_end)]
            if len(hist_data) > 0:
                hist_residuals = hist_data['y'] - initial_beta * hist_data['x']
                kf.R = max(np.var(hist_residuals), 1e-6)
                logger.debug(f"从历史数据估计R: {kf.R:.6f}")
        
        # 存储结果
        results = []
        residuals = []
        position = None
        days_held = 0
        converged = False
        
        for i, row in data.iterrows():
            # Kalman滤波更新
            try:
                kf_result = kf.update(row['y'], row['x'])
                beta_t = kf_result['beta']
            except Exception as e:
                logger.error(f"Kalman滤波更新失败: {e}")
                continue
            
            # 计算残差
            residual = self.calculate_residual(row['y'], row['x'], beta_t)
            residuals.append(residual)
            
            # 计算OLS beta作为对比基准 (REQ-3.1.7a)
            current_idx = data.index.get_loc(i)
            if current_idx >= 60:  # 有足够数据计算60天OLS
                y_window = data['y'].iloc[current_idx-59:current_idx+1].values
                x_window = data['x'].iloc[current_idx-59:current_idx+1].values
                ols_beta = calculate_ols_beta(y_window, x_window, window=60)
            else:
                ols_beta = np.nan
            
            # 确定当前阶段
            current_date = pd.to_datetime(row['date']) if not isinstance(row['date'], pd.Timestamp) else row['date']
            conv_end_date = pd.to_datetime(convergence_end)
            signal_start_date = pd.to_datetime(signal_start)
            
            if current_date <= conv_end_date:
                phase = 'convergence_period'
                
                # 收敛性评估
                if len(residuals) >= self.convergence_days:
                    conv_metrics = kf.get_convergence_metrics(self.convergence_days)
                    if conv_metrics['converged']:
                        converged = True
                
                signal = 'converging'
                z_score = 0.0
                reason = 'converging'
                
            elif current_date >= signal_start_date:
                phase = 'signal_period'
                
                # 计算Z-score（需要足够历史数据）
                # 修正：需要至少window个点（包含当前点的滚动窗口）
                if len(residuals) >= self.window:
                    z_score = self.calculate_zscore(np.array(residuals), self.window)
                    
                    # 生成交易信号
                    signal = self.generate_signal(z_score, position, days_held, 
                                                 max_days=self.max_holding_days)
                    
                    # 更新持仓状态 - 使用清晰的状态机制
                    if signal == 'open_long':
                        position = 'long'
                        days_held = 1
                        reason = 'z_threshold'
                    elif signal == 'open_short':
                        position = 'short'
                        days_held = 1
                        reason = 'z_threshold'
                    elif signal == 'close':
                        reason = 'z_threshold' if abs(z_score) < self.z_close else 'force_close'
                        position = None
                        days_held = 0
                    elif position:
                        days_held += 1
                        reason = 'holding'
                    else:
                        reason = 'no_signal'
                else:
                    signal = 'empty'
                    z_score = 0.0
                    reason = 'insufficient_data'
            else:
                phase = 'transition'
                signal = 'empty'
                z_score = 0.0
                reason = 'transition_period'
            
            # 记录结果 - 严格按照需求文档REQ-4.3格式
            result_row = {
                'date': row['date'],
                'signal': signal,
                'z_score': z_score,
                'residual': residual,
                'beta': beta_t,
                'beta_initial': initial_beta,  # 初始β值（从协整模块获取）
                'days_held': days_held,
                'reason': reason,
                'phase': phase,
                'beta_window_used': beta_window  # 使用的β值时间窗口
            }
            
            # 如果提供了配对信息，添加配对相关字段
            if pair_info:
                result_row['pair'] = pair_info.get('pair', '')
                result_row['symbol_x'] = pair_info.get('symbol_x', '')
                result_row['symbol_y'] = pair_info.get('symbol_y', '')
            else:
                # 从数据中推断（如果可能）
                result_row['pair'] = ''
                result_row['symbol_x'] = ''
                result_row['symbol_y'] = ''
            
            results.append(result_row)
        
        signals_df = pd.DataFrame(results)
        logger.info(f"生成信号: {len(signals_df)}条记录, 收敛状态: {converged}")
        
        return signals_df
    
    def generate_all_signals(self,
                           pairs_params: Dict,
                           price_data: pd.DataFrame,
                           convergence_end: str,
                           signal_start: str,
                           hist_start: Optional[str] = None,
                           hist_end: Optional[str] = None) -> pd.DataFrame:
        """
        批量生成所有配对信号 - REQ-3.4.1
        
        Args:
            pairs_params: 配对参数字典
            price_data: 价格数据DataFrame
            convergence_end: 收敛期结束日期
            signal_start: 信号生成开始日期
            hist_start: 历史数据开始日期
            hist_end: 历史数据结束日期
            
        Returns:
            all_signals_df: 合并的信号DataFrame
        """
        all_signals = []
        
        for pair_name, pair_params in pairs_params.items():
            try:
                symbol_x, symbol_y = pair_name.split('-')
                
                # 准备配对数据
                if f'{symbol_x}' in price_data.columns and f'{symbol_y}' in price_data.columns:
                    pair_data = price_data[['date', symbol_x, symbol_y]].rename(columns={
                        symbol_x: 'x',
                        symbol_y: 'y'
                    }).copy()
                else:
                    logger.warning(f"配对{pair_name}的数据不完整")
                    continue
                
                # 生成信号
                signals = self.process_pair_signals(
                    pair_data=pair_data,
                    initial_beta=pair_params['beta_initial'],
                    convergence_end=convergence_end,
                    signal_start=signal_start,
                    hist_start=hist_start,
                    hist_end=hist_end
                )
                
                if not signals.empty:
                    signals['pair'] = pair_name
                    all_signals.append(signals)
                    
            except Exception as e:
                logger.error(f"处理配对{pair_name}失败: {e}")
                continue
        
        if all_signals:
            result_df = pd.concat(all_signals, ignore_index=True)
            logger.info(f"批量处理完成: {len(all_signals)}个配对, {len(result_df)}条信号")
            return result_df
        else:
            logger.warning("没有成功处理任何配对")
            return pd.DataFrame()


# 验证函数
def validate_kalman_calculation():
    """验证Kalman滤波计算的正确性"""
    print("=== Kalman滤波计算验证 ===")
    
    # 测试用例1: 基本更新
    kf = KalmanFilter1D(initial_beta=1.0, Q=1e-4, R=1e-2, P0=0.1)
    result = kf.update(y_t=2.1, x_t=2.0)
    
    print(f"测试1 - 基本更新:")
    print(f"  输入: y=2.1, x=2.0, β₀=1.0")
    print(f"  输出: β={result['beta']:.6f}, residual={result['residual']:.6f}")
    print(f"  期望: β略微调整, residual≈0.1")
    print(f"  ✓ 通过" if abs(result['residual'] - 0.1) < 0.05 else "  ✗ 失败")
    
    # 测试用例2: 数值稳定性
    kf2 = KalmanFilter1D(initial_beta=0.5, Q=1e-6, R=1e-6, P0=1e-3)
    for i in range(100):
        result = kf2.update(y_t=0.5 + 0.01*np.random.randn(), x_t=1.0)
    
    print(f"\n测试2 - 数值稳定性(100次迭代):")
    print(f"  最终β: {result['beta']:.6f}")
    print(f"  β历史长度: {len(kf2.beta_history)}")
    print(f"  ✓ 通过" if np.isfinite(result['beta']) else "  ✗ 失败")


def validate_signal_calculation():
    """验证信号计算的正确性"""
    print("\n=== 信号计算验证 ===")
    
    sg = SignalGenerator()
    
    # 测试Z-score计算
    residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z_score = sg.calculate_zscore(residuals, 10)
    
    # 手动计算验证
    mean = np.mean(residuals)
    std = np.std(residuals, ddof=1)
    expected_z = (10 - mean) / std
    
    print(f"测试1 - Z-score计算:")
    print(f"  数据: {residuals}")
    print(f"  计算结果: {z_score:.6f}")
    print(f"  期望结果: {expected_z:.6f}")
    print(f"  ✓ 通过" if abs(z_score - expected_z) < 1e-10 else "  ✗ 失败")
    
    # 测试信号逻辑
    test_cases = [
        ((-2.5, None, 0), 'open_long'),
        ((2.5, None, 0), 'open_short'),
        ((0.3, 'open_long', 5), 'close'),
        ((1.0, 'open_long', 30), 'close'),  # 强制平仓
        ((-2.5, 'open_long', 5), 'hold'),   # 防重复开仓
    ]
    
    print(f"\n测试2 - 信号逻辑:")
    for i, ((z, pos, days), expected) in enumerate(test_cases, 1):
        result = sg.generate_signal(z, pos, days)
        status = "✓ 通过" if result == expected else "✗ 失败"
        print(f"  用例{i}: z={z}, pos={pos}, days={days} -> {result} (期望: {expected}) {status}")


def validate_zscore_calculation():
    """验证Z-score计算的数学正确性"""
    print("\n=== Z-score数学验证 ===")
    
    sg = SignalGenerator()
    
    # 测试用例1: 标准正态分布
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    
    for window in [10, 30, 60]:
        z_score = sg.calculate_zscore(normal_data, window)
        print(f"  窗口{window}: Z-score = {z_score:.4f}")
    
    # 测试用例2: 已知分布
    known_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0])  # 最后值为0
    z_score = sg.calculate_zscore(known_data, 10)
    
    # 手动验证
    mean_val = np.mean(known_data)  # 2.5
    std_val = np.std(known_data, ddof=1)  # 约1.58
    expected_z = (0 - mean_val) / std_val  # (0-2.5)/1.58 ≈ -1.58
    
    print(f"  已知数据验证:")
    print(f"    数据: {known_data}")
    print(f"    均值: {mean_val:.2f}, 标准差: {std_val:.2f}")
    print(f"    计算Z-score: {z_score:.4f}")
    print(f"    期望Z-score: {expected_z:.4f}")
    print(f"    ✓ 通过" if abs(z_score - expected_z) < 1e-3 else "    ✗ 失败")


if __name__ == '__main__':
    # 运行所有验证
    validate_kalman_calculation()
    validate_signal_calculation() 
    validate_zscore_calculation()
    
    print("\n=== 验证完成 ===")