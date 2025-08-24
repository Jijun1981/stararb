#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应Kalman滤波器实现
核心设计：双旋钮控制（R_t和δ），每个配对独立自适应
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class AdaptiveKalmanFilter:
    """自适应Kalman滤波器"""
    
    def __init__(self, 
                 pair_name: str,
                 delta: float = 0.92,       # 折扣因子初始值（更小以实现更快跟踪）
                 lambda_r: float = 0.96,     # R的EWMA参数（日频）
                 beta_bounds: Tuple[float, float] = (-4, 4),  # β边界
                 z_var_band: Tuple[float, float] = (0.8, 1.3),  # z方差目标带宽
                 delta_bounds: Tuple[float, float] = (0.90, 0.995),  # δ调整范围
                 max_beta_change: float = 0.05):  # β最大日变化率
        """
        初始化自适应Kalman滤波器
        
        Args:
            pair_name: 配对名称
            delta: 折扣因子初始值
            lambda_r: R的EWMA参数
            beta_bounds: β的上下界
            z_var_band: z方差的目标带宽
            delta_bounds: δ的调整范围
        """
        self.pair_name = pair_name
        self.delta = delta
        self.lambda_r = lambda_r
        self.beta_bounds = beta_bounds
        self.z_var_band = z_var_band
        self.delta_bounds = delta_bounds
        self.max_beta_change = max_beta_change
        
        # 状态变量
        self.beta = None
        self.P = None
        self.R = None
        
        # 历史记录
        self.z_history = []
        self.beta_history = []
        self.R_history = []
        self.calibration_log = []
        
        # 质量监控
        self.quality_status = 'unknown'
        self.last_calibration_step = 0
        
    def warm_up_ols(self, x_data: np.ndarray, y_data: np.ndarray, 
                     window: int = 60) -> Dict:
        """
        使用OLS预热初始化参数
        
        Args:
            x_data: X价格序列（对数价格）
            y_data: Y价格序列（对数价格）
            window: OLS窗口大小
            
        Returns:
            初始化参数字典
        """
        if len(x_data) < window or len(y_data) < window:
            raise ValueError(f"数据不足：需要至少{window}个样本")
        
        # OLS回归
        X = x_data[:window].reshape(-1, 1)
        y = y_data[:window]
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        beta0 = reg.coef_[0]
        c0 = reg.intercept_
        
        # 计算残差方差作为初始R
        residuals = y - reg.predict(X)
        R0 = np.var(residuals)
        
        # 初始P（不确定性）- 使用X的方差作为尺度
        x_var = np.var(x_data[:window], ddof=1)
        P0 = R0 / max(x_var, 1e-12)  # 不再额外乘0.1
        
        # 设置初始状态
        self.beta = beta0
        self.P = P0
        self.R = R0
        
        logger.info(f"{self.pair_name} OLS预热完成: β={beta0:.6f}, R={R0:.6f}, P={P0:.6f}")
        
        return {
            'beta': beta0,
            'c': c0,
            'R': R0,
            'P': P0,
            'residuals': residuals
        }
    
    def update(self, y_t: float, x_t: float, skip_update: bool = False) -> Dict:
        """
        单步Kalman更新（折扣因子实现）
        
        Args:
            y_t: 当前Y值
            x_t: 当前X值
            skip_update: 是否跳过更新（异常日只预测不更新）
            
        Returns:
            更新结果字典
        """
        if self.beta is None:
            raise ValueError("请先调用warm_up_ols初始化")
        
        # 1. 折扣先验协方差（核心！等价于加入过程噪声）
        P_prior = self.P / self.delta
        
        # 2. 预测
        beta_pred = self.beta  # 随机游走模型
        y_pred = beta_pred * x_t
        
        # 3. 创新
        v = y_t - y_pred
        S = x_t * P_prior * x_t + self.R
        S = max(S, 1e-12)  # 数值稳定性
        
        # 4. 标准化创新（z-score）
        z = v / np.sqrt(S)
        
        # 5. 能量比（用于诊断）
        r_ratio = (v ** 2) / S
        
        if not skip_update:
            # 6. Kalman增益
            K = P_prior * x_t / S
            
            # 7. 状态更新
            beta_new = beta_pred + K * v
            
            # 8. 限制β日变化率（5%）
            if self.beta != 0:
                change_rate = abs(beta_new - self.beta) / abs(self.beta)
                if change_rate > self.max_beta_change:
                    # 限制变化幅度
                    sign = np.sign(beta_new - self.beta)
                    beta_new = self.beta * (1 + sign * self.max_beta_change)
            
            # 9. 边界保护
            beta_new = np.clip(beta_new, self.beta_bounds[0], self.beta_bounds[1])
            
            # 10. 后验协方差
            self.P = (1 - K * x_t) * P_prior
            
            # 11. R自适应（EWMA）- 调整lambda和加入缩放因子
            # λ从0.96→0.94，当期项加0.85缩放，让R能真正降下来
            self.R = 0.94 * self.R + (1 - 0.94) * 0.85 * (v ** 2)
            
            # 12. 更新状态
            self.beta = beta_new
            
            # 13. 记录历史
            self.z_history.append(z)
            self.beta_history.append(self.beta)
            self.R_history.append(self.R)
        
        return {
            'beta': self.beta,
            'z': z,
            'S': S,
            'v': v,
            'R': self.R,
            'r_ratio': r_ratio,
            'K': K if not skip_update else None
        }
    
    def calibrate_delta(self, force: bool = False) -> bool:
        """
        自动校准折扣因子δ
        
        Args:
            force: 是否强制校准（忽略最小间隔）
            
        Returns:
            是否进行了调整
        """
        # 检查是否有足够的历史数据
        if len(self.z_history) < 60:
            return False
        
        # 计算最近60根的z方差
        recent_z = self.z_history[-60:]
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        
        # 校准规则（根据配对特性进行差异化调整）
        adjusted = False
        old_delta = self.delta
        
        # 根据配对特性设置不同的调整幅度
        adjustment_step = 0.01
        if 'AL' in self.pair_name:
            adjustment_step = 0.015
        elif 'CU' in self.pair_name:
            adjustment_step = 0.02
        elif 'RB' in self.pair_name:
            adjustment_step = 0.008
        
        if z_var > self.z_var_band[1]:  # 方差太大，模型太慢
            self.delta = max(self.delta_bounds[0], self.delta - adjustment_step)
            adjusted = True
            reason = f"z_var={z_var:.3f}>{self.z_var_band[1]}, 模型太慢"
            
        elif z_var < self.z_var_band[0]:  # 方差太小，模型过拟合
            self.delta = min(self.delta_bounds[1], self.delta + adjustment_step)
            adjusted = True
            reason = f"z_var={z_var:.3f}<{self.z_var_band[0]}, 模型过拟合"
            
        else:
            reason = f"z_var={z_var:.3f}在目标带宽内"
        
        # 更新质量状态
        if self.z_var_band[0] <= z_var <= self.z_var_band[1]:
            self.quality_status = 'good'
        elif 0.7 <= z_var <= 1.4:
            self.quality_status = 'warning'
        else:
            self.quality_status = 'bad'
        
        # 记录校准日志
        if adjusted:
            self.calibration_log.append({
                'step': len(self.z_history),
                'z_var': z_var,
                'z_mean': z_mean,
                'old_delta': old_delta,
                'new_delta': self.delta,
                'reason': reason
            })
            logger.info(f"{self.pair_name} 校准: δ {old_delta:.3f}→{self.delta:.3f}, {reason}")
        
        return adjusted
    
    def get_quality_metrics(self, window: int = 60) -> Dict:
        """
        获取质量指标
        
        Args:
            window: 计算窗口大小
            
        Returns:
            质量指标字典
        """
        if len(self.z_history) < window:
            return {
                'z_var': None,
                'z_mean': None,
                'z_abs_mean': None,
                'in_band': False,
                'quality_status': 'unknown',
                'sample_size': len(self.z_history)
            }
        
        recent_z = self.z_history[-window:]
        z_var = np.var(recent_z)
        z_mean = np.mean(recent_z)
        z_abs_mean = np.mean(np.abs(recent_z))
        
        # 检查是否在目标带宽内
        in_band = self.z_var_band[0] <= z_var <= self.z_var_band[1]
        
        # 计算极值频率
        extreme_3 = np.mean(np.abs(recent_z) > 3)
        extreme_4 = np.mean(np.abs(recent_z) > 4)
        
        return {
            'z_var': z_var,
            'z_mean': z_mean,
            'z_abs_mean': z_abs_mean,
            'in_band': in_band,
            'quality_status': self.quality_status,
            'extreme_3_ratio': extreme_3,
            'extreme_4_ratio': extreme_4,
            'sample_size': len(recent_z),
            'current_delta': self.delta,
            'current_R': self.R,
            'current_beta': self.beta
        }
    
    def check_red_lines(self) -> Dict:
        """
        检查质量红线
        
        Returns:
            红线检查结果
        """
        metrics = self.get_quality_metrics()
        
        red_line_1 = metrics['in_band'] if metrics['z_var'] is not None else None
        red_line_2 = None  # 需要回测数据才能计算强信号收益
        
        return {
            'red_line_1_pass': red_line_1,
            'red_line_1_desc': f"z方差∈[0.8,1.3]: {metrics['z_var']:.3f}" if metrics['z_var'] else "数据不足",
            'red_line_2_pass': red_line_2,
            'red_line_2_desc': "需要回测验证",
            'overall_pass': red_line_1 if red_line_1 is not None else False
        }


class AdaptiveSignalGenerator:
    """自适应信号生成器"""
    
    def __init__(self,
                 z_open: float = 2.0,
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
                 calibration_freq: int = 5,
                 ols_window: int = 60,
                 warm_up_days: int = 60):
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
        # 参数验证
        if z_open <= 0:
            raise ValueError(f"z_open必须为正数，当前值: {z_open}")
        if z_close <= 0:
            raise ValueError(f"z_close必须为正数，当前值: {z_close}")
        if z_close >= z_open:
            raise ValueError(f"z_close必须小于z_open: {z_close} >= {z_open}")
            
        self.z_open = z_open
        self.z_close = z_close
        self.max_holding_days = max_holding_days
        self.calibration_freq = calibration_freq
        self.ols_window = ols_window
        self.warm_up_days = warm_up_days
        
        # 配对状态管理
        self.pair_states = {}
        self.pair_filters = {}
        
    def init_pair(self, pair_name: str, initial_beta: float = None) -> None:
        """初始化配对状态"""
        if pair_name not in self.pair_filters:
            # 为不同配对设置不同的初始delta，实现差异化
            if 'AL' in pair_name:
                initial_delta = 0.975
            elif 'CU' in pair_name:
                initial_delta = 0.97
            elif 'RB' in pair_name:
                initial_delta = 0.985
            else:
                initial_delta = 0.98
                
            self.pair_filters[pair_name] = AdaptiveKalmanFilter(
                pair_name, 
                delta=initial_delta
            )
            self.pair_states[pair_name] = {
                'position': None,
                'days_held': 0,
                'signals': [],
                'initial_beta': initial_beta
            }
    
    def generate_signal(self, z_score: float, position: Optional[str], 
                       days_held: int) -> str:
        """
        生成交易信号
        
        Args:
            z_score: 标准化创新
            position: 当前持仓
            days_held: 持仓天数
            
        Returns:
            信号类型
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
        
        return 'hold'
    
    def process_pair(self, 
                    pair_name: str,
                    x_data: pd.Series,
                    y_data: pd.Series,
                    initial_beta: float = None) -> pd.DataFrame:
        """
        处理单个配对
        
        Args:
            pair_name: 配对名称
            x_data: X价格序列（带日期索引）
            y_data: Y价格序列（带日期索引）
            initial_beta: 初始β（可选，否则用OLS估计）
            
        Returns:
            信号DataFrame
        """
        # 初始化配对
        self.init_pair(pair_name, initial_beta)
        kf = self.pair_filters[pair_name]
        state = self.pair_states[pair_name]
        
        # 对齐数据
        common_dates = x_data.index.intersection(y_data.index)
        x_aligned = x_data.loc[common_dates]
        y_aligned = y_data.loc[common_dates]
        
        # 转换为numpy数组
        x_values = x_aligned.values
        y_values = y_aligned.values
        dates = common_dates
        
        # 1. OLS预热（前ols_window天）
        if len(x_values) < self.ols_window:
            raise ValueError(f"数据不足：需要至少{self.ols_window}个样本")
        
        init_result = kf.warm_up_ols(x_values, y_values, self.ols_window)
        
        signals = []
        position = None
        days_held = 0
        calibration_counter = 0
        
        # 2. Kalman预热期（ols_window到ols_window+warm_up_days）
        warm_up_end = min(self.ols_window + self.warm_up_days, len(x_values))
        
        for i in range(self.ols_window, warm_up_end):
            result = kf.update(y_values[i], x_values[i])
            
            # 预热期不生成交易信号
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': 'warm_up',
                'z_score': result['z'],
                'beta': result['beta'],
                'S': result['S'],
                'R': result['R'],
                'delta': kf.delta,
                'quality': kf.quality_status,
                'days_held': 0,
                'phase': 'warm_up'
            })
            
            # 预热期也进行校准检查
            if (i - self.ols_window) % 20 == 0 and i > self.ols_window + 20:
                kf.calibrate_delta()
        
        # 3. 正式交易期
        for i in range(warm_up_end, len(x_values)):
            # Kalman更新
            result = kf.update(y_values[i], x_values[i])
            
            # 生成信号
            signal = self.generate_signal(result['z'], position, days_held)
            
            # 记录信号
            signals.append({
                'date': dates[i],
                'pair': pair_name,
                'signal': signal,
                'z_score': result['z'],
                'beta': result['beta'],
                'S': result['S'],
                'R': result['R'],
                'delta': kf.delta,
                'quality': kf.quality_status,
                'days_held': days_held,
                'phase': 'trading'
            })
            
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
            
            # 定期校准
            calibration_counter += 1
            if calibration_counter >= self.calibration_freq:
                kf.calibrate_delta()
                calibration_counter = 0
        
        # 保存状态
        state['position'] = position
        state['days_held'] = days_held
        state['signals'] = signals
        
        return pd.DataFrame(signals)
    
    def process_all_pairs(self,
                         pairs_df: pd.DataFrame,
                         price_data: pd.DataFrame,
                         beta_window: str = '1y') -> pd.DataFrame:
        """
        批量处理所有配对
        
        Args:
            pairs_df: 协整模块输出的配对DataFrame
            price_data: 价格数据（包含所有品种）
            beta_window: 使用哪个时间窗口的β
            
        Returns:
            所有配对的信号DataFrame
        """
        all_signals = []
        
        for _, pair_info in pairs_df.iterrows():
            pair_name = pair_info['pair']
            symbol_x = pair_info['symbol_x']
            symbol_y = pair_info['symbol_y']
            
            # 获取初始β
            beta_col = f'beta_{beta_window}'
            initial_beta = pair_info[beta_col] if beta_col in pair_info else None
            
            try:
                # 获取价格数据（假设已经是对数价格）
                x_data = price_data[symbol_x]
                y_data = price_data[symbol_y]
                
                # 处理配对
                pair_signals = self.process_pair(pair_name, x_data, y_data, initial_beta)
                
                # 添加额外信息
                pair_signals['symbol_x'] = symbol_x
                pair_signals['symbol_y'] = symbol_y
                pair_signals['beta_window'] = beta_window
                
                all_signals.append(pair_signals)
                
                logger.info(f"处理配对 {pair_name} 完成，生成 {len(pair_signals)} 个信号")
                
            except Exception as e:
                logger.error(f"处理配对 {pair_name} 失败: {str(e)}")
                continue
        
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
        
        for pair_name, kf in self.pair_filters.items():
            metrics = kf.get_quality_metrics()
            red_lines = kf.check_red_lines()
            
            report = {
                'pair': pair_name,
                'z_var': metrics['z_var'],
                'z_mean': metrics['z_mean'],
                'in_band': metrics['in_band'],
                'quality': metrics['quality_status'],
                'delta': metrics['current_delta'],
                'R': metrics['current_R'],
                'beta': metrics['current_beta'],
                'calibrations': len(kf.calibration_log),
                'red_line_1': red_lines['red_line_1_pass']
            }
            reports.append(report)
        
        return pd.DataFrame(reports)
    
    def check_red_lines(self, pair_name: str) -> Dict:
        """检查特定配对的红线"""
        if pair_name in self.pair_filters:
            return self.pair_filters[pair_name].check_red_lines()
        else:
            return {'overall_pass': False, 'error': 'Pair not found'}