"""
策略模块
实现Kalman滤波动态beta估计和交易信号生成

Test: tests/test_strategy.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import yaml
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Kalman滤波器用于动态beta估计
    状态方程: β(t) = β(t-1) + w(t), w(t) ~ N(0, Q)
    观测方程: Y(t) = β(t) × X(t) + v(t), v(t) ~ N(0, R)
    """
    
    def __init__(self):
        self.beta = None  # 当前beta估计
        self.P = None     # 估计误差协方差
        self.R = None     # 观测噪声方差
        self.Q = None     # 过程噪声方差
        
    def initialize(self, x: np.ndarray, y: np.ndarray, halflife: float = None) -> None:
        """
        使用初始数据估计参数
        
        Args:
            x: 自变量序列（对数价格）
            y: 因变量序列（对数价格）
            halflife: 半衰期（天），用于计算Q
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # 检查数据有效性
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f"数据长度不足：需要至少2个数据点，当前x={len(x)}, y={len(y)}")
        
        if len(x) != len(y):
            raise ValueError(f"X和Y长度不匹配：x={len(x)}, y={len(y)}")
        
        # 检查X是否有变化
        if np.std(x) < 1e-10:
            raise ValueError("X序列无变化，无法进行回归分析")
        
        # 使用OLS估计初始beta
        X = add_constant(x)
        model = OLS(y, X).fit()
        self.beta = model.params[1]
        
        # 估计观测噪声R（残差方差）
        residuals = model.resid
        self.R = np.var(residuals)
        
        # 估计过程噪声Q - 修正计算方法
        if halflife is not None and halflife > 0:
            # 使用更合理的Q估计：Q应该允许beta有一定的变化
            # 经验公式：Q = lambda * R，其中lambda基于半衰期
            lambda_factor = 1.0 / halflife  # 半衰期越短，beta变化越快
            # 限制lambda在合理范围内
            lambda_factor = max(0.001, min(0.1, lambda_factor))
            self.Q = lambda_factor * self.R
        else:
            # 默认值：Q = 0.01 * R （允许适度的beta变化）
            self.Q = 0.01 * self.R
        
        # 初始化估计误差协方差 - 基于初始beta的不确定性
        self.P = np.var([self.beta]) if len(x) > 2 else 1.0
        
        logger.info(f"Kalman初始化: beta={self.beta:.4f}, R={self.R:.6f}, Q={self.Q:.6f}, lambda={self.Q/self.R:.4f}")
    
    def update(self, x_t: float, y_t: float) -> float:
        """
        Kalman滤波递推更新
        
        Args:
            x_t: 当前时刻的x值
            y_t: 当前时刻的y值
            
        Returns:
            更新后的beta估计
        """
        if self.beta is None:
            raise ValueError("Kalman滤波器未初始化，请先调用initialize()")
        
        # 预测步
        beta_pred = self.beta  # β(t|t-1) = β(t-1|t-1)
        P_pred = self.P + self.Q  # P(t|t-1) = P(t-1|t-1) + Q
        
        # 更新步
        e_t = y_t - beta_pred * x_t  # 预测误差
        S_t = x_t**2 * P_pred + self.R  # 误差协方差
        K_t = P_pred * x_t / S_t  # Kalman增益
        
        # 更新估计
        self.beta = beta_pred + K_t * e_t  # β(t|t)
        self.P = (1 - K_t * x_t) * P_pred  # P(t|t)
        
        return self.beta


def calculate_zscore(spreads: np.ndarray, window: int = 60) -> np.ndarray:
    """
    计算Z-score（滚动窗口标准化）
    
    Args:
        spreads: 价差序列
        window: 滚动窗口大小（默认60天）
        
    Returns:
        Z-score序列
    """
    spreads = np.asarray(spreads)
    z_scores = np.full_like(spreads, np.nan)
    
    if len(spreads) < window:
        return z_scores
    
    # 计算滚动均值和标准差
    for i in range(window - 1, len(spreads)):
        window_data = spreads[i - window + 1:i + 1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        if std > 0:
            z_scores[i] = (spreads[i] - mean) / std
        else:
            z_scores[i] = 0
    
    return z_scores


def get_position_ratio(beta: float) -> Tuple[int, int]:
    """
    将动态beta转换为最接近的简单整数比
    
    Args:
        beta: 动态对冲比率
        
    Returns:
        (y_units, x_units) 整数比例
    """
    # 常用交易比例（y:x）
    common_ratios = [
        (1, 1),   # beta = 1.0
        (1, 2),   # beta = 0.5
        (2, 3),   # beta = 0.67
        (3, 4),   # beta = 0.75
        (4, 5),   # beta = 0.8
        (1, 1),   # beta = 1.0 (重复，优先级高)
        (5, 4),   # beta = 1.25
        (4, 3),   # beta = 1.33
        (3, 2),   # beta = 1.5
        (5, 3),   # beta = 1.67
        (2, 1),   # beta = 2.0
    ]
    
    # 找最接近的比例
    best_ratio = min(common_ratios, 
                     key=lambda r: abs(r[0]/r[1] - beta))
    
    return best_ratio


def generate_signals(
    x: np.ndarray,
    y: np.ndarray,
    init_start: str,
    init_end: str,
    backtest_start: str,
    backtest_end: str,
    dates: pd.DatetimeIndex,
    halflife: float = 10.0,
    zscore_window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    pair_name: str = ""
) -> pd.DataFrame:
    """
    生成交易信号
    
    Args:
        x: 自变量价格序列（对数价格）
        y: 因变量价格序列（对数价格）
        init_start/end: 初始化期
        backtest_start/end: 回测期
        dates: 日期索引
        halflife: 半衰期（用于Kalman参数）
        zscore_window: Z-score计算窗口
        entry_threshold: 开仓阈值
        exit_threshold: 平仓阈值
        pair_name: 配对名称
        
    Returns:
        信号DataFrame
    """
    # 转换为Series便于日期筛选
    x_series = pd.Series(x, index=dates)
    y_series = pd.Series(y, index=dates)
    
    # 初始化Kalman滤波器
    init_x = x_series[init_start:init_end].values
    init_y = y_series[init_start:init_end].values
    
    kf = KalmanFilter()
    kf.initialize(init_x, init_y, halflife)
    
    # 回测期数据
    backtest_x = x_series[backtest_start:backtest_end]
    backtest_y = y_series[backtest_start:backtest_end]
    backtest_dates = backtest_x.index
    
    # 计算动态beta和价差
    betas = []
    spreads = []
    
    for i in range(len(backtest_x)):
        beta = kf.update(backtest_x.iloc[i], backtest_y.iloc[i])
        spread = backtest_y.iloc[i] - beta * backtest_x.iloc[i]
        
        betas.append(beta)
        spreads.append(spread)
    
    # 计算Z-score
    z_scores = calculate_zscore(np.array(spreads), zscore_window)
    
    # 生成信号
    signals = []
    position = 0  # 0: 无持仓, 1: 做多价差, -1: 做空价差
    entry_date = None
    entry_z = None
    entry_beta = None
    
    for i in range(len(backtest_dates)):
        date = backtest_dates[i]
        z = z_scores[i]
        beta = betas[i]
        
        # 跳过Z-score无效的时期（前60天）
        if np.isnan(z):
            signals.append({
                'date': date,
                'pair': pair_name,
                'action': 'wait',
                'side': 'none',
                'z_score': z,
                'beta': beta,
                'position_ratio': 'N/A'
            })
            continue
        
        # 信号逻辑
        action = 'hold'
        side = 'none'
        
        if position == 0:  # 无持仓
            if z >= entry_threshold:
                action = 'open'
                side = 'short_spread'
                position = -1
                entry_date = date
                entry_z = z
                entry_beta = beta
            elif z <= -entry_threshold:
                action = 'open'
                side = 'long_spread'
                position = 1
                entry_date = date
                entry_z = z
                entry_beta = beta
        else:  # 有持仓
            if abs(z) <= exit_threshold:
                action = 'close'
                side = 'exit'
                position = 0
                entry_date = None
                entry_z = None
                entry_beta = None
        
        # 计算仓位比例
        y_units, x_units = get_position_ratio(abs(beta))
        position_ratio = f"{y_units}:{x_units}"
        
        signals.append({
            'date': date,
            'pair': pair_name,
            'action': action,
            'side': side,
            'z_score': z,
            'beta': beta,
            'position_ratio': position_ratio
        })
    
    return pd.DataFrame(signals)


def batch_process_pairs(
    config_file: str = 'configs/selected_pairs.yaml',
    data_dir: str = 'data/futures',
    init_start: str = '2020-01-02',
    init_end: str = '2021-12-31',
    backtest_start: str = '2024-01-02',
    backtest_end: str = '2025-08-15'
) -> pd.DataFrame:
    """
    批量处理所有配对
    
    Args:
        config_file: 配对配置文件路径
        data_dir: 数据目录
        init_start/end: 初始化期
        backtest_start/end: 回测期
        
    Returns:
        所有配对的信号汇总
    """
    # 加载配对配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    pairs = config['pairs']
    all_signals = []
    
    # 处理每个配对
    for i, pair_config in enumerate(pairs):
        pair_name = pair_config['pair']
        x_symbol = pair_config['X']
        y_symbol = pair_config['Y']
        halflife = pair_config['statistics']['halflife']
        
        logger.info(f"处理配对 {i+1}/{len(pairs)}: {pair_name}")
        
        try:
            # 加载数据
            x_df = pd.read_parquet(f"{data_dir}/{x_symbol}.parquet")
            y_df = pd.read_parquet(f"{data_dir}/{y_symbol}.parquet")
            
            # 转换日期
            x_df['date'] = pd.to_datetime(x_df['date'])
            y_df['date'] = pd.to_datetime(y_df['date'])
            
            # 对齐数据
            merged = pd.merge(x_df[['date', 'close']], 
                            y_df[['date', 'close']], 
                            on='date', 
                            suffixes=('_x', '_y'))
            merged.set_index('date', inplace=True)
            
            # 计算对数价格
            merged['log_x'] = np.log(merged['close_x'])
            merged['log_y'] = np.log(merged['close_y'])
            
            # 生成信号
            signals = generate_signals(
                x=merged['log_x'].values,
                y=merged['log_y'].values,
                init_start=init_start,
                init_end=init_end,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                dates=merged.index,
                halflife=halflife,
                pair_name=pair_name
            )
            
            all_signals.append(signals)
            
        except Exception as e:
            logger.error(f"处理配对 {pair_name} 失败: {e}")
            continue
    
    # 汇总所有信号
    if all_signals:
        result = pd.concat(all_signals, ignore_index=True)
        logger.info(f"完成批量处理，共生成 {len(result)} 条信号记录")
        return result
    else:
        logger.warning("没有成功处理任何配对")
        return pd.DataFrame()