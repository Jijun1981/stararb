"""
绩效指标计算模块
实现标准化的策略绩效指标计算，符合CFA Institute标准

Test: tests/test_backtest_engine.py (TestPerformanceMetrics)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_returns(trades: pd.DataFrame) -> np.ndarray:
    """
    TC-BE.2.1: 计算收益率序列
    
    Args:
        trades: 交易记录DataFrame
        
    Returns:
        日收益率序列
    """
    # 检查必要的列
    pnl_col = None
    capital_col = None
    
    # 寻找收益列
    for col in ['net_pnl', 'pnl', 'profit', 'returns']:
        if col in trades.columns:
            pnl_col = col
            break
    
    # 寻找资金列
    for col in ['capital', 'balance', 'equity']:
        if col in trades.columns:
            capital_col = col
            break
    
    if pnl_col is None:
        raise ValueError(f"trades DataFrame必须包含收益列(net_pnl/pnl/profit/returns)，当前列: {list(trades.columns)}")
    
    # 如果没有资金列，创建一个
    if capital_col is None:
        trades = trades.copy()
        trades['capital'] = 0.0  # 将在下面正确计算
        capital_col = 'capital'
        
    # 计算每日收益率
    returns = trades[pnl_col] / trades[capital_col].shift(1)
    
    # 处理第一个收益率（相对于初始资金）
    if len(returns) > 0:
        returns.iloc[0] = trades[pnl_col].iloc[0] / trades[capital_col].iloc[0] if trades[capital_col].iloc[0] != 0 else 0
    
    return returns.values


def calculate_sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
    """
    TC-BE.2.3: 计算Sharpe比率
    
    Args:
        daily_returns: 日收益率序列
        risk_free_rate: 年化无风险利率，默认3%
        
    Returns:
        年化Sharpe比率
    """
    if len(daily_returns) == 0:
        return 0.0
        
    # 计算年化收益率
    annualized_return = np.mean(daily_returns) * 252
    
    # 计算年化波动率
    annualized_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)
    
    # 计算Sharpe比率
    if annualized_vol == 0:
        return 0.0
        
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
    
    return sharpe_ratio


def calculate_max_drawdown(cum_returns: np.ndarray) -> Tuple[float, int]:
    """
    TC-BE.2.4: 计算最大回撤
    
    Args:
        cum_returns: 累计收益序列
        
    Returns:
        (最大回撤, 回撤持续期)
    """
    if len(cum_returns) == 0:
        return 0.0, 0
        
    # 计算运行最高点
    running_max = np.maximum.accumulate(cum_returns)
    
    # 计算回撤
    drawdowns = (cum_returns - running_max) / running_max
    
    # 最大回撤
    max_drawdown = np.min(drawdowns)
    
    # 计算回撤持续期
    max_dd_duration = 0
    current_duration = 0
    
    for dd in drawdowns:
        if dd < 0:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0
    
    return max_drawdown, max_dd_duration


def calculate_win_loss_stats(trades: List[Dict]) -> Dict[str, float]:
    """
    TC-BE.2.5: 计算胜率和盈亏比
    
    Args:
        trades: 交易记录列表
        
    Returns:
        包含胜率、平均盈利、平均亏损、盈亏比的字典
    """
    if not trades:
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0
        }
    
    # 分离盈利和亏损交易
    profits = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
    losses = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
    
    # 计算统计指标
    total_trades = len(trades)
    win_trades = len(profits)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    avg_win = np.mean(profits) if profits else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    # 盈亏比（平均盈利/平均亏损的绝对值）
    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': total_trades
    }


def calculate_annualized_metrics(daily_returns: np.ndarray) -> Dict[str, float]:
    """
    TC-BE.2.2: 计算年化指标
    
    Args:
        daily_returns: 日收益率序列
        
    Returns:
        年化收益率和波动率字典
    """
    if len(daily_returns) == 0:
        return {'annualized_return': 0.0, 'annualized_volatility': 0.0}
    
    # 年化收益率
    annualized_return = np.mean(daily_returns) * 252
    
    # 年化波动率
    annualized_volatility = np.std(daily_returns, ddof=1) * np.sqrt(252)
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility
    }


# 算法验证函数

def validate_sharpe_calculation(daily_returns: np.ndarray, risk_free_rate: float = 0.03) -> Tuple[float, float]:
    """
    Sharpe比率计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现
    sharpe1 = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    
    # 方法2: 基于超额收益的计算
    if len(daily_returns) == 0:
        return 0.0, 0.0
        
    daily_rf_rate = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf_rate
    
    if np.std(excess_returns, ddof=1) == 0:
        sharpe2 = 0.0
    else:
        sharpe2 = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)
    
    return sharpe1, sharpe2


def validate_max_drawdown_calculation(cum_returns: np.ndarray) -> Tuple[float, float]:
    """
    最大回撤计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现
    max_dd1, _ = calculate_max_drawdown(cum_returns)
    
    # 方法2: 逐点计算验证
    if len(cum_returns) == 0:
        return 0.0, 0.0
    
    max_dd2 = 0.0
    peak = cum_returns[0]
    
    for value in cum_returns:
        if value > peak:
            peak = value
        
        drawdown = (value - peak) / peak if peak != 0 else 0.0
        max_dd2 = min(max_dd2, drawdown)
    
    return max_dd1, max_dd2


class PerformanceCalculator:
    """绩效计算器类"""
    
    def __init__(self, initial_capital: float = 5000000, risk_free_rate: float = 0.03):
        """
        初始化绩效计算器
        
        Args:
            initial_capital: 初始资金
            risk_free_rate: 年化无风险利率
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
    
    def calculate_comprehensive_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        计算综合绩效指标
        
        Args:
            trades: 交易记录列表
            
        Returns:
            综合绩效指标字典
        """
        if not trades:
            return self._empty_metrics()
        
        # 转换为DataFrame便于计算
        trades_df = pd.DataFrame(trades)
        
        # 计算累计资金变化
        trades_df['capital'] = self.initial_capital
        cumulative_pnl = 0
        
        for i in range(len(trades_df)):
            if i > 0:
                cumulative_pnl += trades_df.loc[i-1, 'net_pnl']
                trades_df.loc[i, 'capital'] = self.initial_capital + cumulative_pnl
        
        daily_returns = calculate_returns(trades_df)
        
        # 过滤无效值
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        daily_returns = daily_returns[np.isfinite(daily_returns)]
        
        if len(daily_returns) == 0:
            return self._empty_metrics()
        
        # 计算累计收益
        cum_returns = np.cumprod(1 + daily_returns)
        
        # 计算各项指标
        annualized_metrics = calculate_annualized_metrics(daily_returns)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns, self.risk_free_rate)
        max_drawdown, dd_duration = calculate_max_drawdown(cum_returns)
        win_loss_stats = calculate_win_loss_stats(trades)
        
        # 汇总所有指标
        final_capital = self.initial_capital + sum(t['net_pnl'] for t in trades)
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        metrics = {
            **annualized_metrics,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'drawdown_duration': dd_duration,
            **win_loss_stats,
            'total_return': total_return,
            'final_capital': final_capital
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """返回空的指标字典"""
        return {
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'drawdown_duration': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0,
            'total_return': 0.0,
            'final_capital': self.initial_capital
        }