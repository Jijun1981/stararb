"""
风险指标分析模块
实现VaR、CVaR、Sortino等风险度量指标

Test: tests/test_backtest_engine.py (TestRiskMetrics)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    TC-BE.3.2: 计算VaR (Value at Risk)
    
    Args:
        returns: 收益率序列
        confidence: 置信水平，默认95%
        
    Returns:
        VaR值（负数表示损失）
    """
    if len(returns) == 0:
        return 0.0
    
    # 计算分位数（例如95%置信度对应5%分位数）
    percentile = (1 - confidence) * 100
    var_value = np.percentile(returns, percentile)
    
    return var_value


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    TC-BE.3.3: 计算CVaR (Conditional VaR)
    
    Args:
        returns: 收益率序列
        confidence: 置信水平，默认95%
        
    Returns:
        CVaR值（超过VaR的平均损失）
    """
    if len(returns) == 0:
        return 0.0
    
    # 先计算VaR
    var_value = calculate_var(returns, confidence)
    
    # 找出超过VaR的损失
    tail_losses = returns[returns <= var_value]
    
    if len(tail_losses) == 0:
        return var_value
    
    # CVaR = 尾部损失的平均值
    cvar_value = np.mean(tail_losses)
    
    return cvar_value


def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0, 
                          risk_free_rate: float = 0.03) -> float:
    """
    TC-BE.3.4: 计算Sortino比率（下行风险调整收益）
    
    Args:
        returns: 日收益率序列
        target_return: 目标收益率，默认0
        risk_free_rate: 年化无风险利率
        
    Returns:
        Sortino比率
    """
    if len(returns) == 0:
        return 0.0
    
    # 计算年化收益率
    annualized_return = np.mean(returns) * 252
    
    # 计算下行偏差（只考虑低于目标收益的部分）
    daily_target = target_return / 252
    downside_returns = returns[returns < daily_target]
    
    if len(downside_returns) == 0:
        return float('inf') if annualized_return > risk_free_rate else 0.0
    
    # 下行风险（年化）
    downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(252)
    
    if downside_deviation == 0:
        return 0.0
    
    # Sortino比率
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    
    return sortino_ratio


def calculate_distribution_stats(returns: np.ndarray) -> Dict[str, float]:
    """
    TC-BE.3.1: 计算收益率分布统计量
    
    Args:
        returns: 收益率序列
        
    Returns:
        包含标准差、偏度、峰度的字典
    """
    if len(returns) == 0:
        return {'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    
    # 标准差
    std = np.std(returns, ddof=1)
    
    # 偏度（三阶矩）
    skewness = stats.skew(returns)
    
    # 峰度（四阶矩，超额峰度）
    kurtosis = stats.kurtosis(returns)
    
    return {
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def analyze_losing_streaks(trades: List[Dict]) -> Dict[str, int]:
    """
    TC-BE.3.5: 分析最大连续亏损
    
    Args:
        trades: 交易记录列表
        
    Returns:
        包含最大连续亏损次数和金额的字典
    """
    if not trades:
        return {'max_losing_streak': 0, 'max_losing_amount': 0.0}
    
    max_streak = 0
    current_streak = 0
    max_losing_amount = 0.0
    current_losing_amount = 0.0
    
    for trade in trades:
        pnl = trade.get('net_pnl', 0)
        
        if pnl < 0:  # 亏损交易
            current_streak += 1
            current_losing_amount += abs(pnl)
            
            # 更新最大值
            max_streak = max(max_streak, current_streak)
            max_losing_amount = max(max_losing_amount, current_losing_amount)
        else:  # 盈利交易
            current_streak = 0
            current_losing_amount = 0.0
    
    return {
        'max_losing_streak': max_streak,
        'max_losing_amount': max_losing_amount
    }


# 算法验证函数

def validate_var_calculation(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    VaR计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现（历史模拟法）
    var1 = calculate_var(returns, confidence)
    
    # 方法2: 参数法验证（假设正态分布）
    if len(returns) == 0:
        return 0.0, 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    # 正态分布的VaR
    z_score = stats.norm.ppf(1 - confidence)
    var2 = mean_return + z_score * std_return
    
    return var1, var2


def validate_cvar_calculation(returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    CVaR计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现
    cvar1 = calculate_cvar(returns, confidence)
    
    # 方法2: 基于排序的计算
    if len(returns) == 0:
        return 0.0, 0.0
    
    # 找到VaR对应的索引
    var_index = int(np.ceil((1 - confidence) * len(returns)))
    
    # 排序并取尾部均值
    sorted_returns = np.sort(returns)
    cvar2 = np.mean(sorted_returns[:var_index]) if var_index > 0 else 0.0
    
    return cvar1, cvar2


class RiskAnalyzer:
    """风险分析器类"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        初始化风险分析器
        
        Args:
            confidence_levels: VaR/CVaR置信水平列表
        """
        self.confidence_levels = confidence_levels
    
    def analyze_comprehensive_risk(self, returns: np.ndarray, trades: List[Dict] = None) -> Dict[str, float]:
        """
        综合风险分析
        
        Args:
            returns: 收益率序列
            trades: 交易记录列表（可选）
            
        Returns:
            综合风险指标字典
        """
        if len(returns) == 0:
            return self._empty_risk_metrics()
        
        # 分布统计
        dist_stats = calculate_distribution_stats(returns)
        
        # VaR和CVaR
        risk_metrics = {}
        for conf in self.confidence_levels:
            conf_pct = int(conf * 100)
            risk_metrics[f'var_{conf_pct}'] = calculate_var(returns, conf)
            risk_metrics[f'cvar_{conf_pct}'] = calculate_cvar(returns, conf)
        
        # Sortino比率
        sortino = calculate_sortino_ratio(returns)
        
        # 连续亏损分析
        losing_streaks = analyze_losing_streaks(trades) if trades else {'max_losing_streak': 0, 'max_losing_amount': 0.0}
        
        # 汇总所有风险指标
        comprehensive_metrics = {
            **dist_stats,
            **risk_metrics,
            'sortino_ratio': sortino,
            **losing_streaks
        }
        
        return comprehensive_metrics
    
    def _empty_risk_metrics(self) -> Dict[str, float]:
        """返回空的风险指标字典"""
        metrics = {
            'std': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'sortino_ratio': 0.0,
            'max_losing_streak': 0,
            'max_losing_amount': 0.0
        }
        
        # 添加各置信水平的VaR和CVaR
        for conf in self.confidence_levels:
            conf_pct = int(conf * 100)
            metrics[f'var_{conf_pct}'] = 0.0
            metrics[f'cvar_{conf_pct}'] = 0.0
        
        return metrics
    
    def generate_risk_report(self, returns: np.ndarray, trades: List[Dict] = None) -> str:
        """
        生成风险报告
        
        Args:
            returns: 收益率序列
            trades: 交易记录列表
            
        Returns:
            格式化的风险报告字符串
        """
        metrics = self.analyze_comprehensive_risk(returns, trades)
        
        report = f"""
=== 风险分析报告 ===

1. 分布特征:
   - 收益率标准差: {metrics['std']:.4f}
   - 偏度: {metrics['skewness']:.4f}
   - 峰度: {metrics['kurtosis']:.4f}

2. 尾部风险:
   - VaR (95%): {metrics.get('var_95', 0):.4f}
   - CVaR (95%): {metrics.get('cvar_95', 0):.4f}
   - VaR (99%): {metrics.get('var_99', 0):.4f}
   - CVaR (99%): {metrics.get('cvar_99', 0):.4f}

3. 下行风险:
   - Sortino比率: {metrics['sortino_ratio']:.4f}

4. 连续亏损:
   - 最大连续亏损次数: {metrics['max_losing_streak']}
   - 最大连续亏损金额: {metrics['max_losing_amount']:.2f}
"""
        
        return report