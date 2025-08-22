#!/usr/bin/env python3
"""
回测核心计算模块（原子服务）
所有收益率以保证金为口径计算
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def calculate_lots(beta: float, available_capital: float,
                   price_y: float, price_x: float,
                   mult_y: int, mult_x: int,
                   margin_rate: float = 0.12) -> Dict:
    """
    根据β值计算最优手数
    
    Args:
        beta: 动态β值（来自信号）
        available_capital: 可用资金
        price_y: Y品种价格
        price_x: X品种价格
        mult_y: Y品种合约乘数
        mult_x: X品种合约乘数
        margin_rate: 保证金率（默认12%）
    
    Returns:
        {
            'lots_y': int,  # Y品种手数
            'lots_x': int,  # X品种手数
            'margin': float,  # 占用保证金
            'feasible': bool,  # 是否可行
            'theoretical_ratio': float,  # 理论比例
            'actual_ratio': float  # 实际比例
        }
    """
    # 处理β的绝对值
    beta_abs = abs(beta)
    
    # 计算单手保证金
    margin_per_y = price_y * mult_y * margin_rate
    margin_per_x = price_x * mult_x * margin_rate
    
    # 搜索最大可行Y手数
    max_lots_y = 0
    best_lots_x = 0
    best_margin = 0
    
    for lots_y in range(1, 100):  # 限制最大100手
        # 根据β计算X手数（四舍五入）
        lots_x = round(lots_y * beta_abs)
        
        # 处理X手数为0的情况
        if lots_x == 0 and beta_abs > 0.1:
            lots_x = 1  # 至少1手
        elif lots_x == 0:
            continue  # β太小，跳过
        
        # 计算总保证金需求
        total_margin = lots_y * margin_per_y + lots_x * margin_per_x
        
        # 检查是否超出可用资金
        if total_margin > available_capital:
            break  # 超出预算，停止搜索
        
        # 更新最优解
        max_lots_y = lots_y
        best_lots_x = lots_x
        best_margin = total_margin
    
    # 返回结果
    return {
        'lots_y': max_lots_y,
        'lots_x': best_lots_x,
        'margin': best_margin,
        'feasible': max_lots_y > 0,
        'theoretical_ratio': beta_abs,
        'actual_ratio': best_lots_x / max_lots_y if max_lots_y > 0 else 0
    }


def calculate_min_lots(beta: float, max_denominator: int = 10) -> Dict:
    """
    根据β值计算最小整数比手数（无资金限制版本）
    
    Args:
        beta: β值（Y/X的比例）
        max_denominator: 最大分母限制（默认10）
    
    Returns:
        {
            'lots_y': int,  # Y品种手数
            'lots_x': int,  # X品种手数
            'theoretical_ratio': float,  # 理论比例
            'actual_ratio': float  # 实际比例
        }
    """
    from fractions import Fraction
    
    # 处理特殊情况
    if beta <= 0:
        return {
            'lots_y': 1,
            'lots_x': 1,
            'theoretical_ratio': abs(beta),
            'actual_ratio': 1.0
        }
    
    beta_abs = abs(beta)
    
    # 使用分数类找最简分数
    frac = Fraction(beta_abs).limit_denominator(max_denominator)
    
    # β = lots_y / lots_x
    lots_y = frac.numerator
    lots_x = frac.denominator
    
    # 确保至少1手
    if lots_y == 0:
        lots_y = 1
    if lots_x == 0:
        lots_x = 1
    
    return {
        'lots_y': lots_y,
        'lots_x': lots_x,
        'theoretical_ratio': beta_abs,
        'actual_ratio': lots_y / lots_x
    }


def apply_slippage(price: float, side: str, tick_size: float, ticks: int = 3) -> float:
    """
    计算滑点后的成交价
    
    Args:
        price: 市场价格
        side: 'buy' 或 'sell'
        tick_size: 最小变动价位
        ticks: 滑点tick数（默认3）
    
    Returns:
        滑点后的成交价
    """
    if side == 'buy':
        return price + tick_size * ticks
    else:  # sell
        return price - tick_size * ticks


def calculate_pnl(position: Dict, exit_price_y: float, exit_price_x: float,
                  mult_y: int, mult_x: int,
                  commission_rate: float = 0.0002) -> Dict:
    """
    计算平仓PnL（以保证金为口径）
    
    Args:
        position: 持仓信息字典，包含：
            - direction: 'long' or 'short'
            - lots_y: Y品种手数
            - lots_x: X品种手数
            - entry_price_y: Y入场价
            - entry_price_x: X入场价
            - margin: 占用保证金
            - open_commission: 开仓手续费
        exit_price_y: Y品种出场价（含滑点）
        exit_price_x: X品种出场价（含滑点）
        mult_y: Y品种合约乘数
        mult_x: X品种合约乘数
        commission_rate: 手续费率（默认万分之2）
    
    Returns:
        {
            'gross_pnl': float,  # 毛利润
            'net_pnl': float,  # 净利润
            'return_pct': float,  # 收益率（基于保证金）
            'pnl_y': float,  # Y腿盈亏
            'pnl_x': float  # X腿盈亏
        }
    """
    # 根据方向计算各腿PnL
    if position['direction'] == 'long':  # 做多价差（买Y卖X）
        pnl_y = (exit_price_y - position['entry_price_y']) * position['lots_y'] * mult_y
        pnl_x = (position['entry_price_x'] - exit_price_x) * position['lots_x'] * mult_x
    else:  # 做空价差（卖Y买X）
        pnl_y = (position['entry_price_y'] - exit_price_y) * position['lots_y'] * mult_y
        pnl_x = (exit_price_x - position['entry_price_x']) * position['lots_x'] * mult_x
    
    # 计算毛利润
    gross_pnl = pnl_y + pnl_x
    
    # 计算平仓手续费
    close_notional = (exit_price_y * position['lots_y'] * mult_y +
                     exit_price_x * position['lots_x'] * mult_x)
    close_commission = close_notional * commission_rate
    
    # 计算净利润
    net_pnl = gross_pnl - position['open_commission'] - close_commission
    
    # 计算收益率（基于保证金）
    return_pct = (net_pnl / position['margin']) * 100 if position['margin'] > 0 else 0
    
    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'return_pct': return_pct,
        'pnl_y': pnl_y,
        'pnl_x': pnl_x,
        'close_commission': close_commission
    }


def calculate_time_weighted_return(trades: List[Dict]) -> Dict:
    """
    计算时间加权总收益率
    
    公式：收益率 = Σ(净PnL) / Σ(保证金×持仓天数)
    
    Args:
        trades: 交易列表，每个交易包含：
            - net_pnl: 净利润
            - margin: 占用保证金
            - holding_days: 持仓天数
    
    Returns:
        {
            'total_pnl': float,  # 总PnL
            'margin_days': float,  # 保证金·天
            'tw_return': float,  # 时间加权收益率(%)
            'daily_return': float,  # 日均收益率(%)
            'annual_return': float,  # 年化收益率(%)
        }
    """
    if not trades:
        return {
            'total_pnl': 0,
            'margin_days': 0,
            'tw_return': 0,
            'daily_return': 0,
            'annual_return': 0
        }
    
    # 计算总PnL和保证金·天
    total_pnl = sum(t['net_pnl'] for t in trades)
    total_margin_days = sum(t['margin'] * t['holding_days'] for t in trades)
    
    if total_margin_days == 0:
        return {
            'total_pnl': total_pnl,
            'margin_days': 0,
            'tw_return': 0,
            'daily_return': 0,
            'annual_return': 0
        }
    
    # 时间加权收益率（每万元·天的收益率）
    tw_return = (total_pnl / total_margin_days) * 100
    
    # 日均收益率（tw_return已经是日均）
    daily_return = tw_return
    
    # 年化收益率
    annual_return = daily_return * 252
    
    return {
        'total_pnl': total_pnl,
        'margin_days': total_margin_days,
        'tw_return': tw_return,
        'daily_return': daily_return,
        'annual_return': annual_return
    }


def calculate_sharpe_ratio(daily_returns: List[float], risk_free_rate: float = 0) -> float:
    """
    计算夏普比率（基于保证金收益率）
    
    Args:
        daily_returns: 日收益率序列（基于保证金）
        risk_free_rate: 无风险收益率（默认0）
    
    Returns:
        年化夏普比率
    """
    if not daily_returns or len(daily_returns) < 2:
        return 0
    
    returns_array = np.array(daily_returns)
    
    # 计算超额收益
    excess_returns = returns_array - risk_free_rate
    
    # 计算平均超额收益和标准差
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return 0
    
    # 年化夏普比率
    sharpe = mean_excess / std_excess * np.sqrt(252)
    
    return sharpe


def calculate_max_drawdown(cumulative_returns: List[float]) -> Tuple[float, int]:
    """
    计算最大回撤（基于累计收益率序列）
    
    Args:
        cumulative_returns: 累计收益率序列
    
    Returns:
        (最大回撤率, 回撤持续期)
    """
    if not cumulative_returns or len(cumulative_returns) < 2:
        return 0, 0
    
    # 转换为numpy数组
    cum_returns = np.array(cumulative_returns)
    
    # 计算累计最高值
    running_max = np.maximum.accumulate(cum_returns)
    
    # 计算回撤
    drawdowns = (cum_returns - running_max) / (1 + running_max)
    
    # 找出最大回撤
    max_drawdown = np.min(drawdowns)
    
    # 计算回撤持续期
    if max_drawdown == 0:
        duration = 0
    else:
        # 找到最大回撤的位置
        max_dd_idx = np.argmin(drawdowns)
        # 找到之前的最高点
        prev_max_idx = np.where(cum_returns[:max_dd_idx+1] == running_max[max_dd_idx])[0]
        if len(prev_max_idx) > 0:
            duration = max_dd_idx - prev_max_idx[0]
        else:
            duration = 0
    
    return max_drawdown, duration


def calculate_win_rate_metrics(trades: List[Dict]) -> Dict:
    """
    计算胜率和盈亏比
    
    Args:
        trades: 交易列表，每个交易包含net_pnl
    
    Returns:
        {
            'win_rate': float,  # 胜率(%)
            'profit_loss_ratio': float,  # 盈亏比
            'avg_win': float,  # 平均盈利
            'avg_loss': float,  # 平均亏损
            'total_trades': int,  # 总交易数
            'winning_trades': int,  # 盈利交易数
            'losing_trades': int  # 亏损交易数
        }
    """
    if not trades:
        return {
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    # 分离盈利和亏损交易
    wins = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
    losses = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
    
    # 计算统计指标
    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    # 胜率
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 平均盈亏
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # 盈亏比
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    return {
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades
    }


def check_stop_loss(position: Dict, current_pnl: float, stop_loss_pct: float = 0.1) -> bool:
    """
    检查是否触发止损（基于保证金）
    
    Args:
        position: 持仓信息
        current_pnl: 当前浮动盈亏
        stop_loss_pct: 止损比例（默认10%）
    
    Returns:
        是否触发止损
    """
    if current_pnl >= 0:
        return False
    
    # 计算亏损比例
    loss_pct = abs(current_pnl) / position['margin']
    
    # 检查是否超过止损线
    return loss_pct >= stop_loss_pct


def check_time_stop(holding_days: int, max_days: int = 30) -> bool:
    """
    检查是否触发时间止损
    
    Args:
        holding_days: 持仓天数
        max_days: 最大持仓天数（默认30天）
    
    Returns:
        是否触发时间止损
    """
    return holding_days >= max_days