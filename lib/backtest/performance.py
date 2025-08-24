"""
绩效分析模块
负责计算PnL和各种绩效指标
对应需求：REQ-4.4
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class PerformanceAnalyzer:
    """
    绩效分析器
    计算组合和配对级别的全面指标
    """
    
    def __init__(self):
        """初始化绩效分析器"""
        pass
    
    # ========== REQ-4.4.1: 组合级别指标 ==========
    
    def calculate_total_return(self, initial_capital: float, final_capital: float) -> float:
        """
        REQ-4.4.1.1: 计算总收益率
        
        Args:
            initial_capital: 初始资金
            final_capital: 最终资金
            
        Returns:
            总收益率
        """
        if initial_capital <= 0:
            return 0
        return (final_capital - initial_capital) / initial_capital
    
    def calculate_annual_return(self, total_return: float, trading_days: int) -> float:
        """
        REQ-4.4.1.2: 计算年化收益
        
        Args:
            total_return: 总收益率
            trading_days: 交易天数
            
        Returns:
            年化收益率
        """
        if trading_days <= 0:
            return 0
        return (1 + total_return) ** (252 / trading_days) - 1
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        REQ-4.4.1.3: 计算夏普比率
        
        Args:
            returns: 日收益率序列
            
        Returns:
            夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        REQ-4.4.1.4: 计算Sortino比率
        
        Args:
            returns: 日收益率序列
            
        Returns:
            Sortino比率
        """
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # 没有下行风险
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        
        return returns.mean() / downside_std * np.sqrt(252)
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        REQ-4.4.1.5: 计算最大回撤
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            最大回撤（0-1之间）
        """
        if len(equity_curve) == 0:
            return 0
        
        # 计算累计最高值
        cummax = equity_curve.cummax()
        
        # 计算回撤
        drawdown = (equity_curve - cummax) / cummax
        
        # 返回最大回撤（绝对值）
        return abs(drawdown.min())
    
    def calculate_win_rate(self, trades: List[Any]) -> float:
        """
        REQ-4.4.1.6: 计算胜率
        
        Args:
            trades: 交易列表
            
        Returns:
            胜率（0-1之间）
        """
        if len(trades) == 0:
            return 0
        
        winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        return winning_trades / len(trades)
    
    def calculate_profit_factor(self, trades: List[Any]) -> float:
        """
        REQ-4.4.1.7: 计算盈亏比
        
        Args:
            trades: 交易列表
            
        Returns:
            盈亏比
        """
        if len(trades) == 0:
            return 0
        
        wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        losses = sum(t.net_pnl for t in trades if t.net_pnl < 0)
        
        if losses == 0:
            return float('inf') if wins > 0 else 0
        
        return wins / abs(losses)
    
    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """
        REQ-4.4.1.8: 计算Calmar比率
        
        Args:
            annual_return: 年化收益率
            max_drawdown: 最大回撤
            
        Returns:
            Calmar比率
        """
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0
        return annual_return / max_drawdown
    
    # ========== REQ-4.4.2: 配对级别指标 ==========
    
    def calculate_pair_metrics(self, pair: str, trades: List[Any]) -> Dict[str, Any]:
        """
        REQ-4.4.2.1: 计算配对级别指标
        
        Args:
            pair: 配对名称
            trades: 所有交易列表
            
        Returns:
            配对指标字典
        """
        # 筛选该配对的交易
        pair_trades = [t for t in trades if t.pair == pair]
        
        if len(pair_trades) == 0:
            return self._empty_pair_metrics(pair)
        
        # REQ-4.4.2.2: 统计交易次数和持仓天数
        num_trades = len(pair_trades)
        avg_holding_days = sum(t.holding_days for t in pair_trades) / num_trades
        
        # 计算PnL
        total_pnl = sum(t.net_pnl for t in pair_trades)
        
        # 计算收益率
        if num_trades > 0:
            avg_return = sum(t.return_pct for t in pair_trades) / num_trades
        else:
            avg_return = 0
        
        # 计算胜率
        win_rate = self.calculate_win_rate(pair_trades)
        
        # REQ-4.4.2.4: 统计止损
        stop_loss_trades = [t for t in pair_trades if t.close_reason == 'stop_loss']
        stop_loss_count = len(stop_loss_trades)
        stop_loss_pnl = sum(t.net_pnl for t in stop_loss_trades)
        
        # 时间止损统计
        time_stop_trades = [t for t in pair_trades if t.close_reason == 'time_stop']
        time_stop_count = len(time_stop_trades)
        
        # REQ-4.4.2.5: 平均手数
        avg_lots_x = sum(t.lots_x for t in pair_trades) / num_trades
        avg_lots_y = sum(t.lots_y for t in pair_trades) / num_trades
        
        # 生成简单的权益曲线用于计算指标
        pair_equity = self.generate_pair_equity_curve(pair, trades, 100000)
        pair_returns = pair_equity.pct_change().dropna()
        
        # 计算风险指标
        sharpe_ratio = self.calculate_sharpe_ratio(pair_returns) if len(pair_returns) > 0 else 0
        sortino_ratio = self.calculate_sortino_ratio(pair_returns) if len(pair_returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(pair_equity)
        
        return {
            'pair': pair,
            'num_trades': num_trades,
            'avg_holding_days': avg_holding_days,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / num_trades if num_trades > 0 else 0,
            'total_return': total_pnl / 100000,  # 基于假设的初始资金
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'stop_loss_count': stop_loss_count,
            'stop_loss_pnl': stop_loss_pnl,
            'time_stop_count': time_stop_count,
            'avg_lots_x': avg_lots_x,
            'avg_lots_y': avg_lots_y
        }
    
    def analyze_all_pairs(self, trades: List[Any]) -> pd.DataFrame:
        """
        分析所有配对的表现
        
        Args:
            trades: 交易列表
            
        Returns:
            配对分析DataFrame
        """
        if len(trades) == 0:
            return pd.DataFrame()
        
        # 获取所有配对
        pairs = list(set(t.pair for t in trades))
        
        # 计算每个配对的指标
        pair_metrics_list = []
        for pair in pairs:
            metrics = self.calculate_pair_metrics(pair, trades)
            pair_metrics_list.append(metrics)
        
        # 转换为DataFrame
        df = pd.DataFrame(pair_metrics_list)
        
        # REQ-4.4.2.3: 计算贡献度
        total_pnl = sum(t.net_pnl for t in trades)
        if total_pnl != 0:
            df['contribution'] = df['total_pnl'] / total_pnl
        else:
            df['contribution'] = 0
        
        return df.sort_values('total_pnl', ascending=False)
    
    def generate_pair_equity_curve(
        self,
        pair: str,
        trades: List[Any],
        initial_capital: float
    ) -> pd.Series:
        """
        REQ-4.4.2.6: 生成配对权益曲线
        
        Args:
            pair: 配对名称
            trades: 交易列表
            initial_capital: 初始资金
            
        Returns:
            权益曲线Series
        """
        pair_trades = [t for t in trades if t.pair == pair]
        
        if len(pair_trades) == 0:
            # 返回平坦的权益曲线
            return pd.Series([initial_capital, initial_capital])
        
        # 按平仓时间排序
        pair_trades.sort(key=lambda x: x.close_date)
        
        # 构建权益曲线
        equity = [initial_capital]
        current_equity = initial_capital
        
        for trade in pair_trades:
            current_equity += trade.net_pnl
            equity.append(current_equity)
        
        return pd.Series(equity)
    
    # ========== REQ-4.4.3: 交易明细（已通过Trade类实现） ==========
    
    def calculate_portfolio_metrics(
        self,
        trades: List[Any],
        equity_curve: pd.Series,
        daily_returns: pd.Series
    ) -> Dict[str, float]:
        """
        计算组合级别的所有指标
        
        Args:
            trades: 交易列表
            equity_curve: 权益曲线
            daily_returns: 日收益率序列
            
        Returns:
            组合指标字典
        """
        if len(trades) == 0:
            return self._empty_portfolio_metrics()
        
        # 基本统计
        total_trades = len(trades)
        total_pnl = sum(t.net_pnl for t in trades)
        
        # 收益指标
        initial = equity_curve.iloc[0] if len(equity_curve) > 0 else 0
        final = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0
        total_return = self.calculate_total_return(initial, final)
        
        trading_days = len(equity_curve) if len(equity_curve) > 0 else 1
        annual_return = self.calculate_annual_return(total_return, trading_days)
        
        # 风险指标
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self.calculate_sortino_ratio(daily_returns)
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        # 交易统计
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        
        # 平均盈亏
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        
        avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # 最大连续亏损
        max_consecutive_losses = self._calculate_max_consecutive_losses(trades)
        
        # Calmar比率
        calmar_ratio = self.calculate_calmar_ratio(annual_return, max_drawdown)
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_losses': max_consecutive_losses,
            'calmar_ratio': calmar_ratio
        }
    
    def generate_report(
        self,
        trades: List[Any],
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        生成完整的绩效报告
        
        Args:
            trades: 交易列表
            equity_curve: 权益曲线
            initial_capital: 初始资金
            
        Returns:
            完整报告字典
        """
        # 计算日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 组合级别指标
        portfolio_metrics = self.calculate_portfolio_metrics(
            trades, equity_curve, daily_returns
        )
        
        # 配对级别分析
        pair_metrics = self.analyze_all_pairs(trades)
        
        # 交易摘要
        trade_summary = {
            'total_trades': len(trades),
            'unique_pairs': len(pair_metrics) if isinstance(pair_metrics, pd.DataFrame) else 0,
            'avg_holding_days': sum(t.holding_days for t in trades) / len(trades) if trades else 0,
            'stop_loss_trades': sum(1 for t in trades if t.close_reason == 'stop_loss'),
            'time_stop_trades': sum(1 for t in trades if t.close_reason == 'time_stop'),
            'signal_closes': sum(1 for t in trades if t.close_reason == 'signal')
        }
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'pair_metrics': pair_metrics,
            'trade_summary': trade_summary,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _empty_portfolio_metrics(self) -> Dict[str, float]:
        """返回空的组合指标"""
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_consecutive_losses': 0,
            'calmar_ratio': 0
        }
    
    def _empty_pair_metrics(self, pair: str) -> Dict[str, Any]:
        """返回空的配对指标"""
        return {
            'pair': pair,
            'num_trades': 0,
            'avg_holding_days': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'total_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'stop_loss_count': 0,
            'stop_loss_pnl': 0,
            'time_stop_count': 0,
            'avg_lots_x': 0,
            'avg_lots_y': 0
        }
    
    def _calculate_max_consecutive_losses(self, trades: List[Any]) -> int:
        """计算最大连续亏损次数"""
        if len(trades) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.net_pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive