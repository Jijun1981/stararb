"""
配对贡献分析模块
分析各配对对整体策略的贡献和表现

Test: tests/test_backtest_engine.py (配对分析相关测试)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_pair_returns(trades: List[Dict]) -> Dict[str, Dict]:
    """
    TC-BE.4.1: 计算每个配对的总收益和收益占比
    
    Args:
        trades: 交易记录列表
        
    Returns:
        每个配对的收益统计字典
    """
    if not trades:
        return {}
    
    # 按配对分组统计
    pair_stats = {}
    total_pnl = sum(trade['net_pnl'] for trade in trades)
    
    for trade in trades:
        pair = trade['pair']
        pnl = trade['net_pnl']
        
        if pair not in pair_stats:
            pair_stats[pair] = {
                'total_pnl': 0.0,
                'trade_count': 0,
                'wins': 0,
                'losses': 0
            }
        
        pair_stats[pair]['total_pnl'] += pnl
        pair_stats[pair]['trade_count'] += 1
        
        if pnl > 0:
            pair_stats[pair]['wins'] += 1
        else:
            pair_stats[pair]['losses'] += 1
    
    # 计算收益占比
    for pair in pair_stats:
        if total_pnl != 0:
            pair_stats[pair]['pnl_contribution'] = pair_stats[pair]['total_pnl'] / total_pnl
        else:
            pair_stats[pair]['pnl_contribution'] = 0.0
        
        # 计算胜率
        total_trades = pair_stats[pair]['trade_count']
        if total_trades > 0:
            pair_stats[pair]['win_rate'] = pair_stats[pair]['wins'] / total_trades
        else:
            pair_stats[pair]['win_rate'] = 0.0
    
    return pair_stats


def analyze_trading_frequency(trades: List[Dict]) -> Dict[str, Dict]:
    """
    TC-BE.4.2: 统计每个配对的交易频率和平均持仓期
    
    Args:
        trades: 交易记录列表
        
    Returns:
        每个配对的交易频率统计字典
    """
    if not trades:
        return {}
    
    frequency_stats = {}
    
    for trade in trades:
        pair = trade['pair']
        holding_days = trade.get('holding_days', 0)
        
        if pair not in frequency_stats:
            frequency_stats[pair] = {
                'trade_count': 0,
                'total_holding_days': 0,
                'holding_periods': []
            }
        
        frequency_stats[pair]['trade_count'] += 1
        frequency_stats[pair]['total_holding_days'] += holding_days
        frequency_stats[pair]['holding_periods'].append(holding_days)
    
    # 计算平均持仓期
    for pair in frequency_stats:
        stats = frequency_stats[pair]
        if stats['trade_count'] > 0:
            stats['avg_holding_days'] = stats['total_holding_days'] / stats['trade_count']
            stats['median_holding_days'] = np.median(stats['holding_periods'])
            stats['max_holding_days'] = max(stats['holding_periods'])
            stats['min_holding_days'] = min(stats['holding_periods'])
        else:
            stats['avg_holding_days'] = 0
            stats['median_holding_days'] = 0
            stats['max_holding_days'] = 0
            stats['min_holding_days'] = 0
        
        # 删除原始数据以节省内存
        del stats['holding_periods']
    
    return frequency_stats


def calculate_correlation_matrix(pair_returns: Dict[str, List[float]]) -> pd.DataFrame:
    """
    TC-BE.4.3: 计算配对间收益的相关性矩阵
    
    Args:
        pair_returns: 每个配对的日收益序列字典
        
    Returns:
        相关性矩阵DataFrame
    """
    if not pair_returns:
        return pd.DataFrame()
    
    # 转换为DataFrame
    returns_df = pd.DataFrame(pair_returns)
    
    # 计算相关性矩阵
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def rank_pairs(pair_stats: Dict[str, Dict], ranking_metric: str = 'total_pnl') -> List[Tuple[str, float]]:
    """
    TC-BE.4.4: 识别Top5最佳和Bottom5最差配对
    
    Args:
        pair_stats: 配对统计字典
        ranking_metric: 排名指标，默认为总收益
        
    Returns:
        按排名指标排序的配对列表
    """
    if not pair_stats:
        return []
    
    # 提取排名指标
    pair_rankings = []
    for pair, stats in pair_stats.items():
        if ranking_metric in stats:
            pair_rankings.append((pair, stats[ranking_metric]))
        else:
            logger.warning(f"配对 {pair} 缺少排名指标 {ranking_metric}")
    
    # 按指标排序（降序）
    pair_rankings.sort(key=lambda x: x[1], reverse=True)
    
    return pair_rankings


def generate_pairs_report(pair_stats: Dict[str, Dict], frequency_stats: Dict[str, Dict] = None) -> str:
    """
    TC-BE.4.5: 生成配对绩效对比表
    
    Args:
        pair_stats: 配对收益统计
        frequency_stats: 配对频率统计（可选）
        
    Returns:
        格式化的配对报告字符串
    """
    if not pair_stats:
        return "无配对数据可供分析"
    
    # 按总收益排序
    ranked_pairs = rank_pairs(pair_stats, 'total_pnl')
    
    report = "=== 配对绩效分析报告 ===\n\n"
    
    # 表头
    report += f"{'配对名称':<15} {'总收益':<12} {'贡献率':<10} {'胜率':<8} {'交易次数':<8}"
    if frequency_stats:
        report += f" {'平均持仓':<8}"
    report += "\n"
    report += "-" * (15 + 12 + 10 + 8 + 8 + (8 if frequency_stats else 0)) + "\n"
    
    # 配对数据
    for pair, _ in ranked_pairs:
        stats = pair_stats[pair]
        freq_stats = frequency_stats.get(pair, {}) if frequency_stats else {}
        
        report += f"{pair:<15} "
        report += f"{stats['total_pnl']:<12.2f} "
        report += f"{stats['pnl_contribution']:<10.2%} "
        report += f"{stats['win_rate']:<8.2%} "
        report += f"{stats['trade_count']:<8d}"
        
        if frequency_stats and 'avg_holding_days' in freq_stats:
            report += f" {freq_stats['avg_holding_days']:<8.1f}"
        
        report += "\n"
    
    # 汇总统计
    total_pnl = sum(stats['total_pnl'] for stats in pair_stats.values())
    total_trades = sum(stats['trade_count'] for stats in pair_stats.values())
    
    report += "\n=== 汇总统计 ===\n"
    report += f"总收益: {total_pnl:.2f}\n"
    report += f"总交易次数: {total_trades}\n"
    report += f"活跃配对数: {len(pair_stats)}\n"
    
    # Top和Bottom配对
    if len(ranked_pairs) >= 3:
        report += f"\nTop 3 最佳配对:\n"
        for i, (pair, pnl) in enumerate(ranked_pairs[:3]):
            report += f"{i+1}. {pair}: {pnl:.2f}\n"
        
        if len(ranked_pairs) >= 3:
            report += f"\nBottom 3 最差配对:\n"
            for i, (pair, pnl) in enumerate(ranked_pairs[-3:]):
                report += f"{i+1}. {pair}: {pnl:.2f}\n"
    
    return report


# 算法验证函数

def validate_pair_returns_calculation(trades: List[Dict]) -> Tuple[Dict, Dict]:
    """
    配对收益计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现
    returns1 = calculate_pair_returns(trades)
    
    # 方法2: 使用pandas分组验证
    if not trades:
        return {}, {}
    
    trades_df = pd.DataFrame(trades)
    returns2 = {}
    
    for pair, group in trades_df.groupby('pair'):
        total_pnl = group['net_pnl'].sum()
        trade_count = len(group)
        wins = len(group[group['net_pnl'] > 0])
        losses = len(group[group['net_pnl'] <= 0])
        
        returns2[pair] = {
            'total_pnl': total_pnl,
            'trade_count': trade_count,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / trade_count if trade_count > 0 else 0.0,
            'pnl_contribution': total_pnl / trades_df['net_pnl'].sum() if trades_df['net_pnl'].sum() != 0 else 0.0
        }
    
    return returns1, returns2


def validate_frequency_analysis(trades: List[Dict]) -> Tuple[Dict, Dict]:
    """
    交易频率分析的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 主实现
    freq1 = analyze_trading_frequency(trades)
    
    # 方法2: 直接计算验证
    if not trades:
        return {}, {}
    
    freq2 = {}
    
    # 按配对分组
    pairs_data = {}
    for trade in trades:
        pair = trade['pair']
        if pair not in pairs_data:
            pairs_data[pair] = []
        pairs_data[pair].append(trade.get('holding_days', 0))
    
    # 计算统计量
    for pair, holding_days in pairs_data.items():
        freq2[pair] = {
            'trade_count': len(holding_days),
            'total_holding_days': sum(holding_days),
            'avg_holding_days': np.mean(holding_days),
            'median_holding_days': np.median(holding_days),
            'max_holding_days': max(holding_days) if holding_days else 0,
            'min_holding_days': min(holding_days) if holding_days else 0
        }
    
    return freq1, freq2


class PairAnalyzer:
    """配对分析器类"""
    
    def __init__(self):
        """初始化配对分析器"""
        self.pair_stats = {}
        self.frequency_stats = {}
        self.correlation_matrix = pd.DataFrame()
    
    def analyze_all_pairs(self, trades: List[Dict]) -> Dict[str, any]:
        """
        综合分析所有配对
        
        Args:
            trades: 交易记录列表
            
        Returns:
            综合分析结果字典
        """
        if not trades:
            return self._empty_analysis()
        
        # 计算各项指标
        self.pair_stats = calculate_pair_returns(trades)
        self.frequency_stats = analyze_trading_frequency(trades)
        
        # 排名分析
        rankings = rank_pairs(self.pair_stats, 'total_pnl')
        
        # 生成报告
        report = generate_pairs_report(self.pair_stats, self.frequency_stats)
        
        return {
            'pair_stats': self.pair_stats,
            'frequency_stats': self.frequency_stats,
            'rankings': rankings,
            'top_pairs': rankings[:5] if len(rankings) >= 5 else rankings,
            'bottom_pairs': rankings[-5:] if len(rankings) >= 5 else [],
            'report': report
        }
    
    def _empty_analysis(self) -> Dict[str, any]:
        """返回空的分析结果"""
        return {
            'pair_stats': {},
            'frequency_stats': {},
            'rankings': [],
            'top_pairs': [],
            'bottom_pairs': [],
            'report': "无交易数据可供分析"
        }