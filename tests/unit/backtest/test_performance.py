"""
绩效分析模块测试用例
严格对应需求文档 REQ-4.4
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lib.backtest.performance import PerformanceAnalyzer
from lib.backtest.trade_executor import Trade


def create_mock_trade(
    pair='CU-SN',
    net_pnl=1000,
    holding_days=5,
    close_reason='signal',
    open_date=None,
    close_date=None
) -> Trade:
    """创建模拟交易记录"""
    if open_date is None:
        open_date = datetime(2024, 1, 1)
    if close_date is None:
        close_date = open_date + timedelta(days=holding_days)
    
    return Trade(
        trade_id=f'T{np.random.randint(10000)}',
        position_id=f'P{np.random.randint(10000)}',
        pair=pair,
        symbol_x=pair.split('-')[0],
        symbol_y=pair.split('-')[1],
        lots_x=2,
        lots_y=5,
        direction='long',
        open_date=open_date,
        open_price_x=60000,
        open_price_y=140000,
        open_commission=260,
        close_date=close_date,
        close_price_x=61000,
        close_price_y=139000,
        close_commission=260,
        gross_pnl=net_pnl + 520,
        net_pnl=net_pnl,
        return_pct=net_pnl / 156000,  # 基于保证金
        holding_days=holding_days,
        close_reason=close_reason,
        margin_released=156000,
        capital_change=net_pnl + 156000
    )


class TestPerformanceAnalyzer(unittest.TestCase):
    """测试绩效分析模块"""
    
    def setUp(self):
        """初始化测试环境"""
        self.analyzer = PerformanceAnalyzer()
        
        # 创建测试交易数据
        self.trades = [
            create_mock_trade('CU-SN', 5000, 5),
            create_mock_trade('CU-SN', -2000, 3),
            create_mock_trade('AL-ZN', 3000, 7),
            create_mock_trade('AL-ZN', 4000, 4),
            create_mock_trade('NI-SS', -1500, 2, 'stop_loss'),
            create_mock_trade('NI-SS', 2500, 30, 'time_stop'),
        ]
        
        # 创建测试权益曲线
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = np.random.randn(100) * 0.01
        self.equity_curve = pd.Series(
            5000000 * (1 + returns).cumprod(),
            index=dates
        )
        
        # 创建日收益率序列
        self.daily_returns = self.equity_curve.pct_change().dropna()
    
    # ========== REQ-4.4.1: 组合级别指标 ==========
    
    def test_req_4_4_1_1_total_return(self):
        """REQ-4.4.1.1: 总收益率 = (final - initial) / initial"""
        initial = 5000000
        final = 5500000
        
        total_return = self.analyzer.calculate_total_return(initial, final)
        
        expected = (final - initial) / initial
        self.assertAlmostEqual(total_return, expected, places=6)
        self.assertEqual(total_return, 0.1)  # 10%收益
    
    def test_req_4_4_1_2_annual_return(self):
        """REQ-4.4.1.2: 年化收益 = (1 + total_return)^(252/days) - 1"""
        total_return = 0.2  # 20%总收益
        trading_days = 100
        
        annual_return = self.analyzer.calculate_annual_return(
            total_return, trading_days
        )
        
        expected = (1 + total_return) ** (252 / trading_days) - 1
        self.assertAlmostEqual(annual_return, expected, places=6)
    
    def test_req_4_4_1_3_sharpe_ratio(self):
        """REQ-4.4.1.3: 夏普比率 = mean(returns) / std(returns) × sqrt(252)"""
        sharpe = self.analyzer.calculate_sharpe_ratio(self.daily_returns)
        
        # 验证计算逻辑
        expected_sharpe = (
            self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252)
        )
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)
    
    def test_req_4_4_1_4_sortino_ratio(self):
        """REQ-4.4.1.4: Sortino比率 = mean(returns) / downside_std × sqrt(252)"""
        sortino = self.analyzer.calculate_sortino_ratio(self.daily_returns)
        
        # 验证只考虑下行波动
        downside_returns = self.daily_returns[self.daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            expected_sortino = (
                self.daily_returns.mean() / downside_std * np.sqrt(252)
            )
            self.assertAlmostEqual(sortino, expected_sortino, places=4)
    
    def test_req_4_4_1_5_max_drawdown(self):
        """REQ-4.4.1.5: 最大回撤 = max(peak - trough) / peak"""
        max_dd = self.analyzer.calculate_max_drawdown(self.equity_curve)
        
        # 验证在0-1之间
        self.assertGreaterEqual(max_dd, 0)
        self.assertLessEqual(max_dd, 1)
        
        # 测试无回撤情况
        rising_curve = pd.Series([100, 110, 120, 130, 140])
        max_dd_rising = self.analyzer.calculate_max_drawdown(rising_curve)
        self.assertEqual(max_dd_rising, 0)
        
        # 测试有回撤情况
        drawdown_curve = pd.Series([100, 120, 90, 110, 105])
        max_dd_drawdown = self.analyzer.calculate_max_drawdown(drawdown_curve)
        expected_dd = (120 - 90) / 120  # 25%回撤
        self.assertAlmostEqual(max_dd_drawdown, expected_dd, places=6)
    
    def test_req_4_4_1_6_win_rate(self):
        """REQ-4.4.1.6: 胜率 = winning_trades / total_trades"""
        win_rate = self.analyzer.calculate_win_rate(self.trades)
        
        # 6笔交易中4笔盈利
        winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        expected_rate = winning_trades / len(self.trades)
        self.assertAlmostEqual(win_rate, expected_rate, places=6)
    
    def test_req_4_4_1_7_profit_factor(self):
        """REQ-4.4.1.7: 盈亏比 = sum(wins) / abs(sum(losses))"""
        profit_factor = self.analyzer.calculate_profit_factor(self.trades)
        
        wins = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        losses = sum(t.net_pnl for t in self.trades if t.net_pnl < 0)
        
        if losses != 0:
            expected_factor = wins / abs(losses)
            self.assertAlmostEqual(profit_factor, expected_factor, places=6)
    
    def test_req_4_4_1_8_calmar_ratio(self):
        """REQ-4.4.1.8: Calmar比率 = annual_return / max_drawdown"""
        annual_return = 0.25  # 25%年化收益
        max_drawdown = 0.10   # 10%最大回撤
        
        calmar = self.analyzer.calculate_calmar_ratio(
            annual_return, max_drawdown
        )
        
        expected = annual_return / max_drawdown if max_drawdown > 0 else 0
        self.assertAlmostEqual(calmar, expected, places=6)
        self.assertEqual(calmar, 2.5)
    
    # ========== REQ-4.4.2: 配对级别指标 ==========
    
    def test_req_4_4_2_1_pair_level_metrics(self):
        """REQ-4.4.2.1: 每个配对独立计算所有组合级别指标"""
        pair_metrics = self.analyzer.calculate_pair_metrics('CU-SN', self.trades)
        
        # 验证包含所有必要指标
        self.assertIn('total_pnl', pair_metrics)
        self.assertIn('total_return', pair_metrics)
        self.assertIn('sharpe_ratio', pair_metrics)
        self.assertIn('sortino_ratio', pair_metrics)
        self.assertIn('max_drawdown', pair_metrics)
        self.assertIn('win_rate', pair_metrics)
    
    def test_req_4_4_2_2_trade_statistics(self):
        """REQ-4.4.2.2: 统计配对交易次数、平均持仓天数"""
        pair_metrics = self.analyzer.calculate_pair_metrics('CU-SN', self.trades)
        
        # CU-SN有2笔交易
        self.assertEqual(pair_metrics['num_trades'], 2)
        
        # 平均持仓天数 = (5 + 3) / 2 = 4
        self.assertEqual(pair_metrics['avg_holding_days'], 4)
    
    def test_req_4_4_2_3_contribution(self):
        """REQ-4.4.2.3: 计算配对贡献度 = pair_pnl / total_pnl"""
        all_pairs_metrics = self.analyzer.analyze_all_pairs(self.trades)
        
        total_pnl = sum(t.net_pnl for t in self.trades)
        
        for _, row in all_pairs_metrics.iterrows():
            if total_pnl != 0:
                expected_contribution = row['total_pnl'] / total_pnl
                self.assertAlmostEqual(
                    row['contribution'], 
                    expected_contribution, 
                    places=6
                )
    
    def test_req_4_4_2_4_stop_loss_stats(self):
        """REQ-4.4.2.4: 统计止损次数、止损损失"""
        pair_metrics = self.analyzer.calculate_pair_metrics('NI-SS', self.trades)
        
        # NI-SS有1次止损
        self.assertEqual(pair_metrics['stop_loss_count'], 1)
        self.assertEqual(pair_metrics['stop_loss_pnl'], -1500)
    
    def test_req_4_4_2_5_avg_lots_and_beta(self):
        """REQ-4.4.2.5: 记录平均手数、平均β值"""
        pair_metrics = self.analyzer.calculate_pair_metrics('CU-SN', self.trades)
        
        # 所有CU-SN交易都是2手X，5手Y
        self.assertEqual(pair_metrics['avg_lots_x'], 2)
        self.assertEqual(pair_metrics['avg_lots_y'], 5)
    
    def test_req_4_4_2_6_pair_equity_curve(self):
        """REQ-4.4.2.6: 生成配对权益曲线"""
        pair_equity = self.analyzer.generate_pair_equity_curve(
            'CU-SN', 
            self.trades,
            initial_capital=100000
        )
        
        # 验证权益曲线
        self.assertIsInstance(pair_equity, pd.Series)
        self.assertEqual(pair_equity.iloc[0], 100000)  # 初始资金
        
        # 最终权益 = 初始 + CU-SN的总PnL
        cu_sn_pnl = sum(t.net_pnl for t in self.trades if t.pair == 'CU-SN')
        expected_final = 100000 + cu_sn_pnl
        self.assertEqual(pair_equity.iloc[-1], expected_final)
    
    # ========== REQ-4.4.3: 交易明细 ==========
    
    def test_req_4_4_3_1_complete_trade_info(self):
        """REQ-4.4.3.1: 记录每笔交易完整信息"""
        trade = self.trades[0]
        
        # 验证交易记录包含所有必要字段
        self.assertIsNotNone(trade.trade_id)
        self.assertIsNotNone(trade.position_id)
        self.assertIsNotNone(trade.pair)
    
    def test_req_4_4_3_2_time_info(self):
        """REQ-4.4.3.2: 包含开仓时间、平仓时间、持仓天数"""
        trade = self.trades[0]
        
        self.assertIsInstance(trade.open_date, datetime)
        self.assertIsInstance(trade.close_date, datetime)
        self.assertGreater(trade.holding_days, 0)
    
    def test_req_4_4_3_3_pair_and_lots(self):
        """REQ-4.4.3.3: 包含配对名称、方向、手数"""
        trade = self.trades[0]
        
        self.assertEqual(trade.pair, 'CU-SN')
        self.assertIn(trade.direction, ['long', 'short'])
        self.assertGreater(trade.lots_x, 0)
        self.assertGreater(trade.lots_y, 0)
    
    def test_req_4_4_3_4_prices_and_costs(self):
        """REQ-4.4.3.4: 包含开仓价、平仓价、滑点、手续费"""
        trade = self.trades[0]
        
        self.assertGreater(trade.open_price_x, 0)
        self.assertGreater(trade.open_price_y, 0)
        self.assertGreater(trade.close_price_x, 0)
        self.assertGreater(trade.close_price_y, 0)
        self.assertGreater(trade.open_commission, 0)
        self.assertGreater(trade.close_commission, 0)
    
    def test_req_4_4_3_5_pnl_info(self):
        """REQ-4.4.3.5: 包含毛利润、净利润、收益率"""
        trade = self.trades[0]
        
        self.assertIsNotNone(trade.gross_pnl)
        self.assertIsNotNone(trade.net_pnl)
        self.assertIsNotNone(trade.return_pct)
        
        # 净利润应该小于等于毛利润（扣除成本）
        self.assertLessEqual(trade.net_pnl, trade.gross_pnl)
    
    def test_req_4_4_3_6_close_reason(self):
        """REQ-4.4.3.6: 包含平仓原因"""
        reasons = [trade.close_reason for trade in self.trades]
        
        # 验证包含各种平仓原因
        self.assertIn('signal', reasons)
        self.assertIn('stop_loss', reasons)
        self.assertIn('time_stop', reasons)
    
    # ========== 综合测试 ==========
    
    def test_portfolio_metrics_calculation(self):
        """测试组合级别指标计算"""
        metrics = self.analyzer.calculate_portfolio_metrics(
            self.trades,
            self.equity_curve,
            self.daily_returns
        )
        
        # 验证返回所有必要指标
        required_metrics = [
            'total_return', 'annual_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'total_trades', 'avg_win',
            'avg_loss', 'max_consecutive_losses'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
    
    def test_generate_report(self):
        """测试生成完整报告"""
        report = self.analyzer.generate_report(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=5000000
        )
        
        # 验证报告结构
        self.assertIn('portfolio_metrics', report)
        self.assertIn('pair_metrics', report)
        self.assertIn('trade_summary', report)
        self.assertIn('equity_curve', report)
        
        # 验证配对分析
        pair_df = report['pair_metrics']
        unique_pairs = set(t.pair for t in self.trades)
        self.assertEqual(len(pair_df), len(unique_pairs))
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 空交易列表
        empty_metrics = self.analyzer.calculate_portfolio_metrics(
            [], self.equity_curve, self.daily_returns
        )
        self.assertEqual(empty_metrics['total_trades'], 0)
        self.assertEqual(empty_metrics['win_rate'], 0)
        
        # 全部盈利
        winning_trades = [create_mock_trade(net_pnl=1000) for _ in range(5)]
        win_metrics = self.analyzer.calculate_portfolio_metrics(
            winning_trades, self.equity_curve, self.daily_returns
        )
        self.assertEqual(win_metrics['win_rate'], 1.0)
        
        # 全部亏损
        losing_trades = [create_mock_trade(net_pnl=-1000) for _ in range(5)]
        lose_metrics = self.analyzer.calculate_portfolio_metrics(
            losing_trades, self.equity_curve, self.daily_returns
        )
        self.assertEqual(lose_metrics['win_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()