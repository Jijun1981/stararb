#!/usr/bin/env python3
"""
回测核心功能测试用例
采用TDD方式，先写测试，后写实现
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.backtest_core import (
    calculate_lots,
    calculate_min_lots,
    apply_slippage,
    calculate_pnl,
    calculate_time_weighted_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)


class TestMinLotsCalculation(unittest.TestCase):
    """最小手数计算测试"""
    
    def test_simple_ratios(self):
        """测试简单比例"""
        # β = 0.5 -> Y:X = 1:2
        result = calculate_min_lots(beta=0.5)
        self.assertEqual(result['lots_y'], 1)
        self.assertEqual(result['lots_x'], 2)
        self.assertAlmostEqual(result['actual_ratio'], 0.5, places=3)
        
        # β = 2.0 -> Y:X = 2:1
        result = calculate_min_lots(beta=2.0)
        self.assertEqual(result['lots_y'], 2)
        self.assertEqual(result['lots_x'], 1)
        self.assertAlmostEqual(result['actual_ratio'], 2.0, places=3)
    
    def test_decimal_ratios(self):
        """测试小数比例"""
        # β = 1.5 -> Y:X = 3:2
        result = calculate_min_lots(beta=1.5)
        self.assertEqual(result['lots_y'], 3)
        self.assertEqual(result['lots_x'], 2)
        self.assertAlmostEqual(result['actual_ratio'], 1.5, places=3)
        
        # β = 0.85 -> 接近 6:7
        result = calculate_min_lots(beta=0.85)
        self.assertEqual(result['lots_y'], 6)
        self.assertEqual(result['lots_x'], 7)
        self.assertAlmostEqual(result['actual_ratio'], 0.857, places=2)
    
    def test_extreme_values(self):
        """测试极端值"""
        # 很小的β
        result = calculate_min_lots(beta=0.1)
        self.assertEqual(result['lots_y'], 1)
        self.assertEqual(result['lots_x'], 10)
        
        # 很大的β
        result = calculate_min_lots(beta=10.0)
        self.assertEqual(result['lots_y'], 10)
        self.assertEqual(result['lots_x'], 1)


class TestLotsCalculation(unittest.TestCase):
    """手数计算测试（有资金限制）"""
    
    def test_normal_beta(self):
        """测试正常β值的手数计算"""
        result = calculate_lots(
            beta=0.85,
            available_capital=100000,
            price_y=1000,
            price_x=2000,
            mult_y=10,
            mult_x=5,
            margin_rate=0.12
        )
        
        # 验证手数比例
        self.assertEqual(result['lots_x'], round(result['lots_y'] * 0.85))
        # 验证保证金不超过可用资金
        self.assertLessEqual(result['margin'], 100000)
        # 验证可行性
        self.assertTrue(result['feasible'])
        
    def test_extreme_beta(self):
        """测试极端β值"""
        # 测试很小的β
        result = calculate_lots(
            beta=0.01,
            available_capital=100000,
            price_y=1000,
            price_x=2000,
            mult_y=10,
            mult_x=5
        )
        # β太小时，X手数可能为0，需要特殊处理
        if result['feasible']:
            self.assertGreater(result['lots_x'], 0)
        
        # 测试很大的β
        result = calculate_lots(
            beta=10,
            available_capital=100000,
            price_y=1000,
            price_x=2000,
            mult_y=10,
            mult_x=5
        )
        if result['feasible']:
            self.assertEqual(result['lots_x'], round(result['lots_y'] * 10))
    
    def test_insufficient_capital(self):
        """测试资金不足的情况"""
        result = calculate_lots(
            beta=0.85,
            available_capital=1000,  # 资金很少
            price_y=1000,
            price_x=2000,
            mult_y=10,
            mult_x=5
        )
        self.assertFalse(result['feasible'])
        self.assertEqual(result['lots_y'], 0)
        self.assertEqual(result['lots_x'], 0)


class TestSlippage(unittest.TestCase):
    """滑点计算测试"""
    
    def test_buy_slippage(self):
        """测试买入滑点"""
        actual_price = apply_slippage(
            price=1000,
            side='buy',
            tick_size=1,
            ticks=3
        )
        self.assertEqual(actual_price, 1003)
    
    def test_sell_slippage(self):
        """测试卖出滑点"""
        actual_price = apply_slippage(
            price=1000,
            side='sell',
            tick_size=1,
            ticks=3
        )
        self.assertEqual(actual_price, 997)
    
    def test_different_tick_sizes(self):
        """测试不同tick_size"""
        actual_price = apply_slippage(
            price=5000,
            side='buy',
            tick_size=5,
            ticks=3
        )
        self.assertEqual(actual_price, 5015)


class TestPnLCalculation(unittest.TestCase):
    """PnL计算测试"""
    
    def test_long_spread_profit(self):
        """测试做多价差盈利"""
        position = {
            'direction': 'long',  # 买Y卖X
            'lots_y': 10,
            'lots_x': 8,
            'entry_price_y': 1000,
            'entry_price_x': 2000,
            'margin': 50000,
            'open_commission': 100
        }
        
        result = calculate_pnl(
            position=position,
            exit_price_y=1100,  # Y涨100
            exit_price_x=2050,  # X涨50
            mult_y=10,
            mult_x=5,
            commission_rate=0.0002
        )
        
        # Y腿盈利: (1100-1000) * 10 * 10 = 10000
        # X腿亏损: (2000-2050) * 8 * 5 = -2000
        # 毛利润: 10000 - 2000 = 8000
        self.assertEqual(result['gross_pnl'], 8000)
        
        # 平仓手续费
        close_notional = 1100*10*10 + 2050*8*5
        close_commission = close_notional * 0.0002
        expected_net = 8000 - 100 - close_commission
        self.assertAlmostEqual(result['net_pnl'], expected_net, places=2)
        
        # 收益率
        expected_return = (expected_net / 50000) * 100
        self.assertAlmostEqual(result['return_pct'], expected_return, places=2)
    
    def test_short_spread_loss(self):
        """测试做空价差亏损"""
        position = {
            'direction': 'short',  # 卖Y买X
            'lots_y': 5,
            'lots_x': 4,
            'entry_price_y': 1000,
            'entry_price_x': 2000,
            'margin': 30000,
            'open_commission': 60
        }
        
        result = calculate_pnl(
            position=position,
            exit_price_y=1050,  # Y涨50（不利）
            exit_price_x=1950,  # X跌50（有利）
            mult_y=10,
            mult_x=5,
            commission_rate=0.0002
        )
        
        # Y腿亏损: (1000-1050) * 5 * 10 = -2500
        # X腿盈利: (1950-2000) * 4 * 5 = -1000
        # 毛利润: -2500 + (-1000) = -3500
        self.assertEqual(result['gross_pnl'], -3500)
        
        # 收益率应该是负的
        self.assertLess(result['return_pct'], 0)


class TestTimeWeightedReturn(unittest.TestCase):
    """时间加权收益率测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        trades = [
            {'net_pnl': 10000, 'margin': 100000, 'holding_days': 5},
            {'net_pnl': -5000, 'margin': 50000, 'holding_days': 10},
            {'net_pnl': 8000, 'margin': 80000, 'holding_days': 3}
        ]
        
        result = calculate_time_weighted_return(trades)
        
        # 总PnL: 10000 - 5000 + 8000 = 13000
        self.assertEqual(result['total_pnl'], 13000)
        
        # 保证金天: 100000*5 + 50000*10 + 80000*3 = 1240000
        self.assertEqual(result['margin_days'], 1240000)
        
        # 收益率: 13000/1240000 * 100 = 1.048%
        self.assertAlmostEqual(result['tw_return'], 1.048, places=3)
        
        # 年化收益率
        self.assertAlmostEqual(result['annual_return'], result['daily_return'] * 252, places=3)
    
    def test_empty_trades(self):
        """测试空交易列表"""
        result = calculate_time_weighted_return([])
        self.assertEqual(result['tw_return'], 0)
        self.assertEqual(result['annual_return'], 0)
    
    def test_all_losses(self):
        """测试全部亏损"""
        trades = [
            {'net_pnl': -5000, 'margin': 100000, 'holding_days': 5},
            {'net_pnl': -3000, 'margin': 50000, 'holding_days': 3}
        ]
        
        result = calculate_time_weighted_return(trades)
        
        # 总亏损
        self.assertEqual(result['total_pnl'], -8000)
        # 收益率为负
        self.assertLess(result['tw_return'], 0)


class TestPerformanceMetrics(unittest.TestCase):
    """绩效指标测试"""
    
    def test_sharpe_ratio(self):
        """测试夏普比率计算"""
        # 基于保证金的日收益率序列
        daily_returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.03, -0.02, 0.01]
        
        sharpe = calculate_sharpe_ratio(daily_returns)
        
        # 夏普比率应该是正数（平均收益为正）
        self.assertGreater(sharpe, 0)
        
        # 验证计算逻辑
        import numpy as np
        expected_sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)
    
    def test_max_drawdown(self):
        """测试最大回撤计算"""
        # 累计收益率序列
        cumulative_returns = [0, 0.05, 0.1, 0.08, 0.03, 0.12, 0.09, 0.15]
        
        max_dd, dd_duration = calculate_max_drawdown(cumulative_returns)
        
        # 最大回撤发生在0.1到0.03之间
        # 正确计算: (0.03 - 0.1) / (1 + 0.1) = -0.07/1.1 = -0.0636
        self.assertAlmostEqual(max_dd, -0.0636, places=3)
        
        # 回撤持续时间
        self.assertEqual(dd_duration, 2)  # 从索引2到索引4


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_complete_trade_cycle(self):
        """测试完整的交易周期"""
        # 1. 计算手数
        lots_result = calculate_lots(
            beta=0.5,
            available_capital=100000,
            price_y=1000,
            price_x=2000,
            mult_y=10,
            mult_x=5
        )
        
        self.assertTrue(lots_result['feasible'])
        
        # 2. 应用滑点开仓
        entry_price_y = apply_slippage(1000, 'buy', 1, 3)  # 买Y
        entry_price_x = apply_slippage(2000, 'sell', 1, 3)  # 卖X
        
        # 3. 构建持仓
        position = {
            'direction': 'long',
            'lots_y': lots_result['lots_y'],
            'lots_x': lots_result['lots_x'],
            'entry_price_y': entry_price_y,
            'entry_price_x': entry_price_x,
            'margin': lots_result['margin'],
            'open_commission': (entry_price_y * lots_result['lots_y'] * 10 + 
                              entry_price_x * lots_result['lots_x'] * 5) * 0.0002
        }
        
        # 4. 平仓
        exit_price_y = apply_slippage(1100, 'sell', 1, 3)  # 卖Y
        exit_price_x = apply_slippage(1950, 'buy', 1, 3)  # 买X
        
        pnl_result = calculate_pnl(
            position=position,
            exit_price_y=exit_price_y,
            exit_price_x=exit_price_x,
            mult_y=10,
            mult_x=5
        )
        
        # 5. 验证结果
        self.assertIn('gross_pnl', pnl_result)
        self.assertIn('net_pnl', pnl_result)
        self.assertIn('return_pct', pnl_result)


if __name__ == '__main__':
    unittest.main()