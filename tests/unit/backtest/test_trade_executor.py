"""
交易执行模块测试用例
严格对应需求文档 REQ-4.2
"""

import unittest
from datetime import datetime
from dataclasses import dataclass
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lib.backtest.trade_executor import TradeExecutor, ExecutionConfig, Position, Trade


class TestTradeExecutor(unittest.TestCase):
    """测试交易执行模块"""
    
    def setUp(self):
        """初始化测试配置"""
        self.config = ExecutionConfig(
            commission_rate=0.0002,  # 万分之2
            slippage_ticks=3,        # 3个tick滑点
            margin_rate=0.12         # 12%保证金率
        )
        self.executor = TradeExecutor(self.config)
        
        # 测试数据
        self.pair_info = {
            'pair': 'CU-SN',
            'symbol_x': 'CU',
            'symbol_y': 'SN',
            'beta': 0.85
        }
        
        self.lots = {'x': 2, 'y': 5}
        self.prices = {'x': 60000.0, 'y': 140000.0}
        
        # 合约规格
        self.contract_specs = {
            'CU': {
                'multiplier': 5,
                'tick_size': 10
            },
            'SN': {
                'multiplier': 1,
                'tick_size': 10
            }
        }
        
        # 设置合约规格
        self.executor.set_contract_specs(self.contract_specs)
    
    # ========== REQ-4.2.1: 开仓执行 ==========
    
    def test_req_4_2_1_1_record_open_info(self):
        """REQ-4.2.1.1: 记录开仓时间、配对名称、方向"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        self.assertIsNotNone(position.open_date)
        self.assertEqual(position.pair, 'CU-SN')
        self.assertEqual(position.direction, 'long')
        
        # 测试做空
        position2 = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_short'
        )
        self.assertEqual(position2.direction, 'short')
    
    def test_req_4_2_1_2_record_prices_and_lots(self):
        """REQ-4.2.1.2: 记录X和Y的开仓价格、手数"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        self.assertEqual(position.lots_x, 2)
        self.assertEqual(position.lots_y, 5)
        # 价格应包含滑点
        self.assertNotEqual(position.open_price_x, 60000)  # 有滑点
        self.assertNotEqual(position.open_price_y, 140000)  # 有滑点
    
    def test_req_4_2_1_3_slippage_calculation(self):
        """REQ-4.2.1.3: 计算开仓滑点：buy加tick，sell减tick"""
        # 做多配对：买X卖Y
        position_long = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # X买入，价格应该增加3个tick
        expected_x_price = 60000 + 3 * 10  # 60030
        self.assertEqual(position_long.open_price_x, expected_x_price)
        
        # Y卖出，价格应该减少3个tick
        expected_y_price = 140000 - 3 * 10  # 139970
        self.assertEqual(position_long.open_price_y, expected_y_price)
        
        # 做空配对：卖X买Y
        position_short = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_short'
        )
        
        # X卖出，价格应该减少3个tick
        expected_x_price = 60000 - 3 * 10  # 59970
        self.assertEqual(position_short.open_price_x, expected_x_price)
        
        # Y买入，价格应该增加3个tick
        expected_y_price = 140000 + 3 * 10  # 140030
        self.assertEqual(position_short.open_price_y, expected_y_price)
    
    def test_req_4_2_1_4_commission_calculation(self):
        """REQ-4.2.1.4: 计算开仓手续费"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 计算名义价值
        value_x = 2 * 60000 * 5  # 600000
        value_y = 5 * 140000 * 1  # 700000
        total_value = value_x + value_y  # 1300000
        
        # 手续费 = 名义价值 × 费率
        expected_commission = total_value * 0.0002  # 260
        
        self.assertAlmostEqual(position.open_commission, expected_commission, delta=1)
    
    def test_req_4_2_1_5_margin_calculation(self):
        """REQ-4.2.1.5: 计算保证金占用"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 保证金 = 名义价值 × 保证金率
        value_x = 2 * 60000 * 5  # 600000
        value_y = 5 * 140000 * 1  # 700000
        expected_margin = (value_x + value_y) * 0.12  # 156000
        
        self.assertAlmostEqual(position.margin, expected_margin, delta=100)
    
    def test_req_4_2_1_6_unique_position_id(self):
        """REQ-4.2.1.6: 生成唯一position_id"""
        positions = []
        for i in range(10):
            position = self.executor.execute_open(
                pair_info=self.pair_info,
                lots=self.lots,
                prices=self.prices,
                signal_type='open_long'
            )
            positions.append(position.position_id)
        
        # 所有ID应该唯一
        self.assertEqual(len(positions), len(set(positions)))
        
        # ID格式应该合理
        for pid in positions:
            self.assertIsInstance(pid, str)
            self.assertGreater(len(pid), 0)
    
    # ========== REQ-4.2.2: 平仓执行 ==========
    
    def test_req_4_2_2_1_record_close_info(self):
        """REQ-4.2.2.1: 记录平仓时间、价格、原因"""
        # 先开仓
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 平仓
        close_prices = {'x': 61000.0, 'y': 138000.0}
        trade = self.executor.execute_close(
            position=position,
            prices=close_prices,
            reason='signal'
        )
        
        self.assertIsNotNone(trade.close_date)
        self.assertIsNotNone(trade.close_price_x)
        self.assertIsNotNone(trade.close_price_y)
        self.assertEqual(trade.close_reason, 'signal')
    
    def test_req_4_2_2_2_close_slippage(self):
        """REQ-4.2.2.2: 计算平仓滑点（方向相反）"""
        # 做多持仓
        position_long = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        position_long.direction = 'long'
        
        # 平仓（做多平仓=卖X买Y）
        close_prices = {'x': 61000.0, 'y': 138000.0}
        trade = self.executor.execute_close(
            position=position_long,
            prices=close_prices,
            reason='signal'
        )
        
        # X卖出，价格减少滑点
        expected_x_close = 61000 - 3 * 10  # 60970
        self.assertEqual(trade.close_price_x, expected_x_close)
        
        # Y买入，价格增加滑点
        expected_y_close = 138000 + 3 * 10  # 138030
        self.assertEqual(trade.close_price_y, expected_y_close)
    
    def test_req_4_2_2_3_close_commission(self):
        """REQ-4.2.2.3: 计算平仓手续费"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        close_prices = {'x': 61000.0, 'y': 138000.0}
        trade = self.executor.execute_close(
            position=position,
            prices=close_prices,
            reason='signal'
        )
        
        # 平仓手续费基于平仓价格
        value_x = 2 * 61000 * 5  # 610000
        value_y = 5 * 138000 * 1  # 690000
        expected_commission = (value_x + value_y) * 0.0002  # 260
        
        self.assertAlmostEqual(trade.close_commission, expected_commission, delta=1)
    
    def test_req_4_2_2_4_pnl_calculation(self):
        """REQ-4.2.2.4: 计算实现盈亏"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 盈利情况
        close_prices = {'x': 62000.0, 'y': 135000.0}
        trade = self.executor.execute_close(
            position=position,
            prices=close_prices,
            reason='signal'
        )
        
        # 验证PnL计算
        self.assertIsNotNone(trade.gross_pnl)
        self.assertIsNotNone(trade.net_pnl)
        self.assertLess(trade.net_pnl, trade.gross_pnl)  # 净利润应该小于毛利润（扣除成本）
    
    def test_req_4_2_2_5_release_margin(self):
        """REQ-4.2.2.5: 释放保证金"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        initial_margin = position.margin
        self.assertGreater(initial_margin, 0)
        
        trade = self.executor.execute_close(
            position=position,
            prices={'x': 61000.0, 'y': 138000.0},
            reason='signal'
        )
        
        # 平仓后保证金应该被释放
        self.assertEqual(trade.margin_released, initial_margin)
    
    def test_req_4_2_2_6_update_capital(self):
        """REQ-4.2.2.6: 更新可用资金"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        trade = self.executor.execute_close(
            position=position,
            prices={'x': 62000.0, 'y': 135000.0},
            reason='signal'
        )
        
        # 应该计算资金变动
        self.assertIsNotNone(trade.capital_change)
        # 资金变动 = 净盈亏 + 释放的保证金
        expected_change = trade.net_pnl + trade.margin_released
        self.assertAlmostEqual(trade.capital_change, expected_change, delta=1)
    
    # ========== REQ-4.2.3: 成本计算 ==========
    
    def test_req_4_2_3_1_commission_rate(self):
        """REQ-4.2.3.1: 手续费率默认0.0002"""
        self.assertEqual(self.executor.commission_rate, 0.0002)
        
        # 测试自定义费率
        custom_config = ExecutionConfig(commission_rate=0.0003)
        custom_executor = TradeExecutor(custom_config)
        self.assertEqual(custom_executor.commission_rate, 0.0003)
    
    def test_req_4_2_3_2_slippage_ticks(self):
        """REQ-4.2.3.2: 滑点默认3个tick"""
        self.assertEqual(self.executor.slippage_ticks, 3)
        
        # 测试自定义滑点
        custom_config = ExecutionConfig(slippage_ticks=5)
        custom_executor = TradeExecutor(custom_config)
        self.assertEqual(custom_executor.slippage_ticks, 5)
    
    def test_req_4_2_3_3_margin_rate(self):
        """REQ-4.2.3.3: 保证金率默认12%"""
        self.assertEqual(self.executor.margin_rate, 0.12)
        
        # 测试自定义保证金率
        custom_config = ExecutionConfig(margin_rate=0.15)
        custom_executor = TradeExecutor(custom_config)
        self.assertEqual(custom_executor.margin_rate, 0.15)
    
    def test_req_4_2_3_4_configurable_params(self):
        """REQ-4.2.3.4: 所有参数可配置"""
        custom_config = ExecutionConfig(
            commission_rate=0.0001,
            slippage_ticks=2,
            margin_rate=0.10
        )
        executor = TradeExecutor(custom_config)
        
        self.assertEqual(executor.commission_rate, 0.0001)
        self.assertEqual(executor.slippage_ticks, 2)
        self.assertEqual(executor.margin_rate, 0.10)
    
    # ========== 综合测试 ==========
    
    def test_complete_trade_cycle(self):
        """完整交易周期测试"""
        # 开仓
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 验证持仓
        self.assertEqual(position.pair, 'CU-SN')
        self.assertEqual(position.direction, 'long')
        self.assertGreater(position.margin, 0)
        
        # 平仓
        close_prices = {'x': 62000.0, 'y': 135000.0}
        trade = self.executor.execute_close(
            position=position,
            prices=close_prices,
            reason='signal'
        )
        
        # 验证交易记录
        self.assertEqual(trade.pair, position.pair)
        self.assertEqual(trade.position_id, position.position_id)
        self.assertIsNotNone(trade.net_pnl)
        self.assertGreater(trade.holding_days, 0)
    
    def test_stop_loss_close(self):
        """止损平仓测试"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 亏损平仓
        close_prices = {'x': 55000.0, 'y': 145000.0}
        trade = self.executor.execute_close(
            position=position,
            prices=close_prices,
            reason='stop_loss'
        )
        
        self.assertEqual(trade.close_reason, 'stop_loss')
        self.assertLess(trade.net_pnl, 0)  # 应该是亏损
    
    def test_time_stop_close(self):
        """时间止损平仓测试"""
        position = self.executor.execute_open(
            pair_info=self.pair_info,
            lots=self.lots,
            prices=self.prices,
            signal_type='open_long'
        )
        
        # 30天后平仓
        position.open_date = datetime(2024, 1, 1)
        
        trade = self.executor.execute_close(
            position=position,
            prices={'x': 60000.0, 'y': 140000.0},
            reason='time_stop',
            close_date=datetime(2024, 1, 31)
        )
        
        self.assertEqual(trade.close_reason, 'time_stop')
        self.assertEqual(trade.holding_days, 30)


if __name__ == '__main__':
    unittest.main()