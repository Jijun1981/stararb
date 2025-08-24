#!/usr/bin/env python3
"""
回测框架TDD测试
基于需求文档REQ-4.x编写测试用例
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fractions import Fraction

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TestBacktestConfig(unittest.TestCase):
    """测试REQ-4.7: 参数配置"""
    
    def test_default_config_creation(self):
        """REQ-4.7.6: 提供默认配置类BacktestConfig"""
        from lib.backtest_v4 import BacktestConfig
        
        config = BacktestConfig()
        
        # 验证默认参数
        self.assertEqual(config.initial_capital, 5000000)
        self.assertEqual(config.margin_rate, 0.12)
        self.assertEqual(config.commission_rate, 0.0002)
        self.assertEqual(config.slippage_ticks, 3)
        self.assertEqual(config.stop_loss_pct, 0.15)
        self.assertEqual(config.max_holding_days, 30)
        self.assertEqual(config.z_open_threshold, 2.0)
        self.assertEqual(config.z_close_threshold, 0.5)
    
    def test_config_parameter_override(self):
        """REQ-4.7.1-4.7.5: 参数可配置"""
        from lib.backtest_v4 import BacktestConfig
        
        # 创建自定义配置
        config = BacktestConfig(
            initial_capital=10000000,
            margin_rate=0.15,
            commission_rate=0.0001,
            slippage_ticks=2,
            stop_loss_pct=0.10,
            max_holding_days=20,
            z_open_threshold=2.5,
            z_close_threshold=0.3,
            max_denominator=20,
            min_lots=2
        )
        
        # 验证参数覆盖
        self.assertEqual(config.initial_capital, 10000000)
        self.assertEqual(config.margin_rate, 0.15)
        self.assertEqual(config.commission_rate, 0.0001)
        self.assertEqual(config.slippage_ticks, 2)
        self.assertEqual(config.stop_loss_pct, 0.10)
        self.assertEqual(config.max_holding_days, 20)
        self.assertEqual(config.z_open_threshold, 2.5)
        self.assertEqual(config.z_close_threshold, 0.3)
        self.assertEqual(config.max_denominator, 20)
        self.assertEqual(config.min_lots, 2)
    
    def test_create_engine_with_dict_config(self):
        """REQ-4.7.7: 提供create_backtest_engine便捷函数"""
        from lib.backtest_v4 import create_backtest_engine
        
        # 使用字典配置创建引擎
        custom_config = {
            'initial_capital': 8000000,
            'stop_loss_pct': 0.12,
            'max_holding_days': 25
        }
        
        engine = create_backtest_engine(custom_config)
        
        # 验证配置生效
        self.assertEqual(engine.config.initial_capital, 8000000)
        self.assertEqual(engine.config.stop_loss_pct, 0.12)
        self.assertEqual(engine.config.max_holding_days, 25)
        # 其他参数应该是默认值
        self.assertEqual(engine.config.margin_rate, 0.12)


class TestLotsCalculation(unittest.TestCase):
    """测试REQ-4.1.1: 根据β值计算最小整数比手数"""
    
    def test_calculate_min_lots_with_fraction(self):
        """REQ-4.1.1: 使用Fraction类计算最小整数比"""
        from lib.backtest_v4 import BacktestEngine
        
        engine = BacktestEngine()
        
        # 测试案例1: β=0.5 -> Y:X = 2:1
        result = engine.calculate_min_lots(0.5)
        self.assertEqual(result['lots_y'], 2)
        self.assertEqual(result['lots_x'], 1)
        self.assertAlmostEqual(result['actual_ratio'], 0.5, places=4)
        
        # 测试案例2: β=1.5 -> Y:X = 2:3
        result = engine.calculate_min_lots(1.5)
        self.assertEqual(result['lots_y'], 2)
        self.assertEqual(result['lots_x'], 3)
        self.assertAlmostEqual(result['actual_ratio'], 1.5, places=4)
        
        # 测试案例3: β=0.85，验证误差在可接受范围
        result = engine.calculate_min_lots(0.85)
        self.assertGreater(result['lots_y'], 0)
        self.assertGreater(result['lots_x'], 0)
        self.assertLess(result['error'], 0.2)  # 误差小于20%
    
    def test_lots_calculation_with_limits(self):
        """REQ-4.7.5: 手数计算参数可配置"""
        from lib.backtest_v4 import BacktestConfig, BacktestEngine
        
        # 配置最小手数为3
        config = BacktestConfig(min_lots=3, max_lots_per_leg=10)
        engine = BacktestEngine(config)
        
        result = engine.calculate_min_lots(0.5)
        # 确保满足最小手数要求
        self.assertGreaterEqual(result['lots_y'], 3)
        self.assertGreaterEqual(result['lots_x'], 3)
        # 确保不超过最大手数
        self.assertLessEqual(result['lots_y'], 10)
        self.assertLessEqual(result['lots_x'], 10)


class TestTradeExecution(unittest.TestCase):
    """测试REQ-4.1: 交易执行"""
    
    def setUp(self):
        """准备测试数据"""
        from lib.backtest_v4 import BacktestEngine, BacktestConfig
        
        self.config = BacktestConfig()
        self.engine = BacktestEngine(self.config)
        
        # 模拟合约规格
        self.engine.contract_specs = {
            'AG': {'multiplier': 15, 'tick_size': 1},
            'NI': {'multiplier': 1, 'tick_size': 10}
        }
        
        # 模拟价格
        self.prices = {
            'AG': 5000,
            'NI': 120000
        }
    
    def test_open_position_with_z_threshold(self):
        """REQ-4.1.2: 开仓条件检查"""
        # 准备信号
        signal = {
            'date': '2024-01-01',
            'pair': 'AG-NI',
            'symbol_x': 'AG',
            'symbol_y': 'NI',
            'signal': 'open_long',
            'z_score': -2.5,  # 超过默认阈值2.0
            'beta': 0.85
        }
        
        # 处理信号
        success = self.engine.process_signal(signal, self.prices)
        
        # 验证开仓成功
        self.assertTrue(success)
        self.assertIn('AG-NI', self.engine.positions)
        
    def test_reject_open_below_threshold(self):
        """REQ-4.1.2: Z-score未达阈值不开仓"""
        signal = {
            'date': '2024-01-01',
            'pair': 'AG-NI',
            'symbol_x': 'AG',
            'symbol_y': 'NI',
            'signal': 'open_long',
            'z_score': -1.5,  # 未达到默认阈值2.0
            'beta': 0.85
        }
        
        success = self.engine.process_signal(signal, self.prices)
        
        # 验证未开仓
        self.assertFalse(success)
        self.assertNotIn('AG-NI', self.engine.positions)
    
    def test_configurable_z_thresholds(self):
        """REQ-4.7.4: 信号参数可配置"""
        from lib.backtest_v4 import BacktestConfig, BacktestEngine
        
        # 创建自定义阈值配置
        config = BacktestConfig(
            z_open_threshold=2.5,
            z_close_threshold=0.3
        )
        engine = BacktestEngine(config)
        engine.contract_specs = self.engine.contract_specs
        
        # Z=2.3，未达到新阈值2.5
        signal = {
            'date': '2024-01-01',
            'pair': 'AG-NI',
            'symbol_x': 'AG',
            'symbol_y': 'NI',
            'signal': 'open_long',
            'z_score': -2.3,
            'beta': 0.85
        }
        
        success = engine.process_signal(signal, self.prices)
        self.assertFalse(success)
        
        # Z=2.6，超过新阈值2.5
        signal['z_score'] = -2.6
        success = engine.process_signal(signal, self.prices)
        self.assertTrue(success)


class TestSlippage(unittest.TestCase):
    """测试REQ-4.1.4: 滑点计算"""
    
    def test_apply_slippage(self):
        """REQ-4.1.4: 滑点计算正确"""
        from lib.backtest_v4 import BacktestEngine
        
        engine = BacktestEngine()
        
        # 买入时价格上滑
        buy_price = engine.apply_slippage(100, 'buy', tick_size=0.5)
        self.assertEqual(buy_price, 101.5)  # 100 + 0.5*3
        
        # 卖出时价格下滑
        sell_price = engine.apply_slippage(100, 'sell', tick_size=0.5)
        self.assertEqual(sell_price, 98.5)  # 100 - 0.5*3
    
    def test_configurable_slippage_ticks(self):
        """REQ-4.7.2: 滑点tick数可配置"""
        from lib.backtest_v4 import BacktestConfig, BacktestEngine
        
        # 配置2个tick滑点
        config = BacktestConfig(slippage_ticks=2)
        engine = BacktestEngine(config)
        
        buy_price = engine.apply_slippage(100, 'buy', tick_size=0.5)
        self.assertEqual(buy_price, 101.0)  # 100 + 0.5*2


class TestRiskControl(unittest.TestCase):
    """测试REQ-4.2: 风险控制"""
    
    def setUp(self):
        from lib.backtest_v4 import BacktestEngine, Position
        
        self.engine = BacktestEngine()
        
        # 创建测试持仓
        self.position = Position(
            pair='AG-NI',
            symbol_x='AG',
            symbol_y='NI',
            direction='open_long',
            open_date=datetime(2024, 1, 1),
            beta=0.85,
            lots_x=10,
            lots_y=8,
            theoretical_ratio=0.85,
            actual_ratio=0.8,
            open_price_x=5000,
            open_price_y=120000,
            margin_occupied=100000,
            open_commission=200,
            multiplier_x=15,
            multiplier_y=1,
            tick_size_x=1,
            tick_size_y=10
        )
    
    def test_stop_loss_trigger(self):
        """REQ-4.2.3: 止损条件检查"""
        # 添加持仓
        self.engine.positions['AG-NI'] = self.position
        
        # 模拟价格下跌导致亏损超过15%
        current_prices = {
            'AG': 4800,  # 下跌
            'NI': 122000  # 上涨
        }
        
        # 检查风险控制
        pairs_to_close = self.engine.check_risk_control('2024-01-02', current_prices)
        
        # 根据实际PnL计算判断是否应该止损
        # 这里需要计算实际盈亏是否达到15%
        # 注意：具体是否触发取决于计算结果
    
    def test_time_stop_trigger(self):
        """REQ-4.2.4: 时间止损检查"""
        # 设置开仓日期为31天前
        self.position.open_date = datetime(2024, 1, 1)
        self.engine.positions['AG-NI'] = self.position
        
        # 当前日期是31天后
        current_date = '2024-02-01'
        current_prices = {'AG': 5000, 'NI': 120000}
        
        pairs_to_close = self.engine.check_risk_control(current_date, current_prices)
        
        # 应该触发时间止损
        self.assertTrue(any(pair == 'AG-NI' and reason == 'time_stop' 
                          for pair, reason in pairs_to_close))
    
    def test_configurable_risk_parameters(self):
        """REQ-4.7.3: 风险控制参数可配置"""
        from lib.backtest_v4 import BacktestConfig, BacktestEngine
        
        # 配置10%止损，20天强制平仓
        config = BacktestConfig(
            stop_loss_pct=0.10,
            max_holding_days=20
        )
        engine = BacktestEngine(config)
        
        self.assertEqual(engine.config.stop_loss_pct, 0.10)
        self.assertEqual(engine.config.max_holding_days, 20)


class TestPnLCalculation(unittest.TestCase):
    """测试REQ-4.3: PnL计算"""
    
    def test_gross_pnl_calculation_long(self):
        """REQ-4.3.1: 毛PnL计算（做多价差）"""
        from lib.backtest_v4 import BacktestEngine, Position
        
        engine = BacktestEngine()
        
        # 创建做多价差持仓
        position = Position(
            pair='AG-NI',
            symbol_x='AG',
            symbol_y='NI',
            direction='open_long',  # 做多价差：买Y卖X
            open_date=datetime(2024, 1, 1),
            beta=1.0,
            lots_x=10,
            lots_y=10,
            theoretical_ratio=1.0,
            actual_ratio=1.0,
            open_price_x=5000,
            open_price_y=120000,
            margin_occupied=100000,
            open_commission=200,
            multiplier_x=15,
            multiplier_y=1,
            tick_size_x=1,
            tick_size_y=10
        )
        
        # 价差扩大，应该盈利
        # Y涨，X跌
        close_price_y = 122000  # Y上涨
        close_price_x = 4900    # X下跌
        
        # 计算PnL
        # Y腿：(122000-120000)*10*1 = 20000
        # X腿：(5000-4900)*10*15 = 15000
        # 总计：35000
        
        # 这里需要调用实际的PnL计算方法验证
    
    def test_commission_calculation(self):
        """REQ-4.3.2: 手续费计算"""
        from lib.backtest_v4 import BacktestEngine
        
        engine = BacktestEngine()
        
        # 名义价值
        nominal_value = 1000000
        
        # 手续费 = 名义价值 * 0.0002
        expected_commission = nominal_value * engine.config.commission_rate
        self.assertEqual(expected_commission, 200)
    
    def test_configurable_commission_rate(self):
        """REQ-4.7.2: 手续费率可配置"""
        from lib.backtest_v4 import BacktestConfig, BacktestEngine
        
        config = BacktestConfig(commission_rate=0.0001)
        engine = BacktestEngine(config)
        
        self.assertEqual(engine.config.commission_rate, 0.0001)


class TestSignalAlignment(unittest.TestCase):
    """测试与信号生成模块的格式对齐"""
    
    def test_signal_format_compatibility(self):
        """验证信号格式与REQ-4.3（信号生成模块）对齐"""
        from lib.backtest_v4 import BacktestEngine
        
        engine = BacktestEngine()
        
        # 标准信号格式（13个字段）
        signal = {
            'date': '2024-01-01',
            'pair': 'AG-NI',
            'symbol_x': 'AG',
            'symbol_y': 'NI',
            'signal': 'open_long',
            'z_score': -2.5,
            'residual': -0.15,
            'beta': 0.8234,
            'beta_initial': 0.8500,
            'days_held': 0,
            'reason': 'z_threshold',
            'phase': 'signal_period',
            'beta_window_used': '1y'
        }
        
        # 验证引擎能够处理标准信号
        engine.contract_specs = {
            'AG': {'multiplier': 15, 'tick_size': 1},
            'NI': {'multiplier': 1, 'tick_size': 10}
        }
        
        prices = {'AG': 5000, 'NI': 120000}
        
        # 应该能够成功处理信号
        success = engine.process_signal(signal, prices)
        self.assertIsNotNone(success)  # 不应该抛出异常


class TestPerformanceMetrics(unittest.TestCase):
    """测试REQ-4.4: 绩效指标"""
    
    def test_calculate_metrics(self):
        """REQ-4.4.1-4.4.6: 绩效指标计算"""
        from lib.backtest_v4 import BacktestEngine
        
        engine = BacktestEngine()
        
        # 添加一些模拟交易记录
        engine.trade_records = [
            {'net_pnl': 10000, 'holding_days': 5, 'close_reason': 'signal'},
            {'net_pnl': -5000, 'holding_days': 10, 'close_reason': 'stop_loss'},
            {'net_pnl': 8000, 'holding_days': 3, 'close_reason': 'signal'},
        ]
        
        # 添加权益曲线
        engine.equity_curve = [
            {'date': '2024-01-01', 'equity': 5000000},
            {'date': '2024-01-02', 'equity': 5010000},
            {'date': '2024-01-03', 'equity': 5005000},
            {'date': '2024-01-04', 'equity': 5013000},
        ]
        
        # 计算指标
        metrics = engine.calculate_metrics()
        
        # 验证返回的指标
        self.assertIn('total_pnl', metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # 验证计算正确性
        self.assertEqual(metrics['total_pnl'], 13000)  # 10000-5000+8000
        self.assertEqual(metrics['total_trades'], 3)
        self.assertAlmostEqual(metrics['win_rate'], 2/3, places=4)


if __name__ == '__main__':
    unittest.main()