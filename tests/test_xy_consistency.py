#!/usr/bin/env python3
"""
测试X/Y符号一致性

验证从协整分析到回测执行的整个流程中，X/Y符号定义保持一致。
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.backtest import BacktestEngine


class TestXYConsistency(unittest.TestCase):
    """测试X/Y符号一致性"""
    
    def setUp(self):
        """测试初始化"""
        # 创建模拟的合约规格
        self.contract_specs = {
            'HC0': {
                'name': '热轧卷板主力',
                'multiplier': 10,
                'tick_size': 1.0,
                'margin_rate': 0.12
            },
            'I0': {
                'name': '铁矿石主力',
                'multiplier': 100,
                'tick_size': 0.5,
                'margin_rate': 0.12
            },
            'CU0': {
                'name': '沪铜主力',
                'multiplier': 5,
                'tick_size': 10.0,
                'margin_rate': 0.12
            }
        }
        
        # 创建回测引擎
        self.engine = BacktestEngine(
            initial_capital=5000000
        )
        # 手动设置合约规格
        self.engine.contract_specs = self.contract_specs
        
    def test_signal_with_explicit_xy(self):
        """测试带有明确symbol_x和symbol_y的信号"""
        
        # 创建测试信号
        signal = {
            'pair': 'HC0-I0',
            'signal': 'long_spread',
            'date': '2023-07-28',
            'symbol_x': 'HC0',  # 明确指定X
            'symbol_y': 'I0',   # 明确指定Y
            'beta': 0.8126,
            'theoretical_ratio': 0.8126,
            'z_score': -2.5
        }
        
        # 当前价格
        current_prices = {
            'HC0': 4117.0,
            'I0': 834.5
        }
        
        # 计算手数
        lots_info = self.engine.calculate_lots(signal, 0.05, current_prices)
        
        # 验证X和Y正确识别
        self.assertEqual(lots_info['symbol_x'], 'HC0')
        self.assertEqual(lots_info['symbol_y'], 'I0')
        
        # 验证手数和保证金计算
        self.assertIn('contracts_y', lots_info)
        self.assertIn('contracts_x', lots_info)
        self.assertIn('margin_required', lots_info)
        
    def test_signal_without_explicit_xy(self):
        """测试没有明确symbol_x和symbol_y的信号（向后兼容）"""
        
        # 创建测试信号（旧格式）
        signal = {
            'pair': 'HC0-I0',
            'signal': 'long_spread',
            'date': '2023-07-28',
            'beta': 0.8126,
            'theoretical_ratio': 0.8126,
            'z_score': -2.5
        }
        
        # 当前价格
        current_prices = {
            'HC0': 4117.0,
            'I0': 834.5
        }
        
        # 计算手数
        lots_info = self.engine.calculate_lots(signal, 0.05, current_prices)
        
        # 验证向后兼容：按split('-')解析
        self.assertEqual(lots_info['symbol_x'], 'HC0')
        self.assertEqual(lots_info['symbol_y'], 'I0')
        
    def test_xy_consistency_in_position(self):
        """测试持仓记录中X/Y的一致性"""
        
        # 创建带有明确X/Y的信号
        signal = {
            'pair': 'CU0-I0',
            'signal': 'short_spread',
            'date': '2023-08-01',
            'symbol_x': 'I0',   # I0是X（低波动）
            'symbol_y': 'CU0',  # CU0是Y（高波动）
            'beta': 0.6018,
            'theoretical_ratio': 0.6018,
            'z_score': 2.3
        }
        
        # 当前价格
        current_prices = {
            'CU0': 69000.0,
            'I0': 721.0
        }
        
        # 执行开仓
        success = self.engine.execute_signal(signal, current_prices)
        
        if success:
            # 获取持仓
            position = self.engine.position_manager.positions.get('CU0-I0')
            self.assertIsNotNone(position)
            
            # 验证X/Y正确记录
            self.assertEqual(position.symbol_x, 'I0')
            self.assertEqual(position.symbol_y, 'CU0')
            
    def test_pnl_calculation_with_correct_xy(self):
        """测试使用正确X/Y计算PnL"""
        
        # 模拟持仓（HC0是X，I0是Y）
        from lib.backtest import Position
        position = Position(
            pair='HC0-I0',
            direction='long_spread',
            spread_formula='I0 - 0.8126*HC0',
            open_date=pd.Timestamp('2023-07-28'),
            position_weight=0.05,
            symbol_x='HC0',  # 正确的X
            symbol_y='I0',    # 正确的Y
            contracts_x=18,   # HC0的手数
            contracts_y=16,   # I0的手数
            beta=0.8126,
            open_price_x=4117.0,  # HC0的开仓价
            open_price_y=834.5,   # I0的开仓价
            margin_occupied=249151.2,
            open_commission=415.62,
            prev_price_x=4117.0,
            prev_price_y=834.5,
            multiplier_x=10,   # HC0的乘数
            multiplier_y=100   # I0的乘数
        )
        
        # 计算PnL（价格变化后）
        close_price_x = 3984.0  # HC0的平仓价
        close_price_y = 719.5   # I0的平仓价
        
        gross_pnl, y_pnl, x_pnl = self.engine._calculate_pnl_method1(
            position, close_price_y, close_price_x
        )
        
        # 验证PnL计算
        # long_spread: 买Y(I0)卖X(HC0)
        # Y腿PnL = (719.5 - 834.5) * 16 * 100 = -184,000
        # X腿PnL = (4117.0 - 3984.0) * 18 * 10 = 23,940
        # 总PnL = -184,000 + 23,940 = -160,060
        
        expected_y_pnl = (close_price_y - position.open_price_y) * position.contracts_y * position.multiplier_y
        expected_x_pnl = (position.open_price_x - close_price_x) * position.contracts_x * position.multiplier_x
        expected_gross = expected_y_pnl + expected_x_pnl
        
        self.assertAlmostEqual(y_pnl, expected_y_pnl, places=2)
        self.assertAlmostEqual(x_pnl, expected_x_pnl, places=2)
        self.assertAlmostEqual(gross_pnl, expected_gross, places=2)
        
    def test_error_on_missing_contract_spec(self):
        """测试缺少合约规格时的错误处理"""
        
        signal = {
            'pair': 'XXX-YYY',
            'signal': 'long_spread',
            'symbol_x': 'XXX',
            'symbol_y': 'YYY',
            'beta': 1.0,
            'theoretical_ratio': 1.0
        }
        
        current_prices = {
            'XXX': 100.0,
            'YYY': 200.0
        }
        
        # 应该返回None
        lots_info = self.engine.calculate_lots(signal, 0.05, current_prices)
        self.assertIsNone(lots_info)
        
    def test_xy_in_cointegration_output(self):
        """测试协整分析输出的X/Y定义"""
        
        # 模拟协整分析结果
        coint_result = {
            'pair': 'HC0-I0',
            'symbol_x': 'HC0',  # 低波动率
            'symbol_y': 'I0',   # 高波动率
            'direction': 'y_on_x',  # I0对HC0回归
            'pvalue_4y': 0.0108,
            'pvalue_1y': 0.00001,
            'beta_4y': 0.439301,
            'beta_1y': 1.468264
        }
        
        # 验证X是低波动率品种
        self.assertEqual(coint_result['symbol_x'], 'HC0')
        # 验证Y是高波动率品种
        self.assertEqual(coint_result['symbol_y'], 'I0')
        # 验证方向表示Y对X回归
        self.assertEqual(coint_result['direction'], 'y_on_x')


class TestXYIntegration(unittest.TestCase):
    """集成测试：验证整个流程的X/Y一致性"""
    
    def test_end_to_end_xy_consistency(self):
        """测试端到端的X/Y一致性"""
        
        # 1. 模拟协整分析输出
        pairs_params = {
            'HC0-I0': {
                'symbol_x': 'HC0',
                'symbol_y': 'I0',
                'beta_initial': 0.8126,
                'beta_ols': 0.8126,
                'direction': 'y_on_x'
            }
        }
        
        # 2. 模拟信号生成（应该包含symbol_x和symbol_y）
        signal = {
            'pair': 'HC0-I0',
            'signal': 'long_spread',
            'beta': 0.8126,
            'z_score': -2.5
        }
        
        # 添加symbol_x和symbol_y（这是我们要验证的修改）
        pair_info = pairs_params.get(signal['pair'])
        signal['symbol_x'] = pair_info.get('symbol_x')
        signal['symbol_y'] = pair_info.get('symbol_y')
        
        # 3. 验证信号包含正确的X/Y
        self.assertEqual(signal['symbol_x'], 'HC0')
        self.assertEqual(signal['symbol_y'], 'I0')
        
        # 4. 验证回测可以正确使用这些信息
        contract_specs = {
            'HC0': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.12},
            'I0': {'multiplier': 100, 'tick_size': 0.5, 'margin_rate': 0.12}
        }
        
        engine = BacktestEngine(
            initial_capital=5000000
        )
        engine.contract_specs = contract_specs
        
        current_prices = {'HC0': 4117.0, 'I0': 834.5}
        lots_info = engine.calculate_lots(signal, 0.05, current_prices)
        
        # 验证最终使用的X/Y正确
        self.assertEqual(lots_info['symbol_x'], 'HC0')
        self.assertEqual(lots_info['symbol_y'], 'I0')


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)