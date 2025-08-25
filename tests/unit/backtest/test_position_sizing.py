"""
手数计算模块测试用例
严格对应需求文档 REQ-4.1
"""

import unittest
from fractions import Fraction
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig


class TestPositionSizing(unittest.TestCase):
    """测试手数计算模块"""
    
    def setUp(self):
        """初始化测试配置"""
        self.config = PositionSizingConfig(
            max_denominator=10,
            min_lots=1,
            max_lots_per_leg=100,
            margin_rate=0.12
        )
        self.sizer = PositionSizer(self.config)
    
    # ========== REQ-4.1.2: 最小整数比计算 ==========
    
    def test_req_4_1_2_1_effective_hedge_ratio(self):
        """REQ-4.1.2.1: 计算有效对冲比 h* = β × (Py × My) / (Px × Mx)"""
        beta = 0.85
        price_x = 60000
        price_y = 140000
        multiplier_x = 5
        multiplier_y = 1
        
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=price_x,
            price_y=price_y,
            multiplier_x=multiplier_x,
            multiplier_y=multiplier_y
        )
        
        # 验证有效对冲比计算
        expected_h_star = beta * (price_y * multiplier_y) / (price_x * multiplier_x)
        # 0.85 * (140000 * 1) / (60000 * 5) = 0.85 * 140000 / 300000 = 0.3967
        self.assertAlmostEqual(expected_h_star, 0.3967, places=3)
        self.assertAlmostEqual(result['effective_ratio'], expected_h_star, places=4)
    
    def test_req_4_1_2_2_fraction_approximation(self):
        """REQ-4.1.2.2: 使用Fraction类对h*进行连分数逼近"""
        beta = 0.85
        price_x = 60000
        price_y = 140000
        multiplier_x = 5
        multiplier_y = 1
        
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=price_x,
            price_y=price_y,
            multiplier_x=multiplier_x,
            multiplier_y=multiplier_y
        )
        
        # h* ≈ 0.3967, Fraction(0.3967).limit_denominator(10) = 2/5
        self.assertEqual(result['lots_x'], 2)
        self.assertEqual(result['lots_y'], 5)
    
    def test_req_4_1_2_3_max_denominator(self):
        """REQ-4.1.2.3: max_denominator默认值10，可配置"""
        # 测试默认值
        beta = 0.123456  # 一个需要大分母的比例
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=100,
            price_y=100,
            multiplier_x=1,
            multiplier_y=1
        )
        
        # 确保分母不超过10
        self.assertLessEqual(result['lots_y'], 10)
        
        # 测试自定义max_denominator
        custom_config = PositionSizingConfig(max_denominator=5)
        custom_sizer = PositionSizer(custom_config)
        result2 = custom_sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=100,
            price_y=100,
            multiplier_x=1,
            multiplier_y=1
        )
        self.assertLessEqual(result2['lots_y'], 5)
    
    def test_req_4_1_2_4_min_lots_constraint(self):
        """REQ-4.1.2.4: 得到最小整数对(nx, ny)，确保每腿至少min_lots手"""
        # 测试极小的beta值
        beta = 0.01
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=100,
            price_y=100,
            multiplier_x=1,
            multiplier_y=1
        )
        
        # 确保每腿至少1手
        self.assertGreaterEqual(result['lots_x'], 1)
        self.assertGreaterEqual(result['lots_y'], 1)
    
    def test_req_4_1_2_5_nominal_error(self):
        """REQ-4.1.2.5: 计算名义价值匹配误差"""
        beta = 0.85
        price_x = 60000
        price_y = 140000
        multiplier_x = 5
        multiplier_y = 1
        
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=price_x,
            price_y=price_y,
            multiplier_x=multiplier_x,
            multiplier_y=multiplier_y
        )
        
        # 验证误差计算
        nominal_x = result['lots_x'] * price_x * multiplier_x
        nominal_y = result['lots_y'] * price_y * multiplier_y * beta
        error = abs(nominal_x - nominal_y) / nominal_y * 100
        
        self.assertAlmostEqual(result['nominal_error_pct'], error, places=2)
        self.assertLess(result['nominal_error_pct'], 10)  # 误差应该较小
    
    # ========== REQ-4.1.3: 资金约束应用 ==========
    
    def test_req_4_1_3_1_margin_calculation(self):
        """REQ-4.1.3.1: 计算最小整数对所需保证金"""
        min_lots = {'lots_x': 2, 'lots_y': 5}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=5000000,
            position_weight=0.05
        )
        
        # 验证保证金计算
        expected_margin = (
            2 * 60000 * 5 * 0.12 +  # X品种保证金
            5 * 140000 * 1 * 0.12    # Y品种保证金
        )
        # = 72000 + 84000 = 156000
        self.assertEqual(expected_margin, 156000)
        self.assertAlmostEqual(result['margin_required'], expected_margin, delta=1)
    
    def test_req_4_1_3_2_scaling_factor(self):
        """REQ-4.1.3.2: 计算整数倍缩放系数 k = floor(allocated × 0.95 / margin)"""
        min_lots = {'lots_x': 2, 'lots_y': 5}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        total_capital = 5000000
        position_weight = 0.05
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=total_capital,
            position_weight=position_weight
        )
        
        # 验证缩放系数计算
        allocated = total_capital * position_weight  # 250000
        min_margin = 156000  # 从上一个测试得出
        expected_k = int(allocated * 0.95 / min_margin)  # floor(237500/156000) = 1
        
        self.assertEqual(result['scaling_factor'], expected_k)
        self.assertEqual(result['scaling_factor'], 1)
    
    def test_req_4_1_3_3_final_lots(self):
        """REQ-4.1.3.3: 最终手数 final_nx = nx × k, final_ny = ny × k"""
        min_lots = {'lots_x': 2, 'lots_y': 5}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=5000000,
            position_weight=0.05
        )
        
        # k=1时，最终手数等于最小手数
        self.assertEqual(result['final_lots_x'], 2 * 1)
        self.assertEqual(result['final_lots_y'], 5 * 1)
        
        # 测试更多资金的情况
        result2 = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=10000000,  # 翻倍
            position_weight=0.05
        )
        
        # k应该是2
        self.assertEqual(result2['scaling_factor'], 3)
        self.assertEqual(result2['final_lots_x'], 2 * 3)
        self.assertEqual(result2['final_lots_y'], 5 * 3)
    
    def test_req_4_1_3_4_insufficient_capital(self):
        """REQ-4.1.3.4: 如果k=0（资金不足），返回can_trade=False"""
        min_lots = {'lots_x': 10, 'lots_y': 20}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=1000000,  # 资金很少
            position_weight=0.05     # 只有50000可用
        )
        
        # 最小保证金需求 = 10*60000*5*0.12 + 20*140000*1*0.12 = 696000
        # 可用资金50000 < 696000，所以k=0
        self.assertEqual(result['scaling_factor'], 0)
        self.assertFalse(result['can_trade'])
        self.assertIn('Insufficient', result['reason'])
    
    def test_req_4_1_3_5_max_lots_limit(self):
        """REQ-4.1.3.5: 检查最大手数限制"""
        min_lots = {'lots_x': 10, 'lots_y': 20}
        prices = {'x': 1000, 'y': 2000}  # 低价格
        multipliers = {'x': 1, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=100000000,  # 大量资金
            position_weight=0.5        # 50%分配
        )
        
        # 即使资金充足，也不能超过max_lots_per_leg=100
        self.assertLessEqual(result['final_lots_x'], 100)
        self.assertLessEqual(result['final_lots_y'], 100)
    
    def test_req_4_1_3_6_utilization_rate(self):
        """REQ-4.1.3.6: 计算资金利用率"""
        min_lots = {'lots_x': 2, 'lots_y': 5}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=5000000,
            position_weight=0.05
        )
        
        # 验证利用率计算
        allocated = 250000
        actual_margin = result['margin_required']
        expected_utilization = actual_margin / allocated
        
        self.assertAlmostEqual(result['utilization_rate'], expected_utilization, places=4)
        self.assertLess(result['utilization_rate'], 1.0)  # 不应超过100%
        self.assertGreater(result['utilization_rate'], 0.5)  # 应该有合理利用率
    
    # ========== REQ-4.1.1: 资金分配机制 ==========
    
    def test_req_4_1_1_1_capital_allocation(self):
        """REQ-4.1.1.1: 每个配对独立分配资金"""
        total_capital = 5000000
        position_weight = 0.05
        
        min_lots = {'lots_x': 2, 'lots_y': 5}
        prices = {'x': 60000, 'y': 140000}
        multipliers = {'x': 5, 'y': 1}
        
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=total_capital,
            position_weight=position_weight
        )
        
        expected_allocation = total_capital * position_weight
        self.assertEqual(result['allocated_capital'], expected_allocation)
        self.assertEqual(result['allocated_capital'], 250000)
    
    def test_req_4_1_1_2_default_weight(self):
        """REQ-4.1.1.2: 默认position_weight = 0.05"""
        min_lots = {'lots_x': 1, 'lots_y': 1}
        prices = {'x': 10000, 'y': 10000}
        multipliers = {'x': 1, 'y': 1}
        
        # 不指定position_weight，应使用默认值0.05
        result = self.sizer.calculate_position_size(
            min_lots=min_lots,
            prices=prices,
            multipliers=multipliers,
            total_capital=1000000
            # position_weight未指定，使用默认值
        )
        
        self.assertEqual(result['allocated_capital'], 50000)  # 1000000 * 0.05
    
    # ========== 边界条件测试 ==========
    
    def test_edge_case_exact_ratio(self):
        """边界条件：beta正好是整数比"""
        beta = 0.4  # 正好是2/5
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=100,
            price_y=100,
            multiplier_x=1,
            multiplier_y=1
        )
        
        self.assertEqual(result['lots_x'], 2)
        self.assertEqual(result['lots_y'], 5)
        self.assertAlmostEqual(result['nominal_error_pct'], 0, places=2)
    
    def test_edge_case_very_large_beta(self):
        """边界条件：非常大的beta值"""
        beta = 9.5
        result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=100,
            price_y=100,
            multiplier_x=1,
            multiplier_y=1
        )
        
        # 应该得到接近9.5的比例，但受max_denominator限制
        ratio = result['lots_x'] / result['lots_y']
        self.assertAlmostEqual(ratio, 9.5, delta=0.5)
    
    def test_integration_full_flow(self):
        """集成测试：完整流程"""
        # 第一步：计算最小整数比
        beta = 0.85
        price_x = 60000
        price_y = 140000
        multiplier_x = 5
        multiplier_y = 1
        
        ratio_result = self.sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=price_x,
            price_y=price_y,
            multiplier_x=multiplier_x,
            multiplier_y=multiplier_y
        )
        
        # 第二步：应用资金约束
        position_result = self.sizer.calculate_position_size(
            min_lots={
                'lots_x': ratio_result['lots_x'],
                'lots_y': ratio_result['lots_y']
            },
            prices={'x': price_x, 'y': price_y},
            multipliers={'x': multiplier_x, 'y': multiplier_y},
            total_capital=5000000,
            position_weight=0.05
        )
        
        # 验证完整流程
        self.assertTrue(position_result['can_trade'])
        self.assertGreater(position_result['final_lots_x'], 0)
        self.assertGreater(position_result['final_lots_y'], 0)
        # 验证比例保持
        final_ratio = position_result['final_lots_x'] / position_result['final_lots_y']
        original_ratio = ratio_result['lots_x'] / ratio_result['lots_y']
        self.assertAlmostEqual(final_ratio, original_ratio, places=6)


if __name__ == '__main__':
    unittest.main()