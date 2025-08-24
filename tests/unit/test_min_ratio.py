#!/usr/bin/env python3
"""
最小整数比计算测试用例
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.min_integer_ratio import calculate_min_integer_ratio


class TestMinIntegerRatio(unittest.TestCase):
    """最小整数比测试"""
    
    def test_simple_ratios(self):
        """测试简单比例"""
        # β = 0.5 -> Y:X = 1:2
        lots_y, lots_x = calculate_min_integer_ratio(0.5)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 2)
        
        # β = 2.0 -> Y:X = 2:1
        lots_y, lots_x = calculate_min_integer_ratio(2.0)
        self.assertEqual(lots_y, 2)
        self.assertEqual(lots_x, 1)
        
        # β = 1.0 -> Y:X = 1:1
        lots_y, lots_x = calculate_min_integer_ratio(1.0)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 1)
    
    def test_decimal_ratios(self):
        """测试小数比例"""
        # β = 1.5 -> Y:X = 3:2
        lots_y, lots_x = calculate_min_integer_ratio(1.5)
        self.assertEqual(lots_y, 3)
        self.assertEqual(lots_x, 2)
        
        # β = 0.333... -> Y:X = 1:3
        lots_y, lots_x = calculate_min_integer_ratio(1/3)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 3)
        
        # β = 0.25 -> Y:X = 1:4
        lots_y, lots_x = calculate_min_integer_ratio(0.25)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 4)
    
    def test_complex_ratios(self):
        """测试复杂比例"""
        # β = 0.85 -> 接近 17:20，但限制分母后应该是 6:7
        lots_y, lots_x = calculate_min_integer_ratio(0.85, max_denominator=10)
        self.assertEqual(lots_y, 6)
        self.assertEqual(lots_x, 7)
        self.assertAlmostEqual(lots_y/lots_x, 0.857, places=2)
        
        # β = 3.5 -> Y:X = 7:2
        lots_y, lots_x = calculate_min_integer_ratio(3.5)
        self.assertEqual(lots_y, 7)
        self.assertEqual(lots_x, 2)
    
    def test_extreme_values(self):
        """测试极端值"""
        # 很小的β
        lots_y, lots_x = calculate_min_integer_ratio(0.1)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 10)
        
        # 很大的β
        lots_y, lots_x = calculate_min_integer_ratio(10.0)
        self.assertEqual(lots_y, 10)
        self.assertEqual(lots_x, 1)
        
        # 接近0的β
        lots_y, lots_x = calculate_min_integer_ratio(0.01, max_denominator=100)
        self.assertEqual(lots_y, 1)
        self.assertLessEqual(lots_x, 100)
    
    def test_max_denominator_limit(self):
        """测试最大分母限制"""
        # β = 0.123 不限制会得到很大的分母
        lots_y, lots_x = calculate_min_integer_ratio(0.123, max_denominator=10)
        self.assertLessEqual(lots_x, 10)
        self.assertLessEqual(lots_y, 10)
        
        # 验证比例接近原值
        ratio = lots_y / lots_x
        self.assertAlmostEqual(ratio, 0.123, delta=0.02)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # β = 0 或负数
        lots_y, lots_x = calculate_min_integer_ratio(0)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 1)
        
        lots_y, lots_x = calculate_min_integer_ratio(-1.5)
        self.assertEqual(lots_y, 1)
        self.assertEqual(lots_x, 1)
    
    def test_accuracy_preservation(self):
        """测试精度保持"""
        test_betas = [0.5, 1.5, 2.5, 0.333, 0.667, 0.85, 1.2, 3.5]
        
        for beta in test_betas:
            lots_y, lots_x = calculate_min_integer_ratio(beta)
            actual_ratio = lots_y / lots_x
            
            # 实际比例应该接近原始β
            self.assertAlmostEqual(actual_ratio, beta, delta=0.1)
            
            # 手数应该是最小的
            self.assertLessEqual(lots_y, 10)
            self.assertLessEqual(lots_x, 10)


if __name__ == '__main__':
    unittest.main()