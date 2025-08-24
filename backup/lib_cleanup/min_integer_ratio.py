#!/usr/bin/env python3
"""
最小整数比计算模块
"""

from math import gcd
from fractions import Fraction

def calculate_min_integer_ratio(beta, max_denominator=10):
    """
    计算β值对应的最小整数比
    
    Args:
        beta: β值（Y/X的比例）
        max_denominator: 最大分母限制（避免手数过大）
    
    Returns:
        (lots_y, lots_x): 最小整数比
    
    Examples:
        beta=0.5 -> (1, 2)
        beta=1.5 -> (3, 2)
        beta=0.33 -> (1, 3)
        beta=2.0 -> (2, 1)
    """
    # 处理特殊情况
    if beta <= 0:
        return (1, 1)
    
    # 使用分数类找最简分数
    frac = Fraction(beta).limit_denominator(max_denominator)
    
    # Y:X = beta:1，所以 lots_x:lots_y = 1:beta
    # 即 lots_y/lots_x = beta
    lots_y = frac.numerator
    lots_x = frac.denominator
    
    # 确保至少1手
    if lots_y == 0:
        lots_y = 1
    if lots_x == 0:
        lots_x = 1
    
    return (lots_y, lots_x)

def test_ratios():
    """测试各种β值的最小整数比"""
    test_cases = [
        0.5,    # 1:2
        1.5,    # 3:2
        0.33,   # 1:3
        2.0,    # 2:1
        0.85,   # 接近17:20
        3.5,    # 7:2
        0.1,    # 1:10
        10.0,   # 10:1
    ]
    
    print("Beta -> (Y, X) -> Actual Ratio")
    print("-" * 40)
    for beta in test_cases:
        lots_y, lots_x = calculate_min_integer_ratio(beta)
        actual_ratio = lots_y / lots_x
        print(f"{beta:6.2f} -> ({lots_y:2d}, {lots_x:2d}) -> {actual_ratio:6.3f}")

if __name__ == "__main__":
    test_ratios()