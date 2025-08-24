"""
手数计算模块
负责根据β值计算最优手数配比，并应用资金约束
对应需求：REQ-4.1
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Any
import math


@dataclass
class PositionSizingConfig:
    """手数计算配置"""
    max_denominator: int = 10      # 最大分母
    min_lots: int = 1               # 最小手数
    max_lots_per_leg: int = 100    # 每腿最大手数
    margin_rate: float = 0.12       # 保证金率
    position_weight: float = 0.05   # 默认仓位权重


class PositionSizer:
    """
    手数计算器
    实现两步计算法：
    1. 根据β和价格/乘数计算最小整数比
    2. 应用资金约束进行整数倍缩放
    """
    
    def __init__(self, config: PositionSizingConfig):
        """
        初始化手数计算器
        
        Args:
            config: 手数计算配置
        """
        self.config = config
        self.max_denominator = config.max_denominator
        self.min_lots = config.min_lots
        self.max_lots_per_leg = config.max_lots_per_leg
        self.margin_rate = config.margin_rate
    
    def calculate_min_integer_ratio(
        self,
        beta: float,
        price_x: float,
        price_y: float,
        multiplier_x: float,
        multiplier_y: float
    ) -> Dict[str, Any]:
        """
        REQ-4.1.2: 计算考虑价格和乘数的最小整数比
        
        Args:
            beta: 动态β值（对冲系数）
            price_x: X品种当前价格
            price_y: Y品种当前价格
            multiplier_x: X品种合约乘数
            multiplier_y: Y品种合约乘数
            
        Returns:
            {
                'lots_x': int,              # X品种最小手数
                'lots_y': int,              # Y品种最小手数
                'effective_ratio': float,   # 有效对冲比h*
                'nominal_error_pct': float  # 名义价值误差百分比
            }
        """
        # REQ-4.1.2.1: 计算有效对冲比 h* = β × (Py × My) / (Px × Mx)
        h_star = beta * (price_y * multiplier_y) / (price_x * multiplier_x)
        
        # REQ-4.1.2.2: 使用Fraction类对h*进行连分数逼近
        # REQ-4.1.2.3: max_denominator限制
        frac = Fraction(h_star).limit_denominator(self.max_denominator)
        
        # 得到初始整数对
        lots_x = frac.numerator
        lots_y = frac.denominator
        
        # REQ-4.1.2.4: 确保每腿至少min_lots手
        if lots_x < self.min_lots or lots_y < self.min_lots:
            # 需要等比例放大
            if lots_x == 0 and lots_y == 0:
                # 特殊情况：比例太小导致都是0
                lots_x = self.min_lots
                lots_y = self.min_lots
            elif lots_x == 0:
                lots_x = self.min_lots
                lots_y = max(self.min_lots, lots_y)
            elif lots_y == 0:
                lots_y = self.min_lots
                lots_x = max(self.min_lots, lots_x)
            else:
                scale_factor = max(
                    self.min_lots / lots_x,
                    self.min_lots / lots_y
                )
                lots_x = max(self.min_lots, int(lots_x * scale_factor))
                lots_y = max(self.min_lots, int(lots_y * scale_factor))
        
        # REQ-4.1.2.5: 计算名义价值匹配误差
        nominal_x = lots_x * price_x * multiplier_x
        nominal_y_target = lots_y * price_y * multiplier_y * beta
        
        if nominal_y_target > 0:
            nominal_error_pct = abs(nominal_x - nominal_y_target) / nominal_y_target * 100
        else:
            nominal_error_pct = 0.0
        
        return {
            'lots_x': lots_x,
            'lots_y': lots_y,
            'effective_ratio': h_star,
            'nominal_error_pct': nominal_error_pct
        }
    
    def calculate_position_size(
        self,
        min_lots: Dict[str, int],
        prices: Dict[str, float],
        multipliers: Dict[str, float],
        total_capital: float,
        position_weight: float = 0.05
    ) -> Dict[str, Any]:
        """
        REQ-4.1.3: 应用资金约束，整数倍缩放手数
        
        Args:
            min_lots: 最小整数对 {'lots_x': nx, 'lots_y': ny}
            prices: 当前价格 {'x': price_x, 'y': price_y}
            multipliers: 合约乘数 {'x': mult_x, 'y': mult_y}
            total_capital: 总资金
            position_weight: 仓位权重（默认0.05）
            
        Returns:
            {
                'final_lots_x': int,        # 最终X品种手数
                'final_lots_y': int,        # 最终Y品种手数
                'scaling_factor': int,      # 整数倍缩放系数k
                'allocated_capital': float, # 分配的资金
                'margin_required': float,   # 实际占用保证金
                'position_value': float,    # 名义价值
                'utilization_rate': float,  # 资金利用率
                'can_trade': bool,          # 是否可交易
                'reason': str              # 不可交易原因（如有）
            }
        """
        # REQ-4.1.1.1: 每个配对独立分配资金
        # REQ-4.1.1.2: 默认position_weight = 0.05
        allocated_capital = total_capital * position_weight
        
        # 获取最小整数对
        nx = min_lots['lots_x']
        ny = min_lots['lots_y']
        price_x = prices['x']
        price_y = prices['y']
        mult_x = multipliers['x']
        mult_y = multipliers['y']
        
        # REQ-4.1.3.1: 计算最小整数对所需保证金
        min_margin_x = nx * price_x * mult_x * self.margin_rate
        min_margin_y = ny * price_y * mult_y * self.margin_rate
        min_margin_required = min_margin_x + min_margin_y
        
        # REQ-4.1.3.2: 计算整数倍缩放系数 k = floor(allocated × 0.95 / margin)
        if min_margin_required > 0:
            k = int(allocated_capital * 0.95 / min_margin_required)
        else:
            k = 0
        
        # REQ-4.1.3.4: 如果k=0，资金不足
        if k == 0:
            return {
                'final_lots_x': 0,
                'final_lots_y': 0,
                'scaling_factor': 0,
                'allocated_capital': allocated_capital,
                'margin_required': 0,
                'position_value': 0,
                'utilization_rate': 0,
                'can_trade': False,
                'reason': 'Insufficient capital for minimum lots'
            }
        
        # REQ-4.1.3.3: 最终手数 = 最小手数 × k
        final_lots_x = nx * k
        final_lots_y = ny * k
        
        # REQ-4.1.3.5: 检查最大手数限制
        if final_lots_x > self.max_lots_per_leg or final_lots_y > self.max_lots_per_leg:
            # 需要限制在最大手数内
            max_k_x = self.max_lots_per_leg // nx if nx > 0 else k
            max_k_y = self.max_lots_per_leg // ny if ny > 0 else k
            k = min(k, max_k_x, max_k_y)
            
            # 重新计算最终手数
            final_lots_x = nx * k
            final_lots_y = ny * k
        
        # 计算实际保证金和名义价值
        actual_margin_x = final_lots_x * price_x * mult_x * self.margin_rate
        actual_margin_y = final_lots_y * price_y * mult_y * self.margin_rate
        actual_margin = actual_margin_x + actual_margin_y
        
        position_value_x = final_lots_x * price_x * mult_x
        position_value_y = final_lots_y * price_y * mult_y
        total_position_value = position_value_x + position_value_y
        
        # REQ-4.1.3.6: 计算资金利用率
        utilization_rate = actual_margin / allocated_capital if allocated_capital > 0 else 0
        
        return {
            'final_lots_x': final_lots_x,
            'final_lots_y': final_lots_y,
            'scaling_factor': k,
            'allocated_capital': allocated_capital,
            'margin_required': actual_margin,
            'position_value': total_position_value,
            'utilization_rate': utilization_rate,
            'can_trade': True,
            'reason': ''
        }