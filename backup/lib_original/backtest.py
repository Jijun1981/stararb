"""
回测引擎模块
实现交易执行模拟、绩效指标计算、风险分析和配对贡献分析

Test: tests/test_backtest_engine.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎核心类"""
    
    def __init__(self, initial_capital: float = 5000000):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金，默认500万
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []  # 完整交易记录
        self.positions = {}  # 当前持仓
        self.margin_manager = MarginManager(initial_capital)
        self.position_tracker = PositionTracker()
        self.trade_recorder = TradeRecorder()
        
    def execute_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        TC-BE.1.1: 按时间顺序执行开仓和平仓信号
        
        Args:
            signals: 信号列表
            
        Returns:
            完整交易记录列表
        """
        # 按日期排序信号
        sorted_signals = sorted(signals, key=lambda x: x['date'])
        
        open_trades = {}  # 未平仓的交易记录
        completed_trades = []
        
        for signal in sorted_signals:
            date = signal['date']
            pair = signal['pair']
            action = signal['action']
            
            if action == 'open':
                # 检查是否已有持仓
                if not self.position_tracker.can_open_position(pair):
                    continue
                    
                # 创建开仓记录
                trade_id = f"{pair}_{date}"
                open_trades[trade_id] = {
                    'pair': pair,
                    'open_date': date,
                    'open_signal': signal
                }
                self.position_tracker.open_position(pair, signal)
                
            elif action == 'close':
                # 寻找对应的开仓记录
                matching_trade = None
                for trade_id, trade in open_trades.items():
                    if trade['pair'] == pair:
                        matching_trade = (trade_id, trade)
                        break
                        
                if matching_trade:
                    trade_id, trade = matching_trade
                    trade['close_date'] = date
                    trade['close_signal'] = signal
                    completed_trades.append(trade)
                    del open_trades[trade_id]
                    self.position_tracker.close_position(pair)
        
        return completed_trades


class MarginManager:
    """保证金管理器"""
    
    def __init__(self, initial_capital: float):
        """
        初始化保证金管理器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.margin_used = 0.0
        self.positions = {}
        
    def check_margin_requirement(self, position: Dict) -> bool:
        """
        TC-BE.1.4: 检查保证金要求
        
        Args:
            position: 持仓信息
            
        Returns:
            是否满足保证金要求
        """
        required_margin = position['notional_value'] * position['margin_rate']
        
        if required_margin <= self.available_capital:
            # 扣除保证金
            self.available_capital -= required_margin
            self.margin_used += required_margin
            self.positions[position['pair']] = {
                'margin': required_margin,
                'notional_value': position['notional_value']
            }
            return True
        else:
            return False


class PositionTracker:
    """持仓状态跟踪器"""
    
    def __init__(self):
        self.positions = {}
        
    def can_open_position(self, pair: str) -> bool:
        """
        TC-BE.1.7: 检查是否可以开仓（避免重复开仓）
        
        Args:
            pair: 配对名称
            
        Returns:
            是否可以开仓
        """
        return pair not in self.positions
        
    def has_position(self, pair: str) -> bool:
        """检查是否有持仓"""
        return pair in self.positions
        
    def open_position(self, pair: str, position_data: Dict):
        """开仓"""
        self.positions[pair] = position_data
        
    def close_position(self, pair: str):
        """平仓"""
        if pair in self.positions:
            del self.positions[pair]


class TradeRecorder:
    """交易记录器"""
    
    def create_record(self, trade_info: Dict) -> Dict:
        """
        TC-BE.1.8: 创建交易记录
        
        Args:
            trade_info: 交易信息
            
        Returns:
            完整的交易记录
        """
        # 计算持仓天数
        open_date = pd.to_datetime(trade_info['open_date'])
        close_date = pd.to_datetime(trade_info['close_date'])
        holding_days = (close_date - open_date).days
        
        # 计算交易成本
        notional_value = (trade_info.get('open_price_y', 1000) * 
                         trade_info.get('open_price_x', 800)) * 0.5  # 简化计算
        transaction_cost = calculate_transaction_cost({
            'notional_value_y': notional_value * 0.5,
            'notional_value_x': notional_value * 0.5,
            'cost_rate': 0.0002
        })
        
        # 计算净收益
        gross_pnl = trade_info.get('pnl', 0)
        net_pnl = gross_pnl - transaction_cost
        
        record = {
            'pair': trade_info['pair'],
            'open_date': trade_info['open_date'],
            'close_date': trade_info['close_date'],
            'holding_days': holding_days,
            'open_price_y': trade_info.get('open_price_y'),
            'open_price_x': trade_info.get('open_price_x'),
            'close_price_y': trade_info.get('close_price_y'),
            'close_price_x': trade_info.get('close_price_x'),
            'position_ratio': trade_info.get('position_ratio'),
            'pnl': gross_pnl,
            'transaction_cost': transaction_cost,
            'net_pnl': net_pnl
        }
        
        return record


def calculate_spread_pnl(trade_data: Dict) -> float:
    """
    TC-BE.1.2: 计算价差收益（使用真实合约乘数）
    
    Args:
        trade_data: 交易数据，包含:
            - position_ratio: 仓位比例 (如 "3:4")
            - open_prices: 开仓价格字典 {'Y': 价格, 'X': 价格}
            - close_prices: 平仓价格字典 {'Y': 价格, 'X': 价格}
            - y_symbol: Y腿品种代码 (可选)
            - x_symbol: X腿品种代码 (可选)
        
    Returns:
        收益金额
    """
    # 导入合约规格
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from configs.contract_specs import CONTRACT_SPECS
    except ImportError:
        # 如果无法导入，使用默认值
        CONTRACT_SPECS = {}
    
    # 解析仓位比例
    y_ratio, x_ratio = map(int, trade_data['position_ratio'].split(':'))
    
    # 获取品种代码（如果提供）
    y_symbol = trade_data.get('y_symbol')
    x_symbol = trade_data.get('x_symbol')
    
    # 如果提供了品种代码，使用真实合约乘数
    if y_symbol and x_symbol and CONTRACT_SPECS:
        y_multiplier = CONTRACT_SPECS.get(y_symbol, {}).get('multiplier', 10)
        x_multiplier = CONTRACT_SPECS.get(x_symbol, {}).get('multiplier', 10)
    else:
        # 根据价格范围推断品种并使用对应乘数
        y_price = trade_data['open_prices']['Y']
        x_price = trade_data['open_prices']['X']
        
        # 简化的品种识别逻辑
        if y_price > 100000:  # 可能是黄金
            y_multiplier = 1000
        elif y_price > 50000:  # 可能是铜或其他有色
            y_multiplier = 5
        elif y_price > 5000:  # 可能是白银
            y_multiplier = 15
        else:  # 默认
            y_multiplier = 10
            
        if x_price > 100000:  # 可能是锡
            x_multiplier = 1
        elif x_price > 50000:  # 可能是铜或其他有色
            x_multiplier = 5
        else:  # 默认
            x_multiplier = 10
    
    # 计算价格变化的收益
    # Y腿收益（做多）
    y_price_change = trade_data['close_prices']['Y'] - trade_data['open_prices']['Y']
    y_pnl = y_price_change * y_ratio * y_multiplier
    
    # X腿收益（做空，所以价格下跌盈利）
    x_price_change = trade_data['open_prices']['X'] - trade_data['close_prices']['X']
    x_pnl = x_price_change * x_ratio * x_multiplier
    
    # 总收益
    pnl = y_pnl + x_pnl
    
    return pnl


def calculate_transaction_cost(trade_data: Dict) -> float:
    """
    TC-BE.1.3: 计算交易成本
    
    Args:
        trade_data: 交易数据
        
    Returns:
        交易成本
    """
    total_notional = (trade_data['notional_value_y'] + 
                     trade_data['notional_value_x'])
    
    # 双边成本 = 名义价值 × 费率 × 2
    cost = total_notional * trade_data['cost_rate'] * 2
    
    return cost


def check_stop_loss(position: Dict) -> bool:
    """
    TC-BE.1.5: 检查止损触发
    
    Args:
        position: 持仓信息
        
    Returns:
        是否应该止损
    """
    loss_rate = (position['current_value'] - position['entry_value']) / position['entry_value']
    
    return loss_rate <= -position['stop_loss_rate']


def apply_slippage(order: Dict) -> float:
    """
    TC-BE.1.6: 应用滑点
    
    Args:
        order: 订单信息
        
    Returns:
        实际成交价格
    """
    signal_price = order['signal_price']
    slippage_ticks = order['slippage_ticks']
    tick_size = order['tick_size']
    direction = order['direction']
    
    # 计算滑点金额
    slippage_amount = slippage_ticks * tick_size
    
    # 买入向上滑，卖出向下滑
    if direction == 'buy':
        actual_price = signal_price + slippage_amount
    else:  # sell
        actual_price = signal_price - slippage_amount
        
    return actual_price


# 算法验证函数（要求至少两种方法）

def validate_spread_pnl_calculation(trade_data: Dict) -> Tuple[float, float]:
    """
    价差收益计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 价差方法（主实现）
    pnl1 = calculate_spread_pnl(trade_data)
    
    # 方法2: 分腿计算验证
    y_ratio, x_ratio = map(int, trade_data['position_ratio'].split(':'))
    multiplier = 10
    
    # Y腿收益（做多）
    y_pnl = ((trade_data['close_prices']['Y'] - trade_data['open_prices']['Y']) * 
             y_ratio * multiplier)
    
    # X腿收益（做空）
    x_pnl = ((trade_data['open_prices']['X'] - trade_data['close_prices']['X']) * 
             x_ratio * multiplier)
    
    pnl2 = y_pnl + x_pnl
    
    return pnl1, pnl2


def validate_transaction_cost_calculation(trade_data: Dict) -> Tuple[float, float]:
    """
    交易成本计算的双重验证
    
    Returns:
        (方法1结果, 方法2结果)
    """
    # 方法1: 总和方法（主实现）
    cost1 = calculate_transaction_cost(trade_data)
    
    # 方法2: 分别计算验证
    cost_y = trade_data['notional_value_y'] * trade_data['cost_rate'] * 2
    cost_x = trade_data['notional_value_x'] * trade_data['cost_rate'] * 2
    cost2 = cost_y + cost_x
    
    return cost1, cost2