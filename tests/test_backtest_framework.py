#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测框架测试用例 - 100%对齐需求文档
基于: /docs/Requirements/04_backtest_framework.md

测试覆盖：
1. 手数计算（正负Beta）
2. 交易执行（考虑Beta符号）
3. 止损逻辑
4. PnL计算
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any


# ============================================================================
# Test Case 4.1: 手数计算测试（包含负Beta）
# 需求: REQ-4.1.2
# ============================================================================

class TestPositionSizing:
    """测试手数计算模块"""
    
    def test_positive_beta_calculation(self):
        """
        测试1：正Beta情况
        需求: REQ-4.1.2.1 ~ REQ-4.1.2.6
        """
        from lib.backtest.position_sizing import PositionSizer
        
        sizer = PositionSizer()
        result = sizer.calculate_min_integer_ratio(
            beta=0.85,
            price_x=60000,
            price_y=140000,
            multiplier_x=5,
            multiplier_y=1
        )
        
        # h* = 0.85 × (140000×1)/(60000×5) = 0.397
        # Fraction(0.397).limit_denominator(10) ≈ 2/5
        assert result['lots_x'] == 2
        assert result['lots_y'] == 5
        assert result['beta_sign'] == 1
        assert abs(result['nominal_error_pct']) < 5
        
    def test_negative_beta_calculation(self):
        """
        测试2：负Beta情况
        需求: REQ-4.1.2.7 - 手数计算时使用abs(β)，但保留β的符号信息
        """
        from lib.backtest.position_sizing import PositionSizer
        
        sizer = PositionSizer()
        result = sizer.calculate_min_integer_ratio(
            beta=-0.5,  # 负Beta
            price_x=60000,
            price_y=140000,
            multiplier_x=5,
            multiplier_y=1
        )
        
        # h* = 0.5 × (140000×1)/(60000×5) = 0.233
        # Fraction(0.233).limit_denominator(10) ≈ 1/4
        assert result['lots_x'] == 1
        assert result['lots_y'] == 4
        assert result['beta_sign'] == -1  # 保留负号信息


# ============================================================================
# Test Case 4.2: 交易执行测试（负Beta）
# 需求: REQ-4.2.1
# ============================================================================

class TestTradeExecution:
    """测试交易执行模块"""
    
    def test_positive_beta_long_signal(self):
        """
        正Beta + Long信号 = 买Y卖X
        需求: REQ-4.2.1.4
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'CU-SN', 'beta': 0.98},
            lots={'x': 2, 'y': 5},
            prices={'x': 60000, 'y': 140000},
            signal_type='long'
        )
        
        assert position.action_x == 'sell'  # 卖X
        assert position.action_y == 'buy'   # 买Y
        
    def test_positive_beta_short_signal(self):
        """
        正Beta + Short信号 = 卖Y买X
        需求: REQ-4.2.1.5
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'CU-SN', 'beta': 0.98},
            lots={'x': 2, 'y': 5},
            prices={'x': 60000, 'y': 140000},
            signal_type='short'
        )
        
        assert position.action_x == 'buy'   # 买X
        assert position.action_y == 'sell'  # 卖Y
        
    def test_negative_beta_long_signal(self):
        """
        负Beta + Long信号 = 买Y买X（同向）
        需求: REQ-4.2.1.6
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'AG-NI', 'beta': -0.5},
            lots={'x': 2, 'y': 5},
            prices={'x': 60000, 'y': 140000},
            signal_type='long'
        )
        
        assert position.action_x == 'buy'  # 买X
        assert position.action_y == 'buy'  # 买Y
        
    def test_negative_beta_short_signal(self):
        """
        负Beta + Short信号 = 卖Y卖X（同向）
        需求: REQ-4.2.1.7
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'AG-NI', 'beta': -0.5},
            lots={'x': 2, 'y': 5},
            prices={'x': 60000, 'y': 140000},
            signal_type='short'
        )
        
        assert position.action_x == 'sell'  # 卖X
        assert position.action_y == 'sell'  # 卖Y


# ============================================================================
# Test Case 4.2.2: PnL计算测试（考虑Beta符号）
# 需求: REQ-4.2.2.4 ~ REQ-4.2.2.9
# ============================================================================

class TestPnLCalculation:
    """测试PnL计算逻辑"""
    
    def test_positive_beta_long_pnl(self):
        """
        正Beta + Long: 买Y卖X的PnL计算
        需求: REQ-4.2.2.5, REQ-4.2.2.6
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        
        # 模拟开仓
        position = type('Position', (), {})()
        position.pair = 'CU-SN'
        position.beta = 0.98
        position.direction = 'long'
        position.lots_x = 2
        position.lots_y = 5
        position.open_price_x = 60000
        position.open_price_y = 140000
        position.action_x = 'sell'
        position.action_y = 'buy'
        
        # 计算PnL（价格都上涨）
        trade = executor.execute_close(
            position=position,
            prices={'x': 61000, 'y': 142000},
            reason='signal'
        )
        
        # 正Beta + Long: 买Y卖X
        # X亏损 = -2 × (61000-60000) × 5 = -10000 (卖X，价格涨了亏损)
        # Y盈利 = 5 × (142000-140000) × 1 = 10000 (买Y，价格涨了盈利)
        # 总毛利 = 0
        assert abs(trade.gross_pnl - 0) < 100  # 允许小误差
        
    def test_negative_beta_long_pnl(self):
        """
        负Beta + Long: 买Y买X的PnL计算
        需求: REQ-4.2.2.7, REQ-4.2.2.8
        """
        from lib.backtest.trade_executor import TradeExecutor
        
        executor = TradeExecutor()
        
        # 模拟开仓
        position = type('Position', (), {})()
        position.pair = 'AG-NI'
        position.beta = -0.5
        position.direction = 'long'
        position.lots_x = 2
        position.lots_y = 5
        position.open_price_x = 60000
        position.open_price_y = 140000
        position.action_x = 'buy'
        position.action_y = 'buy'
        
        # 计算PnL（价格都上涨）
        trade = executor.execute_close(
            position=position,
            prices={'x': 61000, 'y': 142000},
            reason='signal'
        )
        
        # 负Beta + Long: 买Y买X（同向）
        # X盈利 = 2 × (61000-60000) × 5 = 10000
        # Y盈利 = 5 × (142000-140000) × 1 = 10000
        # 总毛利 = 20000
        assert abs(trade.gross_pnl - 20000) < 100


# ============================================================================
# Test Case 4.3: 止损测试
# 需求: REQ-4.3.1
# ============================================================================

class TestStopLoss:
    """测试止损逻辑"""
    
    def test_stop_loss_trigger(self):
        """
        测试：触发止损（亏损超过10%）
        需求: REQ-4.3.1.3 - 触发条件：pnl_pct <= -stop_loss_pct
        """
        from lib.backtest.risk_manager import RiskManager
        
        manager = RiskManager(stop_loss_pct=0.10)  # 10%止损
        
        position = type('Position', (), {})()
        position.pair = 'CU-SN'
        
        allocated_capital = 250000  # 分配的资金
        
        # 测试：触发止损（亏损超过10%）
        current_pnl = -26000  # 亏损26000
        should_stop, reason = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        # -26000/250000 = -10.4% < -10%
        assert should_stop == True
        assert 'stop_loss' in reason.lower()
        
    def test_stop_loss_not_trigger(self):
        """
        测试：不触发（亏损未超过10%）
        需求: REQ-4.3.1.3
        """
        from lib.backtest.risk_manager import RiskManager
        
        manager = RiskManager(stop_loss_pct=0.10)
        
        position = type('Position', (), {})()
        position.pair = 'CU-SN'
        
        allocated_capital = 250000
        
        # 测试：不触发（亏损未超过10%）
        current_pnl = -20000  # 亏损20000
        should_stop, _ = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        # -20000/250000 = -8% > -10%
        assert should_stop == False
        
    def test_stop_loss_with_profit(self):
        """
        测试：盈利时不应触发止损
        需求: REQ-4.3.1.3 - 只有亏损才触发
        """
        from lib.backtest.risk_manager import RiskManager
        
        manager = RiskManager(stop_loss_pct=0.10)
        
        position = type('Position', (), {})()
        position.pair = 'CU-SN'
        
        allocated_capital = 250000
        
        # 测试：盈利时不触发
        current_pnl = 26000  # 盈利26000
        should_stop, _ = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        # 26000/250000 = 10.4% > 0，不应触发止损
        assert should_stop == False


# ============================================================================
# Test Case 4.3.2: 时间止损测试
# 需求: REQ-4.3.2
# ============================================================================

class TestTimeStop:
    """测试时间止损"""
    
    def test_time_stop_trigger(self):
        """
        测试：持仓超过30天触发时间止损
        需求: REQ-4.3.2.2 - 触发条件：days >= max_holding_days
        """
        from lib.backtest.risk_manager import RiskManager
        
        manager = RiskManager(max_holding_days=30)
        
        position = type('Position', (), {})()
        position.open_date = datetime(2024, 1, 1)
        
        # 测试：31天后触发
        current_date = datetime(2024, 2, 1)
        should_stop, reason = manager.check_time_stop(position, current_date)
        
        assert should_stop == True
        assert 'time' in reason.lower()
        
    def test_time_stop_not_trigger(self):
        """
        测试：持仓未超过30天不触发
        需求: REQ-4.3.2.2
        """
        from lib.backtest.risk_manager import RiskManager
        
        manager = RiskManager(max_holding_days=30)
        
        position = type('Position', (), {})()
        position.open_date = datetime(2024, 1, 1)
        
        # 测试：29天不触发
        current_date = datetime(2024, 1, 30)
        should_stop, _ = manager.check_time_stop(position, current_date)
        
        assert should_stop == False


# ============================================================================
# 集成测试：完整交易流程
# ============================================================================

class TestIntegration:
    """测试完整交易流程"""
    
    def test_complete_trade_flow_positive_beta(self):
        """测试正Beta完整交易流程"""
        from lib.backtest.position_sizing import PositionSizer
        from lib.backtest.trade_executor import TradeExecutor
        from lib.backtest.risk_manager import RiskManager
        
        # 1. 计算手数
        sizer = PositionSizer()
        lots = sizer.calculate_min_integer_ratio(
            beta=0.98,
            price_x=60000,
            price_y=140000,
            multiplier_x=5,
            multiplier_y=1
        )
        
        # 2. 开仓
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'CU-SN', 'beta': 0.98},
            lots={'x': lots['lots_x'], 'y': lots['lots_y']},
            prices={'x': 60000, 'y': 140000},
            signal_type='long'
        )
        
        # 3. 风险检查
        manager = RiskManager(stop_loss_pct=0.10)
        
        # 模拟亏损5%
        current_pnl = -12500  # 5%亏损
        allocated_capital = 250000
        should_stop, _ = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        assert should_stop == False
        
        # 4. 平仓
        trade = executor.execute_close(
            position=position,
            prices={'x': 59000, 'y': 141000},
            reason='signal'
        )
        
        # 验证交易记录
        assert trade.close_reason == 'signal'
        assert trade.pair == 'CU-SN'
        
    def test_complete_trade_flow_negative_beta(self):
        """测试负Beta完整交易流程"""
        from lib.backtest.position_sizing import PositionSizer
        from lib.backtest.trade_executor import TradeExecutor
        from lib.backtest.risk_manager import RiskManager
        
        # 1. 计算手数
        sizer = PositionSizer()
        lots = sizer.calculate_min_integer_ratio(
            beta=-0.22,
            price_x=5000,
            price_y=50000,
            multiplier_x=15,
            multiplier_y=1
        )
        
        # 2. 开仓（负Beta + short = 卖Y卖X）
        executor = TradeExecutor()
        position = executor.execute_open(
            pair_info={'pair': 'AG-NI', 'beta': -0.22},
            lots={'x': lots['lots_x'], 'y': lots['lots_y']},
            prices={'x': 5000, 'y': 50000},
            signal_type='short'
        )
        
        assert position.action_x == 'sell'
        assert position.action_y == 'sell'
        
        # 3. 止损检查（亏损15%应触发）
        manager = RiskManager(stop_loss_pct=0.10)
        
        current_pnl = -37500  # 15%亏损
        allocated_capital = 250000
        should_stop, reason = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        assert should_stop == True
        assert 'stop' in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])