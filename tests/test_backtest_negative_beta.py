#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测框架负Beta测试用例
测试负Beta配对的交易方向和PnL计算正确性
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from fractions import Fraction
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 这里假设回测模块的结构
try:
    from lib.backtest.position_sizing import PositionSizer
    from lib.backtest.trade_executor import TradeExecutor
    from lib.backtest.risk_manager import RiskManager
    from lib.backtest.performance import PerformanceAnalyzer
    from lib.backtest.data_structures import Position, Trade, PositionSizingConfig
except ImportError:
    pytest.skip("回测模块尚未实现", allow_module_level=True)


class TestPositionSizingNegativeBeta:
    """测试手数计算模块的负Beta支持"""
    
    def test_negative_beta_ratio_calculation(self):
        """测试负Beta的手数比例计算"""
        config = PositionSizingConfig(
            max_denominator=10,
            min_lots=1,
            max_lots_per_leg=100,
            margin_rate=0.12
        )
        sizer = PositionSizer(config)
        
        # 测试负Beta：-0.5
        result = sizer.calculate_min_integer_ratio(
            beta=-0.5,  # 负Beta
            price_x=60000,
            price_y=140000,
            multiplier_x=5,
            multiplier_y=1
        )
        
        # h* = abs(-0.5) × (140000×1)/(60000×5) = 0.5 × 140000/300000 = 0.233
        # Fraction(0.233).limit_denominator(10) ≈ 1/4
        assert result['lots_x'] == 1
        assert result['lots_y'] == 4
        assert result['beta_sign'] == -1  # 保留负号信息
        assert abs(result['nominal_error_pct']) < 10  # 误差在合理范围
        
        # 验证有效对冲比
        expected_ratio = 0.5 * 140000 / (60000 * 5)  # 使用abs(beta)
        assert abs(result['effective_ratio'] - expected_ratio) < 0.01
    
    def test_positive_beta_comparison(self):
        """对比正Beta和负Beta的计算结果"""
        config = PositionSizingConfig()
        sizer = PositionSizer(config)
        
        # 正Beta
        result_pos = sizer.calculate_min_integer_ratio(
            beta=0.8,
            price_x=50000, price_y=120000,
            multiplier_x=10, multiplier_y=5
        )
        
        # 负Beta（绝对值相同）
        result_neg = sizer.calculate_min_integer_ratio(
            beta=-0.8,
            price_x=50000, price_y=120000,
            multiplier_x=10, multiplier_y=5
        )
        
        # 手数应该相同（都用abs(beta)计算）
        assert result_pos['lots_x'] == result_neg['lots_x']
        assert result_pos['lots_y'] == result_neg['lots_y']
        
        # 但符号标识不同
        assert result_pos['beta_sign'] == 1
        assert result_neg['beta_sign'] == -1
        
    def test_zero_beta_edge_case(self):
        """测试Beta为0的边界情况"""
        config = PositionSizingConfig()
        sizer = PositionSizer(config)
        
        with pytest.raises(ValueError, match="Beta不能为0"):
            sizer.calculate_min_integer_ratio(
                beta=0.0,
                price_x=50000, price_y=120000,
                multiplier_x=10, multiplier_y=5
            )
    
    def test_extreme_negative_beta(self):
        """测试极端负Beta值"""
        config = PositionSizingConfig(max_denominator=20)
        sizer = PositionSizer(config)
        
        # 极端负Beta：-2.5
        result = sizer.calculate_min_integer_ratio(
            beta=-2.5,
            price_x=30000, price_y=100000,
            multiplier_x=5, multiplier_y=10
        )
        
        # h* = 2.5 × (100000×10)/(30000×5) = 16.67
        # 应该能找到合理的整数比
        assert result['lots_x'] >= 1
        assert result['lots_y'] >= 1
        assert result['beta_sign'] == -1
        assert result['effective_ratio'] > 10  # 高对冲比


class TestTradeExecutionNegativeBeta:
    """测试交易执行模块的负Beta支持"""
    
    def test_negative_beta_long_signal(self):
        """测试负Beta + Long信号的交易方向"""
        executor = TradeExecutor()
        
        # 负Beta + Long信号 = 买Y买X
        position = executor.execute_open(
            pair_info={
                'pair': 'AG-NI',
                'symbol_x': 'AG',
                'symbol_y': 'NI', 
                'beta': -0.5
            },
            lots={'x': 2, 'y': 5},
            prices={'x': 4800, 'y': 14000},
            multipliers={'x': 15, 'y': 1},
            signal_type='long',
            timestamp=datetime.now()
        )
        
        # 验证交易方向
        assert position.direction == 'long'
        assert position.action_x == 'buy'   # 买X
        assert position.action_y == 'buy'   # 买Y
        assert position.beta == -0.5
        
        # 验证价格（考虑滑点）
        assert position.open_price_x > 4800  # 买入加滑点
        assert position.open_price_y > 14000 # 买入加滑点
    
    def test_negative_beta_short_signal(self):
        """测试负Beta + Short信号的交易方向"""
        executor = TradeExecutor()
        
        # 负Beta + Short信号 = 卖Y卖X
        position = executor.execute_open(
            pair_info={
                'pair': 'AG-NI',
                'symbol_x': 'AG',
                'symbol_y': 'NI',
                'beta': -0.8
            },
            lots={'x': 3, 'y': 2},
            prices={'x': 4800, 'y': 14000},
            multipliers={'x': 15, 'y': 1},
            signal_type='short',
            timestamp=datetime.now()
        )
        
        # 验证交易方向
        assert position.direction == 'short'
        assert position.action_x == 'sell'  # 卖X
        assert position.action_y == 'sell'  # 卖Y
        assert position.beta == -0.8
        
        # 验证价格（考虑滑点）
        assert position.open_price_x < 4800  # 卖出减滑点
        assert position.open_price_y < 14000 # 卖出减滑点
    
    def test_positive_beta_comparison(self):
        """对比正Beta和负Beta的交易方向"""
        executor = TradeExecutor()
        
        common_params = {
            'pair_info': {'pair': 'CU-AL', 'symbol_x': 'CU', 'symbol_y': 'AL'},
            'lots': {'x': 2, 'y': 4},
            'prices': {'x': 70000, 'y': 18000},
            'multipliers': {'x': 5, 'y': 5},
            'signal_type': 'long',
            'timestamp': datetime.now()
        }
        
        # 正Beta情况
        pos_beta = executor.execute_open(
            pair_info={**common_params['pair_info'], 'beta': 0.6},
            **{k: v for k, v in common_params.items() if k != 'pair_info'}
        )
        
        # 负Beta情况
        neg_beta = executor.execute_open(
            pair_info={**common_params['pair_info'], 'beta': -0.6},
            **{k: v for k, v in common_params.items() if k != 'pair_info'}
        )
        
        # 验证差异
        assert pos_beta.action_x == 'sell' and pos_beta.action_y == 'buy'  # 正Beta
        assert neg_beta.action_x == 'buy' and neg_beta.action_y == 'buy'   # 负Beta
    
    def test_negative_beta_pnl_calculation(self):
        """测试负Beta的PnL计算正确性"""
        executor = TradeExecutor()
        
        # 创建负Beta持仓（买Y买X）
        position = Position(
            position_id='test_001',
            pair='AG-NI',
            symbol_x='AG', symbol_y='NI',
            lots_x=2, lots_y=5,
            direction='long',
            action_x='buy', action_y='buy',
            open_date=datetime.now(),
            open_price_x=4800, open_price_y=14000,
            beta=-0.5,
            margin=50000
        )
        
        # 价格变化：都上涨1000点
        close_prices = {'x': 5800, 'y': 15000}
        
        trade = executor.execute_close(
            position=position,
            prices=close_prices,
            multipliers={'x': 15, 'y': 1},
            reason='signal',
            timestamp=datetime.now()
        )
        
        # 负Beta + Long（都买入）情况的PnL计算
        # X盈利 = 2 × (5800-4800) × 15 = 30000
        # Y盈利 = 5 × (15000-14000) × 1 = 5000
        # 总毛利 = 35000
        
        expected_pnl_x = 2 * (5800 - 4800) * 15  # 买入X的盈利
        expected_pnl_y = 5 * (15000 - 14000) * 1  # 买入Y的盈利
        expected_gross_pnl = expected_pnl_x + expected_pnl_y
        
        assert abs(trade.gross_pnl - expected_gross_pnl) < 100  # 允许小误差
        assert trade.gross_pnl > 0  # 应该盈利
    
    def test_negative_beta_loss_scenario(self):
        """测试负Beta配对的亏损场景"""
        executor = TradeExecutor()
        
        # 创建负Beta空头持仓（卖Y卖X）
        position = Position(
            position_id='test_002',
            pair='AU-AG',
            symbol_x='AU', symbol_y='AG',
            lots_x=1, lots_y=3,
            direction='short',
            action_x='sell', action_y='sell',
            open_date=datetime.now(),
            open_price_x=550, open_price_y=4800,
            beta=-0.3,
            margin=80000
        )
        
        # 价格变化：都上涨（空头亏损）
        close_prices = {'x': 570, 'y': 5000}
        
        trade = executor.execute_close(
            position=position,
            prices=close_prices,
            multipliers={'x': 1000, 'y': 15},
            reason='stop_loss',
            timestamp=datetime.now()
        )
        
        # 负Beta + Short（都卖出）情况的PnL计算
        # X亏损 = 1 × (550-570) × 1000 = -20000
        # Y亏损 = 3 × (4800-5000) × 15 = -9000
        # 总亏损 = -29000
        
        expected_pnl_x = 1 * (550 - 570) * 1000  # 卖出X的亏损
        expected_pnl_y = 3 * (4800 - 5000) * 15  # 卖出Y的亏损
        expected_gross_pnl = expected_pnl_x + expected_pnl_y
        
        assert abs(trade.gross_pnl - expected_gross_pnl) < 100
        assert trade.gross_pnl < 0  # 应该亏损


class TestRiskManagerNegativeBeta:
    """测试风险管理模块对负Beta配对的处理"""
    
    def test_negative_beta_stop_loss(self):
        """测试负Beta配对的止损触发"""
        manager = RiskManager(stop_loss_pct=0.15)  # 15%止损
        
        # 创建负Beta持仓
        position = Position(
            position_id='risk_test_001',
            pair='CU-ZN',
            beta=-0.8,
            margin=100000
        )
        
        allocated_capital = 500000  # 分配资金
        
        # 测试：达到止损线（亏损15%）
        current_pnl = -76000  # 亏损76000
        should_stop, reason = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        assert should_stop == True  # -76000/500000 = -15.2% < -15%
        assert 'stop_loss' in reason
        
        # 测试：未达到止损线
        current_pnl = -60000  # 亏损60000
        should_stop, reason = manager.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        assert should_stop == False  # -60000/500000 = -12% > -15%
    
    def test_negative_beta_unrealized_pnl_calculation(self):
        """测试负Beta持仓的浮动盈亏计算"""
        manager = RiskManager()
        
        # 负Beta多头持仓（买Y买X）
        position = Position(
            position_id='unrealized_test_001',
            pair='AL-NI',
            symbol_x='AL', symbol_y='NI',
            lots_x=4, lots_y=2,
            direction='long',
            action_x='buy', action_y='buy',
            open_price_x=18000, open_price_y=140000,
            beta=-0.7,
            margin=80000
        )
        
        # 当前价格
        current_prices = {'x': 18500, 'y': 142000}
        multipliers = {'x': 5, 'y': 1}
        
        unrealized_pnl = manager.calculate_unrealized_pnl(
            position, current_prices, multipliers
        )
        
        # 计算预期浮动盈亏
        # X盈利 = 4 × (18500-18000) × 5 = 10000
        # Y盈利 = 2 × (142000-140000) × 1 = 4000
        # 总盈利 = 14000
        
        expected_pnl = 4 * (18500 - 18000) * 5 + 2 * (142000 - 140000) * 1
        assert abs(unrealized_pnl - expected_pnl) < 100


class TestPerformanceAnalysisNegativeBeta:
    """测试绩效分析对负Beta配对的支持"""
    
    def test_negative_beta_pair_metrics(self):
        """测试负Beta配对的绩效指标计算"""
        analyzer = PerformanceAnalyzer()
        
        # 创建负Beta配对的交易历史
        trades = [
            Trade(
                trade_id='neg_001',
                pair='AG-NI',
                beta=-0.5,
                direction='long',
                gross_pnl=15000,
                net_pnl=14500,
                return_pct=0.029,  # 2.9%收益
                holding_days=7
            ),
            Trade(
                trade_id='neg_002', 
                pair='AG-NI',
                beta=-0.5,
                direction='short',
                gross_pnl=-8000,
                net_pnl=-8500,
                return_pct=-0.017,  # -1.7%亏损
                holding_days=12
            ),
            Trade(
                trade_id='neg_003',
                pair='AG-NI', 
                beta=-0.5,
                direction='long',
                gross_pnl=22000,
                net_pnl=21400,
                return_pct=0.043,  # 4.3%收益
                holding_days=5
            )
        ]
        
        pair_metrics = analyzer.calculate_pair_metrics('AG-NI', trades)
        
        # 验证基本统计
        assert pair_metrics['total_trades'] == 3
        assert pair_metrics['winning_trades'] == 2
        assert pair_metrics['losing_trades'] == 1
        assert pair_metrics['win_rate'] == 2/3
        
        # 验证收益统计
        assert pair_metrics['total_pnl'] == sum(t.net_pnl for t in trades)
        assert pair_metrics['avg_return_pct'] == pytest.approx(
            sum(t.return_pct for t in trades) / 3, abs=0.001
        )
        
        # 验证持仓统计
        assert pair_metrics['avg_holding_days'] == (7 + 12 + 5) / 3
        assert pair_metrics['beta'] == -0.5  # 应该记录Beta值
    
    def test_portfolio_with_mixed_beta_signs(self):
        """测试包含正负Beta配对的组合绩效"""
        analyzer = PerformanceAnalyzer()
        
        # 混合Beta的交易历史
        trades = [
            # 正Beta配对
            Trade(pair='CU-AL', beta=0.8, net_pnl=12000, return_pct=0.024),
            Trade(pair='CU-AL', beta=0.8, net_pnl=-5000, return_pct=-0.010),
            
            # 负Beta配对
            Trade(pair='AG-NI', beta=-0.5, net_pnl=8000, return_pct=0.016),
            Trade(pair='AG-NI', beta=-0.5, net_pnl=15000, return_pct=0.030),
            
            # 另一个负Beta配对
            Trade(pair='AU-AG', beta=-0.3, net_pnl=-3000, return_pct=-0.006)
        ]
        
        portfolio_metrics = analyzer.calculate_portfolio_metrics(trades)
        
        # 验证组合指标
        assert portfolio_metrics['total_trades'] == 5
        assert portfolio_metrics['total_pnl'] == 27000
        assert portfolio_metrics['positive_beta_pairs'] == 1
        assert portfolio_metrics['negative_beta_pairs'] == 2
        
        # 验证Beta分组统计
        beta_analysis = analyzer.analyze_by_beta_sign(trades)
        assert beta_analysis['positive_beta']['total_pnl'] == 7000
        assert beta_analysis['negative_beta']['total_pnl'] == 20000
        assert beta_analysis['negative_beta']['win_rate'] == 2/3


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])