#!/usr/bin/env python3
"""
信号状态机制单元测试
测试清晰的状态机逻辑：empty, holding_long, holding_short, open_*, close
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TestSignalStateMachine:
    """测试清晰的状态机制"""
    
    def test_initial_state_is_empty(self):
        """测试初始状态为空仓"""
        # 模拟generate_signal_with_clear_states函数
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            """改进的信号生成逻辑"""
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 初始状态：无持仓，Z-score未达到阈值
        signal = generate_signal_with_clear_states(
            z_score=1.5, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'empty'
    
    def test_open_long_signal(self):
        """测试开多头信号"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 无持仓，Z-score <= -2.0
        signal = generate_signal_with_clear_states(
            z_score=-2.5, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'open_long'
    
    def test_open_short_signal(self):
        """测试开空头信号"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 无持仓，Z-score >= 2.0
        signal = generate_signal_with_clear_states(
            z_score=2.8, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'open_short'
    
    def test_holding_long_state(self):
        """测试持多头仓位状态"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 有多头持仓，Z-score未达到平仓条件
        signal = generate_signal_with_clear_states(
            z_score=-1.2, position='long', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_long'
    
    def test_holding_short_state(self):
        """测试持空头仓位状态"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 有空头持仓，Z-score未达到平仓条件
        signal = generate_signal_with_clear_states(
            z_score=1.8, position='short', days_held=8,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_short'
    
    def test_close_by_zscore(self):
        """测试Z-score达到平仓条件"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 有持仓，|Z-score| <= 0.5
        signal = generate_signal_with_clear_states(
            z_score=0.3, position='long', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'
    
    def test_force_close_by_max_days(self):
        """测试强制平仓（超过最大持仓天数）"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 持仓天数 >= 最大持仓天数
        signal = generate_signal_with_clear_states(
            z_score=-1.8, position='long', days_held=30,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'
    
    def test_no_open_when_already_holding(self):
        """测试已持仓时不会开新仓（防重复开仓）"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 已有多头持仓，Z-score达到开空条件
        signal = generate_signal_with_clear_states(
            z_score=2.5, position='long', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_long'  # 不是open_short
    
    def test_position_state_update(self):
        """测试持仓状态更新逻辑"""
        def update_position_state(signal, position, days_held):
            """根据信号更新持仓状态"""
            if signal == 'open_long':
                return 'long', 1
            elif signal == 'open_short':
                return 'short', 1
            elif signal == 'close':
                return None, 0
            elif position:
                return position, days_held + 1
            else:
                return None, 0
        
        # 开多头
        position, days = update_position_state('open_long', None, 0)
        assert position == 'long'
        assert days == 1
        
        # 持仓第二天
        position, days = update_position_state('holding_long', 'long', 1)
        assert position == 'long'
        assert days == 2
        
        # 平仓
        position, days = update_position_state('close', 'long', 5)
        assert position is None
        assert days == 0
        
        # 平仓后第二天
        position, days = update_position_state('empty', None, 0)
        assert position is None
        assert days == 0


class TestSignalStateMachineIntegration:
    """测试状态机制的集成场景"""
    
    def test_complete_trading_cycle(self):
        """测试完整的交易周期"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        def update_position_state(signal, position, days_held):
            if signal == 'open_long':
                return 'long', 1
            elif signal == 'open_short':
                return 'short', 1
            elif signal == 'close':
                return None, 0
            elif position:
                return position, days_held + 1
            else:
                return None, 0
        
        # 模拟完整的交易周期
        z_scores = [1.5, -2.5, -1.8, -1.2, 0.3, 1.0, 2.8, 1.5, 0.2]
        expected_signals = [
            'empty',        # Z=1.5, 空仓等待
            'open_long',    # Z=-2.5, 开多头
            'holding_long', # Z=-1.8, 持仓中
            'holding_long', # Z=-1.2, 持仓中
            'close',        # Z=0.3, 平仓
            'empty',        # Z=1.0, 空仓等待
            'open_short',   # Z=2.8, 开空头
            'holding_short',# Z=1.5, 持仓中
            'close'         # Z=0.2, 平仓
        ]
        
        position = None
        days_held = 0
        
        for i, z_score in enumerate(z_scores):
            signal = generate_signal_with_clear_states(
                z_score=z_score, position=position, days_held=days_held,
                z_open=2.0, z_close=0.5, max_days=30
            )
            
            assert signal == expected_signals[i], \
                f"Day {i}: expected {expected_signals[i]}, got {signal} (Z={z_score})"
            
            position, days_held = update_position_state(signal, position, days_held)
    
    def test_extreme_zscore_during_holding(self):
        """测试持仓期间极端Z值的处理"""
        def generate_signal_with_clear_states(z_score, position, days_held, 
                                            z_open, z_close, max_days):
            if position and days_held >= max_days:
                return 'close'
            if position and abs(z_score) <= z_close:
                return 'close'
            if not position:
                if abs(z_score) >= z_open:
                    if z_score <= -z_open:
                        return 'open_long'
                    elif z_score >= z_open:
                        return 'open_short'
                return 'empty'
            if position == 'long':
                return 'holding_long'
            elif position == 'short':
                return 'holding_short'
            return 'empty'
        
        # 持多头仓位期间，Z-score再次达到极端值
        signal = generate_signal_with_clear_states(
            z_score=-3.5, position='long', days_held=10,  # 极端负Z，但已持多头
            z_open=2.2, z_close=0.3, max_days=30
        )
        assert signal == 'holding_long'  # 应该保持持仓状态，不开新仓
        
        # 持空头仓位期间，Z-score达到开多条件
        signal = generate_signal_with_clear_states(
            z_score=-2.8, position='short', days_held=15,  # 负Z达到开多条件，但已持空头
            z_open=2.2, z_close=0.3, max_days=30
        )
        assert signal == 'holding_short'  # 应该保持持仓状态，不开反向仓位


if __name__ == '__main__':
    pytest.main([__file__])