#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块时间轴配置测试用例
测试REQ-3.0.x相关的时间轴计算和阶段划分功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.signal_generation import SignalGenerator


class TestTimelineConfiguration:
    """测试时间轴配置功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建测试数据
        dates = pd.date_range('2024-04-01', '2024-08-20', freq='D')
        np.random.seed(42)
        
        # 生成协整的价格序列
        x_prices = np.cumsum(np.random.randn(len(dates)) * 0.01) + 10
        noise = np.random.randn(len(dates)) * 0.05  # 增大噪声以产生信号
        y_prices = 1.5 * x_prices + 2.0 + noise  # beta=1.5, alpha=2.0
        
        self.test_data = pd.DataFrame({
            'X': x_prices,
            'Y': y_prices
        }, index=dates)
        
        self.signal_start_date = '2024-07-01'
        self.kalman_warmup_days = 30
        self.ols_training_days = 60
    
    def test_req_3_0_1_signal_start_date_config(self):
        """REQ-3.0.1: 信号期起点用户指定"""
        # 测试不同的信号开始日期
        test_dates = ['2024-07-01', '2024-06-15', '2024-08-01']
        
        for start_date in test_dates:
            generator = SignalGenerator(
                signal_start_date=start_date,
                kalman_warmup_days=30,
                ols_training_days=60
            )
            
            # 验证配置正确设置
            assert generator.signal_start_date == start_date
            assert generator.kalman_warmup_days == 30
            assert generator.ols_training_days == 60
    
    def test_req_3_0_2_kalman_warmup_configurable(self):
        """REQ-3.0.2: Kalman预热期可配置"""
        test_warmup_days = [15, 30, 45, 60]
        
        for warmup_days in test_warmup_days:
            generator = SignalGenerator(
                signal_start_date=self.signal_start_date,
                kalman_warmup_days=warmup_days,
                ols_training_days=60
            )
            
            assert generator.kalman_warmup_days == warmup_days
    
    def test_req_3_0_3_ols_training_configurable(self):
        """REQ-3.0.3: OLS训练期可配置"""
        test_training_days = [30, 60, 90, 120]
        
        for training_days in test_training_days:
            generator = SignalGenerator(
                signal_start_date=self.signal_start_date,
                kalman_warmup_days=30,
                ols_training_days=training_days
            )
            
            assert generator.ols_training_days == training_days
    
    def test_req_3_0_4_auto_calculate_data_range(self):
        """REQ-3.0.4: 自动计算数据范围"""
        generator = SignalGenerator(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        # 测试时间轴计算
        signal_start = pd.to_datetime('2024-07-01')
        expected_kalman_start = signal_start - timedelta(days=30)  # 2024-06-01
        expected_ols_start = expected_kalman_start - timedelta(days=60)  # 2024-04-02
        
        # 验证计算方法
        actual_data_start = generator._calculate_data_start_date()
        assert actual_data_start == expected_ols_start.strftime('%Y-%m-%d')
    
    def test_req_3_0_5_time_phases(self):
        """REQ-3.0.5: 时间阶段划分"""
        generator = SignalGenerator(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60,
            z_open=1.5  # 降低阈值以便测试
        )
        
        # 处理测试配对
        x_data = self.test_data['X']
        y_data = self.test_data['Y']
        
        signals = generator.process_pair(
            pair_name='TEST_PAIR',
            x_data=x_data,
            y_data=y_data
        )
        
        # 验证时间阶段划分
        signal_start = pd.to_datetime('2024-07-01')
        kalman_start = signal_start - timedelta(days=30)
        ols_start = kalman_start - timedelta(days=60)
        
        # 检查各阶段的信号
        ols_phase = signals[signals['date'] < kalman_start]
        kalman_phase = signals[(signals['date'] >= kalman_start) & (signals['date'] < signal_start)]
        signal_phase = signals[signals['date'] >= signal_start]
        
        # OLS阶段：只有'ols_training'状态
        assert all(ols_phase['phase'] == 'ols_training')
        assert all(ols_phase['signal'] == 'warm_up')
        
        # Kalman预热阶段：'kalman_warmup'状态，不出交易信号
        assert all(kalman_phase['phase'] == 'kalman_warmup')
        assert all(kalman_phase['signal'] == 'warm_up')
        
        # 信号生成阶段：'signal_generation'状态，有实际交易信号
        assert all(signal_phase['phase'] == 'signal_generation')
        assert len(signal_phase[signal_phase['signal'].isin(['open_long', 'open_short', 'close'])]) > 0
    
    def test_req_3_0_6_parameter_interface(self):
        """REQ-3.0.6: 参数接口完整性"""
        # 测试所有必需的参数
        generator = SignalGenerator(
            signal_start_date='2024-07-01',
            kalman_warmup_days=25,
            ols_training_days=50,
            z_open=2.5,
            z_close=0.3,
            max_holding_days=20
        )
        
        # 验证所有参数都正确设置
        assert generator.signal_start_date == '2024-07-01'
        assert generator.kalman_warmup_days == 25
        assert generator.ols_training_days == 50
        assert generator.z_open == 2.5
        assert generator.z_close == 0.3
        assert generator.max_holding_days == 20
    
    def test_timeline_calculation_example(self):
        """测试需求文档中的时间轴计算示例"""
        generator = SignalGenerator(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        # 计算各阶段日期
        signal_start = pd.to_datetime('2024-07-01')
        kalman_start = signal_start - timedelta(days=30)  # 2024-06-01
        ols_start = kalman_start - timedelta(days=60)     # 2024-04-02
        
        # 验证计算结果与文档示例一致
        assert kalman_start.strftime('%Y-%m-%d') == '2024-06-01'
        assert ols_start.strftime('%Y-%m-%d') == '2024-04-02'
        
        # 验证数据范围计算
        expected_data_start = '2024-04-02'
        actual_data_start = generator._calculate_data_start_date()
        assert actual_data_start == expected_data_start
    
    def test_phase_transitions(self):
        """测试阶段转换的正确性"""
        generator = SignalGenerator(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        # 处理测试数据
        signals = generator.process_pair(
            pair_name='TEST_PAIR',
            x_data=self.test_data['X'],
            y_data=self.test_data['Y']
        )
        
        # 验证阶段转换无重叠
        phases = signals['phase'].unique()
        expected_phases = ['ols_training', 'kalman_warmup', 'signal_generation']
        
        for phase in expected_phases:
            assert phase in phases, f"Missing phase: {phase}"
        
        # 验证阶段按时间顺序排列
        phase_dates = signals.groupby('phase')['date'].agg(['min', 'max'])
        
        # OLS阶段应该最早
        # Kalman预热阶段在中间
        # 信号生成阶段最晚
        assert phase_dates.loc['ols_training', 'max'] <= phase_dates.loc['kalman_warmup', 'min']
        assert phase_dates.loc['kalman_warmup', 'max'] <= phase_dates.loc['signal_generation', 'min']
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效的信号开始日期格式
        with pytest.raises((ValueError, TypeError)):
            SignalGenerator(
                signal_start_date='invalid-date',
                kalman_warmup_days=30,
                ols_training_days=60
            )
        
        # 测试负数天数
        with pytest.raises(ValueError):
            SignalGenerator(
                signal_start_date='2024-07-01',
                kalman_warmup_days=-10,
                ols_training_days=60
            )
        
        with pytest.raises(ValueError):
            SignalGenerator(
                signal_start_date='2024-07-01',
                kalman_warmup_days=30,
                ols_training_days=-20
            )
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 旧版本的调用方式应该仍然可以工作（使用默认值）
        generator = SignalGenerator(
            signal_start_date='2024-07-01'  # 只提供必需参数
        )
        
        # 验证默认值
        assert generator.kalman_warmup_days == 30
        assert generator.ols_training_days == 60
        assert generator.z_open == 2.0
        assert generator.z_close == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])