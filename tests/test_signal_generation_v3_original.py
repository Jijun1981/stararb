#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块V3测试用例 - 基于原版本KF的正确实现
基于需求文档: /docs/Requirements/03_signal_generation.md V3.1
使用经过验证的原版本Kalman滤波器: z = residual/√R
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.signal_generation import SignalGeneratorV3, OriginalKalmanFilter

class TestOriginalKalmanFilter:
    """测试原版本Kalman滤波器"""
    
    def test_initialization(self):
        """TC-3.1.1: 测试Kalman滤波器初始化"""
        kf = OriginalKalmanFilter(
            warmup=60,
            Q_beta=5e-6,
            Q_alpha=1e-5,
            R_init=0.005,
            R_adapt=True
        )
        
        # 验证参数设置
        assert kf.warmup == 60
        assert kf.Q[0,0] == 5e-6  # Q_beta
        assert kf.Q[1,1] == 1e-5  # Q_alpha
        assert kf.R == 0.005
        assert kf.R_adapt == True
        
        # 验证状态未初始化
        assert kf.state is None
        assert kf.P is None
        assert len(kf.beta_history) == 0
        
    def test_ols_initialization(self):
        """TC-3.1.2: 测试OLS初始化"""
        # 创建简单的线性关系数据
        np.random.seed(42)
        x_data = np.random.randn(100)
        true_beta, true_alpha = 1.5, 0.3
        y_data = true_beta * x_data + true_alpha + 0.1 * np.random.randn(100)
        
        kf = OriginalKalmanFilter(warmup=60)
        kf.initialize(x_data, y_data)
        
        # 验证状态初始化
        assert kf.state is not None
        assert len(kf.state) == 2
        assert kf.P is not None
        assert kf.P.shape == (2, 2)
        
        # 验证β接近真实值
        estimated_beta = kf.state[0]
        assert abs(estimated_beta - true_beta) < 0.2  # 允许一定误差
        
    def test_kalman_update(self):
        """TC-3.1.3: 测试Kalman更新过程"""
        np.random.seed(42)
        x_data = np.random.randn(100)
        true_beta, true_alpha = 1.0, 0.0
        y_data = true_beta * x_data + true_alpha + 0.05 * np.random.randn(100)
        
        kf = OriginalKalmanFilter(warmup=60)
        kf.initialize(x_data, y_data)
        
        initial_beta = kf.state[0]
        
        # 进行几次更新
        for i in range(60, min(80, len(x_data))):
            kf.update(x_data[i], y_data[i])
        
        # 验证历史记录
        assert len(kf.beta_history) > 0
        assert len(kf.z_history) > 0
        assert len(kf.beta_history) == len(kf.z_history)
        
        # 验证β收敛（应该接近真实值）
        final_beta = kf.state[0]
        assert abs(final_beta - true_beta) <= abs(initial_beta - true_beta)  # 应该更接近真实值
        
    def test_z_score_calculation(self):
        """TC-3.1.4: 测试Z-score计算（关键测试）"""
        np.random.seed(42)
        x_data = np.random.randn(100)
        true_beta = 1.0
        y_data = true_beta * x_data + 0.1 * np.random.randn(100)
        
        kf = OriginalKalmanFilter(warmup=60, R_init=0.01)
        kf.initialize(x_data, y_data)
        
        # 更新一些步骤
        z_scores = []
        for i in range(60, min(90, len(x_data))):
            kf.update(x_data[i], y_data[i])
            z_scores.append(kf.z_history[-1])
        
        # 验证Z-score特性
        assert len(z_scores) > 0
        z_var = np.var(z_scores)
        
        # Z方差应该接近1（这是原版本KF的关键特性）
        assert 0.5 <= z_var <= 2.0  # 合理范围
        
        # 应该有一些Z>2的值（约2-5%）
        z_gt2_ratio = np.mean(np.abs(z_scores) > 2.0)
        assert 0.0 <= z_gt2_ratio <= 0.2  # 不超过20%
        
    def test_r_adaptation(self):
        """TC-3.1.5: 测试R自适应调整"""
        np.random.seed(42)
        x_data = np.random.randn(100)
        y_data = x_data + 0.1 * np.random.randn(100)
        
        # 测试自适应R
        kf_adapt = OriginalKalmanFilter(warmup=60, R_adapt=True, R_init=0.01)
        kf_adapt.initialize(x_data, y_data)
        initial_R = kf_adapt.R
        
        for i in range(60, 80):
            kf_adapt.update(x_data[i], y_data[i])
        
        # R应该有所调整
        final_R = kf_adapt.R
        assert initial_R != final_R  # R应该发生变化
        
        # 测试固定R
        kf_fixed = OriginalKalmanFilter(warmup=60, R_adapt=False, R_init=0.01)
        kf_fixed.initialize(x_data, y_data)
        
        for i in range(60, 80):
            kf_fixed.update(x_data[i], y_data[i])
        
        # R应该保持不变
        assert kf_fixed.R == 0.01

class TestSignalGeneratorV3:
    """测试信号生成器V3"""
    
    def test_initialization(self):
        """TC-3.2.1: 测试SignalGenerator初始化"""
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60,
            z_open=2.0,
            z_close=0.5,
            max_holding_days=30
        )
        
        # 验证参数设置
        assert sg.signal_start_date == '2024-07-01'
        assert sg.kalman_warmup_days == 30
        assert sg.ols_training_days == 60
        assert sg.z_open == 2.0
        assert sg.z_close == 0.5
        assert sg.max_holding_days == 30
        
        # 验证时间轴计算
        assert sg.data_start_date == '2024-04-02'  # 2024-07-01 - 30 - 60
        
    def test_time_axis_calculation(self):
        """TC-3.3.1: 测试时间轴自动计算"""
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        # 验证自动计算的时间点
        assert sg.data_start_date == '2024-04-02'
        assert sg.kalman_warmup_start == '2024-06-01'
        assert sg.signal_generation_start == '2024-07-01'
        
    def test_signal_generation_logic(self):
        """TC-3.2.2-3.2.6: 测试信号生成逻辑"""
        sg = SignalGeneratorV3()
        
        # 测试开仓信号
        signal = sg._generate_signal(z_score=-2.5, position=None, days_held=0)
        assert signal == 'open_long'
        
        signal = sg._generate_signal(z_score=2.5, position=None, days_held=0)
        assert signal == 'open_short'
        
        # 测试平仓信号
        signal = sg._generate_signal(z_score=0.3, position='long', days_held=5)
        assert signal == 'close'
        
        # 测试强制平仓
        signal = sg._generate_signal(z_score=1.5, position='long', days_held=30)
        assert signal == 'close'
        
        # 测试持仓状态
        signal = sg._generate_signal(z_score=1.5, position='long', days_held=5)
        assert signal == 'holding_long'
        
        signal = sg._generate_signal(z_score=-1.5, position='short', days_held=5)
        assert signal == 'holding_short'
        
        # 测试空仓等待
        signal = sg._generate_signal(z_score=1.0, position=None, days_held=0)
        assert signal == 'empty'
        
    def test_position_state_update(self):
        """TC-3.2.9-3.2.10: 测试持仓状态更新"""
        sg = SignalGeneratorV3()
        
        # 测试开仓
        position, days_held = sg._update_position_state('open_long', None, 0)
        assert position == 'long'
        assert days_held == 1
        
        # 测试持仓天数增加
        position, days_held = sg._update_position_state('holding_long', 'long', 1)
        assert position == 'long'
        assert days_held == 2
        
        # 测试平仓
        position, days_held = sg._update_position_state('close', 'long', 5)
        assert position is None
        assert days_held == 0
        
    def test_phase_identification(self):
        """TC-3.3.1-3.3.2: 测试阶段识别"""
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        # 创建测试日期
        dates = pd.date_range('2024-04-01', '2024-08-01', freq='D')
        
        # 测试各个阶段
        ols_date = pd.Timestamp('2024-04-15')
        phase = sg._get_phase(ols_date)
        assert phase == 'ols_training'
        
        warmup_date = pd.Timestamp('2024-06-15')
        phase = sg._get_phase(warmup_date)
        assert phase == 'kalman_warmup'
        
        signal_date = pd.Timestamp('2024-07-15')
        phase = sg._get_phase(signal_date)
        assert phase == 'signal_generation'
        
    def test_process_single_pair(self):
        """TC-3.3.3: 测试单配对处理"""
        # 创建模拟数据
        np.random.seed(42)
        dates = pd.date_range('2024-04-01', '2024-08-01', freq='D')
        x_data = pd.Series(np.cumsum(0.01 * np.random.randn(len(dates))), index=dates)
        y_data = pd.Series(1.5 * x_data + 0.05 * np.random.randn(len(dates)), index=dates)
        
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        result = sg.process_pair(
            pair_name='TEST-PAIR',
            x_data=x_data,
            y_data=y_data,
            initial_beta=1.5
        )
        
        # 验证输出格式
        assert isinstance(result, pd.DataFrame)
        required_columns = ['date', 'pair', 'signal', 'z_score', 'beta', 'phase']
        for col in required_columns:
            assert col in result.columns
        
        # 验证阶段分布
        phases = result['phase'].unique()
        expected_phases = ['ols_training', 'kalman_warmup', 'signal_generation']
        for phase in expected_phases:
            assert phase in phases
        
        # 验证信号期的信号类型
        signal_period = result[result['phase'] == 'signal_generation']
        assert len(signal_period) > 0
        
        # 信号应该包含各种类型
        signals = signal_period['signal'].unique()
        assert 'empty' in signals  # 至少应该有空仓状态
        
    def test_batch_processing(self):
        """TC-3.3.4: 测试批量配对处理"""
        # 创建模拟配对数据
        pairs_df = pd.DataFrame({
            'pair': ['AU-AG', 'CU-ZN'],
            'symbol_x': ['AU', 'CU'],
            'symbol_y': ['AG', 'ZN'],
            'beta_1y': [1.2, 0.8],
            'beta_2y': [1.1, 0.9]
        })
        
        # 创建模拟价格数据
        np.random.seed(42)
        dates = pd.date_range('2024-04-01', '2024-08-01', freq='D')
        price_data = pd.DataFrame({
            'AU': np.cumsum(0.01 * np.random.randn(len(dates))),
            'AG': np.cumsum(0.01 * np.random.randn(len(dates))),
            'CU': np.cumsum(0.01 * np.random.randn(len(dates))),
            'ZN': np.cumsum(0.01 * np.random.randn(len(dates)))
        }, index=dates)
        
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60
        )
        
        result = sg.process_all_pairs(
            pairs_df=pairs_df,
            price_data=price_data,
            beta_window='1y'
        )
        
        # 验证输出
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # 验证包含所有配对
        pairs_in_result = result['pair'].unique()
        for pair in pairs_df['pair']:
            assert pair in pairs_in_result
        
        # 验证使用了正确的β窗口
        assert 'beta_window_used' in result.columns
        assert all(result['beta_window_used'] == '1y')
        
    def test_quality_monitoring(self):
        """TC-3.5.1-3.5.4: 测试质量监控"""
        sg = SignalGeneratorV3()
        
        # 模拟处理一些数据以生成质量报告
        np.random.seed(42)
        dates = pd.date_range('2024-04-01', '2024-08-01', freq='D')
        x_data = pd.Series(np.cumsum(0.01 * np.random.randn(len(dates))), index=dates)
        y_data = pd.Series(1.0 * x_data + 0.05 * np.random.randn(len(dates)), index=dates)
        
        sg.process_pair(
            pair_name='QUALITY-TEST',
            x_data=x_data,
            y_data=y_data
        )
        
        # 获取质量报告
        quality_report = sg.get_quality_report()
        
        # 验证质量报告格式
        assert isinstance(quality_report, pd.DataFrame)
        if len(quality_report) > 0:  # 如果有数据
            required_cols = ['pair', 'z_var', 'z_gt2_ratio', 'quality_status']
            for col in required_cols:
                assert col in quality_report.columns
        
    def test_error_handling(self):
        """TC-3.3.5-3.3.7: 测试错误处理"""
        sg = SignalGeneratorV3()
        
        # 测试数据不足的情况
        short_dates = pd.date_range('2024-07-01', '2024-07-05', freq='D')  # 只有5天
        x_data = pd.Series([1, 2, 3, 4, 5], index=short_dates)
        y_data = pd.Series([1, 2, 3, 4, 5], index=short_dates)
        
        # 应该能够处理而不崩溃
        result = sg.process_pair('SHORT-DATA', x_data, y_data)
        assert isinstance(result, pd.DataFrame)
        
        # 测试无效参数
        with pytest.raises((ValueError, TypeError)):
            SignalGeneratorV3(signal_start_date='invalid-date')
            
        # 测试空数据
        empty_df = pd.DataFrame()
        result = sg.process_all_pairs(empty_df, pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_workflow(self):
        """TC-3.3.8: 端到端工作流程测试"""
        # 创建完整的模拟数据，模拟真实的协整配对数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-08-01', freq='D')
        
        # 模拟两个协整的价格序列
        x_prices = np.cumsum(0.005 * np.random.randn(len(dates)))
        noise = 0.002 * np.random.randn(len(dates))
        y_prices = 1.2 * x_prices + 0.1 + noise  # y = 1.2*x + 0.1 + noise
        
        price_data = pd.DataFrame({
            'SYM_X': x_prices,
            'SYM_Y': y_prices
        }, index=dates)
        
        # 模拟协整模块的输出
        pairs_df = pd.DataFrame({
            'pair': ['SYM_X-SYM_Y'],
            'symbol_x': ['SYM_X'],
            'symbol_y': ['SYM_Y'],
            'beta_1y': [1.15],  # 接近真实值1.2
            'pvalue_1y': [0.01]
        })
        
        # 创建信号生成器
        sg = SignalGeneratorV3(
            signal_start_date='2024-07-01',
            kalman_warmup_days=30,
            ols_training_days=60,
            z_open=2.0,
            z_close=0.5
        )
        
        # 生成信号
        signals = sg.process_all_pairs(
            pairs_df=pairs_df,
            price_data=price_data,
            beta_window='1y'
        )
        
        # 验证端到端结果
        assert len(signals) > 0
        
        # 验证时间范围
        signal_dates = pd.to_datetime(signals['date'])
        assert signal_dates.min() >= pd.Timestamp('2024-01-01')  # 应该从数据开始
        
        # 验证阶段分布
        phases = signals['phase'].value_counts()
        assert 'ols_training' in phases
        assert 'kalman_warmup' in phases  
        assert 'signal_generation' in phases
        
        # 验证信号期的质量
        signal_period = signals[signals['phase'] == 'signal_generation']
        assert len(signal_period) > 0
        
        # Z-score应该有合理的分布
        z_scores = signal_period['z_score'].dropna()
        if len(z_scores) > 10:
            z_var = np.var(z_scores)
            assert 0.3 <= z_var <= 3.0  # 合理的方差范围
            
            # 应该有一些信号被触发
            trading_signals = signal_period[
                signal_period['signal'].isin(['open_long', 'open_short', 'close'])
            ]
            # 不要求一定有交易信号，但如果有的话应该合理
            
        # 验证质量报告
        quality_report = sg.get_quality_report()
        assert isinstance(quality_report, pd.DataFrame)


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])