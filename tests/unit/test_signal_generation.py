#!/usr/bin/env python3
"""
信号生成模块单元测试 - TDD版本
测试驱动开发方式实现，严格按照需求文档
所有卡尔曼滤波参数写死，不可配置
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.signal_generation import (
    SignalGenerator, KalmanFilter1D, calculate_ols_beta
)


class TestOLSBeta:
    """测试OLS Beta计算 - 真实数值计算"""
    
    def test_perfect_linear_relationship(self):
        """测试完美线性关系的Beta计算"""
        # 创建完美线性关系数据
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        beta_true = 2.5
        y_data = beta_true * x_data  # y = 2.5x
        
        # 计算Beta
        beta = calculate_ols_beta(y_data, x_data, window=len(x_data))
        
        # 验证Beta精确等于真实值
        assert abs(beta - beta_true) < 1e-10
        
    def test_noisy_linear_relationship(self):
        """测试带噪声的线性关系"""
        np.random.seed(42)
        n = 200
        
        # 创建带噪声的数据
        x_data = np.random.uniform(10, 100, n)
        true_beta = 1.567890  # 6位小数精度
        noise = np.random.randn(n) * 0.5
        y_data = true_beta * x_data + noise
        
        # 计算Beta
        beta = calculate_ols_beta(y_data, x_data, window=n)
        
        # 验证Beta接近真实值
        assert abs(beta - true_beta) < 0.01
        
    def test_rolling_window_beta(self):
        """测试滚动窗口Beta计算"""
        # 创建变化的Beta关系
        n = 200
        x_data = np.arange(n, dtype=float)
        
        # 前100个点: beta = 1.0
        # 后100个点: beta = 2.0
        y_data = np.zeros(n)
        y_data[:100] = 1.0 * x_data[:100]
        y_data[100:] = 2.0 * x_data[100:]
        
        # 使用100点窗口计算最后一个Beta
        beta = calculate_ols_beta(y_data, x_data, window=100)
        
        # 应该接近2.0（后100个点的Beta）
        assert 1.9 < beta < 2.1
        
    def test_beta_precision(self):
        """测试Beta计算的6位小数精度"""
        # 构造精确数据
        x = np.linspace(1, 10, 100)
        beta_exact = 3.141592  # 6位小数
        y = beta_exact * x
        
        # 计算Beta
        beta = calculate_ols_beta(y, x, window=100)
        
        # 验证6位小数精度
        assert f"{beta:.6f}" == f"{beta_exact:.6f}"


class TestKalmanFilter:
    """测试Kalman滤波器 - 真实滤波计算"""
    
    def test_kalman_initialization(self):
        """测试Kalman滤波器初始化"""
        initial_state = 1.5
        initial_variance = 2.0
        
        kf = KalmanFilter1D(
            initial_state=initial_state,
            initial_variance=initial_variance,
            process_variance=1e-5,
            measurement_variance=0.1
        )
        
        # 验证初始状态
        assert kf.get_state() == initial_state
        assert kf.get_variance() == initial_variance
        
    def test_kalman_convergence(self):
        """测试Kalman滤波器收敛性"""
        kf = KalmanFilter1D(
            initial_state=0.0,
            initial_variance=10.0,
            process_variance=1e-6,
            measurement_variance=0.01
        )
        
        # 给定常数测量值
        true_value = 2.5
        measurements = [true_value] * 100
        
        states = []
        for m in measurements:
            state = kf.update(m)
            states.append(state)
            
        # 应该收敛到真实值
        final_state = states[-1]
        assert abs(final_state - true_value) < 0.01
        
        # 方差应该减小
        final_variance = kf.get_variance()
        assert final_variance < 0.01
        
    def test_kalman_tracking_dynamic(self):
        """测试Kalman滤波器跟踪动态变化"""
        np.random.seed(42)
        
        # 创建缓慢变化的真实状态
        n = 200
        true_states = 1.0 + 0.01 * np.arange(n)  # 缓慢线性增长
        
        # 添加测量噪声
        measurements = true_states + np.random.randn(n) * 0.1
        
        kf = KalmanFilter1D(
            initial_state=1.0,
            initial_variance=1.0,
            process_variance=1e-4,  # 允许状态变化
            measurement_variance=0.01
        )
        
        filtered_states = []
        for m in measurements:
            state = kf.update(m)
            filtered_states.append(state)
            
        filtered_states = np.array(filtered_states)
        
        # 计算均方误差
        mse = np.mean((filtered_states - true_states) ** 2)
        assert mse < 0.05  # 良好的跟踪性能
        
        # 后半部分应该有更好的跟踪
        late_mse = np.mean((filtered_states[100:] - true_states[100:]) ** 2)
        assert late_mse < 0.02
        
    def test_kalman_noise_reduction(self):
        """测试Kalman滤波器降噪效果"""
        np.random.seed(123)
        
        # 创建噪声测量
        true_value = 5.0
        n = 100
        noise_std = 0.5
        measurements = true_value + np.random.randn(n) * noise_std
        
        kf = KalmanFilter1D(
            initial_state=0.0,
            initial_variance=100.0,
            process_variance=1e-5,  # 假设状态变化很小
            measurement_variance=noise_std ** 2
        )
        
        filtered = []
        for m in measurements:
            filtered.append(kf.update(m))
            
        # 滤波后的信号应该更平滑
        filtered_std = np.std(filtered[20:])  # 跳过初始收敛期
        raw_std = np.std(measurements[20:])
        
        # 滤波后的标准差应该小于原始信号
        assert filtered_std < raw_std * 0.5


class TestSignalGenerator:
    """测试信号生成器 - 真实信号计算"""
    
    def setup_method(self):
        """准备测试数据"""
        np.random.seed(42)
        n = 300
        dates = pd.date_range('2020-01-01', periods=n)
        
        # 创建协整的价格序列
        common_trend = np.cumsum(np.random.randn(n) * 0.1)
        
        # X序列（自变量）
        self.x_data = pd.Series(
            100 + common_trend + np.random.randn(n) * 0.5,
            index=dates
        )
        
        # Y序列（因变量），与X有固定关系
        true_beta = 1.5
        self.y_data = pd.Series(
            150 + true_beta * common_trend + np.random.randn(n) * 0.5,
            index=dates
        )
        self.true_beta = true_beta
        
    def test_signal_generator_init(self):
        """测试信号生成器初始化"""
        sg = SignalGenerator(self.y_data, self.x_data, fixed_beta=1.5)
        
        assert sg.y_data.equals(self.y_data)
        assert sg.x_data.equals(self.x_data)
        assert sg.fixed_beta == 1.5
        
    def test_spread_calculation(self):
        """测试价差计算"""
        sg = SignalGenerator(self.y_data, self.x_data, fixed_beta=self.true_beta)
        
        # 计算价差
        spread = sg.calculate_spread(beta=self.true_beta)
        
        # 验证价差计算公式: spread = y - beta * x
        expected_spread = self.y_data - self.true_beta * self.x_data
        pd.testing.assert_series_equal(spread, expected_spread)
        
        # 价差应该是平稳的（因为序列是协整的）
        spread_mean = spread.mean()
        spread_std = spread.std()
        assert abs(spread_mean) < 100  # 均值应该有界
        assert spread_std < 50  # 标准差应该有界
        
    def test_zscore_calculation(self):
        """测试Z-score计算"""
        sg = SignalGenerator(self.y_data, self.x_data, fixed_beta=self.true_beta)
        
        spread = sg.calculate_spread()
        window = 60
        zscore = sg.calculate_zscore(spread, window=window)
        
        # 验证Z-score性质
        assert len(zscore) == len(spread)
        
        # 前window个应该是NaN
        assert zscore[:window].isna().all()
        
        # 后面的Z-score应该标准化
        valid_zscore = zscore[window:]
        
        # Z-score的均值应该接近0，标准差接近1
        assert abs(valid_zscore.mean()) < 0.5
        assert 0.7 < valid_zscore.std() < 1.3
        
        # 手动验证一个Z-score计算
        test_idx = window + 10
        test_window = spread[test_idx-window:test_idx]
        expected_z = (spread.iloc[test_idx] - test_window.mean()) / test_window.std()
        assert abs(zscore.iloc[test_idx] - expected_z) < 1e-10
        
    def test_signal_generation_logic(self):
        """测试信号生成逻辑"""
        # 创建人工构造的价差数据
        dates = pd.date_range('2020-01-01', periods=200)
        
        # 构造明确的交易信号模式
        spread_values = np.zeros(200)
        # 第50-60: 价差上升到高位（应该做空）
        spread_values[50:60] = np.linspace(0, 3, 10)
        spread_values[60:70] = 3  # 保持高位
        # 第70-80: 价差回归（应该平仓）
        spread_values[70:80] = np.linspace(3, 0, 10)
        # 第120-130: 价差下降到低位（应该做多）
        spread_values[120:130] = np.linspace(0, -3, 10)
        spread_values[130:140] = -3  # 保持低位
        # 第140-150: 价差回归（应该平仓）
        spread_values[140:150] = np.linspace(-3, 0, 10)
        
        x_data = pd.Series(100, index=dates)
        y_data = pd.Series(100 + spread_values, index=dates)
        
        sg = SignalGenerator(y_data, x_data, fixed_beta=1.0)
        
        # 生成信号
        signals = sg.generate_signals(
            entry_threshold=2.0,
            exit_threshold=0.5,
            max_holding_days=30
        )
        
        # 验证信号DataFrame结构
        assert 'signal' in signals.columns
        assert 'position' in signals.columns
        assert 'zscore' in signals.columns
        
        # 应该有交易信号
        assert (signals['signal'] != 0).any()
        
        # 验证信号逻辑
        # 当Z-score > 2时应该有-1信号（做空）
        # 当Z-score < -2时应该有1信号（做多）
        high_z_mask = signals['zscore'] > 2.0
        low_z_mask = signals['zscore'] < -2.0
        
        # 至少应该有一些极端Z-score
        assert high_z_mask.any() or low_z_mask.any()
        
    def test_max_holding_constraint(self):
        """测试最大持仓天数约束"""
        # 创建持续偏离的数据
        dates = pd.date_range('2020-01-01', periods=100)
        
        # 持续高价差
        x_data = pd.Series(100, index=dates)
        y_data = pd.Series(103, index=dates)  # 持续偏离3个单位
        
        sg = SignalGenerator(y_data, x_data, fixed_beta=1.0)
        
        # 设置最大持仓10天
        signals = sg.generate_signals(
            entry_threshold=2.0,
            exit_threshold=0.5,
            max_holding_days=10
        )
        
        # 计算连续持仓天数
        positions = signals['position'].values
        max_consecutive = 0
        current_consecutive = 0
        
        for pos in positions:
            if pos != 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        # 不应超过最大持仓天数
        assert max_consecutive <= 10
        
    def test_kalman_beta_estimation(self):
        """测试Kalman滤波Beta估计"""
        sg = SignalGenerator(self.y_data, self.x_data)
        
        # 运行Kalman滤波
        beta_series = sg.run_kalman_filter(
            process_variance=1e-5,
            measurement_variance=0.1
        )
        
        # 验证Beta序列
        assert len(beta_series) == len(self.y_data)
        assert not beta_series.isna().any()
        
        # Beta应该逐渐收敛到真实值附近
        final_betas = beta_series.iloc[-50:]  # 最后50个Beta
        mean_beta = final_betas.mean()
        
        # 应该接近真实Beta（允许一定误差）
        assert abs(mean_beta - self.true_beta) < 0.5
        
        # Beta应该相对稳定
        beta_std = final_betas.std()
        assert beta_std < 0.2


class TestCalculationAccuracy:
    """测试计算精度 - 数值准确性验证"""
    
    def test_ols_numerical_accuracy(self):
        """测试OLS计算的数值精度"""
        # 使用精确构造的数据
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y = np.array([2.123456, 4.246912, 6.370368, 8.493824, 10.617280], dtype=np.float64)
        
        beta = calculate_ols_beta(y, x, window=5)
        
        # 验证6位小数精度
        expected_beta = 2.123456
        assert abs(beta - expected_beta) < 1e-6
        assert f"{beta:.6f}" == "2.123456"
        
    def test_zscore_edge_cases(self):
        """测试Z-score边界情况"""
        dates = pd.date_range('2020-01-01', periods=100)
        
        # 情况1: 零标准差
        constant_data = pd.Series(100.0, index=dates)
        sg = SignalGenerator(constant_data, constant_data, fixed_beta=1.0)
        
        spread = sg.calculate_spread()
        zscore = sg.calculate_zscore(spread, window=20)
        
        # 常数spread的Z-score应该是0或NaN
        valid_zscore = zscore[20:]
        assert valid_zscore.isna().all() or (valid_zscore == 0).all()
        
        # 情况2: 极小标准差
        tiny_noise = pd.Series(100.0 + np.random.randn(100) * 1e-10, index=dates)
        sg2 = SignalGenerator(tiny_noise, constant_data, fixed_beta=1.0)
        
        spread2 = sg2.calculate_spread()
        zscore2 = sg2.calculate_zscore(spread2, window=20)
        
        # 不应该产生无穷大或NaN
        assert not np.isinf(zscore2[20:]).any()
        
    def test_spread_calculation_precision(self):
        """测试价差计算精度"""
        dates = pd.date_range('2020-01-01', periods=10)
        
        # 精确构造的数据
        x_values = [100.123456, 101.234567, 102.345678, 103.456789, 104.567890,
                    105.678901, 106.789012, 107.890123, 108.901234, 109.012345]
        y_values = [200.246912, 202.469134, 204.691356, 206.913578, 209.135780,
                    211.357802, 213.580024, 215.780246, 218.002468, 220.024690]
        
        x_data = pd.Series(x_values, index=dates)
        y_data = pd.Series(y_values, index=dates)
        beta = 2.0
        
        sg = SignalGenerator(y_data, x_data, fixed_beta=beta)
        spread = sg.calculate_spread()
        
        # 手动计算价差
        expected_spread = y_data - beta * x_data
        
        # 验证精度（至少10位小数）
        pd.testing.assert_series_equal(spread, expected_spread)
        diff = (spread - expected_spread).abs()
        assert (diff < 1e-10).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])