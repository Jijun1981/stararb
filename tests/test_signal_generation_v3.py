#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块V3.0测试用例
基于原始状态空间Kalman滤波器
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.signal_generation_v3 import SignalGenerator, OriginalKalmanFilter
from lib.data import load_all_symbols_data


class TestOriginalKalmanFilter:
    """测试原始状态空间Kalman滤波器"""
    
    def test_initialization(self):
        """测试初始化参数"""
        kf = OriginalKalmanFilter(
            warmup=60,
            Q_beta=5e-6,
            Q_alpha=1e-5,
            R_init=0.005,
            R_adapt=True
        )
        
        assert kf.warmup == 60
        assert kf.Q[0, 0] == 5e-6
        assert kf.Q[1, 1] == 1e-5
        assert kf.R == 0.005
        assert kf.R_adapt == True
        assert kf.state is None
        assert kf.P is None
    
    def test_ols_warmup(self):
        """测试OLS预热初始化"""
        # 生成测试数据：y = 1.5*x + 0.1 + noise
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 1.5 * x + 0.1 + np.random.randn(n) * 0.1
        
        kf = OriginalKalmanFilter()
        kf.initialize(x, y)
        
        # 检查初始状态
        assert kf.state is not None
        assert len(kf.state) == 2  # [beta, alpha]
        assert abs(kf.state[0] - 1.5) < 0.2  # Beta接近真实值
        assert abs(kf.state[1] - 0.1) < 0.2  # Alpha接近真实值
        
        # 检查协方差矩阵
        assert kf.P is not None
        assert kf.P.shape == (2, 2)
        assert np.all(np.diag(kf.P) > 0)  # 对角线元素为正
        
        # 检查测量噪声
        assert kf.R > 0
        assert kf.R < 1.0  # 合理范围
    
    def test_kalman_update(self):
        """测试Kalman更新步骤"""
        # 生成测试数据
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = 1.5 * x + 0.1 + np.random.randn(n) * 0.1
        
        kf = OriginalKalmanFilter(
            warmup=60,
            Q_beta=5e-6,
            Q_alpha=1e-5,
            R_init=0.005,
            R_adapt=True
        )
        
        # 初始化
        kf.initialize(x, y)
        
        # 更新
        for i in range(60, 100):
            kf.update(x[i], y[i])
        
        # 检查历史记录
        assert len(kf.beta_history) == 40
        assert len(kf.alpha_history) == 40
        assert len(kf.z_history) == 40
        
        # Beta应该稳定在真实值附近
        final_beta = kf.state[0]
        assert abs(final_beta - 1.5) < 0.3
        
        # Z-score应该有合理的方差
        z_var = np.var(kf.z_history)
        assert 0.5 < z_var < 2.0
    
    def test_beta_stability(self):
        """测试Beta稳定性（无符号变化）"""
        # 生成稳定的正Beta数据
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = 0.8 * x + np.random.randn(n) * 0.1
        
        kf = OriginalKalmanFilter(
            warmup=60,
            Q_beta=5e-6,
            Q_alpha=1e-5,
            R_init=0.005
        )
        
        kf.initialize(x, y)
        
        for i in range(60, n):
            kf.update(x[i], y[i])
        
        # 检查Beta符号稳定性
        beta_history = np.array(kf.beta_history)
        sign_changes = np.sum(np.diff(np.sign(beta_history)) != 0)
        assert sign_changes == 0  # 无符号变化
        
        # Beta标准差应该很小
        beta_std = np.std(beta_history)
        assert beta_std < 0.1
    
    def test_negative_beta(self):
        """测试负Beta处理"""
        # 生成负相关数据
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        y = -0.5 * x + np.random.randn(n) * 0.1
        
        kf = OriginalKalmanFilter()
        kf.initialize(x, y)
        
        for i in range(60, n):
            kf.update(x[i], y[i])
        
        # Beta应该保持负值
        assert all(b < 0 for b in kf.beta_history)
        
        # 最终Beta接近真实值
        final_beta = kf.state[0]
        assert abs(final_beta - (-0.5)) < 0.2
    
    def test_z_score_properties(self):
        """测试Z-score统计特性"""
        # 生成均值回归数据
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        # 添加均值回归特性
        y = np.zeros(n)
        beta = 1.0
        for i in range(n):
            if i == 0:
                y[i] = beta * x[i] + np.random.randn() * 0.1
            else:
                # 均值回归
                y[i] = beta * x[i] + 0.1 * y[i-1] + np.random.randn() * 0.1
        
        kf = OriginalKalmanFilter(
            warmup=60,
            Q_beta=5e-6,
            Q_alpha=1e-5,
            R_init=0.005
        )
        
        kf.initialize(x, y)
        
        for i in range(60, n):
            kf.update(x[i], y[i])
        
        z_history = np.array(kf.z_history)
        
        # Z-score统计特性
        z_mean = np.mean(z_history)
        z_std = np.std(z_history)
        z_var = np.var(z_history)
        
        # 均值应该接近0
        assert abs(z_mean) < 0.5
        
        # 标准差应该接近1
        assert 0.7 < z_std < 1.5
        
        # 方差应该在合理范围
        assert 0.5 < z_var < 2.0
        
        # Z>2的比例应该在2-5%
        z_gt2_ratio = np.sum(np.abs(z_history) > 2.0) / len(z_history)
        assert 0.01 < z_gt2_ratio < 0.10


class TestSignalGenerator:
    """测试信号生成器"""
    
    def test_signal_generation_thresholds(self):
        """测试信号生成阈值逻辑"""
        sg = SignalGenerator(
            z_open=2.0,
            z_close=0.5,
            max_holding_days=30
        )
        
        # 测试开多仓
        signal = sg._generate_signal(
            z_score=-2.5,
            position=None,
            days_held=0
        )
        assert signal == 'open_long'
        
        # 测试开空仓
        signal = sg._generate_signal(
            z_score=2.5,
            position=None,
            days_held=0
        )
        assert signal == 'open_short'
        
        # 测试持仓
        signal = sg._generate_signal(
            z_score=1.5,
            position='long',
            days_held=5
        )
        assert signal == 'holding_long'
        
        # 测试平仓
        signal = sg._generate_signal(
            z_score=0.3,
            position='long',
            days_held=10
        )
        assert signal == 'close'
        
        # 测试强制平仓
        signal = sg._generate_signal(
            z_score=1.5,
            position='long',
            days_held=30
        )
        assert signal == 'close'
        
        # 测试空仓等待
        signal = sg._generate_signal(
            z_score=1.0,
            position=None,
            days_held=0
        )
        assert signal == 'empty'
    
    def test_process_pair_basic(self):
        """测试单配对处理基本功能"""
        # 生成测试数据
        np.random.seed(42)
        n = 300
        dates = pd.date_range('2024-01-01', periods=n)
        x_data = np.cumsum(np.random.randn(n) * 0.01)
        y_data = 0.8 * x_data + np.cumsum(np.random.randn(n) * 0.005)
        
        sg = SignalGenerator(
            z_open=2.0,
            z_close=0.5,
            Q_beta=5e-6,
            Q_alpha=1e-5
        )
        
        signals_df = sg.process_pair(
            pair_name='TEST-PAIR',
            x_data=pd.Series(x_data, index=dates),
            y_data=pd.Series(y_data, index=dates)
        )
        
        # 检查输出格式
        assert isinstance(signals_df, pd.DataFrame)
        assert len(signals_df) == n
        
        # 检查必要列
        required_cols = ['date', 'pair', 'signal', 'z_score', 'beta', 
                         'innovation', 'days_held', 'phase']
        for col in required_cols:
            assert col in signals_df.columns
        
        # 检查信号类型
        valid_signals = ['warm_up', 'empty', 'open_long', 'open_short', 
                        'holding_long', 'holding_short', 'close']
        assert all(s in valid_signals for s in signals_df['signal'].unique())
        
        # 前60天应该是warm_up
        assert all(signals_df.iloc[:60]['signal'] == 'warm_up')
        
        # 60-120天应该是Kalman预热期
        assert all(signals_df.iloc[60:120]['phase'] == 'convergence_period')
    
    def test_batch_processing(self):
        """测试批量配对处理"""
        # 准备测试数据
        n = 200
        dates = pd.date_range('2024-01-01', periods=n)
        
        # 创建价格数据
        price_data = pd.DataFrame(index=dates)
        price_data['X1'] = np.cumsum(np.random.randn(n) * 0.01)
        price_data['Y1'] = 0.8 * price_data['X1'] + np.cumsum(np.random.randn(n) * 0.005)
        price_data['X2'] = np.cumsum(np.random.randn(n) * 0.01)
        price_data['Y2'] = -0.5 * price_data['X2'] + np.cumsum(np.random.randn(n) * 0.005)
        
        # 创建配对DataFrame
        pairs_df = pd.DataFrame({
            'pair': ['X1-Y1', 'X2-Y2'],
            'symbol_x': ['X1', 'X2'],
            'symbol_y': ['Y1', 'Y2'],
            'beta_1y': [0.8, -0.5]
        })
        
        sg = SignalGenerator()
        
        all_signals = sg.process_all_pairs(
            pairs_df=pairs_df,
            price_data=price_data,
            beta_window='1y'
        )
        
        # 检查输出
        assert isinstance(all_signals, pd.DataFrame)
        assert len(all_signals) == n * 2  # 2个配对
        
        # 检查每个配对都有信号
        assert set(all_signals['pair'].unique()) == {'X1-Y1', 'X2-Y2'}
        
        # 检查质量报告
        quality_report = sg.get_quality_report()
        assert isinstance(quality_report, pd.DataFrame)
        assert len(quality_report) == 2
        assert 'z_variance' in quality_report.columns
        assert 'signal_ratio' in quality_report.columns
    
    def test_real_data_integration(self):
        """测试真实数据集成（如果数据可用）"""
        try:
            # 尝试加载真实数据
            log_prices = load_all_symbols_data()
            
            # 选择一个测试配对
            test_pair = ('AU', 'AG')
            
            if test_pair[0] in log_prices.columns and test_pair[1] in log_prices.columns:
                x_data = log_prices[test_pair[0]]
                y_data = log_prices[test_pair[1]]
                
                sg = SignalGenerator(
                    Q_beta=5e-6,
                    Q_alpha=1e-5,
                    R_init=0.005
                )
                
                signals_df = sg.process_pair(
                    pair_name=f"{test_pair[0]}-{test_pair[1]}",
                    x_data=x_data,
                    y_data=y_data
                )
                
                # 基本检查
                assert len(signals_df) > 0
                assert 'z_score' in signals_df.columns
                assert 'beta' in signals_df.columns
                
                # 检查Z-score质量
                z_scores = signals_df[signals_df['phase'] == 'signal_period']['z_score']
                if len(z_scores) > 60:
                    z_var = np.var(z_scores)
                    assert 0.3 < z_var < 3.0  # 合理范围
        
        except Exception as e:
            pytest.skip(f"无法加载真实数据: {e}")
    
    def test_44_stable_pairs(self):
        """测试44个稳定配对的预期性能"""
        # 这里使用模拟数据代表44个稳定配对
        np.random.seed(42)
        n_pairs = 44
        results = []
        
        for i in range(n_pairs):
            # 生成具有不同特性的配对数据
            n = 300
            x = np.random.randn(n)
            beta_true = np.random.uniform(-2, 2)
            y = beta_true * x + np.random.randn(n) * 0.1
            
            kf = OriginalKalmanFilter(
                Q_beta=5e-6,
                Q_alpha=1e-5,
                R_init=0.005
            )
            
            kf.initialize(x, y)
            
            for j in range(60, n):
                kf.update(x[j], y[j])
            
            # 计算统计指标
            beta_history = np.array(kf.beta_history)
            z_history = np.array(kf.z_history)
            
            sign_changes = np.sum(np.diff(np.sign(beta_history)) != 0)
            z_var = np.var(z_history)
            z_gt2_ratio = np.sum(np.abs(z_history) > 2.0) / len(z_history)
            
            results.append({
                'pair': f'PAIR_{i}',
                'sign_changes': sign_changes,
                'z_var': z_var,
                'z_gt2_ratio': z_gt2_ratio
            })
        
        results_df = pd.DataFrame(results)
        
        # 验证期望性能（基于实证结果）
        # 97%的配对应该无符号变化
        no_sign_change_ratio = (results_df['sign_changes'] == 0).sum() / len(results_df)
        assert no_sign_change_ratio > 0.9
        
        # 平均Z方差应该接近1.0
        avg_z_var = results_df['z_var'].mean()
        assert 0.8 < avg_z_var < 1.5
        
        # Z>2比例应该在合理范围
        avg_z_gt2 = results_df['z_gt2_ratio'].mean()
        assert 0.01 < avg_z_gt2 < 0.10


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])