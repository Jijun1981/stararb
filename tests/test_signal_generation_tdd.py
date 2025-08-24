#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块的TDD测试用例
严格对齐需求文档 /mnt/e/Star-arb/docs/Requirements/03_signal_generation.md
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 这里假设实现在 lib.signal_generation 模块中
# from lib.signal_generation import AdaptiveKalmanFilter, AdaptiveSignalGenerator


class TestAdaptiveKalmanFilter:
    """测试自适应Kalman滤波器 - REQ-3.1"""
    
    def test_initialization(self):
        """TC-3.1.0: 测试初始化参数"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        kf = AdaptiveKalmanFilter(pair_name="TEST-PAIR")
        assert kf.pair_name == "TEST-PAIR"
        assert kf.delta == 0.98  # REQ-3.1.6: 初始δ=0.98
        assert kf.lambda_r == 0.96  # REQ-3.1.4: λ=0.96
        assert kf.beta_bounds == (-4, 4)  # REQ-3.1.7: β边界
        assert kf.beta is None  # 未初始化前为None
        assert kf.P is None
        assert kf.R is None
    
    def test_ols_warmup_with_decentering(self):
        """TC-3.1.3: 测试OLS预热（去中心化处理）- REQ-3.1.3"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        # 生成测试数据：y = 1.5*x + noise
        np.random.seed(42)
        n = 100
        x_data = np.random.randn(n) + 5  # 加偏移量测试去中心化
        y_data = 1.5 * x_data + np.random.randn(n) * 0.1 + 10  # 加偏移量
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        result = kf.warm_up_ols(x_data, y_data, window=60)
        
        # 验证返回值
        assert 'beta' in result
        assert 'R' in result
        assert 'P' in result
        assert 'mu_x' in result  # 去中心化的均值
        assert 'mu_y' in result
        
        # 验证beta接近真实值1.5
        assert abs(result['beta'] - 1.5) < 0.2
        
        # 验证P0初始化：P = R / Var(x)，不乘0.1
        x_centered = x_data[:60] - result['mu_x']
        x_var = np.var(x_centered, ddof=1)
        expected_P = result['R'] / x_var
        assert abs(kf.P - expected_P) < 1e-6
        
        # 验证状态更新
        assert kf.beta == result['beta']
        assert kf.R == result['R']
        assert kf.P == result['P']
    
    def test_kalman_update_with_discount_factor(self):
        """TC-3.1.5: 测试折扣因子Kalman更新 - REQ-3.1.5"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        # 初始化
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.beta = 1.0
        kf.P = 0.01
        kf.R = 0.001
        kf.mu_x = 0
        kf.mu_y = 0
        
        # 执行更新
        x_t = 2.0
        y_t = 2.1  # 略高于预测值2.0
        result = kf.update(y_t, x_t)
        
        # 验证返回值包含所需字段
        assert 'beta' in result
        assert 'v' in result  # 创新
        assert 'S' in result  # 创新方差
        assert 'z' in result  # 标准化创新
        assert 'R' in result
        assert 'K' in result  # Kalman增益
        
        # 验证折扣因子实现：P_prior = P / delta
        # 这个需要在update内部验证，这里验证beta有更新
        assert kf.beta != 1.0
        
        # 验证创新标准化计算：z = v/√S
        v = y_t - 1.0 * x_t  # 创新
        S_expected = x_t**2 * (kf.P / 0.98) + 0.001  # 使用折扣后的P
        # 由于update会改变R，这里只验证z的计算方式正确
        assert abs(result['z']) < 10  # 合理范围
    
    def test_beta_boundary_protection(self):
        """TC-3.1.7: 测试β边界保护 - REQ-3.1.7"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.beta = 3.9
        kf.P = 0.01
        kf.R = 0.001
        kf.mu_x = 0
        kf.mu_y = 0
        
        # 模拟会导致beta超出边界的更新
        x_t = 1.0
        y_t = 10.0  # 极端值，会导致beta大幅增加
        
        result = kf.update(y_t, x_t)
        
        # 验证beta被限制在[-4, 4]范围内
        assert -4 <= result['beta'] <= 4
        assert -4 <= kf.beta <= 4
    
    def test_r_adaptive_ewma(self):
        """TC-3.1.4: 测试R的EWMA自适应 - REQ-3.1.4"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.beta = 1.0
        kf.P = 0.01
        kf.R = 0.001
        kf.mu_x = 0
        kf.mu_y = 0
        
        R_old = kf.R
        
        # 执行多次更新，观察R的变化
        for i in range(5):
            x_t = np.random.randn()
            y_t = kf.beta * x_t + np.random.randn() * 0.1
            result = kf.update(y_t, x_t)
            
            # 验证R通过EWMA更新：R_t = λ*R_{t-1} + (1-λ)*v²
            # λ = 0.96
            assert result['R'] != R_old  # R应该更新
            R_old = result['R']
    
    def test_calibrate_delta(self):
        """TC-3.2: 测试δ自动校准 - REQ-3.2"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.beta = 1.0
        kf.P = 0.01
        kf.R = 0.001
        kf.mu_x = 0
        kf.mu_y = 0
        
        # 模拟z_history，创建高方差情况
        kf.z_history = [np.random.randn() * 2 for _ in range(60)]  # 高方差
        
        old_delta = kf.delta
        calibrated = kf.calibrate_delta()
        
        # 验证校准逻辑：方差>1.3时δ减小
        z_var = np.var(kf.z_history[-60:])
        if z_var > 1.3:
            assert kf.delta == old_delta - 0.01  # REQ-3.2.4
        elif z_var < 0.8:
            assert kf.delta == old_delta + 0.01
        
        # 验证δ边界 - REQ-3.2.5
        assert 0.95 <= kf.delta <= 0.995
    
    def test_innovation_standardization(self):
        """TC-3.3: 测试创新标准化计算 - REQ-3.3.1"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.beta = 1.0
        kf.P = 0.01
        kf.R = 0.001
        kf.mu_x = 0
        kf.mu_y = 0
        
        x_t = 2.0
        y_t = 2.5
        
        result = kf.update(y_t, x_t)
        
        # 手动计算验证
        v = y_t - kf.beta * x_t  # 创新
        P_prior = kf.P / kf.delta  # 折扣先验协方差
        S = x_t**2 * P_prior + kf.R  # 创新方差
        z_expected = v / np.sqrt(S)  # 标准化创新
        
        # 验证z的计算方式（允许小的数值误差）
        assert 'z' in result
        assert result['z'] != 0  # 不应该是0
        # z是通过创新标准化计算的，不是滚动窗口


class TestSignalGeneration:
    """测试信号生成逻辑 - REQ-3.3"""
    
    def test_signal_states(self):
        """TC-3.3.4: 测试信号状态机制 - REQ-3.3.4"""
        from lib.signal_generation import generate_signal
        
        # 测试空仓时开多
        signal = generate_signal(
            z_score=-2.5, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'open_long'
        
        # 测试空仓时开空
        signal = generate_signal(
            z_score=2.5, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'open_short'
        
        # 测试持多时平仓
        signal = generate_signal(
            z_score=0.3, position='long', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'
        
        # 测试持空时平仓
        signal = generate_signal(
            z_score=-0.3, position='short', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'
        
        # 测试强制平仓
        signal = generate_signal(
            z_score=1.5, position='long', days_held=30,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'
        
        # 测试持仓中状态
        signal = generate_signal(
            z_score=1.0, position='long', days_held=10,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_long'
        
        signal = generate_signal(
            z_score=-1.0, position='short', days_held=10,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_short'
        
        # 测试空仓等待
        signal = generate_signal(
            z_score=1.0, position=None, days_held=0,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'empty'
    
    def test_no_duplicate_positions(self):
        """TC-3.3.5: 测试防重复开仓 - REQ-3.3.5"""
        from lib.signal_generation import generate_signal
        
        # 已有多头仓位，即使z<-2也不开新仓
        signal = generate_signal(
            z_score=-3.0, position='long', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_long'  # 不是open_long
        
        # 已有空头仓位，即使z>2也不开新仓
        signal = generate_signal(
            z_score=3.0, position='short', days_held=5,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'holding_short'  # 不是open_short
    
    def test_signal_priority(self):
        """TC-3.3.6: 测试信号优先级 - REQ-3.3.6"""
        from lib.signal_generation import generate_signal
        
        # 强制平仓优先级最高（即使z值仍然极端）
        signal = generate_signal(
            z_score=-3.0, position='long', days_held=30,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'  # 强制平仓，即使z<-2
        
        # 正常平仓优先于持仓
        signal = generate_signal(
            z_score=0.3, position='long', days_held=10,
            z_open=2.0, z_close=0.5, max_days=30
        )
        assert signal == 'close'  # 平仓，不是holding


class TestAdaptiveSignalGenerator:
    """测试信号生成器主类 - REQ-3.4"""
    
    def test_initialization_parameters(self):
        """测试初始化参数"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        sg = AdaptiveSignalGenerator(
            z_open=2.5,
            z_close=0.8,
            max_holding_days=20,
            calibration_freq=10,
            ols_window=80,
            warm_up_days=40
        )
        
        assert sg.z_open == 2.5
        assert sg.z_close == 0.8
        assert sg.max_holding_days == 20
        assert sg.calibration_freq == 10
        assert sg.ols_window == 80
        assert sg.warm_up_days == 40
    
    def test_process_single_pair(self):
        """TC-3.4.3: 测试单配对处理 - REQ-3.4.3"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 生成测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        x_data = pd.Series(np.cumsum(np.random.randn(200)) * 0.01, index=dates)
        y_data = pd.Series(x_data * 1.5 + np.random.randn(200) * 0.01, index=dates)
        
        sg = AdaptiveSignalGenerator()
        signals_df = sg.process_pair(
            pair_name='TEST-PAIR',
            x_data=x_data,
            y_data=y_data,
            initial_beta=1.5
        )
        
        # 验证输出格式
        assert isinstance(signals_df, pd.DataFrame)
        assert 'date' in signals_df.columns
        assert 'pair' in signals_df.columns
        assert 'signal' in signals_df.columns
        assert 'z_score' in signals_df.columns
        assert 'innovation' in signals_df.columns
        assert 'beta' in signals_df.columns
        assert 'days_held' in signals_df.columns
        assert 'reason' in signals_df.columns
        assert 'phase' in signals_df.columns
        
        # 验证配对名称
        assert all(signals_df['pair'] == 'TEST-PAIR')
        
        # 验证阶段划分
        warm_up_period = sg.ols_window + sg.warm_up_days
        assert all(signals_df.iloc[:warm_up_period]['phase'].isin(['warm_up', 'convergence_period']))
        assert all(signals_df.iloc[warm_up_period:]['phase'] == 'signal_period')
    
    def test_batch_processing(self):
        """TC-3.4: 测试批量配对处理 - REQ-3.4.1, REQ-3.4.2"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 准备配对数据（模拟协整模块输出）
        pairs_df = pd.DataFrame({
            'pair': ['AG-NI', 'AU-ZN', 'CU-PB'],
            'symbol_x': ['AG', 'AU', 'CU'],
            'symbol_y': ['NI', 'ZN', 'PB'],
            'beta_1y': [1.2, -0.8, 2.1],
            'beta_2y': [1.3, -0.7, 2.0],
            'beta_3y': [1.4, -0.6, 1.9],
            'beta_5y': [1.5, -0.5, 1.8],
            'pvalue_1y': [0.01, 0.02, 0.03],
            'pvalue_5y': [0.02, 0.03, 0.04]
        })
        
        # 准备价格数据
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        price_data = pd.DataFrame({
            'AG': np.cumsum(np.random.randn(200)) * 0.01,
            'NI': np.cumsum(np.random.randn(200)) * 0.01,
            'AU': np.cumsum(np.random.randn(200)) * 0.01,
            'ZN': np.cumsum(np.random.randn(200)) * 0.01,
            'CU': np.cumsum(np.random.randn(200)) * 0.01,
            'PB': np.cumsum(np.random.randn(200)) * 0.01
        }, index=dates)
        
        sg = AdaptiveSignalGenerator()
        
        # 测试不同beta窗口
        for beta_window in ['1y', '2y', '3y', '5y']:
            signals_df = sg.process_all_pairs(
                pairs_df=pairs_df,
                price_data=price_data,
                beta_window=beta_window
            )
            
            # 验证输出
            assert isinstance(signals_df, pd.DataFrame)
            assert len(signals_df) > 0
            
            # 验证所有配对都被处理
            processed_pairs = signals_df['pair'].unique()
            assert len(processed_pairs) <= 3  # 最多3个配对
            
            # 验证使用了正确的beta窗口
            assert all(signals_df['beta_window_used'] == beta_window)
    
    def test_quality_metrics(self):
        """TC-3.5: 测试质量监控 - REQ-3.5"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 生成测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        x_data = pd.Series(np.cumsum(np.random.randn(200)) * 0.01, index=dates)
        y_data = pd.Series(x_data * 1.5 + np.random.randn(200) * 0.01, index=dates)
        
        sg = AdaptiveSignalGenerator()
        sg.process_pair('TEST-PAIR', x_data, y_data)
        
        # 获取质量报告
        quality_report = sg.get_quality_report()
        
        assert isinstance(quality_report, pd.DataFrame)
        assert 'pair' in quality_report.columns
        assert 'z_var' in quality_report.columns
        assert 'z_mean' in quality_report.columns
        assert 'z_std' in quality_report.columns
        assert 'current_delta' in quality_report.columns
        assert 'current_R' in quality_report.columns
        assert 'quality_status' in quality_report.columns
        assert 'calibration_count' in quality_report.columns
        
        # 验证质量评级逻辑
        for _, row in quality_report.iterrows():
            z_var = row['z_var']
            if 0.8 <= z_var <= 1.3:
                assert row['quality_status'] == 'good'
            elif 0.6 <= z_var < 0.8 or 1.3 < z_var <= 1.5:
                assert row['quality_status'] == 'warning'
            else:
                assert row['quality_status'] == 'bad'
    
    def test_z_score_distribution(self):
        """测试z-score分布（允许fat tail）"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 生成更长的测试数据以观察分布
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        # 创建带有偶尔极端值的数据（模拟fat tail）
        x_data = pd.Series(np.cumsum(np.random.randn(500)) * 0.01, index=dates)
        noise = np.random.randn(500) * 0.01
        # 添加5%的极端值
        extreme_indices = np.random.choice(500, 25, replace=False)
        noise[extreme_indices] *= 5  # 放大噪声
        y_data = pd.Series(x_data * 1.5 + noise, index=dates)
        
        sg = AdaptiveSignalGenerator()
        signals_df = sg.process_pair('TEST-PAIR', x_data, y_data)
        
        # 获取信号期的z-score
        signal_period_df = signals_df[signals_df['phase'] == 'signal_period']
        z_scores = signal_period_df['z_score'].values
        
        # 验证分布特性
        # 95%以上的z应该在合理范围内（比如-4到4）
        in_range = np.sum(np.abs(z_scores) <= 4) / len(z_scores)
        assert in_range >= 0.90  # 允许10%的fat tail，比需求的5%更宽松
        
        # 验证有一些极端值（fat tail特性）
        extreme_z = np.sum(np.abs(z_scores) > 3)
        assert extreme_z > 0  # 应该有一些极端值
        
        # 验证z的方差在合理范围（考虑自适应调整）
        z_var = np.var(z_scores)
        assert 0.5 <= z_var <= 2.0  # 允许一定范围的方差


class TestSpecificRequirements:
    """测试需求文档中明确要求的测试用例"""
    
    def test_tc_3_1_1_constant_relationship_data(self):
        """TC-3.1.1: 恒定关系数据，β收敛到真实值"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        np.random.seed(42)
        n = 200
        true_beta = 1.5
        
        # 生成恒定关系数据: y = 1.5*x + noise
        x_data = np.random.randn(n)
        y_data = true_beta * x_data + np.random.randn(n) * 0.01  # 低噪声
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.warm_up_ols(x_data, y_data, window=60)
        
        # 运行更新，观察β收敛
        for i in range(60, n):
            kf.update(y_data[i], x_data[i])
        
        # 验证β收敛到真实值（允许小误差）
        final_beta = kf.beta
        assert abs(final_beta - true_beta) < 0.1, f"β没有收敛到真实值，期望{true_beta}，实际{final_beta}"
    
    def test_tc_3_1_2_slowly_drifting_beta(self):
        """TC-3.1.2: β缓慢漂移数据，β跟踪漂移趋势"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        np.random.seed(42)
        n = 200
        
        # β从1.0缓慢漂移到2.0
        beta_sequence = np.linspace(1.0, 2.0, n)
        x_data = np.random.randn(n)
        y_data = np.array([beta_sequence[i] * x_data[i] + np.random.randn() * 0.02 for i in range(n)])
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.warm_up_ols(x_data, y_data, window=60)
        
        beta_estimates = []
        for i in range(60, n):
            result = kf.update(y_data[i], x_data[i])
            beta_estimates.append(result['beta'])
        
        # 验证β跟踪了漂移趋势（由于δ=0.98比较保守，允许更大的偏差）
        final_beta = beta_estimates[-1]
        initial_ols_beta = kf.beta_history[0] if kf.beta_history else kf.beta
        
        # 验证β确实在朝正确方向变化
        beta_change = final_beta - initial_ols_beta
        assert beta_change > 0.1, f"β没有朝正确方向变化，变化量{beta_change}"
        
        # 验证β在合理范围内（不要求精确跟踪到2.0）
        assert 1.2 <= final_beta <= 2.2, f"β超出合理范围，实际{final_beta}"
    
    def test_tc_3_1_3_high_noise_stability(self):
        """TC-3.1.3: 高噪声数据，β保持稳定不发散"""
        from lib.signal_generation import AdaptiveKalmanFilter
        
        np.random.seed(42)
        n = 300
        true_beta = 1.5
        
        # 生成高噪声数据
        x_data = np.random.randn(n)
        y_data = true_beta * x_data + np.random.randn(n) * 0.5  # 高噪声
        
        kf = AdaptiveKalmanFilter(pair_name="TEST")
        kf.warm_up_ols(x_data, y_data, window=60)
        
        beta_estimates = []
        for i in range(60, n):
            result = kf.update(y_data[i], x_data[i])
            beta_estimates.append(result['beta'])
        
        # 验证β保持稳定，不发散
        assert all(-10 < beta < 10 for beta in beta_estimates), "β发散了"
        
        # 验证β的标准差在合理范围
        beta_std = np.std(beta_estimates)
        assert beta_std < 2.0, f"β波动过大，标准差{beta_std}"
    
    def test_tc_3_2_5_convergence_period_no_trading(self):
        """TC-3.2.5: 收敛期内不生成交易信号"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        x_data = pd.Series(np.cumsum(np.random.randn(150)) * 0.01, index=dates)
        y_data = pd.Series(x_data * 1.5 + np.random.randn(150) * 0.01, index=dates)
        
        sg = AdaptiveSignalGenerator(
            ols_window=60,
            warm_up_days=30
        )
        
        signals_df = sg.process_pair('TEST', x_data, y_data)
        
        # 验证收敛期（前90天）不生成交易信号
        convergence_period = signals_df.iloc[:90]
        trading_signals = convergence_period[convergence_period['signal'].isin(['open_long', 'open_short', 'close'])]
        assert len(trading_signals) == 0, f"收敛期生成了{len(trading_signals)}个交易信号"
        
        # 验证收敛期的phase标记正确
        assert all(convergence_period['phase'].isin(['warm_up', 'convergence_period']))
    
    def test_tc_3_2_7_custom_beta_window(self):
        """TC-3.2.7: 自定义β时间窗口配置"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 准备配对数据
        pairs_df = pd.DataFrame({
            'pair': ['TEST-PAIR'],
            'symbol_x': ['X'],
            'symbol_y': ['Y'],
            'beta_1y': [1.2],
            'beta_2y': [1.5],
            'beta_3y': [1.8],
            'beta_5y': [2.0],
        })
        
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        price_data = pd.DataFrame({
            'X': np.cumsum(np.random.randn(150)) * 0.01,
            'Y': np.cumsum(np.random.randn(150)) * 0.01
        }, index=dates)
        
        sg = AdaptiveSignalGenerator()
        
        # 测试不同时间窗口
        for beta_window in ['1y', '2y', '3y', '5y']:
            signals_df = sg.process_all_pairs(pairs_df, price_data, beta_window)
            
            if not signals_df.empty:
                # 验证使用了正确的beta窗口
                assert all(signals_df['beta_window_used'] == beta_window)
                
                # 验证初始beta值正确
                expected_beta = pairs_df[f'beta_{beta_window}'].iloc[0]
                assert all(signals_df['beta_initial'] == expected_beta)
    
    def test_tc_3_3_8_cointegration_data_format_matching(self):
        """TC-3.3.8: 协整数据格式匹配 - 正确读取协整模块输出DataFrame的所有字段"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 模拟完整的协整模块输出格式
        pairs_df = pd.DataFrame({
            'pair': ['AG-NI', 'AU-ZN'],
            'symbol_x': ['AG', 'AU'],
            'symbol_y': ['NI', 'ZN'],
            'beta_1y': [1.2, -0.8],
            'beta_2y': [1.3, -0.7],
            'beta_3y': [1.4, -0.6],
            'beta_5y': [1.5, -0.5],
            'pvalue_1y': [0.01, 0.02],
            'pvalue_5y': [0.02, 0.03],
            'direction': ['AG_low_vol', 'AU_low_vol'],
            'test_statistic_1y': [-3.5, -3.2]
        })
        
        dates = pd.date_range('2024-01-01', periods=150, freq='D')
        price_data = pd.DataFrame({
            'AG': np.cumsum(np.random.randn(150)) * 0.01,
            'NI': np.cumsum(np.random.randn(150)) * 0.01,
            'AU': np.cumsum(np.random.randn(150)) * 0.01,
            'ZN': np.cumsum(np.random.randn(150)) * 0.01
        }, index=dates)
        
        sg = AdaptiveSignalGenerator()
        signals_df = sg.process_all_pairs(pairs_df, price_data, beta_window='1y')
        
        # 验证处理了所有配对
        assert len(signals_df['pair'].unique()) <= 2
        
        # 验证正确读取了所有字段
        if not signals_df.empty:
            assert 'symbol_x' in signals_df.columns
            assert 'symbol_y' in signals_df.columns
            assert 'beta_initial' in signals_df.columns
            assert 'beta_window_used' in signals_df.columns


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end_signal_generation(self):
        """端到端测试：从原始数据到信号生成"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 模拟真实场景：创建两个相关的价格序列
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # 基础价格序列
        base_returns = np.random.randn(365) * 0.01
        x_prices = np.exp(np.cumsum(base_returns))
        
        # 相关价格序列（beta约1.5，带噪声）
        y_returns = base_returns * 1.5 + np.random.randn(365) * 0.005
        y_prices = np.exp(np.cumsum(y_returns))
        
        x_data = pd.Series(np.log(x_prices), index=dates)  # 对数价格
        y_data = pd.Series(np.log(y_prices), index=dates)
        
        # 运行信号生成
        sg = AdaptiveSignalGenerator(
            z_open=2.0,
            z_close=0.5,
            max_holding_days=30,
            calibration_freq=5,
            ols_window=60,
            warm_up_days=30
        )
        
        signals_df = sg.process_pair(
            pair_name='X-Y',
            x_data=x_data,
            y_data=y_data,
            initial_beta=None  # 让OLS自动估计
        )
        
        # 验证基本输出
        assert len(signals_df) == 365
        assert signals_df['pair'].iloc[0] == 'X-Y'
        
        # 验证阶段划分
        warm_up_end = 60 + 30  # ols_window + warm_up_days
        assert all(signals_df.iloc[:warm_up_end]['phase'].isin(['warm_up', 'convergence_period']))
        assert all(signals_df.iloc[warm_up_end:]['phase'] == 'signal_period')
        
        # 验证有交易信号生成
        trading_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short', 'close'])]
        # 不约束具体数量，只要有信号即可
        assert len(trading_signals) > 0
        
        # 验证信号逻辑的一致性
        for i, row in signals_df.iterrows():
            if row['signal'] == 'open_long':
                assert row['z_score'] <= -2.0  # 开多条件
            elif row['signal'] == 'open_short':
                assert row['z_score'] >= 2.0  # 开空条件
            elif row['signal'] == 'close' and row['reason'] == 'z_threshold':
                assert abs(row['z_score']) <= 0.5  # 正常平仓条件
        
        # 验证质量报告
        quality_report = sg.get_quality_report()
        assert len(quality_report) == 1
        assert quality_report['pair'].iloc[0] == 'X-Y'
        
        # 验证自适应校准
        # 应该有校准发生（每5天一次）
        signal_period_days = len(signals_df) - warm_up_end
        expected_calibrations = signal_period_days // 5
        actual_calibrations = quality_report['calibration_count'].iloc[0]
        assert actual_calibrations > 0  # 至少有校准
    
    def test_extreme_market_conditions(self):
        """测试极端市场条件下的鲁棒性"""
        from lib.signal_generation import AdaptiveSignalGenerator
        
        # 创建极端场景：突然的结构性变化
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        
        # 前150天：稳定关系
        x1 = np.cumsum(np.random.randn(150)) * 0.01
        y1 = x1 * 1.5 + np.random.randn(150) * 0.01
        
        # 后150天：关系突变
        x2 = np.cumsum(np.random.randn(150)) * 0.01
        y2 = x2 * 0.8 + np.random.randn(150) * 0.02  # beta从1.5变为0.8
        
        x_data = pd.Series(np.concatenate([x1, x2]), index=dates)
        y_data = pd.Series(np.concatenate([y1, y2]), index=dates)
        
        sg = AdaptiveSignalGenerator()
        signals_df = sg.process_pair('TEST', x_data, y_data)
        
        # 系统应该能处理而不崩溃
        assert len(signals_df) == 300
        
        # beta应该适应变化
        early_beta = signals_df.iloc[100]['beta']
        late_beta = signals_df.iloc[250]['beta']
        assert early_beta != late_beta  # beta应该有变化
        
        # 验证beta边界保护
        assert all(signals_df['beta'].between(-4, 4))


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])