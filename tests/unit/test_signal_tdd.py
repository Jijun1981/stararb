#!/usr/bin/env python3
"""
信号生成模块TDD测试
测试卡尔曼滤波参数必须写死的需求
"""

import unittest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestKalmanParametersFixed(unittest.TestCase):
    """测试卡尔曼滤波参数必须写死"""
    
    def test_kalman_filter_no_configurable_params(self):
        """
        TDD测试：KalmanFilter1D不应该接受可配置参数
        需求：所有卡尔曼滤波参数写死，不可配置
        """
        from lib.signal_generation import KalmanFilter1D
        
        # 应该只接受initial_beta参数，其他参数都写死
        kf = KalmanFilter1D(initial_beta=0.8)
        
        # 验证所有参数都是固定值
        self.assertEqual(kf.Q, 1e-4)  # 固定过程噪声
        self.assertEqual(kf.P, 0.1)   # 固定初始不确定性
        self.assertEqual(kf.R, 1.0)   # 固定观测噪声（或从历史估计但算法固定）
        
        # 不应该有配置参数的构造函数
        import inspect
        sig = inspect.signature(KalmanFilter1D.__init__)
        params = list(sig.parameters.keys())
        
        # 除了self和initial_beta，不应该有其他参数
        expected_params = ['self', 'initial_beta']
        self.assertEqual(params, expected_params, 
                        f"KalmanFilter1D构造函数有多余参数: {params}")

class TestSignalGenerationTDD(unittest.TestCase):
    """测试实际信号生成功能"""
    
    def test_generate_simple_signal(self):
        """
        TDD测试：生成一个简单的信号
        使用前面模块的真实数据
        """
        import pandas as pd
        import numpy as np
        from lib.signal_generation import SignalGenerator, KalmanFilter1D
        
        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 模拟协整数据：Y = β * X + 噪声
        np.random.seed(42)
        true_beta = 0.8
        x_prices = np.cumsum(np.random.randn(100) * 0.1) + 100  # 随机游走
        y_prices = true_beta * x_prices + np.random.randn(100) * 0.1
        
        # 创建配对数据
        pair_data = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'x': x_prices,
            'y': y_prices
        })
        
        # 协整模块输出格式
        pair_info = {
            'pair': 'TEST-PAIR',
            'symbol_x': 'TEST',
            'symbol_y': 'PAIR', 
            'beta_1y': true_beta
        }
        
        # 创建信号生成器
        sg = SignalGenerator(
            window=20,  # 小窗口便于测试
            z_open=2.0,
            z_close=0.5,
            convergence_days=10,
            convergence_threshold=0.02,
            max_holding_days=30
        )
        
        # 生成信号
        try:
            signals = sg.process_pair_signals(
                pair_data=pair_data,
                initial_beta=pair_info['beta_1y'],
                convergence_end='2024-01-15',
                signal_start='2024-01-16',
                pair_info=pair_info,  # 传递配对信息
                beta_window='1y'  # 指定β窗口
            )
            
            # 验证输出格式
            self.assertIsInstance(signals, pd.DataFrame)
            self.assertGreater(len(signals), 0)
            
            # 严格按需求文档REQ-4.3验证所有必需字段
            required_columns = [
                'date',           # 无硬编码日期
                'pair',           # 与协整模块格式一致：纯符号，无后缀
                'symbol_x',       # X品种（低波动）
                'symbol_y',       # Y品种（高波动） 
                'signal',         # converging, open_long, open_short, close, hold
                'z_score',        # 残差Z-score
                'residual',       # 当前残差值
                'beta',           # 当前β值
                'beta_initial',   # 初始β值（从协整模块获取）
                'days_held',      # 持仓天数（新开仓为0）
                'reason',         # 信号原因：converging, z_threshold, force_close等
                'phase',          # 阶段标识：convergence_period, signal_period
                'beta_window_used' # 使用的β值时间窗口
            ]
            
            print(f"需求字段: {required_columns}")
            print(f"现有字段: {list(signals.columns)}")
            
            # 严格检查所有字段必须存在
            missing_fields = [col for col in required_columns if col not in signals.columns]
            if missing_fields:
                self.fail(f"❌ 缺失必需字段: {missing_fields}")
            else:
                print(f"✅ 所有必需字段都存在")
            
            # 验证现有数据
            self.assertGreater(len(signals), 0)
            print(f"前3行数据:\n{signals.head(3)}")
            
            # 验证关键字段类型和值
            if 'signal' in signals.columns:
                signal_types = signals['signal'].unique()
                valid_signals = ['converging', 'open_long', 'open_short', 'close', 'hold']
                invalid_signals = [s for s in signal_types if s not in valid_signals]
                if invalid_signals:
                    print(f"❌ 无效信号类型: {invalid_signals}")
                else:
                    print(f"✅ 信号类型正确: {list(signal_types)}")
            
            # 验证字段值符合需求
            # 验证配对信息
            self.assertEqual(signals['pair'].iloc[0], 'TEST-PAIR')
            self.assertEqual(signals['symbol_x'].iloc[0], 'TEST')
            self.assertEqual(signals['symbol_y'].iloc[0], 'PAIR')
            self.assertEqual(signals['beta_initial'].iloc[0], true_beta)
            self.assertEqual(signals['beta_window_used'].iloc[0], '1y')
            print(f"✅ 配对信息正确")
            
            # 验证阶段标识
            phases = signals['phase'].unique()
            valid_phases = ['convergence_period', 'signal_period']
            for phase in phases:
                self.assertIn(phase, valid_phases, f"无效阶段: {phase}")
            print(f"✅ 阶段标识正确: {list(phases)}")
            
            # 验证信号原因
            reasons = signals['reason'].unique()
            valid_reasons = ['converging', 'z_threshold', 'force_close', 'holding', 'no_signal', 'insufficient_data', 'transition_period']
            for reason in reasons:
                self.assertIn(reason, valid_reasons, f"无效原因: {reason}")
            print(f"✅ 信号原因正确: {list(reasons)}")
            
            # 验证Kalman参数写死（通过创建新实例检查）
            kf_test = KalmanFilter1D(initial_beta=1.0)
            self.assertEqual(kf_test.Q, 1e-4, "Q参数未写死")
            self.assertEqual(kf_test.P, 0.1, "P参数未写死") 
            self.assertEqual(kf_test.R, 1.0, "R参数未写死")
            print(f"✅ Kalman参数已写死: Q={kf_test.Q}, R={kf_test.R}, P={kf_test.P}")
            
            # 验证信号类型
            signal_types = signals['signal'].unique()
            valid_signals = ['converging', 'open_long', 'open_short', 'close', 'hold']
            for signal_type in signal_types:
                self.assertIn(signal_type, valid_signals, f"无效信号类型: {signal_type}")
            
            print(f"✅ 成功生成 {len(signals)} 条信号记录")
            print(f"信号分布: {dict(signals['signal'].value_counts())}")
            
        except Exception as e:
            self.fail(f"信号生成失败: {e}")
            
    def test_kalman_filter_update(self):
        """测试卡尔曼滤波器更新功能"""
        from lib.signal_generation import KalmanFilter1D
        
        kf = KalmanFilter1D(initial_beta=0.5)
        
        # 模拟多步更新，让β收敛
        for i in range(10):
            y_t = 2.0  # 目标值
            x_t = 1.0
            result = kf.update(y_t, x_t)
        
        # 验证返回值
        self.assertIsInstance(result, dict)
        self.assertIn('beta', result)
        self.assertIn('residual', result)
        
        # β值应该向真实值（2.0）移动
        self.assertNotEqual(kf.beta, 0.5, "β值应该有变化")
        self.assertGreater(kf.beta, 0.5, "β值应该增加")
        
        # 验证历史记录
        self.assertEqual(len(kf.beta_history), 11)  # 初始值 + 10次更新

if __name__ == '__main__':
    unittest.main()