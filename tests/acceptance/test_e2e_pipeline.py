#!/usr/bin/env python3
"""
端到端测试用例 - 完整配对交易流程
按照TDD模式，先定义测试用例，再实现功能
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TestE2EPipeline(unittest.TestCase):
    """端到端测试：完整配对交易流程"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config = {
            # 时间参数
            'beta_train_start': '2023-01-01',
            'beta_train_end': '2023-12-31',
            'kalman_start': '2024-01-01',
            'kalman_converge_end': '2024-06-30',
            'signal_start': '2024-07-01',
            'backtest_end': '2025-08-20',
            
            # 协整筛选参数
            'p_value_5y': 0.05,
            'p_value_1y': 0.05,
            'use_halflife_filter': False,  # 可选的半衰期筛选
            'halflife_5y_range': [5, 60],
            'halflife_1y_range': [2, 60],
            
            # 信号生成参数
            'z_open_threshold': 2.2,
            'z_close_threshold': 0.3,
            'rolling_window': 60,
            'max_holding_days': 30,
            
            # Beta约束参数
            'beta_min_abs': 0.3,
            'beta_max_abs': 3.0,
            
            # 回测参数
            'initial_capital': 5000000,
            'margin_rate': 0.12,
            'commission_rate': 0.0002,
            'slippage_ticks': 3,
            'stop_loss_pct': 0.15,
            
            # 品种列表
            'symbols': [
                'AG0', 'AU0', 'AL0', 'CU0', 'NI0', 'PB0', 
                'SN0', 'ZN0', 'HC0', 'I0', 'RB0', 'SF0', 
                'SM0', 'SS0'
            ]
        }
        
        cls.output_dir = project_root / 'output' / 'e2e_test'
        cls.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_01_data_loading(self):
        """测试1：数据加载和质量检查"""
        from scripts.pipeline.e2e_pipeline import load_and_validate_data
        
        # 执行数据加载
        data = load_and_validate_data(self.config['symbols'])
        
        # 验证数据格式
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 14, "应该有14个品种的数据")
        
        # 验证每个品种的数据
        for symbol in self.config['symbols']:
            self.assertIn(symbol, data)
            df = data[symbol]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('close', df.columns)
            self.assertIn('volume', df.columns)
            
            # 验证时间范围
            self.assertGreaterEqual(
                df.index[-1].strftime('%Y-%m-%d'),
                '2025-08-01',
                f"{symbol}数据应包含到2025年8月"
            )
        
        print("✅ 测试1通过：数据加载成功")
    
    def test_02_cointegration_screening(self):
        """测试2：协整配对筛选"""
        from scripts.pipeline.e2e_pipeline import (
            load_and_validate_data,
            screen_cointegrated_pairs
        )
        
        # 加载数据
        data = load_and_validate_data(self.config['symbols'])
        
        # 执行协整筛选
        pairs = screen_cointegrated_pairs(
            data,
            p_value_5y=self.config['p_value_5y'],
            p_value_1y=self.config['p_value_1y'],
            use_halflife_filter=self.config['use_halflife_filter']
        )
        
        # 验证结果
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0, "应该至少有一对通过协整检验")
        
        # 验证每对的格式
        for pair in pairs:
            self.assertIsInstance(pair, dict)
            self.assertIn('pair', pair)
            self.assertIn('p_value_5y', pair)
            self.assertIn('p_value_1y', pair)
            self.assertIn('beta_ols', pair)
            
            # 验证p值条件
            self.assertLess(pair['p_value_5y'], self.config['p_value_5y'])
            self.assertLess(pair['p_value_1y'], self.config['p_value_1y'])
        
        print(f"✅ 测试2通过：筛选出{len(pairs)}对协整配对")
    
    def test_03_beta_estimation(self):
        """测试3：Beta系数估计"""
        from scripts.pipeline.e2e_pipeline import (
            load_and_validate_data,
            screen_cointegrated_pairs,
            estimate_betas
        )
        
        # 准备数据
        data = load_and_validate_data(self.config['symbols'])
        pairs = screen_cointegrated_pairs(data)
        
        # 估计Beta
        betas = estimate_betas(
            data,
            pairs,
            train_start=self.config['beta_train_start'],
            train_end=self.config['beta_train_end']
        )
        
        # 验证Beta结果
        self.assertIsInstance(betas, dict)
        # 由于Beta约束筛选，betas数量应该小于等于pairs数量
        self.assertLessEqual(len(betas), len(pairs), "Beta数量应≤配对数量（因约束筛选）")
        self.assertGreater(len(betas), 0, "应至少有一对通过Beta约束")
        
        # 验证Beta约束
        for pair_name, beta in betas.items():
            self.assertIsInstance(beta, float)
            self.assertGreaterEqual(
                abs(beta), 
                self.config['beta_min_abs'],
                f"{pair_name}的Beta绝对值应≥{self.config['beta_min_abs']}"
            )
            self.assertLessEqual(
                abs(beta),
                self.config['beta_max_abs'],
                f"{pair_name}的Beta绝对值应≤{self.config['beta_max_abs']}"
            )
        
        print(f"✅ 测试3通过：估计了{len(betas)}对的Beta系数")
    
    def test_04_signal_generation(self):
        """测试4：信号生成"""
        from scripts.pipeline.e2e_pipeline import (
            load_and_validate_data,
            screen_cointegrated_pairs,
            estimate_betas,
            generate_signals
        )
        
        # 准备数据
        data = load_and_validate_data(self.config['symbols'])
        pairs = screen_cointegrated_pairs(data)
        betas = estimate_betas(data, pairs)
        
        # 生成信号
        signals = generate_signals(
            data,
            pairs,
            betas,
            kalman_start=self.config['kalman_start'],
            signal_start=self.config['signal_start'],
            signal_end=self.config['backtest_end'],
            z_open_threshold=self.config['z_open_threshold'],
            z_close_threshold=self.config['z_close_threshold']
        )
        
        # 验证信号格式
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertGreater(len(signals), 0, "应该生成至少一个信号")
        
        # 验证必需的13个字段
        required_fields = [
            'date', 'pair', 'x_symbol', 'y_symbol',
            'signal', 'zscore', 'spread', 'beta',
            'x_price', 'y_price', 'spread_mean', 
            'spread_std', 'position'
        ]
        for field in required_fields:
            self.assertIn(field, signals.columns, f"缺少必需字段：{field}")
        
        # 验证信号类型
        valid_signals = {'open_long', 'open_short', 'close', 'hold'}
        unique_signals = signals['signal'].unique()
        for sig in unique_signals:
            self.assertIn(sig, valid_signals, f"无效信号类型：{sig}")
        
        # 验证Z-score阈值
        open_signals = signals[signals['signal'].isin(['open_long', 'open_short'])]
        if len(open_signals) > 0:
            self.assertTrue(
                (abs(open_signals['zscore']) >= self.config['z_open_threshold'] - 0.1).all(),
                "开仓信号的Z-score应接近或超过阈值"
            )
        
        print(f"✅ 测试4通过：生成了{len(signals)}个交易信号")
    
    def test_05_backtest_execution(self):
        """测试5：回测执行"""
        from scripts.pipeline.e2e_pipeline import (
            load_and_validate_data,
            screen_cointegrated_pairs,
            estimate_betas,
            generate_signals,
            run_backtest
        )
        
        # 准备数据
        data = load_and_validate_data(self.config['symbols'])
        pairs = screen_cointegrated_pairs(data)
        betas = estimate_betas(data, pairs)
        signals = generate_signals(data, pairs, betas)
        
        # 运行回测
        results = run_backtest(
            signals,
            data,
            initial_capital=self.config['initial_capital'],
            margin_rate=self.config['margin_rate'],
            commission_rate=self.config['commission_rate'],
            slippage_ticks=self.config['slippage_ticks'],
            stop_loss_pct=self.config['stop_loss_pct'],
            max_holding_days=self.config['max_holding_days']
        )
        
        # 验证回测结果
        self.assertIsInstance(results, dict)
        
        # 验证必需的输出
        required_outputs = [
            'trades', 'daily_pnl', 'metrics', 'positions'
        ]
        for output in required_outputs:
            self.assertIn(output, results, f"缺少输出：{output}")
        
        # 验证交易记录
        trades = results['trades']
        if len(trades) > 0:
            self.assertIsInstance(trades, pd.DataFrame)
            trade_columns = [
                'date', 'pair', 'action', 'lots_y', 'lots_x',
                'price_y', 'price_x', 'pnl', 'commission'
            ]
            for col in trade_columns:
                self.assertIn(col, trades.columns)
        
        # 验证绩效指标
        metrics = results['metrics']
        self.assertIsInstance(metrics, dict)
        metric_keys = [
            'total_pnl', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'total_trades'
        ]
        for key in metric_keys:
            self.assertIn(key, metrics)
            
        print(f"✅ 测试5通过：回测执行完成")
    
    def test_06_performance_metrics(self):
        """测试6：绩效指标计算"""
        from scripts.pipeline.e2e_pipeline import (
            load_and_validate_data,
            screen_cointegrated_pairs,
            estimate_betas,
            generate_signals,
            run_backtest,
            calculate_performance_metrics
        )
        
        # 运行完整流程
        data = load_and_validate_data(self.config['symbols'])
        pairs = screen_cointegrated_pairs(data)
        betas = estimate_betas(data, pairs)
        signals = generate_signals(data, pairs, betas)
        results = run_backtest(signals, data)
        
        # 计算绩效指标
        metrics = calculate_performance_metrics(results)
        
        # 验证指标完整性
        self.assertIsInstance(metrics, dict)
        
        # 验证关键指标
        self.assertIn('total_pnl', metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        
        # 验证指标合理性
        self.assertIsInstance(metrics['total_pnl'], (int, float))
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertLessEqual(metrics['max_drawdown'], 0)
        
        print(f"✅ 测试6通过：绩效指标计算完成")
    
    def test_07_report_generation(self):
        """测试7：报告生成"""
        from scripts.pipeline.e2e_pipeline import (
            run_complete_pipeline,
            generate_report
        )
        
        # 运行完整流程
        results = run_complete_pipeline(self.config)
        
        # 生成报告
        report_path = generate_report(
            results,
            output_dir=self.output_dir
        )
        
        # 验证报告文件
        self.assertTrue(report_path.exists(), "报告文件应该存在")
        
        # 验证报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键部分
        self.assertIn('配对交易回测报告', content)
        self.assertIn('参数配置', content)
        self.assertIn('协整筛选结果', content)
        self.assertIn('交易统计', content)
        self.assertIn('绩效指标', content)
        
        print(f"✅ 测试7通过：报告生成成功")
    
    def test_08_end_to_end_integration(self):
        """测试8：完整端到端集成测试"""
        from scripts.pipeline.e2e_pipeline import run_complete_pipeline
        
        # 运行完整流程
        results = run_complete_pipeline(self.config)
        
        # 验证完整结果
        self.assertIsInstance(results, dict)
        
        # 验证所有模块输出
        expected_keys = [
            'config', 'data_info', 'pairs', 'betas',
            'signals', 'backtest_results', 'metrics',
            'report_path'
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"缺少输出：{key}")
        
        # 验证数据流完整性
        pairs = results['pairs']
        betas = results['betas']
        signals = results['signals']
        
        # 配对数量一致性
        self.assertEqual(
            len(pairs), 
            len(betas),
            "配对数量和Beta数量应该一致"
        )
        
        # 信号配对验证
        signal_pairs = signals['pair'].unique()
        for pair in signal_pairs:
            self.assertIn(
                pair,
                [p['pair'] for p in pairs],
                f"信号中的配对{pair}应该在筛选结果中"
            )
        
        # 验证最终绩效
        metrics = results['metrics']
        self.assertIsNotNone(metrics['total_pnl'])
        self.assertIsNotNone(metrics['sharpe_ratio'])
        
        print("✅ 测试8通过：端到端集成测试成功")
        print(f"  - 筛选配对: {len(pairs)}对")
        print(f"  - 生成信号: {len(signals)}个")
        print(f"  - 总收益: {metrics['total_pnl']:,.0f}")
        print(f"  - 夏普比率: {metrics['sharpe_ratio']:.2f}")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestE2EPipeline)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    # 运行测试
    success = run_tests()
    
    if success:
        print("\n" + "="*60)
        print("🎉 所有端到端测试通过！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 部分测试失败，请检查实现")
        print("="*60)
        sys.exit(1)