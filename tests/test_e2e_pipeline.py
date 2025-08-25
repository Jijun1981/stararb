#!/usr/bin/env python3
"""
端到端完整流程测试脚本
基于 e2e_pipeline_config.yaml 配置文件

测试流程：
1. 数据获取模块测试
2. 协整配对分析测试
3. 信号生成模块测试
4. 回测框架模块测试
5. 完整流程集成测试

Author: Claude Code
Date: 2025-08-25
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from lib.data import load_data, SYMBOLS
    from lib.coint import CointegrationAnalyzer
    from lib.signal_generation import SignalGeneratorV3
    from lib.backtest.engine import BacktestEngine, BacktestConfig
    from lib.backtest.position_sizing import PositionSizingConfig
    from lib.backtest.trade_executor import ExecutionConfig
    from lib.backtest.risk_manager import RiskConfig
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_e2e_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """端到端测试运行器"""
    
    def __init__(self, config_path: str):
        """
        初始化测试运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []
        
        # 创建输出目录
        self.output_dir = Path(self.config['output']['base_dir']) / 'test_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ 端到端测试初始化完成")
        logger.info(f"配置文件: {self.config_path}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            raise
    
    def _assert_test(self, condition: bool, test_name: str, message: str = ""):
        """测试断言辅助函数"""
        if condition:
            logger.info(f"✅ {test_name}: PASSED {message}")
            self.passed_tests.append(test_name)
            return True
        else:
            logger.error(f"❌ {test_name}: FAILED {message}")
            self.failed_tests.append(test_name)
            return False
    
    def test_1_data_module(self) -> bool:
        """测试1: 数据管理模块"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试1: 数据管理模块")
        logger.info("="*50)
        
        try:
            # 获取配置参数
            symbols = []
            for category in self.config['symbols']['metals'].values():
                symbols.extend(category)
            
            start_date = self.config['time_config']['data_start_date']
            end_date = self.config['time_config']['data_end_date']
            
            logger.info(f"测试品种: {symbols}")
            logger.info(f"时间范围: {start_date} 到 {end_date}")
            
            # 加载数据
            logger.info("正在加载数据...")
            data = load_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # 验证数据基本属性
            test_passed = True
            test_passed &= self._assert_test(
                isinstance(data, pd.DataFrame),
                "数据类型检查",
                "数据应该是DataFrame"
            )
            
            test_passed &= self._assert_test(
                len(data.columns) == len(symbols),
                "品种数量检查",
                f"应该有{len(symbols)}个品种，实际{len(data.columns)}"
            )
            
            test_passed &= self._assert_test(
                isinstance(data.index, pd.DatetimeIndex),
                "索引类型检查",
                "索引应该是DatetimeIndex"
            )
            
            test_passed &= self._assert_test(
                len(data) > 1000,
                "数据量检查",
                f"数据量: {len(data)} 行"
            )
            
            # 检查数据质量
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            test_passed &= self._assert_test(
                missing_ratio < 0.1,
                "数据完整性检查",
                f"缺失率: {missing_ratio:.2%}"
            )
            
            # 保存测试数据
            self.test_data = data
            logger.info(f"数据形状: {data.shape}")
            logger.info(f"时间跨度: {data.index.min()} 到 {data.index.max()}")
            
            self.test_results['data_module'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'data_shape': data.shape,
                'symbols': list(data.columns),
                'date_range': [str(data.index.min()), str(data.index.max())],
                'missing_ratio': missing_ratio
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 数据模块测试异常: {e}")
            self.test_results['data_module'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def test_2_cointegration_module(self) -> bool:
        """测试2: 协整配对分析模块"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试2: 协整配对分析模块")
        logger.info("="*50)
        
        try:
            if not hasattr(self, 'test_data'):
                logger.error("❌ 需要先运行数据模块测试")
                return False
            
            # 创建协整分析器
            logger.info("创建协整分析器...")
            analyzer = CointegrationAnalyzer(self.test_data)
            
            # 测试基本功能
            test_passed = True
            test_passed &= self._assert_test(
                hasattr(analyzer, 'data'),
                "分析器初始化检查",
                "分析器应该包含数据属性"
            )
            
            # 进行协整分析（测试少量配对）
            logger.info("进行协整分析...")
            test_symbols = list(self.test_data.columns)[:6]  # 只测试前6个品种
            
            # 生成配对
            from itertools import combinations
            pairs = list(combinations(test_symbols, 2))
            logger.info(f"测试配对数量: {len(pairs)}")
            
            # 测试单个配对分析
            if pairs:
                test_pair = pairs[0]
                logger.info(f"测试配对: {test_pair[0]}-{test_pair[1]}")
                
                try:
                    result = analyzer.test_cointegration(
                        test_pair[0], 
                        test_pair[1], 
                        window='5y'
                    )
                    
                    test_passed &= self._assert_test(
                        isinstance(result, dict),
                        "单配对分析结果类型",
                        "结果应该是字典"
                    )
                    
                    required_keys = ['pvalue', 'beta', 'halflife', 'r_squared']
                    for key in required_keys:
                        test_passed &= self._assert_test(
                            key in result,
                            f"结果包含{key}",
                            f"协整结果应包含{key}"
                        )
                    
                    logger.info(f"示例结果: pvalue={result.get('pvalue', 'N/A'):.4f}, beta={result.get('beta', 'N/A'):.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ 单配对分析失败: {e}")
                    test_passed = False
            
            # 保存测试结果
            self.test_analyzer = analyzer
            self.test_pairs = pairs[:10]  # 保存前10个配对用于后续测试
            
            self.test_results['cointegration_module'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'total_pairs': len(pairs),
                'test_symbols': test_symbols
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 协整模块测试异常: {e}")
            self.test_results['cointegration_module'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def test_3_signal_generation_module(self) -> bool:
        """测试3: 信号生成模块"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试3: 信号生成模块")
        logger.info("="*50)
        
        try:
            if not hasattr(self, 'test_data') or not hasattr(self, 'test_pairs'):
                logger.error("❌ 需要先运行前面的测试")
                return False
            
            # 获取配置参数
            signal_config = self.config['signal_generation']
            
            # 创建信号生成器
            logger.info("创建信号生成器...")
            generator = SignalGeneratorV3(
                signal_start_date=self.config['time_config']['signal_generation_start'],
                kalman_warmup_days=signal_config['kalman_warmup'],
                ols_training_days=signal_config['ols_window'],
                z_open=signal_config['signal_thresholds']['z_open'],
                z_close=signal_config['signal_thresholds']['z_close'],
                max_holding_days=signal_config['signal_thresholds']['max_holding_days'],
                Q_beta=signal_config['kalman_params']['Q_beta'],
                Q_alpha=signal_config['kalman_params']['Q_alpha'],
                R_init=signal_config['kalman_params']['R_init'],
                R_adapt=signal_config['kalman_params']['R_adapt']
            )
            
            test_passed = True
            test_passed &= self._assert_test(
                hasattr(generator, 'signal_start_date'),
                "生成器初始化检查",
                "生成器应该包含signal_start_date属性"
            )
            
            # 测试单个配对的信号生成
            if self.test_pairs:
                test_pair = self.test_pairs[0]
                symbol_x, symbol_y = test_pair
                logger.info(f"测试信号生成配对: {symbol_x}-{symbol_y}")
                
                try:
                    signals = generator.process_pair(
                        pair_name=f"{symbol_x}-{symbol_y}",
                        x_data=self.test_data[symbol_x],
                        y_data=self.test_data[symbol_y],
                        initial_beta=1.0  # 使用默认beta
                    )
                    
                    test_passed &= self._assert_test(
                        isinstance(signals, pd.DataFrame),
                        "信号结果类型检查",
                        "信号结果应该是DataFrame"
                    )
                    
                    test_passed &= self._assert_test(
                        'z_score' in signals.columns,
                        "信号包含z_score",
                        "信号结果应包含z_score列"
                    )
                    
                    test_passed &= self._assert_test(
                        'trade_signal' in signals.columns,
                        "信号包含trade_signal",
                        "信号结果应包含trade_signal列"
                    )
                    
                    test_passed &= self._assert_test(
                        'beta' in signals.columns,
                        "信号包含beta",
                        "信号结果应包含beta列"
                    )
                    
                    # 检查信号数量
                    signal_counts = signals['trade_signal'].value_counts()
                    logger.info(f"信号统计: {dict(signal_counts)}")
                    
                    test_passed &= self._assert_test(
                        len(signals) > 0,
                        "信号数量检查",
                        f"生成了{len(signals)}个信号点"
                    )
                    
                    # 保存测试信号
                    self.test_signals = signals
                    self.test_signal_pair = f"{symbol_x}-{symbol_y}"
                    self.test_signal_symbols = (symbol_x, symbol_y)
                    
                    logger.info(f"信号生成成功，形状: {signals.shape}")
                    
                except Exception as e:
                    logger.error(f"❌ 信号生成失败: {e}")
                    test_passed = False
            
            self.test_results['signal_generation_module'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'config_params': signal_config,
                'test_pair': self.test_signal_pair if hasattr(self, 'test_signal_pair') else None
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 信号生成模块测试异常: {e}")
            self.test_results['signal_generation_module'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def test_4_backtest_module(self) -> bool:
        """测试4: 回测框架模块"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试4: 回测框架模块")
        logger.info("="*50)
        
        try:
            if not hasattr(self, 'test_signals') or not hasattr(self, 'test_data'):
                logger.error("❌ 需要先运行前面的测试")
                return False
            
            # 获取回测配置
            backtest_config = self.config['backtest']
            contract_specs = self.config['contract_specs']
            
            # 创建配置对象
            sizing_config = PositionSizingConfig(
                max_denominator=backtest_config['position_sizing']['max_denominator'],
                min_lots=backtest_config['position_sizing']['min_lots'],
                max_lots_per_leg=backtest_config['position_sizing']['max_lots_per_leg'],
                margin_rate=backtest_config['capital_management']['margin_rate'],
                position_weight=backtest_config['capital_management']['position_weight']
            )
            
            execution_config = ExecutionConfig(
                commission_rate=backtest_config['trading_costs']['commission_rate'],
                slippage_ticks=backtest_config['trading_costs']['slippage_ticks'],
                margin_rate=backtest_config['capital_management']['margin_rate']
            )
            
            risk_config = RiskConfig(
                stop_loss_pct=backtest_config['risk_management']['stop_loss_pct'],
                max_holding_days=backtest_config['risk_management']['max_holding_days'],
                max_positions=20  # 设置最大持仓数
            )
            
            config = BacktestConfig(
                initial_capital=backtest_config['capital_management']['initial_capital'],
                sizing_config=sizing_config,
                execution_config=execution_config,
                risk_config=risk_config
            )
            
            test_passed = True
            test_passed &= self._assert_test(
                config.initial_capital > 0,
                "回测配置检查",
                f"初始资金: {config.initial_capital:,.0f}"
            )
            
            # 创建回测引擎
            logger.info("创建回测引擎...")
            engine = BacktestEngine(config)
            
            # 设置合约规格
            symbol_x, symbol_y = self.test_signal_symbols
            test_specs = {
                symbol_x: contract_specs[symbol_x],
                symbol_y: contract_specs[symbol_y]
            }
            engine.executor.set_contract_specs(test_specs)
            
            test_passed &= self._assert_test(
                hasattr(engine, 'executor'),
                "回测引擎初始化",
                "引擎应该包含executor属性"
            )
            
            # 准备回测数据
            signals_df = self.test_signals.copy()
            signals_df['date'] = signals_df.index
            signals_df['pair'] = self.test_signal_pair
            signals_df['symbol_x'] = symbol_x
            signals_df['symbol_y'] = symbol_y
            
            # 确保包含所有必需的列
            required_columns = ['date', 'pair', 'symbol_x', 'symbol_y', 'trade_signal', 'beta']
            missing_columns = [col for col in required_columns if col not in signals_df.columns]
            
            test_passed &= self._assert_test(
                len(missing_columns) == 0,
                "信号数据格式检查",
                f"缺失列: {missing_columns}" if missing_columns else "包含所有必需列"
            )
            
            # 运行回测
            logger.info("运行回测...")
            try:
                results = engine.run(
                    signals=signals_df,
                    prices=self.test_data
                )
                
                test_passed &= self._assert_test(
                    isinstance(results, dict),
                    "回测结果类型检查",
                    "结果应该是字典"
                )
                
                required_keys = ['trades', 'equity_curve', 'metrics']
                for key in required_keys:
                    test_passed &= self._assert_test(
                        key in results,
                        f"结果包含{key}",
                        f"回测结果应包含{key}"
                    )
                
                # 检查回测指标
                if 'metrics' in results:
                    metrics = results['metrics']
                    logger.info(f"回测指标: {list(metrics.keys())}")
                    
                    if 'total_trades' in metrics:
                        logger.info(f"总交易次数: {metrics['total_trades']}")
                    
                    if 'total_pnl' in metrics:
                        logger.info(f"总盈亏: {metrics['total_pnl']:,.2f}")
                
                self.test_backtest_results = results
                
            except Exception as e:
                logger.error(f"❌ 回测执行失败: {e}")
                test_passed = False
            
            self.test_results['backtest_module'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'config': {
                    'initial_capital': config.initial_capital,
                    'margin_rate': config.sizing_config.margin_rate,
                    'commission_rate': config.execution_config.commission_rate
                }
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 回测模块测试异常: {e}")
            self.test_results['backtest_module'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def test_5_full_pipeline_integration(self) -> bool:
        """测试5: 完整流程集成测试"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试5: 完整流程集成测试")
        logger.info("="*50)
        
        try:
            # 运行精简版的完整流程
            logger.info("运行完整流程集成测试...")
            
            # 1. 重新加载数据（少量品种）
            test_symbols = ['CU', 'ZN', 'AL']  # 只测试3个品种
            logger.info(f"集成测试品种: {test_symbols}")
            
            data = load_data(
                symbols=test_symbols,
                start_date='2024-01-01',  # 使用较短时间段
                end_date='2024-12-31'
            )
            
            # 2. 协整分析
            logger.info("执行协整分析...")
            analyzer = CointegrationAnalyzer(data)
            
            # 手动创建一个测试配对
            test_pair_name = f"{test_symbols[0]}-{test_symbols[1]}"
            logger.info(f"集成测试配对: {test_pair_name}")
            
            # 3. 信号生成
            logger.info("生成交易信号...")
            signal_config = self.config['signal_generation']
            generator = SignalGeneratorV3(
                signal_start_date='2024-07-01',
                kalman_warmup_days=30,
                ols_training_days=60,
                z_open=2.0,
                z_close=0.5,
                max_holding_days=30,
                Q_beta=5e-6,
                Q_alpha=1e-5,
                R_init=0.005,
                R_adapt=True
            )
            
            signals = generator.process_pair(
                pair_name=test_pair_name,
                x_data=data[test_symbols[0]],
                y_data=data[test_symbols[1]],
                initial_beta=1.0
            )
            
            # 4. 回测执行
            logger.info("执行回测...")
            # 创建配置（使用简化的配置）
            config = BacktestConfig(
                initial_capital=1000000,  # 100万测试资金
                sizing_config=PositionSizingConfig(
                    margin_rate=0.12,
                    position_weight=0.1
                ),
                execution_config=ExecutionConfig(
                    commission_rate=0.0002,
                    slippage_ticks=3,
                    margin_rate=0.12
                ),
                risk_config=RiskConfig(
                    stop_loss_pct=0.30,
                    max_holding_days=30,
                    max_positions=10
                )
            )
            
            engine = BacktestEngine(config)
            
            # 设置合约规格
            contract_specs = self.config['contract_specs']
            test_specs = {
                test_symbols[0]: contract_specs[test_symbols[0]],
                test_symbols[1]: contract_specs[test_symbols[1]]
            }
            engine.executor.set_contract_specs(test_specs)
            
            # 准备信号数据
            signals_df = signals.copy()
            signals_df['date'] = signals_df.index
            signals_df['pair'] = test_pair_name
            signals_df['symbol_x'] = test_symbols[0]
            signals_df['symbol_y'] = test_symbols[1]
            
            # 运行回测
            results = engine.run(signals=signals_df, prices=data)
            
            # 验证结果
            test_passed = True
            test_passed &= self._assert_test(
                isinstance(results, dict),
                "集成测试结果类型",
                "完整流程应该返回字典结果"
            )
            
            test_passed &= self._assert_test(
                'metrics' in results,
                "集成测试包含指标",
                "完整流程应该包含绩效指标"
            )
            
            # 记录最终结果
            if 'metrics' in results and results['metrics']:
                logger.info("📊 完整流程测试结果:")
                metrics = results['metrics']
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:,.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            self.test_results['full_pipeline_integration'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'test_symbols': test_symbols,
                'test_pair': test_pair_name,
                'data_points': len(data),
                'signal_points': len(signals),
                'final_metrics': results.get('metrics', {}) if test_passed else None
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 完整流程集成测试异常: {e}")
            self.test_results['full_pipeline_integration'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def test_6_detailed_validation(self) -> bool:
        """测试6: 详细结果验证和报告生成"""
        logger.info("\n" + "="*50)
        logger.info("🧪 测试6: 详细结果验证和报告生成")
        logger.info("="*50)
        
        try:
            # 生成测试报告
            report = self._generate_test_report()
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"e2e_test_report_{timestamp}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"📄 测试报告已保存: {report_path}")
            
            # 保存详细结果
            results_path = self.output_dir / f"e2e_test_results_{timestamp}.json"
            import json
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 详细结果已保存: {results_path}")
            
            # 验证测试完整性
            expected_tests = [
                'data_module',
                'cointegration_module', 
                'signal_generation_module',
                'backtest_module',
                'full_pipeline_integration'
            ]
            
            completed_tests = list(self.test_results.keys())
            missing_tests = [t for t in expected_tests if t not in completed_tests]
            
            test_passed = self._assert_test(
                len(missing_tests) == 0,
                "测试完整性检查",
                f"缺失测试: {missing_tests}" if missing_tests else "所有测试已完成"
            )
            
            self.test_results['detailed_validation'] = {
                'status': 'PASSED' if test_passed else 'FAILED',
                'report_path': str(report_path),
                'results_path': str(results_path),
                'completed_tests': completed_tests,
                'missing_tests': missing_tests
            }
            
            return test_passed
            
        except Exception as e:
            logger.error(f"❌ 详细验证测试异常: {e}")
            self.test_results['detailed_validation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def _generate_test_report(self) -> str:
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "# 端到端流程测试报告",
            f"**生成时间**: {timestamp}",
            f"**配置文件**: {self.config_path}",
            "",
            "## 测试概要",
            f"- ✅ 通过测试: {len(self.passed_tests)}",
            f"- ❌ 失败测试: {len(self.failed_tests)}",
            f"- 📊 总体成功率: {len(self.passed_tests)/(len(self.passed_tests)+len(self.failed_tests))*100:.1f}%",
            "",
            "## 测试详情",
        ]
        
        for test_name, result in self.test_results.items():
            report_lines.extend([
                f"### {test_name}",
                f"**状态**: {result['status']}",
                ""
            ])
            
            if result['status'] == 'ERROR':
                report_lines.extend([
                    f"**错误信息**: {result.get('error', 'N/A')}",
                    ""
                ])
            else:
                # 添加具体测试结果
                for key, value in result.items():
                    if key != 'status':
                        report_lines.append(f"- {key}: {value}")
                report_lines.append("")
        
        if self.failed_tests:
            report_lines.extend([
                "## 失败测试列表",
                ""
            ])
            for test in self.failed_tests:
                report_lines.append(f"- ❌ {test}")
            report_lines.append("")
        
        report_lines.extend([
            "## 配置参数",
            f"- 初始资金: {self.config['backtest']['capital_management']['initial_capital']:,}",
            f"- 保证金率: {self.config['backtest']['capital_management']['margin_rate']*100}%",
            f"- 仓位权重: {self.config['backtest']['capital_management']['position_weight']*100}%",
            f"- 开仓阈值: {self.config['signal_generation']['signal_thresholds']['z_open']}",
            f"- 平仓阈值: {self.config['signal_generation']['signal_thresholds']['z_close']}",
            "",
            "---",
            "*报告由端到端测试脚本自动生成*"
        ])
        
        return "\n".join(report_lines)
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        logger.info("🚀 开始端到端测试")
        logger.info(f"测试配置: {self.config['pipeline']['name']}")
        
        start_time = datetime.now()
        
        # 按顺序执行所有测试
        all_tests = [
            self.test_1_data_module,
            self.test_2_cointegration_module,
            self.test_3_signal_generation_module,
            self.test_4_backtest_module,
            self.test_5_full_pipeline_integration,
            self.test_6_detailed_validation
        ]
        
        overall_success = True
        
        for test_func in all_tests:
            try:
                success = test_func()
                overall_success &= success
            except Exception as e:
                logger.error(f"❌ 测试执行异常 {test_func.__name__}: {e}")
                overall_success = False
        
        # 测试总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("📋 测试总结")
        logger.info("="*60)
        logger.info(f"⏱️  测试耗时: {duration}")
        logger.info(f"✅ 通过测试: {len(self.passed_tests)}")
        logger.info(f"❌ 失败测试: {len(self.failed_tests)}")
        logger.info(f"📊 成功率: {len(self.passed_tests)/(len(self.passed_tests)+len(self.failed_tests))*100:.1f}%")
        
        if overall_success:
            logger.info("🎉 所有测试通过！端到端流程验证成功")
        else:
            logger.error("🚫 部分测试失败，请检查详细日志")
            if self.failed_tests:
                logger.error(f"失败测试: {', '.join(self.failed_tests)}")
        
        return overall_success


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='端到端流程测试脚本')
    parser.add_argument(
        '--config', 
        default='configs/e2e_pipeline_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        default='output/test_results',
        help='输出目录'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细日志输出'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 运行测试
    try:
        runner = E2ETestRunner(str(config_path))
        success = runner.run_all_tests()
        
        if success:
            logger.info("🎯 端到端测试完成 - 全部成功")
            sys.exit(0)
        else:
            logger.error("🎯 端到端测试完成 - 部分失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ 测试运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()