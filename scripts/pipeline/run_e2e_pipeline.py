#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端配对交易流水线
基于YAML配置的完整流程：协整筛选 → 信号生成 → 回测执行

特色功能:
- 1-5年全协整配对筛选
- 灵活的时间轴配置（OLS训练期 → Kalman预热期 → 信号生成期）
- 完整的回测引擎（考虑负Beta交易）
- 详细的执行报告和结果分析

版本: V1.0
创建时间: 2025-08-25
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
# 直接导入backtest.py（避免与backtest目录冲突）
import importlib.util
backtest_spec = importlib.util.spec_from_file_location("backtest_module", project_root / "lib" / "backtest.py")
backtest_module = importlib.util.module_from_spec(backtest_spec)
backtest_spec.loader.exec_module(backtest_module)

BacktestEngine = backtest_module.BacktestEngine
BacktestConfig = backtest_module.BacktestConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2EPipelineRunner:
    """端到端流水线执行器"""
    
    def __init__(self, config_path: str):
        """
        初始化流水线执行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 初始化结果存储
        self.results = {
            'cointegration': None,
            'signals': None, 
            'backtest': None,
            'execution_time': {},
            'summary': {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"配置文件加载成功: {self.config_path}")
            logger.info(f"流水线版本: {config['pipeline']['version']}")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的端到端流水线
        
        Returns:
            完整的执行结果字典
        """
        logger.info("=" * 80)
        logger.info(f"开始执行端到端配对交易流水线 - {self.config['pipeline']['name']}")
        logger.info("=" * 80)
        
        pipeline_start = datetime.now()
        
        # 步骤1: 数据加载
        self._step_1_load_data()
        
        # 步骤2: 协整配对筛选
        self._step_2_cointegration_screening()
        
        # 步骤3: 信号生成
        self._step_3_signal_generation()
        
        # 步骤4: 回测执行
        self._step_4_backtest_execution()
        
        # 步骤5: 结果分析和保存
        self._step_5_results_analysis()
        
        # 计算总执行时间
        pipeline_end = datetime.now()
        total_time = (pipeline_end - pipeline_start).total_seconds()
        self.results['execution_time']['total'] = total_time
        
        logger.info("=" * 80)
        logger.info(f"流水线执行完成! 总耗时: {total_time:.1f}秒")
        logger.info("=" * 80)
        
        return self.results
    
    def _step_1_load_data(self):
        """步骤1: 数据加载"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤1: 数据加载")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            # 加载价格数据
            logger.info("从数据源加载期货价格数据...")
            self.price_data = load_all_symbols_data()
            
            logger.info(f"数据加载成功:")
            logger.info(f"  品种数量: {len(self.price_data.columns)}")
            logger.info(f"  数据时间范围: {self.price_data.index[0]} 至 {self.price_data.index[-1]}")
            logger.info(f"  数据点数: {len(self.price_data)}")
            logger.info(f"  品种列表: {list(self.price_data.columns)}")
            
            # 验证品种完整性
            expected_symbols = []
            for category in self.config['symbols']['metals'].values():
                expected_symbols.extend(category)
            
            missing_symbols = set(expected_symbols) - set(self.price_data.columns)
            if missing_symbols:
                logger.warning(f"缺失的品种: {missing_symbols}")
            else:
                logger.info("所有期望的品种都已加载成功")
                
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.results['execution_time']['data_loading'] = step_time
        logger.info(f"数据加载完成，耗时: {step_time:.1f}秒")
    
    def _step_2_cointegration_screening(self):
        """步骤2: 协整配对筛选"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤2: 协整配对筛选")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            # 创建协整分析器
            analyzer = CointegrationAnalyzer(self.price_data)
            
            # 获取配置参数
            coint_config = self.config['cointegration']
            
            logger.info("协整筛选参数:")
            logger.info(f"  筛选窗口: {coint_config['screening_windows']}")
            logger.info(f"  p值阈值: {coint_config['p_value_threshold']}")
            logger.info(f"  筛选逻辑: {coint_config['filter_logic']}")
            
            # 运行协整筛选
            logger.info("开始协整配对筛选...")
            coint_results = analyzer.screen_all_pairs(
                screening_windows=coint_config['screening_windows'],
                p_thresholds={
                    window: coint_config['p_value_threshold'] 
                    for window in coint_config['screening_windows']
                },
                filter_logic=coint_config['filter_logic'],
                sort_by=coint_config.get('sort_by', 'pvalue_1y')
            )
            
            self.results['cointegration'] = coint_results
            
            logger.info(f"协整筛选结果:")
            logger.info(f"  通过筛选的配对数: {len(coint_results)}")
            
            if len(coint_results) > 0:
                logger.info("  前5个最佳配对:")
                for i, (_, pair) in enumerate(coint_results.head().iterrows()):
                    logger.info(f"    {i+1}. {pair['pair']}: 1年p值={pair['pvalue_1y']:.4f}")
            else:
                logger.warning("没有配对通过协整筛选条件!")
                return
                
        except Exception as e:
            logger.error(f"协整筛选失败: {e}")
            raise
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.results['execution_time']['cointegration'] = step_time
        logger.info(f"协整筛选完成，耗时: {step_time:.1f}秒")
    
    def _step_3_signal_generation(self):
        """步骤3: 信号生成"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤3: 信号生成")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            if self.results['cointegration'] is None or len(self.results['cointegration']) == 0:
                logger.error("没有协整配对可用于信号生成")
                return
            
            # 获取信号生成配置
            signal_config = self.config['signal_generation']
            time_config = self.config['time_config']
            
            # 创建信号生成器
            generator = SignalGenerator(
                # 时间配置
                signal_start_date=time_config['signal_generation_start'],
                kalman_warmup_days=signal_config['kalman_warmup'],
                ols_training_days=signal_config['ols_window'],
                
                # Kalman参数
                Q_beta=signal_config['kalman_params']['Q_beta'],
                Q_alpha=signal_config['kalman_params']['Q_alpha'],
                R_init=signal_config['kalman_params']['R_init'],
                R_adapt=signal_config['kalman_params']['R_adapt'],
                
                # 交易阈值
                z_open=signal_config['signal_thresholds']['z_open'],
                z_close=signal_config['signal_thresholds']['z_close'],
                max_holding_days=signal_config['signal_thresholds']['max_holding_days']
            )
            
            logger.info("信号生成参数:")
            logger.info(f"  信号开始日期: {time_config['signal_generation_start']}")
            logger.info(f"  OLS训练期: {signal_config['ols_window']}天")
            logger.info(f"  Kalman预热期: {signal_config['kalman_warmup']}天")
            logger.info(f"  开仓阈值: {signal_config['signal_thresholds']['z_open']}")
            logger.info(f"  平仓阈值: {signal_config['signal_thresholds']['z_close']}")
            
            # 计算数据需求范围
            data_start = generator._calculate_data_start_date()
            logger.info(f"  数据开始日期: {data_start}")
            
            # 生成信号
            logger.info(f"开始为{len(self.results['cointegration'])}个配对生成信号...")
            all_signals = generator.process_all_pairs(
                pairs_df=self.results['cointegration'],
                price_data=self.price_data,
                beta_window='1y'  # 使用1年期Beta
            )
            
            self.results['signals'] = all_signals
            
            # 统计信号结果
            logger.info(f"信号生成结果:")
            logger.info(f"  总信号记录数: {len(all_signals)}")
            logger.info(f"  信号时间范围: {all_signals['date'].min()} 至 {all_signals['date'].max()}")
            
            # 统计交易信号
            trading_signals = all_signals[all_signals['signal'].isin(['open_long', 'open_short', 'close'])]
            logger.info(f"  交易信号数: {len(trading_signals)}")
            
            if len(trading_signals) > 0:
                signal_counts = trading_signals['signal'].value_counts()
                for signal_type, count in signal_counts.items():
                    logger.info(f"    {signal_type}: {count}条")
            else:
                logger.warning("没有生成任何交易信号!")
                
        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            raise
        
        step_time = (datetime.now() - step_start).total_seconds() 
        self.results['execution_time']['signal_generation'] = step_time
        logger.info(f"信号生成完成，耗时: {step_time:.1f}秒")
    
    def _step_4_backtest_execution(self):
        """步骤4: 回测执行"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤4: 回测执行")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            if self.results['signals'] is None or len(self.results['signals']) == 0:
                logger.error("没有信号可用于回测")
                return
            
            # 获取回测配置
            backtest_config = self.config['backtest']
            
            # 创建回测配置对象
            config = BacktestConfig(
                # 资金管理
                initial_capital=backtest_config['capital_management']['initial_capital'],
                margin_rate=backtest_config['capital_management']['margin_rate'],
                position_weight=backtest_config['capital_management']['position_weight'],
                
                # 交易成本
                commission_rate=backtest_config['trading_costs']['commission_rate'],
                slippage_ticks=backtest_config['trading_costs']['slippage_ticks'],
                
                # 风险控制
                stop_loss_pct=backtest_config['risk_management']['stop_loss_pct'],
                max_holding_days=backtest_config['risk_management']['max_holding_days'],
                enable_stop_loss=backtest_config['risk_management']['enable_stop_loss'],
                enable_time_stop=backtest_config['risk_management']['enable_time_stop'],
                
                # 仓位配置
                max_denominator=backtest_config['position_sizing']['max_denominator'],
                min_lots=backtest_config['position_sizing']['min_lots'],
                max_lots_per_leg=backtest_config['position_sizing']['max_lots_per_leg'],
                
                # 执行控制
                allow_multiple_positions=backtest_config['execution']['allow_multiple_positions'],
                force_close_at_end=backtest_config['execution']['force_close_at_end']
            )
            
            logger.info("回测配置:")
            logger.info(f"  初始资金: {config.initial_capital:,.0f}元")
            logger.info(f"  仓位权重: {config.position_weight:.1%}")
            logger.info(f"  手续费率: {config.commission_rate:.4f}")
            logger.info(f"  止损比例: {config.stop_loss_pct:.1%}")
            logger.info(f"  最大持仓天数: {config.max_holding_days}")
            
            # 准备价格数据（转换为原始价格）
            backtest_prices = np.exp(self.price_data)
            
            # 准备交易信号
            trading_signals = self.results['signals'][
                self.results['signals']['signal'].isin(['open_long', 'open_short', 'close'])
            ].copy()
            
            # 保持信号列名为'signal'（回测引擎期望的格式）
            # trading_signals = trading_signals.rename(columns={'signal': 'trade_signal'})
            
            logger.info(f"开始回测执行，交易信号数: {len(trading_signals)}")
            
            # 创建回测引擎并运行
            engine = BacktestEngine(config)
            backtest_results = engine.run_backtest(
                signals=trading_signals,
                prices=backtest_prices
            )
            
            self.results['backtest'] = backtest_results
            
            # 统计回测结果
            logger.info(f"回测执行结果:")
            logger.info(f"  初始资金: {backtest_results['initial_capital']:,.0f}元")
            logger.info(f"  最终资金: {backtest_results['final_capital']:,.0f}元")
            logger.info(f"  总收益: {backtest_results['final_capital'] - backtest_results['initial_capital']:,.0f}元")
            logger.info(f"  总收益率: {(backtest_results['final_capital'] / backtest_results['initial_capital'] - 1) * 100:.2f}%")
            logger.info(f"  总交易数: {len(backtest_results['trades'])}")
            
            # 组合绩效指标
            if 'portfolio_metrics' in backtest_results:
                metrics = backtest_results['portfolio_metrics']
                logger.info(f"  年化收益率: {metrics.get('annual_return', 0):.2%}")
                logger.info(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
                logger.info(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"  胜率: {metrics.get('win_rate', 0):.1%}")
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            raise
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.results['execution_time']['backtest'] = step_time
        logger.info(f"回测执行完成，耗时: {step_time:.1f}秒")
    
    def _step_5_results_analysis(self):
        """步骤5: 结果分析和保存"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤5: 结果分析和保存")
        logger.info("=" * 60)
        
        step_start = datetime.now()
        
        try:
            # 创建输出目录
            output_dir = Path(self.config['output']['base_dir'])
            output_dir.mkdir(exist_ok=True)
            
            for subdir in self.config['output']['subdirs'].values():
                (output_dir / subdir).mkdir(exist_ok=True)
            
            # 保存协整结果
            if self.results['cointegration'] is not None:
                coint_file = output_dir / self.config['output']['subdirs']['analysis'] / f"cointegration_results_{self.timestamp}.csv"
                self.results['cointegration'].to_csv(coint_file, index=False)
                logger.info(f"协整结果已保存: {coint_file}")
            
            # 保存信号数据
            if self.results['signals'] is not None:
                signals_file = output_dir / self.config['output']['subdirs']['signals'] / f"signals_{self.timestamp}.csv"
                self.results['signals'].to_csv(signals_file, index=False)
                logger.info(f"信号数据已保存: {signals_file}")
            
            # 保存回测结果
            if self.results['backtest'] is not None:
                backtest_dir = output_dir / self.config['output']['subdirs']['backtest']
                
                # 保存交易记录
                if self.results['backtest']['trades']:
                    trades_df = pd.DataFrame([{
                        'pair': trade.pair,
                        'direction': trade.direction,
                        'open_date': trade.open_date,
                        'close_date': trade.close_date,
                        'holding_days': trade.holding_days,
                        'net_pnl': trade.net_pnl,
                        'return_pct': trade.return_pct,
                        'close_reason': trade.close_reason
                    } for trade in self.results['backtest']['trades']])
                    
                    trades_file = backtest_dir / f"trades_{self.timestamp}.csv"
                    trades_df.to_csv(trades_file, index=False)
                    logger.info(f"交易记录已保存: {trades_file}")
                
                # 保存绩效指标
                if 'portfolio_metrics' in self.results['backtest']:
                    metrics_file = backtest_dir / f"metrics_{self.timestamp}.csv"
                    pd.Series(self.results['backtest']['portfolio_metrics']).to_csv(metrics_file, header=['value'])
                    logger.info(f"绩效指标已保存: {metrics_file}")
            
            # 生成执行摘要
            self._generate_summary_report()
            
        except Exception as e:
            logger.error(f"结果保存失败: {e}")
            raise
        
        step_time = (datetime.now() - step_start).total_seconds()
        self.results['execution_time']['results_analysis'] = step_time
        logger.info(f"结果分析和保存完成，耗时: {step_time:.1f}秒")
    
    def _generate_summary_report(self):
        """生成执行摘要报告"""
        summary = {
            'pipeline_info': {
                'version': self.config['pipeline']['version'],
                'execution_time': self.timestamp,
                'total_runtime': self.results['execution_time']['total']
            },
            'data_info': {
                'symbols_count': len(self.price_data.columns),
                'data_range': f"{self.price_data.index[0]} to {self.price_data.index[-1]}"
            }
        }
        
        if self.results['cointegration'] is not None:
            summary['cointegration_info'] = {
                'pairs_screened': len(self.results['cointegration']),
                'screening_criteria': self.config['cointegration']['screening_windows']
            }
        
        if self.results['signals'] is not None:
            trading_signals = self.results['signals'][self.results['signals']['signal'].isin(['open_long', 'open_short', 'close'])]
            summary['signals_info'] = {
                'total_records': len(self.results['signals']),
                'trading_signals': len(trading_signals),
                'signal_period': f"{self.results['signals']['date'].min()} to {self.results['signals']['date'].max()}"
            }
        
        if self.results['backtest'] is not None:
            summary['backtest_info'] = {
                'initial_capital': self.results['backtest']['initial_capital'],
                'final_capital': self.results['backtest']['final_capital'],
                'total_return': (self.results['backtest']['final_capital'] / self.results['backtest']['initial_capital'] - 1) * 100,
                'total_trades': len(self.results['backtest']['trades'])
            }
            
            if 'portfolio_metrics' in self.results['backtest']:
                summary['performance_metrics'] = self.results['backtest']['portfolio_metrics']
        
        self.results['summary'] = summary
        
        # 保存摘要报告
        output_dir = Path(self.config['output']['base_dir'])
        summary_file = output_dir / self.config['output']['subdirs']['reports'] / f"pipeline_summary_{self.timestamp}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"执行摘要已保存: {summary_file}")


def main():
    """主函数"""
    # 默认配置文件路径
    default_config = project_root / 'configs' / 'e2e_pipeline_config.yaml'
    
    # 从命令行获取配置文件路径
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    
    try:
        # 创建流水线执行器
        runner = E2EPipelineRunner(config_path)
        
        # 运行完整流水线
        results = runner.run_full_pipeline()
        
        # 输出最终摘要
        logger.info("\n" + "=" * 80)
        logger.info("流水线执行摘要")
        logger.info("=" * 80)
        
        summary = results['summary']
        if 'cointegration_info' in summary:
            logger.info(f"协整配对数: {summary['cointegration_info']['pairs_screened']}")
        
        if 'signals_info' in summary:
            logger.info(f"交易信号数: {summary['signals_info']['trading_signals']}")
        
        if 'backtest_info' in summary:
            logger.info(f"总收益率: {summary['backtest_info']['total_return']:.2f}%")
            logger.info(f"总交易数: {summary['backtest_info']['total_trades']}")
        
        if 'performance_metrics' in summary:
            metrics = summary['performance_metrics']
            logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        
        logger.info(f"总执行时间: {results['execution_time']['total']:.1f}秒")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()