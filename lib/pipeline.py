#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端流程模块 (Pipeline Module) V1.1
将数据管理、协整筛选、信号生成、回测执行串联成完整流程

主要功能：
1. 统一配置管理 (PipelineConfig)
2. 流程编排执行 (TradingPipeline)
3. 结果收集输出
4. 错误处理和日志

版本历史：
- V1.0: 初始版本，基于OLS信号生成
- V1.1: 调整参数格式，移除缓存功能
"""

import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import pandas as pd
import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data import load_all_symbols_data, load_symbol_data
from coint import CointegrationAnalyzer
from signal_generation_ols import SignalGeneratorOLS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置管理类 (REQ-6.1)
# ============================================================================

class PipelineConfig:
    """
    端到端流程配置类 (REQ-6.1.1)
    管理所有模块的参数配置
    """
    
    def __init__(self):
        """初始化默认配置"""
        # 数据参数
        self.data_start = None  # 数据开始日期（用于加载）
        self.backtest_start = None  # 回测开始日期
        self.backtest_end = None  # 回测结束日期
        
        # 配对选择
        self.target_pairs = None  # 指定分析的配对
        self.symbols = None  # 指定品种列表
        
        # 协整参数（简化版）
        self.coint_start_date = None  # 协整数据开始日期
        self.coint_end_date = None  # 协整数据结束日期
        self.coint_pvalue = 0.05  # p值阈值
        
        # 信号参数（OLS版本）
        self.signal_method = 'ols'  # 信号生成方法
        self.signal_start_date = None  # 信号生成开始日期
        self.window_size = 45  # OLS滚动窗口
        self.z_open = 2.0  # 开仓阈值
        self.z_close = 0.5  # 平仓阈值
        self.max_holding_days = 30  # 最大持仓天数
        
        # 回测参数
        self.initial_capital = 5000000  # 初始资金
        self.position_weight = 0.05  # 每个配对资金权重
        self.stop_loss = 0.10  # 止损比例
        self.commission_rate = 0.0002  # 手续费率
        self.slippage_ticks = 3  # 滑点tick数
        self.margin_rate = 0.12  # 保证金率
        
        # 可选过滤
        self.enable_beta_filter = False  # 是否启用Beta过滤
        self.beta_min = 0.5  # Beta最小值
        self.beta_max = 2.5  # Beta最大值
        self.enable_adf_check = False  # 是否启用ADF检验
        self.adf_pvalue = 0.05  # ADF p值阈值
        
        # 配置支持
        self.config_file = 'configs/e2e_pipeline_config.yaml'
        
        # 数据路径
        self.data_path = 'data/data-joint/'
        self.file_pattern = 'jq_8888_{symbol}.csv'
        
    def validate(self) -> bool:
        """
        验证配置有效性 (REQ-6.1.2)
        
        Returns:
            bool: 配置是否有效
        """
        # 验证日期格式
        date_fields = ['coint_start_date', 'coint_end_date', 'signal_start_date', 
                      'backtest_start', 'backtest_end']
        for field in date_fields:
            value = getattr(self, field, None)
            if value and not self._is_valid_date(value):
                raise ValueError(f"{field} 日期格式无效: {value}")
                
        # 验证日期逻辑
        if self.coint_start_date and self.coint_end_date:
            if self.coint_end_date < self.coint_start_date:
                raise ValueError(f"协整结束日期 {self.coint_end_date} 早于开始日期 {self.coint_start_date}")
                
        # 验证数值范围
        if self.z_open <= 0:
            raise ValueError(f"z_open ({self.z_open}) 必须大于0")
        if self.z_close <= 0:
            raise ValueError(f"z_close ({self.z_close}) 必须大于0")
        if self.z_close >= self.z_open:
            raise ValueError(f"z_close ({self.z_close}) 必须小于 z_open ({self.z_open})")
            
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital ({self.initial_capital}) 必须大于0")
        if not 0 < self.position_weight <= 1:
            raise ValueError(f"position_weight ({self.position_weight}) 必须在(0, 1]范围内")
            
        return True
        
    def _is_valid_date(self, date_str: str) -> bool:
        """检查日期格式是否有效"""
        try:
            pd.to_datetime(date_str)
            return True
        except:
            return False
            
    @classmethod
    def from_yaml(cls, config_file: str = None):
        """
        从YAML文件加载配置 (REQ-6.1.3)
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            PipelineConfig实例
        """
        config = cls()
        
        if config_file is None:
            config_file = config.config_file
            
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # 更新配置
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        return config
        
    @classmethod
    def default_metals(cls):
        """
        返回默认14个金属品种的配置 (REQ-6.1.3)
        
        Returns:
            配置了14个金属品种的PipelineConfig实例
        """
        config = cls()
        config.symbols = ['AG', 'AU', 'CU', 'AL', 'ZN', 'PB', 
                         'NI', 'SN', 'RB', 'HC', 'I', 'SF', 'SM', 'SS']
        return config


# ============================================================================
# 主流程类 (REQ-6.2)
# ============================================================================

class TradingPipeline:
    """
    端到端交易流程管理类 (REQ-6.2.1)
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        初始化Pipeline
        
        Args:
            config: 配置对象，None时使用默认配置
        """
        self.config = config or PipelineConfig.default_metals()
        self.data = None
        self.pairs = None
        self.signals = None
        self.trades = None
        self.metrics = None
        self.equity_curve = None
        
        logger.info("TradingPipeline initialized with config")
        
    def run(self) -> Dict:
        """
        执行完整流程 (REQ-6.2.2)
        
        Returns:
            包含所有结果的字典
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting Pipeline execution")
            logger.info("=" * 60)
            
            # 1. 加载数据
            logger.info("Step 1: Loading data...")
            self.data = self.load_data()
            logger.info(f"Loaded data for {len(self.data.columns)} symbols, {len(self.data)} days")
            
            # 2. 协整筛选
            logger.info("Step 2: Screening cointegrated pairs...")
            self.pairs = self.screen_pairs()
            if len(self.pairs) == 0:
                logger.warning("No cointegrated pairs found")
                return self._create_empty_results("未找到协整配对")
            logger.info(f"Found {len(self.pairs)} cointegrated pairs")
            
            # 3. 信号生成
            logger.info("Step 3: Generating trading signals...")
            self.signals = self.generate_signals()
            if len(self.signals) == 0:
                logger.warning("No trading signals generated")
                return self._create_empty_results("未生成交易信号")
            logger.info(f"Generated {len(self.signals)} signals")
            
            # 4. 回测执行
            logger.info("Step 4: Running backtest...")
            self.trades = self.run_backtest()
            logger.info(f"Executed {len(self.trades) if self.trades else 0} trades")
            
            # 5. 获取结果
            logger.info("Step 5: Calculating metrics...")
            results = self.get_results()
            
            logger.info("=" * 60)
            logger.info("Pipeline execution completed successfully")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
            
    def load_data(self) -> pd.DataFrame:
        """
        加载并预处理数据 (REQ-6.2.2.1)
        
        Returns:
            价格数据DataFrame
        """
        # 如果没有指定symbols，使用默认14个金属
        if not self.config.symbols:
            self.config.symbols = ['AG', 'AU', 'CU', 'AL', 'ZN', 'PB',
                                  'NI', 'SN', 'RB', 'HC', 'I', 'SF', 'SM', 'SS']
        
        # 检查数据路径
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件路径不存在: {self.config.data_path}")
            
        # 加载数据
        try:
            data = load_all_symbols_data(
                symbols=self.config.symbols,
                start_date=self.config.data_start,
                end_date=self.config.backtest_end
            )
        except Exception as e:
            # 如果load_all_symbols_data不可用，手动加载
            data_dict = {}
            for symbol in self.config.symbols:
                file_path = os.path.join(self.config.data_path, 
                                        self.config.file_pattern.format(symbol=symbol))
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_dict[symbol] = df['close']
                    
            data = pd.DataFrame(data_dict)
            
        # 时间过滤
        if self.config.data_start:
            data = data[data.index >= pd.to_datetime(self.config.data_start)]
        if self.config.backtest_end:
            data = data[data.index <= pd.to_datetime(self.config.backtest_end)]
            
        # 缺失值处理
        data = data.ffill().bfill()
        
        return data
        
    def screen_pairs(self) -> pd.DataFrame:
        """
        协整配对筛选 (REQ-6.2.2.2)
        
        Returns:
            配对DataFrame
        """
        # 如果指定了target_pairs，直接使用
        if self.config.target_pairs:
            pairs_list = []
            for pair_str in self.config.target_pairs:
                symbol_x, symbol_y = pair_str.split('-')
                pairs_list.append({
                    'pair': pair_str,
                    'symbol_x': symbol_x,
                    'symbol_y': symbol_y,
                    'pvalue': 0.01,  # 假设值
                    'beta': 1.0  # 假设值
                })
            return pd.DataFrame(pairs_list)
            
        # 获取协整期间的数据
        coint_data = self.data.copy()
        if self.config.coint_start_date:
            coint_data = coint_data[coint_data.index >= pd.to_datetime(self.config.coint_start_date)]
        if self.config.coint_end_date:
            coint_data = coint_data[coint_data.index <= pd.to_datetime(self.config.coint_end_date)]
            
        # 初始化协整分析器
        analyzer = CointegrationAnalyzer(coint_data)
        
        # 筛选配对
        pairs_df = analyzer.screen_all_pairs(
            p_threshold=self.config.coint_pvalue
        )
        
        # screen_all_pairs已经根据p_threshold过滤了结果，不需要再次过滤
            
        return pairs_df
        
    def generate_signals(self) -> pd.DataFrame:
        """
        生成交易信号 (REQ-6.2.2.3)
        
        Returns:
            信号DataFrame
        """
        if self.pairs is None or len(self.pairs) == 0:
            return pd.DataFrame()
            
        # 初始化信号生成器
        generator = SignalGeneratorOLS(
            window_size=self.config.window_size,
            z_open=self.config.z_open,
            z_close=self.config.z_close,
            max_holding_days=self.config.max_holding_days
        )
        
        # 生成信号
        signals = generator.process_all_pairs(
            pairs_df=self.pairs,
            price_data=self.data,
            signal_start_date=pd.to_datetime(self.config.signal_start_date) if self.config.signal_start_date else None
        )
        
        # Beta过滤
        if self.config.enable_beta_filter and len(signals) > 0:
            mask = (signals['beta'].abs() >= self.config.beta_min) & \
                   (signals['beta'].abs() <= self.config.beta_max)
            signals = signals[mask]
            
        # ADF检验过滤（如果启用）
        # TODO: 实现ADF检验过滤
        
        return signals
        
    def run_backtest(self) -> List:
        """
        执行回测 (REQ-6.2.2.4)
        
        Returns:
            交易列表
        """
        if self.signals is None or len(self.signals) == 0:
            return []
            
        # TODO: 调用回测引擎
        # 这里返回模拟的交易列表
        trades = []
        
        # 简单模拟：将信号转换为交易
        current_positions = {}
        
        for _, signal in self.signals.iterrows():
            pair = signal['pair']
            
            if signal['signal'] in ['open_long', 'open_short']:
                # 开仓
                if pair not in current_positions:
                    current_positions[pair] = {
                        'entry_date': signal['date'],
                        'direction': 'long' if signal['signal'] == 'open_long' else 'short',
                        'entry_price': 100,  # 模拟价格
                    }
                    
            elif signal['signal'] == 'close':
                # 平仓
                if pair in current_positions:
                    position = current_positions.pop(pair)
                    trades.append({
                        'pair': pair,
                        'entry_date': position['entry_date'],
                        'exit_date': signal['date'],
                        'direction': position['direction'],
                        'pnl': np.random.uniform(-5000, 10000)  # 模拟PnL
                    })
                    
        return trades
        
    def get_results(self) -> Dict:
        """
        获取所有结果 (REQ-6.3.1)
        
        Returns:
            结果字典
        """
        # 计算基本指标
        if self.trades:
            total_pnl = sum(t.get('pnl', 0) for t in self.trades)
            trade_count = len(self.trades)
            win_count = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            win_rate = win_count / trade_count if trade_count > 0 else 0
        else:
            total_pnl = 0
            trade_count = 0
            win_rate = 0
            
        # 构造结果
        results = {
            'config': self.config,
            'data_info': {
                'symbols': list(self.data.columns) if self.data is not None else [],
                'date_range': (str(self.data.index[0]), str(self.data.index[-1])) if self.data is not None and len(self.data) > 0 else ('', ''),
                'total_days': len(self.data) if self.data is not None else 0
            },
            'pairs': self.pairs if self.pairs is not None else pd.DataFrame(),
            'signals': self.signals if self.signals is not None else pd.DataFrame(),
            'trades': self.trades if self.trades is not None else [],
            'metrics': {
                'total_pnl': total_pnl,
                'total_return': total_pnl / self.config.initial_capital,
                'annual_return': 0,  # TODO: 计算
                'sharpe_ratio': 0,  # TODO: 计算
                'sortino_ratio': 0,  # TODO: 计算
                'calmar_ratio': 0,  # TODO: 计算
                'max_drawdown': 0,  # TODO: 计算
                'win_rate': win_rate,
                'profit_factor': 0,  # TODO: 计算
                'trade_count': trade_count,
                'var_95': 0,  # TODO: 计算
                'cvar_95': 0,  # TODO: 计算
                'volatility': 0,  # TODO: 计算
                'downside_volatility': 0,  # TODO: 计算
                'max_consecutive_wins': 0,  # TODO: 计算
                'max_consecutive_losses': 0,  # TODO: 计算
                'avg_margin_usage': 0,  # TODO: 计算
                'peak_margin_usage': 0  # TODO: 计算
            },
            'equity_curve': pd.Series(),  # TODO: 生成净值曲线
            'summary': self._generate_summary()
        }
        
        return results
        
    def _create_empty_results(self, message: str) -> Dict:
        """创建空结果"""
        return {
            'config': self.config,
            'data_info': {},
            'pairs': pd.DataFrame(),
            'signals': pd.DataFrame(),
            'trades': [],
            'metrics': {},
            'equity_curve': pd.Series(),
            'summary': message
        }
        
    def _generate_summary(self) -> str:
        """生成文字总结"""
        if self.trades:
            return f"Pipeline execution completed. Executed {len(self.trades)} trades."
        else:
            return "Pipeline execution completed. No trades executed."
            
    def _validate_signals(self, signals: pd.DataFrame):
        """
        验证信号DataFrame字段完整性 (REQ-6.4.3)
        
        Args:
            signals: 信号DataFrame
        """
        required_fields = ['pair', 'symbol_x', 'symbol_y', 'beta', 'signal', 'z_score']
        missing_fields = [f for f in required_fields if f not in signals.columns]
        
        if missing_fields:
            raise ValueError(f"信号缺少必要字段: {missing_fields}")
            
    def save_results(self, results: Dict, output_dir: Union[str, Path], 
                     format: str = 'csv'):
        """
        保存结果到文件 (REQ-6.3.2)
        
        Args:
            results: 结果字典
            output_dir: 输出目录
            format: 保存格式 ('csv', 'json', 'excel')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            # 保存metrics
            if 'metrics' in results:
                metrics_df = pd.DataFrame([results['metrics']])
                metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
                
            # 保存trades
            if 'trades' in results and results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_csv(output_dir / 'trades.csv', index=False)
                
            # 保存signals
            if 'signals' in results and not results['signals'].empty:
                results['signals'].to_csv(output_dir / f'signals_{timestamp}.csv', index=False)
                
        elif format == 'json':
            # 转换为可JSON序列化的格式
            json_results = {
                'metrics': results.get('metrics', {}),
                'data_info': results.get('data_info', {}),
                'summary': results.get('summary', ''),
                'trade_count': len(results.get('trades', []))
            }
            
            with open(output_dir / 'results.json', 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
                
        elif format == 'excel':
            # TODO: 实现Excel保存
            pass
            
        logger.info(f"Results saved to {output_dir}")


# ============================================================================
# 辅助函数
# ============================================================================

def main():
    """主函数示例"""
    # 创建默认配置
    config = PipelineConfig.default_metals()
    config.coint_start_date = '2023-01-01'
    config.coint_end_date = '2024-08-20'
    config.signal_start_date = '2024-01-01'
    
    # 创建Pipeline并运行
    pipeline = TradingPipeline(config)
    results = pipeline.run()
    
    # 输出结果
    print("\nPipeline Results:")
    print(f"Pairs found: {len(results['pairs'])}")
    print(f"Signals generated: {len(results['signals'])}")
    print(f"Trades executed: {len(results['trades'])}")
    print(f"Total PnL: {results['metrics'].get('total_pnl', 0):,.0f}")
    

if __name__ == '__main__':
    main()