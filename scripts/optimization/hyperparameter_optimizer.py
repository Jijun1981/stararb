#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配对交易策略超参数多目标优化框架
基于网格搜索和贝叶斯优化的综合优化系统
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer  
from lib.signal_generation import AdaptiveSignalGenerator
from lib.backtest.engine import BacktestEngine, BacktestConfig
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.trade_executor import ExecutionConfig
from lib.backtest.risk_manager import RiskConfig


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 数据设置
    start_date: str = '2024-01-01'
    end_date: str = '2025-08-24'
    warm_up_days: int = 90
    
    # 优化目标权重
    sharpe_weight: float = 0.4      # 夏普比率权重
    return_weight: float = 0.3      # 收益率权重  
    drawdown_weight: float = 0.2    # 最大回撤权重（负向）
    winrate_weight: float = 0.1     # 胜率权重
    
    # 约束条件
    min_sharpe: float = 0.5         # 最低夏普比率
    max_drawdown: float = 0.3       # 最大回撤限制
    min_trades: int = 10            # 最少交易数
    
    # 优化算法设置
    optimization_method: str = 'grid_search'  # 'grid_search', 'random', 'bayesian'
    max_evaluations: int = 1000     # 最大评估次数
    n_jobs: int = 1                 # 并行作业数


class HyperparameterOptimizer:
    """
    超参数优化器
    优化配对交易策略的所有关键参数
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        初始化优化器
        
        Args:
            config: 优化配置
        """
        self.config = config
        self.price_data = None
        self.coint_pairs = None
        self.optimization_results = []
        
        # 合约规格
        self.contract_specs = {
            'AG': {'multiplier': 15, 'tick_size': 1},
            'AU': {'multiplier': 1000, 'tick_size': 0.01}, 
            'AL': {'multiplier': 5, 'tick_size': 5},
            'CU': {'multiplier': 5, 'tick_size': 10},
            'NI': {'multiplier': 1, 'tick_size': 10},
            'PB': {'multiplier': 5, 'tick_size': 5},
            'SN': {'multiplier': 1, 'tick_size': 10},
            'ZN': {'multiplier': 5, 'tick_size': 5},
            'HC': {'multiplier': 10, 'tick_size': 1},
            'I': {'multiplier': 100, 'tick_size': 0.5},
            'RB': {'multiplier': 10, 'tick_size': 1},
            'SF': {'multiplier': 5, 'tick_size': 2},
            'SM': {'multiplier': 5, 'tick_size': 2},
            'SS': {'multiplier': 5, 'tick_size': 5}
        }
    
    def define_parameter_space(self) -> Dict[str, List]:
        """
        定义超参数搜索空间
        
        Returns:
            参数空间字典
        """
        parameter_space = {
            # === Kalman滤波器参数 (暂时使用固定值) ===
            # 'delta': [0.96],          # 固定最优值
            # 'lambda_r': [0.92],       # 固定最优值
            
            # === 信号生成参数 ===
            'z_open': [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0],         # 开仓Z阈值
            'z_close': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0],              # 平仓Z阈值
            'ols_window': [30, 45, 60, 75, 90, 120],                 # OLS预热窗口
            'calibration_freq': [3, 5, 7, 10, 15, 20],               # 校准频率
            
            # === 风险管理参数 ===
            'stop_loss_pct': [0.08, 0.10, 0.12, 0.15, 0.20],        # 止损比例
            'max_holding_days': [20, 25, 30, 35, 40, 45],            # 最大持仓天数
            
            # === 仓位管理参数 ===
            'position_weight': [0.03, 0.04, 0.05, 0.06, 0.08],      # 仓位权重
            'max_denominator': [5, 8, 10, 12, 15],                   # 最大分母
            
            # === 交易执行参数 ===
            'commission_rate': [0.0001, 0.0002, 0.0003],             # 手续费率
            'slippage_ticks': [1, 2, 3, 4, 5],                       # 滑点tick数
            
            # === 配对筛选参数 ===
            'coint_pvalue': [0.01, 0.03, 0.05, 0.08],                # 协整p值阈值
            'min_pairs': [10, 15, 20, 25, 30]                        # 最少配对数量
        }
        
        return parameter_space
    
    def calculate_optimization_score(self, backtest_results: Dict) -> float:
        """
        计算综合优化评分
        
        Args:
            backtest_results: 回测结果
            
        Returns:
            综合评分 (越高越好)
        """
        try:
            metrics = backtest_results.get('portfolio_metrics', {})
            
            # 提取关键指标
            sharpe = metrics.get('sharpe_ratio', 0)
            annual_return = metrics.get('annual_return', 0) 
            max_drawdown = abs(metrics.get('max_drawdown', 1))  # 转为正值
            win_rate = metrics.get('win_rate', 0)
            total_trades = len(backtest_results.get('trades', []))
            
            # 约束检查
            if sharpe < self.config.min_sharpe:
                return -1000  # 严重惩罚
            if max_drawdown > self.config.max_drawdown:
                return -1000
            if total_trades < self.config.min_trades:
                return -1000
                
            # 标准化指标 (0-1范围)
            normalized_sharpe = min(max(sharpe / 3.0, 0), 1)          # 夏普3.0为满分
            normalized_return = min(max(annual_return / 0.5, 0), 1)    # 50%年化收益为满分  
            normalized_drawdown = 1 - min(max_drawdown / 0.3, 1)      # 30%回撤为0分
            normalized_winrate = min(max(win_rate, 0), 1)              # 胜率已经是0-1
            
            # 加权综合评分
            score = (
                self.config.sharpe_weight * normalized_sharpe +
                self.config.return_weight * normalized_return + 
                self.config.drawdown_weight * normalized_drawdown +
                self.config.winrate_weight * normalized_winrate
            )
            
            # 交易数量奖励（鼓励适中的交易频率）
            trade_bonus = min(total_trades / 100, 0.1)  # 最多10%奖励
            score += trade_bonus
            
            return score
            
        except Exception as e:
            print(f"评分计算错误: {e}")
            return -1000
    
    def run_single_backtest(self, params: Dict) -> Tuple[Dict, float]:
        """
        执行单次回测
        
        Args:
            params: 参数组合
            
        Returns:
            (回测结果, 综合评分)
        """
        try:
            # 1. 配对筛选 (使用缓存的协整结果)
            if self.coint_pairs is None:
                print("协整配对未准备，跳过此次评估")
                return {}, -1000
                
            # 根据p值阈值筛选配对
            filtered_pairs = self.coint_pairs[
                self.coint_pairs['pvalue_5y'] <= params['coint_pvalue']
            ].head(params['min_pairs'])
            
            if len(filtered_pairs) < params['min_pairs']:
                return {}, -1000  # 配对数量不足
            
            # 2. 信号生成
            sg = AdaptiveSignalGenerator(
                z_open=params['z_open'],
                z_close=params['z_close'], 
                max_holding_days=params['max_holding_days'],
                calibration_freq=params['calibration_freq'],
                ols_window=params['ols_window'],
                warm_up_days=30  # 固定30天Kalman预热
            )
            
            # Kalman参数使用优化后的固定值
            # (delta=0.96, lambda_r=0.92已经在代码中写死)
            
            # 生成信号 (不使用initial_beta覆盖)
            signals_df = sg.process_all_pairs(
                pairs_df=filtered_pairs,
                price_data=self.price_data,
                beta_window='1y'
            )
            
            if signals_df.empty:
                return {}, -1000
                
            # 筛选交易信号
            trade_signals = signals_df[
                signals_df['signal'].isin(['open_long', 'open_short', 'close'])
            ].copy()
            trade_signals = trade_signals.rename(columns={'signal': 'trade_signal'})
            
            if len(trade_signals) < self.config.min_trades:
                return {}, -1000
            
            # 3. 回测执行
            backtest_config = BacktestConfig(
                initial_capital=5000000,
                sizing_config=PositionSizingConfig(
                    max_denominator=params['max_denominator'],
                    position_weight=params['position_weight']
                ),
                execution_config=ExecutionConfig(
                    commission_rate=params['commission_rate'],
                    slippage_ticks=params['slippage_ticks']
                ),
                risk_config=RiskConfig(
                    stop_loss_pct=params['stop_loss_pct'],
                    max_holding_days=params['max_holding_days']
                )
            )
            
            engine = BacktestEngine(backtest_config)
            
            # 转换价格数据为原始价格
            original_prices = np.exp(self.price_data)
            
            results = engine.run(
                signals=trade_signals,
                prices=original_prices,
                contract_specs=self.contract_specs
            )
            
            # 4. 计算评分
            score = self.calculate_optimization_score(results)
            
            return results, score
            
        except Exception as e:
            print(f"回测执行错误: {e}")
            return {}, -1000
    
    def prepare_data(self):
        """准备优化所需的数据"""
        print("正在准备数据...")
        
        # 加载价格数据
        print("- 加载价格数据")
        full_price_data = load_all_symbols_data()
        
        # 截取优化期间的数据
        start_date = pd.to_datetime(self.config.start_date) - pd.Timedelta(days=self.config.warm_up_days)
        end_date = pd.to_datetime(self.config.end_date)
        self.price_data = full_price_data[start_date:end_date].copy()
        
        print(f"- 数据期间: {self.price_data.index[0]} 至 {self.price_data.index[-1]}")
        print(f"- 数据点数: {len(self.price_data)}")
        
        # 协整分析 (一次性完成，后续复用)
        print("- 执行协整分析")
        analyzer = CointegrationAnalyzer(full_price_data)
        self.coint_pairs = analyzer.screen_all_pairs(
            screening_windows=['1y', '5y'],
            p_thresholds={'1y': 0.1, '5y': 0.1},  # 宽松阈值，后续再筛选
            filter_logic='AND'
        )
        
        print(f"- 协整配对数: {len(self.coint_pairs)}")
        
    def grid_search_optimization(self) -> List[Dict]:
        """
        网格搜索优化
        
        Returns:
            优化结果列表
        """
        print("开始网格搜索优化...")
        
        param_space = self.define_parameter_space()
        
        # 生成参数组合 (采样以控制计算量)
        if self.config.max_evaluations < np.prod([len(v) for v in param_space.values()]):
            print(f"参数空间过大，采样{self.config.max_evaluations}个组合")
            # 随机采样
            sampled_combinations = []
            for _ in range(self.config.max_evaluations):
                combination = {}
                for param, values in param_space.items():
                    combination[param] = np.random.choice(values)
                sampled_combinations.append(combination)
            param_combinations = sampled_combinations
        else:
            # 全网格搜索
            keys = list(param_space.keys())
            values = list(param_space.values())
            param_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        
        print(f"总计评估{len(param_combinations)}个参数组合")
        
        results = []
        best_score = -float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations):
            if i % 50 == 0:
                print(f"进度: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%)")
            
            backtest_results, score = self.run_single_backtest(params)
            
            result = {
                'params': params.copy(),
                'score': score,
                'backtest_results': backtest_results
            }
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"新的最佳评分: {score:.4f}")
                print(f"参数: {best_params}")
        
        return results
    
    def optimize(self) -> Dict:
        """
        执行优化
        
        Returns:
            优化结果
        """
        print("=" * 80)
        print("配对交易策略超参数优化")
        print("=" * 80)
        
        # 准备数据
        self.prepare_data()
        
        # 执行优化
        if self.config.optimization_method == 'grid_search':
            results = self.grid_search_optimization()
        else:
            raise NotImplementedError(f"优化方法 {self.config.optimization_method} 未实现")
        
        # 整理结果
        self.optimization_results = results
        
        # 找到最佳结果
        valid_results = [r for r in results if r['score'] > -1000]
        if not valid_results:
            print("❌ 没有找到有效的参数组合!")
            return {}
        
        best_result = max(valid_results, key=lambda x: x['score'])
        
        print("\n" + "=" * 80)
        print("优化完成!")
        print("=" * 80)
        print(f"有效评估: {len(valid_results)}/{len(results)}")
        print(f"最佳评分: {best_result['score']:.4f}")
        print("\n最佳参数组合:")
        for param, value in best_result['params'].items():
            print(f"  {param}: {value}")
        
        # 显示最佳结果的回测指标
        best_metrics = best_result['backtest_results'].get('portfolio_metrics', {})
        if best_metrics:
            print(f"\n最佳结果回测指标:")
            print(f"  夏普比率: {best_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  年化收益: {best_metrics.get('annual_return', 0):.3f}")
            print(f"  最大回撤: {best_metrics.get('max_drawdown', 0):.3f}")
            print(f"  胜率: {best_metrics.get('win_rate', 0):.3f}")
            print(f"  交易次数: {len(best_result['backtest_results'].get('trades', []))}")
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['score'],
            'best_results': best_result['backtest_results'],
            'all_results': results
        }
    
    def save_results(self, optimization_result: Dict, filename: Optional[str] = None):
        """
        保存优化结果
        
        Args:
            optimization_result: 优化结果
            filename: 保存文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hyperparameter_optimization_{timestamp}.csv"
        
        # 整理结果为DataFrame
        results_data = []
        for result in optimization_result['all_results']:
            row = result['params'].copy()
            row['score'] = result['score']
            
            # 添加关键回测指标
            metrics = result['backtest_results'].get('portfolio_metrics', {})
            row['sharpe_ratio'] = metrics.get('sharpe_ratio', 0)
            row['annual_return'] = metrics.get('annual_return', 0)
            row['max_drawdown'] = metrics.get('max_drawdown', 0)
            row['win_rate'] = metrics.get('win_rate', 0)
            row['total_trades'] = len(result['backtest_results'].get('trades', []))
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('score', ascending=False)
        results_df.to_csv(filename, index=False)
        
        print(f"优化结果已保存到: {filename}")


def main():
    """主函数"""
    # 配置优化参数
    config = OptimizationConfig(
        start_date='2024-07-01',        # 缩短测试期间加速优化
        end_date='2025-08-24',
        max_evaluations=200,            # 先用200个评估快速测试
        min_sharpe=0.3,                 # 降低约束以找到更多候选
        max_drawdown=0.4
    )
    
    # 创建优化器
    optimizer = HyperparameterOptimizer(config)
    
    # 执行优化
    results = optimizer.optimize()
    
    # 保存结果
    if results:
        optimizer.save_results(results)


if __name__ == "__main__":
    main()