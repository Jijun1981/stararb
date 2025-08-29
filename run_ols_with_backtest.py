#!/usr/bin/env python3
"""
基于OLS信号模块运行完整回测
集成信号生成和回测框架
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml
import json

sys.path.append('/mnt/e/Star-arb')

# 导入数据和信号模块
from lib.data import load_data, SYMBOLS
from lib.signal_generation_ols import SignalGeneratorOLS

# 导入回测模块
from lib.backtest.engine import BacktestEngine, BacktestConfig
from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig
from lib.backtest.trade_executor import TradeExecutor, ExecutionConfig
from lib.backtest.risk_manager import RiskManager, RiskConfig
from lib.backtest.performance import PerformanceAnalyzer


def load_contract_specs():
    """加载合约规格信息"""
    specs_file = 'configs/contract_specs.json'
    if os.path.exists(specs_file):
        with open(specs_file, 'r') as f:
            return json.load(f)
    else:
        # 默认合约规格
        return {
            'AG': {'multiplier': 15, 'tick_size': 1},
            'AL': {'multiplier': 5, 'tick_size': 5},
            'AU': {'multiplier': 1000, 'tick_size': 0.02},
            'CU': {'multiplier': 5, 'tick_size': 10},
            'HC': {'multiplier': 10, 'tick_size': 1},
            'I': {'multiplier': 100, 'tick_size': 0.5},
            'NI': {'multiplier': 1, 'tick_size': 10},
            'PB': {'multiplier': 5, 'tick_size': 5},
            'RB': {'multiplier': 10, 'tick_size': 1},
            'SF': {'multiplier': 5, 'tick_size': 2},
            'SM': {'multiplier': 5, 'tick_size': 2},
            'SN': {'multiplier': 1, 'tick_size': 10},
            'SS': {'multiplier': 5, 'tick_size': 5},
            'ZN': {'multiplier': 5, 'tick_size': 5}
        }


def prepare_signals_for_backtest(signals_df, price_data, contract_specs):
    """
    准备回测所需的信号数据格式
    
    Args:
        signals_df: OLS生成的信号
        price_data: 价格数据
        contract_specs: 合约规格
    
    Returns:
        适合回测的信号DataFrame
    """
    backtest_signals = []
    
    for _, signal in signals_df.iterrows():
        pair = signal['pair']
        x_symbol, y_symbol = pair.split('-')
        
        # 获取当前价格（使用原始价格，不是对数价格）
        date = pd.to_datetime(signal['date'])
        
        # 需要从对数价格转换回原始价格
        if date in price_data.index:
            log_price_x = price_data.loc[date, x_symbol]
            log_price_y = price_data.loc[date, y_symbol]
            price_x = np.exp(log_price_x)
            price_y = np.exp(log_price_y)
        else:
            continue
        
        # 构造回测信号（使用回测引擎期望的字段名）
        backtest_signal = {
            'date': date,
            'pair': pair,
            'symbol_x': x_symbol,  
            'symbol_y': y_symbol,  
            'signal': signal['signal'],  # 使用'signal'字段名，引擎期望的
            'trade_signal': signal['signal'],  # 也保留trade_signal以防万一
            'beta': signal.get('beta', 1.0),
            'z_score': signal.get('z_score', 0.0),
            'price_x': price_x,
            'price_y': price_y,
            'multiplier_x': contract_specs[x_symbol]['multiplier'],
            'multiplier_y': contract_specs[y_symbol]['multiplier'],
            'tick_x': contract_specs[x_symbol]['tick_size'],
            'tick_y': contract_specs[y_symbol]['tick_size']
        }
        
        backtest_signals.append(backtest_signal)
    
    return pd.DataFrame(backtest_signals)


def run_backtest_on_signals(signals_df, price_data, contract_specs, initial_capital=5000000):
    """
    对信号运行回测
    
    Args:
        signals_df: 准备好的回测信号
        price_data: 价格数据
        contract_specs: 合约规格
        initial_capital: 初始资金
    
    Returns:
        回测结果
    """
    # 创建各个子配置
    sizing_config = PositionSizingConfig()
    sizing_config.position_weight = 0.05  # 每个配对5%资金
    
    execution_config = ExecutionConfig()
    execution_config.commission_rate = 0.0002  # 万分之2
    execution_config.slippage_ticks = 3
    
    risk_config = RiskConfig()
    risk_config.stop_loss_pct = 0.15  # 15%止损
    risk_config.max_holding_days = 30
    risk_config.margin_rate = 0.12
    risk_config.beta_filter_enabled = True  # 启用Beta过滤
    risk_config.beta_min = 0.2  # Beta最小值（绝对值）
    risk_config.beta_max = 5.0  # Beta最大值（绝对值）
    
    # 配置回测参数
    backtest_config = BacktestConfig()
    backtest_config.initial_capital = initial_capital
    backtest_config.sizing_config = sizing_config
    backtest_config.execution_config = execution_config
    backtest_config.risk_config = risk_config
    
    # 显示配置信息
    print(f"  回测配置: 初始资金={initial_capital}, 仓位权重=5%, 止损=15%, 时间止损=30交易日, OLS窗口=60天, Z=2.5/0.5, Beta过滤=[0.2,5.0]")
    
    # 创建回测引擎
    engine = BacktestEngine(backtest_config)
    
    # 运行回测
    print("\n开始运行回测...")
    # 注意：signals_df应该是我们准备好的backtest格式信号
    results = engine.run(signals_df, price_data, contract_specs)
    
    # 调试：查看执行情况
    print(f"执行完成，持仓记录数: {len(engine.positions)}")
    print(f"执行完成，交易记录数: {len(engine.trades)}")
    
    return results


def analyze_backtest_results(results):
    """分析回测结果"""
    print("\n" + "=" * 60)
    print("回测结果分析")
    print("=" * 60)
    
    # 基础统计
    if 'trades' in results and len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        
        print(f"\n交易统计:")
        print(f"总交易数: {len(trades_df)}")
        
        # PnL统计
        total_pnl = trades_df['net_pnl'].sum()
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        
        print(f"总盈亏: {total_pnl:,.2f}")
        print(f"盈利交易: {len(winning_trades)}笔")
        print(f"亏损交易: {len(losing_trades)}笔")
        
        if len(winning_trades) > 0:
            print(f"平均盈利: {winning_trades['net_pnl'].mean():,.2f}")
            print(f"最大盈利: {winning_trades['net_pnl'].max():,.2f}")
        
        if len(losing_trades) > 0:
            print(f"平均亏损: {losing_trades['net_pnl'].mean():,.2f}")
            print(f"最大亏损: {losing_trades['net_pnl'].min():,.2f}")
        
        # 胜率
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        print(f"胜率: {win_rate:.1f}%")
        
        # 盈亏比
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum())
            print(f"盈亏比: {profit_factor:.2f}")
        
        # 持仓时间分析
        if 'holding_days' in trades_df.columns:
            print(f"\n持仓时间分析:")
            print(f"平均持仓: {trades_df['holding_days'].mean():.1f}天")
            print(f"最短持仓: {trades_df['holding_days'].min()}天")
            print(f"最长持仓: {trades_df['holding_days'].max()}天")
        
        # 平仓原因分析
        if 'close_reason' in trades_df.columns:
            print(f"\n平仓原因分析:")
            reason_counts = trades_df['close_reason'].value_counts()
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}笔 ({count/len(trades_df)*100:.1f}%)")
    
    # 绩效指标
    if 'metrics' in results:
        print(f"\n绩效指标:")
        metrics = results['metrics']
        
        if 'total_return' in metrics:
            print(f"总收益率: {metrics['total_return']*100:.2f}%")
        if 'annual_return' in metrics:
            print(f"年化收益: {metrics['annual_return']*100:.2f}%")
        if 'sharpe_ratio' in metrics:
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        if 'sortino_ratio' in metrics:
            print(f"Sortino比率: {metrics['sortino_ratio']:.2f}")
        if 'max_drawdown' in metrics:
            print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        if 'calmar_ratio' in metrics:
            print(f"Calmar比率: {metrics['calmar_ratio']:.2f}")
    
    # 净值曲线
    if 'equity_curve' in results:
        equity_curve = results['equity_curve']
        print(f"\n净值曲线:")
        print(f"起始净值: {equity_curve.iloc[0]:,.0f}")
        print(f"结束净值: {equity_curve.iloc[-1]:,.0f}")
        print(f"峰值净值: {equity_curve.max():,.0f}")
        print(f"谷值净值: {equity_curve.min():,.0f}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("OLS信号 + 回测框架集成测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 加载协整配对
    print("\n加载协整配对结果...")
    coint_files = list(Path('output/cointegration').glob('significant_pairs_20230701_20240701_*.csv'))
    if not coint_files:
        print("错误：未找到协整配对结果文件")
        return
    
    latest_file = max(coint_files, key=lambda x: x.stat().st_mtime)
    pairs_df = pd.read_csv(latest_file)
    print(f"协整配对数: {len(pairs_df)}")
    
    # 2. 加载价格数据
    print("\n加载价格数据...")
    try:
        # 对数价格用于信号生成
        log_price_data = load_data(
            symbols=SYMBOLS,
            start_date='2020-01-01',
            end_date=None,
            columns=['close'],
            log_price=True,
            fill_method='ffill'
        )
        print(f"对数价格数据形状: {log_price_data.shape}")
        
        # 原始价格用于回测
        raw_price_data = load_data(
            symbols=SYMBOLS,
            start_date='2020-01-01',
            end_date=None,
            columns=['close'],
            log_price=False,
            fill_method='ffill'
        )
        print(f"原始价格数据形状: {raw_price_data.shape}")
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 3. 生成OLS信号
    print("\n生成OLS信号...")
    generator = SignalGeneratorOLS(
        window_size=60,          # OLS滚动窗口60天
        z_open=2.5,              # 提高开仓阈值从2.0到2.5
        z_close=0.5,
        max_holding_days=30,
        enable_adf_check=False
    )
    
    signal_start_date = pd.to_datetime('2024-07-01')
    
    try:
        signals_df = generator.process_all_pairs(
            pairs_df=pairs_df,
            price_data=log_price_data,
            beta_window='1y',
            signal_start_date=signal_start_date
        )
        print(f"生成信号记录数: {len(signals_df)}")
        
    except Exception as e:
        print(f"信号生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if signals_df is None or len(signals_df) == 0:
        print("未生成有效信号")
        return
    
    # 4. 准备回测数据
    print("\n准备回测数据...")
    contract_specs = load_contract_specs()
    
    # 转换信号格式
    backtest_signals = prepare_signals_for_backtest(
        signals_df, 
        log_price_data,
        contract_specs
    )
    print(f"回测信号数: {len(backtest_signals)}")
    
    # 调试：查看信号内容
    if len(backtest_signals) > 0:
        print("\n信号样本:")
        print(backtest_signals.head())
        
        # 统计信号类型
        signal_counts = backtest_signals['signal'].value_counts()
        print("\n信号类型统计:")
        print(signal_counts)
    
    # 5. 运行回测
    print("\n" + "=" * 60)
    print("运行回测")
    print("=" * 60)
    
    try:
        # 注意：传入的应该是格式化后的backtest_signals
        results = run_backtest_on_signals(
            backtest_signals,  # 这是转换后的信号
            raw_price_data,
            contract_specs
        )
        
        # 6. 分析结果
        analyze_backtest_results(results)
        
        # 7. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存交易记录
        if 'trades' in results and len(results['trades']) > 0:
            trades_df = pd.DataFrame(results['trades'])
            trades_file = f'ols_backtest_trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"\n交易记录已保存: {trades_file}")
        
        # 保存净值曲线
        if 'equity_curve' in results:
            equity_file = f'ols_backtest_equity_{timestamp}.csv'
            results['equity_curve'].to_csv(equity_file)
            print(f"净值曲线已保存: {equity_file}")
        
        # 保存绩效指标
        if 'metrics' in results:
            metrics_file = f'ols_backtest_metrics_{timestamp}.csv'
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(metrics_file, index=False)
            print(f"绩效指标已保存: {metrics_file}")
        
    except Exception as e:
        print(f"回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("OLS信号回测完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()