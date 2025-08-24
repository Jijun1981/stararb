#!/usr/bin/env python3
"""
运行参数化回测v4
展示如何使用灵活配置的回测框架
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lib.backtest_v4 import BacktestEngine, BacktestConfig, create_backtest_engine
from lib.data import load_all_symbols_data


def prepare_test_signals() -> pd.DataFrame:
    """准备测试信号（模拟信号生成模块的输出）"""
    
    # 创建一些测试信号
    signals = []
    
    # 信号1：AG-NI 开多
    signals.append({
        'date': '2024-01-10',
        'pair': 'AG-NI',
        'symbol_x': 'AG',
        'symbol_y': 'NI',
        'signal': 'open_long',
        'z_score': -2.5,
        'residual': -0.15,
        'beta': 0.8234,
        'beta_initial': 0.8500,
        'days_held': 0,
        'reason': 'z_threshold',
        'phase': 'signal_period',
        'beta_window_used': '1y'
    })
    
    # 信号2：AG-NI 平仓
    signals.append({
        'date': '2024-01-15',
        'pair': 'AG-NI',
        'symbol_x': 'AG',
        'symbol_y': 'NI',
        'signal': 'close',
        'z_score': 0.3,
        'residual': 0.02,
        'beta': 0.8234,
        'beta_initial': 0.8500,
        'days_held': 5,
        'reason': 'z_threshold',
        'phase': 'signal_period',
        'beta_window_used': '1y'
    })
    
    # 信号3：CU-SN 开空
    signals.append({
        'date': '2024-01-12',
        'pair': 'CU-SN',
        'symbol_x': 'CU',
        'symbol_y': 'SN',
        'signal': 'open_short',
        'z_score': 2.3,
        'residual': 0.18,
        'beta': 1.456,
        'beta_initial': 1.500,
        'days_held': 0,
        'reason': 'z_threshold',
        'phase': 'signal_period',
        'beta_window_used': '1y'
    })
    
    # 信号4：AL-ZN 开多
    signals.append({
        'date': '2024-01-20',
        'pair': 'AL-ZN',
        'symbol_x': 'AL',
        'symbol_y': 'ZN',
        'signal': 'open_long',
        'z_score': -2.1,
        'residual': -0.12,
        'beta': 0.567,
        'beta_initial': 0.600,
        'days_held': 0,
        'reason': 'z_threshold',
        'phase': 'signal_period',
        'beta_window_used': '2y'
    })
    
    return pd.DataFrame(signals)


def test_basic_backtest():
    """测试基础回测功能"""
    print("=" * 60)
    print("测试基础回测功能（默认参数）")
    print("=" * 60)
    
    # 创建默认配置的回测引擎
    engine = BacktestEngine()
    
    # 加载合约规格
    specs_file = project_root / "configs" / "contract_specs.json"
    engine.load_contract_specs(str(specs_file))
    
    # 准备信号
    signals = prepare_test_signals()
    print(f"\n准备了 {len(signals)} 个测试信号")
    
    # 加载价格数据
    print("\n加载价格数据...")
    prices = load_all_symbols_data(start_date='2024-01-01', end_date='2024-02-01')
    
    # 运行回测
    print("\n运行回测...")
    results = engine.run_backtest(signals, prices)
    
    # 显示结果
    print("\n回测结果:")
    print(f"  总交易数: {results['total_trades']}")
    print(f"  总PnL: {results['total_pnl']:,.0f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    print(f"  胜率: {results['win_rate']:.1%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    
    return results


def test_custom_config():
    """测试自定义配置"""
    print("\n" + "=" * 60)
    print("测试自定义配置")
    print("=" * 60)
    
    # 创建自定义配置
    custom_config = {
        'initial_capital': 10000000,      # 1000万
        'margin_rate': 0.15,              # 15%保证金
        'commission_rate': 0.0001,        # 万分之1手续费
        'slippage_ticks': 2,              # 2个tick滑点
        'stop_loss_pct': 0.10,            # 10%止损
        'max_holding_days': 20,           # 20天强制平仓
        'z_open_threshold': 2.5,          # 更严格的开仓阈值
        'z_close_threshold': 0.3,         # 更严格的平仓阈值
        'max_denominator': 20,            # 允许更复杂的手数比例
        'position_weight': 0.03           # 3%仓位
    }
    
    print("\n自定义参数:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # 创建回测引擎
    engine = create_backtest_engine(custom_config)
    
    # 加载合约规格
    specs_file = project_root / "configs" / "contract_specs.json"
    engine.load_contract_specs(str(specs_file))
    
    # 准备信号
    signals = prepare_test_signals()
    
    # 加载价格数据
    prices = load_all_symbols_data(start_date='2024-01-01', end_date='2024-02-01')
    
    # 运行回测
    results = engine.run_backtest(signals, prices)
    
    # 显示结果
    print("\n回测结果（自定义配置）:")
    print(f"  总交易数: {results['total_trades']}")
    print(f"  总PnL: {results['total_pnl']:,.0f}")
    print(f"  总收益率: {results['total_return']:.2%}")
    
    return results


def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n" + "=" * 60)
    print("参数敏感性分析")
    print("=" * 60)
    
    # 准备数据
    signals = prepare_test_signals()
    prices = load_all_symbols_data(start_date='2024-01-01', end_date='2024-02-01')
    
    # 测试不同参数组合
    test_cases = [
        {
            'name': '基准配置',
            'config': {}
        },
        {
            'name': '低手续费',
            'config': {'commission_rate': 0.00005}  # 万分之0.5
        },
        {
            'name': '高手续费',
            'config': {'commission_rate': 0.0005}   # 万分之5
        },
        {
            'name': '无滑点',
            'config': {'slippage_ticks': 0}
        },
        {
            'name': '高滑点',
            'config': {'slippage_ticks': 5}
        },
        {
            'name': '严格止损',
            'config': {'stop_loss_pct': 0.05}       # 5%止损
        },
        {
            'name': '宽松止损',
            'config': {'stop_loss_pct': 0.25}       # 25%止损
        },
        {
            'name': '短持仓',
            'config': {'max_holding_days': 10}      # 10天强制平仓
        },
        {
            'name': '长持仓',
            'config': {'max_holding_days': 60}      # 60天强制平仓
        }
    ]
    
    results_summary = []
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        
        # 创建引擎
        engine = create_backtest_engine(test_case['config'])
        
        # 加载合约规格
        specs_file = project_root / "configs" / "contract_specs.json"
        engine.load_contract_specs(str(specs_file))
        
        # 运行回测
        results = engine.run_backtest(signals.copy(), prices)
        
        # 记录结果
        results_summary.append({
            '配置': test_case['name'],
            '总PnL': results['total_pnl'],
            '收益率': f"{results['total_return']:.2%}",
            '交易数': results['total_trades']
        })
        
        print(f"  PnL: {results['total_pnl']:,.0f}")
        print(f"  收益率: {results['total_return']:.2%}")
    
    # 显示汇总
    print("\n" + "=" * 60)
    print("参数敏感性汇总")
    print("=" * 60)
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    return summary_df


def test_lots_calculation():
    """测试手数计算逻辑"""
    print("\n" + "=" * 60)
    print("测试手数计算")
    print("=" * 60)
    
    # 创建引擎
    engine = BacktestEngine()
    
    # 测试不同β值
    test_betas = [0.5, 0.85, 1.0, 1.5, 2.34, 3.5]
    
    for beta in test_betas:
        result = engine.calculate_min_lots(beta)
        print(f"\nβ = {beta:.2f}")
        print(f"  Y:X = {result['lots_y']}:{result['lots_x']}")
        print(f"  实际比例: {result['actual_ratio']:.4f}")
        print(f"  误差: {result['error']*100:.2f}%")


def main():
    """主函数"""
    print("参数化回测框架v4测试")
    print("=" * 60)
    
    # 1. 基础回测测试
    test_basic_backtest()
    
    # 2. 自定义配置测试
    test_custom_config()
    
    # 3. 参数敏感性分析
    test_parameter_sensitivity()
    
    # 4. 手数计算测试
    test_lots_calculation()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    
    # 展示如何为实际使用创建配置
    print("\n实际使用示例:")
    print("-" * 40)
    print("""
# 创建你的自定义配置
my_config = {
    'initial_capital': 8000000,     # 800万初始资金
    'margin_rate': 0.12,            # 12%保证金
    'commission_rate': 0.00015,     # 万分之1.5手续费
    'slippage_ticks': 3,            # 3个tick滑点
    'stop_loss_pct': 0.15,          # 15%止损
    'max_holding_days': 30,         # 30天强制平仓
    'z_open_threshold': 2.0,        # Z>2开仓
    'z_close_threshold': 0.5,       # Z<0.5平仓
    'position_weight': 0.05,        # 5%仓位
    'save_trades': True,            # 保存交易记录
    'output_dir': 'output/my_test'  # 输出目录
}

# 创建回测引擎
engine = create_backtest_engine(my_config)

# 加载合约规格
engine.load_contract_specs('configs/contract_specs.json')

# 运行回测
results = engine.run_backtest(signals, prices)
    """)


if __name__ == "__main__":
    main()