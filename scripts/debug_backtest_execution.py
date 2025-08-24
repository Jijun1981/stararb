#!/usr/bin/env python3
"""
调试BacktestEngine执行问题
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.backtest import BacktestEngine, BacktestConfig


def debug_backtest():
    print("=" * 60)
    print("调试BacktestEngine执行问题")
    print("=" * 60)
    
    # 1. 加载测试数据
    print("\n1. 加载测试数据")
    ag_data = load_symbol_data('AG')
    au_data = load_symbol_data('AU')
    
    # 准备价格数据
    prices = pd.DataFrame({
        'AG0': ag_data['close'],
        'AU0': au_data['close']
    })
    prices = prices.dropna().fillna(method='ffill')
    print(f"价格数据形状: {prices.shape}")
    print(f"价格数据索引类型: {type(prices.index[0])}")
    print(f"价格数据日期范围: {prices.index[0]} 至 {prices.index[-1]}")
    
    # 2. 创建测试信号
    print("\n2. 创建测试信号")
    
    # 手工创建一个简单的开仓信号
    test_signals = [
        {
            'date': prices.index[500],  # 使用存在的日期
            'pair': 'AG0-AU0',
            'signal': 'open_long',
            'z_score': -2.5,  # 超过阈值
            'symbol_x': 'AG0',
            'symbol_y': 'AU0',
            'beta': 0.5,
            'residual': -10.0
        },
        {
            'date': prices.index[510],  # 10天后平仓
            'pair': 'AG0-AU0', 
            'signal': 'close',
            'z_score': -0.3,  # 低于平仓阈值
            'symbol_x': 'AG0',
            'symbol_y': 'AU0',
            'beta': 0.5,
            'residual': -1.0
        }
    ]
    
    signals_df = pd.DataFrame(test_signals)
    print(f"测试信号:")
    print(signals_df[['date', 'pair', 'signal', 'z_score']])
    
    # 3. 创建BacktestEngine
    print("\n3. 创建BacktestEngine") 
    config = BacktestConfig(
        initial_capital=1000000,
        margin_rate=0.12,
        commission_rate=0.0002,
        slippage_ticks=3,
        stop_loss_pct=0.15,
        max_holding_days=30,
        z_open_threshold=2.0,
        z_close_threshold=0.5
    )
    
    engine = BacktestEngine(config)
    print(f"初始资金: {engine.available_capital:,.0f}")
    
    # 4. 检查合约规格
    print("\n4. 检查合约规格")
    contract_specs_path = project_root / 'configs' / 'contract_specs.json'
    if contract_specs_path.exists():
        engine.load_contract_specs(str(contract_specs_path))
        print(f"加载合约规格: {len(engine.contract_specs)}个品种")
        
        # 检查我们需要的品种
        for symbol in ['AG0', 'AU0']:
            if symbol in engine.contract_specs:
                spec = engine.contract_specs[symbol]
                print(f"  {symbol}: 合约乘数={spec.get('multiplier', 'N/A')}, tick_size={spec.get('tick_size', 'N/A')}")
            else:
                print(f"  {symbol}: 缺少规格")
    else:
        print("合约规格文件不存在")
    
    # 5. 手动测试process_signal
    print("\n5. 手动测试process_signal")
    
    # 获取第一个信号的日期对应的价格
    first_signal = test_signals[0]
    signal_date = first_signal['date']
    
    if signal_date in prices.index:
        current_prices = prices.loc[signal_date].to_dict()
        print(f"信号日期: {signal_date}")
        print(f"当天价格: AG0={current_prices['AG0']:.2f}, AU0={current_prices['AU0']:.2f}")
        
        print(f"\n执行process_signal...")
        print(f"信号: {first_signal}")
        
        # 调用process_signal
        result = engine.process_signal(first_signal, current_prices)
        print(f"process_signal返回: {result}")
        
        # 检查持仓
        print(f"当前持仓数量: {len(engine.positions)}")
        if engine.positions:
            for pair, position in engine.positions.items():
                print(f"  {pair}: {position}")
        
        # 检查交易记录
        print(f"交易记录数量: {len(engine.trade_records)}")
        if engine.trade_records:
            for trade in engine.trade_records:
                print(f"  {trade}")
    else:
        print(f"信号日期 {signal_date} 不在价格数据中")
    
    # 6. 运行完整回测
    print("\n6. 运行完整回测")
    
    try:
        # 重置引擎
        engine = BacktestEngine(config)
        if contract_specs_path.exists():
            engine.load_contract_specs(str(contract_specs_path))
        
        metrics = engine.run_backtest(signals_df, prices)
        
        print(f"回测结果:")
        print(f"  总交易: {metrics.get('total_trades', 0)}笔")
        print(f"  总收益: {metrics.get('total_pnl', 0):,.0f}")
        
        # 检查最终状态
        print(f"\n最终状态:")
        print(f"  持仓数: {len(engine.positions)}")
        print(f"  交易记录: {len(engine.trade_records)}")
        print(f"  权益曲线: {len(engine.equity_curve)}条")
        
        if engine.trade_records:
            print("\n交易记录详情:")
            for i, trade in enumerate(engine.trade_records):
                print(f"  {i+1}: {trade}")
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    debug_backtest()