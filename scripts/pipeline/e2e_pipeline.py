#!/usr/bin/env python3
"""
端到端配对交易流程 - 完整业务流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
from lib.backtest import BacktestEngine, BacktestConfig




def main():
    print("=" * 80)
    print("端到端配对交易流程")
    print("=" * 80)
    
    # 参数配置
    symbols = ['AG', 'AU', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 
               'HC', 'I', 'RB', 'SF', 'SM', 'SS']
    
    # 1. 数据加载
    print("\n1. 数据加载")
    data = {}
    for symbol in symbols:
        df = load_symbol_data(symbol)
        data[symbol] = df
        print(f"  {symbol}: {len(df)}条")
    
    # 2. 协整分析
    print("\n2. 协整分析")
    prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
    prices = prices.dropna()
    log_prices = np.log(prices)
    
    analyzer = CointegrationAnalyzer(log_prices)
    pairs_df = analyzer.screen_all_pairs(
        p_threshold=0.05,
        volatility_start_date='2024-01-01'
    )
    print(f"  筛选结果: {len(pairs_df)}对")
    
    # Beta约束筛选
    filtered_pairs = pairs_df[
        (abs(pairs_df['beta_1y']) >= 0.3) & 
        (abs(pairs_df['beta_1y']) <= 3.0)
    ].copy()
    print(f"  Beta约束后: {len(filtered_pairs)}对")
    
    # 3. 信号生成
    print("\n3. 信号生成")
    price_data = prices.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
    pairs_params = {}
    for _, row in filtered_pairs.iterrows():
        pair_name = row['pair']
        pairs_params[pair_name] = {
            'x': row['symbol_x'],
            'y': row['symbol_y'],
            'beta': row['beta_1y'],
            'beta_initial': row['beta_1y']
        }
    
    generator = SignalGenerator(
        window=60,
        z_open=2.0,
        z_close=0.5,
        max_holding_days=30
    )
    
    signals = generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end='2024-07-01',
        signal_start='2024-07-01',
        hist_start='2024-01-01',
        hist_end='2025-08-20'
    )
    
    # 修复symbol_x和symbol_y字段
    if len(signals) > 0:
        signals['symbol_x'] = signals['pair'].apply(lambda x: x.split('-')[0])
        signals['symbol_y'] = signals['pair'].apply(lambda x: x.split('-')[1])
    
    print(f"  生成信号: {len(signals)}条")
    
    open_signals = signals[signals['signal'].isin(['open_long', 'open_short'])]
    print(f"  开仓信号: {len(open_signals)}条")
    
    # 4. 回测执行
    print("\n4. 回测执行")
    config = BacktestConfig(
        initial_capital=5000000,
        margin_rate=0.12,
        commission_rate=0.0002,
        slippage_ticks=3,
        stop_loss_pct=0.15,
        max_holding_days=30,
        z_open_threshold=2.0,
        z_close_threshold=0.5
    )
    
    engine = BacktestEngine(config)
    
    contract_specs_path = project_root / 'configs' / 'contract_specs.json'
    if contract_specs_path.exists():
        engine.load_contract_specs(str(contract_specs_path))
    
    metrics = engine.run_backtest(signals, prices)
    
    print(f"\n回测结果:")
    print(f"  总交易: {metrics.get('total_trades', 0)}笔")
    print(f"  总收益: {metrics.get('total_pnl', 0):,.0f}")
    print(f"  收益率: {metrics.get('total_return', 0):.2%}")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  胜率: {metrics.get('win_rate', 0):.2%}")
    
    # 导出交易记录到CSV
    if hasattr(engine, 'trade_records') and len(engine.trade_records) > 0:
        from datetime import datetime
        
        trades_df = pd.DataFrame(engine.trade_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"/mnt/e/Star-arb/backtest_trades_{timestamp}.csv"
        trades_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n交易记录已导出: {csv_path}")
        print(f"共{len(trades_df)}条交易记录")
        
        # 显示前几条交易
        print(f"\n前5条交易记录:")
        print(trades_df.head().to_string())
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()