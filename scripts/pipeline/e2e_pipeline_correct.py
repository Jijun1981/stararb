#!/usr/bin/env python3
"""
端到端配对交易流程 - 正确API版本
严格按照各模块的实际接口实现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 正确的导入 - 根据实际API
from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
from lib.backtest import BacktestEngine, BacktestConfig


def main():
    """完整端到端流程 - 正确API版本"""
    
    print("="*80)
    print("端到端配对交易流程 - 正确API版本")
    print("="*80)
    
    # 配置参数 - 使用正确的品种名称（不带0后缀）
    symbols = ['AG', 'AU', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 
               'HC', 'I', 'RB', 'SF', 'SM', 'SS']
    
    # =============== 1. 数据模块 ===============
    print("\n1. 数据加载")
    print("-"*40)
    
    data = {}
    for symbol in symbols:
        try:
            df = load_symbol_data(symbol)
            if df is not None and len(df) > 0:
                data[symbol] = df
                print(f"✓ {symbol}: {len(df)}条")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
    
    if len(data) < 2:
        print("数据不足")
        return
    
    # =============== 2. 协整模块 ===============
    print("\n2. 协整筛选")
    print("-"*40)
    
    # 准备数据 - CointegrationAnalyzer需要对数价格
    prices = pd.DataFrame()
    for symbol, df in data.items():
        prices[symbol] = df['close']
    prices = prices.dropna(how='all').fillna(method='ffill')
    log_prices = np.log(prices)
    
    # 创建分析器
    analyzer = CointegrationAnalyzer(log_prices)
    
    # 使用screen_all_pairs方法
    pairs_df = analyzer.screen_all_pairs(
        p_threshold=0.05,
        volatility_start_date='2024-01-01'  # 用于确定方向
    )
    
    print(f"筛选结果: {len(pairs_df)}对")
    if len(pairs_df) == 0:
        print("没有协整配对")
        return
        
    # 显示前5对
    for i in range(min(5, len(pairs_df))):
        row = pairs_df.iloc[i]
        print(f"  {i+1}. {row['pair']}: p={row['pvalue_1y']:.4f}, β={row['beta_1y']:.4f}")
    
    # =============== 3. Beta筛选 ===============
    print("\n3. Beta约束筛选")
    print("-"*40)
    
    # 应用Beta约束
    beta_min, beta_max = 0.3, 3.0
    filtered_pairs = pairs_df[
        (abs(pairs_df['beta_1y']) >= beta_min) & 
        (abs(pairs_df['beta_1y']) <= beta_max)
    ].copy()
    
    print(f"通过Beta约束: {len(filtered_pairs)}/{len(pairs_df)}")
    
    if len(filtered_pairs) == 0:
        print("没有配对通过Beta约束")
        return
    
    # =============== 4. 信号生成模块 ===============
    print("\n4. 信号生成")
    print("-"*40)
    
    # 准备价格数据 - SignalGenerator需要date列
    price_data = prices.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
    # 准备配对参数 - 根据SignalGenerator.generate_all_signals的需求
    pairs_params = {}
    for _, row in filtered_pairs.iterrows():
        pair_name = row['pair']
        x_symbol = row['symbol_x'] 
        y_symbol = row['symbol_y']
        beta = row['beta_1y']
        
        pairs_params[pair_name] = {
            'x': x_symbol,
            'y': y_symbol,
            'beta': beta,
            'beta_initial': beta  # SignalGenerator需要这个字段
        }
    
    # 创建信号生成器
    generator = SignalGenerator(
        window=60,
        z_open=2.0,  # 降低阈值以产生更多信号
        z_close=0.5,
        max_holding_days=30
    )
    
    # 生成信号
    signals = generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end='2024-07-01',
        signal_start='2024-07-01',
        hist_start='2024-01-01',
        hist_end='2025-08-20'
    )
    
    print(f"生成信号: {len(signals)}条")
    if len(signals) == 0:
        print("未生成信号")
        return
    
    # 补充缺失的字段
    if 'symbol_x' not in signals.columns or signals['symbol_x'].isna().all():
        signals['symbol_x'] = signals['pair'].apply(lambda x: x.split('-')[0])
    if 'symbol_y' not in signals.columns or signals['symbol_y'].isna().all():
        signals['symbol_y'] = signals['pair'].apply(lambda x: x.split('-')[1])
        
    # 信号统计
    print("信号分布:")
    for signal_type, count in signals['signal'].value_counts().items():
        print(f"  {signal_type}: {count}")
    
    # =============== 5. 回测模块 ===============
    print("\n5. 回测执行")
    print("-"*40)
    
    # 创建回测配置
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
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 加载合约规格
    contract_specs_path = project_root / 'configs' / 'contract_specs.json'
    if contract_specs_path.exists():
        engine.load_contract_specs(str(contract_specs_path))
    
    # BacktestEngine.run_backtest需要的格式
    # signals: DataFrame, prices: DataFrame (index为date, columns为symbols)
    
    # 确保signals有必需字段
    # BacktestEngine.process_signal需要: pair, signal, z_score, date, symbol_x, symbol_y
    if 'z_score' not in signals.columns and 'zscore' in signals.columns:
        signals['z_score'] = signals['zscore']
    elif 'z_score' not in signals.columns:
        signals['z_score'] = 0  # 临时
    
    # 调试：检查开仓信号
    open_signals = signals[signals['signal'].isin(['open_long', 'open_short'])]
    print(f"\\n调试信息:")
    print(f"  开仓信号数: {len(open_signals)}")
    print(f"  Z-score阈值设置: {config.z_open_threshold}")
    if len(open_signals) > 0:
        first_signal = open_signals.iloc[0]
        print(f"  首个开仓信号: {first_signal['pair']}, z_score={first_signal['z_score']:.2f}, signal={first_signal['signal']}")
        print(f"  包含字段: {[col for col in ['pair', 'signal', 'z_score', 'date', 'symbol_x', 'symbol_y'] if col in first_signal.index]}")
    
    # 运行回测
    try:
        # prices需要是DataFrame，index为日期，columns为品种
        backtest_prices = prices  # 这个格式是对的
        
        metrics = engine.run_backtest(signals, backtest_prices)
        
        print(f"回测结果:")
        print(f"  总交易: {metrics.get('total_trades', 0)}笔")
        print(f"  总收益: {metrics.get('total_pnl', 0):,.0f}")
        print(f"  收益率: {metrics.get('total_return', 0):.2%}")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  胜率: {metrics.get('win_rate', 0):.2%}")
        
        # 如果有交易记录，显示一些统计
        if hasattr(engine, 'trade_records') and len(engine.trade_records) > 0:
            trades_df = pd.DataFrame(engine.trade_records)
            print(f"\\n交易详情:")
            print(f"  开仓交易: {len(trades_df[trades_df['action'] == 'open'])}笔")
            print(f"  平仓交易: {len(trades_df[trades_df['action'] == 'close'])}笔")
            
            # 盈亏分析
            close_trades = trades_df[trades_df['action'] == 'close']
            if len(close_trades) > 0:
                profitable = len(close_trades[close_trades['pnl'] > 0])
                print(f"  盈利交易: {profitable}/{len(close_trades)} ({profitable/len(close_trades):.1%})")
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "="*80)
    print("端到端流程完成")
    print("="*80)


if __name__ == '__main__':
    main()