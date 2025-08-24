#!/usr/bin/env python3
"""
单独测试信号生成模块
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.signal_generation import SignalGenerator

# 时间配置
TIME_CONFIG = {
    'data_start': '2020-01-01',  # 数据起始（用于协整分析）
    'beta_training_start': '2023-01-01',  # Beta训练开始
    'beta_training_end': '2023-12-31',     # Beta训练结束
    'convergence_start': '2024-01-01',     # Kalman收敛期开始
    'convergence_end': '2024-06-30',       # Kalman收敛期结束
    'signal_start': '2024-07-01',          # 信号生成开始
    'backtest_end': '2025-08-20'           # 回测结束
}

# 信号生成参数
SIGNAL_CONFIG = {
    'z_open': 2.2,
    'z_close': 0.3,
    'window': 60,
    'max_holding_days': 30,
    'convergence_threshold': 0.01  # 1%收敛阈值
}

def main():
    print("=" * 60)
    print("测试信号生成模块")
    print("=" * 60)
    
    # 1. 加载协整结果
    print("1. 加载协整结果")
    coint_df = pd.read_csv("/mnt/e/Star-arb/cointegration_results.csv")
    print(f"  协整配对数: {len(coint_df)}")
    
    # 先测试前5对
    test_pairs = coint_df.head(5)
    print(f"  测试配对: {len(test_pairs)}对")
    for i, row in test_pairs.iterrows():
        print(f"    {i+1}. {row['pair']}: β={row['beta_1y']:.4f}")
    
    # 2. 加载价格数据
    print("\n2. 加载价格数据")
    symbols = ['AG', 'AU', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 
               'HC', 'I', 'RB', 'SF', 'SM', 'SS']
    
    data = {}
    for symbol in symbols:
        df = load_symbol_data(symbol)
        data[symbol] = df
        print(f"  {symbol}: {len(df)}条")
    
    # 3. 准备价格数据
    print("\n3. 准备价格数据")
    prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
    prices = prices.dropna()
    
    # 转换为SignalGenerator需要的格式 (带date列)
    price_data = prices.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
    print(f"价格数据形状: {price_data.shape}")
    print(f"日期范围: {price_data['date'].min()} 至 {price_data['date'].max()}")
    print(f"列名: {list(price_data.columns)}")
    
    # 4. 准备配对参数
    print("\n4. 准备配对参数")
    pairs_params = {}
    for _, row in test_pairs.iterrows():
        pair_name = row['pair']
        pairs_params[pair_name] = {
            'x': row['symbol_x'],
            'y': row['symbol_y'],
            'beta': row['beta_1y'],
            'beta_initial': row['beta_1y']
        }
    
    print(f"配对参数:")
    for pair_name, params in pairs_params.items():
        print(f"  {pair_name}: x={params['x']}, y={params['y']}, β={params['beta']:.4f}")
    
    # 5. 创建信号生成器
    print("\n5. 创建信号生成器")
    generator = SignalGenerator(
        window=SIGNAL_CONFIG['window'],
        z_open=SIGNAL_CONFIG['z_open'], 
        z_close=SIGNAL_CONFIG['z_close'],
        max_holding_days=SIGNAL_CONFIG['max_holding_days']
    )
    
    print(f"信号参数:")
    print(f"  滚动窗口: {SIGNAL_CONFIG['window']}")
    print(f"  开仓阈值: {SIGNAL_CONFIG['z_open']}")
    print(f"  平仓阈值: {SIGNAL_CONFIG['z_close']}")
    print(f"  最大持仓: {SIGNAL_CONFIG['max_holding_days']}天")
    
    # 6. 生成信号
    print("\n6. 生成信号")
    print(f"时间配置:")
    print(f"  收敛期: {TIME_CONFIG['convergence_start']} 至 {TIME_CONFIG['convergence_end']}")
    print(f"  信号期: {TIME_CONFIG['signal_start']} 至 {TIME_CONFIG['backtest_end']}")
    
    try:
        signals = generator.generate_all_signals(
            pairs_params=pairs_params,
            price_data=price_data,
            convergence_end=TIME_CONFIG['convergence_end'],
            signal_start=TIME_CONFIG['signal_start'],
            hist_start=TIME_CONFIG['convergence_start'],
            hist_end=TIME_CONFIG['backtest_end']
        )
        
        # 修复symbol_x和symbol_y字段
        if len(signals) > 0:
            signals['symbol_x'] = signals['pair'].apply(lambda x: x.split('-')[0] if '-' in str(x) else '')
            signals['symbol_y'] = signals['pair'].apply(lambda x: x.split('-')[1] if '-' in str(x) else '')
        
        print(f"\n信号生成结果:")
        print(f"  总信号数: {len(signals)}")
        
        if len(signals) > 0:
            # 统计信号类型
            signal_counts = signals['signal'].value_counts()
            print(f"  信号分布:")
            for signal_type, count in signal_counts.items():
                print(f"    {signal_type}: {count}")
            
            # 检查开仓信号
            open_signals = signals[signals['signal'].isin(['open_long', 'open_short'])]
            print(f"  开仓信号: {len(open_signals)}条")
            
            # 导出信号到CSV
            csv_path = "/mnt/e/Star-arb/signals_test.csv"
            signals.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  信号已导出: {csv_path}")
            
            # 显示前几条信号
            print(f"\n前5条信号:")
            print(signals.head().to_string())
            
            # 显示开仓信号样本
            if len(open_signals) > 0:
                print(f"\n前3条开仓信号:")
                print(open_signals.head(3)[['date', 'pair', 'signal', 'z_score', 'symbol_x', 'symbol_y']].to_string())
        
    except Exception as e:
        print(f"信号生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()