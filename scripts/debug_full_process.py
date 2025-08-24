#!/usr/bin/env python3
"""
调试完整的信号生成流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.signal_generation import SignalGenerator

def main():
    print("=" * 60)
    print("调试完整信号生成流程")
    print("=" * 60)
    
    # 加载数据
    symbols = ['AU', 'ZN']
    data = {}
    for symbol in symbols:
        data[symbol] = load_symbol_data(symbol)
        print(f"{symbol}: {len(data[symbol])}条")
    
    # 准备价格数据
    prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
    prices = prices.dropna()
    price_data = prices.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
    # 重命名列为x和y（AU作为x，ZN作为y）
    price_data = price_data.rename(columns={'AU': 'x', 'ZN': 'y'})
    
    # 用2023年数据估计beta
    data_2023 = price_data[(price_data['date'] >= '2023-01-01') & (price_data['date'] <= '2023-12-31')]
    print(f"2023年数据: {len(data_2023)}条")
    
    if len(data_2023) > 60:  # 确保有足够数据
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        import numpy as np
        
        # OLS估计: log(ZN) = alpha + beta * log(AU) + epsilon
        x_2023 = np.log(data_2023['x'])  # AU
        y_2023 = np.log(data_2023['y'])  # ZN
        X = add_constant(x_2023)
        model = OLS(y_2023, X).fit()
        beta_2023 = model.params[1]
        
        print(f"协整分析beta: -0.306369")
        print(f"2023年OLS beta: {beta_2023:.6f}")
    else:
        beta_2023 = -0.306369
        print(f"2023年数据不足，使用协整分析beta: {beta_2023}")
    
    # 配对参数
    pair_params = {
        'x': 'AU',
        'y': 'ZN', 
        'beta': beta_2023,
        'beta_initial': beta_2023
    }
    
    print(f"配对: AU-ZN, beta={pair_params['beta']}")
    print(f"数据日期范围: {price_data['date'].min()} 至 {price_data['date'].max()}")
    
    # 创建信号生成器
    generator = SignalGenerator(window=60, z_open=2.2, z_close=0.3, max_holding_days=30)
    
    # 调用process_pair_signals方法，加上调试参数
    print("\n开始处理配对信号...")
    
    signals = generator.process_pair_signals(
        pair_data=price_data,
        initial_beta=pair_params['beta_initial'],
        convergence_end='2024-06-30',
        signal_start='2024-07-01',
        hist_start='2024-01-01',
        hist_end='2025-08-20',
        pair_info={'pair': 'AU-ZN', 'x': 'AU', 'y': 'ZN'}
    )
    
    print(f"\n处理结果:")
    print(f"总信号数: {len(signals)}")
    
    if len(signals) > 0:
        # 检查各阶段信号
        phase_counts = signals['phase'].value_counts()
        print(f"阶段分布:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count}")
        
        signal_counts = signals['signal'].value_counts()
        print(f"信号分布:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # 查看信号生成期的数据
        signal_period = signals[signals['phase'] == 'signal_period']
        print(f"\n信号生成期数据: {len(signal_period)}条")
        
        if len(signal_period) > 0:
            # 查看Z-score分布
            z_scores = signal_period['z_score'].dropna()
            if len(z_scores) > 0:
                print(f"Z-score统计:")
                print(f"  均值: {z_scores.mean():.4f}")
                print(f"  标准差: {z_scores.std():.4f}")
                print(f"  最小值: {z_scores.min():.4f}")
                print(f"  最大值: {z_scores.max():.4f}")
                print(f"  |Z|>=2.2的个数: {len(z_scores[abs(z_scores) >= 2.2])}")
                
                # 显示极值点
                extreme_points = signal_period[abs(signal_period['z_score']) >= 2.0]
                if len(extreme_points) > 0:
                    print(f"\n|Z|>=2.0的时点:")
                    for _, row in extreme_points.iterrows():
                        print(f"  {row['date'].strftime('%Y-%m-%d')}: Z={row['z_score']:.4f}, signal={row['signal']}")
        
        # 导出调试结果
        debug_csv = "/mnt/e/Star-arb/debug_AU_ZN_signals.csv"
        signals.to_csv(debug_csv, index=False)
        print(f"\n调试数据已导出: {debug_csv}")

if __name__ == '__main__':
    main()