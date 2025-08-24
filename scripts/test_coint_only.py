#!/usr/bin/env python3
"""
单独测试协整模块
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer

def main():
    print("=" * 60)
    print("测试协整模块")
    print("=" * 60)
    
    # 1. 数据加载
    symbols = ['AG', 'AU', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 
               'HC', 'I', 'RB', 'SF', 'SM', 'SS']
    
    print("1. 数据加载")
    data = {}
    for symbol in symbols:
        df = load_symbol_data(symbol)
        data[symbol] = df
        print(f"  {symbol}: {len(df)}条")
    
    # 2. 准备协整分析的输入
    print("\n2. 准备协整分析输入")
    prices = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})
    prices = prices.dropna()
    log_prices = np.log(prices)
    
    print(f"价格数据形状: {prices.shape}")
    print(f"日期范围: {prices.index[0]} 至 {prices.index[-1]}")
    
    # 3. 创建协整分析器
    print("\n3. 创建协整分析器")
    analyzer = CointegrationAnalyzer(log_prices)
    
    # 4. 运行协整筛选
    print("\n4. 运行协整筛选 (5年和1年 p<0.05)")
    pairs_df = analyzer.screen_all_pairs(
        p_threshold=0.05,
        volatility_start_date='2024-01-01'
    )
    
    print(f"\n协整筛选结果: {len(pairs_df)}对")
    
    # 5. 导出结果到CSV
    if len(pairs_df) > 0:
        csv_path = "/mnt/e/Star-arb/cointegration_results.csv"
        pairs_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"结果已导出到: {csv_path}")
        
        # 显示前10对
        print(f"\n前10对协整配对:")
        print(pairs_df.head(10).to_string())
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"  beta_1y 范围: {pairs_df['beta_1y'].min():.4f} 至 {pairs_df['beta_1y'].max():.4f}")
        print(f"  pvalue_1y 范围: {pairs_df['pvalue_1y'].min():.6f} 至 {pairs_df['pvalue_1y'].max():.6f}")
        print(f"  pvalue_5y 范围: {pairs_df['pvalue_5y'].min():.6f} 至 {pairs_df['pvalue_5y'].max():.6f}")
        
    else:
        print("没有找到符合条件的协整配对")

if __name__ == '__main__':
    main()