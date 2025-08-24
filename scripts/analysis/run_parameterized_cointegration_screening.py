#!/usr/bin/env python3
"""
参数化协整配对筛选脚本

使用新的参数化协整模块生成符合指定条件的协整配对：
- 最近1年p值 < 0.05
- 最近3年p值 < 0.05
- AND逻辑筛选

输出文件：output/cointegration/results/filtered_pairs_{timestamp}.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

def ensure_output_directory():
    """确保输出目录存在"""
    output_dir = project_root / "output" / "cointegration" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    """主函数"""
    print("=" * 60)
    print("参数化协整配对筛选")
    print("=" * 60)
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        # 加载所有14个品种的对数价格数据，用于协整分析
        symbols = ['AG', 'AL', 'AU', 'CU', 'HC', 'I', 'NI', 'PB', 'RB', 'SF', 'SM', 'SN', 'SS', 'ZN']
        log_prices = load_data(symbols, log_price=True)
        print(f"   数据加载成功: {log_prices.shape[1]}个品种, {log_prices.shape[0]}个交易日")
        print(f"   数据时间范围: {log_prices.index[0].strftime('%Y-%m-%d')} 至 {log_prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"   品种列表: {', '.join(log_prices.columns)}")
    except Exception as e:
        print(f"   数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 初始化协整分析器
    print("\n2. 初始化协整分析器...")
    try:
        analyzer = CointegrationAnalyzer(log_prices)
        print(f"   分析器初始化成功: {analyzer.n_symbols}个品种")
    except Exception as e:
        print(f"   分析器初始化失败: {e}")
        return
    
    # 3. 运行参数化筛选
    print("\n3. 运行协整配对筛选...")
    print("   筛选条件:")
    print("   - 1年窗口 p值 < 0.05")
    print("   - 3年窗口 p值 < 0.05") 
    print("   - AND逻辑筛选")
    print("   - 按1年p值升序排序")
    
    try:
        # 使用新的参数化接口
        results = analyzer.screen_all_pairs(
            screening_windows=['1y', '3y'],           # 筛选用的时间窗口
            p_thresholds={'1y': 0.05, '3y': 0.05},   # 各窗口的p值阈值
            filter_logic='AND',                       # AND逻辑筛选
            sort_by='pvalue_1y',                      # 按1年p值排序
            ascending=True                            # 升序排序
        )
        
        print(f"   筛选完成: 找到 {len(results)} 个符合条件的配对")
        
        if len(results) == 0:
            print("   ⚠️ 没有找到符合条件的配对")
            return
            
    except Exception as e:
        print(f"   筛选失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 显示结果摘要
    print("\n4. 结果摘要:")
    print("   前5个最佳配对:")
    display_cols = ['pair', 'symbol_x', 'symbol_y', 'pvalue_1y', 'pvalue_3y', 'beta_1y', 'beta_3y']
    available_cols = [col for col in display_cols if col in results.columns]
    
    for i, (idx, row) in enumerate(results.head().iterrows()):
        print(f"   {i+1}. {row['pair']}: p_1y={row['pvalue_1y']:.4f}, p_3y={row['pvalue_3y']:.4f}, β_1y={row['beta_1y']:.4f}")
    
    # 5. 保存结果到CSV
    print("\n5. 保存结果...")
    try:
        output_dir = ensure_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"filtered_pairs_{timestamp}.csv"
        
        # 保存完整结果
        results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"   结果已保存到: {output_file}")
        
        # 显示保存的文件信息
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"   文件大小: {file_size:.1f} KB")
        print(f"   包含列数: {len(results.columns)}")
        
        # 显示所有列名
        print("   \n   包含的列:")
        for i, col in enumerate(results.columns):
            if i % 4 == 0:
                print("   ", end="")
            print(f"{col:<20}", end="")
            if (i + 1) % 4 == 0:
                print()
        if len(results.columns) % 4 != 0:
            print()
            
    except Exception as e:
        print(f"   保存失败: {e}")
        return
    
    # 6. 统计信息
    print(f"\n6. 统计信息:")
    print(f"   总配对数: {len(results)}")
    if len(results) > 0:
        print(f"   1年p值范围: {results['pvalue_1y'].min():.6f} - {results['pvalue_1y'].max():.6f}")
        print(f"   3年p值范围: {results['pvalue_3y'].min():.6f} - {results['pvalue_3y'].max():.6f}")
        if 'beta_1y' in results.columns:
            print(f"   1年β系数范围: {results['beta_1y'].min():.4f} - {results['beta_1y'].max():.4f}")
        if 'volatility_x' in results.columns:
            print(f"   X品种波动率范围: {results['volatility_x'].min():.4f} - {results['volatility_x'].max():.4f}")
    
    print(f"\n✅ 协整配对筛选完成！")
    print(f"📁 结果文件: {output_file}")

if __name__ == "__main__":
    main()