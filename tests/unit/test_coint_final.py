#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协整模块最终验证测试
验证REQ-2.4.3和REQ-2.4.5的实现
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

def main():
    print("协整模块需求验证")
    print("=" * 50)
    
    # 加载数据
    symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 
               'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']
    data = load_data(symbols, columns=['close'], log_price=True)
    
    # 创建分析器
    analyzer = CointegrationAnalyzer(data)
    
    # 筛选配对（使用0.05阈值）
    results = analyzer.screen_all_pairs(p_threshold=0.05)
    
    print(f"\n✅ 筛选出 {len(results)} 个配对")
    
    # 验证REQ-2.4.3: 双重筛选条件
    if len(results) > 0:
        all_meet_5y = (results['pvalue_5y'] < 0.05).all()
        all_meet_1y = (results['pvalue_1y'] < 0.05).all()
        
        if all_meet_5y and all_meet_1y:
            print("✅ REQ-2.4.3: 所有配对满足双重条件（5年且1年 < 0.05）")
        else:
            print("❌ REQ-2.4.3: 存在不满足条件的配对")
            return
    
    # 验证REQ-2.4.5: 按1年p值排序
    if len(results) > 0:
        pvalues = results['pvalue_1y'].values
        is_sorted = all(pvalues[i] <= pvalues[i+1] for i in range(len(pvalues)-1))
        
        if is_sorted:
            print("✅ REQ-2.4.5: 结果按1年p值升序排序")
            print(f"\n前5个配对:")
            for i in range(min(5, len(results))):
                row = results.iloc[i]
                print(f"  {i+1}. {row['pair']}: 1年p={row['pvalue_1y']:.6f}")
        else:
            print("❌ REQ-2.4.5: 未按1年p值排序")
    
    # 验证数据完整性
    required_cols = ['pvalue_1y', 'pvalue_2y', 'pvalue_3y', 'pvalue_4y', 'pvalue_5y',
                     'beta_1y', 'beta_2y', 'beta_3y', 'beta_4y', 'beta_5y',
                     'halflife_1y', 'halflife_2y', 'halflife_3y', 'halflife_4y', 'halflife_5y']
    
    missing = [col for col in required_cols if col not in results.columns]
    if not missing:
        print("✅ 所有5个时间窗口的数据完整")
    else:
        print(f"❌ 缺少: {missing}")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()