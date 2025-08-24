#!/usr/bin/env python
"""
验证满足协整条件的配对
条件：
1. 5年p值 < 0.05
2. 1年p值 < 0.05  
3. 1年内半衰期 < 30天
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_from_parquet
from coint import engle_granger_test, calculate_halflife

# 14个金属期货品种
SYMBOLS = ['AG0', 'AU0', 'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0', 
           'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0']

def verify_pairs():
    """验证满足条件的配对"""
    
    print("="*80)
    print("金属期货配对协整验证")
    print("="*80)
    print(f"分析品种: {', '.join(SYMBOLS)}")
    print(f"总配对数: {len(SYMBOLS) * (len(SYMBOLS) - 1) // 2}")
    print()
    
    # 加载所有品种数据
    print("正在加载数据...")
    data_dict = {}
    for symbol in SYMBOLS:
        try:
            df = load_from_parquet(symbol)
            if df is not None and not df.empty:
                # 设置日期为索引
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                data_dict[symbol] = df
                print(f"  {symbol}: {len(df)} 条记录")
        except Exception as e:
            print(f"  {symbol}: 加载失败 - {e}")
    
    print(f"\n成功加载 {len(data_dict)} 个品种数据")
    
    # 设置时间范围
    end_date = datetime.now()
    start_5y = end_date - timedelta(days=5*365)
    start_1y = end_date - timedelta(days=365)
    
    # 测试所有配对
    results = []
    total_pairs = 0
    
    print("\n开始协整检验...")
    print("-"*80)
    
    for i, sym1 in enumerate(SYMBOLS):
        for j, sym2 in enumerate(SYMBOLS):
            if i >= j:  # 避免重复测试
                continue
                
            if sym1 not in data_dict or sym2 not in data_dict:
                continue
                
            total_pairs += 1
            
            try:
                # 获取数据
                df1 = data_dict[sym1]
                df2 = data_dict[sym2]
                
                # 对齐数据
                common_dates = df1.index.intersection(df2.index)
                df1_aligned = df1.loc[common_dates]
                df2_aligned = df2.loc[common_dates]
                
                # 5年数据
                dates_5y = common_dates[common_dates >= pd.Timestamp(start_5y)]
                
                # 1年数据  
                dates_1y = common_dates[common_dates >= pd.Timestamp(start_1y)]
                
                if len(dates_5y) < 100 or len(dates_1y) < 100:
                    continue
                
                # 5年协整检验
                result_5y = engle_granger_test(
                    df1_aligned.loc[dates_5y, 'close'].values,
                    df2_aligned.loc[dates_5y, 'close'].values
                )
                
                # 1年协整检验
                result_1y = engle_granger_test(
                    df1_aligned.loc[dates_1y, 'close'].values,
                    df2_aligned.loc[dates_1y, 'close'].values
                )
                
                # 计算1年半衰期
                if result_1y['pvalue'] < 0.05:
                    # 使用1年数据的残差计算半衰期
                    x_1y = df1_aligned.loc[dates_1y, 'close'].values
                    y_1y = df2_aligned.loc[dates_1y, 'close'].values
                    residuals = y_1y - result_1y['beta'] * x_1y - result_1y['alpha']
                    
                    halflife_1y = calculate_halflife(residuals)
                else:
                    halflife_1y = np.nan
                
                # 记录结果
                result = {
                    'pair': f"{sym1}-{sym2}",
                    'sym1': sym1,
                    'sym2': sym2,
                    'p_value_5y': result_5y['pvalue'],
                    'p_value_1y': result_1y['pvalue'],
                    'halflife_1y': halflife_1y,
                    'beta_5y': result_5y['beta'],
                    'beta_1y': result_1y['beta'],
                    'pass_5y': result_5y['pvalue'] < 0.05,
                    'pass_1y': result_1y['pvalue'] < 0.05,
                    'pass_halflife': halflife_1y < 30 if not np.isnan(halflife_1y) else False,
                    'pass_all': (result_5y['pvalue'] < 0.05 and 
                                result_1y['pvalue'] < 0.05 and 
                                halflife_1y < 30)
                }
                
                results.append(result)
                
                # 打印进度
                if total_pairs % 10 == 0:
                    print(f"已测试 {total_pairs} 对...")
                    
            except Exception as e:
                print(f"  {sym1}-{sym2}: 测试失败 - {e}")
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 统计分析
    print("\n" + "="*80)
    print("统计结果")
    print("="*80)
    
    print(f"\n总测试配对数: {len(df_results)}")
    
    if len(df_results) > 0:
        print(f"5年p值<0.05: {df_results['pass_5y'].sum()} 对")
        print(f"1年p值<0.05: {df_results['pass_1y'].sum()} 对")
        print(f"1年半衰期<30天: {df_results['pass_halflife'].sum()} 对")
        print(f"全部条件满足: {df_results['pass_all'].sum()} 对")
    else:
        print("没有成功测试的配对")
    
    # 显示满足所有条件的配对
    print("\n" + "="*80)
    print("满足所有条件的配对")
    print("="*80)
    
    if len(df_results) > 0:
        qualified = df_results[df_results['pass_all']].sort_values('halflife_1y')
    else:
        qualified = pd.DataFrame()
    
    if len(qualified) > 0:
        print(f"\n{'配对':<12} {'5年p值':<10} {'1年p值':<10} {'1年半衰期':<12} {'5年Beta':<10} {'1年Beta':<10}")
        print("-"*80)
        
        for _, row in qualified.iterrows():
            print(f"{row['pair']:<12} {row['p_value_5y']:<10.6f} {row['p_value_1y']:<10.6f} "
                  f"{row['halflife_1y']:<12.2f} {row['beta_5y']:<10.4f} {row['beta_1y']:<10.4f}")
    else:
        print("没有配对满足所有条件")
    
    # 显示部分满足条件的配对
    print("\n" + "="*80)
    print("满足5年和1年p值条件的配对（不考虑半衰期）")
    print("="*80)
    
    if len(df_results) > 0:
        partial = df_results[(df_results['pass_5y']) & (df_results['pass_1y'])].sort_values('p_value_5y')
    else:
        partial = pd.DataFrame()
    
    if len(partial) > 0:
        print(f"\n{'配对':<12} {'5年p值':<10} {'1年p值':<10} {'1年半衰期':<12} {'状态':<10}")
        print("-"*80)
        
        for _, row in partial.iterrows():
            hl_str = f"{row['halflife_1y']:.2f}" if not np.isnan(row['halflife_1y']) else "N/A"
            status = "✓ 合格" if row['pass_all'] else "✗ 半衰期过长" if row['halflife_1y'] >= 30 else "✗ 无效"
            print(f"{row['pair']:<12} {row['p_value_5y']:<10.6f} {row['p_value_1y']:<10.6f} "
                  f"{hl_str:<12} {status:<10}")
    else:
        print("没有配对同时满足5年和1年p值条件")
    
    # 保存结果
    if len(df_results) > 0:
        output_file = '/mnt/e/Star-arb/data/pairs_verification_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n结果已保存至: {output_file}")
    else:
        print("\n没有结果可保存")
    
    return df_results

if __name__ == "__main__":
    results = verify_pairs()