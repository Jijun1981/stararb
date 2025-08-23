#!/usr/bin/env python3
"""
聚宽Research环境脚本 - 直接运行，无需认证
获取8888主力连续合约数据并保存为CSV
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================
# 14个金属期货品种的8888主力连续合约
# ============================================
symbols_map = {
    'AG': 'AG8888.XSGE',  # 白银
    'AL': 'AL8888.XSGE',  # 铝
    'AU': 'AU8888.XSGE',  # 黄金
    'CU': 'CU8888.XSGE',  # 铜
    'HC': 'HC8888.XSGE',  # 热卷
    'I': 'I8888.XDCE',    # 铁矿
    'NI': 'NI8888.XSGE',  # 镍
    'PB': 'PB8888.XSGE',  # 铅
    'RB': 'RB8888.XSGE',  # 螺纹钢
    'SF': 'SF8888.XZCE',  # 硅铁
    'SM': 'SM8888.XZCE',  # 锰硅
    'SN': 'SN8888.XSGE',  # 锡
    'SS': 'SS8888.XSGE',  # 不锈钢
    'ZN': 'ZN8888.XSGE',  # 锌
}

# 数据获取时间范围
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# ============================================
# 1. 获取所有品种的8888数据
# ============================================
print("开始获取聚宽8888主力连续合约数据...")
print("="*60)

all_data = {}
for symbol, jq_code in symbols_map.items():
    print(f"\n获取 {symbol} ({jq_code})...")
    
    try:
        # 获取后复权数据
        df = get_price(jq_code,
                      start_date=START_DATE,
                      end_date=END_DATE,
                      frequency='daily',
                      fields=['open', 'high', 'low', 'close', 'volume', 'open_interest'],
                      skip_paused=False,
                      fq='post')  # 后复权
        
        # 添加symbol列
        df['symbol'] = symbol
        
        # 保存到字典
        all_data[symbol] = df
        
        print(f"  ✓ 成功: {len(df)} 条数据")
        print(f"  日期范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        
        # 检查是否有异常跳空
        returns = df['close'].pct_change()
        large_jumps = returns[abs(returns) > 0.10]
        if len(large_jumps) > 0:
            print(f"  ⚠️ 发现超过10%的价格变动: {len(large_jumps)}次")
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")

# ============================================
# 2. 重点检查I8888的换月日数据
# ============================================
print("\n" + "="*60)
print("重点检查I8888在AkShare换月日的表现")
print("="*60)

if 'I' in all_data:
    df_i = all_data['I']
    
    # 检查关键日期
    check_dates = [
        ('2023-08-04', '2023-08-07', '2023-08-08'),  # AkShare显示-12%跳空
        ('2024-03-27', '2024-03-28', '2024-03-29'),  # AkShare显示-5.9%跳空
    ]
    
    for date1, date2, date3 in check_dates:
        print(f"\n检查 {date2} 换月日:")
        print("-"*40)
        
        # 提取这几天的数据
        dates_to_check = pd.to_datetime([date1, date2, date3])
        subset = df_i.loc[df_i.index.isin(dates_to_check)]
        
        if len(subset) > 0:
            # 计算收益率
            subset_copy = subset.copy()
            subset_copy['return'] = subset_copy['close'].pct_change() * 100
            subset_copy['oi_change'] = subset_copy['open_interest'].pct_change() * 100
            
            print("聚宽后复权数据:")
            for idx, row in subset_copy.iterrows():
                date_str = idx.strftime('%Y-%m-%d')
                print(f"  {date_str}: close={row['close']:.1f}, "
                      f"return={row['return']:.2f}%, "
                      f"oi={row['open_interest']:.0f}, "
                      f"oi_chg={row['oi_change']:.2f}%")
            
            # 对比AkShare数据
            if date2 == '2023-08-07':
                print("\nAkShare原始数据（有换月跳空）:")
                print("  2023-08-04: close=817.5")
                print("  2023-08-07: close=719.5 (跳空-12.0%)")
            elif date2 == '2024-03-28':
                print("\nAkShare原始数据（有换月跳空）:")
                print("  2024-03-27: close=805.5")
                print("  2024-03-28: close=758.0 (跳空-5.9%)")

# ============================================
# 3. 合并所有数据并保存
# ============================================
print("\n" + "="*60)
print("保存数据到CSV文件")
print("="*60)

# 合并所有品种数据
all_dfs = []
for symbol, df in all_data.items():
    df_copy = df.copy()
    df_copy['symbol'] = symbol
    df_copy.reset_index(inplace=True)
    df_copy.rename(columns={'index': 'date'}, inplace=True)
    all_dfs.append(df_copy)

# 合并成一个大DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# 保存合并数据
combined_file = 'jq_8888_all_symbols.csv'
combined_df.to_csv(combined_file, index=False)
print(f"\n✓ 所有品种数据已保存到: {combined_file}")
print(f"  总记录数: {len(combined_df)}")
print(f"  品种数: {combined_df['symbol'].nunique()}")

# ============================================
# 4. 单独保存每个品种
# ============================================
print("\n单独保存每个品种的CSV文件:")
for symbol, df in all_data.items():
    filename = f'jq_8888_{symbol}.csv'
    df.to_csv(filename)
    print(f"  ✓ {symbol}: {filename} ({len(df)} 条记录)")

# ============================================
# 5. 生成数据质量报告
# ============================================
print("\n" + "="*60)
print("数据质量报告")
print("="*60)

quality_report = []
for symbol, df in all_data.items():
    returns = df['close'].pct_change()
    
    report = {
        'symbol': symbol,
        'records': len(df),
        'start_date': df.index[0].strftime('%Y-%m-%d'),
        'end_date': df.index[-1].strftime('%Y-%m-%d'),
        'max_return': returns.max() * 100,
        'min_return': returns.min() * 100,
        'jumps_over_10pct': len(returns[abs(returns) > 0.10]),
        'jumps_over_5pct': len(returns[abs(returns) > 0.05]),
        'missing_days': df['close'].isna().sum()
    }
    quality_report.append(report)

quality_df = pd.DataFrame(quality_report)
quality_df.to_csv('jq_8888_quality_report.csv', index=False)

print("\n数据质量统计:")
print(quality_df.to_string())

print("\n" + "="*60)
print("数据导出完成！")
print("="*60)
print("\n生成的文件:")
print("1. jq_8888_all_symbols.csv - 所有品种合并数据")
print("2. jq_8888_{symbol}.csv - 各品种单独文件")
print("3. jq_8888_quality_report.csv - 数据质量报告")
print("\n请下载这些CSV文件进行后续分析")