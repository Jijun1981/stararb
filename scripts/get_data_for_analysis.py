"""获取数据用于分析止损问题"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data import DataManager
import pandas as pd
from datetime import datetime

# 初始化数据管理器
dm = DataManager()

# 要获取的品种
symbols = ['SN', 'SF', 'CU', 'AU', 'AG', 'AL', 'ZN', 'PB', 'HC', 'SM', 'SS', 'RB', 'NI', 'I']

print("开始获取数据...")

for symbol in symbols:
    print(f"获取 {symbol} 数据...")
    try:
        # 获取数据
        df = dm.load_futures_data(
            symbol=symbol,
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        # 保存到parquet文件
        output_path = f'data/{symbol}.parquet'
        df.to_parquet(output_path)
        print(f"  已保存到 {output_path}, 数据行数: {len(df)}")
        
    except Exception as e:
        print(f"  获取 {symbol} 失败: {e}")

print("\n数据获取完成！")