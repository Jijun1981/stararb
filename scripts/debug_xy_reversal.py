#!/usr/bin/env python3
"""
检查X和Y是否搞反了
"""

import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("检查X和Y是否搞反了")
print("=" * 80)

# 读取协整结果
coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
coint_results = pd.read_csv(coint_file)

print("检查几个有问题的配对:")

# 1. CU0-SM0配对
print("\n1. CU0-SM0配对:")
cu_sm = coint_results[coint_results['pair'] == 'CU0-SM0']
if not cu_sm.empty:
    row = cu_sm.iloc[0]
    print(f"  协整结果: X={row['symbol_x']}, Y={row['symbol_y']}")
    print(f"  Beta: {row.get('beta_4y', 'N/A')}")
    
    print(f"  当前理解: 价差 = {row['symbol_y']} - β×{row['symbol_x']}")
    print(f"  即: 价差 = SM0 - {row.get('beta_4y', 'N/A')}×CU0")
    print(f"  这不合理！SM0(6334) - 0.56×CU0(73770) = 6334 - 41312 = -35000")
    
    print(f"  如果X和Y反了:")
    print(f"  价差 = CU0 - β×SM0")  
    print(f"  则β应该 = CU0/SM0 ≈ 73770/6334 ≈ 11.6")
    print(f"  实际β = 0.56，可能是 11.6的倒数？ 1/11.6 ≈ 0.086")

# 2. AL0-CU0配对
print("\n2. AL0-CU0配对:")
al_cu = coint_results[coint_results['pair'] == 'AL0-CU0']
if not al_cu.empty:
    row = al_cu.iloc[0]
    print(f"  协整结果: X={row['symbol_x']}, Y={row['symbol_y']}")
    print(f"  Beta: {row.get('beta_4y', 'N/A')}")
    
    # AL0价格约18930, CU0价格约73770
    print(f"  AL0价格~18930, CU0价格~73770")
    print(f"  当前理解: 价差 = CU0 - β×AL0")
    print(f"  β = 0.87，价差 = 73770 - 0.87×18930 ≈ 73770 - 16469 ≈ 57301")
    print(f"  这看起来可能合理")

# 3. 检查I0相关配对  
print("\n3. 检查I0相关配对:")
i0_pairs = coint_results[coint_results['pair'].str.contains('I0')]
print(f"找到{len(i0_pairs)}个I0相关配对:")

for _, row in i0_pairs.head(3).iterrows():
    print(f"  {row['pair']}: X={row['symbol_x']}, Y={row['symbol_y']}, β={row.get('beta_4y', 'N/A')}")
    
    # I0价格约839
    if row['symbol_y'] == 'I0':
        print(f"    价差 = I0 - β×{row['symbol_x']}")
        print(f"    I0价格很低(839)，作为Y可能不合理")
    elif row['symbol_x'] == 'I0':
        print(f"    价差 = {row['symbol_y']} - β×I0") 
        print(f"    I0价格低(839)，作为X可能合理")

print(f"\n🔍 问题分析:")
print("协整分析时可能有以下问题:")
print("1. X和Y的定义不一致")
print("2. 高价格品种和低价格品种的配对方向错误")
print("3. Beta值没有考虑价格量级差异")

print(f"\n💡 解决方案:")
print("需要检查协整分析的代码，确保:")
print("1. 价差公式的X和Y定义一致")
print("2. Beta计算考虑价格量级")
print("3. 配对方向选择合理（通常高价格做X，低价格做Y）")

# 4. 检查我们的协整代码逻辑
print(f"\n4. 检查协整分析逻辑:")
print("需要查看CointegrationAnalyzer的实现")
print("特别是如何确定X和Y，以及如何计算Beta")