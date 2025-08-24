#!/usr/bin/env python3
"""
验证Beta方向和手数计算的正确性
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/mnt/e/Star-arb')

from lib.data import load_from_parquet

# 测试AL0-SN0配对
# direction = y_on_x, 意思是 SN0 = α + β×AL0
# 协整结果显示 beta_1y = 2.225015

print("="*60)
print("验证Beta计算方向 - AL0-SN0")
print("="*60)

# 加载数据
df_al = load_from_parquet('AL0')
df_sn = load_from_parquet('SN0')

# 选择2023年3月到5月的数据做60天窗口测试
start_date = '2023-03-01'
end_date = '2023-05-01'

df_al = df_al[(df_al.index >= start_date) & (df_al.index <= end_date)]
df_sn = df_sn[(df_sn.index >= start_date) & (df_sn.index <= end_date)]

# 合并数据
data = pd.merge(df_al[['close']], df_sn[['close']], 
               left_index=True, right_index=True, 
               how='inner', suffixes=('_al', '_sn'))

print(f"数据点数: {len(data)}")

# 计算对数价格
data['log_al'] = np.log(data['close_al'])
data['log_sn'] = np.log(data['close_sn'])

# 方法1: 按照y_on_x方向 - SN0对AL0回归
# Y = SN0, X = AL0
cov_sn_al = np.cov(data['log_sn'], data['log_al'])[0, 1]
var_al = np.var(data['log_al'], ddof=1)
beta_sn_on_al = cov_sn_al / var_al
print(f"\nSN0对AL0回归 (y_on_x): beta = {beta_sn_on_al:.6f}")

# 方法2: 反向计算 - AL0对SN0回归
# Y = AL0, X = SN0
cov_al_sn = np.cov(data['log_al'], data['log_sn'])[0, 1]
var_sn = np.var(data['log_sn'], ddof=1)
beta_al_on_sn = cov_al_sn / var_sn
print(f"AL0对SN0回归 (x_on_y): beta = {beta_al_on_sn:.6f}")

# 验证倒数关系
print(f"\n1/beta_sn_on_al = {1/beta_sn_on_al:.6f}")
print(f"beta_al_on_sn = {beta_al_on_sn:.6f}")
print(f"关系验证: 接近倒数关系")

# 手数计算示例
print("\n" + "="*60)
print("手数计算验证")
print("="*60)

# 假设的价格和合约乘数
price_al = 18000  # AL0价格
price_sn = 230000  # SN0价格
multiplier_al = 5  # AL0合约乘数
multiplier_sn = 1  # SN0合约乘数

# 如果Beta = 2.225 (SN0 = 2.225 * AL0)
# 意味着1手SN0对应2.225手AL0
# 但实际我们需要整数手数
beta = 2.225015
print(f"理论Beta (SN0/AL0): {beta}")
print(f"理论比率: 1手SN0 对 {beta:.3f}手AL0")

# 反向: 如果用AL0/SN0的Beta
beta_inverse = 1/beta
print(f"反向Beta (AL0/SN0): {beta_inverse:.6f}")
print(f"理论比率: 1手AL0 对 {beta_inverse:.3f}手SN0")

# 实际手数配比（需要是整数）
# 寻找最接近的整数配比
best_ratio = 0
best_diff = float('inf')
for sn_lots in range(1, 10):
    for al_lots in range(1, 20):
        ratio = al_lots / sn_lots
        diff = abs(ratio - beta)
        if diff < best_diff:
            best_diff = diff
            best_ratio = (sn_lots, al_lots)
            
print(f"\n最优整数手数配比: {best_ratio[0]}手SN0 : {best_ratio[1]}手AL0")
print(f"实际比率: {best_ratio[1]/best_ratio[0]:.3f}")
print(f"与理论Beta差异: {abs(best_ratio[1]/best_ratio[0] - beta):.3f}")

# 计算名义价值
nominal_sn = price_sn * best_ratio[0] * multiplier_sn
nominal_al = price_al * best_ratio[1] * multiplier_al
print(f"\n名义价值:")
print(f"SN0: {nominal_sn:,.0f} ({best_ratio[0]}手 × {price_sn} × {multiplier_sn})")
print(f"AL0: {nominal_al:,.0f} ({best_ratio[1]}手 × {price_al} × {multiplier_al})")
print(f"价值比率: {nominal_al/nominal_sn:.3f}")