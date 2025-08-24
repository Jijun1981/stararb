#!/usr/bin/env python3
"""
调试Beta计算，找出问题所在
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/mnt/e/Star-arb')

from lib.data import load_from_parquet

print("="*80)
print("调试Beta计算逻辑")
print("="*80)

# 测试AL0-SN0配对
print("\n1. 测试AL0-SN0配对")
print("-"*40)

# 加载数据
df_al = load_from_parquet('AL0')
df_sn = load_from_parquet('SN0')

# 选择特定日期范围
start_date = '2023-04-01'
end_date = '2023-07-03'

df_al = df_al[(df_al.index >= start_date) & (df_al.index <= end_date)]
df_sn = df_sn[(df_sn.index >= start_date) & (df_sn.index <= end_date)]

# 合并数据
data = pd.merge(df_al[['close']], df_sn[['close']], 
               left_index=True, right_index=True, 
               how='inner', suffixes=('_x', '_y'))

print(f"symbol_x = AL0, symbol_y = SN0")
print(f"direction = y_on_x (SN0对AL0回归)")
print(f"数据点数: {len(data)}")

# 计算对数价格
data['log_x'] = np.log(data['close_x'])  # log(AL0)
data['log_y'] = np.log(data['close_y'])  # log(SN0)

# 取最后60天数据
window = 60
if len(data) >= window:
    window_data = data.iloc[-window:]
    
    # 按照y_on_x方向计算
    y_var = window_data['log_y']  # SN0
    x_var = window_data['log_x']  # AL0
    
    # 计算Beta: SN0 = alpha + beta * AL0
    covariance = np.cov(y_var, x_var, ddof=1)[0, 1]
    variance_x = np.var(x_var, ddof=1)
    beta = covariance / variance_x
    
    print(f"\n最后60天窗口Beta计算:")
    print(f"  Cov(log_SN0, log_AL0) = {covariance:.6f}")
    print(f"  Var(log_AL0) = {variance_x:.6f}")
    print(f"  Beta = {beta:.6f}")
    
    # 验证：用statsmodels OLS
    from statsmodels.api import OLS, add_constant
    X = add_constant(x_var.values)
    model = OLS(y_var.values, X).fit()
    print(f"\nstatsmodels验证:")
    print(f"  Beta = {model.params[1]:.6f}")
    print(f"  一致性检查: {'✓' if abs(beta - model.params[1]) < 0.0001 else '✗'}")

print("\n" + "="*80)
print("2. 测试HC0-I0配对")
print("-"*40)

# HC0-I0配对
df_hc = load_from_parquet('HC0')
df_i = load_from_parquet('I0')

df_hc = df_hc[(df_hc.index >= start_date) & (df_hc.index <= end_date)]
df_i = df_i[(df_i.index >= start_date) & (df_i.index <= end_date)]

data2 = pd.merge(df_hc[['close']], df_i[['close']], 
                left_index=True, right_index=True, 
                how='inner', suffixes=('_x', '_y'))

print(f"symbol_x = HC0, symbol_y = I0")
print(f"direction = y_on_x (I0对HC0回归)")
print(f"数据点数: {len(data2)}")

data2['log_x'] = np.log(data2['close_x'])  # log(HC0)
data2['log_y'] = np.log(data2['close_y'])  # log(I0)

if len(data2) >= window:
    window_data2 = data2.iloc[-window:]
    
    # 按照y_on_x方向计算
    y_var2 = window_data2['log_y']  # I0
    x_var2 = window_data2['log_x']  # HC0
    
    # 计算Beta: I0 = alpha + beta * HC0
    covariance2 = np.cov(y_var2, x_var2, ddof=1)[0, 1]
    variance_x2 = np.var(x_var2, ddof=1)
    beta2 = covariance2 / variance_x2
    
    print(f"\n最后60天窗口Beta计算:")
    print(f"  Cov(log_I0, log_HC0) = {covariance2:.6f}")
    print(f"  Var(log_HC0) = {variance_x2:.6f}")
    print(f"  Beta = {beta2:.6f}")
    
    # 验证反向Beta
    beta2_inverse = np.var(y_var2, ddof=1) / covariance2
    print(f"\n反向Beta (HC0对I0): {1/beta2:.6f}")

print("\n" + "="*80)
print("3. 手数配比验证")
print("-"*40)

# AL0-SN0的手数配比
print(f"\nAL0-SN0配对:")
print(f"  Beta = {beta:.3f} (SN0 = {beta:.3f} * AL0)")
print(f"  理论配比: 1手SN0 : {beta:.3f}手AL0")

# 寻找最佳整数配比
from math import gcd
from fractions import Fraction

# 将Beta转换为最接近的分数
frac = Fraction(beta).limit_denominator(10)
print(f"  最接近的简单分数: {frac}")
print(f"  建议配比: {frac.denominator}手SN0 : {frac.numerator}手AL0")

# HC0-I0的手数配比
print(f"\nHC0-I0配对:")
print(f"  Beta = {beta2:.3f} (I0 = {beta2:.3f} * HC0)")
print(f"  理论配比: 1手I0 : {beta2:.3f}手HC0")

frac2 = Fraction(beta2).limit_denominator(10)
print(f"  最接近的简单分数: {frac2}")
print(f"  建议配比: {frac2.denominator}手I0 : {frac2.numerator}手HC0")

print("\n" + "="*80)
print("结论")
print("-"*40)
print("Beta计算逻辑验证完成。")
print("如果实际交易中Beta值与这里计算的不一致，")
print("可能是数据时间窗口不同导致的。")