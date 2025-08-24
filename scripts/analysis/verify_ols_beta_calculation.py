#!/usr/bin/env python3
"""
验证OLS Pipeline中的Beta计算逻辑
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/mnt/e/Star-arb')

from lib.data import load_from_parquet
from statsmodels.api import OLS, add_constant

print("="*80)
print("验证OLS Pipeline Beta计算")
print("="*80)

# 测试配对1: AL0-SN0 (direction=y_on_x)
print("\n1. AL0-SN0配对 (y_on_x: SN0 = α + β×AL0)")
print("-"*40)

# 加载数据
df_al = load_from_parquet('AL0')
df_sn = load_from_parquet('SN0')

# 使用与OLS pipeline相同的日期范围
data_start = '2023-03-01'
signal_start = '2023-07-01'
end_date = '2024-12-31'

df_al = df_al[(df_al.index >= data_start) & (df_al.index <= end_date)]
df_sn = df_sn[(df_sn.index >= data_start) & (df_sn.index <= end_date)]

# 合并数据
data = pd.merge(df_al[['close']], df_sn[['close']], 
               left_index=True, right_index=True, 
               how='inner', suffixes=('_x', '_y'))

print(f"数据点数: {len(data)}")
print(f"日期范围: {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}")

# 计算对数价格
data['log_x'] = np.log(data['close_x'])  # log(AL0)
data['log_y'] = np.log(data['close_y'])  # log(SN0)

# 找到2023-07-01之后的第一个交易日
test_date = pd.Timestamp('2023-07-01')
available_dates = data.index[data.index >= test_date]
if len(available_dates) > 0:
    test_date = available_dates[0]
    print(f"\n使用交易日: {test_date.strftime('%Y-%m-%d')}")
if test_date in data.index:
    end_idx = data.index.get_loc(test_date)
    start_idx = max(0, end_idx - 60 + 1)
    window_data = data.iloc[start_idx:end_idx+1]
    
    print(f"\n2023-07-01的60天窗口:")
    print(f"  窗口大小: {len(window_data)}天")
    print(f"  窗口起始: {window_data.index[0].strftime('%Y-%m-%d')}")
    print(f"  窗口结束: {window_data.index[-1].strftime('%Y-%m-%d')}")
    
    # direction = y_on_x: SN0对AL0回归
    y_var = window_data['log_y']  # log(SN0)
    x_var = window_data['log_x']  # log(AL0)
    
    # 计算Beta
    covariance = np.cov(y_var, x_var, ddof=1)[0, 1]
    variance_x = np.var(x_var, ddof=1)
    beta = covariance / variance_x
    
    print(f"\n手动计算Beta:")
    print(f"  Cov(log_SN0, log_AL0) = {covariance:.6f}")
    print(f"  Var(log_AL0) = {variance_x:.6f}")
    print(f"  Beta = {beta:.6f}")
    
    # 用statsmodels验证
    X = add_constant(x_var.values)
    model = OLS(y_var.values, X).fit()
    print(f"\nstatsmodels验证:")
    print(f"  Beta = {model.params[1]:.6f}")
    print(f"  R-squared = {model.rsquared:.4f}")
    
    # 计算残差和Z-score
    residuals = y_var - beta * x_var
    current_residual = residuals.iloc[-1]
    z_score = (current_residual - residuals.mean()) / residuals.std()
    
    print(f"\n残差分析:")
    print(f"  当前残差 = {current_residual:.6f}")
    print(f"  残差均值 = {residuals.mean():.6f}")
    print(f"  残差标准差 = {residuals.std():.6f}")
    print(f"  Z-score = {z_score:.3f}")
    
    # 理论手数比例
    print(f"\n理论手数比例:")
    print(f"  1手SN0 : {beta:.3f}手AL0")
    print(f"  或者: {1/beta:.3f}手SN0 : 1手AL0")

print("\n" + "="*80)
print("2. HC0-I0配对 (y_on_x: I0 = α + β×HC0)")
print("-"*40)

# 测试配对2: HC0-I0
df_hc = load_from_parquet('HC0')
df_i = load_from_parquet('I0')

df_hc = df_hc[(df_hc.index >= data_start) & (df_hc.index <= end_date)]
df_i = df_i[(df_i.index >= data_start) & (df_i.index <= end_date)]

data2 = pd.merge(df_hc[['close']], df_i[['close']], 
                left_index=True, right_index=True, 
                how='inner', suffixes=('_x', '_y'))

data2['log_x'] = np.log(data2['close_x'])  # log(HC0)
data2['log_y'] = np.log(data2['close_y'])  # log(I0)

if test_date in data2.index:
    end_idx2 = data2.index.get_loc(test_date)
    start_idx2 = max(0, end_idx2 - 60 + 1)
    window_data2 = data2.iloc[start_idx2:end_idx2+1]
    
    print(f"2023-07-01的60天窗口:")
    print(f"  窗口大小: {len(window_data2)}天")
    
    # direction = y_on_x: I0对HC0回归
    y_var2 = window_data2['log_y']  # log(I0)
    x_var2 = window_data2['log_x']  # log(HC0)
    
    # 计算Beta
    covariance2 = np.cov(y_var2, x_var2, ddof=1)[0, 1]
    variance_x2 = np.var(x_var2, ddof=1)
    beta2 = covariance2 / variance_x2
    
    print(f"\n手动计算Beta:")
    print(f"  Cov(log_I0, log_HC0) = {covariance2:.6f}")
    print(f"  Var(log_HC0) = {variance_x2:.6f}")
    print(f"  Beta = {beta2:.6f}")
    
    # 理论手数比例
    print(f"\n理论手数比例:")
    print(f"  1手I0 : {beta2:.3f}手HC0")
    print(f"  或者: {1/beta2:.3f}手I0 : 1手HC0")

print("\n" + "="*80)
print("结论")
print("-"*40)
print("以上计算显示了OLS Pipeline应该得到的Beta值。")
print("如果实际运行结果与此不同，说明代码存在逻辑错误。")