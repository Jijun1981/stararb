#!/usr/bin/env python3
"""
分析协整分析与信号生成时间窗口不匹配问题
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def analyze_time_mismatch():
    """分析时间窗口不匹配问题"""
    
    print("🕐 分析协整分析与信号生成的时间窗口匹配问题")
    print("=" * 70)
    
    # 加载数据
    data = load_all_symbols_data()
    
    # 获取NI-AG数据
    ni_data = data['NI'].dropna()
    ag_data = data['AG'].dropna()
    common_dates = ni_data.index.intersection(ag_data.index)
    ni_aligned = ni_data[common_dates]
    ag_aligned = ag_data[common_dates]
    
    print(f"总数据范围: {common_dates[0]} 至 {common_dates[-1]}")
    print(f"总数据长度: {len(common_dates)} 个交易日")
    
    # 1. 协整分析的时间窗口
    print(f"\n=== 协整分析的时间窗口 (最近252个交易日) ===")
    
    # 最近252个交易日
    coint_window_data = ni_aligned.iloc[-252:]
    coint_dates = coint_window_data.index
    
    print(f"协整分析窗口: {coint_dates[0]} 至 {coint_dates[-1]}")
    print(f"协整分析长度: {len(coint_dates)} 个交易日")
    
    # 计算协整分析窗口的β值
    ni_coint = ni_aligned[coint_dates]
    ag_coint = ag_aligned[coint_dates]
    
    reg_coint = LinearRegression()
    reg_coint.fit(ni_coint.values.reshape(-1, 1), ag_coint.values)
    beta_coint = reg_coint.coef_[0]
    
    print(f"协整分析β值: {beta_coint:.6f}")
    
    # 2. 信号生成的预热期
    print(f"\n=== 信号生成的预热期 (2024-02-08开始60天) ===")
    
    signal_start = pd.to_datetime('2024-02-08')
    
    # 找到信号开始日期在数据中的位置
    signal_start_idx = common_dates.get_indexer([signal_start], method='nearest')[0]
    warmup_end_idx = signal_start_idx + 60
    
    warmup_dates = common_dates[signal_start_idx:warmup_end_idx]
    
    print(f"信号预热窗口: {warmup_dates[0]} 至 {warmup_dates[-1]}")
    print(f"信号预热长度: {len(warmup_dates)} 个交易日")
    
    # 计算预热期的β值
    ni_warmup = ni_aligned[warmup_dates]
    ag_warmup = ag_aligned[warmup_dates]
    
    reg_warmup = LinearRegression()
    reg_warmup.fit(ni_warmup.values.reshape(-1, 1), ag_warmup.values)
    beta_warmup = reg_warmup.coef_[0]
    
    print(f"信号预热β值: {beta_warmup:.6f}")
    
    # 3. 时间窗口重叠度分析
    print(f"\n=== 时间窗口重叠度分析 ===")
    
    overlap_dates = coint_dates.intersection(warmup_dates)
    overlap_ratio_coint = len(overlap_dates) / len(coint_dates)
    overlap_ratio_warmup = len(overlap_dates) / len(warmup_dates)
    
    print(f"重叠交易日数: {len(overlap_dates)}")
    print(f"协整窗口重叠度: {overlap_ratio_coint:.1%}")
    print(f"预热窗口重叠度: {overlap_ratio_warmup:.1%}")
    
    if overlap_ratio_coint < 0.5:
        print("❌ 时间窗口严重不匹配！")
    elif overlap_ratio_coint < 0.8:
        print("⚠️ 时间窗口部分不匹配")
    else:
        print("✅ 时间窗口匹配良好")
    
    # 4. 计算"正确"的协整β值（应该用什么时间窗口）
    print(f"\n=== 应该使用的协整分析窗口建议 ===")
    
    # 方案1: 使用信号生成期之前的252天作为协整分析窗口
    if signal_start_idx >= 252:
        coint_proper_start_idx = signal_start_idx - 252
        coint_proper_dates = common_dates[coint_proper_start_idx:signal_start_idx]
        
        print(f"方案1 - 信号前252天: {coint_proper_dates[0]} 至 {coint_proper_dates[-1]}")
        
        ni_proper = ni_aligned[coint_proper_dates]
        ag_proper = ag_aligned[coint_proper_dates]
        
        reg_proper = LinearRegression()
        reg_proper.fit(ni_proper.values.reshape(-1, 1), ag_proper.values)
        beta_proper = reg_proper.coef_[0]
        
        print(f"方案1 β值: {beta_proper:.6f}")
        
        # 与预热期β值比较
        beta_diff_proper = abs(beta_proper - beta_warmup)
        print(f"与预热期β值差异: {beta_diff_proper:.6f}")
        
    # 方案2: 使用预热期结束时的最近252天
    warmup_end_date = warmup_dates[-1]
    warmup_end_idx = common_dates.get_indexer([warmup_end_date], method='nearest')[0]
    
    if warmup_end_idx >= 252:
        recent_start_idx = warmup_end_idx - 252 + 1
        recent_dates = common_dates[recent_start_idx:warmup_end_idx+1]
        
        print(f"\n方案2 - 预热结束时最近252天: {recent_dates[0]} 至 {recent_dates[-1]}")
        
        ni_recent = ni_aligned[recent_dates]
        ag_recent = ag_aligned[recent_dates]
        
        reg_recent = LinearRegression()
        reg_recent.fit(ni_recent.values.reshape(-1, 1), ag_recent.values)
        beta_recent = reg_recent.coef_[0]
        
        print(f"方案2 β值: {beta_recent:.6f}")
        
        # 与预热期β值比较
        beta_diff_recent = abs(beta_recent - beta_warmup)
        print(f"与预热期β值差异: {beta_diff_recent:.6f}")
    
    # 5. 推荐解决方案
    print(f"\n=== 推荐解决方案 ===")
    print("问题根源: 协整分析使用的时间窗口与信号生成期不匹配")
    print("当前情况:")
    print(f"  - 协整β值: {beta_coint:.6f} (2024年8月-2025年8月)")
    print(f"  - 预热β值: {beta_warmup:.6f} (2024年2-4月)")
    print(f"  - β值差异: {abs(beta_coint - beta_warmup):.6f}")
    print()
    print("建议修改:")
    print("1. 修改协整分析，使其使用信号生成期之前的历史数据")
    print("2. 或者调整信号生成的预热期，使其与协整分析窗口一致")
    print("3. 或者在Kalman滤波初始化时使用预热期的β值而不是协整分析的β值")

if __name__ == "__main__":
    analyze_time_mismatch()