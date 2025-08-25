#!/usr/bin/env python3
"""
按照需求文档验证信号生成的合规性
检查：β数值，方向，手数计算，乘数，long/short方向
"""
import pandas as pd
import numpy as np
import os
from lib.data import load_all_symbols_data

def verify_requirements_compliance():
    """按照需求文档验证信号生成合规性"""
    
    print("📋 按需求文档验证信号生成合规性")
    print("=" * 60)
    
    # 加载最新信号文件
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if not signal_files:
        print("❌ 未找到信号文件")
        return
        
    latest_signal_file = max(signal_files)
    signals_df = pd.read_csv(latest_signal_file)
    print(f"分析信号文件: {latest_signal_file}")
    print(f"总信号数: {len(signals_df)}")
    
    # 验证1: 符号映射正确性 (REQ-3.4.1)
    print(f"\n=== 验证1: 符号映射正确性 (REQ-3.4.1) ===")
    
    # 检查必需字段是否存在
    required_fields = ['pair', 'symbol_x', 'symbol_y', 'signal', 'z_score', 'beta', 'beta_initial']
    missing_fields = [field for field in required_fields if field not in signals_df.columns]
    if missing_fields:
        print(f"❌ 缺失必需字段: {missing_fields}")
        return
    else:
        print("✅ 所有必需字段存在")
    
    # 检查配对格式
    sample_pairs = signals_df['pair'].unique()[:5]
    for pair in sample_pairs:
        pair_data = signals_df[signals_df['pair'] == pair].iloc[0]
        expected_pair = f"{pair_data['symbol_x']}-{pair_data['symbol_y']}"
        if pair == expected_pair:
            print(f"✅ {pair}: 符号映射正确")
        else:
            print(f"❌ {pair}: 期望{expected_pair}")
    
    # 验证2: β值合理性 (REQ-3.1.3, REQ-3.1.7)
    print(f"\n=== 验证2: β值合理性 (REQ-3.1.3, REQ-3.1.7) ===")
    
    # REQ-3.1.7: β边界保护 [-4, 4]
    beta_values = signals_df['beta'].values
    beta_out_of_bounds = (beta_values < -4) | (beta_values > 4)
    out_of_bounds_count = np.sum(beta_out_of_bounds)
    
    if out_of_bounds_count > 0:
        print(f"❌ {out_of_bounds_count}个β值超出[-4, 4]边界")
        print(f"   范围: [{beta_values.min():.6f}, {beta_values.max():.6f}]")
    else:
        print(f"✅ 所有β值在边界内: [{beta_values.min():.6f}, {beta_values.max():.6f}]")
    
    # 检查β值初始化合理性
    pairs_beta_stats = []
    for pair in signals_df['pair'].unique():
        pair_data = signals_df[signals_df['pair'] == pair]
        if len(pair_data) > 0:
            beta_initial = pair_data['beta_initial'].iloc[0]
            beta_range = [pair_data['beta'].min(), pair_data['beta'].max()]
            beta_change = beta_range[1] - beta_range[0]
            
            pairs_beta_stats.append({
                'pair': pair,
                'beta_initial': beta_initial,
                'beta_range': beta_range,
                'beta_change': beta_change
            })
    
    # 显示β值变化最大的配对
    pairs_beta_df = pd.DataFrame(pairs_beta_stats)
    top_changes = pairs_beta_df.nlargest(5, 'beta_change')
    print(f"\nβ值变化最大的5个配对:")
    for _, row in top_changes.iterrows():
        print(f"  {row['pair']}: 初始={row['beta_initial']:.3f}, "
              f"范围=[{row['beta_range'][0]:.3f}, {row['beta_range'][1]:.3f}], "
              f"变化={row['beta_change']:.3f}")
    
    # 验证3: 信号生成逻辑 (REQ-3.3.1, REQ-3.3.2, REQ-3.3.4)
    print(f"\n=== 验证3: 信号生成逻辑 (REQ-3.3.1~3.3.4) ===")
    
    # REQ-3.3.4: 信号类型检查
    expected_signals = {'open_long', 'open_short', 'holding_long', 'holding_short', 'close', 'empty', 'warm_up'}
    actual_signals = set(signals_df['signal'].unique())
    unexpected_signals = actual_signals - expected_signals
    missing_signals = expected_signals - actual_signals
    
    if unexpected_signals:
        print(f"❌ 发现未预期的信号类型: {unexpected_signals}")
    if missing_signals:
        print(f"ℹ️ 未出现的信号类型: {missing_signals}")
    else:
        print(f"✅ 信号类型符合规范: {actual_signals}")
    
    # REQ-3.3.1: 开仓阈值 |z| > 2.0
    # REQ-3.3.2: 平仓阈值 |z| < 0.5  
    open_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short'])]
    close_signals = signals_df[signals_df['signal'] == 'close']
    
    if len(open_signals) > 0:
        z_open = open_signals['z_score'].values
        z_open_violations = np.abs(z_open) < 2.0
        if np.any(z_open_violations):
            violation_count = np.sum(z_open_violations)
            print(f"❌ {violation_count}个开仓信号违反|z|>2.0阈值")
        else:
            print(f"✅ 所有{len(open_signals)}个开仓信号符合|z|>2.0阈值")
    
    if len(close_signals) > 0:
        z_close = close_signals['z_score'].values
        # 注意：强制平仓可能不满足z<0.5
        force_close = close_signals[close_signals['reason'] == 'force_close']
        normal_close = close_signals[close_signals['reason'] != 'force_close']
        
        if len(normal_close) > 0:
            z_normal_close = normal_close['z_score'].values  
            z_close_violations = np.abs(z_normal_close) >= 0.5
            if np.any(z_close_violations):
                violation_count = np.sum(z_close_violations)
                print(f"❌ {violation_count}个平仓信号违反|z|<0.5阈值")
            else:
                print(f"✅ 所有{len(normal_close)}个正常平仓信号符合|z|<0.5阈值")
    
    # 验证4: Long/Short方向逻辑
    print(f"\n=== 验证4: Long/Short方向逻辑 ===")
    
    # 检查开仓信号的z-score方向
    if len(open_signals) > 0:
        long_signals = open_signals[open_signals['signal'] == 'open_long']
        short_signals = open_signals[open_signals['signal'] == 'open_short']
        
        long_direction_correct = True
        short_direction_correct = True
        
        if len(long_signals) > 0:
            long_z_scores = long_signals['z_score'].values
            if not np.all(long_z_scores < -2.0):
                long_direction_correct = False
                print(f"❌ Long信号应对应z<-2.0，实际范围: [{long_z_scores.min():.3f}, {long_z_scores.max():.3f}]")
            else:
                print(f"✅ {len(long_signals)}个Long信号方向正确 (z<-2.0)")
        
        if len(short_signals) > 0:
            short_z_scores = short_signals['z_score'].values
            if not np.all(short_z_scores > 2.0):
                short_direction_correct = False
                print(f"❌ Short信号应对应z>2.0，实际范围: [{short_z_scores.min():.3f}, {short_z_scores.max():.3f}]")
            else:
                print(f"✅ {len(short_signals)}个Short信号方向正确 (z>2.0)")
    
    # 验证5: 持仓天数限制 (REQ-3.3.3)
    print(f"\n=== 验证5: 持仓天数限制 (REQ-3.3.3) ===")
    
    # REQ-3.3.3: 最大持仓30天
    holding_signals = signals_df[signals_df['signal'].str.contains('holding_')]
    if len(holding_signals) > 0:
        days_held = holding_signals['days_held'].values
        max_days = days_held.max()
        over_limit = np.sum(days_held > 30)
        
        if over_limit > 0:
            print(f"❌ {over_limit}个持仓信号超过30天限制，最大持仓{max_days}天")
        else:
            print(f"✅ 所有持仓信号符合30天限制，最大持仓{max_days}天")
    
    # 验证6: 合约乘数和手数计算合理性
    print(f"\n=== 验证6: 合约乘数和手数计算合理性 ===")
    
    # 期货合约乘数（常识检查）
    multipliers = {
        'AG': 15,  # 白银15千克/手
        'AL': 5,   # 铝5吨/手  
        'AU': 1000,# 黄金1000克/手
        'CU': 5,   # 铜5吨/手
        'HC': 10,  # 热卷10吨/手
        'I': 100,  # 铁矿100吨/手
        'NI': 1,   # 镍1吨/手
        'PB': 25,  # 铅25吨/手
        'RB': 10,  # 螺纹10吨/手
        'SF': 5,   # 硅铁5吨/手
        'SM': 5,   # 锰硅5吨/手
        'SN': 1,   # 锡1吨/手
        'SS': 5,   # 不锈钢5吨/手
        'ZN': 5    # 锌5吨/手
    }
    
    print("理论对冲比例计算（h* = β × (Py × My) / (Px × Mx)）:")
    
    # 检查几个主要配对的对冲比例
    key_pairs = ['NI-AG', 'AU-ZN', 'CU-SN', 'RB-SM']
    for pair in key_pairs:
        if pair in signals_df['pair'].unique():
            pair_data = signals_df[signals_df['pair'] == pair].iloc[-1]  # 最新数据
            symbol_x = pair_data['symbol_x']
            symbol_y = pair_data['symbol_y']
            beta = pair_data['beta']
            
            mx = multipliers.get(symbol_x, 1)
            my = multipliers.get(symbol_y, 1)
            
            # h* = β × (Py × My) / (Px × Mx)，假设价格比例为1
            hedge_ratio = beta * my / mx
            
            print(f"  {pair}: β={beta:.3f}, {symbol_x}({mx}) : {symbol_y}({my})")
            print(f"    理论对冲比例: 1 : {abs(hedge_ratio):.2f}")
            
            if abs(hedge_ratio) > 100:
                print(f"    ⚠️ 对冲比例过高")
            elif abs(hedge_ratio) < 0.01:
                print(f"    ⚠️ 对冲比例过低") 
            else:
                print(f"    ✅ 对冲比例合理")
    
    # 验证7: Z-score质量 (REQ-3.5.1)
    print(f"\n=== 验证7: Z-score质量 (REQ-3.5.1) ===")
    
    # REQ-3.5.1: 最近60根z方差 ∈ [0.8, 1.3]
    signal_period_data = signals_df[signals_df['phase'] == 'signal_period']
    if len(signal_period_data) > 0:
        pairs_quality = []
        for pair in signal_period_data['pair'].unique():
            pair_signals = signal_period_data[signal_period_data['pair'] == pair]
            if len(pair_signals) >= 60:
                z_recent = pair_signals['z_score'].iloc[-60:].values
                z_var = np.var(z_recent, ddof=1)
                
                if 0.8 <= z_var <= 1.3:
                    quality = 'good'
                elif 0.6 <= z_var <= 1.6:
                    quality = 'warning'  
                else:
                    quality = 'bad'
                
                pairs_quality.append({
                    'pair': pair,
                    'z_var': z_var,
                    'quality': quality,
                    'n_signals': len(pair_signals)
                })
        
        if pairs_quality:
            quality_df = pd.DataFrame(pairs_quality)
            good_count = len(quality_df[quality_df['quality'] == 'good'])
            warning_count = len(quality_df[quality_df['quality'] == 'warning'])
            bad_count = len(quality_df[quality_df['quality'] == 'bad'])
            
            print(f"配对质量分布: Good={good_count}, Warning={warning_count}, Bad={bad_count}")
            print(f"质量合格率: {good_count/len(quality_df)*100:.1f}%")
            
            # 显示质量最差的配对
            worst_pairs = quality_df.nsmallest(3, 'z_var') if bad_count > 0 else quality_df.head(3)
            print("质量需要关注的配对:")
            for _, row in worst_pairs.iterrows():
                print(f"  {row['pair']}: z_var={row['z_var']:.3f}, {row['quality']}")
    
    print(f"\n{'='*60}")
    print("🎯 合规性验证完成")
    
    return {
        'signals_df': signals_df,
        'pairs_beta_stats': pairs_beta_df if 'pairs_beta_df' in locals() else None,
        'pairs_quality': quality_df if 'quality_df' in locals() else None
    }

if __name__ == "__main__":
    results = verify_requirements_compliance()