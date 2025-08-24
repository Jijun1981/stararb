#!/usr/bin/env python3
"""
分析符号映射不一致问题的根本原因
"""
import pandas as pd
import numpy as np
import os

def analyze_symbol_mapping_issue():
    """分析符号映射不一致的根本原因"""
    
    print("🔍 分析符号映射不一致问题")
    print("=" * 50)
    
    # 1. 读取协整分析结果
    coint_file = "./output/pipeline_v21/cointegrated_pairs_20250822_171214.csv"
    if os.path.exists(coint_file):
        coint_df = pd.read_csv(coint_file)
        print(f"✅ 协整文件存在: {len(coint_df)} 个配对")
        
        # 查找AG-NI相关配对
        ag_ni_patterns = ['NI.*AG', 'AG.*NI']
        ag_ni_pairs = []
        
        for _, row in coint_df.iterrows():
            pair_name = row['pair']
            if any(pd.Series([pair_name]).str.contains(pattern).iloc[0] for pattern in ag_ni_patterns):
                ag_ni_pairs.append(row)
        
        if ag_ni_pairs:
            print(f"\n🎯 找到AG-NI相关配对 ({len(ag_ni_pairs)}个):")
            for pair_info in ag_ni_pairs:
                print(f"  配对名: {pair_info['pair']}")
                print(f"  symbol_x: {pair_info['symbol_x']}")
                print(f"  symbol_y: {pair_info['symbol_y']}")  
                print(f"  β_1y: {pair_info['beta_1y']:.6f}")
                print(f"  direction: {pair_info['direction']}")
                print("-" * 30)
        else:
            print("❌ 协整文件中未找到AG-NI配对")
            
    else:
        print(f"❌ 协整文件不存在: {coint_file}")
        return
    
    # 2. 读取信号生成结果
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if signal_files:
        latest_signal_file = max(signal_files)
        signal_df = pd.read_csv(latest_signal_file)
        print(f"\n✅ 信号文件存在: {latest_signal_file}")
        
        # 查找AG-NI信号
        ag_ni_signals = signal_df[signal_df['pair'] == 'AG-NI']
        
        if len(ag_ni_signals) > 0:
            first_signal = ag_ni_signals.iloc[0]
            print(f"\n🎯 AG-NI信号映射:")
            print(f"  配对名: {first_signal['pair']}")
            print(f"  symbol_x: {first_signal['symbol_x']}")
            print(f"  symbol_y: {first_signal['symbol_y']}")
            print(f"  beta_initial: {first_signal['beta_initial']:.6f}")
            print(f"  beta_window_used: {first_signal['beta_window_used']}")
        else:
            print("❌ 信号文件中未找到AG-NI配对")
    else:
        print("❌ 未找到信号文件")
        return
    
    # 3. 问题定位
    print(f"\n🚨 问题分析:")
    
    if ag_ni_pairs and len(ag_ni_signals) > 0:
        coint_pair = ag_ni_pairs[0]
        signal_pair = ag_ni_signals.iloc[0]
        
        print(f"协整文件: {coint_pair['symbol_x']} -> {coint_pair['symbol_y']}, β={coint_pair['beta_1y']:.6f}")
        print(f"信号文件: {signal_pair['symbol_x']} -> {signal_pair['symbol_y']}, β={signal_pair['beta_initial']:.6f}")
        
        # 检查符号映射
        if coint_pair['symbol_x'] != signal_pair['symbol_x']:
            print("❌ X符号不匹配!")
            print(f"   协整: {coint_pair['symbol_x']} vs 信号: {signal_pair['symbol_x']}")
            
        if coint_pair['symbol_y'] != signal_pair['symbol_y']:
            print("❌ Y符号不匹配!")
            print(f"   协整: {coint_pair['symbol_y']} vs 信号: {signal_pair['symbol_y']}")
            
        # 检查β值
        beta_diff = abs(coint_pair['beta_1y'] - signal_pair['beta_initial'])
        if beta_diff > 0.01:
            print(f"❌ β值差异过大: {beta_diff:.6f}")
        else:
            print(f"✅ β值匹配: 差异仅{beta_diff:.6f}")
            
    # 4. 分析配对名称生成逻辑
    print(f"\n🔧 配对名称生成逻辑分析:")
    
    if ag_ni_pairs:
        coint_pair = ag_ni_pairs[0]
        original_pair_name = coint_pair['pair']
        expected_x = coint_pair['symbol_x'] 
        expected_y = coint_pair['symbol_y']
        
        print(f"协整文件中的pair名称: {original_pair_name}")
        print(f"协整文件中的X符号: {expected_x}")
        print(f"协整文件中的Y符号: {expected_y}")
        
        # 检查配对名称是否是 symbol_x + '-' + symbol_y 格式
        expected_pair_name = f"{expected_x}-{expected_y}"
        if original_pair_name == expected_pair_name:
            print(f"✅ 配对名称格式正确: {expected_pair_name}")
        else:
            print(f"⚠️ 配对名称格式异常:")
            print(f"   实际: {original_pair_name}")
            print(f"   期望: {expected_pair_name}")
            
        # 分析符号后缀问题 (可能有0_close等后缀)
        clean_x = expected_x.replace('0_close', '').replace('_close', '')
        clean_y = expected_y.replace('0_close', '').replace('_close', '')
        print(f"清理后的符号: {clean_x}-{clean_y}")
        
        if len(ag_ni_signals) > 0:
            signal_pair_name = ag_ni_signals.iloc[0]['pair']
            signal_x = ag_ni_signals.iloc[0]['symbol_x']
            signal_y = ag_ni_signals.iloc[0]['symbol_y']
            
            print(f"信号中的配对名称: {signal_pair_name}")
            print(f"信号中的X符号: {signal_x}")
            print(f"信号中的Y符号: {signal_y}")
            
            # 检查是否是符号后缀清理问题
            if clean_x == signal_x and clean_y == signal_y:
                print("✅ 符号映射正确，只是后缀问题")
            elif clean_x == signal_y and clean_y == signal_x:
                print("❌ 符号映射完全颠倒!")
                print("   这是导致β值符号错误的根本原因")
            else:
                print("❓ 符号映射存在其他问题")
    
    # 5. 解决方案
    print(f"\n💡 解决方案:")
    print("1. 检查协整结果读取逻辑，确保正确解析symbol_x和symbol_y")
    print("2. 检查信号生成时的配对名称解析逻辑")
    print("3. 确保符号后缀处理的一致性")
    print("4. 修复后重新生成信号")

if __name__ == "__main__":
    analyze_symbol_mapping_issue()