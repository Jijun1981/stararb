#!/usr/bin/env python3
"""
信号分析调试脚本 - 系统检查β值、方向、手数计算、乘数、long/short方向
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
import os

def debug_signal_analysis():
    """系统性检查信号生成的所有关键要素"""
    print("=== 信号分析调试 ===")
    
    # 1. 加载最新的信号文件
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if not signal_files:
        print("❌ 没有找到信号文件")
        return
    
    latest_signal_file = max(signal_files)
    print(f"📊 分析信号文件: {latest_signal_file}")
    
    signals_df = pd.read_csv(latest_signal_file)
    print(f"📈 总信号数量: {len(signals_df)}")
    
    # 2. 检查β值不匹配问题
    print("\n=== β值不匹配分析 ===")
    
    # 获取所有配对的β值比较
    pairs_beta_comparison = []
    for pair in signals_df['pair'].unique():
        pair_data = signals_df[signals_df['pair'] == pair]
        if len(pair_data) > 0:
            beta_initial = pair_data['beta_initial'].iloc[0]
            beta_current = pair_data['beta'].iloc[-1]  # 最新的β值
            
            pairs_beta_comparison.append({
                'pair': pair,
                'beta_initial': beta_initial,
                'beta_current': beta_current,
                'beta_diff': beta_current - beta_initial,
                'beta_ratio': beta_current / beta_initial if beta_initial != 0 else None,
                'sign_match': np.sign(beta_initial) == np.sign(beta_current)
            })
    
    beta_df = pd.DataFrame(pairs_beta_comparison)
    print(f"🔍 配对总数: {len(beta_df)}")
    
    # 符号不匹配的配对
    sign_mismatch = beta_df[~beta_df['sign_match']]
    print(f"❌ 符号不匹配配对数: {len(sign_mismatch)}")
    if len(sign_mismatch) > 0:
        print("符号不匹配的配对:")
        for _, row in sign_mismatch.head(10).iterrows():
            print(f"  {row['pair']}: 初始={row['beta_initial']:.6f}, 当前={row['beta_current']:.6f}")
    
    # 3. 检查AG-NI配对的详细情况
    print("\n=== AG-NI配对详细分析 ===")
    ag_ni_signals = signals_df[signals_df['pair'] == 'AG-NI']
    
    if len(ag_ni_signals) > 0:
        print(f"AG-NI信号数量: {len(ag_ni_signals)}")
        print(f"初始β值: {ag_ni_signals['beta_initial'].iloc[0]}")
        print(f"当前β值: {ag_ni_signals['beta'].iloc[-1]}")
        print(f"符号X: {ag_ni_signals['symbol_x'].iloc[0]}")
        print(f"符号Y: {ag_ni_signals['symbol_y'].iloc[0]}")
        
        # 检查交易信号
        trade_signals = ag_ni_signals[ag_ni_signals['signal'].str.contains('open_|holding_')]
        if len(trade_signals) > 0:
            print(f"\n交易信号数量: {len(trade_signals)}")
            latest_trade = trade_signals.iloc[-1]
            print(f"最新交易信号: {latest_trade['signal']}")
            print(f"Z-score: {latest_trade['z_score']:.6f}")
            print(f"创新项: {latest_trade['innovation']:.6f}")
            print(f"当前β: {latest_trade['beta']:.6f}")
    
    # 4. 重新验证协整分析的β值
    print("\n=== 重新验证协整分析β值 ===")
    try:
        # 加载数据
        data = load_all_symbols_data()
        print(f"数据加载成功，形状: {data.shape}")
        
        # 重新运行协整分析
        analyzer = CointegrationAnalyzer()
        coint_results = analyzer.analyze_pairs(
            data=data,
            reference_date='2024-01-01'
        )
        
        # 查找AG-NI配对
        ag_ni_coint = None
        for result in coint_results:
            if (result['symbol_x'] == 'AG' and result['symbol_y'] == 'NI') or \
               (result['symbol_x'] == 'NI' and result['symbol_y'] == 'AG'):
                ag_ni_coint = result
                break
        
        if ag_ni_coint:
            print("✅ 找到AG-NI协整结果:")
            print(f"  符号X: {ag_ni_coint['symbol_x']}")
            print(f"  符号Y: {ag_ni_coint['symbol_y']}")
            print(f"  β值: {ag_ni_coint['beta']:.6f}")
            print(f"  p值: {ag_ni_coint['p_value']:.6f}")
            print(f"  方向: {ag_ni_coint.get('direction', 'N/A')}")
        else:
            print("❌ 未找到AG-NI协整结果")
            
    except Exception as e:
        print(f"❌ 协整分析验证失败: {e}")
    
    # 5. 检查配对方向定义问题
    print("\n=== 配对方向定义检查 ===")
    
    # 分析波动率来验证X/Y分配是否正确
    try:
        data = load_all_symbols_data()
        recent_data = data['2024-01-01':]
        
        ag_vol = recent_data['AG'].std()
        ni_vol = recent_data['NI'].std()
        
        print(f"AG 2024年至今波动率: {ag_vol:.6f}")
        print(f"NI 2024年至今波动率: {ni_vol:.6f}")
        
        if ag_vol < ni_vol:
            print("✅ AG应该作为X（低波动率），NI应该作为Y（高波动率）")
        else:
            print("⚠️ AG波动率更高，可能需要重新分配X/Y")
            
    except Exception as e:
        print(f"❌ 波动率分析失败: {e}")
    
    # 6. 合约乘数检查
    print("\n=== 合约乘数检查 ===")
    multipliers = {
        'AG': 15,  # 白银
        'NI': 1     # 镍
    }
    
    print(f"AG合约乘数: {multipliers['AG']}")
    print(f"NI合约乘数: {multipliers['NI']}")
    
    # 理论对冲比例 h* = β × (Py × My) / (Px × Mx)
    if ag_ni_coint:
        beta = ag_ni_coint['beta']
        # 假设AG作为X，NI作为Y
        hedge_ratio = beta * (1 * multipliers['NI']) / (1 * multipliers['AG'])
        print(f"理论对冲比例 h*: {hedge_ratio:.6f}")
        print(f"AG:NI手数比例约为: 1:{abs(hedge_ratio):.2f}")
    
    return {
        'beta_comparison': beta_df,
        'ag_ni_signals': ag_ni_signals if len(ag_ni_signals) > 0 else None,
        'ag_ni_coint': ag_ni_coint
    }

if __name__ == "__main__":
    results = debug_signal_analysis()