#!/usr/bin/env python3
"""
验证β值修复效果和完整的信号逻辑
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression

def verify_beta_and_signal_logic():
    """验证β值修复效果和信号逻辑"""
    
    print("🔍 验证β值修复效果和完整信号逻辑")
    print("=" * 60)
    
    # 1. 加载最新信号数据
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if signal_files:
        latest_signal_file = max(signal_files)
        signals_df = pd.read_csv(latest_signal_file)
        print(f"📊 分析信号文件: {latest_signal_file}")
        
        # 查看NI-AG配对
        ni_ag_signals = signals_df[signals_df['pair'] == 'NI-AG']
        if len(ni_ag_signals) > 0:
            print(f"\n=== NI-AG配对信号分析 ===")
            print(f"信号数量: {len(ni_ag_signals)}")
            print(f"symbol_x: {ni_ag_signals['symbol_x'].iloc[0]}")
            print(f"symbol_y: {ni_ag_signals['symbol_y'].iloc[0]}")
            print(f"beta_initial: {ni_ag_signals['beta_initial'].iloc[0]:.6f}")
            
            # 分析β值变化
            beta_values = ni_ag_signals['beta'].values
            print(f"β值范围: {min(beta_values):.6f} 至 {max(beta_values):.6f}")
            print(f"β值变化: {max(beta_values) - min(beta_values):.6f}")
            
            # 检查实际交易信号
            trade_signals = ni_ag_signals[ni_ag_signals['signal'].str.contains('open_|holding_')]
            if len(trade_signals) > 0:
                print(f"\n实际交易信号: {len(trade_signals)}个")
                
                # 显示前几个交易信号
                for i, (_, signal) in enumerate(trade_signals.head(5).iterrows()):
                    print(f"  {i+1}. {signal['date']}: {signal['signal']}, "
                          f"Z={signal['z_score']:.3f}, β={signal['beta']:.6f}")
        else:
            print("❌ 未找到NI-AG配对信号")
    else:
        print("❌ 未找到信号文件")
        return
    
    # 2. 验证预热期β值的合理性
    print(f"\n=== 验证预热期β值的合理性 ===")
    
    data = load_all_symbols_data()
    
    # 模拟信号生成的数据范围和预热期
    signal_data = data['2024-02-08':'2025-08-20'].copy()
    ni_signal = signal_data['NI'].dropna()
    ag_signal = signal_data['AG'].dropna()
    
    # 对齐数据
    common_dates = ni_signal.index.intersection(ag_signal.index)
    ni_aligned = ni_signal[common_dates]
    ag_aligned = ag_signal[common_dates]
    
    # 预热期（前60天）
    ols_window = 60
    ni_warmup = ni_aligned[:ols_window]
    ag_warmup = ag_aligned[:ols_window]
    
    # 计算预热期β值
    reg_warmup = LinearRegression()
    reg_warmup.fit(ni_warmup.values.reshape(-1, 1), ag_warmup.values)
    beta_warmup = reg_warmup.coef_[0]
    
    print(f"理论预热期β值: {beta_warmup:.6f}")
    print(f"信号中初始β值: {ni_ag_signals['beta_initial'].iloc[0]:.6f}")
    print(f"差异: {abs(beta_warmup - ni_ag_signals['beta_initial'].iloc[0]):.6f}")
    
    if abs(beta_warmup - ni_ag_signals['beta_initial'].iloc[0]) < 0.01:
        print("✅ 修复成功：β值初始化合理")
    else:
        print("❌ β值初始化仍有问题")
    
    # 3. 验证Long/Short方向逻辑
    print(f"\n=== 验证Long/Short方向逻辑 ===")
    
    if len(trade_signals) > 0:
        for signal_type in ['open_long', 'open_short']:
            type_signals = trade_signals[trade_signals['signal'] == signal_type]
            if len(type_signals) > 0:
                z_scores = type_signals['z_score'].values
                print(f"{signal_type}: {len(type_signals)}个信号")
                print(f"  Z-score范围: {min(z_scores):.3f} 至 {max(z_scores):.3f}")
                print(f"  平均Z-score: {np.mean(z_scores):.3f}")
                
                # 检查方向逻辑是否正确
                if signal_type == 'open_long' and np.mean(z_scores) < -1:
                    print("  ✅ Long信号对应负Z-score，逻辑正确")
                elif signal_type == 'open_short' and np.mean(z_scores) > 1:
                    print("  ✅ Short信号对应正Z-score，逻辑正确")
                else:
                    print("  ⚠️ 信号方向可能有问题")
    
    # 4. 检查合约乘数和手数计算
    print(f"\n=== 检查合约乘数和手数计算 ===")
    
    # 期货合约乘数
    multipliers = {
        'NI': 1,   # 镍：1吨/手
        'AG': 15,  # 白银：15千克/手
    }
    
    print(f"NI合约乘数: {multipliers['NI']}")
    print(f"AG合约乘数: {multipliers['AG']}")
    
    # 理论对冲比例计算
    if len(ni_ag_signals) > 0:
        # 使用最新的β值
        latest_beta = ni_ag_signals['beta'].iloc[-1]
        
        # h* = β × (Py × My) / (Px × Mx)
        # 其中 NI是X，AG是Y
        hedge_ratio = latest_beta * (1 * multipliers['AG']) / (1 * multipliers['NI'])
        
        print(f"当前β值: {latest_beta:.6f}")
        print(f"理论对冲比例: {hedge_ratio:.6f}")
        print(f"NI:AG手数比例约为: 1:{abs(hedge_ratio):.2f}")
        
        # 实际交易中的手数分配建议
        if abs(hedge_ratio) > 50:
            print("⚠️ 对冲比例过高，可能需要调整")
        else:
            print("✅ 对冲比例在合理范围内")
    
    # 5. Z-score分布合理性检查
    print(f"\n=== Z-score分布合理性检查 ===")
    
    if len(ni_ag_signals) > 0:
        signal_period_signals = ni_ag_signals[ni_ag_signals['phase'] == 'signal_period']
        if len(signal_period_signals) > 0:
            z_scores = signal_period_signals['z_score'].values
            
            print(f"信号期Z-score统计:")
            print(f"  数量: {len(z_scores)}")
            print(f"  均值: {np.mean(z_scores):.3f}")
            print(f"  标准差: {np.std(z_scores):.3f}")
            print(f"  范围: [{min(z_scores):.3f}, {max(z_scores):.3f}]")
            print(f"  |Z|>2的比例: {np.mean(np.abs(z_scores) > 2):.1%}")
            
            if np.abs(np.mean(z_scores)) < 0.2 and 0.5 < np.std(z_scores) < 2.0:
                print("✅ Z-score分布合理")
            else:
                print("⚠️ Z-score分布可能异常")

import os

if __name__ == "__main__":
    verify_beta_and_signal_logic()