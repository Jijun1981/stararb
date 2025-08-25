#!/usr/bin/env python3
"""
修复信号生成问题的系统分析脚本
"""
import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
import os

def analyze_xy_assignment_problem():
    """分析X/Y分配问题"""
    print("=== 分析X/Y分配问题 ===")
    
    # 加载数据
    data = load_all_symbols_data()
    recent_data = data['2024-01-01':]
    
    # 计算所有品种的波动率
    volatilities = recent_data.std().sort_values()
    print("2024年至今各品种波动率（从低到高）:")
    for symbol, vol in volatilities.items():
        print(f"  {symbol}: {vol:.6f}")
    
    # 检查当前信号文件中的X/Y分配
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if signal_files:
        latest_signal_file = max(signal_files)
        signals_df = pd.read_csv(latest_signal_file)
        
        print(f"\n当前信号文件中的X/Y分配:")
        pairs_info = []
        for pair in signals_df['pair'].unique()[:10]:  # 检查前10个配对
            pair_data = signals_df[signals_df['pair'] == pair].iloc[0]
            symbol_x = pair_data['symbol_x']
            symbol_y = pair_data['symbol_y']
            
            vol_x = volatilities[symbol_x] if symbol_x in volatilities else None
            vol_y = volatilities[symbol_y] if symbol_y in volatilities else None
            
            correct_assignment = vol_x < vol_y if (vol_x and vol_y) else None
            
            pairs_info.append({
                'pair': pair,
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                'vol_x': vol_x,
                'vol_y': vol_y,
                'correct_assignment': correct_assignment
            })
        
        pairs_df = pd.DataFrame(pairs_info)
        print(pairs_df)
        
        # 统计错误分配
        wrong_assignments = pairs_df[pairs_df['correct_assignment'] == False]
        print(f"\n❌ 错误分配的配对数量: {len(wrong_assignments)} / {len(pairs_df)}")
        
        return pairs_df, volatilities
    
    return None, volatilities

def analyze_beta_calculation_logic():
    """分析β值计算逻辑"""
    print("\n=== 分析β值计算逻辑 ===")
    
    # 手动计算AG-NI的正确β值
    data = load_all_symbols_data()
    
    # 根据波动率正确分配：NI应该作为X（低波动率），AG应该作为Y（高波动率）
    ag_prices = data['AG'].dropna()
    ni_prices = data['NI'].dropna()
    
    # 对齐数据
    common_dates = ag_prices.index.intersection(ni_prices.index)
    ag_aligned = ag_prices[common_dates]
    ni_aligned = ni_prices[common_dates]
    
    print(f"数据对齐后长度: {len(common_dates)}")
    
    # 计算对数价格
    log_ag = np.log(ag_aligned)
    log_ni = np.log(ni_aligned)
    
    # 使用最近1年数据进行回归
    recent_start = '2024-01-01'
    recent_ag = log_ag[recent_start:]
    recent_ni = log_ni[recent_start:]
    
    if len(recent_ag) > 0 and len(recent_ni) > 0:
        # 情况1: 按当前分配 AG(X) vs NI(Y), β = Δlog(NI)/Δlog(AG)
        beta_current_assignment = np.cov(recent_ag, recent_ni)[0,1] / np.var(recent_ag)
        print(f"当前分配 AG(X)->NI(Y) 的β: {beta_current_assignment:.6f}")
        
        # 情况2: 正确分配 NI(X) vs AG(Y), β = Δlog(AG)/Δlog(NI) 
        beta_correct_assignment = np.cov(recent_ni, recent_ag)[0,1] / np.var(recent_ni)
        print(f"正确分配 NI(X)->AG(Y) 的β: {beta_correct_assignment:.6f}")
        
        # 使用OLS验证
        from sklearn.linear_model import LinearRegression
        
        # 当前分配的OLS
        reg_current = LinearRegression()
        reg_current.fit(recent_ag.values.reshape(-1,1), recent_ni.values)
        beta_ols_current = reg_current.coef_[0]
        print(f"当前分配 OLS β: {beta_ols_current:.6f}")
        
        # 正确分配的OLS
        reg_correct = LinearRegression()
        reg_correct.fit(recent_ni.values.reshape(-1,1), recent_ag.values)
        beta_ols_correct = reg_correct.coef_[0]
        print(f"正确分配 OLS β: {beta_ols_correct:.6f}")
        
        return {
            'beta_current': beta_current_assignment,
            'beta_correct': beta_correct_assignment,
            'beta_ols_current': beta_ols_current,
            'beta_ols_correct': beta_ols_correct
        }
    
    return None

def check_kalman_initialization_logic():
    """检查Kalman滤波初始化逻辑"""
    print("\n=== 检查Kalman滤波初始化逻辑 ===")
    
    # 检查signal_generation.py中的初始化逻辑
    try:
        with open('lib/signal_generation.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找关键的初始化代码段
        lines = content.split('\n')
        
        # 查找warm_up_ols方法
        in_warmup_method = False
        warmup_lines = []
        
        for i, line in enumerate(lines):
            if 'def warm_up_ols' in line:
                in_warmup_method = True
                warmup_lines.append(f"{i+1:4d}: {line}")
            elif in_warmup_method:
                if line.strip().startswith('def ') and 'warm_up_ols' not in line:
                    break
                warmup_lines.append(f"{i+1:4d}: {line}")
        
        print("warm_up_ols 方法关键代码:")
        for line in warmup_lines[:20]:  # 显示前20行
            print(line)
            
        # 检查β值符号处理
        if 'np.sign' in content or 'abs(' in content:
            print("\n⚠️ 发现符号处理相关代码")
        else:
            print("\n✅ 未发现符号强制处理代码")
            
    except Exception as e:
        print(f"❌ 无法读取signal_generation.py: {e}")

def main():
    """主函数"""
    print("🔍 系统分析信号生成问题")
    print("=" * 50)
    
    # 1. 分析X/Y分配问题
    pairs_df, volatilities = analyze_xy_assignment_problem()
    
    # 2. 分析β值计算逻辑
    beta_results = analyze_beta_calculation_logic()
    
    # 3. 检查Kalman初始化逻辑
    check_kalman_initialization_logic()
    
    # 4. 总结问题和解决方案
    print("\n" + "=" * 50)
    print("🎯 问题总结和解决方案:")
    
    if pairs_df is not None:
        wrong_count = len(pairs_df[pairs_df['correct_assignment'] == False])
        total_count = len(pairs_df)
        if wrong_count > 0:
            print(f"1. ❌ X/Y分配问题: {wrong_count}/{total_count} 配对分配错误")
            print("   解决方案: 修改协整分析阶段的配对逻辑，低波动率作为X")
        else:
            print("1. ✅ X/Y分配正确")
    
    if beta_results:
        print(f"2. ❌ β值计算问题: 当前={beta_results['beta_ols_current']:.6f}, 应该={beta_results['beta_ols_correct']:.6f}")
        print("   解决方案: 重新运行协整分析，使用正确的X/Y分配")
    
    print("3. ⚠️ Kalman滤波初始化: 需要确保使用正确的初始β值")
    print("   解决方案: 修复协整分析后重新生成信号")

if __name__ == "__main__":
    main()