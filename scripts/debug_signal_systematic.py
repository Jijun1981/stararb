#!/usr/bin/env python3
"""
系统性排查信号数量问题
按照专家建议的5个关键点逐一检查
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.signal_generation import SignalGenerator
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def debug_data_format_and_alignment():
    """1. 检查列名与对齐"""
    print("=" * 80)
    print("1. 检查列名与对齐")
    print("=" * 80)
    
    # 加载数据
    au_data = load_symbol_data('AU')
    zn_data = load_symbol_data('ZN')
    
    print(f"AU数据列名: {list(au_data.columns)}")
    print(f"ZN数据列名: {list(zn_data.columns)}")
    print(f"AU数据形状: {au_data.shape}")
    print(f"ZN数据形状: {zn_data.shape}")
    print(f"AU日期范围: {au_data.index.min()} to {au_data.index.max()}")
    print(f"ZN日期范围: {zn_data.index.min()} to {zn_data.index.max()}")
    
    # 准备配对数据
    prices = pd.DataFrame({'AU': au_data['close'], 'ZN': zn_data['close']})
    prices = prices.dropna()
    
    # 检查配对数据格式
    pair_data = prices.reset_index()
    pair_data.rename(columns={'index': 'date'}, inplace=True)
    
    # 重要：确认方向 - AU应该是x(低波动)，ZN应该是y(高波动)
    recent_data = pair_data[pair_data['date'] >= '2024-01-01']
    au_vol = recent_data['AU'].pct_change().std()
    zn_vol = recent_data['ZN'].pct_change().std()
    
    print(f"\n方向检查 (2024年至今波动率):")
    print(f"AU波动率: {au_vol:.6f}")
    print(f"ZN波动率: {zn_vol:.6f}")
    print(f"正确方向: {'AU(x)低波动, ZN(y)高波动' if au_vol < zn_vol else 'ZN(x)低波动, AU(y)高波动'}")
    
    # 按照正确方向设置
    if au_vol < zn_vol:
        pair_data = pair_data.rename(columns={'AU': 'x', 'ZN': 'y'})
        print("✓ 使用AU作为x，ZN作为y")
        symbol_x, symbol_y = 'AU', 'ZN'
    else:
        pair_data = pair_data.rename(columns={'ZN': 'x', 'AU': 'y'})
        print("✓ 使用ZN作为x，AU作为y")
        symbol_x, symbol_y = 'ZN', 'AU'
    
    print(f"最终pair_data列名: {list(pair_data.columns)}")
    print(f"最终pair_data形状: {pair_data.shape}")
    
    return pair_data, symbol_x, symbol_y, au_vol, zn_vol

def debug_cointegration_window_and_direction(pair_data, symbol_x, symbol_y):
    """2. 检查协整筛选窗口与方向"""
    print("\n" + "=" * 80)
    print("2. 检查协整筛选窗口与方向")
    print("=" * 80)
    
    # 读取协整结果
    coint_results = pd.read_csv("/mnt/e/Star-arb/cointegration_results.csv")
    
    # 查找AU-ZN配对
    au_zn_pairs = coint_results[
        ((coint_results['symbol_x'] == symbol_x) & (coint_results['symbol_y'] == symbol_y)) |
        ((coint_results['symbol_x'] == symbol_y) & (coint_results['symbol_y'] == symbol_x))
    ]
    
    print(f"协整结果中AU-ZN相关配对:")
    if len(au_zn_pairs) > 0:
        for _, row in au_zn_pairs.iterrows():
            print(f"  {row['pair']}: {row['symbol_x']}(x) -> {row['symbol_y']}(y), beta_1y={row['beta_1y']:.6f}")
            print(f"    5年p值: {row['pvalue_5y']:.6f}, 1年p值: {row['pvalue_1y']:.6f}")
            
        # 使用第一个匹配的配对
        selected = au_zn_pairs.iloc[0]
        print(f"\n✓ 选择配对: {selected['pair']}")
        
        # 检查方向一致性
        if selected['symbol_x'] == symbol_x and selected['symbol_y'] == symbol_y:
            print("✓ 方向一致")
            beta_coint = selected['beta_1y']
        else:
            print("⚠ 方向不一致，需要调整beta符号")
            beta_coint = -selected['beta_1y']
    else:
        print("❌ 未找到AU-ZN配对")
        beta_coint = -0.828799  # 使用2023年估计的值
    
    return beta_coint

def debug_convergence_signal_boundaries(pair_data):
    """3. 检查收敛期与信号期边界"""
    print("\n" + "=" * 80)
    print("3. 检查收敛期与信号期边界")
    print("=" * 80)
    
    convergence_end = pd.Timestamp('2024-06-30')
    signal_start = pd.Timestamp('2024-07-01')
    
    # 边界处理检查
    convergence_mask = pair_data['date'] <= convergence_end
    signal_mask = pair_data['date'] >= signal_start
    
    convergence_count = convergence_mask.sum()
    signal_count = signal_mask.sum()
    
    print(f"收敛期数据点: {convergence_count}个 (≤ {convergence_end.strftime('%Y-%m-%d')})")
    print(f"信号期数据点: {signal_count}个 (≥ {signal_start.strftime('%Y-%m-%d')})")
    
    # 检查边界日期
    boundary_data = pair_data[
        (pair_data['date'] >= '2024-06-28') & 
        (pair_data['date'] <= '2024-07-03')
    ]
    
    print(f"\n边界日期数据:")
    for _, row in boundary_data.iterrows():
        date = row['date']
        phase = 'convergence' if date <= convergence_end else 'signal'
        print(f"  {date.strftime('%Y-%m-%d')}: {phase}期")
    
    return convergence_count, signal_count

def debug_zscore_window_and_protection(pair_data, beta_initial):
    """4. 检查Z-score窗口与保护"""
    print("\n" + "=" * 80)
    print("4. 检查Z-score窗口与保护机制")
    print("=" * 80)
    
    window = 60
    signal_start = pd.Timestamp('2024-07-01')
    
    # 模拟Z-score计算
    signal_data = pair_data[pair_data['date'] >= signal_start].copy()
    print(f"信号期数据点: {len(signal_data)}个")
    
    if len(signal_data) < window:
        print(f"❌ 信号期数据不足window要求: {len(signal_data)} < {window}")
        return
    
    # 计算残差
    residuals = []
    log_x = np.log(signal_data['x'])
    log_y = np.log(signal_data['y'])
    
    for i in range(len(signal_data)):
        residual = log_y.iloc[i] - beta_initial * log_x.iloc[i]
        residuals.append(residual)
    
    residuals = np.array(residuals)
    
    # Z-score计算与保护检查
    valid_zscores = 0
    protected_by_std = 0
    protected_by_sample = 0
    extreme_zscores = 0
    
    for i in range(window, len(residuals)):
        window_residuals = residuals[i-window+1:i+1]  # 包含当前点的window个点
        
        # 样本数检查
        if len(window_residuals) < window:
            protected_by_sample += 1
            continue
            
        # 计算统计量
        mean = np.mean(window_residuals)
        std = np.std(window_residuals, ddof=0)  # ddof=0
        
        # std保护
        if std < 1e-6:
            protected_by_std += 1
            continue
            
        z_score = (residuals[i] - mean) / std
        valid_zscores += 1
        
        if abs(z_score) >= 2.2:
            extreme_zscores += 1
    
    print(f"Z-score统计:")
    print(f"  有效Z-score计算: {valid_zscores}次")
    print(f"  std过小保护: {protected_by_std}次")
    print(f"  样本不足保护: {protected_by_sample}次")
    print(f"  |Z|>=2.2的次数: {extreme_zscores}次")
    print(f"  极端Z-score比例: {extreme_zscores/max(valid_zscores,1)*100:.2f}%")
    
    # 检查ddof差异
    if valid_zscores > 0:
        # 取最后一个窗口比较ddof=0 vs ddof=1
        window_residuals = residuals[-window:]
        mean = np.mean(window_residuals)
        std_ddof0 = np.std(window_residuals, ddof=0)
        std_ddof1 = np.std(window_residuals, ddof=1)
        current_residual = residuals[-1]
        
        z_ddof0 = (current_residual - mean) / std_ddof0
        z_ddof1 = (current_residual - mean) / std_ddof1
        
        print(f"\nddof影响检查 (最后一个窗口):")
        print(f"  ddof=0: std={std_ddof0:.6f}, Z={z_ddof0:.4f}")
        print(f"  ddof=1: std={std_ddof1:.6f}, Z={z_ddof1:.4f}")
        print(f"  差异: {abs(z_ddof0 - z_ddof1):.4f}")
    
    return extreme_zscores, valid_zscores

def debug_filters_and_event_mapping(pair_data, beta_initial):
    """5. 检查过滤与事件映射"""
    print("\n" + "=" * 80)
    print("5. 检查过滤与事件映射")
    print("=" * 80)
    
    # 运行实际的信号生成
    generator = SignalGenerator(window=60, z_open=2.2, z_close=0.3, max_holding_days=30)
    
    signals = generator.process_pair_signals(
        pair_data=pair_data,
        initial_beta=beta_initial,
        convergence_end='2024-06-30',
        signal_start='2024-07-01',
        hist_start='2024-01-01',
        hist_end='2025-08-20',
        pair_info={'pair': 'AU-ZN', 'x': 'AU', 'y': 'ZN'}
    )
    
    # 统计各类事件
    signal_counts = signals['signal'].value_counts()
    phase_counts = signals['phase'].value_counts()
    
    print(f"信号类型分布:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")
    
    print(f"\n阶段分布:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count}")
    
    # 检查信号期的具体情况
    signal_period = signals[signals['phase'] == 'signal_period']
    if len(signal_period) > 0:
        print(f"\n信号期详细分析:")
        print(f"  总记录数: {len(signal_period)}")
        
        # 检查各种过滤条件
        valid_z = signal_period['z_score'].notna()
        extreme_z = abs(signal_period['z_score']) >= 2.2
        
        print(f"  有效Z-score: {valid_z.sum()}")
        print(f"  |Z|>=2.2: {extreme_z.sum()}")
        
        # 检查开仓被过滤的情况
        extreme_but_not_open = signal_period[extreme_z & ~signal_period['signal'].isin(['open_long', 'open_short'])]
        if len(extreme_but_not_open) > 0:
            print(f"  极端Z但未开仓: {len(extreme_but_not_open)}次")
            print(f"    原因分布: {extreme_but_not_open['signal'].value_counts().to_dict()}")
    
    return signals

def main():
    """主函数：系统性排查"""
    print("系统性排查信号数量问题")
    
    # 1. 检查数据格式与对齐
    pair_data, symbol_x, symbol_y, au_vol, zn_vol = debug_data_format_and_alignment()
    
    # 2. 检查协整方向
    beta_coint = debug_cointegration_window_and_direction(pair_data, symbol_x, symbol_y)
    
    # 使用2023年数据重新估计beta
    data_2023 = pair_data[(pair_data['date'] >= '2023-01-01') & (pair_data['date'] <= '2023-12-31')]
    if len(data_2023) > 60:
        x_2023 = np.log(data_2023['x'])
        y_2023 = np.log(data_2023['y'])
        X = add_constant(x_2023)
        model = OLS(y_2023, X).fit()
        beta_2023 = model.params[1]
        print(f"\n2023年OLS beta: {beta_2023:.6f}")
        print(f"协整分析beta: {beta_coint:.6f}")
        print(f"beta差异: {abs(beta_2023 - beta_coint):.6f}")
    else:
        beta_2023 = beta_coint
        print(f"\n使用协整beta: {beta_coint:.6f}")
    
    # 3. 检查边界处理
    convergence_count, signal_count = debug_convergence_signal_boundaries(pair_data)
    
    # 4. 检查Z-score保护
    extreme_zscores, valid_zscores = debug_zscore_window_and_protection(pair_data, beta_2023)
    
    # 5. 检查过滤与映射
    signals = debug_filters_and_event_mapping(pair_data, beta_2023)
    
    # 最终总结
    print("\n" + "=" * 80)
    print("排查总结")
    print("=" * 80)
    print(f"数据点: 收敛期{convergence_count}个，信号期{signal_count}个")
    print(f"Z-score: {extreme_zscores}/{valid_zscores} 次达到开仓阈值")
    print(f"实际开仓: {len(signals[signals['signal'].isin(['open_long', 'open_short'])])} 次")
    open_count = len(signals[signals['signal'].isin(['open_long', 'open_short'])])
    efficiency = open_count / max(extreme_zscores, 1) * 100
    print(f"信号效率: {open_count}/{extreme_zscores} = {efficiency:.1f}%")

if __name__ == '__main__':
    main()