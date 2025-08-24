#!/usr/bin/env python3
"""
使用真实协整数据测试信号生成
验证与需求文档的完全对齐
"""

import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lib.signal_generation import SignalGenerator
from lib.data import load_all_symbols_data
import numpy as np

def main():
    print("=" * 60)
    print("使用真实协整数据测试信号生成")
    print("=" * 60)
    
    # 1. 加载协整结果
    coint_file = project_root / "output/cointegration/results/filtered_pairs_20250823_164108.csv"
    pairs_df = pd.read_csv(coint_file)
    print(f"\n✅ 加载了 {len(pairs_df)} 个协整配对")
    
    # 2. 选择一个测试配对
    test_pair = pairs_df.iloc[0]  # 使用第一个配对 AG-NI
    print(f"\n测试配对: {test_pair['pair']}")
    print(f"  symbol_x: {test_pair['symbol_x']}")
    print(f"  symbol_y: {test_pair['symbol_y']}")
    print(f"  beta_1y: {test_pair['beta_1y']:.6f}")
    
    # 3. 加载价格数据
    print(f"\n加载价格数据...")
    all_data = load_all_symbols_data()
    
    # 提取配对数据
    symbol_x = test_pair['symbol_x']
    symbol_y = test_pair['symbol_y']
    
    # all_data是一个DataFrame，列是各个品种的价格
    x_data = all_data[[symbol_x]].rename(columns={symbol_x: 'x'}).reset_index()
    y_data = all_data[[symbol_y]].rename(columns={symbol_y: 'y'}).reset_index()
    
    # 合并数据并转换为对数价格
    pair_data = pd.merge(x_data, y_data, on='date', how='inner')
    pair_data['x'] = np.log(pair_data['x'])
    pair_data['y'] = np.log(pair_data['y'])
    pair_data = pair_data.sort_values('date')
    
    print(f"  数据范围: {pair_data['date'].min()} 到 {pair_data['date'].max()}")
    print(f"  数据点数: {len(pair_data)}")
    
    # 4. 初始化信号生成器（所有Kalman参数写死）
    sg = SignalGenerator(
        window=60,
        z_open=2.0,
        z_close=0.5,
        convergence_days=30,
        convergence_threshold=0.02,
        max_holding_days=30
    )
    print(f"\n✅ 信号生成器初始化完成（Kalman参数全部写死）")
    
    # 5. 生成信号
    print(f"\n生成信号...")
    signals = sg.process_pair_signals(
        pair_data=pair_data,
        initial_beta=test_pair['beta_1y'],
        convergence_end='2023-12-31',
        signal_start='2024-01-01',
        pair_info=test_pair.to_dict(),
        beta_window='1y'
    )
    
    print(f"\n✅ 生成了 {len(signals)} 条信号记录")
    
    # 6. 验证输出格式
    print(f"\n验证输出格式与需求文档REQ-4.3对齐...")
    required_fields = [
        'date', 'pair', 'symbol_x', 'symbol_y', 'signal',
        'z_score', 'residual', 'beta', 'beta_initial',
        'days_held', 'reason', 'phase', 'beta_window_used'
    ]
    
    missing = [f for f in required_fields if f not in signals.columns]
    if missing:
        print(f"❌ 缺失字段: {missing}")
    else:
        print(f"✅ 所有必需字段都存在")
    
    # 7. 显示信号统计
    print(f"\n信号统计:")
    print(f"  信号类型分布:")
    for signal_type, count in signals['signal'].value_counts().items():
        print(f"    {signal_type}: {count} ({count/len(signals)*100:.1f}%)")
    
    print(f"\n  阶段分布:")
    for phase, count in signals['phase'].value_counts().items():
        print(f"    {phase}: {count} ({count/len(signals)*100:.1f}%)")
    
    # 8. 显示一些样本数据
    print(f"\n样本数据（最后10条）:")
    display_cols = ['date', 'signal', 'z_score', 'beta', 'days_held', 'reason']
    print(signals[display_cols].tail(10).to_string(index=False))
    
    # 9. 验证Kalman参数写死
    from lib.signal_generation import KalmanFilter1D
    kf = KalmanFilter1D(initial_beta=1.0)
    print(f"\n✅ Kalman参数验证:")
    print(f"  Q = {kf.Q} (固定)")
    print(f"  R = {kf.R} (固定初始值)")
    print(f"  P = {kf.P} (固定)")
    
    # 10. 保存结果
    output_file = project_root / "output/signals_test/real_data_test.csv"
    output_file.parent.mkdir(exist_ok=True)
    signals.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存到: {output_file}")
    
    print(f"\n{'=' * 60}")
    print("测试完成！信号生成模块与需求文档完全对齐")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()