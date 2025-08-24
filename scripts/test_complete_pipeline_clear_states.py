#!/usr/bin/env python3
"""
完整的两模块联调测试：协整+信号生成
使用清晰的状态机制和所有满足条件的配对

=== 测试条件说明 ===
1. 配对筛选条件：
   - 5年p值 < 0.05 AND 1年p值 < 0.05
   - 使用协整模块(lib.coint)的结果

2. 初始Beta估计：
   - 使用2023年(2023-01-01至2023-12-31)的数据
   - 用OLS回归: log(Y) = α + β*log(X) + ε
   - β作为Kalman滤波器的初始值

3. 时间配置：
   - 收敛期: 2024-01-01至2024-06-30 (Kalman收敛)
   - 信号期: 2024-07-01至2025-08-20 (生成交易信号)

4. 信号参数：
   - 滚动窗口: 60天
   - 开仓阈值: |Z| >= 2.2
   - 平仓阈值: |Z| <= 0.3
   - 最大持仓: 30天

5. Kalman参数：
   - Q = 1e-5 (过程噪声)
   - R = 1e-3 (观测噪声)
   - P0 = 0.1 (初始不确定性)

6. 清晰状态机制：
   - empty: 空仓等待
   - open_long/open_short: 当日开仓
   - holding_long/holding_short: 持仓中
   - close: 当日平仓
   - converging: 收敛期状态
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# 时间配置
TIME_CONFIG = {
    'beta_training_start': '2023-01-01',    # 2023年OLS训练开始
    'beta_training_end': '2023-12-31',      # 2023年OLS训练结束
    'convergence_start': '2024-01-01',      # Kalman收敛期开始
    'convergence_end': '2024-06-30',        # Kalman收敛期结束
    'signal_start': '2024-07-01',           # 信号生成开始
    'backtest_end': '2025-08-20'            # 信号生成结束
}

# 信号参数
SIGNAL_CONFIG = {
    'window': 60,           # 60天滚动窗口
    'z_open': 2.2,          # 开仓阈值
    'z_close': 0.3,         # 平仓阈值
    'max_holding_days': 30, # 最大持仓天数
}

# Kalman参数（已修复）
KALMAN_CONFIG = {
    'Q': 1e-5,              # 过程噪声
    'R': 1e-3,              # 观测噪声
    'P0': 0.1               # 初始不确定性
}

def load_cointegration_results():
    """
    加载协整分析结果并筛选满足条件的配对
    
    筛选条件：
    - pvalue_5y < 0.05 AND pvalue_1y < 0.05
    """
    print("1. 加载协整分析结果")
    print("=" * 60)
    
    coint_df = pd.read_csv("/mnt/e/Star-arb/cointegration_results.csv")
    print(f"总配对数: {len(coint_df)}")
    
    # 筛选条件：5年和1年p值都小于0.05
    filtered_df = coint_df[
        (coint_df['pvalue_5y'] < 0.05) & 
        (coint_df['pvalue_1y'] < 0.05)
    ]
    
    print(f"满足条件的配对数: {len(filtered_df)}")
    print("筛选条件: pvalue_5y < 0.05 AND pvalue_1y < 0.05")
    
    if len(filtered_df) > 0:
        print("\n前10个配对:")
        for i, (_, row) in enumerate(filtered_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['pair']}: "
                  f"5y_p={row['pvalue_5y']:.6f}, "
                  f"1y_p={row['pvalue_1y']:.6f}, "
                  f"1y_β={row['beta_1y']:.6f}")
    
    return filtered_df

def load_all_price_data(symbols):
    """
    加载所有需要的价格数据
    """
    print(f"\n2. 加载价格数据")
    print("=" * 60)
    
    data = {}
    for symbol in symbols:
        df = load_symbol_data(symbol)
        data[symbol] = df
        print(f"{symbol}: {len(df)}条记录, "
              f"{df.index.min().strftime('%Y-%m-%d')} 至 "
              f"{df.index.max().strftime('%Y-%m-%d')}")
    
    return data

def calculate_2023_ols_beta(pair_row, all_data):
    """
    使用2023年数据计算OLS beta作为Kalman初始值
    
    计算方法：
    - 时间范围：2023-01-01至2023-12-31
    - 回归方程：log(Y) = α + β*log(X) + ε
    - 返回β系数
    """
    symbol_x = pair_row['symbol_x']
    symbol_y = pair_row['symbol_y']
    
    # 获取2023年数据
    start_date = TIME_CONFIG['beta_training_start']
    end_date = TIME_CONFIG['beta_training_end']
    
    data_x = all_data[symbol_x]
    data_y = all_data[symbol_y]
    
    # 筛选2023年数据
    mask_2023 = (data_x.index >= start_date) & (data_x.index <= end_date)
    x_2023 = data_x[mask_2023]['close']
    y_2023 = data_y[mask_2023]['close']
    
    # 对齐数据（处理可能的缺失值）
    common_dates = x_2023.index.intersection(y_2023.index)
    x_aligned = x_2023.loc[common_dates]
    y_aligned = y_2023.loc[common_dates]
    
    if len(common_dates) < 60:  # 至少需要60个交易日
        print(f"  ⚠️  {pair_row['pair']}: 2023年数据不足({len(common_dates)}天)")
        return np.nan, len(common_dates)
    
    # OLS回归：log(Y) = α + β*log(X) + ε
    log_x = np.log(x_aligned)
    log_y = np.log(y_aligned)
    
    X = add_constant(log_x)
    model = OLS(log_y, X).fit()
    beta_2023 = model.params[1]
    
    print(f"  ✓ {pair_row['pair']}: "
          f"协整β={pair_row['beta_1y']:.6f}, "
          f"2023年β={beta_2023:.6f}, "
          f"数据点={len(common_dates)}")
    
    return beta_2023, len(common_dates)

def prepare_pair_data(symbol_x, symbol_y, all_data):
    """
    准备单个配对的价格数据，格式化为信号生成模块需要的格式
    
    返回格式：
    - DataFrame包含列：date, x, y
    - x是低波动品种，y是高波动品种
    - 已对齐并去除缺失值
    """
    data_x = all_data[symbol_x]['close']
    data_y = all_data[symbol_y]['close']
    
    # 对齐数据
    prices = pd.DataFrame({symbol_x: data_x, symbol_y: data_y})
    prices = prices.dropna()
    
    # 转换为信号生成模块需要的格式
    pair_data = prices.reset_index()
    pair_data.rename(columns={'index': 'date'}, inplace=True)
    
    # 根据协整分析结果确定x和y的映射
    # 这里假设协整分析已经正确确定了x和y的方向
    pair_data = pair_data.rename(columns={symbol_x: 'x', symbol_y: 'y'})
    
    return pair_data

def run_signal_generation_for_all_pairs(filtered_pairs, all_data):
    """
    为所有满足条件的配对运行信号生成
    """
    print(f"\n3. 运行信号生成")
    print("=" * 60)
    print(f"时间配置:")
    print(f"  收敛期: {TIME_CONFIG['convergence_start']} 至 {TIME_CONFIG['convergence_end']}")
    print(f"  信号期: {TIME_CONFIG['signal_start']} 至 {TIME_CONFIG['backtest_end']}")
    print(f"\n信号参数:")
    print(f"  滚动窗口: {SIGNAL_CONFIG['window']}天")
    print(f"  开仓阈值: |Z| >= {SIGNAL_CONFIG['z_open']}")
    print(f"  平仓阈值: |Z| <= {SIGNAL_CONFIG['z_close']}")
    print(f"  最大持仓: {SIGNAL_CONFIG['max_holding_days']}天")
    
    # 创建信号生成器
    generator = SignalGenerator(
        window=SIGNAL_CONFIG['window'],
        z_open=SIGNAL_CONFIG['z_open'],
        z_close=SIGNAL_CONFIG['z_close'],
        max_holding_days=SIGNAL_CONFIG['max_holding_days']
    )
    
    all_signals = []
    successful_pairs = 0
    failed_pairs = []
    
    print(f"\n开始处理{len(filtered_pairs)}个配对...")
    
    for i, (_, pair_row) in enumerate(filtered_pairs.iterrows()):
        pair_name = pair_row['pair']
        symbol_x = pair_row['symbol_x']
        symbol_y = pair_row['symbol_y']
        
        print(f"\n处理配对 {i+1}/{len(filtered_pairs)}: {pair_name}")
        
        try:
            # 1. 计算2023年OLS beta
            beta_2023, data_points = calculate_2023_ols_beta(pair_row, all_data)
            
            if np.isnan(beta_2023):
                failed_pairs.append((pair_name, "2023年数据不足"))
                continue
            
            # 2. 准备配对数据
            pair_data = prepare_pair_data(symbol_x, symbol_y, all_data)
            
            # 3. 生成信号
            signals = generator.process_pair_signals(
                pair_data=pair_data,
                initial_beta=beta_2023,  # 使用2023年OLS beta
                convergence_end=TIME_CONFIG['convergence_end'],
                signal_start=TIME_CONFIG['signal_start'],
                hist_start=TIME_CONFIG['convergence_start'],
                hist_end=TIME_CONFIG['backtest_end'],
                pair_info={
                    'pair': pair_name, 
                    'x': symbol_x, 
                    'y': symbol_y
                }
            )
            
            # 4. 只保留2023年以后的数据
            signals = signals[signals['date'] >= '2023-01-01']
            
            # 5. 添加配对信息到信号中
            signals['symbol_x'] = symbol_x
            signals['symbol_y'] = symbol_y
            signals['beta_2023'] = beta_2023
            signals['beta_coint'] = pair_row['beta_1y']
            
            all_signals.append(signals)
            successful_pairs += 1
            
            # 统计信号
            signal_counts = signals['signal'].value_counts()
            signal_period = signals[signals['phase'] == 'signal_period']
            open_signals = len(signals[signals['signal'].isin(['open_long', 'open_short'])])
            
            print(f"  ✓ 生成{len(signals)}条记录，信号期{len(signal_period)}条")
            print(f"    开仓信号: {open_signals}个")
            if len(signal_period) > 0:
                extreme_z = len(signal_period[abs(signal_period['z_score']) >= SIGNAL_CONFIG['z_open']])
                print(f"    极端Z值: {extreme_z}次 (|Z|>={SIGNAL_CONFIG['z_open']})")
                
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            failed_pairs.append((pair_name, str(e)))
    
    # 合并所有信号
    if all_signals:
        combined_signals = pd.concat(all_signals, ignore_index=True)
    else:
        combined_signals = pd.DataFrame()
    
    print(f"\n处理结果统计:")
    print(f"  成功配对: {successful_pairs}")
    print(f"  失败配对: {len(failed_pairs)}")
    
    if failed_pairs:
        print("失败详情:")
        for pair_name, reason in failed_pairs[:5]:  # 只显示前5个
            print(f"    {pair_name}: {reason}")
    
    return combined_signals

def analyze_results(combined_signals):
    """
    分析信号生成结果
    """
    print(f"\n4. 结果分析")
    print("=" * 60)
    
    if len(combined_signals) == 0:
        print("没有生成任何信号")
        return
    
    print(f"总信号记录: {len(combined_signals)}")
    
    # 按阶段统计
    phase_counts = combined_signals['phase'].value_counts()
    print(f"\n阶段分布:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count}")
    
    # 信号类型分布
    signal_counts = combined_signals['signal'].value_counts()
    print(f"\n信号类型分布:")
    for signal_type, count in signal_counts.items():
        print(f"  {signal_type}: {count}")
    
    # 信号期详细分析
    signal_period = combined_signals[combined_signals['phase'] == 'signal_period']
    if len(signal_period) > 0:
        print(f"\n信号期详细分析:")
        print(f"  信号期总记录: {len(signal_period)}")
        
        # 按配对统计开仓信号
        open_signals = signal_period[signal_period['signal'].isin(['open_long', 'open_short'])]
        if len(open_signals) > 0:
            pair_open_counts = open_signals['pair'].value_counts()
            print(f"  开仓信号总数: {len(open_signals)}")
            print(f"  开仓配对数: {len(pair_open_counts)}")
            print("  开仓最多的配对:")
            for pair, count in pair_open_counts.head(5).items():
                print(f"    {pair}: {count}次")
        
        # 极端Z值分析
        extreme_z = signal_period[abs(signal_period['z_score']) >= SIGNAL_CONFIG['z_open']]
        print(f"  极端Z值次数: {len(extreme_z)} (|Z|>={SIGNAL_CONFIG['z_open']})")
        
        if len(extreme_z) > 0:
            extreme_by_signal = extreme_z['signal'].value_counts()
            print("  极端Z值时的信号分布:")
            for signal, count in extreme_by_signal.items():
                print(f"    {signal}: {count}")
        
        # 状态机制效果分析
        print(f"\n状态机制效果:")
        holding_signals = signal_period[signal_period['signal'].isin(['holding_long', 'holding_short'])]
        print(f"  持仓状态记录: {len(holding_signals)}")
        empty_signals = signal_period[signal_period['signal'] == 'empty']
        print(f"  空仓状态记录: {len(empty_signals)}")
        
        # 持仓期间的极端Z值分析
        extreme_while_holding = extreme_z[extreme_z['signal'].isin(['holding_long', 'holding_short'])]
        print(f"  持仓期间极端Z值: {len(extreme_while_holding)}次")
        print(f"  空仓期间极端Z值: {len(extreme_z) - len(extreme_while_holding)}次")
    
    return combined_signals

def export_results(combined_signals):
    """
    导出结果到CSV文件
    """
    if len(combined_signals) == 0:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"/mnt/e/Star-arb/signals_complete_pipeline_{timestamp}.csv"
    
    combined_signals.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已导出: {csv_path}")
    
    return csv_path

def main():
    """
    主函数：完整的两模块联调测试
    """
    print("完整的两模块联调测试：协整+信号生成")
    print("使用清晰状态机制和所有满足条件的配对")
    print("=" * 80)
    
    try:
        # 1. 加载协整分析结果
        filtered_pairs = load_cointegration_results()
        
        if len(filtered_pairs) == 0:
            print("没有满足条件的配对，退出")
            return
        
        # 获取所有需要的品种
        all_symbols = set(filtered_pairs['symbol_x'].tolist() + 
                         filtered_pairs['symbol_y'].tolist())
        
        # 2. 加载价格数据
        all_data = load_all_price_data(sorted(all_symbols))
        
        # 3. 运行信号生成
        combined_signals = run_signal_generation_for_all_pairs(filtered_pairs, all_data)
        
        # 4. 分析结果
        analyze_results(combined_signals)
        
        # 5. 导出结果
        export_results(combined_signals)
        
        print(f"\n测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()