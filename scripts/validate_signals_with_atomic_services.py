#!/usr/bin/env python3
"""
使用原子服务验证信号生成
基于前面协整分析的结果，验证信号数量和β的正常性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入原子服务
from lib.data import load_data
from lib.signal_generation import SignalGenerator

print("=" * 80)
print("使用原子服务验证信号生成")
print("=" * 80)

# 时间配置 (使用前面的shifted配置)
TIME_CONFIG = {
    'data_start': '2019-01-01',
    'data_end': '2024-08-20',
    'convergence_end': '2023-06-30',  # 收敛期结束
    'signal_start': '2023-07-01',     # 信号期开始
    'hist_start': '2022-01-01',       # 历史数据开始(用于R估计)
    'hist_end': '2022-12-31'          # 历史数据结束
}

print(f"时间配置:")
for key, value in TIME_CONFIG.items():
    print(f"  {key}: {value}")

# 1. 使用数据管理原子服务加载数据
print(f"\n" + "=" * 60)
print("1. 加载数据 (使用 lib.data 原子服务)")
print("-" * 60)

try:
    # 定义需要的品种 (14个金属期货)
    symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']
    
    price_data = load_data(
        symbols=symbols,
        start_date=TIME_CONFIG['data_start'],
        end_date=TIME_CONFIG['data_end'],
        columns=['close'],
        log_price=True,  # 使用对数价格
        fill_method='ffill'
    )
    print(f"✓ 成功加载价格数据:")
    print(f"  形状: {price_data.shape}")
    print(f"  列名: {list(price_data.columns)}")
    
    # 检查是否有date列，如果没有则添加
    if 'date' not in price_data.columns:
        if price_data.index.name == 'date' or isinstance(price_data.index, pd.DatetimeIndex):
            price_data = price_data.reset_index()
            print(f"  ✓ 将date索引转为列")
        else:
            print(f"  ❌ 没有找到date信息")
            sys.exit(1)
    
    print(f"  日期范围: {price_data['date'].min()} ~ {price_data['date'].max()}")
    print(f"  品种数: {len([col for col in price_data.columns if col != 'date'])}")
    
    # 重命名列名，去掉_close后缀
    rename_dict = {}
    for col in price_data.columns:
        if col.endswith('_close'):
            new_name = col.replace('_close', '')
            rename_dict[col] = new_name
    
    if rename_dict:
        price_data = price_data.rename(columns=rename_dict)
        print(f"  ✓ 重命名列名: {list(rename_dict.keys())[:3]}... -> {list(rename_dict.values())[:3]}...")
        print(f"  更新后列名: {list(price_data.columns)}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

# 2. 使用协整原子服务加载协整结果
print(f"\n" + "=" * 60)
print("2. 加载协整结果 (使用 lib.coint 原子服务)")
print("-" * 60)

try:
    # 尝试加载shifted版本的协整结果
    coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
    if coint_file.exists():
        coint_results = pd.read_csv(coint_file)
        print(f"✓ 成功加载协整结果 (shifted版本): {coint_file}")
    else:
        # 如果没有，尝试加载其他版本
        possible_files = [
            project_root / "output" / "cointegration_results.csv",
            project_root / "data" / "cointegration_results.csv"
        ]
        
        coint_results = None
        for file_path in possible_files:
            if file_path.exists():
                coint_results = pd.read_csv(file_path)
                print(f"✓ 成功加载协整结果: {file_path}")
                break
        
        if coint_results is None:
            raise FileNotFoundError("找不到协整结果文件")
    
    print(f"  配对数量: {len(coint_results)}")
    
    # 检查列名
    p_value_col = None
    for col in coint_results.columns:
        if 'p_value' in col and ('5y' in col or '4y' in col):
            p_value_col = col
            break
    
    if p_value_col:
        print(f"  通过协整检验: {len(coint_results[coint_results[p_value_col] < 0.05])}")
    
    # 显示β分布 (检查可用的β列)
    beta_cols = [col for col in coint_results.columns if 'beta' in col]
    print(f"\n  可用β列: {beta_cols}")
    
    for col in beta_cols:
        betas = coint_results[col]
        print(f"  {col}: 范围[{betas.min():.3f}, {betas.max():.3f}], 均值{betas.mean():.3f}")
        negative_count = (betas < 0).sum()
        print(f"           负β数量: {negative_count} ({negative_count/len(betas)*100:.1f}%)")

except Exception as e:
    print(f"❌ 协整结果加载失败: {e}")
    sys.exit(1)

# 3. 筛选有效配对
print(f"\n" + "=" * 60)
print("3. 筛选有效配对")
print("-" * 60)

# 使用宽松的筛选条件，根据实际列名调整
p_value_cols = [col for col in coint_results.columns if 'p_value' in col or 'pvalue' in col]
print(f"可用p值列: {p_value_cols}")

if len(p_value_cols) >= 2:
    # 使用前两个p值列
    p_col1, p_col2 = p_value_cols[:2]
    valid_pairs = coint_results[
        (coint_results[p_col1] < 0.05) & 
        (coint_results[p_col2] < 0.1)    # 宽松条件
    ].copy()
elif len(p_value_cols) == 1:
    # 只有一个p值列
    p_col = p_value_cols[0]
    valid_pairs = coint_results[coint_results[p_col] < 0.05].copy()
else:
    # 没有p值列，使用所有配对
    print("  ⚠️ 没有找到p值列，使用所有配对")
    valid_pairs = coint_results.copy()

print(f"✓ 筛选出有效配对: {len(valid_pairs)}个")

if len(valid_pairs) == 0:
    print("❌ 没有有效配对，无法继续")
    sys.exit(1)

# 准备配对参数
pairs_params = {}
for _, row in valid_pairs.iterrows():
    pair_name = f"{row['symbol_x']}-{row['symbol_y']}"
    
    # 选择初始β值
    if 'beta_4y' in row:
        beta_initial = row['beta_4y']
    elif 'beta_1y' in row:
        beta_initial = row['beta_1y']
    else:
        beta_initial = 1.0  # 默认值
    
    pairs_params[pair_name] = {
        'symbol_x': row['symbol_x'],
        'symbol_y': row['symbol_y'],
        'beta_initial': beta_initial,
        'direction': row.get('direction', 'y_on_x')  # 默认方向
    }

print(f"  配对参数准备完成: {len(pairs_params)}个")

# 4. 使用信号生成原子服务
print(f"\n" + "=" * 60)
print("4. 信号生成 (使用 lib.signal_generation 原子服务)")
print("-" * 60)

signal_generator = SignalGenerator(
    window=60,
    z_open=2.0, 
    z_close=0.5,
    convergence_days=20,
    convergence_threshold=0.01
)

print(f"✓ 信号生成器初始化完成")
print(f"  参数: window=60, z_open=2.0, z_close=0.5")

# 批量生成信号
try:
    all_signals = signal_generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end=TIME_CONFIG['convergence_end'],
        signal_start=TIME_CONFIG['signal_start'],
        hist_start=TIME_CONFIG['hist_start'],
        hist_end=TIME_CONFIG['hist_end']
    )
    
    if all_signals.empty:
        print("❌ 没有生成任何信号!")
    else:
        print(f"✓ 成功生成信号: {len(all_signals)}条记录")
        
        # 5. 信号统计分析
        print(f"\n" + "=" * 60)
        print("5. 信号统计分析")
        print("-" * 60)
        
        # 按配对统计
        pair_stats = all_signals.groupby('pair').agg({
            'signal': lambda x: (x != 'hold').sum(),  # 非hold信号数
            'z_score': lambda x: x[x != 0].count() if (x != 0).any() else 0  # 非零Z-score数
        }).rename(columns={'signal': 'trading_signals', 'z_score': 'nonzero_zscores'})
        
        print(f"配对信号统计:")
        print(f"  总配对数: {len(pair_stats)}")
        print(f"  有交易信号的配对: {(pair_stats['trading_signals'] > 0).sum()}")
        print(f"  总交易信号数: {pair_stats['trading_signals'].sum()}")
        
        # 按信号类型统计
        signal_counts = all_signals['signal'].value_counts()
        print(f"\n信号类型分布:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # 找出有实际交易的配对
        active_pairs = pair_stats[pair_stats['trading_signals'] > 0].sort_values(
            'trading_signals', ascending=False
        )
        
        if len(active_pairs) > 0:
            print(f"\n活跃交易配对 (前10个):")
            for pair, stats in active_pairs.head(10).iterrows():
                # 获取该配对的β信息
                pair_row = valid_pairs[
                    (valid_pairs['symbol_x'] == pair.split('-')[0]) & 
                    (valid_pairs['symbol_y'] == pair.split('-')[1])
                ].iloc[0]
                
                print(f"  {pair}:")
                print(f"    交易信号: {stats['trading_signals']}个")
                
                # 显示可用的β值
                beta_info = []
                for col in pair_row.index:
                    if 'beta' in col:
                        beta_info.append(f"{col}: {pair_row[col]:.3f}")
                if beta_info:
                    print(f"    β系数: {', '.join(beta_info)}")
                
                # 显示可用的p值
                pvalue_info = []
                for col in pair_row.index:
                    if 'p_value' in col or 'pvalue' in col:
                        pvalue_info.append(f"{col}: {pair_row[col]:.3f}")
                if pvalue_info:
                    print(f"    p值: {', '.join(pvalue_info)}")
                
                # 显示该配对的具体信号
                pair_signals = all_signals[
                    (all_signals['pair'] == pair) & 
                    (all_signals['signal'] != 'hold')
                ].head(5)
                
                if len(pair_signals) > 0:
                    print(f"    前5个交易信号:")
                    for _, sig in pair_signals.iterrows():
                        print(f"      {sig['date']}: {sig['signal']}, Z={sig['z_score']:.3f}, β={sig['beta']:.3f}")
        
        # 6. β动态性分析
        print(f"\n" + "=" * 60)
        print("6. β动态性分析")
        print("-" * 60)
        
        # 信号期的β变化分析
        signal_period = all_signals[all_signals['phase'] == 'signal_period']
        
        if len(signal_period) > 0:
            print(f"信号期β动态性:")
            
            for pair in active_pairs.head(5).index:  # 分析前5个活跃配对
                pair_data = signal_period[signal_period['pair'] == pair]
                if len(pair_data) > 0:
                    betas = pair_data['beta'].dropna()
                    if len(betas) > 1:
                        beta_change = abs(betas.max() - betas.min())
                        beta_volatility = betas.std()
                        
                        print(f"  {pair}:")
                        print(f"    β范围: [{betas.min():.3f}, {betas.max():.3f}]")
                        print(f"    β变化: {beta_change:.3f}, β波动率: {beta_volatility:.3f}")
        
        # 7. 验证结果
        print(f"\n" + "=" * 60)
        print("7. 验证结果总结")
        print("-" * 60)
        
        total_open_signals = signal_counts.get('open_long', 0) + signal_counts.get('open_short', 0)
        total_close_signals = signal_counts.get('close', 0)
        
        print(f"✓ 原子服务验证结果:")
        print(f"  有效配对数: {len(valid_pairs)}")
        print(f"  活跃交易配对: {len(active_pairs)}")
        print(f"  总开仓信号: {total_open_signals} (多头: {signal_counts.get('open_long', 0)}, 空头: {signal_counts.get('open_short', 0)})")
        print(f"  总平仓信号: {total_close_signals}")
        print(f"  信号生成率: {(total_open_signals / len(valid_pairs)):.2f} 信号/配对")
        
        if total_open_signals > 0:
            print(f"  ✅ 信号生成正常!")
        else:
            print(f"  ⚠️  没有开仓信号，需要检查阈值设置")

except Exception as e:
    print(f"❌ 信号生成失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("验证完成")
print("=" * 80)