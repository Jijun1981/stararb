#!/usr/bin/env python3
"""
基于信号验证结果进行回测
使用原子服务进行完整的配对交易回测
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
from lib.backtest import BacktestEngine

print("=" * 80)
print("配对交易回测 - 基于信号验证结果")
print("=" * 80)

# 时间配置
TIME_CONFIG = {
    'data_start': '2019-01-01',
    'data_end': '2024-08-20',
    'convergence_end': '2023-06-30',  # 收敛期结束
    'signal_start': '2023-07-01',     # 信号期开始
    'backtest_start': '2023-07-01',   # 回测开始
    'hist_start': '2022-01-01',       # 历史数据开始
    'hist_end': '2022-12-31'          # 历史数据结束
}

# 回测配置
BACKTEST_CONFIG = {
    'initial_capital': 5000000,  # 500万初始资金
    'margin_rate': 0.12,         # 12%保证金率
    'commission_rate': 0.0002,   # 万分之2手续费
    'slippage_ticks': 3,         # 3个tick滑点
    'position_weight': 0.05      # 每配对5%资金
}

print(f"时间配置:")
for key, value in TIME_CONFIG.items():
    print(f"  {key}: {value}")

print(f"\n回测配置:")
for key, value in BACKTEST_CONFIG.items():
    print(f"  {key}: {value}")

# 1. 加载数据
print(f"\n" + "=" * 60)
print("1. 加载价格数据")
print("-" * 60)

symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']

price_data = load_data(
    symbols=symbols,
    start_date=TIME_CONFIG['data_start'],
    end_date=TIME_CONFIG['data_end'],
    columns=['close'],
    log_price=False,  # 回测需要真实价格
    fill_method='ffill'
)

# 处理列名
if 'date' not in price_data.columns:
    price_data = price_data.reset_index()

rename_dict = {col: col.replace('_close', '') for col in price_data.columns if col.endswith('_close')}
if rename_dict:
    price_data = price_data.rename(columns=rename_dict)

print(f"✓ 价格数据加载完成: {price_data.shape}")
print(f"  日期范围: {price_data['date'].min()} ~ {price_data['date'].max()}")

# 2. 加载协整结果
print(f"\n" + "=" * 60)
print("2. 加载协整结果")
print("-" * 60)

coint_file = project_root / "output" / "pipeline_shifted" / "cointegration_results.csv"
coint_results = pd.read_csv(coint_file)

# 筛选有效配对
p_value_cols = [col for col in coint_results.columns if 'p_value' in col or 'pvalue' in col]
if len(p_value_cols) >= 2:
    p_col1, p_col2 = p_value_cols[:2]
    valid_pairs = coint_results[
        (coint_results[p_col1] < 0.05) & 
        (coint_results[p_col2] < 0.1)
    ].copy()
else:
    valid_pairs = coint_results.copy()

print(f"✓ 有效配对: {len(valid_pairs)}个")

# 准备配对参数
pairs_params = {}
position_weights = {}

for _, row in valid_pairs.iterrows():
    pair_name = f"{row['symbol_x']}-{row['symbol_y']}"
    
    beta_initial = row.get('beta_4y', row.get('beta_1y', 1.0))
    
    pairs_params[pair_name] = {
        'symbol_x': row['symbol_x'],
        'symbol_y': row['symbol_y'],
        'beta_initial': beta_initial,
        'direction': row.get('direction', 'y_on_x')
    }
    
    # 每个配对分配相等权重
    position_weights[pair_name] = BACKTEST_CONFIG['position_weight']

print(f"✓ 配对参数准备完成: {len(pairs_params)}个")

# 3. 生成交易信号
print(f"\n" + "=" * 60)
print("3. 生成交易信号")
print("-" * 60)

# 为信号生成准备对数价格数据
log_price_data = load_data(
    symbols=symbols,
    start_date=TIME_CONFIG['data_start'],
    end_date=TIME_CONFIG['data_end'],
    columns=['close'],
    log_price=True,  # 信号生成需要对数价格
    fill_method='ffill'
)

if 'date' not in log_price_data.columns:
    log_price_data = log_price_data.reset_index()

rename_dict = {col: col.replace('_close', '') for col in log_price_data.columns if col.endswith('_close')}
if rename_dict:
    log_price_data = log_price_data.rename(columns=rename_dict)

signal_generator = SignalGenerator(
    window=60,
    z_open=2.0,
    z_close=0.5,
    convergence_days=20,
    convergence_threshold=0.01
)

all_signals = signal_generator.generate_all_signals(
    pairs_params=pairs_params,
    price_data=log_price_data,
    convergence_end=TIME_CONFIG['convergence_end'],
    signal_start=TIME_CONFIG['signal_start'],
    hist_start=TIME_CONFIG['hist_start'],
    hist_end=TIME_CONFIG['hist_end']
)

print(f"✓ 信号生成完成: {len(all_signals)}条")

# 统计信号
signal_counts = all_signals['signal'].value_counts()
total_open = signal_counts.get('open_long', 0) + signal_counts.get('open_short', 0)
total_close = signal_counts.get('close', 0)

print(f"  信号统计:")
print(f"    开仓信号: {total_open} (多头: {signal_counts.get('open_long', 0)}, 空头: {signal_counts.get('open_short', 0)})")
print(f"    平仓信号: {total_close}")
print(f"    收敛信号: {signal_counts.get('converging', 0)}")

# 4. 运行回测
print(f"\n" + "=" * 60)
print("4. 运行回测引擎")
print("-" * 60)

try:
    # 初始化回测引擎
    backtest_engine = BacktestEngine(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        margin_rate=BACKTEST_CONFIG['margin_rate'],
        commission_rate=BACKTEST_CONFIG['commission_rate'],
        slippage_ticks=BACKTEST_CONFIG['slippage_ticks'],
        position_weights=position_weights  # 直接在初始化时传入
    )
    
    # 加载合约规格（使用JSON格式，乘数更准确）
    contract_specs_file = project_root / "configs" / "contract_specs.json"
    backtest_engine.load_contract_specs(str(contract_specs_file))
    
    print(f"✓ 回测引擎初始化完成")
    print(f"  初始资金: {BACKTEST_CONFIG['initial_capital']:,}")
    print(f"  配对数量: {len(position_weights)}")
    
    # 准备回测期间的信号
    backtest_signals = all_signals[
        all_signals['date'] >= TIME_CONFIG['backtest_start']
    ].copy()
    
    # 确保信号按时间排序
    backtest_signals = backtest_signals.sort_values(['date', 'pair']).reset_index(drop=True)
    
    print(f"✓ 回测信号准备完成: {len(backtest_signals)}条")
    
    # 运行回测 - 逐日执行信号
    print(f"\n开始回测...")
    
    # 按日期分组信号
    daily_signals = backtest_signals.groupby('date')
    dates = sorted(daily_signals.groups.keys())
    
    executed_signals = 0
    total_days = len(dates)
    
    print(f"回测期间: {len(dates)}个交易日")
    
    for i, date in enumerate(dates):
        if i % 50 == 0:  # 每50天打印一次进度
            progress = (i / total_days) * 100
            print(f"  进度: {progress:.1f}% ({i}/{total_days})")
        
        # 获取当日价格
        date_prices = price_data[price_data['date'] == date]
        if date_prices.empty:
            continue
            
        # 转换为价格字典
        current_prices = {}
        for col in date_prices.columns:
            if col != 'date':
                current_prices[col] = date_prices.iloc[0][col]
        
        # 获取当日信号
        day_signals = daily_signals.get_group(date)
        
        # 执行每个信号
        for _, signal in day_signals.iterrows():
            if signal['signal'] not in ['hold', 'converging']:
                success = backtest_engine.execute_signal(
                    signal.to_dict(), 
                    current_prices, 
                    pd.to_datetime(date)
                )
                if success:
                    executed_signals += 1
        
        # 执行风险管理
        backtest_engine.run_risk_management(pd.to_datetime(date), current_prices)
        
        # 执行逐日结算
        backtest_engine.position_manager.daily_settlement(current_prices)
    
    print(f"✓ 回测完成!")
    print(f"  执行信号数: {executed_signals}")
    print(f"  回测天数: {total_days}")
    
    # 5. 结果分析
    print(f"\n" + "=" * 60)
    print("5. 回测结果分析")
    print("-" * 60)
    
    # 获取最终状态
    final_capital = backtest_engine.position_manager.total_equity
    total_pnl = final_capital - BACKTEST_CONFIG['initial_capital']
    total_return = (total_pnl / BACKTEST_CONFIG['initial_capital']) * 100
    
    # 时间统计
    backtest_days = (pd.to_datetime(TIME_CONFIG['data_end']) - pd.to_datetime(TIME_CONFIG['backtest_start'])).days
    annualized_return = (total_return / backtest_days) * 365
    
    print(f"📊 基本绩效指标:")
    print(f"  初始资金: {BACKTEST_CONFIG['initial_capital']:,}")
    print(f"  最终资金: {final_capital:,.2f}")
    print(f"  总盈亏: {total_pnl:,.2f}")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  年化收益率: {annualized_return:.2f}%")
    print(f"  回测天数: {backtest_days}天")
    
    # 交易统计
    trades = backtest_engine.trade_records
    if len(trades) > 0:
        print(f"\n📈 交易统计:")
        print(f"  总交易次数: {len(trades)}")
        
        # 盈亏统计
        trade_pnls = [trade.get('pnl', 0) for trade in trades]
        profitable_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        if trade_pnls:
            print(f"  平均盈亏: {np.mean(trade_pnls):,.2f}")
            print(f"  盈利交易: {len(profitable_trades)} ({len(profitable_trades)/len(trades)*100:.1f}%)")
            print(f"  亏损交易: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
            
            if profitable_trades:
                print(f"  平均盈利: {np.mean(profitable_trades):,.2f}")
            if losing_trades:
                print(f"  平均亏损: {np.mean(losing_trades):,.2f}")
    
    # 持仓统计
    current_positions = backtest_engine.position_manager.positions
    print(f"\n📋 持仓统计:")
    print(f"  当前持仓: {len(current_positions)}个")
    print(f"  占用保证金: {backtest_engine.position_manager.occupied_margin:,.2f}")
    print(f"  可用资金: {backtest_engine.position_manager.available_capital:,.2f}")
    
    # 保存结果
    output_dir = project_root / "output" / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存交易记录
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_file = output_dir / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\n💾 结果已保存:")
        print(f"  交易记录: {trades_file}")
    
    # 保存每日记录
    daily_records = backtest_engine.position_manager.daily_records
    if daily_records:
        daily_df = pd.DataFrame(daily_records)
        daily_file = output_dir / f"daily_records_{timestamp}.csv"
        daily_df.to_csv(daily_file, index=False)
        print(f"  每日记录: {daily_file}")
    
    # 输出总结
    print(f"\n" + "=" * 60)
    print("6. 回测总结")
    print("-" * 60)
    
    if total_return > 0:
        print(f"🎉 回测成功! 策略表现良好")
        print(f"   年化收益率: {annualized_return:.2f}%")
    else:
        print(f"⚠️  策略存在亏损，需要优化")
        print(f"   总亏损: {total_pnl:,.2f}")
    
    print(f"\n✅ 回测完成!")

except Exception as e:
    print(f"❌ 回测失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("回测结束")
print("=" * 80)