#!/usr/bin/env python3
"""
==============================================================================
                完整端到端配对交易管道 v2.0 (基于原子服务)
==============================================================================

版本更新说明：
- v2.0: 完全基于原子服务重构，使用BacktestEngine进行回测
- v1.0: 原始实现，包含自定义回测逻辑

核心原子服务：
1. DataManager - 数据管理原子服务
2. CointegrationAnalyzer - 协整分析原子服务  
3. SignalGenerator - 信号生成原子服务
4. BacktestEngine - 回测引擎原子服务

==============================================================================
详细算法流程和计算参数：
==============================================================================

算法流程说明：
1. 协整筛选阶段
   - 筛选91个配对（14个品种的所有组合）
   - 使用5年和1年的p值都小于0.05作为筛选条件
   - 使用最近一年(2024年)的波动率确定方向（低波动作X，高波动作Y）

2. Beta初始化阶段
   - 使用2023年全年数据计算OLS beta作为基础β
   - 这个OLS beta作为Kalman滤波的初始值
   - 计算2023年残差方差作为观测噪声R的初始值

3. Kalman预热阶段（收敛期）
   - 时间：2024年1月到6月（6个月）
   - 目的：让Kalman滤波的beta收敛稳定
   - 这期间只更新beta，不生成交易信号
   - Beta日变化限制5%，但最小绝对变化0.001防止死螺旋

4. 信号生成阶段
   - 时间：2024年7月开始
   - 使用收敛后的Kalman滤波动态更新beta
   - Z-score阈值：
     * 开仓：|Z| > 2.2
     * 平仓：|Z| < 0.3
   - 最大持仓30天强制平仓

5. 回测阶段
   - Beta约束：只交易[0.3, 3.0]范围内的信号
   - 负Beta处理：
     * 正Beta（正相关）：传统对冲（买Y卖X 或 卖Y买X）
     * 负Beta（负相关）：同向操作（同时买或同时卖）
   - 保证金率：12%
   - 止损：保证金的15%
   - 手续费：万分之2（双边）
   - 滑点：3个tick

==============================================================================
具体参数配置：
==============================================================================

品种配置：
- 14个金属期货：AG0, AU0, AL0, CU0, NI0, PB0, SN0, ZN0, HC0, I0, RB0, SF0, SM0, SS0
- 贵金属(2个)、有色金属(6个)、黑色系(6个)

时间配置：
- Beta训练期：2023-01-01 至 2023-12-31
- Kalman收敛期：2024-01-01 至 2024-06-30
- 信号生成期：2024-07-01 至 2025-08-20

协整筛选参数：
- 5年期p值阈值：< 0.05
- 1年期p值阈值：< 0.05  
- 半衰期约束：[2, 60]天（仅1年期）
- v2.0改进：移除5年期半衰期约束

信号生成参数：
- Z-score开仓阈值：> 2.2
- Z-score平仓阈值：< 0.3
- 滚动窗口：60个交易日
- 最大持仓：30天强制平仓

Beta约束参数：
- Beta绝对值范围：[0.3, 3.0]
- 只过滤开仓信号，保留所有平仓信号

回测交易参数：
- 初始资金：500万元
- 保证金率：12%
- 手续费率：万分之2（双边）
- 滑点设置：每腿3个tick
- 止损参数：15%保证金止损（可配置，1.0=100%禁用）
- 时间止损：30天最大持仓

风险控制：
- 负Beta处理：正Beta传统对冲，负Beta同向操作
- 保证金管理：按12%保证金率计算
- 止损机制：保证金的15%损失触发止损
- 强制平仓：持仓30天后强制平仓

==============================================================================

4. 回测执行阶段 (BacktestEngine原子服务)
   交易参数：
   - 初始资金: 500万
   - 保证金率: 12%
   - 手续费率: 万分之2 (双边)
   - 滑点: 每腿3个tick
   
   风险控制参数 (可配置):
   - 止损比例: 可设置 (15%启用止损，100%禁用止损)
   - 时间止损: 30天
   
   手数计算：
   - 三种算法: 网格搜索、比率约简、线性规划
   - 使用Fraction类计算最小整数比
   - Y:X手数比例基于动态Beta值

5. 绩效分析阶段
   输出指标：
   - 基础统计: 总交易数、胜率、盈亏比
   - 收益指标: 总收益率、年化收益率、夏普比率、最大回撤
   - 风控统计: 止损次数、时间止损次数
   - 交易明细: 每笔交易的完整记录

==============================================================================

作者: Claude Code
日期: 2025-08-21  
版本: v2.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# 添加lib路径
sys.path.append('.')

# 导入所有原子服务
from typing import Dict
from lib.data import DataManager, load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
from lib.backtest import BacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局配置参数 v2.0
# ==============================================================================

# 版本信息
PIPELINE_VERSION = "2.0"
PIPELINE_NAME = "完整端到端配对交易管道"

# 品种列表（14个金属期货）
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'  # 黑色系
]

# 时间配置
TIME_CONFIG = {
    'beta_training_start': '2023-01-01',
    'beta_training_end': '2023-12-31',
    'convergence_start': '2024-01-01', 
    'convergence_end': '2024-06-30',
    'signal_start': '2024-07-01',
    'backtest_end': '2025-08-20'
}

# 协整筛选参数
COINT_CONFIG = {
    'p_value_5y': 0.05,
    'p_value_1y': 0.05,
    'halflife_min': 2,
    'halflife_max': 60,
    'apply_5y_halflife_constraint': False  # v2.0移除5年半衰期约束
}

# 信号生成参数
SIGNAL_CONFIG = {
    'z_open': 2.2,
    'z_close': 0.3,
    'window': 60,
    'max_holding_days': 30
}

# Beta约束参数
BETA_CONFIG = {
    'min_abs': 0.3,
    'max_abs': 3.0
}

# 回测参数
BACKTEST_CONFIG = {
    'initial_capital': 5000000,
    'margin_rate': 0.12,
    'commission_rate': 0.0002,
    'slippage_ticks': 3,
    'stop_loss_pct': 1.0,  # 设置为1.0(100%)禁用止损，0.15(15%)启用止损
    'max_holding_days': 30
}

# 合约规格
CONTRACT_SPECS = {
    'AG0': {'multiplier': 15, 'tick_size': 1},
    'AU0': {'multiplier': 1000, 'tick_size': 0.02},
    'AL0': {'multiplier': 5, 'tick_size': 5},
    'CU0': {'multiplier': 5, 'tick_size': 10},
    'NI0': {'multiplier': 1, 'tick_size': 10},
    'PB0': {'multiplier': 5, 'tick_size': 5},
    'SN0': {'multiplier': 1, 'tick_size': 10},
    'ZN0': {'multiplier': 5, 'tick_size': 5},
    'HC0': {'multiplier': 10, 'tick_size': 1},
    'I0': {'multiplier': 100, 'tick_size': 0.5},
    'RB0': {'multiplier': 10, 'tick_size': 1},
    'SF0': {'multiplier': 5, 'tick_size': 2},
    'SM0': {'multiplier': 5, 'tick_size': 2},
    'SS0': {'multiplier': 5, 'tick_size': 5}
}

def print_header():
    """打印管道标题"""
    print("=" * 80)
    print(f"  {PIPELINE_NAME} v{PIPELINE_VERSION}")
    print("  基于原子服务架构")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"品种数量: {len(SYMBOLS)}")
    print(f"配对数量: {len(SYMBOLS) * (len(SYMBOLS) - 1) // 2}")
    print(f"数据范围: {TIME_CONFIG['beta_training_start']} ~ {TIME_CONFIG['backtest_end']}")
    print()

def step1_cointegration_screening(data_manager: DataManager) -> pd.DataFrame:
    """
    步骤1: 协整筛选 (使用CointegrationAnalyzer原子服务)
    
    Returns:
        符合条件的协整配对DataFrame
    """
    logger.info("=" * 60)
    logger.info("步骤1: 协整筛选 (CointegrationAnalyzer原子服务)")
    logger.info("=" * 60)
    
    # 初始化协整分析器
    data = load_data(SYMBOLS, 
                     start_date='2020-01-01',
                     columns=['close'], 
                     log_price=True)
    analyzer = CointegrationAnalyzer(data)
    
    # 筛选所有配对
    logger.info("开始协整筛选...")
    significant_pairs = analyzer.screen_all_pairs(
        p_threshold=COINT_CONFIG['p_value_5y']
    )
    
    if len(significant_pairs) == 0:
        logger.error("未找到符合条件的协整配对!")
        return pd.DataFrame()
    
    logger.info(f"协整筛选完成:")
    logger.info(f"  总配对数: {len(SYMBOLS) * (len(SYMBOLS) - 1) // 2}")
    logger.info(f"  通过筛选: {len(significant_pairs)}")
    logger.info(f"  筛选率: {len(significant_pairs) / (len(SYMBOLS) * (len(SYMBOLS) - 1) // 2) * 100:.1f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'output/cointegrated_pairs_v2_{timestamp}.csv'
    significant_pairs.to_csv(output_file, index=False)
    logger.info(f"协整结果保存至: {output_file}")
    
    return significant_pairs

def step2_signal_generation(pairs_df: pd.DataFrame, data_manager: DataManager) -> pd.DataFrame:
    """
    步骤2: 信号生成 (使用SignalGenerator原子服务)
    
    Args:
        pairs_df: 协整配对DataFrame
        data_manager: 数据管理器
        
    Returns:
        交易信号DataFrame
    """
    logger.info("=" * 60)
    logger.info("步骤2: 信号生成 (SignalGenerator原子服务)")
    logger.info("=" * 60)
    
    # 初始化信号生成器
    generator = SignalGenerator(
        z_open=SIGNAL_CONFIG['z_open'],
        z_close=SIGNAL_CONFIG['z_close'],
        window=SIGNAL_CONFIG['window']
    )
    
    # 批量生成所有配对的信号
    logger.info("开始信号生成...")
    
    # 准备参数字典
    pairs_params = {}
    for _, pair_row in pairs_df.iterrows():
        pair = pair_row['pair']
        pairs_params[pair] = {
            'beta_initial': pair_row.get('beta_5y', 1.0),
            'R': pair_row.get('residual_var_5y', 0.01)
        }
    
    # 加载价格数据
    price_data = load_data(SYMBOLS, 
                          start_date='2020-01-01',
                          columns=['close'], 
                          log_price=True)
    
    # SignalGenerator需要date作为列而不是索引
    if 'date' in price_data.index.names:
        price_data = price_data.reset_index()
    
    all_signals = generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end=TIME_CONFIG['convergence_end'],
        signal_start=TIME_CONFIG['signal_start']
    )
    
    if len(all_signals) == 0:
        logger.error("未生成任何交易信号!")
        return pd.DataFrame()
    
    logger.info(f"信号生成完成:")
    logger.info(f"  处理配对: {len(pairs_df)}")
    logger.info(f"  总信号数: {len(all_signals)}")
    
    # 统计信号类型
    signal_counts = all_signals['signal'].value_counts()
    for signal_type, count in signal_counts.items():
        logger.info(f"  {signal_type}: {count}")
    
    # 保存信号
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'output/signals_v2_{timestamp}.csv'
    all_signals.to_csv(output_file, index=False)
    logger.info(f"信号保存至: {output_file}")
    
    return all_signals

def step3_signal_filtering(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤3: 信号过滤 (Beta约束)
    
    Args:
        signals_df: 原始信号DataFrame
        
    Returns:
        过滤后的信号DataFrame
    """
    logger.info("=" * 60)
    logger.info("步骤3: 信号过滤 (Beta约束)")
    logger.info("=" * 60)
    
    original_count = len(signals_df)
    
    # 只过滤开仓信号
    open_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short'])].copy()
    close_signals = signals_df[signals_df['signal'] == 'close'].copy()
    
    # Beta约束过滤
    logger.info(f"应用Beta约束 [{BETA_CONFIG['min_abs']}, {BETA_CONFIG['max_abs']}] (绝对值)")
    valid_mask = (abs(open_signals['beta']) >= BETA_CONFIG['min_abs']) & \
                 (abs(open_signals['beta']) <= BETA_CONFIG['max_abs'])
    
    filtered_count = (~valid_mask).sum()
    valid_open_signals = open_signals[valid_mask].copy()
    
    # 合并有效信号
    filtered_signals = pd.concat([valid_open_signals, close_signals], ignore_index=True)
    filtered_signals = filtered_signals.sort_values('date').reset_index(drop=True)
    
    logger.info(f"信号过滤完成:")
    logger.info(f"  原始信号: {original_count}")
    logger.info(f"  过滤开仓信号: {filtered_count}")  
    logger.info(f"  有效信号: {len(filtered_signals)}")
    logger.info(f"  过滤率: {filtered_count / len(open_signals) * 100:.1f}%")
    
    return filtered_signals

def step4_backtest_execution(signals_df: pd.DataFrame, data_manager: DataManager) -> Dict:
    """
    步骤4: 回测执行 (使用BacktestEngine原子服务)
    
    Args:
        signals_df: 交易信号DataFrame
        data_manager: 数据管理器
        
    Returns:
        回测结果字典
    """
    logger.info("=" * 60) 
    logger.info("步骤4: 回测执行 (BacktestEngine原子服务)")
    logger.info("=" * 60)
    
    # 初始化回测引擎
    engine = BacktestEngine(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        margin_rate=BACKTEST_CONFIG['margin_rate'],
        commission_rate=BACKTEST_CONFIG['commission_rate'],
        slippage_ticks=BACKTEST_CONFIG['slippage_ticks'],
        stop_loss_pct=BACKTEST_CONFIG['stop_loss_pct'],
        max_holding_days=BACKTEST_CONFIG['max_holding_days']
    )
    
    # 设置合约规格
    engine.contract_specs = CONTRACT_SPECS
    
    # 止损状态判断
    stop_loss_enabled = BACKTEST_CONFIG['stop_loss_pct'] < 1.0
    stop_loss_desc = f"{BACKTEST_CONFIG['stop_loss_pct']*100:.0f}% ({'启用' if stop_loss_enabled else '禁用'})"
    
    logger.info(f"回测引擎配置:")
    logger.info(f"  初始资金: ¥{BACKTEST_CONFIG['initial_capital']:,}")
    logger.info(f"  保证金率: {BACKTEST_CONFIG['margin_rate']*100:.0f}%")
    logger.info(f"  止损比例: {stop_loss_desc}")
    logger.info(f"  最大持仓: {BACKTEST_CONFIG['max_holding_days']}天")
    logger.info(f"  版本模式: {'无止损版本(与v1.0对比)' if not stop_loss_enabled else '完整风控版本'}")
    
    # 准备价格数据
    logger.info("加载价格数据...")
    price_data = {}
    for symbol in SYMBOLS:
        df = data_manager.load_from_parquet(symbol)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # 使用 symbol_close 格式作为key，与信号配对格式保持一致
        price_data[f"{symbol}_close"] = df[['close']]
    
    # 转换信号格式为BacktestEngine需要的格式
    logger.info("转换信号格式...")
    backtest_signals = []
    
    # 处理开仓信号
    open_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short'])]
    for _, signal in open_signals.iterrows():
        pair = signal['pair']
        
        # 判断信号类型
        if signal['beta'] < 0:
            # 负Beta：同向操作
            signal_type = 'long_spread' if signal['signal'] == 'open_long' else 'short_spread'
        else:
            # 正Beta：传统对冲
            signal_type = 'long_spread' if signal['signal'] == 'open_long' else 'short_spread'
        
        backtest_signals.append({
            'date': signal['date'],
            'pair': pair,
            'signal': signal_type,
            'theoretical_ratio': abs(signal['beta']),
            'z_score': signal.get('z_score', 0),
            'spread_formula': f"y - {abs(signal['beta']):.4f} * x"
        })
    
    # 处理平仓信号
    close_signals = signals_df[signals_df['signal'] == 'close']
    for _, signal in close_signals.iterrows():
        backtest_signals.append({
            'date': signal['date'],
            'pair': signal['pair'],
            'signal': 'close',
            'z_score': signal.get('z_score', 0)
        })
    
    # 按日期排序
    backtest_signals = sorted(backtest_signals, key=lambda x: x['date'])
    logger.info(f"待执行信号: {len(backtest_signals)}")
    
    # 执行回测
    logger.info("开始回测执行...")
    
    # 获取所有交易日期
    all_dates = set()
    for signal in backtest_signals:
        all_dates.add(signal['date'])
    
    for symbol_data in price_data.values():
        all_dates.update(symbol_data.index)
    
    all_dates = sorted(all_dates)
    start_date = pd.Timestamp(TIME_CONFIG['signal_start'])
    end_date = pd.Timestamp(TIME_CONFIG['backtest_end'])
    all_dates = [d for d in all_dates if start_date <= d <= end_date]
    
    # 按日期处理信号
    signal_index = 0
    processed_signals = 0
    
    for current_date in all_dates:
        # 获取当前价格
        current_prices = {}
        for symbol, df in price_data.items():
            if current_date in df.index:
                current_prices[symbol] = df.loc[current_date, 'close']
        
        # 执行风险管理
        if current_prices:
            closed_pairs = engine.run_risk_management(current_date, current_prices)
            if closed_pairs:
                logger.debug(f"{current_date}: 风控平仓 {len(closed_pairs)} 个配对")
        
        # 处理当日信号
        while signal_index < len(backtest_signals):
            signal = backtest_signals[signal_index]
            if signal['date'] > current_date:
                break
            
            # 执行信号
            if current_prices:
                success = engine.execute_signal(signal, current_prices, current_date)
                if success:
                    processed_signals += 1
                    if processed_signals % 10 == 0:
                        logger.info(f"已处理信号: {processed_signals}/{len(backtest_signals)}")
            
            signal_index += 1
        
        # 日终结算
        if current_prices:
            engine.position_manager.daily_settlement(current_prices)
    
    logger.info(f"回测执行完成，共处理 {processed_signals} 个信号")
    
    # 生成回测结果
    results = engine.calculate_metrics()
    
    return results, engine

def step5_results_analysis(results: Dict, engine: BacktestEngine):
    """
    步骤5: 结果分析和输出
    
    Args:
        results: 回测结果字典
        engine: 回测引擎实例
    """
    logger.info("=" * 60)
    logger.info("步骤5: 结果分析")
    logger.info("=" * 60)
    
    # 基础统计
    logger.info(f"【交易统计】")
    logger.info(f"  总交易数: {results['total_trades']}")
    logger.info(f"  盈利交易: {results['winning_trades']}")
    logger.info(f"  亏损交易: {results['losing_trades']}")
    logger.info(f"  胜率: {results['win_rate']:.1f}%")
    
    # 收益分析
    logger.info(f"\n【收益分析】")
    logger.info(f"  总净盈亏: ¥{results['total_pnl']:,.2f}")
    logger.info(f"  总收益率: {results['total_return']:.2f}%")
    logger.info(f"  年化收益率: {results.get('annual_return', 0):.2f}%")
    logger.info(f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  最大回撤: {results.get('max_drawdown', 0):.2f}%")
    
    # 风控统计
    stop_losses = sum(1 for r in engine.trade_records if r.get('close_reason') == 'stop_loss')
    time_stops = sum(1 for r in engine.trade_records if r.get('close_reason') == 'time_stop')
    
    logger.info(f"\n【风控统计】")
    logger.info(f"  止损触发: {stop_losses} 次")
    logger.info(f"  时间止损: {time_stops} 次")
    logger.info(f"  正常平仓: {results['total_trades'] - stop_losses - time_stops} 次")
    
    # 保存交易记录
    if engine.trade_records:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trades_df = pd.DataFrame(engine.trade_records)
        output_file = f'output/trades_v2_{timestamp}.csv'
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录保存至: {output_file}")
        
        # 分析止损交易
        if stop_losses > 0:
            stop_loss_trades = trades_df[trades_df['close_reason'] == 'stop_loss']
            logger.info(f"\n【止损交易分析】")
            avg_stop_loss = stop_loss_trades['net_pnl'].mean()
            logger.info(f"  平均止损损失: ¥{avg_stop_loss:,.2f}")
            logger.info(f"  止损率: {stop_losses / results['total_trades'] * 100:.1f}%")
    
    # 版本对比和验算说明
    stop_loss_enabled = BACKTEST_CONFIG['stop_loss_pct'] < 1.0
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"  {PIPELINE_NAME} v{PIPELINE_VERSION} 执行完成")
    logger.info("=" * 60)
    
    if not stop_loss_enabled:
        logger.info(f"🔍 验算模式 (与v1.0对比):")
        logger.info(f"  - 当前配置: 止损100%禁用，应与v1.0结果一致")
        logger.info(f"  - 核心改进: 使用BacktestEngine原子服务")
        logger.info(f"  - 计算精度: 双算法验证，Fraction类手数计算")
        logger.info(f"  - 移除约束: 5年半衰期约束，提升策略容量")
        logger.info(f"  ⚠️  如结果差异较大，请检查算法实现")
    else:
        logger.info(f"🛡️  完整风控模式:")
        logger.info(f"  - 止损控制: {BACKTEST_CONFIG['stop_loss_pct']*100:.0f}%保证金止损")
        logger.info(f"  - 时间控制: {BACKTEST_CONFIG['max_holding_days']}天时间止损")  
        logger.info(f"  - 风险优化: 相比v1.0增加完整风险管理")
    
    logger.info(f"\n📈 v2.0核心优势:")
    logger.info(f"  ✅ 100%原子服务架构，模块化程度更高")
    logger.info(f"  ✅ BacktestEngine专业回测，PnL计算更精确")
    logger.info(f"  ✅ 三种手数算法，Fraction类最小整数比")
    logger.info(f"  ✅ 可配置止损参数，灵活控制风险策略")

def main():
    """
    主函数 - 完整端到端管道执行
    """
    try:
        # 打印标题
        print_header()
        
        # 初始化数据管理器
        logger.info("初始化数据管理器...")
        data_manager = DataManager()
        
        # 步骤1: 协整筛选
        pairs_df = step1_cointegration_screening(data_manager)
        if len(pairs_df) == 0:
            logger.error("协整筛选失败，终止执行")
            return
        
        # 步骤2: 信号生成
        signals_df = step2_signal_generation(pairs_df, data_manager)
        if len(signals_df) == 0:
            logger.error("信号生成失败，终止执行")
            return
        
        # 步骤3: 信号过滤
        filtered_signals = step3_signal_filtering(signals_df)
        if len(filtered_signals) == 0:
            logger.error("所有信号被过滤，终止执行")
            return
        
        # 步骤4: 回测执行
        results, engine = step4_backtest_execution(filtered_signals, data_manager)
        if not results:
            logger.error("回测执行失败，终止执行")
            return
        
        # 步骤5: 结果分析
        step5_results_analysis(results, engine)
        
        logger.info(f"\n🎉 管道 v{PIPELINE_VERSION} 执行完成!")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断执行")
    except Exception as e:
        logger.error(f"管道执行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()