#!/usr/bin/env python3
"""
==============================================================================
                完整端到端配对交易管道 - 往前平移一年版本
==============================================================================

时间平移说明：
- 原版本(v2.1): 2024年7月开始信号期
- 本版本(shifted): 2023年7月开始信号期（往前平移整整一年）

★★★ 重要时间逻辑修复（2025-08-23）★★★
==============================================================================
问题发现：
- 原代码错误地从2022年1月开始运行Kalman滤波
- 导致Beta在2023年1月时已经演化了一整年
- 例如AU0-ZN0的Beta从-0.0996变化到1.04

正确的时间逻辑应该是：
1. 2022年1-12月：仅用于OLS计算初始Beta（静态，不运行Kalman）
2. 2023年1-6月：Kalman滤波预热期（从初始Beta开始演化）
3. 2023年7月-2024年8月：信号生成和回测期

修复方案：
- 确保Kalman滤波从convergence_start（2023-01-01）开始
- 不是从beta_training_start（2022-01-01）开始

参数配置总览：
==============================================================================

【品种配置】
- 品种数量: 14个金属期货
- 贵金属: AG0(银), AU0(金)
- 有色金属: AL0(铝), CU0(铜), NI0(镍), PB0(铅), SN0(锡), ZN0(锌)
- 黑色系: HC0(热卷), I0(铁矿), RB0(螺纹), SF0(硅铁), SM0(锰硅), SS0(不锈钢)

【时间配置】
- 数据起始: 2019-01-01 (用于协整分析，需要4年历史数据)
- Beta训练期: 2022-01-01 至 2022-12-31 (用于OLS计算初始Beta)
- Kalman收敛期: 2023-01-01 至 2023-06-30 (往前平移1年)
- 信号生成期: 2023-07-01 至 2024-08-20 (往前平移1年)
- 回测结束: 2024-08-20

【协整筛选参数】****重要调整****
- 4年期p值阈值: 0.05 (使用4年数据，不是5年)
- 1年期p值阈值: 0.10 (放宽到0.10，不是0.05)
- 波动率计算期间: 2023-01-01开始 (用于方向判定)
- 筛选逻辑: 4年p值 < 0.05 AND 1年p值 < 0.10

【信号生成参数】
- 开仓Z-score阈值: 2.2 (回到原始设置)
- 平仓Z-score阈值: 0.3 (回到原始设置)
- Z-score上限: 3.2 (超过此值不开仓)
- 滚动窗口: 60个交易日
- 最大持仓天数: 30天
- 收敛阈值: 1%

【Beta约束参数】
- 最小绝对值: 0.3
- 最大绝对值: 3.0
- 过滤条件: |beta| ∈ [0.3, 3.0]

【回测参数】
- 初始资金: 500万元
- 保证金率: 12%
- 手续费率: 万分之2 (0.02%)
- 滑点: 3个tick
- 止损比例: 15% (基于保证金)
- 最大持仓天数: 30天

【合约规格】
- 使用JSON文件中的规格 (configs/contract_specs.json)
- 注意: JSON格式的乘数是正确的（如I0=100，不是100000）

【输出目录】
- 输出根目录: output/pipeline_shifted/
- 协整结果: output/pipeline_shifted/cointegration_results.csv
- 信号文件: output/pipeline_shifted/signals_YYYYMMDD_HHMMSS.csv
- 交易记录: output/pipeline_shifted/trades_YYYYMMDD_HHMMSS.csv
- 回测报告: output/pipeline_shifted/backtest_report_YYYYMMDD_HHMMSS.json

版本说明：
- shifted: 往前平移一年版本，用于验证策略在不同时间段的表现
- 协整条件调整: 使用4年p值<0.05 + 1年p值<0.10的双重筛选
- 信号阈值调整: 使用实际验证过的2.0/0.5阈值

==============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入最新的原子服务
from lib.data import load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator, calculate_ols_beta
from lib.backtest import BacktestEngine, PositionManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局配置参数 - 往前平移一年版本
# ==============================================================================

# 版本信息
PIPELINE_VERSION = "shifted"
PIPELINE_NAME = "完整端到端配对交易管道(往前平移一年版)"

# 品种列表（14个金属期货）
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'  # 黑色系
]

# 时间配置 - 往前平移一年
TIME_CONFIG = {
    'data_start': '2019-01-01',         # 数据起始（需要4年历史做协整）
    'beta_training_start': '2022-01-01', # Beta训练开始（原2023→2022）
    'beta_training_end': '2022-12-31',   # Beta训练结束（原2023→2022）
    'convergence_start': '2023-01-01',   # Kalman收敛期开始（原2024→2023）
    'convergence_end': '2023-06-30',     # Kalman收敛期结束（原2024→2023）
    'signal_start': '2023-07-01',        # 信号生成开始（原2024→2023）
    'backtest_end': '2024-08-20'         # 回测结束（原2025→2024）
}

# 协整筛选参数 - 使用新的筛选条件
COINT_CONFIG = {
    'p_threshold_4y': 0.05,   # 4年p值阈值（不是5年）
    'p_threshold_1y': 0.05,   # 1年p值阈值（调整为0.05）
    'use_halflife_filter': False,  # 不使用半衰期筛选
    'volatility_start': '2023-01-01'  # 波动率计算起始（用于方向判定）
}

# 信号生成参数 - 回到用户要求的设置
SIGNAL_CONFIG = {
    'z_open': 2.2,     # 开仓阈值回到2.2
    'z_open_max': 3.2, # 重新启用上限限制，z>3.2不操作
    'z_close': 0.3,    # 平仓阈值回到0.3
    'window': 60,      # 滚动窗口
    'max_holding_days': 30,  # 最大持仓天数
    'convergence_threshold': 0.01  # 1%收敛阈值
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
    'stop_loss_pct': 0.15,  # 止损调整到15%
    'max_holding_days': 30
}

# 创建输出目录 - 使用新的目录名避免覆盖
OUTPUT_DIR = Path("output/pipeline_shifted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 步骤1: 协整筛选（使用4年+1年双重条件）
# ==============================================================================

def step1_cointegration_screening() -> pd.DataFrame:
    """
    步骤1: 协整筛选
    使用4年p值<0.05 + 1年p值<0.10的双重条件筛选
    """
    logger.info("=" * 60)
    logger.info("步骤1: 协整配对筛选（4年+1年双重条件）")
    logger.info("=" * 60)
    
    # 加载数据（对数价格）
    logger.info(f"加载数据: {TIME_CONFIG['data_start']} 至 {TIME_CONFIG['backtest_end']}")
    data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['data_start'],
        end_date=TIME_CONFIG['backtest_end'],
        columns=['close'],
        log_price=True
    )
    
    # 处理列名（去除_close后缀）
    data.columns = [col.replace('_close', '') if col.endswith('_close') else col for col in data.columns]
    
    logger.info(f"数据形状: {data.shape}")
    
    # 创建协整分析器
    analyzer = CointegrationAnalyzer(data)
    
    # 执行协整分析（所有配对）
    logger.info("执行全配对协整分析...")
    
    all_results = []
    pair_count = 0
    
    # 获取所有配对组合
    from itertools import combinations
    for symbol1, symbol2 in combinations(SYMBOLS, 2):
        pair_count += 1
        pair_name = f"{symbol1}-{symbol2}"
        
        # 方向判定（基于2023年波动率）
        direction, symbol_x, symbol_y = analyzer.determine_direction(
            symbol1, symbol2, 
            use_recent=True, 
            recent_start=COINT_CONFIG['volatility_start']
        )
        
        # 确定X和Y序列
        if symbol_x == symbol1:
            x_data = data[symbol1].values
            y_data = data[symbol2].values
        else:
            x_data = data[symbol2].values
            y_data = data[symbol1].values
        
        # 4年窗口协整检验（1008个交易日）
        end_idx = len(data)
        start_idx_4y = max(0, end_idx - 1008)
        
        if start_idx_4y < end_idx:
            x_4y = x_data[start_idx_4y:end_idx]
            y_4y = y_data[start_idx_4y:end_idx]
            
            from lib.coint import engle_granger_test
            result_4y = engle_granger_test(x_4y, y_4y, direction)
            pvalue_4y = result_4y['pvalue']
            beta_4y = result_4y['beta']
        else:
            pvalue_4y = 1.0
            beta_4y = np.nan
        
        # 1年窗口协整检验（252个交易日）
        start_idx_1y = max(0, end_idx - 252)
        
        if start_idx_1y < end_idx:
            x_1y = x_data[start_idx_1y:end_idx]
            y_1y = y_data[start_idx_1y:end_idx]
            
            result_1y = engle_granger_test(x_1y, y_1y, direction)
            pvalue_1y = result_1y['pvalue']
            beta_1y = result_1y['beta']
        else:
            pvalue_1y = 1.0
            beta_1y = np.nan
        
        # 记录结果
        result_dict = {
            'pair': pair_name,
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'direction': direction,
            'pvalue_4y': pvalue_4y,
            'pvalue_1y': pvalue_1y,
            'beta_4y': beta_4y,
            'beta_1y': beta_1y
        }
        
        all_results.append(result_dict)
        
        if pair_count % 10 == 0:
            logger.info(f"  已处理 {pair_count} 个配对...")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 应用筛选条件：4年p值<0.05 AND 1年p值<0.10
    logger.info(f"应用筛选条件: 4年p值<{COINT_CONFIG['p_threshold_4y']} AND 1年p值<{COINT_CONFIG['p_threshold_1y']}")
    
    filtered_df = results_df[
        (results_df['pvalue_4y'] < COINT_CONFIG['p_threshold_4y']) &
        (results_df['pvalue_1y'] < COINT_CONFIG['p_threshold_1y'])
    ].copy()
    
    logger.info(f"筛选结果: {len(filtered_df)}/{len(results_df)} 个配对通过")
    
    # 保存结果
    output_file = OUTPUT_DIR / "cointegration_results.csv"
    filtered_df.to_csv(output_file, index=False)
    logger.info(f"协整结果保存至: {output_file}")
    
    # 输出前10个配对
    if len(filtered_df) > 0:
        logger.info("前10个配对:")
        for idx, row in filtered_df.head(10).iterrows():
            logger.info(f"  {row['pair']}: 4年p={row['pvalue_4y']:.4f}, 1年p={row['pvalue_1y']:.4f}, β={row['beta_4y']:.3f}")
    
    return filtered_df

# ==============================================================================
# 步骤2: 计算初始Beta参数
# ==============================================================================

def step2_calculate_initial_betas(pairs_df: pd.DataFrame) -> Dict:
    """
    步骤2: 为每个配对计算初始Beta
    使用2022年数据进行OLS估计
    """
    logger.info("=" * 60)
    logger.info("步骤2: 计算初始Beta参数")
    logger.info("=" * 60)
    
    # 加载训练期数据（对数价格）
    training_data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['beta_training_start'],
        end_date=TIME_CONFIG['beta_training_end'],
        columns=['close'],
        log_price=True
    )
    
    # 处理列名（去除_close后缀）
    training_data.columns = [col.replace('_close', '') if col.endswith('_close') else col for col in training_data.columns]
    
    pairs_params = {}
    
    for _, row in pairs_df.iterrows():
        pair_name = row['pair']
        symbol_x = row['symbol_x']
        symbol_y = row['symbol_y']
        
        # 提取训练数据
        x_train = training_data[symbol_x].values
        y_train = training_data[symbol_y].values
        
        # 计算2022年一年OLS Beta作为Kalman初始值
        beta_initial = calculate_ols_beta(y_train, x_train, window=len(y_train))
        
        pairs_params[pair_name] = {
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'beta_initial': beta_initial,
            'direction': row['direction']
        }
        
        logger.info(f"{pair_name}: β_initial={beta_initial:.4f}")
    
    return pairs_params

# ==============================================================================
# 步骤3: 信号生成（Kalman滤波）
# ==============================================================================

def step3_signal_generation(pairs_params: Dict) -> pd.DataFrame:
    """
    步骤3: 使用Kalman滤波生成交易信号
    """
    logger.info("=" * 60)
    logger.info("步骤3: 信号生成 (Kalman滤波)")
    logger.info("=" * 60)
    
    # 初始化信号生成器
    generator = SignalGenerator(
        window=SIGNAL_CONFIG['window'],
        z_open=SIGNAL_CONFIG['z_open'],
        z_close=SIGNAL_CONFIG['z_close'],
        z_open_max=SIGNAL_CONFIG.get('z_open_max'),  # 添加z_open_max参数
        convergence_threshold=SIGNAL_CONFIG['convergence_threshold']
    )
    
    # 加载完整价格数据（对数价格用于信号生成）
    # 重要修复：从convergence_start开始，而不是beta_training_start
    # 这样Kalman滤波只从2023年1月开始运行
    price_data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['convergence_start'],  # 修改：从2023-01-01开始
        end_date=TIME_CONFIG['backtest_end'],
        columns=['close'],
        log_price=True  # 信号生成使用对数价格
    )
    
    # 重置索引使date成为列
    price_data = price_data.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
    # 移除列名中的_close后缀，因为信号生成器期望纯符号名
    price_data.columns = [col.replace('_close', '') if col.endswith('_close') else col for col in price_data.columns]
    
    # 批量生成信号
    logger.info(f"处理 {len(pairs_params)} 个配对的信号...")
    all_signals = generator.generate_all_signals(
        pairs_params=pairs_params,
        price_data=price_data,
        convergence_end=TIME_CONFIG['convergence_end'],
        signal_start=TIME_CONFIG['signal_start'],
        hist_start=TIME_CONFIG['beta_training_start'],
        hist_end=TIME_CONFIG['beta_training_end']
    )
    
    if all_signals.empty:
        logger.warning("未生成任何信号")
        return pd.DataFrame()
    
    # 统计信号
    logger.info(f"信号生成完成:")
    logger.info(f"  总信号数: {len(all_signals)}")
    
    # 按信号类型统计
    signal_counts = all_signals['signal'].value_counts()
    for signal_type, count in signal_counts.items():
        logger.info(f"  {signal_type}: {count}")
    
    # 保存信号（包含所有beta信息）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"signals_{timestamp}.csv"
    
    # 确保保存所有重要列
    important_cols = ['date', 'pair', 'signal', 'z_score', 'beta', 'ols_beta', 
                     'residual', 'converged', 'phase', 'reason']
    save_cols = [col for col in important_cols if col in all_signals.columns]
    
    all_signals[save_cols].to_csv(output_file, index=False)
    logger.info(f"信号保存至: {output_file}")
    
    return all_signals

# ==============================================================================
# 步骤4: 回测执行
# ==============================================================================

def step4_backtest_execution(signals_df: pd.DataFrame, pairs_params: Dict = None) -> Dict:
    """
    步骤4: 回测执行
    
    Args:
        signals_df: 信号数据
        pairs_params: 配对参数（包含symbol_x和symbol_y）
    """
    logger.info("=" * 60)
    logger.info("步骤4: 回测执行")
    logger.info("=" * 60)
    
    # Beta约束过滤
    open_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short'])].copy()
    
    if 'beta' in open_signals.columns:
        logger.info(f"应用Beta约束: |β| ∈ [{BETA_CONFIG['min_abs']}, {BETA_CONFIG['max_abs']}]")
        valid_mask = (
            (abs(open_signals['beta']) >= BETA_CONFIG['min_abs']) &
            (abs(open_signals['beta']) <= BETA_CONFIG['max_abs'])
        )
        filtered_count = (~valid_mask).sum()
        logger.info(f"Beta约束过滤了 {filtered_count} 个开仓信号")
        
        # 保留有效的开仓信号和所有平仓信号
        valid_open = open_signals[valid_mask]
        close_signals = signals_df[signals_df['signal'] == 'close']
        filtered_signals = pd.concat([valid_open, close_signals], ignore_index=True)
    else:
        filtered_signals = signals_df
    
    filtered_signals = filtered_signals.sort_values('date').reset_index(drop=True)
    
    # 初始化回测引擎
    logger.info("初始化回测引擎...")
    engine = BacktestEngine(
        initial_capital=BACKTEST_CONFIG['initial_capital'],
        margin_rate=BACKTEST_CONFIG['margin_rate'],
        commission_rate=BACKTEST_CONFIG['commission_rate'],
        slippage_ticks=BACKTEST_CONFIG['slippage_ticks'],
        stop_loss_pct=BACKTEST_CONFIG['stop_loss_pct'],
        max_holding_days=BACKTEST_CONFIG['max_holding_days']
    )
    
    # 加载合约规格（使用JSON格式）
    specs_file = project_root / "configs" / "contract_specs.json"
    if specs_file.exists():
        engine.load_contract_specs(str(specs_file))
        logger.info(f"加载合约规格: {specs_file}")
    else:
        logger.error(f"合约规格文件不存在: {specs_file}")
        return {}
    
    # 加载价格数据（原始价格用于回测）
    logger.info("加载回测价格数据...")
    price_data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['signal_start'],
        end_date=TIME_CONFIG['backtest_end'],
        columns=['close'],
        log_price=False  # 回测使用原始价格
    )
    
    # 执行回测
    logger.info(f"执行回测: {len(filtered_signals)} 个信号")
    
    # 获取所有交易日期
    all_dates = sorted(price_data.index.unique())
    start_date = pd.Timestamp(TIME_CONFIG['signal_start'])
    end_date = pd.Timestamp(TIME_CONFIG['backtest_end'])
    trading_dates = [d for d in all_dates if start_date <= d <= end_date]
    
    # 按日期分组信号
    signals_by_date = {}
    for _, signal in filtered_signals.iterrows():
        date = pd.Timestamp(signal['date'])
        if date not in signals_by_date:
            signals_by_date[date] = []
        
        # 转换信号格式
        if signal['signal'] == 'open_long':
            signal_type = 'long_spread'
        elif signal['signal'] == 'open_short':
            signal_type = 'short_spread'
        else:
            signal_type = 'close'
        
        # 准备theoretical_ratio（用于手数计算）
        theoretical_ratio = abs(signal.get('beta', 1.0))
        
        # 获取symbol_x和symbol_y（如果提供了pairs_params）
        formatted_signal = {
            'pair': signal['pair'],
            'signal': signal_type,
            'date': date,
            'beta': signal.get('beta', 1.0),
            'ols_beta': signal.get('ols_beta', np.nan),  # 添加OLS beta
            'theoretical_ratio': theoretical_ratio,
            'z_score': signal.get('z_score', 0)
        }
        
        # 添加symbol_x和symbol_y（如果pairs_params可用）
        if pairs_params and signal['pair'] in pairs_params:
            pair_info = pairs_params[signal['pair']]
            formatted_signal['symbol_x'] = pair_info.get('symbol_x')
            formatted_signal['symbol_y'] = pair_info.get('symbol_y')
        signals_by_date[date].append(formatted_signal)
    
    # 设置仓位权重（每个配对5%）
    position_weights = {}
    for pair in signals_df['pair'].unique():
        position_weights[pair] = 0.05
    engine.position_weights = position_weights
    
    # 逐日处理
    processed_signals = 0
    for current_date in trading_dates:
        # 获取当前价格
        current_prices = {}
        for symbol in SYMBOLS:
            col_name = f"{symbol}_close"
            if col_name in price_data.columns:
                current_prices[symbol] = price_data.loc[current_date, col_name]
        
        # 处理当日信号
        if current_date in signals_by_date:
            for signal in signals_by_date[current_date]:
                if engine.execute_signal(signal, current_prices, current_date):
                    processed_signals += 1
        
        # 风险管理 - 检查并执行止损等风险控制
        force_close_list = engine.run_risk_management(current_date, current_prices)
        
        # 执行强制平仓
        for item in force_close_list:
            pair = item['pair']
            reason = item['reason']
            if pair in engine.position_manager.positions:
                # 对于止损和强制平仓，Z-score不可用，使用NaN
                engine._close_position(pair, current_prices, reason, current_date, np.nan)
        
        # 逐日盯市结算
        if current_prices:
            engine.position_manager.daily_settlement(current_prices)
            engine.equity_curve.append(engine.position_manager.total_equity)
    
    logger.info(f"回测完成: 处理了 {processed_signals} 个信号")
    
    # 计算绩效指标
    results = engine.calculate_metrics()
    
    # 保存交易记录
    if engine.trade_records:
        trades_df = pd.DataFrame(engine.trade_records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_file = OUTPUT_DIR / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"交易记录保存至: {trades_file}")
        
        # 输出统计
        logger.info("=" * 60)
        logger.info("回测统计")
        logger.info("=" * 60)
        logger.info(f"总交易数: {results.get('total_trades', 0)}")
        logger.info(f"盈利交易: {results.get('winning_trades', 0)}")
        logger.info(f"亏损交易: {results.get('losing_trades', 0)}")
        logger.info(f"总净盈亏: {results.get('total_pnl', 0):,.2f}")
        logger.info(f"总收益率: {results.get('total_return', 0):.2%}")
        logger.info(f"年化收益率: {results.get('annual_return', 0):.2%}")
        logger.info(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"最大回撤: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"胜率: {results.get('win_rate', 0):.2%}")
        logger.info(f"盈亏比: {results.get('profit_loss_ratio', 0):.2f}")
    else:
        logger.warning("没有交易记录")
    
    # 保存回测报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = OUTPUT_DIR / f"backtest_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"回测报告保存至: {report_file}")
    
    return results

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    try:
        print("=" * 80)
        print(f"  {PIPELINE_NAME} v{PIPELINE_VERSION}")
        print("=" * 80)
        print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("时间配置:")
        for key, value in TIME_CONFIG.items():
            print(f"  {key}: {value}")
        print()
        print("协整筛选条件:")
        print(f"  4年p值 < {COINT_CONFIG['p_threshold_4y']}")
        print(f"  1年p值 < {COINT_CONFIG['p_threshold_1y']}")
        print()
        
        # 步骤1: 协整筛选
        pairs_df = step1_cointegration_screening()
        if pairs_df.empty:
            logger.error("协整筛选未找到配对，退出")
            return 1
        
        # 步骤2: 计算初始Beta
        initial_params = step2_calculate_initial_betas(pairs_df)
        if not initial_params:
            logger.error("无法计算初始Beta，退出")
            return 1
        
        # 步骤3: 信号生成
        signals_df = step3_signal_generation(initial_params)
        if signals_df.empty:
            logger.error("未生成信号，退出")
            return 1
        
        # 步骤4: 回测执行
        results = step4_backtest_execution(signals_df, initial_params)
        
        # 生成报告
        logger.info("=" * 60)
        logger.info("管道执行完成")
        logger.info("=" * 60)
        
        # 总结
        print("\n执行总结:")
        print(f"  协整配对数: {len(pairs_df)}")
        print(f"  生成信号数: {len(signals_df)}")
        print(f"  总交易数: {results.get('total_trades', 0)}")
        print(f"  总收益率: {results.get('total_return', 0):.2%}")
        print(f"  年化收益率: {results.get('annual_return', 0):.2%}")
        print(f"  输出目录: {OUTPUT_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())