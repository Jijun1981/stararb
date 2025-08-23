#!/usr/bin/env python3
"""
==============================================================================
                    完整端到端配对交易管道
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

关键原子服务调用：
- lib.data.DataManager: 数据加载和管理
- lib.coint.CointegrationAnalyzer: 协整分析和配对筛选
- lib.signal_generation.SignalGenerator: Kalman滤波和信号生成
- lib.backtest_core: 回测核心计算函数

作者：Claude
日期：2025-08-21
==============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple

# 添加lib路径
sys.path.append('.')

# 导入原子服务
from lib.data import DataManager
from lib.coint import CointegrationAnalyzer, determine_direction
from lib.signal_generation import SignalGenerator, KalmanFilter1D, calculate_ols_beta
from lib.backtest import BacktestEngine
from lib.backtest_core import calculate_min_lots, apply_slippage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局参数配置
# ==============================================================================

# 品种列表（14个金属期货）
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'  # 黑色系
]

# 时间参数
OLS_START = '2023-01-01'  # OLS计算起始
OLS_END = '2023-12-31'    # OLS计算结束
CONVERGENCE_START = '2024-01-01'  # Kalman收敛期开始
CONVERGENCE_END = '2024-06-30'    # Kalman收敛期结束
SIGNAL_START = '2024-07-01'        # 信号生成期开始
BACKTEST_END = '2025-08-20'        # 回测结束

# 协整筛选参数
P_VALUE_THRESHOLD = 0.05  # p值阈值
HALFLIFE_MIN = 2          # 最小半衰期
HALFLIFE_MAX = 60         # 最大半衰期

# 信号生成参数
Z_OPEN = 2.2   # 开仓Z-score阈值
Z_CLOSE = 0.3  # 平仓Z-score阈值
WINDOW = 60    # 滚动窗口

# Beta约束 - 使用绝对值
BETA_MIN = 0.3  # 绝对值最小值
BETA_MAX = 3.0  # 绝对值最大值

# 回测参数
INITIAL_CAPITAL = 5000000  # 初始资金
MARGIN_RATE = 0.12         # 保证金率
STOP_LOSS_PCT = 0.15       # 止损比例（相对保证金）
MAX_HOLDING_DAYS = 30      # 最大持仓天数
COMMISSION_RATE = 0.0002   # 手续费率（双边）
SLIPPAGE_TICKS = 3         # 滑点tick数
MAX_TOTAL_LOTS = 10        # 最大总手数

# 合约参数
MULTIPLIERS = {
    'CU0': 5, 'AL0': 5, 'ZN0': 5, 'PB0': 5, 'NI0': 1, 'SN0': 1,
    'AU0': 1000, 'AG0': 15, 'I0': 100, 'RB0': 10, 'HC0': 10,
    'SS0': 5, 'SF0': 5, 'SM0': 5
}

TICK_SIZES = {
    'CU0': 10, 'AL0': 5, 'ZN0': 5, 'PB0': 5, 'NI0': 10, 'SN0': 10,
    'AU0': 0.02, 'AG0': 1, 'I0': 0.5, 'RB0': 1, 'HC0': 1,
    'SS0': 5, 'SF0': 2, 'SM0': 2
}

# ==============================================================================
# 步骤1: 协整筛选
# ==============================================================================

def screen_cointegrated_pairs(data_manager: DataManager) -> pd.DataFrame:
    """
    筛选协整配对
    
    使用lib.coint.CointegrationAnalyzer进行协整分析
    筛选条件：5年和1年p值都<0.05
    方向判定：基于最近一年的波动率
    """
    logger.info("=" * 60)
    logger.info("步骤1: 协整筛选")
    logger.info("=" * 60)
    
    # 先加载所有品种的数据
    all_data = {}
    for symbol in SYMBOLS:
        try:
            data = data_manager.load_from_parquet(symbol)
            if data is not None and len(data) > 0:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')['close']
                all_data[symbol] = np.log(data)  # 对数价格
        except Exception as e:
            logger.warning(f"加载{symbol}失败: {e}")
    
    # 创建对齐的DataFrame
    price_df = pd.DataFrame(all_data)
    price_df = price_df.dropna()
    
    # 筛选2019年以来的数据（确保有5年数据）
    if len(price_df) > 0:
        price_df = price_df[price_df.index >= pd.Timestamp('2019-01-01')]
    
    logger.info(f"加载数据: {len(price_df)}天, {len(price_df.columns)}个品种")
    
    # 创建协整分析器
    analyzer = CointegrationAnalyzer(data=price_df)
    
    # 测试所有配对
    results = []
    pair_count = 0
    
    for i in range(len(SYMBOLS)):
        for j in range(i+1, len(SYMBOLS)):
            symbol_1 = SYMBOLS[i]
            symbol_2 = SYMBOLS[j]
            pair_count += 1
            
            try:
                # 获取数据（已经是对数价格）
                if symbol_1 not in price_df.columns or symbol_2 not in price_df.columns:
                    continue
                    
                log_prices_1 = price_df[symbol_1].dropna()
                log_prices_2 = price_df[symbol_2].dropna()
                
                # 找共同日期
                common_dates = log_prices_1.index.intersection(log_prices_2.index)
                if len(common_dates) < 252:  # 至少需要1年数据
                    continue
                
                # 对齐数据
                log_prices_1 = log_prices_1.loc[common_dates]
                log_prices_2 = log_prices_2.loc[common_dates]
                
                # 确定方向（基于最近一年的波动率）
                direction, symbol_x, symbol_y = determine_direction(
                    log_prices_1.values, log_prices_2.values,
                    common_dates, common_dates,
                    symbol_1, symbol_2,
                    start_date='2024-01-01'  # 使用最近一年
                )
                
                # 根据方向排列数据
                if direction == 'y_on_x':
                    x = log_prices_1.values if symbol_x == symbol_1 else log_prices_2.values
                    y = log_prices_2.values if symbol_y == symbol_2 else log_prices_1.values
                else:
                    x = log_prices_2.values if symbol_x == symbol_2 else log_prices_1.values
                    y = log_prices_1.values if symbol_y == symbol_1 else log_prices_2.values
                
                # 多窗口协整检验
                from lib.coint import multi_window_test, calculate_halflife
                test_results = multi_window_test(x, y)
                
                # 检查5年和1年p值
                if '5y' in test_results and test_results['5y']:
                    p_5y = test_results['5y'].get('pvalue', 1.0)  # 注意是pvalue不是p_value
                    beta_5y = test_results['5y'].get('beta', np.nan)
                    # 计算半衰期
                    residuals_5y = test_results['5y'].get('residuals', None)
                    halflife_5y = calculate_halflife(residuals_5y) if residuals_5y is not None else np.nan
                else:
                    p_5y = 1.0
                    beta_5y = np.nan
                    halflife_5y = np.nan
                    
                if '1y' in test_results and test_results['1y']:
                    p_1y = test_results['1y'].get('pvalue', 1.0)  # 注意是pvalue不是p_value
                    beta_1y = test_results['1y'].get('beta', np.nan)
                    # 计算半衰期
                    residuals_1y = test_results['1y'].get('residuals', None)
                    halflife_1y = calculate_halflife(residuals_1y) if residuals_1y is not None else np.nan
                else:
                    p_1y = 1.0
                    beta_1y = np.nan
                    halflife_1y = np.nan
                
                # 调试输出
                if p_5y < 0.1 or p_1y < 0.1:
                    logger.debug(f"{symbol_x}-{symbol_y}: p_5y={p_5y:.4f}, p_1y={p_1y:.4f}")
                
                if p_5y < P_VALUE_THRESHOLD and p_1y < P_VALUE_THRESHOLD:
                    # 半衰期检查 - 只检查1年的半衰期，放松5年的
                    if halflife_5y is None:
                        halflife_5y = np.nan
                    if halflife_1y is None:
                        halflife_1y = np.nan
                    
                    # 修改：只要求1年半衰期在合理范围内
                    if HALFLIFE_MIN <= halflife_1y <= HALFLIFE_MAX:
                        
                        results.append({
                            'pair': f"{symbol_x}-{symbol_y}",
                            'symbol_x': symbol_x,
                            'symbol_y': symbol_y,
                            'direction': direction,
                            'p_value_5y': p_5y,
                            'p_value_1y': p_1y,
                            'beta_5y': beta_5y,
                            'beta_1y': beta_1y,
                            'halflife_5y': halflife_5y,
                            'halflife_1y': halflife_1y
                        })
                        
                        logger.info(f"通过筛选: {symbol_x}-{symbol_y}, "
                                  f"p_5y={p_5y:.4f}, p_1y={p_1y:.4f}, "
                                  f"HL_5y={halflife_5y:.1f}, HL_1y={halflife_1y:.1f}")
                        
            except Exception as e:
                logger.warning(f"配对{symbol_1}-{symbol_2}分析失败: {e}")
                continue
    
    pairs_df = pd.DataFrame(results)
    logger.info(f"\n筛选结果: {len(pairs_df)}/{pair_count}个配对通过")
    
    # 保存结果
    output_file = f'output/cointegrated_pairs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    pairs_df.to_csv(output_file, index=False)
    logger.info(f"配对结果保存至: {output_file}")
    
    return pairs_df

# ==============================================================================
# 步骤2: 计算初始Beta（2023年OLS）
# ==============================================================================

def calculate_initial_betas(pairs_df: pd.DataFrame, data_manager: DataManager) -> Dict:
    """
    计算2023年OLS beta作为Kalman初始值
    
    使用lib.signal_generation.calculate_ols_beta
    """
    logger.info("=" * 60)
    logger.info("步骤2: 计算初始Beta（2023年OLS）")
    logger.info("=" * 60)
    
    initial_params = {}
    
    for _, row in pairs_df.iterrows():
        pair = row['pair']
        symbol_x = row['symbol_x']
        symbol_y = row['symbol_y']
        
        try:
            # 加载2023年数据
            data_x = data_manager.load_from_parquet(symbol_x)
            data_y = data_manager.load_from_parquet(symbol_y)
            
            # 转换日期
            data_x['date'] = pd.to_datetime(data_x['date'])
            data_y['date'] = pd.to_datetime(data_y['date'])
            
            # 筛选2023年数据
            mask_x = (data_x['date'] >= OLS_START) & (data_x['date'] <= OLS_END)
            mask_y = (data_y['date'] >= OLS_START) & (data_y['date'] <= OLS_END)
            
            data_x_2023 = data_x[mask_x].set_index('date')
            data_y_2023 = data_y[mask_y].set_index('date')
            
            # 对齐数据
            common_dates = data_x_2023.index.intersection(data_y_2023.index)
            
            if len(common_dates) < 100:  # 至少需要100天数据
                logger.warning(f"{pair}: 2023年数据不足({len(common_dates)}天)")
                continue
            
            # 计算对数价格
            log_x = np.log(data_x_2023.loc[common_dates, 'close'].values)
            log_y = np.log(data_y_2023.loc[common_dates, 'close'].values)
            
            # 计算OLS beta（使用全年数据）
            initial_beta = calculate_ols_beta(log_y, log_x, window=len(log_x))
            
            # 计算残差方差（作为Kalman的观测噪声R）
            residuals = log_y - initial_beta * log_x
            residual_var = np.var(residuals)
            
            initial_params[pair] = {
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                'initial_beta': initial_beta,
                'residual_var': residual_var,
                'num_obs': len(common_dates)
            }
            
            logger.info(f"{pair}: β={initial_beta:.4f}, R={residual_var:.6f}, n={len(common_dates)}")
            
        except Exception as e:
            logger.error(f"{pair} 初始Beta计算失败: {e}")
            continue
    
    logger.info(f"成功计算{len(initial_params)}个配对的初始参数")
    return initial_params

# ==============================================================================
# 步骤3: Kalman预热和信号生成
# ==============================================================================

def generate_signals_with_kalman(initial_params: Dict, data_manager: DataManager) -> pd.DataFrame:
    """
    使用Kalman滤波生成信号
    
    1. 2024年1-6月：收敛期，只更新beta不生成信号
    2. 2024年7月起：生成交易信号
    """
    logger.info("=" * 60)
    logger.info("步骤3: Kalman预热和信号生成")
    logger.info("=" * 60)
    
    # 创建信号生成器
    signal_generator = SignalGenerator(
        window=WINDOW,
        z_open=Z_OPEN,
        z_close=Z_CLOSE,
        convergence_days=20,
        convergence_threshold=0.01
    )
    
    all_signals = []
    
    for pair, params in initial_params.items():
        symbol_x = params['symbol_x']
        symbol_y = params['symbol_y']
        
        try:
            # 加载完整数据（2024年起）
            data_x = data_manager.load_from_parquet(symbol_x)
            data_y = data_manager.load_from_parquet(symbol_y)
            
            # 转换日期
            data_x['date'] = pd.to_datetime(data_x['date'])
            data_y['date'] = pd.to_datetime(data_y['date'])
            
            # 筛选2024年起数据
            mask_x = data_x['date'] >= CONVERGENCE_START
            mask_y = data_y['date'] >= CONVERGENCE_START
            
            data_x_full = data_x[mask_x].set_index('date')
            data_y_full = data_y[mask_y].set_index('date')
            
            # 对齐数据
            common_dates = data_x_full.index.intersection(data_y_full.index)
            common_dates = sorted(common_dates)
            
            # 准备数据（process_pair_signals期望x和y列）
            pair_data = pd.DataFrame({
                'date': common_dates,
                'x': data_x_full.loc[common_dates, 'close'].values,
                'y': data_y_full.loc[common_dates, 'close'].values
            })
            
            # 生成信号（包含收敛期和信号期）
            signals = signal_generator.process_pair_signals(
                pair_data=pair_data,
                initial_beta=params['initial_beta'],
                convergence_end=CONVERGENCE_END,
                signal_start=SIGNAL_START
            )
            
            if signals is not None and len(signals) > 0:
                # 添加配对信息
                signals['pair'] = pair
                signals['symbol_x'] = symbol_x
                signals['symbol_y'] = symbol_y
                all_signals.append(signals)
                
                # 统计信号
                signal_counts = signals['signal'].value_counts()
                logger.info(f"{pair}: 生成{len(signals)}个信号")
                for sig_type, count in signal_counts.items():
                    if sig_type != 'hold':
                        logger.info(f"  {sig_type}: {count}")
        
        except Exception as e:
            logger.error(f"{pair} 信号生成失败: {e}")
            continue
    
    # 合并所有信号
    if all_signals:
        signals_df = pd.concat(all_signals, ignore_index=True)
        signals_df = signals_df.sort_values(['date', 'pair'])
        
        # 保存信号
        output_file = f'output/signals_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        signals_df.to_csv(output_file, index=False)
        logger.info(f"\n信号保存至: {output_file}")
        logger.info(f"总信号数: {len(signals_df)}")
        logger.info(f"信号类型分布:\n{signals_df['signal'].value_counts()}")
        
        return signals_df
    else:
        logger.warning("未生成任何信号")
        return pd.DataFrame()

# ==============================================================================
# 步骤4: 回测执行
# ==============================================================================

def run_backtest_with_engine(signals_df: pd.DataFrame, data_manager: DataManager) -> Dict:
    """
    执行回测
    
    使用lib.backtest_core的函数进行计算
    处理负Beta的方向问题
    """
    logger.info("=" * 60)
    logger.info("步骤4: 回测执行")
    logger.info("=" * 60)
    
    # Beta约束过滤
    open_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short'])]
    filtered_mask = (abs(open_signals['beta']) >= BETA_MIN) & (abs(open_signals['beta']) <= BETA_MAX)
    filtered_count = (~filtered_mask).sum()
    
    logger.info(f"Beta约束[{BETA_MIN}, {BETA_MAX}]过滤了{filtered_count}个开仓信号")
    
    # 应用过滤
    valid_opens = open_signals[filtered_mask]
    close_signals = signals_df[signals_df['signal'] == 'close']
    valid_signals = pd.concat([valid_opens, close_signals]).sort_values('date')
    
    # 回测变量
    positions = {}
    trades = []
    available_capital = INITIAL_CAPITAL
    
    # 按日期处理信号
    for date in valid_signals['date'].unique():
        day_signals = valid_signals[valid_signals['date'] == date]
        
        # 先处理平仓
        for _, signal in day_signals[day_signals['signal'] == 'close'].iterrows():
            pair = signal['pair']
            if pair in positions:
                position = positions[pair]
                trades.append(close_position(position, signal, data_manager))
                del positions[pair]
        
        # 再处理开仓
        for _, signal in day_signals[day_signals['signal'].isin(['open_long', 'open_short'])].iterrows():
            pair = signal['pair']
            if pair not in positions:  # 避免重复开仓
                position = open_position(signal, data_manager)
                if position:
                    positions[pair] = position
    
    # 强制平仓剩余持仓
    for pair, position in positions.items():
        close_signal = pd.Series({
            'date': pd.Timestamp(BACKTEST_END),
            'pair': pair,
            'signal': 'close',
            'reason': 'force_close'
        })
        trades.append(close_position(position, close_signal, data_manager))
    
    # 计算统计
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # 基础统计
        total_trades = len(trades_df)
        total_pnl = trades_df['net_pnl'].sum()
        return_rate = (total_pnl / INITIAL_CAPITAL) * 100
        
        # 胜率统计
        win_rate_stats = calculate_win_rate_metrics(trades)
        
        # 输出结果
        logger.info(f"\n回测结果:")
        logger.info(f"  交易次数: {total_trades}")
        logger.info(f"  净PnL: {total_pnl:,.2f}")
        logger.info(f"  收益率: {return_rate:.2f}%")
        logger.info(f"  胜率: {win_rate_stats['win_rate']:.1f}%")
        
        # 保存交易记录
        output_file = f'output/trades_complete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(output_file, index=False)
        logger.info(f"\n交易记录保存至: {output_file}")
        
        return {
            'trades_df': trades_df,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'return_rate': return_rate,
            'win_rate': win_rate_stats['win_rate']
        }
    else:
        logger.warning("没有产生任何交易")
        return {}

def open_position(signal: pd.Series, data_manager: DataManager) -> Dict:
    """开仓"""
    pair = signal['pair']
    symbols = pair.split('-')
    symbol_x = symbols[0]
    symbol_y = symbols[1]
    
    try:
        # 获取价格
        data_x = data_manager.load_from_parquet(symbol_x)
        data_y = data_manager.load_from_parquet(symbol_y)
        
        data_x['date'] = pd.to_datetime(data_x['date'])
        data_y['date'] = pd.to_datetime(data_y['date'])
        
        price_x = float(data_x[data_x['date'] == signal['date']]['close'].iloc[0])
        price_y = float(data_y[data_y['date'] == signal['date']]['close'].iloc[0])
        
        # 计算手数（限制总手数）
        lots_info = calculate_lots_with_constraints(signal['beta'], MAX_TOTAL_LOTS)
        
        # 应用滑点
        direction = 'long' if signal['signal'] == 'open_long' else 'short'
        beta = signal['beta']
        
        if beta > 0:  # 正相关：对冲
            if direction == 'long':
                entry_y = apply_slippage(price_y, 'buy', TICK_SIZES[symbol_y], SLIPPAGE_TICKS)
                entry_x = apply_slippage(price_x, 'sell', TICK_SIZES[symbol_x], SLIPPAGE_TICKS)
            else:
                entry_y = apply_slippage(price_y, 'sell', TICK_SIZES[symbol_y], SLIPPAGE_TICKS)
                entry_x = apply_slippage(price_x, 'buy', TICK_SIZES[symbol_x], SLIPPAGE_TICKS)
        else:  # 负相关：同向
            if direction == 'long':
                entry_y = apply_slippage(price_y, 'buy', TICK_SIZES[symbol_y], SLIPPAGE_TICKS)
                entry_x = apply_slippage(price_x, 'buy', TICK_SIZES[symbol_x], SLIPPAGE_TICKS)
            else:
                entry_y = apply_slippage(price_y, 'sell', TICK_SIZES[symbol_y], SLIPPAGE_TICKS)
                entry_x = apply_slippage(price_x, 'sell', TICK_SIZES[symbol_x], SLIPPAGE_TICKS)
        
        return {
            'pair': pair,
            'direction': direction,
            'open_date': signal['date'],
            'beta': beta,
            'lots_y': lots_info['lots_y'],
            'lots_x': lots_info['lots_x'],
            'entry_price_y': entry_y,
            'entry_price_x': entry_x,
            'symbol_y': symbol_y,
            'symbol_x': symbol_x
        }
        
    except Exception as e:
        logger.warning(f"开仓失败 {pair}: {e}")
        return None

def close_position(position: Dict, signal: pd.Series, data_manager: DataManager) -> Dict:
    """平仓"""
    if not position:
        return {}
    
    try:
        # 获取平仓价格
        data_x = data_manager.load_from_parquet(position['symbol_x'])
        data_y = data_manager.load_from_parquet(position['symbol_y'])
        
        data_x['date'] = pd.to_datetime(data_x['date'])
        data_y['date'] = pd.to_datetime(data_y['date'])
        
        price_x = float(data_x[data_x['date'] == signal['date']]['close'].iloc[0])
        price_y = float(data_y[data_y['date'] == signal['date']]['close'].iloc[0])
        
        # 应用滑点
        beta = position['beta']
        
        if beta > 0:  # 正相关：对冲
            if position['direction'] == 'long':
                exit_y = apply_slippage(price_y, 'sell', TICK_SIZES[position['symbol_y']], SLIPPAGE_TICKS)
                exit_x = apply_slippage(price_x, 'buy', TICK_SIZES[position['symbol_x']], SLIPPAGE_TICKS)
            else:
                exit_y = apply_slippage(price_y, 'buy', TICK_SIZES[position['symbol_y']], SLIPPAGE_TICKS)
                exit_x = apply_slippage(price_x, 'sell', TICK_SIZES[position['symbol_x']], SLIPPAGE_TICKS)
        else:  # 负相关：同向
            if position['direction'] == 'long':
                exit_y = apply_slippage(price_y, 'sell', TICK_SIZES[position['symbol_y']], SLIPPAGE_TICKS)
                exit_x = apply_slippage(price_x, 'sell', TICK_SIZES[position['symbol_x']], SLIPPAGE_TICKS)
            else:
                exit_y = apply_slippage(price_y, 'buy', TICK_SIZES[position['symbol_y']], SLIPPAGE_TICKS)
                exit_x = apply_slippage(price_x, 'buy', TICK_SIZES[position['symbol_x']], SLIPPAGE_TICKS)
        
        # 计算PnL
        mult_y = MULTIPLIERS[position['symbol_y']]
        mult_x = MULTIPLIERS[position['symbol_x']]
        
        if beta > 0:  # 正相关
            if position['direction'] == 'long':
                y_pnl = (exit_y - position['entry_price_y']) * position['lots_y'] * mult_y
                x_pnl = (position['entry_price_x'] - exit_x) * position['lots_x'] * mult_x
            else:
                y_pnl = (position['entry_price_y'] - exit_y) * position['lots_y'] * mult_y
                x_pnl = (exit_x - position['entry_price_x']) * position['lots_x'] * mult_x
        else:  # 负相关
            if position['direction'] == 'long':
                y_pnl = (exit_y - position['entry_price_y']) * position['lots_y'] * mult_y
                x_pnl = (exit_x - position['entry_price_x']) * position['lots_x'] * mult_x
            else:
                y_pnl = (position['entry_price_y'] - exit_y) * position['lots_y'] * mult_y
                x_pnl = (position['entry_price_x'] - exit_x) * position['lots_x'] * mult_x
        
        gross_pnl = y_pnl + x_pnl
        
        # 手续费
        commission = ((position['entry_price_y'] + exit_y) * position['lots_y'] * mult_y +
                     (position['entry_price_x'] + exit_x) * position['lots_x'] * mult_x) * COMMISSION_RATE
        
        net_pnl = gross_pnl - commission
        
        return {
            **position,
            'close_date': signal['date'],
            'exit_price_y': exit_y,
            'exit_price_x': exit_x,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'commission': commission,
            'holding_days': (signal['date'] - position['open_date']).days
        }
        
    except Exception as e:
        logger.warning(f"平仓失败 {position['pair']}: {e}")
        return position

def calculate_lots_with_constraints(beta: float, max_total: int) -> Dict:
    """计算受约束的手数"""
    base_result = calculate_min_lots(abs(beta), max_denominator=max_total)
    
    total = base_result['lots_y'] + base_result['lots_x']
    if total > max_total:
        # 寻找最佳近似
        best_y, best_x = 1, 1
        best_error = float('inf')
        
        for y in range(1, max_total):
            for x in range(1, max_total - y + 1):
                if y + x <= max_total:
                    ratio = y / x
                    error = abs(abs(beta) - ratio)
                    if error < best_error:
                        best_error = error
                        best_y, best_x = y, x
        
        return {'lots_y': best_y, 'lots_x': best_x}
    
    return base_result

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数：执行完整管道"""
    logger.info("=" * 80)
    logger.info("                    开始执行完整配对交易管道")
    logger.info("=" * 80)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 步骤1: 协整筛选
    pairs_df = screen_cointegrated_pairs(data_manager)
    if pairs_df.empty:
        logger.error("没有找到协整配对，退出")
        return
    
    # 步骤2: 计算初始Beta
    initial_params = calculate_initial_betas(pairs_df, data_manager)
    if not initial_params:
        logger.error("无法计算初始参数，退出")
        return
    
    # 步骤3: Kalman预热和信号生成
    signals_df = generate_signals_with_kalman(initial_params, data_manager)
    if signals_df.empty:
        logger.error("未生成任何信号，退出")
        return
    
    # 步骤4: 回测执行
    results = run_backtest_with_engine(signals_df, data_manager)
    
    logger.info("=" * 80)
    logger.info("                    管道执行完成")
    logger.info("=" * 80)
    
    return results

if __name__ == "__main__":
    main()