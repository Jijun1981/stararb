#!/usr/bin/env python3
"""
==============================================================================
                完整端到端配对交易管道 v2.1 (使用最新原子服务)
==============================================================================

参数配置总览：
==============================================================================

【品种配置】
- 品种数量: 14个金属期货
- 贵金属: AG0(银), AU0(金)
- 有色金属: AL0(铝), CU0(铜), NI0(镍), PB0(铅), SN0(锡), ZN0(锌)
- 黑色系: HC0(热卷), I0(铁矿), RB0(螺纹), SF0(硅铁), SM0(锰硅), SS0(不锈钢)

【时间配置】
- Beta训练期: 2023-01-01 至 2023-12-31 (使用2023年全年数据)
- Kalman收敛期: 2024-01-01 至 2024-06-30 (6个月收敛)
- 信号生成期: 2024-07-01 至 2025-08-20
- 回测结束: 2025-08-20

【协整筛选参数】
- 5年期p值阈值: 0.05
- 1年期p值阈值: 0.05 (双重条件筛选)
- 半衰期筛选: 可选（通过use_halflife_filter参数控制）
  * 5年半衰期范围: [5, 60]天（如启用）
  * 1年半衰期范围: [2, 60]天（如启用）

【信号生成参数】
- 开仓Z-score阈值: 2.2
- 平仓Z-score阈值: 0.3
- 滚动窗口: 60个交易日
- 最大持仓天数: 30天

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

版本说明：
- v2.1: 使用最新的原子服务接口
  * 协整模块: 双重p值筛选（REQ-2.4.3）
  * 信号模块: Kalman滤波动态Beta（REQ-3.1.x）
  * 回测模块: 双算法验证PnL（REQ-4.3.x）

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
from lib.data import load_data  # 使用统一的data-joint数据源
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGeneratorV3
from lib.backtest.engine import BacktestEngine
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.trade_executor import ExecutionConfig
from lib.backtest.risk_manager import RiskConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局配置参数 v2.1
# ==============================================================================

# 版本信息
PIPELINE_VERSION = "2.1"
PIPELINE_NAME = "完整端到端配对交易管道(最新原子服务版)"

# 品种列表（14个金属期货）
SYMBOLS = [
    'AG0', 'AU0',  # 贵金属
    'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',  # 有色金属
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'  # 黑色系
]

# 时间配置
TIME_CONFIG = {
    'data_start': '2020-01-01',  # 数据起始（用于协整分析）
    'beta_training_start': '2023-01-01',  # Beta训练开始
    'beta_training_end': '2023-12-31',     # Beta训练结束
    'convergence_start': '2024-01-01',     # Kalman收敛期开始
    'convergence_end': '2024-06-30',       # Kalman收敛期结束
    'signal_start': '2024-07-01',          # 信号生成开始
    'backtest_end': '2025-08-20'           # 回测结束
}

# 协整筛选参数（REQ-2.4.3: 双重条件筛选）
COINT_CONFIG = {
    'p_threshold': 0.05,  # 同时要求5年和1年p值 < 0.05
    'use_halflife_filter': False,  # 是否启用半衰期筛选（默认不启用）
    'halflife_min': 5,     # 最小半衰期（天）
    'halflife_max': 60,    # 最大半衰期（天）
}

# 信号生成参数（REQ-3.3.x）
SIGNAL_CONFIG = {
    'z_open': 2.2,
    'z_close': 0.3,
    'window': 60,
    'max_holding_days': 30,
    'convergence_threshold': 0.01  # 1%收敛阈值
}

# Beta约束参数
BETA_CONFIG = {
    'min_abs': 0.3,
    'max_abs': 3.0
}

# 回测参数（REQ-4.x.x）
BACKTEST_CONFIG = {
    'initial_capital': 5000000,
    'margin_rate': 0.12,         # REQ-4.1.5
    'commission_rate': 0.0002,   # REQ-4.1.6
    'slippage_ticks': 3,         # REQ-4.1.4
    'stop_loss_pct': 0.15,       # REQ-4.2.3: 15%止损
    'max_holding_days': 30       # REQ-4.2.4: 30天强平
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

# 创建输出目录
OUTPUT_DIR = Path("output/pipeline_v21")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 步骤1: 协整筛选（使用最新的双重条件筛选）
# ==============================================================================

def step1_cointegration_screening() -> pd.DataFrame:
    """
    步骤1: 协整筛选
    使用最新的协整模块，实现REQ-2.4.3双重条件筛选
    """
    logger.info("=" * 60)
    logger.info("步骤1: 协整配对筛选")
    logger.info("=" * 60)
    
    # 加载数据（对数价格）
    logger.info(f"加载数据: {TIME_CONFIG['data_start']} 至今")
    data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['data_start'],
        columns=['close'],
        log_price=True
    )
    
    # 创建协整分析器
    analyzer = CointegrationAnalyzer(data)
    
    # 筛选配对（REQ-2.4.3: 双重条件筛选）
    if COINT_CONFIG['use_halflife_filter']:
        logger.info(f"执行协整筛选 (p值阈值: {COINT_CONFIG['p_threshold']}, "
                   f"半衰期: [{COINT_CONFIG['halflife_min']}, {COINT_CONFIG['halflife_max']}])")
    else:
        logger.info(f"执行协整筛选 (p值阈值: {COINT_CONFIG['p_threshold']}, 不使用半衰期筛选)")
    
    # 调用更新后的screen_all_pairs方法
    filtered_pairs = analyzer.screen_all_pairs(
        p_threshold=COINT_CONFIG['p_threshold'],
        halflife_min=COINT_CONFIG['halflife_min'] if COINT_CONFIG['use_halflife_filter'] else None,
        halflife_max=COINT_CONFIG['halflife_max'] if COINT_CONFIG['use_halflife_filter'] else None,
        use_halflife_filter=COINT_CONFIG['use_halflife_filter']
    )
    
    if len(filtered_pairs) == 0:
        logger.error("未找到符合条件的协整配对!")
        return pd.DataFrame()
    
    logger.info(f"筛选结果:")
    logger.info(f"  总配对数: {len(SYMBOLS) * (len(SYMBOLS) - 1) // 2}")
    logger.info(f"  通过筛选条件: {len(filtered_pairs)}")
    
    # 显示前10个配对
    if len(filtered_pairs) > 0:
        logger.info("\n前10个配对:")
        for i, row in filtered_pairs.head(10).iterrows():
            logger.info(f"  {i+1}. {row['pair']}: "
                       f"1年p={row['pvalue_1y']:.4f}, "
                       f"5年p={row['pvalue_5y']:.4f}, "
                       f"β={row['beta_5y']:.4f}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"cointegrated_pairs_{timestamp}.csv"
    filtered_pairs.to_csv(output_file, index=False)
    logger.info(f"协整配对保存至: {output_file}")
    
    return filtered_pairs

# ==============================================================================
# 步骤2: 计算初始Beta（使用2023年数据）
# ==============================================================================

def step2_calculate_initial_betas(pairs_df: pd.DataFrame) -> Dict:
    """
    步骤2: 计算初始Beta
    使用2023年数据计算OLS beta作为Kalman滤波初始值
    """
    logger.info("=" * 60)
    logger.info("步骤2: 计算初始Beta (2023年数据)")
    logger.info("=" * 60)
    
    # 加载2023年数据
    data_2023 = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['beta_training_start'],
        end_date=TIME_CONFIG['beta_training_end'],
        columns=['close'],
        log_price=True
    )
    
    initial_params = {}
    
    for _, row in pairs_df.iterrows():
        pair = row['pair']
        symbol_x = row['symbol_x'].replace('_close', '')
        symbol_y = row['symbol_y'].replace('_close', '')
        
        try:
            # 获取2023年对数价格
            log_x = data_2023[f"{symbol_x}_close"].values
            log_y = data_2023[f"{symbol_y}_close"].values
            
            # 计算OLS beta（使用全年数据）
            initial_beta = calculate_ols_beta(log_y, log_x, window=len(log_x))
            
            if np.isnan(initial_beta):
                logger.warning(f"{pair}: 无法计算初始Beta")
                continue
            
            # 计算残差方差（作为Kalman的观测噪声R）
            residuals = log_y - initial_beta * log_x
            residual_var = np.var(residuals)
            
            initial_params[pair] = {
                'beta_initial': initial_beta,
                'symbol_x': f"{symbol_x}_close",
                'symbol_y': f"{symbol_y}_close",
                'R': residual_var
            }
            
            logger.info(f"{pair}: β_2023={initial_beta:.4f}, R={residual_var:.6f}")
            
        except Exception as e:
            logger.error(f"{pair} 初始Beta计算失败: {e}")
    
    logger.info(f"成功计算 {len(initial_params)} 个配对的初始Beta")
    return initial_params

# ==============================================================================
# 步骤3: 信号生成（使用Kalman滤波）
# ==============================================================================

def step3_signal_generation(pairs_params: Dict) -> pd.DataFrame:
    """
    步骤3: 信号生成
    使用Kalman滤波动态更新Beta（REQ-3.1.x）
    """
    logger.info("=" * 60)
    logger.info("步骤3: 信号生成 (Kalman滤波)")
    logger.info("=" * 60)
    
    # 初始化信号生成器
    generator = SignalGenerator(
        window=SIGNAL_CONFIG['window'],
        z_open=SIGNAL_CONFIG['z_open'],
        z_close=SIGNAL_CONFIG['z_close'],
        convergence_threshold=SIGNAL_CONFIG['convergence_threshold']
    )
    
    # 加载完整价格数据
    price_data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['beta_training_start'],
        columns=['close'],
        log_price=False  # 原始价格
    )
    
    # 重置索引使date成为列
    price_data = price_data.reset_index()
    price_data.rename(columns={'index': 'date'}, inplace=True)
    
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
    
    # 保存信号
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"signals_{timestamp}.csv"
    all_signals.to_csv(output_file, index=False)
    logger.info(f"信号保存至: {output_file}")
    
    return all_signals

# ==============================================================================
# 步骤4: 回测执行（使用最新的回测引擎）
# ==============================================================================

def step4_backtest_execution(signals_df: pd.DataFrame) -> Dict:
    """
    步骤4: 回测执行
    使用最新的回测引擎，包含双算法PnL验证
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
    
    # 加载合约规格
    specs_file = project_root / "configs" / "contract_specs.json"
    if specs_file.exists():
        engine.load_contract_specs(str(specs_file))
    else:
        engine.contract_specs = CONTRACT_SPECS
    
    # 加载价格数据
    logger.info("加载回测价格数据...")
    price_data = load_data(
        symbols=SYMBOLS,
        start_date=TIME_CONFIG['signal_start'],
        end_date=TIME_CONFIG['backtest_end'],
        columns=['close'],
        log_price=False
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
        
        formatted_signal = {
            'pair': signal['pair'],
            'signal': signal_type,
            'date': date,
            'beta': signal.get('beta', 1.0),
            'z_score': signal.get('z_score', 0)
        }
        signals_by_date[date].append(formatted_signal)
    
    # 逐日处理
    processed_signals = 0
    for current_date in trading_dates:
        # 获取当前价格
        current_prices = {}
        for symbol in SYMBOLS:
            col_name = f"{symbol}_close"
            if col_name in price_data.columns:
                current_prices[col_name] = price_data.loc[current_date, col_name]
        
        # 检查风险控制（止损、强平）
        if current_prices:
            pairs_to_close = engine.run_risk_management(current_date, current_prices)
            if pairs_to_close:
                for pair_info in pairs_to_close:
                    logger.info(f"{current_date}: 风控平仓 {pair_info}")
                    # 实际执行平仓 - 使用_close_position内部方法
                    engine._close_position(
                        pair=pair_info['pair'],
                        current_prices=current_prices,
                        current_date=current_date,
                        reason=pair_info['reason']
                    )
        
        # 处理当日信号
        if current_date in signals_by_date:
            for signal in signals_by_date[current_date]:
                if engine.execute_signal(signal, current_prices, current_date):
                    processed_signals += 1
        
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
        results = step4_backtest_execution(signals_df)
        
        # 生成报告
        logger.info("=" * 60)
        logger.info("管道执行完成")
        logger.info("=" * 60)
        
        # 保存最终报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report = {
            'pipeline_version': PIPELINE_VERSION,
            'execution_time': timestamp,
            'pairs_count': len(pairs_df),
            'signals_count': len(signals_df),
            **results
        }
        
        report_file = OUTPUT_DIR / f"pipeline_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"最终报告保存至: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())