"""
端到端配对交易流水线
按照配置文件运行完整的交易流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 导入核心模块
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGeneratorV3 as SignalGenerator
from lib.backtest.engine import BacktestEngine, BacktestConfig
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.trade_executor import ExecutionConfig
from lib.backtest.risk_manager import RiskConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_cointegration_step(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    """运行协整分析步骤"""
    logger.info("=" * 80)
    logger.info("开始协整分析...")
    
    # 创建协整分析器
    analyzer = CointegrationAnalyzer(data)
    
    # 运行所有配对的协整分析
    from lib.coint import screen_all_pairs
    results = screen_all_pairs(analyzer)
    
    # 筛选满足条件的配对
    valid_pairs = []
    for _, row in results.iterrows():
        # 检查所有窗口的p值
        all_pass = True
        for window in config['cointegration']['screening_windows']:
            pval_col = f'pvalue_{window}'
            if pval_col in row and (pd.isna(row[pval_col]) or row[pval_col] > config['cointegration']['p_value_threshold']):
                all_pass = False
                break
        
        if all_pass:
            valid_pairs.append(row)
    
    pairs_df = pd.DataFrame(valid_pairs)
    logger.info(f"筛选出 {len(pairs_df)} 个满足条件的配对")
    
    # 保存配对结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs_df.to_csv(f'cointegration_pairs_{timestamp}.csv', index=False)
    
    return pairs_df

def generate_signals_step(config: dict, data: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    """生成交易信号"""
    logger.info("=" * 80)
    logger.info("开始生成交易信号...")
    
    # 创建信号生成器（使用正确的参数）
    generator = SignalGenerator(
        signal_start_date=config['time_config']['signal_generation_start'],
        kalman_warmup_days=config['signal_generation']['kalman_warmup'],
        ols_training_days=config['signal_generation']['ols_window'],
        z_open=config['signal_generation']['signal_thresholds']['z_open'],
        z_close=config['signal_generation']['signal_thresholds']['z_close'],
        max_holding_days=config['signal_generation']['signal_thresholds']['max_holding_days'],
        Q_beta=config['signal_generation']['kalman_params']['Q_beta'],
        Q_alpha=config['signal_generation']['kalman_params']['Q_alpha'],
        R_init=config['signal_generation']['kalman_params']['R_init'],
        R_adapt=config['signal_generation']['kalman_params']['R_adapt']
    )
    
    # 设置时间参数
    time_config = config['time_config']
    
    # 生成所有配对的信号
    all_signals = []
    
    for _, pair in pairs_df.iterrows():
        pair_name = pair['pair']
        symbol_x = pair['symbol_x']
        symbol_y = pair['symbol_y']
        
        logger.info(f"处理配对 {pair_name}...")
        
        # 获取配对数据
        if symbol_x not in data.columns or symbol_y not in data.columns:
            logger.warning(f"跳过配对 {pair_name}: 数据不完整")
            continue
        
        px = data[symbol_x]
        py = data[symbol_y]
        
        # 生成信号
        try:
            # 使用process_pair方法（正确的参数名）
            signals = generator.process_pair(
                pair_name=pair_name,
                x_data=px,
                y_data=py,
                initial_beta=pair.get('beta_5y', 1.0)  # 使用5年Beta
            )
            
            if signals is not None and len(signals) > 0:
                # 添加配对信息
                signals['pair'] = pair_name
                signals['symbol_x'] = symbol_x
                signals['symbol_y'] = symbol_y
                all_signals.append(signals)
                
        except Exception as e:
            logger.error(f"配对 {pair_name} 信号生成失败: {e}")
            continue
    
    # 合并所有信号
    if all_signals:
        signals_df = pd.concat(all_signals, ignore_index=True)
        logger.info(f"生成了 {len(signals_df)} 条信号")
        
        # 保存信号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signals_df.to_csv(f'trading_signals_{timestamp}.csv', index=False)
        
        return signals_df
    else:
        logger.warning("没有生成任何信号")
        return pd.DataFrame()

def run_backtest_step(config: dict, data: pd.DataFrame, signals_df: pd.DataFrame) -> dict:
    """运行回测"""
    logger.info("=" * 80)
    logger.info("开始运行回测...")
    
    # 创建回测配置
    sizing_config = PositionSizingConfig(
        max_denominator=config['backtest']['position_sizing']['max_denominator'],
        min_lots=config['backtest']['position_sizing']['min_lots'],
        max_lots_per_leg=config['backtest']['position_sizing']['max_lots_per_leg'],
        margin_rate=config['backtest']['capital_management']['margin_rate'],
        position_weight=config['backtest']['capital_management']['position_weight']
    )
    
    exec_config = ExecutionConfig(
        commission_rate=config['backtest']['trading_costs']['commission_rate'],
        slippage_ticks=config['backtest']['trading_costs']['slippage_ticks'],
        margin_rate=config['backtest']['capital_management']['margin_rate']
    )
    
    risk_config = RiskConfig(
        stop_loss_pct=config['backtest']['risk_management']['stop_loss_pct'],
        max_holding_days=config['backtest']['risk_management']['max_holding_days'],
        max_positions=config['backtest']['risk_management'].get('max_concurrent_positions', 20) or 20
    )
    
    backtest_config = BacktestConfig(
        initial_capital=config['backtest']['capital_management']['initial_capital'],
        sizing_config=sizing_config,
        execution_config=exec_config,
        risk_config=risk_config
    )
    
    # 创建回测引擎
    engine = BacktestEngine(backtest_config)
    
    # 设置合约规格（通过executor）
    engine.executor.set_contract_specs(config['contract_specs'])
    
    # 准备回测数据
    backtest_start = pd.to_datetime(config['time_config']['backtest_start'])
    backtest_end = pd.to_datetime(config['time_config']['backtest_end'])
    
    # 过滤信号到回测期间
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    backtest_signals = signals_df[
        (signals_df['date'] >= backtest_start) & 
        (signals_df['date'] <= backtest_end)
    ].copy()
    
    logger.info(f"回测期间: {backtest_start} 至 {backtest_end}")
    logger.info(f"回测信号数: {len(backtest_signals)}")
    
    # 运行回测
    results = engine.run(
        signals=backtest_signals,
        prices=data[backtest_start:backtest_end]
    )
    
    logger.info("回测完成")
    
    # 保存回测结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存交易记录
    if results['trades']:
        trades_df = pd.DataFrame([
            {
                'pair': trade.pair,
                'direction': trade.direction,
                'open_date': trade.open_date,
                'close_date': trade.close_date,
                'holding_days': trade.holding_days,
                'net_pnl': trade.net_pnl,
                'gross_pnl': trade.gross_pnl,
                'open_commission': trade.open_commission,
                'close_commission': trade.close_commission,
                'total_commission': trade.open_commission + trade.close_commission,
                'return_pct': trade.return_pct,
                'close_reason': trade.close_reason,
                'symbol_x': trade.symbol_x,
                'symbol_y': trade.symbol_y,
                'lots_x': trade.lots_x,
                'lots_y': trade.lots_y,
                'open_price_x': trade.open_price_x,
                'open_price_y': trade.open_price_y,
                'close_price_x': trade.close_price_x,
                'close_price_y': trade.close_price_y
            }
            for trade in results['trades']
        ])
        
        trades_df.to_csv(f'backtest_trades_{timestamp}.csv', index=False)
        logger.info(f"保存交易记录: backtest_trades_{timestamp}.csv")
        
        # 打印交易统计
        logger.info("=" * 80)
        logger.info("交易统计:")
        logger.info(f"总交易数: {len(trades_df)}")
        logger.info(f"盈利交易: {len(trades_df[trades_df['net_pnl'] > 0])}")
        logger.info(f"亏损交易: {len(trades_df[trades_df['net_pnl'] < 0])}")
        logger.info(f"总盈亏: {trades_df['net_pnl'].sum():,.2f}")
        logger.info(f"平均盈亏: {trades_df['net_pnl'].mean():,.2f}")
        
        # 按平仓原因统计
        reason_stats = trades_df.groupby('close_reason').agg({
            'net_pnl': ['count', 'sum', 'mean']
        })
        logger.info("\n按平仓原因统计:")
        print(reason_stats)
    
    # 保存权益曲线
    if results['equity_curve']:
        equity_df = pd.DataFrame({
            'date': pd.date_range(start=backtest_start, periods=len(results['equity_curve']), freq='D'),
            'equity': results['equity_curve']
        })
        equity_df.to_csv(f'backtest_equity_{timestamp}.csv', index=False)
        logger.info(f"保存权益曲线: backtest_equity_{timestamp}.csv")
        
        # 打印绩效指标
        logger.info("=" * 80)
        logger.info("绩效指标:")
        initial_equity = results['equity_curve'][0]
        final_equity = results['equity_curve'][-1]
        total_return = (final_equity - initial_equity) / initial_equity
        logger.info(f"初始资金: {initial_equity:,.0f}")
        logger.info(f"最终资金: {final_equity:,.0f}")
        logger.info(f"总收益率: {total_return:.2%}")
    
    # 保存配对统计
    if results['pair_metrics']:
        pair_metrics_df = pd.DataFrame(results['pair_metrics']).T
        pair_metrics_df.to_csv(f'backtest_pairs_{timestamp}.csv')
        logger.info(f"保存配对统计: backtest_pairs_{timestamp}.csv")
    
    # 保存绩效指标
    metrics = results.get('metrics', {})
    if metrics:
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'backtest_metrics_{timestamp}.csv', index=False)
        logger.info(f"保存绩效指标: backtest_metrics_{timestamp}.csv")
        
        # 打印关键指标
        logger.info("=" * 80)
        logger.info("关键绩效指标:")
        logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"胜率: {metrics.get('win_rate', 0):.2%}")
    
    return results

def main():
    """主函数"""
    # 加载配置
    config_path = 'configs/e2e_pipeline_config.yaml'
    config = load_config(config_path)
    
    logger.info("=" * 80)
    logger.info("端到端配对交易流水线")
    logger.info("=" * 80)
    
    # 步骤1: 加载数据
    logger.info("步骤1: 加载数据...")
    all_symbols = (
        config['symbols']['metals']['precious'] +
        config['symbols']['metals']['nonferrous'] +
        config['symbols']['metals']['ferrous']
    )
    
    data = load_all_symbols_data(
        symbols=all_symbols,
        start_date=config['time_config']['data_start_date'],
        end_date=config['time_config']['data_end_date']
    )
    logger.info(f"加载了 {len(data.columns)} 个品种的数据")
    
    # 步骤2: 协整分析
    pairs_df = run_cointegration_step(config, data)
    
    if pairs_df.empty:
        logger.error("没有找到满足条件的配对，流程终止")
        return
    
    # 步骤3: 生成信号
    signals_df = generate_signals_step(config, data, pairs_df)
    
    if signals_df.empty:
        logger.error("没有生成交易信号，流程终止")
        return
    
    # 步骤4: 运行回测
    backtest_results = run_backtest_step(config, data, signals_df)
    
    logger.info("=" * 80)
    logger.info("流水线执行完成!")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()