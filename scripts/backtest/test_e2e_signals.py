#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端信号回测脚本
使用generate_signals_e2e.py生成的信号进行完整回测
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.data import load_all_symbols_data
from lib.backtest.engine import BacktestEngine, BacktestConfig
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.trade_executor import ExecutionConfig
from lib.backtest.risk_manager import RiskConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    # ================== 1. 加载信号数据 ==================
    logger.info("=" * 60)
    logger.info("步骤1: 加载E2E信号数据")
    logger.info("=" * 60)
    
    # 查找最新的信号文件
    signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
    if not signal_files:
        logger.error("未找到信号文件！请先运行generate_signals_e2e.py")
        return
    
    latest_signal_file = sorted(signal_files)[-1]
    logger.info(f"加载信号文件: {latest_signal_file}")
    
    try:
        signals_df = pd.read_csv(latest_signal_file)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        
        logger.info(f"成功加载信号: {len(signals_df)}条记录")
        logger.info(f"信号时间范围: {signals_df['date'].min()} 至 {signals_df['date'].max()}")
        logger.info(f"涉及配对: {signals_df['pair'].nunique()}个")
        
        # 筛选交易信号
        trade_signals = signals_df[signals_df['signal'].isin(['open_long', 'open_short', 'close'])].copy()
        logger.info(f"实际交易信号: {len(trade_signals)}条")
        
        # 重命名列以匹配回测引擎的期望
        trade_signals = trade_signals.rename(columns={'signal': 'trade_signal'})
        
        # 统计信号分布
        signal_counts = trade_signals['trade_signal'].value_counts()
        logger.info("信号分布:")
        for signal_type, count in signal_counts.items():
            logger.info(f"  {signal_type}: {count}条")
            
    except Exception as e:
        logger.error(f"加载信号文件失败: {e}")
        return
    
    # ================== 2. 加载价格数据 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤2: 加载价格数据")
    logger.info("=" * 60)
    
    try:
        # 加载完整价格数据（对数价格）
        log_price_data = load_all_symbols_data()
        
        # 转换为原始价格（回测需要原始价格）
        price_data = np.exp(log_price_data)
        
        # 截取信号期间的数据（包含预热期）
        signal_start = signals_df['date'].min()
        signal_end = signals_df['date'].max()
        
        # 往前推90天作为预热期
        data_start = signal_start - pd.Timedelta(days=120)  # 多留一些余量
        
        backtest_prices = price_data[data_start:signal_end].copy()
        
        logger.info(f"价格数据范围: {backtest_prices.index[0]} 至 {backtest_prices.index[-1]}")
        logger.info(f"数据点数: {len(backtest_prices)}")
        
        # 检查信号中的品种是否都有价格数据
        signal_symbols = set()
        for _, row in trade_signals.iterrows():
            if pd.notna(row.get('symbol_x')):
                signal_symbols.add(row['symbol_x'])
            if pd.notna(row.get('symbol_y')):
                signal_symbols.add(row['symbol_y'])
        
        missing_symbols = signal_symbols - set(backtest_prices.columns)
        if missing_symbols:
            logger.warning(f"缺少价格数据的品种: {missing_symbols}")
        
        available_symbols = signal_symbols & set(backtest_prices.columns)
        logger.info(f"可用于回测的品种: {len(available_symbols)}个")
        
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        return
    
    # ================== 3. 准备合约规格 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤3: 设置合约规格")
    logger.info("=" * 60)
    
    # 14个金属期货品种的合约规格
    contract_specs = {
        # 贵金属
        'AG': {'multiplier': 15, 'tick_size': 1},     # 白银
        'AU': {'multiplier': 1000, 'tick_size': 0.01}, # 黄金
        
        # 有色金属
        'AL': {'multiplier': 5, 'tick_size': 5},      # 铝
        'CU': {'multiplier': 5, 'tick_size': 10},     # 铜
        'NI': {'multiplier': 1, 'tick_size': 10},     # 镍
        'PB': {'multiplier': 5, 'tick_size': 5},      # 铅
        'SN': {'multiplier': 1, 'tick_size': 10},     # 锡
        'ZN': {'multiplier': 5, 'tick_size': 5},      # 锌
        
        # 黑色系
        'HC': {'multiplier': 10, 'tick_size': 1},     # 热卷
        'I': {'multiplier': 100, 'tick_size': 0.5},   # 铁矿石
        'RB': {'multiplier': 10, 'tick_size': 1},     # 螺纹钢
        'SF': {'multiplier': 5, 'tick_size': 2},      # 硅铁
        'SM': {'multiplier': 5, 'tick_size': 2},      # 锰硅
        'SS': {'multiplier': 5, 'tick_size': 5}       # 不锈钢
    }
    
    logger.info(f"合约规格已配置: {len(contract_specs)}个品种")
    
    # ================== 4. 配置回测参数 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤4: 配置回测参数")
    logger.info("=" * 60)
    
    # 回测配置
    config = BacktestConfig(
        initial_capital=5000000,  # 500万初始资金
        sizing_config=PositionSizingConfig(
            max_denominator=10,       # 最大分母
            min_lots=1,              # 最小手数
            max_lots_per_leg=100,    # 每腿最大手数
            margin_rate=0.12,        # 12%保证金率
            position_weight=0.05     # 5%仓位权重
        ),
        execution_config=ExecutionConfig(
            commission_rate=0.0002,   # 万分之2手续费
            slippage_ticks=3,         # 3个tick滑点
            margin_rate=0.12          # 12%保证金率
        ),
        risk_config=RiskConfig(
            stop_loss_pct=0.10,       # 10%止损
            max_holding_days=30,      # 最大持仓30天
            max_positions=20          # 最大同时持仓20个
        )
    )
    
    logger.info("回测配置:")
    logger.info(f"  初始资金: {config.initial_capital:,.0f}元")
    logger.info(f"  仓位权重: {config.sizing_config.position_weight:.1%}")
    logger.info(f"  手续费率: {config.execution_config.commission_rate:.4f}")
    logger.info(f"  止损比例: {config.risk_config.stop_loss_pct:.1%}")
    logger.info(f"  最大持仓天数: {config.risk_config.max_holding_days}天")
    
    # ================== 5. 运行回测 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤5: 运行回测")
    logger.info("=" * 60)
    
    try:
        # 创建回测引擎
        engine = BacktestEngine(config)
        
        # 运行回测
        logger.info("开始执行回测...")
        results = engine.run(
            signals=trade_signals,
            prices=backtest_prices,
            contract_specs=contract_specs
        )
        
        logger.info("回测执行完成！")
        
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ================== 6. 分析结果 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤6: 回测结果分析")
    logger.info("=" * 60)
    
    # 基本统计
    logger.info(f"初始资金: {results['initial_capital']:,.0f}元")
    logger.info(f"最终资金: {results['final_capital']:,.0f}元")
    logger.info(f"总收益: {results['final_capital'] - results['initial_capital']:,.0f}元")
    logger.info(f"总收益率: {(results['final_capital'] / results['initial_capital'] - 1) * 100:.2f}%")
    logger.info(f"总交易数: {len(results['trades'])}")
    
    # 组合绩效指标
    portfolio_metrics = results.get('portfolio_metrics', {})
    if portfolio_metrics:
        logger.info("\n组合绩效指标:")
        logger.info(f"  年化收益率: {portfolio_metrics.get('annual_return', 0):.2%}")
        logger.info(f"  波动率: {portfolio_metrics.get('volatility', 0):.2%}")
        logger.info(f"  夏普比率: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  索提诺比率: {portfolio_metrics.get('sortino_ratio', 0):.3f}")
        logger.info(f"  最大回撤: {portfolio_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  胜率: {portfolio_metrics.get('win_rate', 0):.1%}")
        logger.info(f"  平均盈利: {portfolio_metrics.get('avg_win_pnl', 0):,.0f}元")
        logger.info(f"  平均亏损: {portfolio_metrics.get('avg_loss_pnl', 0):,.0f}元")
    
    # 配对级别绩效
    pair_metrics = results.get('pair_metrics', pd.DataFrame())
    if not pair_metrics.empty:
        logger.info(f"\n配对级别绩效 (共{len(pair_metrics)}个配对):")
        
        # 按收益贡献排序，显示前5名
        top_pairs = pair_metrics.nlargest(5, 'contribution')
        logger.info("  收益贡献前5名:")
        for _, row in top_pairs.iterrows():
            logger.info(f"    {row['pair']}: {row['contribution']:,.0f}元 "
                      f"(夏普{row['sharpe_ratio']:.2f}, 胜率{row['win_rate']:.1%})")
        
        # 显示后5名
        bottom_pairs = pair_metrics.nsmallest(5, 'contribution')
        logger.info("  收益贡献后5名:")
        for _, row in bottom_pairs.iterrows():
            logger.info(f"    {row['pair']}: {row['contribution']:,.0f}元 "
                      f"(夏普{row['sharpe_ratio']:.2f}, 胜率{row['win_rate']:.1%})")
    
    # 风险统计
    risk_stats = results.get('risk_statistics', {})
    if risk_stats:
        logger.info("\n风险控制统计:")
        stop_loss_stats = risk_stats.get('stop_loss_stats', {})
        if stop_loss_stats:
            logger.info(f"  止损次数: {stop_loss_stats.get('count', 0)}")
            logger.info(f"  止损损失: {stop_loss_stats.get('total_loss', 0):,.0f}元")
        
        time_stop_stats = risk_stats.get('time_stop_stats', {})
        if time_stop_stats:
            logger.info(f"  时间止损次数: {time_stop_stats.get('count', 0)}")
    
    # ================== 7. 保存结果 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤7: 保存结果")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存交易记录
    if results['trades']:
        trades_df = pd.DataFrame([{
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
            'margin_released': trade.margin_released
        } for trade in results['trades']])
        
        trades_file = f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"交易记录已保存: {trades_file}")
    
    # 保存绩效报告
    if portfolio_metrics:
        metrics_file = f"backtest_metrics_{timestamp}.csv"
        pd.Series(portfolio_metrics).to_csv(metrics_file, header=['value'])
        logger.info(f"绩效指标已保存: {metrics_file}")
    
    # 保存配对绩效
    if not pair_metrics.empty:
        pair_file = f"backtest_pairs_{timestamp}.csv"
        pair_metrics.to_csv(pair_file, index=False)
        logger.info(f"配对绩效已保存: {pair_file}")
    
    # 保存权益曲线
    if 'equity_curve' in results:
        equity_df = pd.DataFrame({
            'date': pd.date_range(start=signal_start, periods=len(results['equity_curve']), freq='D'),
            'equity': results['equity_curve']
        })
        equity_file = f"backtest_equity_{timestamp}.csv"
        equity_df.to_csv(equity_file, index=False)
        logger.info(f"权益曲线已保存: {equity_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("回测完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()