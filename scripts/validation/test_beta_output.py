#!/usr/bin/env python3
"""
测试Beta值输出功能

验证回测模块是否正确记录和输出每笔交易的beta值
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator, calculate_ols_beta
from lib.backtest import BacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试品种（选择少量品种快速测试）
TEST_SYMBOLS = ['CU0', 'ZN0', 'AL0', 'PB0']

# 合约规格
CONTRACT_SPECS = {
    'CU0': {'multiplier': 5, 'tick_size': 10},
    'ZN0': {'multiplier': 5, 'tick_size': 5},
    'AL0': {'multiplier': 5, 'tick_size': 5},
    'PB0': {'multiplier': 5, 'tick_size': 5}
}

def test_beta_output():
    """测试Beta值输出功能"""
    
    logger.info("=" * 80)
    logger.info("测试Beta值输出功能")
    logger.info("=" * 80)
    
    # 1. 加载数据
    logger.info("\n步骤1: 加载测试数据...")
    data = load_data(
        symbols=TEST_SYMBOLS,
        start_date='2023-01-01',
        end_date='2024-12-31',
        columns=['close'],
        log_price=True
    )
    
    if data.empty:
        logger.error("数据加载失败")
        return
    
    logger.info(f"数据加载成功: {len(data)}条记录, {len(TEST_SYMBOLS)}个品种")
    
    # 2. 协整分析
    logger.info("\n步骤2: 协整分析...")
    analyzer = CointegrationAnalyzer(data)
    pairs_df = analyzer.screen_all_pairs(p_threshold=0.1)  # 使用宽松阈值确保有配对
    
    if len(pairs_df) == 0:
        logger.warning("未找到协整配对，尝试更宽松的阈值")
        pairs_df = analyzer.screen_all_pairs(p_threshold=0.5)
    
    if len(pairs_df) == 0:
        logger.error("无法找到任何配对进行测试")
        return
        
    logger.info(f"找到 {len(pairs_df)} 个配对")
    
    # 3. 选择第一个配对进行测试
    test_pair = pairs_df.iloc[0]
    pair_name = test_pair['pair']
    symbol_x = test_pair['symbol_x']
    symbol_y = test_pair['symbol_y']
    
    logger.info(f"\n使用配对 {pair_name} 进行测试")
    logger.info(f"  5年p值: {test_pair.get('pvalue_5y', np.nan):.4f}")
    logger.info(f"  1年p值: {test_pair.get('pvalue_1y', np.nan):.4f}")
    
    # 4. 计算Beta值
    logger.info("\n步骤3: 计算Beta值...")
    
    # 获取2023年数据用于Beta估计
    training_data = data.loc['2023-01-01':'2023-12-31']
    x_train = training_data[symbol_x].values
    y_train = training_data[symbol_y].values
    
    # 计算OLS Beta
    beta_ols = calculate_ols_beta(x_train, y_train)
    logger.info(f"OLS Beta值: {beta_ols:.6f}")
    
    # 5. 生成测试信号
    logger.info("\n步骤4: 生成测试信号...")
    
    # 创建简单的测试信号
    test_signals = []
    
    # 信号1: 开多仓
    test_signals.append({
        'date': '2024-01-10',
        'pair': pair_name,
        'signal': 'open_long',
        'theoretical_ratio': beta_ols,  # 关键：包含beta值
        'beta': beta_ols,  # 也用beta字段
        'z_score': -2.5,
        'converged': True
    })
    
    # 信号2: 平仓
    test_signals.append({
        'date': '2024-01-20',
        'pair': pair_name,
        'signal': 'close',
        'theoretical_ratio': beta_ols,
        'beta': beta_ols,
        'z_score': 0.3,
        'converged': True
    })
    
    # 信号3: 开空仓
    test_signals.append({
        'date': '2024-02-01',
        'pair': pair_name,
        'signal': 'open_short',
        'theoretical_ratio': beta_ols * 1.1,  # 稍微不同的beta
        'beta': beta_ols * 1.1,
        'z_score': 2.3,
        'converged': True
    })
    
    # 信号4: 平仓
    test_signals.append({
        'date': '2024-02-15',
        'pair': pair_name,
        'signal': 'close',
        'theoretical_ratio': beta_ols * 1.1,
        'beta': beta_ols * 1.1,
        'z_score': -0.2,
        'converged': True
    })
    
    signals_df = pd.DataFrame(test_signals)
    logger.info(f"生成 {len(signals_df)} 个测试信号")
    
    # 6. 执行回测
    logger.info("\n步骤5: 执行回测...")
    
    # 获取回测期价格数据
    backtest_data = data.loc['2024-01-01':'2024-12-31']
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=5000000,
        margin_rate=0.12,
        commission_rate=0.0002,
        slippage_ticks=3,
        stop_loss_pct=0.15,
        max_holding_days=30,
        contract_specs=CONTRACT_SPECS
    )
    
    # 执行信号
    for _, signal in signals_df.iterrows():
        date = pd.to_datetime(signal['date'])
        if date in backtest_data.index:
            current_prices = backtest_data.loc[date].to_dict()
            success = engine.execute_signal(signal.to_dict(), current_prices, date)
            if success:
                logger.info(f"  {signal['date']}: 执行 {signal['signal']} 信号成功")
            else:
                logger.warning(f"  {signal['date']}: 执行 {signal['signal']} 信号失败")
    
    # 7. 验证Beta值输出
    logger.info("\n步骤6: 验证Beta值输出...")
    
    # 获取交易记录
    trades_df = pd.DataFrame(engine.trade_records)
    
    if len(trades_df) == 0:
        logger.error("没有生成交易记录")
        return
    
    logger.info(f"\n生成了 {len(trades_df)} 笔交易记录")
    
    # 显示交易记录，重点关注Beta值
    logger.info("\n交易记录详情:")
    for i, trade in trades_df.iterrows():
        logger.info(f"\n交易 {i+1}:")
        logger.info(f"  配对: {trade['pair']}")
        logger.info(f"  方向: {trade['direction']}")
        logger.info(f"  Beta值: {trade.get('beta', 'N/A')}")  # 关键：检查beta字段
        logger.info(f"  开仓日期: {trade['open_date']}")
        logger.info(f"  平仓日期: {trade['close_date']}")
        logger.info(f"  Y手数: {trade['contracts_y']}")
        logger.info(f"  X手数: {trade['contracts_x']}")
        logger.info(f"  实际比率: {trade['contracts_y']/trade['contracts_x']:.4f}")
        logger.info(f"  净盈亏: {trade['net_pnl']:,.2f}")
    
    # 8. 保存输出文件
    output_dir = Path("output/beta_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存交易记录
    trades_file = output_dir / f"trades_with_beta_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"\n交易记录已保存至: {trades_file}")
    
    # 保存汇总报告
    summary = {
        'test_date': timestamp,
        'pair': pair_name,
        'ols_beta': beta_ols,
        'total_trades': len(trades_df),
        'trades': []
    }
    
    for _, trade in trades_df.iterrows():
        summary['trades'].append({
            'trade_id': trade['trade_id'],
            'beta': trade.get('beta', None),
            'contracts_y': trade['contracts_y'],
            'contracts_x': trade['contracts_x'],
            'actual_ratio': trade['contracts_y'] / trade['contracts_x'] if trade['contracts_x'] > 0 else 0,
            'net_pnl': trade['net_pnl']
        })
    
    summary_file = output_dir / f"beta_test_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"测试汇总已保存至: {summary_file}")
    
    # 9. 验证结论
    logger.info("\n" + "=" * 60)
    logger.info("测试结论:")
    logger.info("=" * 60)
    
    has_beta = all('beta' in trade for _, trade in trades_df.iterrows())
    
    if has_beta:
        logger.info("✅ 所有交易记录都包含Beta值")
        
        # 检查Beta值的合理性
        beta_values = trades_df['beta'].values
        logger.info(f"✅ Beta值范围: [{beta_values.min():.4f}, {beta_values.max():.4f}]")
        
        # 检查实际手数比率与Beta的匹配度
        for _, trade in trades_df.iterrows():
            if trade['contracts_x'] > 0:
                actual_ratio = trade['contracts_y'] / trade['contracts_x']
                beta = trade['beta']
                error = abs(actual_ratio - beta) / beta * 100
                logger.info(f"  交易{trade['trade_id']}: Beta={beta:.4f}, "
                          f"实际比率={actual_ratio:.4f}, 误差={error:.2f}%")
    else:
        logger.error("❌ 部分交易记录缺少Beta值")
        missing = trades_df[~trades_df.index.isin(trades_df[trades_df['beta'].notna()].index)]
        logger.error(f"   缺少Beta的交易: {missing['trade_id'].tolist()}")
    
    return trades_df

if __name__ == "__main__":
    try:
        trades_df = test_beta_output()
        logger.info("\n测试完成!")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        sys.exit(1)