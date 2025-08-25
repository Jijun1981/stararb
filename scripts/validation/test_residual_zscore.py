#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试残差滚动Z-score方法与创新标准化方法的对比
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveSignalGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_zscore_methods():
    """测试两种Z-score方法"""
    
    logger.info("="*60)
    logger.info("测试两种Z-score计算方法")
    logger.info("="*60)
    
    # 1. 加载数据
    logger.info("\n1. 加载价格数据...")
    log_prices = load_all_symbols_data()
    
    # 选择一个测试配对
    test_pair = "NI-AG"
    symbol_x = "NI"
    symbol_y = "AG"
    
    x_data = log_prices[symbol_x]
    y_data = log_prices[symbol_y]
    
    # 截取2024年数据
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-12-31')
    x_test = x_data[start_date:end_date]
    y_test = y_data[start_date:end_date]
    
    logger.info(f"测试配对: {test_pair}")
    logger.info(f"数据范围: {x_test.index[0]} 至 {x_test.index[-1]}")
    
    # 2. 测试创新标准化方法
    logger.info("\n2. 测试创新标准化方法 (innovation)...")
    generator_innovation = AdaptiveSignalGenerator(
        z_open=2.0,
        z_close=0.5,
        max_holding_days=30,
        z_score_method='innovation'  # 使用创新标准化
    )
    
    signals_innovation = generator_innovation.process_pair(
        pair_name=test_pair,
        x_data=x_test,
        y_data=y_test,
        initial_beta=1.483  # 使用之前分析的beta值
    )
    
    # 3. 测试残差滚动方法
    logger.info("\n3. 测试残差滚动方法 (residual)...")
    generator_residual = AdaptiveSignalGenerator(
        z_open=2.0,
        z_close=0.5,
        max_holding_days=30,
        z_score_method='residual'  # 使用残差滚动
    )
    
    signals_residual = generator_residual.process_pair(
        pair_name=test_pair,
        x_data=x_test,
        y_data=y_test,
        initial_beta=1.483
    )
    
    # 4. 比较结果
    logger.info("\n4. 比较两种方法的结果:")
    logger.info("="*60)
    
    # 过滤出交易信号
    trade_signals_innovation = signals_innovation[
        signals_innovation['signal'].isin(['open_long', 'open_short', 'close'])
    ]
    trade_signals_residual = signals_residual[
        signals_residual['signal'].isin(['open_long', 'open_short', 'close'])
    ]
    
    logger.info(f"创新标准化方法:")
    logger.info(f"  总信号数: {len(trade_signals_innovation)}")
    if len(trade_signals_innovation) > 0:
        signal_counts_i = trade_signals_innovation['signal'].value_counts()
        for signal, count in signal_counts_i.items():
            logger.info(f"    {signal}: {count}")
    
    logger.info(f"\n残差滚动方法:")
    logger.info(f"  总信号数: {len(trade_signals_residual)}")
    if len(trade_signals_residual) > 0:
        signal_counts_r = trade_signals_residual['signal'].value_counts()
        for signal, count in signal_counts_r.items():
            logger.info(f"    {signal}: {count}")
    
    # 5. 分析Z-score分布
    logger.info("\n5. Z-score分布分析:")
    logger.info("="*60)
    
    # 获取信号期的Z-score
    signal_period_innovation = signals_innovation[signals_innovation['phase'] == 'signal_period']
    signal_period_residual = signals_residual[signals_residual['phase'] == 'signal_period']
    
    if len(signal_period_innovation) > 0:
        z_scores_i = signal_period_innovation['z_score'].values
        logger.info(f"创新标准化方法:")
        logger.info(f"  Z-score均值: {np.mean(z_scores_i):.4f}")
        logger.info(f"  Z-score标准差: {np.std(z_scores_i):.4f}")
        logger.info(f"  |Z|>2的比例: {np.mean(np.abs(z_scores_i) > 2)*100:.2f}%")
        logger.info(f"  |Z|>1的比例: {np.mean(np.abs(z_scores_i) > 1)*100:.2f}%")
    
    if len(signal_period_residual) > 0:
        z_scores_r = signal_period_residual['z_score'].values
        logger.info(f"\n残差滚动方法:")
        logger.info(f"  Z-score均值: {np.mean(z_scores_r):.4f}")
        logger.info(f"  Z-score标准差: {np.std(z_scores_r):.4f}")
        logger.info(f"  |Z|>2的比例: {np.mean(np.abs(z_scores_r) > 2)*100:.2f}%")
        logger.info(f"  |Z|>1的比例: {np.mean(np.abs(z_scores_r) > 1)*100:.2f}%")
    
    # 6. 保存结果对比
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存创新标准化结果
    innovation_file = f"signals_innovation_{timestamp}.csv"
    signals_innovation.to_csv(innovation_file, index=False)
    logger.info(f"\n创新标准化信号已保存: {innovation_file}")
    
    # 保存残差滚动结果
    residual_file = f"signals_residual_{timestamp}.csv"
    signals_residual.to_csv(residual_file, index=False)
    logger.info(f"残差滚动信号已保存: {residual_file}")
    
    return signals_innovation, signals_residual


def run_full_test():
    """运行完整测试（所有配对）"""
    
    logger.info("\n" + "="*60)
    logger.info("运行完整测试 - 所有配对")
    logger.info("="*60)
    
    # 加载配对数据
    pairs_file = 'cointegrated_pairs_20250823_210522.csv'
    if not os.path.exists(pairs_file):
        logger.error(f"配对文件不存在: {pairs_file}")
        return
    
    pairs_df = pd.read_csv(pairs_file)
    logger.info(f"加载配对数据: {len(pairs_df)}个配对")
    
    # 加载价格数据
    log_prices = load_all_symbols_data()
    
    # 对前5个配对进行测试
    test_pairs = pairs_df.head(5)
    
    results = []
    for idx, pair_info in test_pairs.iterrows():
        pair_name = pair_info['pair']
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        beta = pair_info.get('beta_1y', 1.0)
        
        logger.info(f"\n处理配对: {pair_name}")
        
        # 测试两种方法
        for method in ['innovation', 'residual']:
            generator = AdaptiveSignalGenerator(
                z_open=2.0,
                z_close=0.5,
                max_holding_days=30,
                z_score_method=method
            )
            
            signals = generator.process_pair(
                pair_name=pair_name,
                x_data=log_prices[symbol_x],
                y_data=log_prices[symbol_y],
                initial_beta=beta
            )
            
            # 统计信号
            trade_signals = signals[signals['signal'].isin(['open_long', 'open_short', 'close'])]
            signal_period = signals[signals['phase'] == 'signal_period']
            
            if len(signal_period) > 0:
                z_scores = signal_period['z_score'].values
                result = {
                    'pair': pair_name,
                    'method': method,
                    'total_signals': len(trade_signals),
                    'z_mean': np.mean(z_scores),
                    'z_std': np.std(z_scores),
                    'z_gt_2_pct': np.mean(np.abs(z_scores) > 2) * 100,
                    'z_gt_1_pct': np.mean(np.abs(z_scores) > 1) * 100
                }
                results.append(result)
                
                logger.info(f"  {method}: 信号数={result['total_signals']}, "
                          f"|Z|>2比例={result['z_gt_2_pct']:.2f}%")
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"zscore_method_comparison_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"\n对比结果已保存: {results_file}")
        
        # 显示汇总
        logger.info("\n" + "="*60)
        logger.info("方法对比汇总:")
        logger.info("="*60)
        
        for method in ['innovation', 'residual']:
            method_data = results_df[results_df['method'] == method]
            logger.info(f"\n{method}方法:")
            logger.info(f"  平均信号数: {method_data['total_signals'].mean():.1f}")
            logger.info(f"  平均|Z|>2比例: {method_data['z_gt_2_pct'].mean():.2f}%")
            logger.info(f"  平均Z标准差: {method_data['z_std'].mean():.3f}")
    
    return results_df


if __name__ == "__main__":
    # 运行单配对测试
    signals_i, signals_r = test_zscore_methods()
    
    # 运行多配对测试
    results = run_full_test()
    
    logger.info("\n测试完成！")