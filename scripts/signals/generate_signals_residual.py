#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用残差滚动方法生成端到端信号
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
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import AdaptiveSignalGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    logger.info("="*60)
    logger.info("生成端到端信号 - 残差滚动方法")
    logger.info("="*60)
    
    # 1. 加载数据
    logger.info("\n步骤1: 加载数据")
    log_prices = load_all_symbols_data()
    
    # 2. 协整分析
    logger.info("\n步骤2: 协整分析")
    analyzer = CointegrationAnalyzer(log_prices)
    pairs_df = analyzer.screen_all_pairs()
    
    # 筛选p值小于0.05的配对
    pairs_df = pairs_df[pairs_df['pvalue_1y'] < 0.05].copy()
    logger.info(f"找到 {len(pairs_df)} 个协整配对")
    
    # 3. 生成信号 - 使用残差滚动方法
    logger.info("\n步骤3: 生成信号 - 残差滚动方法")
    generator = AdaptiveSignalGenerator(
        z_open=2.0,
        z_close=0.5,
        max_holding_days=30,
        z_score_method='residual'  # 使用残差滚动方法
    )
    
    all_signals = generator.process_all_pairs(
        pairs_df=pairs_df,
        price_data=log_prices,
        beta_window='1y'
    )
    
    logger.info(f"生成信号总数: {len(all_signals)}")
    
    # 4. 统计信号
    trade_signals = all_signals[all_signals['signal'].isin(['open_long', 'open_short', 'close'])]
    logger.info(f"交易信号数: {len(trade_signals)}")
    
    signal_counts = trade_signals['signal'].value_counts()
    logger.info("信号分布:")
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
    # 分析Z-score分布
    signal_period = all_signals[all_signals['phase'] == 'signal_period']
    if len(signal_period) > 0:
        z_scores = signal_period['z_score'].values
        logger.info(f"\nZ-score统计:")
        logger.info(f"  均值: {np.mean(z_scores):.4f}")
        logger.info(f"  标准差: {np.std(z_scores):.4f}")
        logger.info(f"  |Z|>2比例: {np.mean(np.abs(z_scores) > 2)*100:.2f}%")
        logger.info(f"  |Z|>1比例: {np.mean(np.abs(z_scores) > 1)*100:.2f}%")
    
    # 5. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    signals_file = f"signals_residual_{timestamp}.csv"
    all_signals.to_csv(signals_file, index=False)
    logger.info(f"\n信号已保存: {signals_file}")
    
    # 6. 获取质量报告
    quality_report = generator.get_quality_report()
    if not quality_report.empty:
        quality_file = f"quality_residual_{timestamp}.csv"
        quality_report.to_csv(quality_file, index=False)
        logger.info(f"质量报告已保存: {quality_file}")
    
    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)
    
    return all_signals


if __name__ == "__main__":
    signals = main()