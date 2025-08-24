#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端信号生成脚本
1. 使用协整模块筛选配对（5年和1年p值都<0.05）
2. 对筛选出的配对生成交易信号
3. 信号期：2024年7月到2025年8月（当前）
4. 90天预热期
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # ================== 1. 加载数据 ==================
    logger.info("=" * 60)
    logger.info("步骤1: 加载期货数据")
    logger.info("=" * 60)
    
    try:
        # 加载所有品种数据（已经是对数价格）
        price_data = load_all_symbols_data()
        logger.info(f"成功加载数据: {price_data.shape}")
        logger.info(f"数据范围: {price_data.index[0]} 至 {price_data.index[-1]}")
        logger.info(f"品种列表: {list(price_data.columns)}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # ================== 2. 协整筛选 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤2: 协整配对筛选")
    logger.info("=" * 60)
    
    try:
        # 创建协整分析器
        analyzer = CointegrationAnalyzer(price_data)
        
        # 运行协整筛选
        coint_results = analyzer.screen_all_pairs(
            screening_windows=['1y', '5y'],  # 筛选1年和5年窗口
            p_thresholds={'1y': 0.05, '5y': 0.05},  # p值阈值
            filter_logic='AND',  # 需要同时满足
            sort_by='pvalue_1y'  # 按1年p值排序
        )
        
        logger.info(f"协整筛选完成，共{len(coint_results)}个配对")
        
        # screen_all_pairs已经根据条件筛选，直接使用结果
        filtered_pairs = coint_results.copy()
        
        logger.info(f"通过筛选的配对数: {len(filtered_pairs)}")
        
        if len(filtered_pairs) > 0:
            logger.info("\n筛选出的配对:")
            for _, pair in filtered_pairs.iterrows():
                logger.info(f"  {pair['pair']}: "
                          f"β_1y={pair['beta_1y']:.4f}, "
                          f"p_1y={pair['pvalue_1y']:.4f}, "
                          f"p_5y={pair['pvalue_5y']:.4f}")
        else:
            logger.warning("没有配对通过筛选条件")
            return
            
    except Exception as e:
        logger.error(f"协整筛选失败: {e}")
        return
    
    # ================== 3. 设置时间参数 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤3: 设置时间参数")
    logger.info("=" * 60)
    
    # 时间设置
    signal_start_date = '2024-07-01'  # 信号期开始
    signal_end_date = '2025-08-24'    # 信号期结束（当前日期）
    
    # 预热期设置
    ols_window = 60        # OLS预热窗口
    warm_up_days = 30      # 额外Kalman预热（总计90天）
    total_warm_up = ols_window + warm_up_days
    
    # 计算实际需要的数据起始日期（往前推90个交易日）
    signal_start_pd = pd.to_datetime(signal_start_date)
    
    # 找到数据中信号开始日期的位置
    signal_start_idx = price_data.index.get_indexer([signal_start_pd], method='nearest')[0]
    
    # 往前推90个交易日
    if signal_start_idx >= total_warm_up:
        data_start_idx = signal_start_idx - total_warm_up
        data_start_date = price_data.index[data_start_idx]
    else:
        data_start_date = price_data.index[0]
        logger.warning(f"预热期数据不足，使用最早可用日期: {data_start_date}")
    
    logger.info(f"数据起始日期: {data_start_date} (包含90天预热)")
    logger.info(f"信号生成期: {signal_start_date} 至 {signal_end_date}")
    logger.info(f"预热设置: OLS {ols_window}天 + Kalman {warm_up_days}天 = 总计{total_warm_up}天")
    
    # 截取需要的数据
    analysis_data = price_data[data_start_date:signal_end_date].copy()
    logger.info(f"分析数据范围: {analysis_data.index[0]} 至 {analysis_data.index[-1]}")
    logger.info(f"数据点数: {len(analysis_data)}")
    
    # ================== 4. 生成交易信号 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤4: 生成交易信号")
    logger.info("=" * 60)
    
    # 创建信号生成器
    sg = AdaptiveSignalGenerator(
        z_open=2.0,              # 开仓阈值
        z_close=0.5,             # 平仓阈值
        max_holding_days=30,     # 最大持仓天数
        calibration_freq=5,      # 校准频率
        ols_window=ols_window,   # OLS预热窗口
        warm_up_days=warm_up_days # Kalman预热天数
    )
    
    # 使用批量处理
    logger.info(f"批量处理{len(filtered_pairs)}个配对...")
    
    try:
        # 批量生成信号
        all_signals_df = sg.process_all_pairs(
            pairs_df=filtered_pairs,
            price_data=analysis_data,
            beta_window='1y'
        )
        
        if not all_signals_df.empty:
            logger.info(f"成功生成信号: {len(all_signals_df)}个数据点")
            
            # 统计各配对的信号
            for pair in all_signals_df['pair'].unique():
                pair_signals = all_signals_df[all_signals_df['pair'] == pair]
                trade_signals = pair_signals[pair_signals['signal'].isin(['open_long', 'open_short', 'close'])]
                logger.info(f"  {pair}: 总计{len(pair_signals)}个点, 交易信号{len(trade_signals)}个")
        else:
            logger.warning("没有生成任何信号")
            
    except Exception as e:
        logger.error(f"批量信号生成失败: {e}")
        return
    
    # ================== 5. 汇总结果 ==================
    logger.info("\n" + "=" * 60)
    logger.info("步骤5: 汇总结果")
    logger.info("=" * 60)
    
    if not all_signals_df.empty:
        # 只保留信号期的数据（去除预热期）
        all_signals_df['date'] = pd.to_datetime(all_signals_df['date'])
        trading_signals = all_signals_df[all_signals_df['date'] >= signal_start_date].copy()
        
        logger.info(f"成功处理配对数: {len(all_signals_df['pair'].unique())}")
        logger.info(f"总信号数: {len(all_signals_df)}")
        logger.info(f"交易期信号数: {len(trading_signals)}")
        
        # 统计交易信号
        trade_signals = trading_signals[trading_signals['signal'].isin(['open_long', 'open_short', 'close'])]
        logger.info(f"实际交易信号数: {len(trade_signals)}")
        
        if len(trade_signals) > 0:
            logger.info("\n交易信号统计:")
            signal_summary = trade_signals.groupby(['pair', 'signal']).size().unstack(fill_value=0)
            logger.info(f"\n{signal_summary}")
        
        # 统计z-score分布
        signal_period_z = trading_signals[trading_signals['phase'] == 'signal_period']['z_score']
        if len(signal_period_z) > 0:
            logger.info(f"\nz-score统计 (信号期):")
            logger.info(f"  总数: {len(signal_period_z)}")
            logger.info(f"  均值: {signal_period_z.mean():.3f}")
            logger.info(f"  标准差: {signal_period_z.std():.3f}")
            logger.info(f"  95%分位数范围: [{signal_period_z.quantile(0.025):.2f}, {signal_period_z.quantile(0.975):.2f}]")
            
            # Fat tail分析
            extreme_count = len(signal_period_z[np.abs(signal_period_z) > 4])
            fat_tail_pct = extreme_count / len(signal_period_z) * 100
            logger.info(f"  |z|>4的比例: {fat_tail_pct:.2f}% ({extreme_count}个)")
        
        # 保存结果
        output_file = f"signals_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        all_signals_df.to_csv(output_file, index=False)
        logger.info(f"\n信号已保存到: {output_file}")
        
        # 获取质量报告
        quality_report = sg.get_quality_report()
        quality_file = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        quality_report.to_csv(quality_file, index=False)
        logger.info(f"质量报告已保存到: {quality_file}")
        
        # 打印质量摘要
        logger.info("\n质量摘要:")
        for _, row in quality_report.iterrows():
            logger.info(f"  {row['pair']}: z_var={row['z_var']:.3f}, "
                      f"质量={row['quality_status']}, delta={row['current_delta']:.3f}")
    else:
        logger.warning("没有生成任何信号")
    
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()