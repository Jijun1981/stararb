#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出在5年、4年、3年、2年、1年都协整的稳定配对
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

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_stable_cointegrated_pairs():
    """找出在所有时间窗口都协整的配对"""
    
    logger.info("="*60)
    logger.info("寻找稳定协整配对")
    logger.info("要求：5年、4年、3年、2年、1年p值都<0.05")
    logger.info("="*60)
    
    # 1. 加载数据
    logger.info("\n步骤1: 加载数据")
    log_prices = load_all_symbols_data()
    logger.info(f"数据范围: {log_prices.index[0]} 至 {log_prices.index[-1]}")
    
    # 2. 创建协整分析器
    logger.info("\n步骤2: 协整分析")
    analyzer = CointegrationAnalyzer(log_prices)
    
    # 获取所有配对的协整结果
    all_pairs = analyzer.screen_all_pairs()
    logger.info(f"总配对数: {len(all_pairs)}")
    
    # 3. 筛选在所有窗口都协整的配对
    logger.info("\n步骤3: 筛选稳定配对")
    
    # 需要检查的p值列
    pvalue_columns = ['pvalue_5y', 'pvalue_4y', 'pvalue_3y', 'pvalue_2y', 'pvalue_1y']
    
    # 筛选条件：所有p值都小于0.05
    stable_pairs = all_pairs.copy()
    for col in pvalue_columns:
        if col in stable_pairs.columns:
            stable_pairs = stable_pairs[stable_pairs[col] < 0.05]
            logger.info(f"  {col} < 0.05: 剩余{len(stable_pairs)}个配对")
    
    logger.info(f"\n找到{len(stable_pairs)}个稳定配对")
    
    # 4. 显示稳定配对的详细信息
    if len(stable_pairs) > 0:
        logger.info("\n稳定配对列表:")
        logger.info("-"*60)
        
        # 按5年p值排序
        stable_pairs = stable_pairs.sort_values('pvalue_5y')
        
        for idx, row in stable_pairs.iterrows():
            logger.info(f"\n配对: {row['pair']} ({row['symbol_x']}-{row['symbol_y']})")
            logger.info(f"  5年p值: {row['pvalue_5y']:.6f}")
            logger.info(f"  4年p值: {row.get('pvalue_4y', np.nan):.6f}")
            logger.info(f"  3年p值: {row.get('pvalue_3y', np.nan):.6f}")
            logger.info(f"  2年p值: {row.get('pvalue_2y', np.nan):.6f}")
            logger.info(f"  1年p值: {row['pvalue_1y']:.6f}")
            logger.info(f"  Beta(1y): {row.get('beta_1y', np.nan):.6f}")
            logger.info(f"  半衰期: {row.get('halflife_1y', np.nan):.2f}天")
        
        # 5. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stable_pairs_{timestamp}.csv"
        stable_pairs.to_csv(filename, index=False)
        logger.info(f"\n稳定配对已保存: {filename}")
        
        # 6. 统计分析
        logger.info("\n统计分析:")
        logger.info("-"*40)
        
        # Beta分布
        beta_1y = stable_pairs['beta_1y']
        logger.info(f"Beta分布:")
        logger.info(f"  正Beta: {(beta_1y > 0).sum()}个")
        logger.info(f"  负Beta: {(beta_1y < 0).sum()}个")
        logger.info(f"  Beta均值: {beta_1y.mean():.4f}")
        logger.info(f"  Beta标准差: {beta_1y.std():.4f}")
        
        # 半衰期分布
        if 'halflife_1y' in stable_pairs.columns:
            halflife = stable_pairs['halflife_1y'].dropna()
            if len(halflife) > 0:
                logger.info(f"\n半衰期分布:")
                logger.info(f"  最短: {halflife.min():.1f}天")
                logger.info(f"  最长: {halflife.max():.1f}天")
                logger.info(f"  平均: {halflife.mean():.1f}天")
                logger.info(f"  中位数: {halflife.median():.1f}天")
        
        # 按品种统计
        logger.info(f"\n品种出现频率:")
        all_symbols = list(stable_pairs['symbol_x']) + list(stable_pairs['symbol_y'])
        symbol_counts = pd.Series(all_symbols).value_counts()
        for symbol, count in symbol_counts.head(10).items():
            logger.info(f"  {symbol}: {count}次")
    
    else:
        logger.warning("没有找到在所有时间窗口都协整的配对")
    
    return stable_pairs


def analyze_pair_stability(stable_pairs):
    """分析配对的稳定性"""
    
    if len(stable_pairs) == 0:
        return
    
    logger.info("\n" + "="*60)
    logger.info("配对稳定性分析")
    logger.info("="*60)
    
    # 计算p值的变异系数
    pvalue_columns = ['pvalue_5y', 'pvalue_4y', 'pvalue_3y', 'pvalue_2y', 'pvalue_1y']
    
    stability_scores = []
    for idx, row in stable_pairs.iterrows():
        pvalues = []
        for col in pvalue_columns:
            if col in row and not pd.isna(row[col]):
                pvalues.append(row[col])
        
        if len(pvalues) > 1:
            # 变异系数 = 标准差 / 均值
            cv = np.std(pvalues) / np.mean(pvalues)
            stability_scores.append({
                'pair': row['pair'],
                'mean_pvalue': np.mean(pvalues),
                'std_pvalue': np.std(pvalues),
                'cv': cv,
                'min_pvalue': np.min(pvalues),
                'max_pvalue': np.max(pvalues)
            })
    
    if stability_scores:
        stability_df = pd.DataFrame(stability_scores)
        stability_df = stability_df.sort_values('cv')  # 按变异系数排序
        
        logger.info("\n最稳定的配对（p值变异系数最小）:")
        for idx, row in stability_df.head(10).iterrows():
            logger.info(f"  {row['pair']}: CV={row['cv']:.3f}, 平均p值={row['mean_pvalue']:.4f}")
        
        # 保存稳定性分析
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stability_file = f"pair_stability_{timestamp}.csv"
        stability_df.to_csv(stability_file, index=False)
        logger.info(f"\n稳定性分析已保存: {stability_file}")
        
        return stability_df
    
    return None


if __name__ == "__main__":
    # 找出稳定配对
    stable_pairs = find_stable_cointegrated_pairs()
    
    # 分析稳定性
    if len(stable_pairs) > 0:
        stability_analysis = analyze_pair_stability(stable_pairs)
    
    logger.info("\n" + "="*60)
    logger.info("分析完成！")
    logger.info("="*60)