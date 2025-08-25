#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版信号生成：使用60天滚动OLS
- 2024年1-3月: OLS估计初始β
- 2024年3-6月: 预热期
- 2024年7月至今: 使用60天滚动OLS生成信号
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys
import os
from sklearn.linear_model import LinearRegression

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer, estimate_parameters

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rolling_ols_beta(x_data, y_data, window=60):
    """计算滚动OLS的beta"""
    if len(x_data) < window:
        return None
    
    x_window = x_data[-window:].reshape(-1, 1)
    y_window = y_data[-window:]
    
    model = LinearRegression()
    model.fit(x_window, y_window)
    return model.coef_[0]


def generate_signals_rolling_ols(pair_name, x_data, y_data, 
                                z_open=2.2, z_close=0.3):
    """
    使用滚动OLS生成信号
    """
    # 准备数据 (从2024年1月开始)
    start_date = pd.Timestamp('2024-01-01')
    signal_start = pd.Timestamp('2024-07-01')
    
    x_full = x_data[start_date:].dropna()
    y_full = y_data[start_date:].dropna()
    
    # 对齐数据
    common_index = x_full.index.intersection(y_full.index)
    x_aligned = x_full[common_index]
    y_aligned = y_full[common_index]
    
    if len(x_aligned) < 120:
        logger.warning(f"{pair_name}: 数据不足")
        return pd.DataFrame()
    
    signals = []
    position = None
    days_held = 0
    
    # 存储历史残差用于计算Z-score
    residual_window = []
    
    for i in range(60, len(x_aligned)):
        date = x_aligned.index[i]
        
        # 计算60天滚动OLS的beta
        x_window = x_aligned.values[i-60:i]
        y_window = y_aligned.values[i-60:i]
        
        beta = rolling_ols_beta(x_window, y_window, 60)
        if beta is None:
            continue
        
        # 计算残差
        residual = y_aligned.iloc[i] - beta * x_aligned.iloc[i]
        residual_window.append(residual)
        
        # 保持窗口长度为60
        if len(residual_window) > 60:
            residual_window.pop(0)
        
        # 信号生成期（7月开始）
        if date >= signal_start and len(residual_window) >= 20:
            # 计算Z-score
            residual_mean = np.mean(residual_window)
            residual_std = np.std(residual_window)
            
            if residual_std > 1e-8:
                z = (residual - residual_mean) / residual_std
            else:
                z = 0.0
            
            # 生成交易信号
            signal = 'empty'
            
            # 强制平仓
            if position and days_held >= 30:
                signal = 'close'
                position = None
                days_held = 0
            # 平仓条件
            elif position and abs(z) <= z_close:
                signal = 'close'
                position = None
                days_held = 0
            # 开仓条件
            elif not position:
                if z <= -z_open:
                    signal = 'open_long'
                    position = 'long'
                    days_held = 1
                elif z >= z_open:
                    signal = 'open_short'
                    position = 'short'
                    days_held = 1
            # 持仓
            elif position:
                days_held += 1
                signal = f'holding_{position}'
            
            phase = 'signal'
        else:
            # 预热期
            signal = 'warm_up'
            phase = 'warmup'
            z = 0.0
        
        signals.append({
            'date': date,
            'pair': pair_name,
            'signal': signal,
            'z_score': z,
            'residual': residual,
            'beta': beta,
            'phase': phase,
            'position': position,
            'days_held': days_held
        })
    
    return pd.DataFrame(signals)


def main():
    """主函数"""
    
    logger.info("="*60)
    logger.info("滚动OLS信号生成")
    logger.info("60天滚动窗口计算beta")
    logger.info("7月至今: 信号生成")
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
    
    # 3. 逐个处理配对
    logger.info("\n步骤3: 生成信号")
    
    all_results = []
    
    for idx, pair_info in pairs_df.iterrows():
        pair_name = pair_info['pair']
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        
        if symbol_x not in log_prices.columns or symbol_y not in log_prices.columns:
            continue
        
        logger.info(f"处理配对: {pair_name}")
        
        signals = generate_signals_rolling_ols(
            pair_name,
            log_prices[symbol_x],
            log_prices[symbol_y],
            z_open=2.2,
            z_close=0.3
        )
        
        if not signals.empty:
            signals['symbol_x'] = symbol_x
            signals['symbol_y'] = symbol_y
            all_results.append(signals)
    
    # 4. 合并结果
    if all_results:
        all_signals = pd.concat(all_results, ignore_index=True)
        
        # 统计
        trade_signals = all_signals[all_signals['signal'].isin(['open_long', 'open_short', 'close'])]
        signal_period = all_signals[all_signals['phase'] == 'signal']
        
        logger.info(f"\n统计:")
        logger.info(f"  总信号数: {len(trade_signals)}")
        
        if len(signal_period) > 0:
            z_scores = signal_period['z_score'].values
            z_scores_nonzero = z_scores[z_scores != 0]
            if len(z_scores_nonzero) > 0:
                logger.info(f"  |Z|>2比例: {np.mean(np.abs(z_scores_nonzero) > 2)*100:.2f}%")
                logger.info(f"  Z均值: {np.mean(z_scores_nonzero):.4f}")
                logger.info(f"  Z标准差: {np.std(z_scores_nonzero):.4f}")
        
        # 检查beta符号
        open_signals = all_signals[all_signals['signal'].str.contains('open')]
        if len(open_signals) > 0:
            logger.info(f"\nBeta分析:")
            logger.info(f"  正Beta: {(open_signals['beta'] > 0).sum()}个")
            logger.info(f"  负Beta: {(open_signals['beta'] < 0).sum()}个")
            
            # 显示几个负beta的例子
            neg_beta = open_signals[open_signals['beta'] < 0].head(5)
            if len(neg_beta) > 0:
                logger.info(f"\n负Beta示例:")
                for _, row in neg_beta.iterrows():
                    logger.info(f"    {row['pair']}: beta={row['beta']:.4f}")
        
        # 5. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals_rolling_ols_{timestamp}.csv"
        all_signals.to_csv(filename, index=False)
        logger.info(f"\n信号已保存: {filename}")
    
    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)
    
    return all_signals if all_results else pd.DataFrame()


if __name__ == "__main__":
    signals = main()