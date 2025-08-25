#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照正确的时间线生成信号：
- 2024年1-3月: OLS估计初始β
- 2024年3-6月: Kalman滤波预热
- 2024年7月至今: 信号生成期
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
from lib.coint import CointegrationAnalyzer, estimate_parameters

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_initial_beta(x_data, y_data, start_date='2024-01-01', end_date='2024-03-31'):
    """使用2024年1-3月数据估计初始beta"""
    x_period = x_data[start_date:end_date].dropna()
    y_period = y_data[start_date:end_date].dropna()
    
    # 对齐数据
    common_index = x_period.index.intersection(y_period.index)
    x_aligned = x_period[common_index].values
    y_aligned = y_period[common_index].values
    
    if len(x_aligned) < 20:
        logger.warning(f"数据不足: 只有{len(x_aligned)}个点")
        return None
    
    # OLS估计
    params = estimate_parameters(x_aligned, y_aligned)
    return params['beta']


def generate_signals_with_timeline(pair_name, x_data, y_data, method='innovation'):
    """
    按照正确时间线生成信号
    """
    from lib.signal_generation import AdaptiveKalmanFilter
    
    # 1. 估计初始beta (2024年1-3月)
    initial_beta = estimate_initial_beta(x_data, y_data)
    if initial_beta is None:
        logger.warning(f"{pair_name}: 无法估计初始beta")
        return pd.DataFrame()
    
    logger.info(f"{pair_name}: 初始beta = {initial_beta:.6f}")
    
    # 2. 准备数据 (从2024年3月开始，用于Kalman预热)
    start_date = pd.Timestamp('2024-03-01')
    x_full = x_data[start_date:].dropna()
    y_full = y_data[start_date:].dropna()
    
    # 对齐数据
    common_index = x_full.index.intersection(y_full.index)
    x_aligned = x_full[common_index]
    y_aligned = y_full[common_index]
    
    if len(x_aligned) < 100:
        logger.warning(f"{pair_name}: 数据不足")
        return pd.DataFrame()
    
    # 3. 创建Kalman滤波器
    kf = AdaptiveKalmanFilter(
        pair_name=pair_name,
        delta=0.93,
        lambda_r=0.89,
        beta_bounds=None  # 不限制beta边界
    )
    
    # 4. 初始化（使用前60个点，约3个月）
    kf.warm_up_ols(x_aligned.values[:60], y_aligned.values[:60], window=60)
    # 覆盖为我们计算的初始beta
    kf.beta = initial_beta
    
    # 5. Kalman预热期 (3-6月，约60个交易日)
    signals = []
    position = None
    days_held = 0
    
    # 预热期更新
    for i in range(60, min(120, len(x_aligned))):
        result = kf.update(y_aligned.iloc[i], x_aligned.iloc[i])
        signals.append({
            'date': x_aligned.index[i],
            'pair': pair_name,
            'signal': 'warm_up',
            'z_score': result['z'],
            'residual': result['v'],
            'beta': result['beta'],
            'phase': 'warmup'
        })
    
    # 6. 信号生成期 (7月开始)
    signal_start = pd.Timestamp('2024-07-01')
    signal_start_idx = None
    
    for i, date in enumerate(x_aligned.index):
        if date >= signal_start:
            signal_start_idx = i
            break
    
    if signal_start_idx is None:
        logger.warning(f"{pair_name}: 没有7月之后的数据")
        return pd.DataFrame(signals)
    
    # 根据方法选择计算Z-score
    if method == 'residual':
        residual_window = []
    
    for i in range(signal_start_idx, len(x_aligned)):
        # Kalman更新
        result = kf.update(y_aligned.iloc[i], x_aligned.iloc[i])
        
        # 计算Z-score
        if method == 'residual':
            # 残差滚动方法
            residual = result['v']
            residual_window.append(residual)
            if len(residual_window) > 60:
                residual_window.pop(0)
            
            if len(residual_window) >= 20:
                residual_mean = np.mean(residual_window)
                residual_std = np.std(residual_window)
                if residual_std > 1e-8:
                    z = (residual - residual_mean) / residual_std
                else:
                    z = 0.0
            else:
                z = 0.0
        else:
            # 创新标准化方法
            z = result['z']
        
        # 生成交易信号
        signal = 'empty'
        
        # 强制平仓
        if position and days_held >= 30:
            signal = 'close'
            position = None
            days_held = 0
        # 平仓条件（改为0.3）
        elif position and abs(z) <= 0.3:
            signal = 'close'
            position = None
            days_held = 0
        # 开仓条件（改为2.2）
        elif not position:
            if z <= -2.2:
                signal = 'open_long'
                position = 'long'
                days_held = 1
            elif z >= 2.2:
                signal = 'open_short'
                position = 'short'
                days_held = 1
        # 持仓
        elif position:
            days_held += 1
            signal = f'holding_{position}'
        
        signals.append({
            'date': x_aligned.index[i],
            'pair': pair_name,
            'signal': signal,
            'z_score': z,
            'residual': result['v'],
            'beta': result['beta'],
            'phase': 'signal',
            'position': position,
            'days_held': days_held
        })
    
    return pd.DataFrame(signals)


def main():
    """主函数"""
    
    logger.info("="*60)
    logger.info("按正确时间线生成信号")
    logger.info("1-3月: OLS估计β")
    logger.info("3-6月: Kalman预热")
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
    
    # 测试两种方法
    for method in ['innovation', 'residual']:
        logger.info(f"\n使用 {method} 方法:")
        method_results = []
        
        for idx, pair_info in pairs_df.iterrows():
            pair_name = pair_info['pair']
            symbol_x = pair_info['symbol_x']
            symbol_y = pair_info['symbol_y']
            
            if symbol_x not in log_prices.columns or symbol_y not in log_prices.columns:
                continue
            
            logger.info(f"处理配对: {pair_name}")
            
            signals = generate_signals_with_timeline(
                pair_name,
                log_prices[symbol_x],
                log_prices[symbol_y],
                method=method
            )
            
            if not signals.empty:
                signals['symbol_x'] = symbol_x
                signals['symbol_y'] = symbol_y
                signals['method'] = method
                method_results.append(signals)
        
        if method_results:
            method_df = pd.concat(method_results, ignore_index=True)
            all_results.append(method_df)
            
            # 统计
            trade_signals = method_df[method_df['signal'].isin(['open_long', 'open_short', 'close'])]
            signal_period = method_df[method_df['phase'] == 'signal']
            
            logger.info(f"\n{method}方法统计:")
            logger.info(f"  总信号数: {len(trade_signals)}")
            
            if len(signal_period) > 0:
                z_scores = signal_period['z_score'].values
                logger.info(f"  |Z|>2比例: {np.mean(np.abs(z_scores) > 2)*100:.2f}%")
                logger.info(f"  Z均值: {np.mean(z_scores):.4f}")
                logger.info(f"  Z标准差: {np.std(z_scores):.4f}")
    
    # 4. 保存结果
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for df in all_results:
            method = df['method'].iloc[0]
            filename = f"signals_{method}_timeline_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"\n{method}信号已保存: {filename}")
    
    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()