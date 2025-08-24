"""
批量协整分析模块
处理所有配对的1年和5年协整分析
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
import logging

from .coint import eg_test, calculate_halflife
from .direction import determine_direction_by_volatility
from .halflife_utils import calculate_halflife_relaxed

logger = logging.getLogger(__name__)


def analyze_all_pairs(data: pd.DataFrame, 
                      data_1y: pd.DataFrame = None) -> pd.DataFrame:
    """
    分析所有配对的1年和5年协整关系
    
    Args:
        data: 5年完整数据（对数价格）
        data_1y: 最近1年数据（如果None，从data提取）
        
    Returns:
        DataFrame with columns:
        - pair: 配对名称
        - symbol1, symbol2: 品种代码
        - direction: 回归方向
        - pvalue_1y, pvalue_5y: p值
        - beta_1y, beta_5y: beta系数
        - halflife_1y, halflife_5y: 半衰期
    """
    if data_1y is None:
        data_1y = data.iloc[-252:]  # 最近1年
    
    symbols = data.columns.tolist()
    all_pairs = list(combinations(symbols, 2))
    
    results = []
    
    for idx, (sym1, sym2) in enumerate(all_pairs):
        if idx % 10 == 0:
            logger.info(f"Processing pair {idx+1}/{len(all_pairs)}")
        
        try:
            # 5年数据
            x_5y = data[sym1].values
            y_5y = data[sym2].values
            
            # 1年数据
            x_1y = data_1y[sym1].values
            y_1y = data_1y[sym2].values
            
            # 基于5年数据判定方向
            dir_result = determine_direction_by_volatility(x_5y, y_5y)
            direction = dir_result['direction']
            
            # 5年协整
            result_5y = eg_test(x_5y, y_5y, direction=direction)
            halflife_5y = calculate_halflife(result_5y['residuals'])
            
            # 1年协整
            result_1y = eg_test(x_1y, y_1y, direction=direction)
            halflife_1y = calculate_halflife(result_1y['residuals'])
            
            results.append({
                'pair': f"{sym1}-{sym2}",
                'symbol1': sym1,
                'symbol2': sym2,
                'direction': direction,
                'volatility_1': dir_result['volatility_x'],
                'volatility_2': dir_result['volatility_y'],
                'pvalue_1y': result_1y['pvalue'],
                'pvalue_5y': result_5y['pvalue'],
                'beta_1y': result_1y['beta'],
                'beta_5y': result_5y['beta'],
                'halflife_1y': halflife_1y if halflife_1y else np.nan,
                'halflife_5y': halflife_5y if halflife_5y else np.nan
            })
            
        except Exception as e:
            logger.warning(f"Error processing {sym1}-{sym2}: {e}")
            results.append({
                'pair': f"{sym1}-{sym2}",
                'symbol1': sym1,
                'symbol2': sym2,
                'direction': None,
                'volatility_1': np.nan,
                'volatility_2': np.nan,
                'pvalue_1y': np.nan,
                'pvalue_5y': np.nan,
                'beta_1y': np.nan,
                'beta_5y': np.nan,
                'halflife_1y': np.nan,
                'halflife_5y': np.nan
            })
    
    return pd.DataFrame(results)


def rolling_window_analysis(x: np.ndarray, 
                           y: np.ndarray,
                           windows: List[int] = [30, 45, 60, 90],
                           step: int = 5) -> Dict:
    """
    对一个配对进行滚动窗口分析
    
    Args:
        x: 自变量价格序列（1年数据）
        y: 因变量价格序列（1年数据）
        windows: 窗口大小列表
        step: 滚动步长
        
    Returns:
        Dict with keys for each window size
    """
    results = {}
    
    for window in windows:
        pvalues = []
        betas = []
        halflifes = []
        residual_means = []
        residual_stds = []
        
        # 滚动
        for i in range(0, len(x) - window + 1, step):
            x_win = x[i:i+window]
            y_win = y[i:i+window]
            
            try:
                # 每个窗口都判定方向
                dir_result = determine_direction_by_volatility(x_win, y_win)
                
                # 协整检验
                result = eg_test(x_win, y_win, direction=dir_result['direction'])
                halflife = calculate_halflife(result['residuals'])
                
                pvalues.append(result['pvalue'])
                betas.append(result['beta'])
                halflifes.append(halflife if halflife else np.nan)
                residual_means.append(np.mean(result['residuals']))
                residual_stds.append(np.std(result['residuals']))
                
            except:
                pvalues.append(np.nan)
                betas.append(np.nan)
                halflifes.append(np.nan)
                residual_means.append(np.nan)
                residual_stds.append(np.nan)
        
        results[f'window_{window}'] = {
            'pvalues': np.array(pvalues),
            'betas': np.array(betas),
            'halflifes': np.array(halflifes),
            'means': np.array(residual_means),
            'stds': np.array(residual_stds),
            'window_starts': np.arange(0, len(x) - window + 1, step)
        }
    
    return results


def batch_rolling_analysis(data: pd.DataFrame,
                          pairs_df: pd.DataFrame,
                          p_threshold: float = 0.5) -> Dict:
    """
    对满足p值阈值的配对批量进行滚动分析
    
    Args:
        data: 最近1年的对数价格数据
        pairs_df: 所有配对的协整结果（需包含pvalue_5y列）
        p_threshold: p值筛选阈值
        
    Returns:
        Dict with pair names as keys
    """
    # 筛选满足阈值的配对
    selected = pairs_df[pairs_df['pvalue_5y'] < p_threshold].copy()
    
    logger.info(f"Selected {len(selected)} pairs with p<{p_threshold}")
    
    rolling_results = {}
    
    for idx, row in selected.iterrows():
        pair = row['pair']
        sym1 = row['symbol1']
        sym2 = row['symbol2']
        
        logger.info(f"Rolling analysis for {pair}")
        
        x = data[sym1].values
        y = data[sym2].values
        
        # 根据原始方向判定结果
        if row['direction'] == 'x_on_y':
            x, y = y, x  # 交换
        
        rolling_results[pair] = rolling_window_analysis(x, y)
    
    return rolling_results