"""
协整分析模块
处理协整检验、方向判定、批量筛选

Test: tests/test_coint.py
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from statsmodels.tsa.stattools import adfuller, coint as sm_coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def eg_test(x: np.ndarray, y: np.ndarray, direction: str = 'y_on_x') -> Dict:
    """
    Engle-Granger两步法协整检验
    
    Args:
        x: 价格序列1
        y: 价格序列2
        direction: 回归方向 'y_on_x' 或 'x_on_y'
        
    Returns:
        {
            'pvalue': float,        # p值
            'adf_stat': float,      # ADF统计量
            'beta': float,          # 回归系数
            'residuals': array,     # 残差序列
            'direction': str        # 回归方向
        }
        
    Example:
        >>> result = eg_test(x, y)
        >>> print(f"P-value: {result['pvalue']:.4f}")
    """
    # 确保输入是numpy数组
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 检查长度
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")
    
    # 执行回归
    if direction == 'y_on_x':
        # Y ~ X
        X = add_constant(x)
        model = OLS(y, X).fit()
        beta = model.params[1]
        residuals = model.resid
    else:
        # X ~ Y
        Y = add_constant(y)
        model = OLS(x, Y).fit()
        beta = model.params[1]
        residuals = model.resid
    
    # ADF检验残差平稳性
    adf_result = adfuller(residuals, maxlag=1, regression='n', autolag=None)
    adf_stat = adf_result[0]
    pvalue = adf_result[1]
    
    return {
        'pvalue': pvalue,
        'adf_stat': adf_stat,
        'beta': beta,
        'residuals': residuals,
        'direction': direction
    }


def calculate_halflife(residuals: np.ndarray) -> Optional[float]:
    """
    计算均值回复半衰期（使用OU过程）
    
    Args:
        residuals: 残差序列或价格序列
        
    Returns:
        半衰期（天数），如果无法计算返回None
        
    Example:
        >>> hl = calculate_halflife(residuals)
        >>> print(f"半衰期: {hl:.1f}天")
    """
    residuals = np.asarray(residuals).flatten()
    
    # 使用AR(1)模型估计均值回复速度
    # y_t = a + b * y_{t-1} + e_t
    # 半衰期 = -ln(2) / ln(b)
    
    try:
        y = residuals[1:]
        x = residuals[:-1]
        
        # 添加常数项
        X = add_constant(x)
        model = OLS(y, X).fit()
        
        # AR系数
        b = model.params[1]
        
        # 如果b >= 1或b <= 0，无法计算有效半衰期
        if b >= 1 or b <= 0:
            return None
        
        # 计算半衰期
        halflife = -np.log(2) / np.log(b)
        
        # 只保留基本合理性检查
        if halflife < 0:
            return None
            
        return halflife
        
    except:
        return None


def find_direction(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    确定最优回归方向
    
    综合考虑：
    1. p值（协整显著性）
    2. 波动率（低波动作为自变量）
    3. Beta稳定性
    
    Args:
        x: 价格序列1
        y: 价格序列2
        
    Returns:
        {
            'recommended_direction': str,  # 'y_on_x' 或 'x_on_y'
            'volatility_x': float,
            'volatility_y': float,
            'pvalue_yx': float,
            'pvalue_xy': float,
            'beta_stability_yx': float,
            'beta_stability_xy': float,
            'score': float  # 0-100
        }
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 1. 计算波动率（使用收益率的标准差）
    ret_x = np.diff(x) / x[:-1]
    ret_y = np.diff(y) / y[:-1]
    vol_x = np.std(ret_x)
    vol_y = np.std(ret_y)
    
    # 2. 双向协整检验
    result_yx = eg_test(x, y, direction='y_on_x')
    result_xy = eg_test(x, y, direction='x_on_y')
    
    # 3. 计算Beta稳定性（滚动窗口标准差）
    window = min(60, len(x) // 4)
    betas_yx = []
    betas_xy = []
    
    for i in range(window, len(x)):
        x_win = x[i-window:i]
        y_win = y[i-window:i]
        
        # Y~X
        X_win = add_constant(x_win)
        beta_yx = OLS(y_win, X_win).fit().params[1]
        betas_yx.append(beta_yx)
        
        # X~Y
        Y_win = add_constant(y_win)
        beta_xy = OLS(x_win, Y_win).fit().params[1]
        betas_xy.append(beta_xy)
    
    beta_stability_yx = np.std(betas_yx) if betas_yx else float('inf')
    beta_stability_xy = np.std(betas_xy) if betas_xy else float('inf')
    
    # 4. 综合评分
    # 低波动率作为自变量（+30分）
    # p值更小（+40分）
    # Beta更稳定（+30分）
    
    score_yx = 0
    score_xy = 0
    
    # 波动率评分
    if vol_x < vol_y:
        score_yx += 30  # x波动小，适合作为自变量（Y~X）
    else:
        score_xy += 30  # y波动小，适合作为自变量（X~Y）
    
    # p值评分
    if result_yx['pvalue'] < result_xy['pvalue']:
        score_yx += 40
    else:
        score_xy += 40
    
    # Beta稳定性评分
    if beta_stability_yx < beta_stability_xy:
        score_yx += 30
    else:
        score_xy += 30
    
    recommended = 'y_on_x' if score_yx >= score_xy else 'x_on_y'
    
    return {
        'recommended_direction': recommended,
        'volatility_x': vol_x,
        'volatility_y': vol_y,
        'pvalue_yx': result_yx['pvalue'],
        'pvalue_xy': result_xy['pvalue'],
        'beta_stability_yx': beta_stability_yx,
        'beta_stability_xy': beta_stability_xy,
        'score': max(score_yx, score_xy)
    }


def screen_pairs(
    data: pd.DataFrame, 
    p_threshold: float = 0.1,
    halflife_range: Tuple[float, float] = (1, 365)
) -> List[Dict]:
    """
    批量筛选协整配对
    
    Args:
        data: 宽表数据，列为各品种价格
        p_threshold: p值阈值
        halflife_range: 半衰期范围
        
    Returns:
        配对列表，按p值升序排序
        [
            {
                'pair': 'RB0-HC0',
                'symbol1': 'RB0',
                'symbol2': 'HC0',
                'pvalue': 0.01,
                'beta': 1.2,
                'halflife': 10,
                'direction': 'RB0_on_HC0'
            }
        ]
    """
    results = []
    symbols = data.columns.tolist()
    
    # 生成所有配对组合
    for sym1, sym2 in combinations(symbols, 2):
        try:
            # 获取价格序列
            x = data[sym1].values
            y = data[sym2].values
            
            # 移除NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) < 100:  # 数据太少
                continue
            
            # 找最优方向
            direction_result = find_direction(x, y)
            
            # 使用推荐方向进行协整检验
            if direction_result['recommended_direction'] == 'y_on_x':
                coint_result = eg_test(x, y, direction='y_on_x')
                direction_str = f"{sym2}_on_{sym1}"
            else:
                coint_result = eg_test(x, y, direction='x_on_y')
                direction_str = f"{sym1}_on_{sym2}"
            
            # 检查p值
            if coint_result['pvalue'] > p_threshold:
                continue
            
            # 计算半衰期
            halflife = calculate_halflife(coint_result['residuals'])
            
            # 检查半衰期范围
            if halflife is not None:
                if halflife < halflife_range[0] or halflife > halflife_range[1]:
                    continue
            
            # 添加结果
            results.append({
                'pair': f"{sym1}-{sym2}",
                'symbol1': sym1,
                'symbol2': sym2,
                'pvalue': coint_result['pvalue'],
                'beta': coint_result['beta'],
                'halflife': halflife,
                'direction': direction_str
            })
            
        except Exception as e:
            # 跳过有问题的配对
            continue
    
    # 按p值排序
    results.sort(key=lambda x: x['pvalue'])
    
    return results


def multi_window_analysis(
    x: np.ndarray, 
    y: np.ndarray,
    windows: List[int] = None
) -> Dict[int, Dict]:
    """
    多时间窗口协整分析
    
    Args:
        x: 价格序列1
        y: 价格序列2
        windows: 窗口大小列表（数据点数）
        
    Returns:
        {
            252: {'pvalue': 0.01, 'beta': 1.2, ...},
            504: {'pvalue': 0.02, 'beta': 1.3, ...},
            ...
        }
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if windows is None:
        # 默认：1年、2年、5年（约）
        windows = [252, 504, 1260]
    
    results = {}
    
    for window in windows:
        if window > len(x):
            continue
        
        # 使用最近的window个数据点
        x_window = x[-window:]
        y_window = y[-window:]
        
        # 协整检验
        coint_result = eg_test(x_window, y_window)
        
        # 计算半衰期
        halflife = calculate_halflife(coint_result['residuals'])
        
        results[window] = {
            'window_size': window,
            'pvalue': coint_result['pvalue'],
            'beta': coint_result['beta'],
            'halflife': halflife,
            'adf_stat': coint_result['adf_stat']
        }
    
    return results


def calculate_stability_score(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    计算跨时间窗口的稳定性评分
    
    Args:
        x: 价格序列1
        y: 价格序列2
        
    Returns:
        {
            'score': float,  # 0-100
            'beta_std': float,
            'pvalue_consistency': float,
            'window_results': dict
        }
    """
    # 多窗口分析
    windows = [252, 504, 1260]
    window_results = multi_window_analysis(x, y, windows)
    
    if len(window_results) < 2:
        return {
            'score': 0,
            'beta_std': float('inf'),
            'pvalue_consistency': 0,
            'window_results': window_results
        }
    
    # 提取指标
    betas = [r['beta'] for r in window_results.values()]
    pvalues = [r['pvalue'] for r in window_results.values()]
    
    # Beta稳定性（标准差越小越稳定）
    beta_std = np.std(betas)
    
    # P值一致性（都小于0.05的比例）
    pvalue_consistency = sum(p < 0.05 for p in pvalues) / len(pvalues)
    
    # 计算评分
    # Beta稳定性评分（0-50分）
    beta_score = max(0, 50 - beta_std * 100)
    
    # P值一致性评分（0-50分）
    pvalue_score = pvalue_consistency * 50
    
    score = beta_score + pvalue_score
    
    return {
        'score': min(100, max(0, score)),
        'beta_std': beta_std,
        'pvalue_consistency': pvalue_consistency,
        'window_results': window_results
    }


def calculate_coint_score(metrics: Dict) -> float:
    """
    计算协整强度综合评分
    
    Args:
        metrics: {
            'pvalue': float,
            'adf_stat': float,
            'halflife': float,
            'beta_stability': float
        }
        
    Returns:
        0-100的综合评分
    """
    score = 0
    
    # P值评分（0-40分）
    pvalue = metrics.get('pvalue', 1.0)
    if pvalue < 0.01:
        score += 40
    elif pvalue < 0.05:
        score += 30
    elif pvalue < 0.1:
        score += 20
    else:
        score += max(0, 10 * (1 - pvalue))
    
    # ADF统计量评分（0-30分）
    adf_stat = metrics.get('adf_stat', 0)
    if adf_stat < -4:
        score += 30
    elif adf_stat < -3:
        score += 20
    elif adf_stat < -2:
        score += 10
    
    # 半衰期评分（0-20分）
    halflife = metrics.get('halflife', 100)
    if halflife is not None:
        if 1 <= halflife <= 30:
            score += 20
        elif 30 < halflife <= 60:
            score += 10
        elif 60 < halflife <= 90:
            score += 5
    
    # Beta稳定性评分（0-10分）
    beta_stability = metrics.get('beta_stability', 1.0)
    if beta_stability < 0.1:
        score += 10
    elif beta_stability < 0.3:
        score += 7
    elif beta_stability < 0.5:
        score += 4
    
    return min(100, max(0, score))