"""
方向判定模块
基于波动率判定协整回归方向
"""
import numpy as np
from typing import Tuple, Dict


def determine_direction_by_volatility(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    基于波动率判定回归方向
    低波动品种作为自变量
    
    Args:
        x: 对数价格序列1
        y: 对数价格序列2
        
    Returns:
        {
            'direction': str,  # 'y_on_x' 或 'x_on_y'
            'independent': str,  # 自变量名称 'x' 或 'y'
            'dependent': str,  # 因变量名称 'y' 或 'x'
            'volatility_x': float,
            'volatility_y': float
        }
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 计算收益率波动率
    ret_x = np.diff(x) / x[:-1]
    ret_y = np.diff(y) / y[:-1]
    vol_x = np.std(ret_x)
    vol_y = np.std(ret_y)
    
    # 低波动品种作为自变量
    if vol_x < vol_y:
        # x波动率低，x作为自变量，回归方向是y~x
        direction = 'y_on_x'
        independent = 'x'
        dependent = 'y'
    else:
        # y波动率低，y作为自变量，回归方向是x~y
        direction = 'x_on_y'
        independent = 'y'
        dependent = 'x'
    
    return {
        'direction': direction,
        'independent': independent,
        'dependent': dependent,
        'volatility_x': vol_x,
        'volatility_y': vol_y
    }