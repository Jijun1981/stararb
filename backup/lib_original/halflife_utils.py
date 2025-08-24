"""
半衰期计算工具
提供更宽松和详细的半衰期计算
"""
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from typing import Optional, Dict


def calculate_halflife_detailed(residuals: np.ndarray, 
                               adf_threshold: float = 0.2,
                               max_halflife: float = 500) -> Dict:
    """
    计算半衰期并返回详细信息
    
    Args:
        residuals: 残差序列
        adf_threshold: ADF检验p值阈值（默认0.2，更宽松）
        max_halflife: 最大合理半衰期（默认500天）
        
    Returns:
        {
            'halflife': float or None,
            'ar_coef': float,  # AR系数
            'adf_pvalue': float,  # ADF检验p值
            'is_stationary': bool,  # 是否平稳
            'reason': str  # 如果返回None的原因
        }
    """
    residuals = np.asarray(residuals).flatten()
    
    # ADF检验
    adf_result = adfuller(residuals, maxlag=1, regression='c', autolag=None)
    adf_pvalue = adf_result[1]
    is_stationary = adf_pvalue < adf_threshold
    
    # 使用AR(1)模型
    y = residuals[1:]
    x = residuals[:-1]
    
    # 添加常数项
    X = add_constant(x)
    model = OLS(y, X).fit()
    
    # AR系数
    b = model.params[1]
    
    result = {
        'ar_coef': b,
        'adf_pvalue': adf_pvalue,
        'is_stationary': is_stationary,
        'halflife': None,
        'reason': ''
    }
    
    # 计算半衰期
    if not is_stationary:
        result['reason'] = f'非平稳(ADF p={adf_pvalue:.4f})'
    elif b >= 1:
        result['reason'] = f'AR系数>=1 (b={b:.4f})'
    elif b <= 0:
        result['reason'] = f'AR系数<=0 (b={b:.4f})'
    else:
        # 可以计算半衰期
        halflife = -np.log(2) / np.log(b)
        
        if halflife < 0:
            result['reason'] = f'半衰期<0 (HL={halflife:.1f})'
        elif halflife > max_halflife:
            result['reason'] = f'半衰期>{max_halflife} (HL={halflife:.1f})'
        else:
            result['halflife'] = halflife
            result['reason'] = 'OK'
    
    return result


def calculate_halflife_relaxed(residuals: np.ndarray) -> Optional[float]:
    """
    更宽松的半衰期计算（用于替换原版本）
    
    Args:
        residuals: 残差序列
        
    Returns:
        半衰期（天数），如果无法计算返回None
    """
    result = calculate_halflife_detailed(
        residuals, 
        adf_threshold=0.2,  # 放宽到20%
        max_halflife=500    # 允许更长的半衰期
    )
    return result['halflife']


def diagnose_halflife_issues(data: np.ndarray, symbol1: str, symbol2: str) -> None:
    """
    诊断半衰期计算问题
    
    Args:
        data: 价格数据
        symbol1, symbol2: 品种名称
    """
    from .coint import eg_test
    
    x = data[symbol1].values
    y = data[symbol2].values
    
    # 协整检验
    result = eg_test(x, y)
    
    # 详细半衰期分析
    hl_detail = calculate_halflife_detailed(result['residuals'])
    
    print(f"\n{symbol1}-{symbol2} 半衰期诊断:")
    print("="*50)
    print(f"协整p值: {result['pvalue']:.6f}")
    print(f"Beta系数: {result['beta']:.4f}")
    print(f"\n半衰期分析:")
    print(f"  ADF p值: {hl_detail['adf_pvalue']:.6f}")
    print(f"  是否平稳: {hl_detail['is_stationary']}")
    print(f"  AR系数: {hl_detail['ar_coef']:.6f}")
    print(f"  半衰期: {hl_detail['halflife']:.2f}天" if hl_detail['halflife'] else f"  半衰期: None")
    print(f"  原因: {hl_detail['reason']}")
    
    # 残差统计
    residuals = result['residuals']
    print(f"\n残差统计:")
    print(f"  均值: {np.mean(residuals):.6f}")
    print(f"  标准差: {np.std(residuals):.6f}")
    print(f"  最小值: {np.min(residuals):.6f}")
    print(f"  最大值: {np.max(residuals):.6f}")