"""
改进的β计算模块
基于名义价值×波动率对齐的动态β计算方法

Author: Metal Verify Research Team
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import sys
sys.path.append('/mnt/f/metal verify')
from configs.contract_specs import CONTRACT_SPECS


def calculate_volatility_adjusted_beta(
    y_prices: np.ndarray,
    x_prices: np.ndarray, 
    y_symbol: str,
    x_symbol: str,
    volatility_window: int = 30,
    price_method: str = 'close'
) -> Dict:
    """
    基于名义价值×波动率对齐的β计算
    
    核心思想：
    - 名义价值对齐：确保两边资金敞口相当
    - 波动率调整：确保两边风险敞口匹配
    - β = (名义价值Y × 波动率Y) / (名义价值X × 波动率X)
    
    Args:
        y_prices: Y品种价格序列（做多方）
        x_prices: X品种价格序列（做空方）
        y_symbol: Y品种代码
        x_symbol: X品种代码
        volatility_window: 波动率计算窗口
        price_method: 价格选择方法（'close', 'mean'等）
        
    Returns:
        {
            'beta': float,                    # 调整后的β系数
            'y_notional': float,             # Y品种名义价值
            'x_notional': float,             # X品种名义价值  
            'y_volatility': float,           # Y品种波动率
            'x_volatility': float,           # X品种波动率
            'risk_weighted_y': float,        # Y品种风险权重
            'risk_weighted_x': float,        # X品种风险权重
            'position_ratio': str,           # 建议仓位比例
            'explanation': str               # 计算解释
        }
    """
    
    # 获取合约规格
    y_specs = CONTRACT_SPECS[y_symbol]
    x_specs = CONTRACT_SPECS[x_symbol]
    
    # 获取最新价格（用于名义价值计算）
    y_current_price = y_prices[-1]
    x_current_price = x_prices[-1]
    
    # 计算名义价值（单手）
    y_notional_per_lot = y_current_price * y_specs['multiplier']
    x_notional_per_lot = x_current_price * x_specs['multiplier']
    
    # 计算近期波动率（使用收益率标准差）
    y_returns = np.diff(y_prices[-volatility_window:]) / y_prices[-volatility_window:-1]
    x_returns = np.diff(x_prices[-volatility_window:]) / x_prices[-volatility_window:-1]
    
    y_volatility = np.std(y_returns) * np.sqrt(252)  # 年化波动率
    x_volatility = np.std(x_returns) * np.sqrt(252)  # 年化波动率
    
    # 计算风险权重（名义价值 × 波动率）
    y_risk_weight = y_notional_per_lot * y_volatility
    x_risk_weight = x_notional_per_lot * x_volatility
    
    # 计算β系数（风险权重对齐）
    beta = y_risk_weight / x_risk_weight
    
    # 简化为整数比例
    ratio_y, ratio_x = simplify_ratio(beta)
    
    # 验证比例的有效性
    if ratio_y > 10 or ratio_x > 10:
        # 如果比例过大，使用名义价值对齐方法
        beta_nominal = y_notional_per_lot / x_notional_per_lot
        ratio_y, ratio_x = simplify_ratio(beta_nominal)
        explanation = f"比例过大，采用名义价值对齐: {beta_nominal:.3f} → {ratio_y}:{ratio_x}"
    else:
        explanation = f"风险权重对齐: {beta:.3f} → {ratio_y}:{ratio_x}"
    
    return {
        'beta': beta,
        'y_notional': y_notional_per_lot,
        'x_notional': x_notional_per_lot,
        'y_volatility': y_volatility,
        'x_volatility': x_volatility,
        'risk_weighted_y': y_risk_weight,
        'risk_weighted_x': x_risk_weight,
        'position_ratio': f"{ratio_y}:{ratio_x}",
        'ratio_y': ratio_y,
        'ratio_x': ratio_x,
        'explanation': explanation
    }


def simplify_ratio(ratio: float, max_ratio: int = 10) -> Tuple[int, int]:
    """
    将浮点数比例简化为整数比例
    
    Args:
        ratio: 浮点数比例
        max_ratio: 允许的最大整数比例
        
    Returns:
        (分子, 分母) 整数元组
        
    Example:
        >>> simplify_ratio(1.33)
        (4, 3)  # 1.33 ≈ 4/3
    """
    if ratio <= 0:
        return (1, 1)
    
    # 尝试不同的精度来找到合适的整数比例
    for denominator in range(1, max_ratio + 1):
        numerator = round(ratio * denominator)
        
        # 检查比例是否合理
        if numerator <= max_ratio and numerator > 0:
            actual_ratio = numerator / denominator
            error = abs(actual_ratio - ratio) / ratio
            
            # 如果误差小于5%，接受这个比例
            if error < 0.05:
                return (numerator, denominator)
    
    # 如果找不到合适的比例，返回最接近的1:1
    if ratio > 1:
        return (min(max_ratio, round(ratio)), 1)
    else:
        return (1, min(max_ratio, round(1/ratio)))


def calculate_dynamic_beta(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    y_symbol: str,
    x_symbol: str,
    method: str = 'volatility_adjusted'
) -> Dict:
    """
    计算动态β系数
    
    支持多种计算方法：
    1. 'volatility_adjusted': 波动率调整法（推荐）
    2. 'notional_value': 纯名义价值法
    3. 'equal_risk': 等风险法
    4. 'traditional': 传统协整法
    
    Args:
        y_prices: Y品种价格序列
        x_prices: X品种价格序列
        y_symbol: Y品种代码
        x_symbol: X品种代码
        method: 计算方法
        
    Returns:
        计算结果字典
    """
    
    if method == 'volatility_adjusted':
        return calculate_volatility_adjusted_beta(y_prices, x_prices, y_symbol, x_symbol)
    
    elif method == 'notional_value':
        return calculate_notional_value_beta(y_prices, x_prices, y_symbol, x_symbol)
    
    elif method == 'equal_risk':
        return calculate_equal_risk_beta(y_prices, x_prices, y_symbol, x_symbol)
    
    elif method == 'traditional':
        return calculate_traditional_beta(y_prices, x_prices, y_symbol, x_symbol)
    
    else:
        raise ValueError(f"不支持的计算方法: {method}")


def calculate_notional_value_beta(
    y_prices: np.ndarray,
    x_prices: np.ndarray, 
    y_symbol: str,
    x_symbol: str
) -> Dict:
    """纯名义价值对齐法"""
    
    y_specs = CONTRACT_SPECS[y_symbol]
    x_specs = CONTRACT_SPECS[x_symbol]
    
    y_notional = y_prices[-1] * y_specs['multiplier']
    x_notional = x_prices[-1] * x_specs['multiplier']
    
    beta = y_notional / x_notional
    ratio_y, ratio_x = simplify_ratio(beta)
    
    return {
        'beta': beta,
        'y_notional': y_notional,
        'x_notional': x_notional,
        'position_ratio': f"{ratio_y}:{ratio_x}",
        'ratio_y': ratio_y,
        'ratio_x': ratio_x,
        'explanation': f"名义价值对齐: {beta:.3f} → {ratio_y}:{ratio_x}"
    }


def calculate_equal_risk_beta(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    y_symbol: str, 
    x_symbol: str,
    volatility_window: int = 30
) -> Dict:
    """等风险敞口法"""
    
    y_specs = CONTRACT_SPECS[y_symbol]
    x_specs = CONTRACT_SPECS[x_symbol]
    
    # 计算波动率
    y_returns = np.diff(y_prices[-volatility_window:]) / y_prices[-volatility_window:-1]
    x_returns = np.diff(x_prices[-volatility_window:]) / x_prices[-volatility_window:-1]
    
    y_volatility = np.std(y_returns)
    x_volatility = np.std(x_returns)
    
    # 计算风险敞口（价格 × 合约乘数 × 波动率）
    y_risk = y_prices[-1] * y_specs['multiplier'] * y_volatility
    x_risk = x_prices[-1] * x_specs['multiplier'] * x_volatility
    
    beta = y_risk / x_risk
    ratio_y, ratio_x = simplify_ratio(beta)
    
    return {
        'beta': beta,
        'y_volatility': y_volatility,
        'x_volatility': x_volatility,
        'y_risk': y_risk,
        'x_risk': x_risk,
        'position_ratio': f"{ratio_y}:{ratio_x}",
        'ratio_y': ratio_y,
        'ratio_x': ratio_x,
        'explanation': f"等风险敞口: {beta:.3f} → {ratio_y}:{ratio_x}"
    }


def calculate_traditional_beta(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    y_symbol: str,
    x_symbol: str
) -> Dict:
    """传统协整β法"""
    
    from lib.coint import eg_test
    
    # 协整检验
    result = eg_test(x_prices, y_prices, direction='y_on_x')
    beta = result['beta']
    ratio_y, ratio_x = simplify_ratio(beta)
    
    return {
        'beta': beta,
        'pvalue': result['pvalue'],
        'position_ratio': f"{ratio_y}:{ratio_x}",
        'ratio_y': ratio_y,
        'ratio_x': ratio_x,
        'explanation': f"协整β: {beta:.3f} → {ratio_y}:{ratio_x}"
    }


def compare_beta_methods(
    y_prices: np.ndarray,
    x_prices: np.ndarray,
    y_symbol: str,
    x_symbol: str
) -> Dict:
    """
    对比不同β计算方法
    
    Returns:
        {
            'volatility_adjusted': {...},
            'notional_value': {...}, 
            'equal_risk': {...},
            'traditional': {...},
            'recommendation': str
        }
    """
    
    methods = ['volatility_adjusted', 'notional_value', 'equal_risk', 'traditional']
    results = {}
    
    for method in methods:
        try:
            results[method] = calculate_dynamic_beta(y_prices, x_prices, y_symbol, x_symbol, method)
        except Exception as e:
            results[method] = {'error': str(e)}
    
    # 推荐方法（默认使用波动率调整法）
    recommendation = 'volatility_adjusted'
    
    results['recommendation'] = recommendation
    return results


def calculate_portfolio_balance_score(beta_result: Dict) -> float:
    """
    计算配对平衡性评分
    
    考虑因素：
    1. 风险敞口是否匹配
    2. 名义价值是否合理
    3. 仓位比例是否可执行
    
    Args:
        beta_result: β计算结果
        
    Returns:
        0-100的平衡性评分
    """
    score = 0
    
    # 1. 仓位比例合理性（0-40分）
    ratio_y = beta_result.get('ratio_y', 1)
    ratio_x = beta_result.get('ratio_x', 1)
    max_ratio = max(ratio_y, ratio_x)
    
    if max_ratio <= 3:
        score += 40
    elif max_ratio <= 5:
        score += 30
    elif max_ratio <= 8:
        score += 20
    else:
        score += 10
    
    # 2. 风险权重平衡性（0-30分）
    if 'risk_weighted_y' in beta_result and 'risk_weighted_x' in beta_result:
        risk_ratio = beta_result['risk_weighted_y'] / beta_result['risk_weighted_x']
        risk_balance = min(risk_ratio, 1/risk_ratio)  # 0.5-1之间
        score += risk_balance * 30
    
    # 3. 波动率匹配度（0-30分）
    if 'y_volatility' in beta_result and 'x_volatility' in beta_result:
        vol_ratio = beta_result['y_volatility'] / beta_result['x_volatility']
        vol_balance = min(vol_ratio, 1/vol_ratio)  # 0.5-1之间
        score += vol_balance * 30
    
    return min(100, max(0, score))


def batch_calculate_improved_beta(data: pd.DataFrame, pairs: List[str]) -> pd.DataFrame:
    """
    批量计算改进的β系数
    
    Args:
        data: 价格数据DataFrame，列为品种代码
        pairs: 配对列表，格式['Y-X', ...]
        
    Returns:
        包含新β系数的DataFrame
    """
    results = []
    
    for pair in pairs:
        try:
            y_symbol, x_symbol = pair.split('-')
            
            if y_symbol not in data.columns or x_symbol not in data.columns:
                continue
            
            # 获取价格数据
            y_prices = data[y_symbol].dropna().values
            x_prices = data[x_symbol].dropna().values
            
            # 对齐数据长度
            min_len = min(len(y_prices), len(x_prices))
            y_prices = y_prices[-min_len:]
            x_prices = x_prices[-min_len:]
            
            # 计算新β
            beta_result = calculate_volatility_adjusted_beta(
                y_prices, x_prices, y_symbol, x_symbol
            )
            
            # 计算平衡性评分
            balance_score = calculate_portfolio_balance_score(beta_result)
            
            # 保存结果
            results.append({
                'pair': pair,
                'y_symbol': y_symbol,
                'x_symbol': x_symbol,
                'new_beta': beta_result['beta'],
                'position_ratio': beta_result['position_ratio'],
                'y_notional': beta_result['y_notional'],
                'x_notional': beta_result['x_notional'],
                'y_volatility': beta_result['y_volatility'],
                'x_volatility': beta_result['x_volatility'],
                'balance_score': balance_score,
                'explanation': beta_result['explanation']
            })
            
        except Exception as e:
            print(f"计算{pair}的β时出错: {e}")
            continue
    
    return pd.DataFrame(results)


def validate_beta_improvement(
    original_signals: pd.DataFrame,
    new_beta_results: pd.DataFrame
) -> Dict:
    """
    验证新β方法相比原方法的改进效果
    
    Args:
        original_signals: 原始信号数据
        new_beta_results: 新β计算结果
        
    Returns:
        改进效果分析
    """
    
    improvements = {
        'beta_changes': [],
        'ratio_changes': [],
        'balance_improvements': [],
        'summary': {}
    }
    
    for _, new_result in new_beta_results.iterrows():
        pair = new_result['pair']
        
        # 查找原始β
        original_pair_data = original_signals[original_signals['pair'] == pair]
        if len(original_pair_data) == 0:
            continue
        
        original_beta = original_pair_data['beta'].iloc[0] if 'beta' in original_pair_data.columns else 1.0
        original_ratio = original_pair_data['position_ratio'].iloc[0] if 'position_ratio' in original_pair_data.columns else '1:1'
        
        # 记录变化
        improvements['beta_changes'].append({
            'pair': pair,
            'original_beta': original_beta,
            'new_beta': new_result['new_beta'],
            'beta_change': new_result['new_beta'] - original_beta,
            'beta_change_pct': (new_result['new_beta'] - original_beta) / original_beta * 100 if original_beta != 0 else 0
        })
        
        improvements['ratio_changes'].append({
            'pair': pair,
            'original_ratio': original_ratio,
            'new_ratio': new_result['position_ratio'],
            'balance_score': new_result['balance_score']
        })
    
    # 计算改进摘要
    if improvements['beta_changes']:
        beta_changes_df = pd.DataFrame(improvements['beta_changes'])
        avg_beta_change = beta_changes_df['beta_change_pct'].mean()
        significant_changes = (abs(beta_changes_df['beta_change_pct']) > 10).sum()
        
        improvements['summary'] = {
            'total_pairs': len(beta_changes_df),
            'avg_beta_change_pct': avg_beta_change,
            'significant_changes': significant_changes,
            'avg_balance_score': new_beta_results['balance_score'].mean()
        }
    
    return improvements


if __name__ == "__main__":
    # 简单测试
    print("β计算模块加载成功")
    print("支持的方法：")
    print("1. volatility_adjusted - 波动率调整法（推荐）")
    print("2. notional_value - 名义价值对齐法") 
    print("3. equal_risk - 等风险敞口法")
    print("4. traditional - 传统协整法")