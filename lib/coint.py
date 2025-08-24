#!/usr/bin/env python3
"""
协整配对模块
实现REQ-2.x.x相关需求

功能:
- Engle-Granger两步法协整检验
- 多时间窗口协整分析  
- 基于波动率的方向判定
- 参数估计和统计分析
- 批量配对筛选

Author: Claude Code
Date: 2025-08-20
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging
from itertools import combinations
from datetime import datetime, timedelta
import warnings

# 统计计算库
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS  
from statsmodels.tools import add_constant
from scipy import stats

# 抑制警告
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 异常定义
class CointegrationError(Exception):
    """协整分析异常"""
    pass

class DataError(Exception):
    """数据相关异常"""
    pass


def get_window_days(window_name: str, windows: Optional[Dict[str, int]] = None) -> int:
    """
    获取窗口名称对应的天数
    
    Args:
        window_name: 窗口名称，如'1y', '2y', '6m'
        windows: 自定义窗口字典
    
    Returns:
        窗口对应的交易日天数
    """
    # 首先检查自定义窗口
    if windows is not None and window_name in windows:
        return windows[window_name]
    
    # 默认窗口映射
    default_windows = {
        '5y': 1260,
        '4y': 1008, 
        '3y': 756,
        '2y': 504,
        '1y': 252,
        '18m': 378,
        '6m': 126
    }
    
    return default_windows.get(window_name, 0)


def engle_granger_test(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Engle-Granger两步法协整检验
    
    实现: REQ-2.1.1
    
    Args:
        x: 价格序列1 (自变量X)
        y: 价格序列2 (因变量Y)
        
    Returns:
        {
            'pvalue': float,        # ADF检验p值
            'adf_stat': float,      # ADF统计量
            'beta': float,          # 回归系数β
            'alpha': float,         # 截距α
            'residuals': np.ndarray,# 残差序列
            'r_squared': float,     # R²
            'n_obs': int           # 观测数量
        }
        
    Raises:
        CointegrationError: 输入数据无效
    """
    # 输入验证
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise CointegrationError(f"x和y长度不匹配: {len(x)} vs {len(y)}")
    
    if len(x) < 20:
        raise CointegrationError(f"数据点太少: {len(x)} < 20")
    
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise CointegrationError("输入数据包含NaN值")
    
    try:
        # 第一步: 统一使用Y对X回归 Y = α + β*X + ε
        X = add_constant(x)
        model = OLS(y, X).fit()
        alpha = model.params[0]
        beta = model.params[1] 
        residuals = model.resid
        
        # 第二步: ADF检验残差平稳性
        # 使用无截距、无趋势的ADF检验
        maxlag = int(np.floor(len(residuals)**(1/3)))  # T^(1/3)公式
        adf_result = adfuller(residuals, maxlag=maxlag, regression='n', autolag='AIC')
        
        adf_stat = adf_result[0]
        pvalue = adf_result[1] 
        
        return {
            'pvalue': pvalue,
            'adf_stat': adf_stat,
            'beta': round(beta, 6),  # REQ-2.3.1: β系数精确到6位小数
            'alpha': alpha,
            'residuals': residuals,
            'r_squared': model.rsquared,
            'n_obs': len(x)
        }
        
    except Exception as e:
        logger.error(f"Engle-Granger检验失败: {str(e)}")
        raise CointegrationError(f"协整检验计算失败: {str(e)}")


def multi_window_test(x: np.ndarray, y: np.ndarray, 
                     windows: Optional[Dict[str, int]] = None) -> Dict:
    """
    多时间窗口协整检验
    
    实现: REQ-2.1.2至REQ-2.1.6
    
    Args:
        x: 价格序列1
        y: 价格序列2
        windows: 自定义时间窗口配置，格式为 {'名称': 交易日数量}
        
    Returns:
        按窗口名称返回检验结果字典，数据不足的窗口返回None
    """
    # 默认时间窗口定义 (交易日)
    if windows is None:
        windows = {
            '1y': 252,   # 1年 = 252
            '2y': 504,   # 2年 = 2×252
            '3y': 756,   # 3年 = 3×252
            '4y': 1008,  # 4年 = 4×252  
            '5y': 1260,  # 5年 = 5×252
        }
    
    results = {}
    total_length = len(x)
    
    for window_name, window_size in windows.items():
        if total_length >= window_size:
            try:
                # 使用最新的window_size个数据点
                x_window = x[-window_size:]
                y_window = y[-window_size:]
                
                result = engle_granger_test(x_window, y_window)
                results[window_name] = result
                
            except Exception as e:
                logger.warning(f"{window_name}窗口协整检验失败: {str(e)}")
                results[window_name] = None
        else:
            logger.info(f"{window_name}窗口数据不足: {total_length} < {window_size}")
            results[window_name] = None
    
    return results


def adf_test(series: np.ndarray) -> Tuple[float, float]:
    """
    ADF平稳性检验
    
    Args:
        series: 时间序列
        
    Returns:
        (adf_statistic, p_value)
    """
    try:
        result = adfuller(series, autolag='AIC')
        return result[0], result[1]
    except Exception as e:
        logger.error(f"ADF检验失败: {str(e)}")
        return np.nan, np.nan


def calculate_volatility(log_prices: np.ndarray, 
                        dates: pd.DatetimeIndex,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> float:
    """
    计算年化波动率
    
    实现: REQ-2.2.1至REQ-2.2.3, REQ-2.2.5, REQ-2.2.6, REQ-2.2.7
    
    Args:
        log_prices: 对数价格序列
        dates: 对应的日期索引
        start_date: 波动率计算起始日期（可选）
        end_date: 波动率计算结束日期（可选）
        
    Returns:
        年化波动率
    """
    try:
        # REQ-2.2.5, REQ-2.2.6: 支持指定时间段计算
        if start_date is None and end_date is None:
            # REQ-2.2.6: 默认使用全部可用数据
            mask = np.ones(len(dates), dtype=bool)
        else:
            mask = np.ones(len(dates), dtype=bool)
            
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                mask &= (dates >= start_dt)
            
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                mask &= (dates <= end_dt)
        
        if not np.any(mask):
            logger.warning(f"指定时间段内没有数据，使用全部数据")
            recent_prices = log_prices
        else:
            recent_prices = log_prices[mask]
        
        if len(recent_prices) < 2:
            raise DataError("计算波动率的数据点不足")
        
        # 计算对数收益率: returns = np.diff(np.log(prices))
        returns = np.diff(recent_prices)
        
        # 计算年化波动率: vol = std(returns) * sqrt(252)
        volatility = np.std(returns) * np.sqrt(252)
        
        return float(volatility)
        
    except Exception as e:
        logger.error(f"波动率计算失败: {str(e)}")
        return np.nan


def determine_direction(series_1: np.ndarray, series_2: np.ndarray,
                       dates_1: pd.DatetimeIndex, dates_2: pd.DatetimeIndex,
                       symbol_1: str, symbol_2: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Tuple[str, str, str]:
    """
    基于波动率确定回归方向
    
    实现: REQ-2.2.4至REQ-2.2.7
    
    Args:
        series_1: 品种1的对数价格
        series_2: 品种2的对数价格
        dates_1: 品种1的日期
        dates_2: 品种2的日期  
        symbol_1: 品种1代码
        symbol_2: 品种2代码
        start_date: 计算波动率的起始日期（可选）
        end_date: 计算波动率的结束日期（可选）
        
    Returns:
        (direction, symbol_x, symbol_y)
        direction: 'y_on_x' 或 'x_on_y'
        symbol_x: 自变量品种 (低波动率)
        symbol_y: 因变量品种 (高波动率)
    """
    try:
        # 计算两个品种的波动率
        vol_1 = calculate_volatility(series_1, dates_1, start_date, end_date)
        vol_2 = calculate_volatility(series_2, dates_2, start_date, end_date)
        
        if np.isnan(vol_1) or np.isnan(vol_2):
            logger.warning("波动率计算失败，使用默认方向")
            return 'y_on_x', symbol_1, symbol_2
        
        # REQ-2.2.4: 低波动作为X，高波动作为Y
        # 判断逻辑：谁波动率低，谁作为X（自变量）
        if vol_1 < vol_2:
            # series_1波动率更低，作为自变量X
            # 所以是 series_2 对 series_1 回归 (y_on_x)
            direction = 'y_on_x'
            symbol_x = symbol_1  # 低波动
            symbol_y = symbol_2  # 高波动
        elif vol_1 > vol_2:
            # series_2波动率更低，作为自变量X
            # 所以是 series_1 对 series_2 回归 (x_on_y)
            direction = 'x_on_y'
            symbol_x = symbol_2  # 低波动
            symbol_y = symbol_1  # 高波动
        else:
            # REQ-2.2.4: 波动率相等时默认选择第一个为X
            direction = 'y_on_x'
            symbol_x = symbol_1
            symbol_y = symbol_2
            
        logger.info(f"方向判定: {symbol_x}(vol={vol_1 if symbol_x==symbol_1 else vol_2:.4f}) -> "
                   f"{symbol_y}(vol={vol_2 if symbol_y==symbol_2 else vol_1:.4f}), {direction}")
        
        return direction, symbol_x, symbol_y
        
    except Exception as e:
        logger.error(f"方向判定失败: {str(e)}")
        return 'y_on_x', symbol_1, symbol_2


def estimate_parameters(x: np.ndarray, y: np.ndarray, 
                       window: str = '5y') -> Dict:
    """
    估计配对参数
    
    实现: REQ-2.3.1至REQ-2.3.6
    
    Args:
        x: 自变量序列
        y: 因变量序列
        window: 时间窗口标识
        
    Returns:
        参数字典
    """
    try:
        # 基础回归
        X = add_constant(x)
        model = OLS(y, X).fit()
        
        beta = model.params[1]
        alpha = model.params[0]
        residuals = model.resid
        
        # REQ-2.3.2: 计算半衰期
        halflife = calculate_halflife(residuals)
        
        # REQ-2.3.3: 残差统计
        residual_stats = residual_statistics(residuals)
        
        # REQ-2.3.5: R²和调整R²  
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        
        return {
            'window': window,
            'beta': round(beta, 6),  # REQ-2.3.1: β系数精确到6位小数
            'alpha': alpha, 
            'halflife': halflife,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            **residual_stats  # 解包残差统计
        }
        
    except Exception as e:
        logger.error(f"参数估计失败: {str(e)}")
        return {
            'window': window,
            'beta': np.nan,
            'alpha': np.nan,
            'halflife': np.nan,
            'r_squared': np.nan,
            'adj_r_squared': np.nan,
            'residual_mean': np.nan,
            'residual_std': np.nan,
            'residual_skew': np.nan,
            'residual_kurt': np.nan,
            'jarque_bera_pvalue': np.nan
        }


def calculate_halflife(residuals: np.ndarray) -> Optional[float]:
    """
    计算均值回复半衰期
    
    实现: REQ-2.3.2
    方法: halflife = -log(2) / log(1 + λ)，其中λ来自AR(1): ε_t = λ * ε_{t-1} + u_t
    
    Args:
        residuals: 残差序列
        
    Returns:
        半衰期（天数），失败返回None
    """
    try:
        if len(residuals) < 10:
            return None
            
        # 使用标准AR(1)模型: ε_t = α + λ * ε_{t-1} + u_t
        lagged_resid = residuals[:-1]  # ε_{t-1}
        current_resid = residuals[1:]  # ε_t
        
        if len(lagged_resid) > 0 and np.var(lagged_resid) > 1e-10:
            # 添加截距项
            X = add_constant(lagged_resid)
            model = OLS(current_resid, X).fit()
            lambda_coef = model.params[1]  # AR系数（不是截距）
            
            # 半衰期计算：当ε_t衰减到初始值的一半所需时间
            # 对于AR(1)过程，半衰期 = log(0.5) / log(λ)，其中0 < λ < 1
            if 0 < lambda_coef < 1:
                halflife = np.log(0.5) / np.log(lambda_coef)
                if halflife > 0:
                    return float(halflife)
            elif lambda_coef < 0:
                # 如果系数为负，使用原方法
                halflife = -np.log(2) / lambda_coef
                if halflife > 0:
                    return float(halflife)
            
        return None
        
    except Exception as e:
        logger.warning(f"半衰期计算失败: {str(e)}")
        return None


def residual_statistics(residuals: np.ndarray) -> Dict:
    """
    计算残差统计特征
    
    实现: REQ-2.3.3, REQ-2.3.6
    
    Args:
        residuals: 残差序列
        
    Returns:
        统计特征字典
    """
    try:
        # 基础统计
        mean = float(np.mean(residuals))
        std = float(np.std(residuals, ddof=1))  # 样本标准差
        skewness = float(stats.skew(residuals))
        kurtosis = float(stats.kurtosis(residuals))
        
        # REQ-2.3.6: Jarque-Bera正态性检验
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            jb_pvalue = float(jb_pvalue)
        except:
            jb_pvalue = np.nan
        
        return {
            'residual_mean': mean,
            'residual_std': std,
            'residual_skew': skewness,
            'residual_kurt': kurtosis,
            'jarque_bera_pvalue': jb_pvalue
        }
        
    except Exception as e:
        logger.error(f"残差统计计算失败: {str(e)}")
        return {
            'residual_mean': np.nan,
            'residual_std': np.nan,
            'residual_skew': np.nan,
            'residual_kurt': np.nan,
            'jarque_bera_pvalue': np.nan
        }


def screen_all_pairs(analyzer: 'CointegrationAnalyzer', 
                    p_threshold: float = 0.05,
                    halflife_min: Optional[float] = None,
                    halflife_max: Optional[float] = None,
                    use_halflife_filter: bool = False) -> pd.DataFrame:
    """
    批量筛选所有配对
    
    实现: REQ-2.4.1至REQ-2.4.8
    
    Args:
        analyzer: 协整分析器实例
        p_threshold: p值筛选阈值
        halflife_min: 最小半衰期阈值
        halflife_max: 最大半衰期阈值  
        use_halflife_filter: 是否启用半衰期筛选
        
    Returns:
        筛选结果DataFrame
    """
    try:
        return analyzer.screen_all_pairs(
            p_threshold=p_threshold,
            halflife_min=halflife_min,
            halflife_max=halflife_max,
            use_halflife_filter=use_halflife_filter
        )
    except Exception as e:
        logger.error(f"批量筛选失败: {str(e)}")
        return pd.DataFrame()


def export_results(results: pd.DataFrame, filepath: str) -> None:
    """
    导出分析结果
    
    实现: REQ-2.4.6
    
    Args:
        results: 分析结果DataFrame
        filepath: 输出文件路径
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"结果已导出至: {filepath}")
    except Exception as e:
        logger.error(f"结果导出失败: {str(e)}")


class CointegrationAnalyzer:
    """
    协整分析器
    
    统一的协整分析接口，封装所有协整相关操作
    实现接口定义 4.1
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化分析器
        
        Args:
            data: 对数价格数据，index为日期，columns为品种代码
        """
        self.data = data.copy()
        self.symbols = list(data.columns)
        self.n_symbols = len(self.symbols)
        
        # 验证数据
        if self.n_symbols < 2:
            raise DataError("至少需要2个品种进行协整分析")
        
        if len(data) < 252:
            logger.warning(f"数据长度较短: {len(data)}天，建议至少252天（1年）")
        
        logger.info(f"协整分析器初始化完成: {self.n_symbols}个品种，{len(data)}个数据点")
    
    def engle_granger_test(self, x: np.ndarray, y: np.ndarray, 
                          window: str = '5y') -> Dict:
        """协整检验接口"""
        return engle_granger_test(x, y)
    
    def multi_window_test(self, x: np.ndarray, y: np.ndarray,
                         windows: Optional[Dict[str, int]] = None) -> Dict:
        """多窗口协整检验接口"""
        return multi_window_test(x, y, windows)
    
    def adf_test(self, series: np.ndarray) -> Tuple[float, float]:
        """ADF检验接口"""
        return adf_test(series)
    
    def calculate_volatility(self, log_prices: np.ndarray,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> float:
        """波动率计算接口"""
        # 创建对应长度的日期索引
        if len(log_prices) == len(self.data.index):
            dates = self.data.index
        else:
            # 如果长度不匹配，使用最新的对应长度的日期
            dates = self.data.index[-len(log_prices):]
        return calculate_volatility(log_prices, dates, start_date, end_date)
    
    def determine_direction(self, symbol_x: str, symbol_y: str, 
                          use_recent: bool = True,
                          recent_start: Optional[str] = None) -> Tuple[str, str, str]:
        """
        方向判定接口
        
        Args:
            symbol_x: 品种1代码
            symbol_y: 品种2代码
            use_recent: 是否使用最近数据计算波动率
            recent_start: 波动率计算起始日期（默认为最近1年）
            
        Returns:
            (direction, symbol_x_low_vol, symbol_y_high_vol)
        """
        # 如果use_recent为True但没有提供recent_start，使用默认值None（会自动计算最近1年）
        if use_recent and recent_start is None:
            start_date = None  # 将使用calculate_volatility的默认行为（最近1年）
        elif use_recent:
            start_date = recent_start  # 使用用户提供的起始日期
        else:
            start_date = None  # 使用全部数据
            
        return determine_direction(
            self.data[symbol_x].values,
            self.data[symbol_y].values,
            self.data.index, self.data.index,
            symbol_x, symbol_y, start_date
        )
    
    def test_pair_cointegration(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """协整检验接口（别名）"""
        return self.engle_granger_test(x, y)
    
    def calculate_beta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Beta系数估计接口"""
        params = estimate_parameters(x, y)
        return params.get('beta', np.nan)
    
    def estimate_parameters(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """参数估计接口"""
        return estimate_parameters(x, y)
    
    def calculate_halflife(self, residuals: np.ndarray) -> float:
        """半衰期计算接口"""
        return calculate_halflife(residuals)
    
    def residual_statistics(self, residuals: np.ndarray) -> Dict:
        """残差统计接口"""
        return residual_statistics(residuals)
    
    def determine_symbols(self, symbol1: str, symbol2: str,
                         vol_start_date: Optional[str] = None,
                         vol_end_date: Optional[str] = None) -> Tuple[str, str]:
        """
        基于波动率确定品种角色
        
        实现: REQ-2.2.4至REQ-2.2.7
        
        Args:
            symbol1: 品种1代码
            symbol2: 品种2代码
            vol_start_date: 波动率计算起始日期（可选）
            vol_end_date: 波动率计算结束日期（可选）
            
        Returns:
            (symbol_x, symbol_y): 低波动品种作为X，高波动品种作为Y
        """
        try:
            # 获取两个品种的对数价格数据
            data1 = self.data[symbol1].values
            data2 = self.data[symbol2].values
            
            # 计算波动率
            vol1 = self.calculate_volatility(data1, vol_start_date, vol_end_date)
            vol2 = self.calculate_volatility(data2, vol_start_date, vol_end_date)
            
            # 如果波动率计算失败，使用字母顺序
            if np.isnan(vol1) or np.isnan(vol2):
                logger.warning(f"波动率计算失败，使用字母顺序: {symbol1}, {symbol2}")
                if symbol1 <= symbol2:
                    return symbol1, symbol2
                else:
                    return symbol2, symbol1
            
            # 低波动作为X，高波动作为Y
            if vol1 < vol2:
                return symbol1, symbol2  # symbol1是X，symbol2是Y
            elif vol1 > vol2:
                return symbol2, symbol1  # symbol2是X，symbol1是Y
            else:
                # 波动率相等时按字母顺序
                if symbol1 <= symbol2:
                    return symbol1, symbol2
                else:
                    return symbol2, symbol1
                    
        except Exception as e:
            logger.error(f"品种角色确定失败: {e}")
            # 回退到字母顺序
            if symbol1 <= symbol2:
                return symbol1, symbol2
            else:
                return symbol2, symbol1
    
    def screen_all_pairs(self,
                        screening_windows: Optional[List[str]] = None,
                        p_thresholds: Optional[Dict[str, float]] = None,
                        filter_logic: str = 'AND',
                        sort_by: str = 'pvalue_1y',
                        ascending: bool = True,
                        vol_start_date: Optional[str] = None,
                        vol_end_date: Optional[str] = None,
                        windows: Optional[Dict[str, int]] = None,
                        # 向后兼容参数
                        p_threshold: Optional[float] = None,
                        halflife_min: Optional[float] = None,
                        halflife_max: Optional[float] = None,
                        use_halflife_filter: bool = False,
                        volatility_start_date: Optional[str] = None) -> pd.DataFrame:
        """
        批量筛选所有可能的配对
        
        实现: REQ-2.4.1至REQ-2.4.9
        
        Args:
            screening_windows: 筛选用的时间窗口列表（可选，默认['5y'])
            p_thresholds: 各窗口的p值阈值字典（可选，默认{'5y': 0.05}）
            filter_logic: 筛选逻辑，'AND'或'OR'（默认'AND'）
            sort_by: 排序字段（默认'pvalue_1y'）
            ascending: 排序方向（默认True升序）
            vol_start_date: 波动率计算起始日期（可选）
            vol_end_date: 波动率计算结束日期（可选）
            windows: 自定义时间窗口字典（可选）
            
        Returns:
            配对结果DataFrame
        """
        # 向后兼容处理
        if p_threshold is not None:
            # 使用旧参数格式
            screening_windows = ['5y', '1y']
            p_thresholds = {'5y': p_threshold, '1y': p_threshold}
            if volatility_start_date is not None:
                vol_start_date = volatility_start_date
        
        # 默认参数设置
        if screening_windows is None:
            screening_windows = ['5y']
        if p_thresholds is None:
            p_thresholds = {'5y': 0.05}
        
        logger.info(f"开始批量分析{self.n_symbols}个品种的所有配对...")
        logger.info(f"筛选窗口: {screening_windows}")
        logger.info(f"p值阈值: {p_thresholds}")
        logger.info(f"筛选逻辑: {filter_logic}")
        
        # REQ-2.4.1: 生成所有可能配对 C(n,2)
        all_pairs = list(combinations(self.symbols, 2))
        total_pairs = len(all_pairs)
        
        logger.info(f"总配对数: {total_pairs}")
        
        results = []
        
        for i, (symbol1, symbol2) in enumerate(all_pairs, 1):
            if i % 10 == 1:
                logger.info(f"分析进度: {i}/{total_pairs}")
            
            try:
                # 品种角色确定
                symbol_x, symbol_y = self.determine_symbols(
                    symbol1, symbol2, vol_start_date=vol_start_date, vol_end_date=vol_end_date
                )
                
                # 获取最终的X和Y序列
                x_final = self.data[symbol_x].values
                y_final = self.data[symbol_y].values
                
                # 多窗口协整检验
                multi_results = multi_window_test(x_final, y_final, windows)
                
                # 构建结果记录
                pair_result = {
                    'pair': f"{symbol_x}-{symbol_y}",
                    'symbol_x': symbol_x,
                    'symbol_y': symbol_y,
                }
                
                # 添加各窗口的p值和β值
                for window_name, window_result in multi_results.items():
                    if window_result is not None:
                        pair_result[f'pvalue_{window_name}'] = window_result['pvalue']
                        # REQ-2.3.1: β系数精确到6位小数
                        pair_result[f'beta_{window_name}'] = round(window_result['beta'], 6)
                        
                        # REQ-2.3.2: 计算半衰期
                        halflife = calculate_halflife(window_result['residuals'])
                        pair_result[f'halflife_{window_name}'] = halflife
                    else:
                        pair_result[f'pvalue_{window_name}'] = np.nan
                        pair_result[f'beta_{window_name}'] = np.nan
                        pair_result[f'halflife_{window_name}'] = np.nan
                
                # 计算波动率信息  
                vol_x = self.calculate_volatility(
                    self.data[symbol_x].values, vol_start_date, vol_end_date
                )
                vol_y = self.calculate_volatility(
                    self.data[symbol_y].values, vol_start_date, vol_end_date
                )
                
                # 记录波动率计算的时间段
                if vol_start_date and vol_end_date:
                    vol_period = f'{vol_start_date} to {vol_end_date}'
                elif vol_start_date:
                    vol_period = f'{vol_start_date} to latest'
                elif vol_end_date:
                    vol_period = f'earliest to {vol_end_date}'
                else:
                    vol_period = 'all available data'
                    
                pair_result.update({
                    'volatility_x': vol_x,
                    'volatility_y': vol_y,
                    'volatility_period': vol_period
                })
                
                # 添加最长窗口的详细统计 (如果存在)
                longest_window = None
                longest_days = 0
                for window_name, window_result in multi_results.items():
                    if window_result is not None:
                        # 获取窗口的天数
                        window_days = get_window_days(window_name, windows)
                        if window_days > longest_days:
                            longest_days = window_days
                            longest_window = window_name
                
                if longest_window is not None:
                    pair_result[f'r_squared_{longest_window}'] = multi_results[longest_window]['r_squared']
                    
                    # 残差统计
                    resid_stats = residual_statistics(multi_results[longest_window]['residuals'])
                    pair_result.update(resid_stats)
                
                results.append(pair_result)
                
            except Exception as e:
                logger.error(f"配对 {symbol1}-{symbol2} 分析失败: {str(e)}")
                continue
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            logger.warning("没有成功分析的配对")
            return results_df
        
        # REQ-2.4.3至2.4.5: 灵活筛选条件
        # 构建筛选条件
        filter_masks = []
        for window in screening_windows:
            pvalue_col = f'pvalue_{window}'
            if pvalue_col in results_df.columns and window in p_thresholds:
                threshold = p_thresholds[window]
                mask = (results_df[pvalue_col] < threshold) & (results_df[pvalue_col].notna())
                filter_masks.append(mask)
                logger.info(f"窗口 {window}: p值阈值 {threshold}")
        
        if filter_masks:
            # 应用筛选逻辑
            if filter_logic == 'AND':
                final_mask = filter_masks[0]
                for mask in filter_masks[1:]:
                    final_mask &= mask
            else:  # OR
                final_mask = filter_masks[0]
                for mask in filter_masks[1:]:
                    final_mask |= mask
                    
            filtered_df = results_df[final_mask].copy()
            logger.info(f"筛选结果: {len(filtered_df)}/{len(results_df)} 个配对通过筛选 ({filter_logic}逻辑)")
        else:
            # 没有筛选条件，返回所有结果
            filtered_df = results_df.copy()
            logger.info("无筛选条件，返回所有结果")
        
        # REQ-2.4.7: 排序
        if len(filtered_df) > 0 and sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending, na_position='last').reset_index(drop=True)
            logger.info(f"按 {sort_by} {'升序' if ascending else '降序'} 排序")
        
        return filtered_df
    
    def screen_all_pairs_legacy(self, 
                               p_threshold: float = 0.05,
                               halflife_min: Optional[float] = None,
                               halflife_max: Optional[float] = None,
                               use_halflife_filter: bool = False,
                               volatility_start_date: Optional[str] = None) -> pd.DataFrame:
        """
        向后兼容的筛选方法
        
        将旧接口参数转换为新接口调用
        """
        # 转换为新接口参数
        screening_windows = ['5y', '1y']
        p_thresholds = {'5y': p_threshold, '1y': p_threshold}
        
        return self.screen_all_pairs(
            screening_windows=screening_windows,
            p_thresholds=p_thresholds,
            filter_logic='AND',
            sort_by='pvalue_1y',
            ascending=True,
            vol_start_date=volatility_start_date,
            vol_end_date=None
        )
    
    def get_top_pairs(self, n: int = 20, **kwargs) -> pd.DataFrame:
        """
        获取前N个最佳配对
        
        Args:
            n: 返回的配对数量
            **kwargs: 传递给screen_all_pairs的参数
        """
        # 检查是否使用了旧的参数格式
        if any(key in kwargs for key in ['p_threshold', 'halflife_min', 'halflife_max', 'use_halflife_filter', 'volatility_start_date']):
            # 使用向后兼容方法
            all_results = self.screen_all_pairs_legacy(**kwargs)
        else:
            # 使用新接口，默认返回所有结果
            if 'p_thresholds' not in kwargs:
                kwargs['p_thresholds'] = {'5y': 1.0}  # 宽松阈值
            all_results = self.screen_all_pairs(**kwargs)
        return all_results.head(n)
    
    def export_results(self, filepath: str) -> None:
        """导出分析结果"""
        results = self.screen_all_pairs()
        export_results(results, filepath)