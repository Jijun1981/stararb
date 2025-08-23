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


def engle_granger_test(x: np.ndarray, y: np.ndarray, 
                      direction: str = 'y_on_x') -> Dict:
    """
    Engle-Granger两步法协整检验
    
    实现: REQ-2.1.1
    
    Args:
        x: 价格序列1 (自变量候选)
        y: 价格序列2 (因变量候选)  
        direction: 回归方向 'y_on_x' 或 'x_on_y'
        
    Returns:
        {
            'pvalue': float,        # ADF检验p值
            'adf_stat': float,      # ADF统计量
            'beta': float,          # 回归系数β
            'alpha': float,         # 截距α
            'residuals': np.ndarray,# 残差序列
            'direction': str,       # 回归方向
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
        # 第一步: OLS回归
        if direction == 'y_on_x':
            # Y = α + β*X + ε
            X = add_constant(x)
            model = OLS(y, X).fit()
            alpha = model.params[0]
            beta = model.params[1] 
            residuals = model.resid
        elif direction == 'x_on_y':
            # X = α + β*Y + ε  
            Y = add_constant(y)
            model = OLS(x, Y).fit()
            alpha = model.params[0]
            beta = model.params[1]
            residuals = model.resid
        else:
            raise CointegrationError(f"无效的回归方向: {direction}")
        
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
            'direction': direction,
            'r_squared': model.rsquared,
            'n_obs': len(x)
        }
        
    except Exception as e:
        logger.error(f"Engle-Granger检验失败: {str(e)}")
        raise CointegrationError(f"协整检验计算失败: {str(e)}")


def multi_window_test(x: np.ndarray, y: np.ndarray, 
                     direction: Optional[str] = None) -> Dict:
    """
    多时间窗口协整检验
    
    实现: REQ-2.1.2至REQ-2.1.6
    
    Args:
        x: 价格序列1
        y: 价格序列2
        direction: 回归方向，None表示自动判定
        
    Returns:
        {
            '5y': {...},  # 5年窗口结果，None如果数据不足
            '4y': {...},  # 4年窗口结果
            '3y': {...},  # 3年窗口结果  
            '2y': {...},  # 2年窗口结果
            '1y': {...}   # 1年窗口结果
        }
    """
    # 时间窗口定义 (交易日)
    windows = {
        '5y': 1260,  # 5年 = 5×252
        '4y': 1008,  # 4年 = 4×252  
        '3y': 756,   # 3年 = 3×252
        '2y': 504,   # 2年 = 2×252
        '1y': 252    # 1年 = 252
    }
    
    results = {}
    total_length = len(x)
    
    # 自动方向判定
    if direction is None:
        vol_x = np.std(np.diff(x)) if len(x) > 1 else 0
        vol_y = np.std(np.diff(y)) if len(y) > 1 else 0
        direction = 'y_on_x' if vol_y >= vol_x else 'x_on_y'
    
    for window_name, window_size in windows.items():
        if total_length >= window_size:
            try:
                # 使用最新的window_size个数据点
                x_window = x[-window_size:]
                y_window = y[-window_size:]
                
                result = engle_granger_test(x_window, y_window, direction)
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
                        start_date: Optional[str] = None) -> float:
    """
    计算年化波动率
    
    实现: REQ-2.2.1至REQ-2.2.3, REQ-2.2.7
    
    Args:
        log_prices: 对数价格序列
        dates: 对应的日期索引
        start_date: 波动率计算起始日期（默认为最近1年）
        
    Returns:
        年化波动率
    """
    try:
        # REQ-2.2.7: 默认使用最近1年数据
        if start_date is None:
            # 计算最近1年的起始日期
            latest_date = dates[-1] if len(dates) > 0 else pd.Timestamp.now()
            start_dt = latest_date - pd.Timedelta(days=365)
        else:
            start_dt = pd.to_datetime(start_date)
        mask = dates >= start_dt
        
        if not np.any(mask):
            logger.warning(f"没有找到{start_date}之后的数据，使用全部数据")
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
                       start_date: Optional[str] = None) -> Tuple[str, str, str]:
    """
    基于波动率确定回归方向
    
    实现: REQ-2.2.4至REQ-2.2.5, REQ-2.2.7
    
    Args:
        series_1: 品种1的对数价格
        series_2: 品种2的对数价格
        dates_1: 品种1的日期
        dates_2: 品种2的日期  
        symbol_1: 品种1代码
        symbol_2: 品种2代码
        start_date: 计算波动率的起始日期（默认为最近1年）
        
    Returns:
        (direction, symbol_x, symbol_y)
        direction: 'y_on_x' 或 'x_on_y'
        symbol_x: 自变量品种 (低波动率)
        symbol_y: 因变量品种 (高波动率)
    """
    try:
        # 计算两个品种的波动率
        vol_1 = calculate_volatility(series_1, dates_1, start_date)
        vol_2 = calculate_volatility(series_2, dates_2, start_date)
        
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
    
    def multi_window_test(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """多窗口协整检验接口"""
        return multi_window_test(x, y)
    
    def adf_test(self, series: np.ndarray) -> Tuple[float, float]:
        """ADF检验接口"""
        return adf_test(series)
    
    def calculate_volatility(self, log_prices: np.ndarray, 
                           start_date: Optional[str] = None) -> float:
        """波动率计算接口"""
        return calculate_volatility(log_prices, self.data.index, start_date)
    
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
    
    def estimate_parameters(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """参数估计接口"""
        return estimate_parameters(x, y)
    
    def calculate_halflife(self, residuals: np.ndarray) -> float:
        """半衰期计算接口"""
        return calculate_halflife(residuals)
    
    def residual_statistics(self, residuals: np.ndarray) -> Dict:
        """残差统计接口"""
        return residual_statistics(residuals)
    
    def screen_all_pairs(self, 
                        p_threshold: float = 0.05,
                        halflife_min: Optional[float] = None,
                        halflife_max: Optional[float] = None,
                        use_halflife_filter: bool = False,
                        volatility_start_date: Optional[str] = None) -> pd.DataFrame:
        """
        批量筛选所有可能的配对
        
        实现: REQ-2.4.1至REQ-2.4.8
        
        Args:
            p_threshold: p值筛选阈值（默认0.05）
            halflife_min: 最小半衰期阈值（可选）
            halflife_max: 最大半衰期阈值（可选）
            use_halflife_filter: 是否启用半衰期筛选（默认False）
            volatility_start_date: 波动率计算起始日期（默认为最近1年）
            
        Returns:
            配对结果DataFrame，按1年p值升序排序
        """
        logger.info(f"开始批量分析{self.n_symbols}个品种的所有配对...")
        
        # REQ-2.4.1: 生成所有可能配对 C(n,2)
        all_pairs = list(combinations(self.symbols, 2))
        total_pairs = len(all_pairs)
        
        logger.info(f"总配对数: {total_pairs}")
        
        results = []
        
        for i, (symbol1, symbol2) in enumerate(all_pairs, 1):
            if i % 10 == 1:
                logger.info(f"分析进度: {i}/{total_pairs}")
            
            try:
                # 获取数据
                x_data = self.data[symbol1].values
                y_data = self.data[symbol2].values
                
                # 方向判定 (基于最近数据波动率)
                direction, symbol_x, symbol_y = self.determine_direction(
                    symbol1, symbol2, use_recent=True, recent_start=volatility_start_date
                )
                
                # 确定最终的X和Y序列
                if symbol_x == symbol1:
                    x_final = x_data
                    y_final = y_data
                else:
                    x_final = y_data  # 交换
                    y_final = x_data
                
                # 多窗口协整检验
                multi_results = multi_window_test(x_final, y_final, direction)
                
                # 构建结果记录
                pair_result = {
                    'pair': f"{symbol_x}-{symbol_y}",
                    'symbol_x': symbol_x,
                    'symbol_y': symbol_y,
                    'direction': direction,
                }
                
                # 添加各窗口的p值和β值
                windows = ['5y', '4y', '3y', '2y', '1y']
                for window in windows:
                    if multi_results[window] is not None:
                        pair_result[f'pvalue_{window}'] = multi_results[window]['pvalue']
                        # REQ-2.3.1: β系数精确到6位小数
                        pair_result[f'beta_{window}'] = round(multi_results[window]['beta'], 6)
                        
                        # REQ-2.3.2: 计算所有窗口的半衰期（API要求）
                        halflife = calculate_halflife(multi_results[window]['residuals'])
                        pair_result[f'halflife_{window}'] = halflife
                    else:
                        pair_result[f'pvalue_{window}'] = np.nan
                        pair_result[f'beta_{window}'] = np.nan
                        pair_result[f'halflife_{window}'] = np.nan
                
                # 计算波动率信息  
                vol_x = self.calculate_volatility(
                    self.data[symbol_x].values, start_date=volatility_start_date
                )
                vol_y = self.calculate_volatility(
                    self.data[symbol_y].values, start_date=volatility_start_date
                )
                
                # 记录波动率计算的时间段
                if volatility_start_date:
                    vol_period = f'{volatility_start_date} to latest'
                else:
                    vol_period = 'recent 1 year'
                    
                pair_result.update({
                    'volatility_x': vol_x,
                    'volatility_y': vol_y,
                    'volatility_period': vol_period
                })
                
                # 添加5年窗口的详细统计 (如果存在)
                if multi_results['5y'] is not None:
                    pair_result['r_squared_5y'] = multi_results['5y']['r_squared']
                    
                    # 残差统计
                    resid_stats = residual_statistics(multi_results['5y']['residuals'])
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
        
        # REQ-2.4.3: 筛选条件：5年p值 < 阈值 且 1年p值 < 阈值
        if 'pvalue_5y' in results_df.columns and 'pvalue_1y' in results_df.columns:
            # p值筛选
            significant_mask = (results_df['pvalue_5y'] < p_threshold) & \
                             (results_df['pvalue_1y'] < p_threshold)
            
            # REQ-2.4.7-2.4.8: 可选的半衰期筛选
            if use_halflife_filter and ('halflife_5y' in results_df.columns):
                logger.info(f"启用半衰期筛选: [{halflife_min}, {halflife_max}]")
                halflife_mask = pd.Series([True] * len(results_df))
                
                if halflife_min is not None:
                    halflife_mask &= (results_df['halflife_5y'] >= halflife_min)
                    halflife_mask &= results_df['halflife_5y'].notna()
                
                if halflife_max is not None:
                    halflife_mask &= (results_df['halflife_5y'] <= halflife_max)
                    halflife_mask &= results_df['halflife_5y'].notna()
                
                # 组合筛选条件
                significant_mask &= halflife_mask
                
                filtered_df = results_df[significant_mask].copy()
                logger.info(f"筛选结果: {len(filtered_df)}/{len(results_df)} 个配对通过p值和半衰期筛选")
            else:
                filtered_df = results_df[significant_mask].copy()
                logger.info(f"筛选结果: {len(filtered_df)}/{len(results_df)} 个配对通过p值筛选 (5年且1年 < {p_threshold})")
            
            if len(filtered_df) > 0:
                # REQ-2.4.5: 按1年p值升序排序（近期协整性优先）
                filtered_df = filtered_df.sort_values('pvalue_1y').reset_index(drop=True)
                return filtered_df
        
        # 如果没有5年数据或筛选结果为空，返回所有结果
        logger.info("返回所有分析结果")
        return results_df.sort_values('pvalue_1y', na_position='last').reset_index(drop=True)
    
    def get_top_pairs(self, n: int = 20, **kwargs) -> pd.DataFrame:
        """
        获取前N个最佳配对
        
        Args:
            n: 返回的配对数量
            **kwargs: 传递给screen_all_pairs的参数
        """
        # 默认使用宽松的p值，获取所有配对
        if 'p_threshold' not in kwargs:
            kwargs['p_threshold'] = 1.0
        
        all_results = self.screen_all_pairs(**kwargs)
        return all_results.head(n)
    
    def export_results(self, filepath: str) -> None:
        """导出分析结果"""
        results = self.screen_all_pairs()
        export_results(results, filepath)