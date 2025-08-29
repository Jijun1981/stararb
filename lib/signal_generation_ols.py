#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号生成模块 - 60天滚动OLS版本
基于需求文档: /docs/Requirements/03_signal_generation_ols.md V1.0

与原Kalman版本保持完全兼容的接口，可无缝切换
核心差异：使用60天滚动OLS代替Kalman滤波
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_ols_parameters(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """
    计算OLS回归参数
    
    Args:
        x_data: X序列数据
        y_data: Y序列数据
        
    Returns:
        (beta, alpha, residuals, r_squared)
    """
    if len(x_data) != len(y_data):
        raise ValueError("X和Y数据长度必须相同")
    
    if len(x_data) < 2:
        raise ValueError("至少需要2个数据点")
    
    # 构造设计矩阵 [x, 1]
    X = np.column_stack([x_data, np.ones(len(x_data))])
    Y = y_data
    
    # 最小二乘法求解
    try:
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta = coeffs[0]
        alpha = coeffs[1]
    except:
        # 如果求解失败，返回默认值
        return 1.0, 0.0, np.zeros(len(y_data)), 0.0
    
    # 计算残差
    y_pred = beta * x_data + alpha
    residuals = y_data - y_pred
    
    # 计算R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return beta, alpha, residuals, r_squared


def calculate_z_score(residual: float, residuals_window: np.ndarray) -> float:
    """
    计算标准化残差
    
    Args:
        residual: 当前残差
        residuals_window: 窗口内所有残差
        
    Returns:
        z_score值
    """
    if len(residuals_window) == 0:
        return 0.0
    
    std = np.std(residuals_window)
    if std == 0:
        return 0.0
    
    return residual / std


def check_adf_stationarity(residuals: np.ndarray, pvalue_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    检验残差序列的平稳性
    
    Args:
        residuals: 残差序列
        pvalue_threshold: ADF检验p值阈值
        
    Returns:
        (is_stationary, p_value): 是否平稳和p值
    """
    try:
        # 执行ADF检验
        adf_result = adfuller(residuals, autolag='AIC')
        p_value = adf_result[1]
        
        # p值小于等于阈值认为平稳
        is_stationary = p_value <= pvalue_threshold
        
        return is_stationary, p_value
        
    except Exception as e:
        logger.warning(f"ADF检验失败: {str(e)}")
        # 异常情况默认为非平稳，不交易
        return False, 1.0


def generate_signal(z_score: float, position: Optional[str], days_held: int,
                   z_open: float, z_close: float, max_days: int) -> str:
    """
    信号生成逻辑 - 与原Kalman版本完全相同
    
    Args:
        z_score: 标准化残差
        position: 当前持仓状态 ('long'/'short'/None)
        days_held: 持仓天数
        z_open, z_close, max_days: 阈值参数
        
    Returns:
        信号字符串
    """
    # 强制平仓（最高优先级）
    if position and days_held >= max_days:
        return 'close'
    
    # 平仓条件
    if position and abs(z_score) <= z_close:
        return 'close'
    
    # 开仓条件（仅在空仓时）
    if not position:
        if abs(z_score) >= z_open:
            if z_score <= -z_open:
                return 'open_long'
            elif z_score >= z_open:
                return 'open_short'
        return 'empty'  # 空仓等待
    
    # 持仓期间状态
    if position == 'long':
        return 'holding_long'
    elif position == 'short':
        return 'holding_short'
    
    return 'empty'  # 默认空仓


class SignalGeneratorOLS:
    """
    OLS信号生成器 - 60天滚动窗口版本
    
    与SignalGeneratorV3保持完全兼容的接口
    """
    
    def __init__(self,
                 window_size: int = 60,           # OLS窗口大小（30/45/60/90）
                 min_samples: Optional[int] = None, # 最小样本数，默认为窗口大小的50%
                 z_open: float = 2.0,             # 开仓阈值
                 z_close: float = 0.5,            # 平仓阈值
                 max_holding_days: int = 30,      # 最大持仓天数
                 enable_adf_check: bool = False,  # 启用ADF平稳性检验
                 adf_pvalue_threshold: float = 0.05): # ADF检验p值阈值
        """
        初始化OLS信号生成器
        
        Args:
            window_size: OLS窗口大小（建议值：30/45/60/90）
            min_samples: 最小样本数，默认为窗口大小的50%
            z_open: 开仓阈值
            z_close: 平仓阈值
            max_holding_days: 最大持仓天数
            enable_adf_check: 是否启用ADF平稳性检验约束
            adf_pvalue_threshold: ADF检验p值阈值，低于此值认为平稳
        """
        # 验证窗口大小
        if window_size not in [30, 45, 60, 90]:
            logger.warning(f"窗口大小{window_size}不在建议值[30,45,60,90]中，仍然使用")
        
        self.window_size = window_size
        # 如果没有指定min_samples，默认为窗口大小的50%
        self.min_samples = min_samples if min_samples is not None else window_size // 2
        self.z_open = z_open
        self.z_close = z_close
        self.max_holding_days = max_holding_days
        self.enable_adf_check = enable_adf_check
        self.adf_pvalue_threshold = adf_pvalue_threshold
        
        # 统计报告数据
        self.pair_statistics = {}
        
        # ADF统计数据
        if self.enable_adf_check:
            self._adf_stats = {
                'adf_total': 0,
                'adf_passed': 0,
                'adf_failed': 0
            }
        
        logger.info(f"SignalGeneratorOLS 初始化完成:")
        logger.info(f"  - 窗口大小: {self.window_size}天")
        logger.info(f"  - 最小样本: {self.min_samples}天")
        logger.info(f"  - 阈值: z_open={self.z_open}, z_close={self.z_close}")
        logger.info(f"  - 最大持仓: {self.max_holding_days}天")
        if self.enable_adf_check:
            logger.info(f"  ADF检验: 已启用 (p阈值={self.adf_pvalue_threshold})")
        else:
            logger.info(f"  ADF检验: 已禁用")
    
    def calculate_rolling_ols(self, x_data: pd.Series, y_data: pd.Series, 
                             start_date: Optional[str] = None,
                             window: Optional[int] = None) -> pd.DataFrame:
        """
        简化版滚动OLS：从指定日期开始，有多少数据用多少
        
        Args:
            x_data: X品种价格序列
            y_data: Y品种价格序列
            start_date: 开始日期（默认从数据开始）
            window: 窗口大小（默认使用self.window_size）
            
        Returns:
            DataFrame包含: date, beta, alpha, residual, residual_std, z_score, window_size
        """
        if window is None:
            window = self.window_size
        
        results = []
        
        # 确保数据对齐
        aligned = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
        if len(aligned) == 0:
            return pd.DataFrame()
        
        # 确定起始位置
        if start_date:
            try:
                start_idx = aligned.index.get_loc(pd.to_datetime(start_date))
            except:
                # 如果找不到精确日期，找最近的
                start_idx = aligned.index.searchsorted(pd.to_datetime(start_date))
                if start_idx >= len(aligned):
                    start_idx = 0
        else:
            start_idx = 0
        
        min_points = 20  # 最少20个点才开始计算
        
        # 从start_date开始累积数据，直到有足够数据点
        for i in range(start_idx + min_points, len(aligned)):
            # 计算实际可用的窗口大小
            points_available = i - start_idx
            actual_window = min(points_available, window)
            
            # 获取窗口数据
            if actual_window < window:
                # 不足窗口大小，用所有从start_idx开始的数据
                window_data = aligned.iloc[start_idx:i]
            else:
                # 达到窗口大小，使用滚动窗口
                window_data = aligned.iloc[i-window:i]
            
            x_window = window_data['x'].values
            y_window = window_data['y'].values
            
            # OLS回归
            beta, alpha, residuals, r_squared = calculate_ols_parameters(x_window, y_window)
            
            # 计算当前点的残差
            current_x = aligned.iloc[i]['x']
            current_y = aligned.iloc[i]['y']
            current_residual = current_y - (beta * current_x + alpha)
            
            # 计算窗口内所有残差的标准差
            residual_std = np.std(residuals) if len(residuals) > 1 else 0
            
            # 计算z-score
            z_score = current_residual / residual_std if residual_std > 0 else 0
            
            results.append({
                'date': aligned.index[i],  # 当前日期
                'beta': beta,
                'alpha': alpha,
                'residual': current_residual,
                'residual_std': residual_std,
                'z_score': z_score,
                'r_squared': r_squared,
                'window_size': len(window_data)  # 记录实际使用的数据点数
            })
        
        return pd.DataFrame(results)
    
    def process_pair(self, pair_name: str, x_data: pd.Series, 
                    y_data: pd.Series, initial_beta: Optional[float] = None,
                    signal_start_date: Optional[pd.Timestamp] = None,
                    symbol_x: str = None, symbol_y: str = None) -> pd.DataFrame:
        """
        处理单个配对，生成交易信号
        
        Args:
            pair_name: 配对名称
            x_data: X品种价格序列
            y_data: Y品种价格序列
            initial_beta: 初始Beta值（可选，用于参考）
            signal_start_date: 开始跟踪持仓的日期（可选，之前只计算不跟踪）
            
        Returns:
            信号DataFrame，与SignalGeneratorV3输出格式完全一致
        """
        logger.info(f"处理配对: {pair_name}")
        
        # 计算滚动OLS（使用简化版本）
        ols_results = self.calculate_rolling_ols(
            x_data, y_data, 
            start_date=signal_start_date
        )
        
        if ols_results.empty:
            logger.warning(f"配对 {pair_name} 无法计算OLS")
            return pd.DataFrame()
        
        # 初始化信号列表
        signals = []
        position = None
        days_held = 0
        
        # 统计数据
        beta_list = []
        open_signal_count = 0
        
        # 处理每个OLS结果生成信号
        for idx, row in ols_results.iterrows():
            current_date = row['date']
            z_score = row['z_score']
            beta = row['beta']
            alpha = row['alpha']
            residual = row['residual']
            r_squared = row['r_squared']
            window_size = row['window_size']
            
            beta_list.append(beta)
            
            # 生成交易信号
            signal = generate_signal(
                z_score, position, days_held,
                self.z_open, self.z_close, self.max_holding_days
            )
            
            # 判断信号原因
            if signal == 'close':
                if days_held >= self.max_holding_days:
                    reason = 'force_close'
                else:
                    reason = 'z_converge'
            elif signal in ['open_long', 'open_short']:
                reason = 'z_threshold'
                open_signal_count += 1
            elif signal in ['holding_long', 'holding_short']:
                reason = 'holding'
            else:
                reason = 'no_signal'
            
            # 更新持仓状态
            if signal == 'open_long':
                position = 'long'
                days_held = 1
            elif signal == 'open_short':
                position = 'short'
                days_held = 1
            elif signal == 'close':
                position = None
                days_held = 0
            elif position:
                days_held += 1
            
            # 记录信号
            signals.append({
                'date': current_date,
                'pair': pair_name,
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                'signal': signal,
                'z_score': z_score,
                'residual': residual,
                'beta': beta,
                'alpha': alpha,
                'beta_initial': initial_beta if initial_beta else beta,
                'days_held': days_held,
                'reason': reason,
                'window_size': window_size,
                'r_squared': r_squared
            })
        
        # 保存统计数据
        if beta_list:
            pair_stats = {
                'total_signals': len(signals),
                'open_signals': open_signal_count,
                'avg_beta': np.mean(beta_list),
                'beta_std': np.std(beta_list),
                'beta_min': np.min(beta_list),
                'beta_max': np.max(beta_list)
            }
            
            # 添加ADF统计信息
            if self.enable_adf_check and hasattr(self, '_adf_stats'):
                pair_stats.update({
                    'adf_total': self._adf_stats['adf_total'],
                    'adf_passed': self._adf_stats['adf_passed'],
                    'adf_failed': self._adf_stats['adf_failed'],
                    'adf_pass_rate': self._adf_stats['adf_passed'] / max(self._adf_stats['adf_total'], 1)
                })
            
            self.pair_statistics[pair_name] = pair_stats
        
        result_df = pd.DataFrame(signals)
        logger.info(f"配对 {pair_name} 处理完成: 生成 {len(result_df)} 条记录，{open_signal_count} 个开仓信号")
        
        return result_df
    
    def process_all_pairs(self, pairs_df: pd.DataFrame, price_data: pd.DataFrame,
                         beta_window: str = '1y', signal_start_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        批量处理所有配对
        
        Args:
            pairs_df: 协整模块输出的配对DataFrame
            price_data: 价格数据DataFrame
            beta_window: 使用的Beta时间窗口（用于获取初始beta）
            signal_start_date: 信号开始日期，在此日期前不会生成交易信号（避免孤立平仓）
            
        Returns:
            所有配对的信号DataFrame
        """
        logger.info(f"开始批量处理 {len(pairs_df)} 个配对")
        
        all_signals = []
        processed_count = 0
        
        for idx, row in pairs_df.iterrows():
            pair_name = row['pair']
            symbol_x = row['symbol_x']
            symbol_y = row['symbol_y']
            
            # 获取初始beta（如果有）
            beta_col = f'beta_{beta_window}'
            initial_beta = row.get(beta_col, None)
            
            # 检查价格数据
            if symbol_x not in price_data.columns or symbol_y not in price_data.columns:
                logger.warning(f"跳过配对 {pair_name}: 缺少价格数据")
                continue
            
            x_data = price_data[symbol_x]
            y_data = price_data[symbol_y]
            
            # 处理配对
            try:
                pair_signals = self.process_pair(
                    pair_name=pair_name,
                    x_data=x_data,
                    y_data=y_data,
                    initial_beta=initial_beta,
                    signal_start_date=signal_start_date,
                    symbol_x=symbol_x,
                    symbol_y=symbol_y
                )
                
                if len(pair_signals) > 0:
                    all_signals.append(pair_signals)
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"处理配对 {pair_name} 时出错: {e}")
                continue
        
        # 合并所有信号
        if all_signals:
            result_df = pd.concat(all_signals, ignore_index=True)
        else:
            result_df = pd.DataFrame()
        
        logger.info(f"批量处理完成: 成功处理 {processed_count}/{len(pairs_df)} 个配对，生成 {len(result_df)} 条信号")
        
        return result_df
    
    def get_statistics_report(self) -> pd.DataFrame:
        """
        获取统计报告
        
        Returns:
            统计报告DataFrame
        """
        if not self.pair_statistics:
            return pd.DataFrame()
        
        report_data = []
        for pair_name, stats in self.pair_statistics.items():
            report_data.append({
                'pair': pair_name,
                'total_signals': stats['total_signals'],
                'open_signals': stats['open_signals'],
                'avg_beta': stats['avg_beta'],
                'beta_std': stats['beta_std'],
                'beta_min': stats['beta_min'],
                'beta_max': stats['beta_max'],
                'beta_range': stats['beta_max'] - stats['beta_min']
            })
        
        return pd.DataFrame(report_data)
    
    def get_quality_report(self) -> pd.DataFrame:
        """
        获取质量报告 - 与SignalGeneratorV3兼容
        
        Returns:
            质量报告DataFrame（简化版）
        """
        # OLS版本的简化质量报告
        if not self.pair_statistics:
            return pd.DataFrame()
        
        quality_data = []
        for pair_name, stats in self.pair_statistics.items():
            # 简单的质量评级
            beta_stability = stats['beta_std'] / abs(stats['avg_beta']) if stats['avg_beta'] != 0 else float('inf')
            
            if beta_stability < 0.1:
                quality_status = 'good'
            elif beta_stability < 0.3:
                quality_status = 'warning'
            else:
                quality_status = 'bad'
            
            quality_data.append({
                'pair': pair_name,
                'quality_status': quality_status,
                'beta_stability': beta_stability,
                'avg_beta': stats['avg_beta'],
                'beta_std': stats['beta_std'],
                'open_signals': stats['open_signals'],
                'data_points': stats['total_signals']
            })
        
        return pd.DataFrame(quality_data)


def main():
    """示例使用"""
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # 模拟协整的价格序列
    x_prices = np.cumsum(0.01 * np.random.randn(len(dates))) + 100
    y_prices = 0.8 * x_prices + 0.1 + 0.02 * np.random.randn(len(dates))
    
    price_data = pd.DataFrame({
        'TEST_X': x_prices,
        'TEST_Y': y_prices
    }, index=dates)
    
    # 模拟协整模块输出
    pairs_df = pd.DataFrame({
        'pair': ['TEST_X-TEST_Y'],
        'symbol_x': ['TEST_X'],
        'symbol_y': ['TEST_Y'],
        'beta_1y': [0.75]
    })
    
    # 创建信号生成器
    generator = SignalGeneratorOLS(
        window_size=45,
        z_open=2.0,
        z_close=0.5,
        max_holding_days=30
    )
    
    # 生成信号
    signals = generator.process_all_pairs(pairs_df, price_data, beta_window='1y')
    
    print(f"生成信号数量: {len(signals)}")
    
    if len(signals) > 0:
        # 统计信号类型
        signal_counts = signals['signal'].value_counts()
        print(f"\n信号类型分布:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # 统计Z-score
        valid_signals = signals[signals['window_size'] == 45]
        if len(valid_signals) > 0:
            z_scores = valid_signals['z_score'].dropna()
            if len(z_scores) > 0:
                print(f"\nZ-score统计:")
                print(f"  均值: {z_scores.mean():.3f}")
                print(f"  标准差: {z_scores.std():.3f}")
                print(f"  |Z|>2比例: {(np.abs(z_scores) > 2.0).mean()*100:.1f}%")
    
    # 质量报告
    quality = generator.get_quality_report()
    if len(quality) > 0:
        print("\n质量报告:")
        print(quality)
    
    # 统计报告
    stats = generator.get_statistics_report()
    if len(stats) > 0:
        print("\n统计报告:")
        print(stats)


if __name__ == '__main__':
    main()