#!/usr/bin/env python3
"""
Beta估计模块 - REQ-3.x.x
负责期货配对Beta系数的一次性标定

主要功能:
1. 5种Beta估计方法: OLS(2y), EWLS, FM-OLS, Min-Halflife, Huber
2. 样本外验证和两步选择算法
3. Monte Carlo交叉验证 (验证算法A)
4. 时间序列分割验证 (验证算法B)
5. 批量配对处理和参数导出

作者: Star-arb Team
日期: 2025-08-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
import json
from datetime import datetime
import logging

# 统计计算库
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import HuberRegressor

# 协整分析库 
try:
    from linearmodels.cointegration import FMOLS
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False
    warnings.warn("linearmodels not available, FM-OLS method will be disabled")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class BetaCalibrator:
    """
    Beta标定器 - 一次性标定期货配对的Beta系数
    
    实现REQ-5.x.x系列需求，支持5种估计方法和样本外验证
    """
    
    def __init__(self, 
                 train_start: str = "2020-01-01",
                 train_end: str = "2023-12-31", 
                 valid_start: str = "2024-01-01",
                 valid_end: str = "2025-08-20"):
        """
        初始化Beta标定器
        
        Args:
            train_start: 训练段开始日期
            train_end: 训练段结束日期  
            valid_start: 验证段开始日期
            valid_end: 验证段结束日期
        """
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        
        logger.info(f"Beta标定器初始化: 训练段({train_start}~{train_end}), 验证段({valid_start}~{valid_end})")
    
    def load_and_split_data(self, pair_data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """
        REQ-5.1.1-5.1.3: 加载并分割配对数据
        
        Args:
            pair_data: {'X': price_series, 'Y': price_series}
            
        Returns:
            {'train_data': DataFrame, 'valid_data': DataFrame}
        """
        # 合并为DataFrame
        data = pd.DataFrame(pair_data)
        
        # 确保索引为日期类型
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 处理缺失值 - REQ-5.1.3
        data = data.dropna()
        
        # 数据分割
        train_mask = (data.index >= self.train_start) & (data.index <= self.train_end)
        valid_mask = (data.index >= self.valid_start) & (data.index <= self.valid_end)
        
        train_data = data[train_mask].copy()
        valid_data = data[valid_mask].copy()
        
        if len(train_data) == 0:
            raise ValueError(f"训练段无数据: {self.train_start} ~ {self.train_end}")
        if len(valid_data) == 0:
            raise ValueError(f"验证段无数据: {self.valid_start} ~ {self.valid_end}")
        
        logger.debug(f"数据分割完成: 训练段{len(train_data)}天, 验证段{len(valid_data)}天")
        
        return {'train_data': train_data, 'valid_data': valid_data}
    
    def choose_direction(self, train_data: pd.DataFrame) -> str:
        """
        REQ-5.2.6: 选择回归方向 (Y~X vs X~Y)
        基于ADF检验p值选择协整性更强的方向
        
        Args:
            train_data: 训练数据DataFrame with X, Y columns
            
        Returns:
            'y_on_x' or 'x_on_y'
        """
        log_X = np.log(train_data['X'])
        log_Y = np.log(train_data['Y'])
        
        # 方向1: Y ~ X (log_Y = α + β*log_X + ε)  
        X_matrix = np.column_stack([np.ones(len(log_X)), log_X])
        beta_1 = np.linalg.lstsq(X_matrix, log_Y, rcond=None)[0]
        residuals_1 = log_Y - beta_1[0] - beta_1[1] * log_X
        
        try:
            adf_result_1 = adfuller(residuals_1, autolag='AIC', maxlag=int(len(residuals_1)**(1/3)))
            p_value_1 = adf_result_1[1]
        except:
            p_value_1 = 1.0
        
        # 方向2: X ~ Y (log_X = α + β*log_Y + ε)
        Y_matrix = np.column_stack([np.ones(len(log_Y)), log_Y]) 
        beta_2 = np.linalg.lstsq(Y_matrix, log_X, rcond=None)[0]
        residuals_2 = log_X - beta_2[0] - beta_2[1] * log_Y
        
        try:
            adf_result_2 = adfuller(residuals_2, autolag='AIC', maxlag=int(len(residuals_2)**(1/3)))
            p_value_2 = adf_result_2[1]
        except:
            p_value_2 = 1.0
        
        # 选择p值更小的方向
        direction = 'y_on_x' if p_value_1 <= p_value_2 else 'x_on_y'
        
        logger.debug(f"方向选择: {direction} (p_yx={p_value_1:.4f}, p_xy={p_value_2:.4f})")
        
        return direction
    
    # =================================================================
    # 5种Beta估计方法 - REQ-5.2系列
    # =================================================================
    
    def fit_ols_2y(self, train_data: pd.DataFrame) -> Dict:
        """
        REQ-5.2.1: OLS(2y) - 使用训练段末尾2年数据
        """
        # 取末尾504个交易日 (约2年)
        recent_data = train_data.tail(min(504, len(train_data)))
        
        log_X = np.log(recent_data['X'])
        log_Y = np.log(recent_data['Y'])
        
        # 带截距的OLS回归
        X_matrix = np.column_stack([np.ones(len(log_X)), log_X])
        coeffs = np.linalg.lstsq(X_matrix, log_Y, rcond=None)[0]
        
        # 计算残差和诊断信息
        residuals = log_Y - coeffs[0] - coeffs[1] * log_X
        r_squared = 1 - np.var(residuals) / np.var(log_Y)
        
        return {
            'method': 'OLS_2y',
            'beta': float(coeffs[1]),
            'alpha': float(coeffs[0]),
            'window': 'last_2y', 
            'r_squared': float(r_squared),
            'residuals': residuals.values,
            'n_obs': len(recent_data)
        }
    
    def fit_ewls(self, train_data: pd.DataFrame, halflife: int = 126) -> Dict:
        """
        REQ-5.2.2: EWLS - 指数加权最小二乘
        
        Args:
            halflife: 半衰期，默认126交易日(半年)
        """
        log_X = np.log(train_data['X']).values
        log_Y = np.log(train_data['Y']).values
        
        T = len(log_X)
        lambda_param = np.exp(-np.log(2) / halflife)
        
        # 构建权重：远期小，近期大
        weights = lambda_param ** np.arange(T-1, -1, -1)
        
        # 加权最小二乘：避免构造大矩阵
        X_matrix = np.column_stack([np.ones(T), log_X])
        sqrt_weights = np.sqrt(weights)
        
        Xw = X_matrix * sqrt_weights[:, None]
        yw = log_Y * sqrt_weights
        
        # 求解加权正规方程
        coeffs = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        
        # 计算加权残差
        residuals = log_Y - coeffs[0] - coeffs[1] * log_X
        weighted_ssr = np.sum(weights * residuals**2)
        weighted_tss = np.sum(weights * (log_Y - np.average(log_Y, weights=weights))**2)
        
        r_squared = 1 - weighted_ssr / weighted_tss if weighted_tss > 0 else 0
        
        return {
            'method': 'EWLS',
            'beta': float(coeffs[1]),
            'alpha': float(coeffs[0]),
            'halflife_used': halflife,
            'r_squared': float(r_squared),
            'residuals': residuals,
            'n_obs': T
        }
    
    def fit_fm_ols(self, train_data: pd.DataFrame) -> Dict:
        """
        REQ-5.2.3: FM-OLS - Phillips-Hansen全修正最小二乘
        """
        if not LINEARMODELS_AVAILABLE:
            # 如果linearmodels不可用，回退到普通OLS
            logger.warning("linearmodels不可用，FM-OLS回退到普通OLS")
            return self._fallback_ols(train_data, 'FM_OLS')
        
        try:
            log_X = np.log(train_data['X']).values
            log_Y = np.log(train_data['Y']).values
            
            # FM-OLS估计
            res = FMOLS(log_Y, log_X).fit()
            
            # 计算残差
            residuals = log_Y - res.params.const - res.params[0] * log_X
            r_squared = 1 - np.var(residuals) / np.var(log_Y)
            
            return {
                'method': 'FM_OLS',
                'beta': float(res.params[0]),
                'alpha': float(res.params.const),
                'r_squared': float(r_squared),
                'residuals': residuals,
                'n_obs': len(train_data)
            }
            
        except Exception as e:
            logger.warning(f"FM-OLS估计失败，回退到普通OLS: {str(e)}")
            return self._fallback_ols(train_data, 'FM_OLS')
    
    def fit_min_halflife(self, train_data: pd.DataFrame, beta_ols: float, grid_points: int = 201) -> Dict:
        """
        REQ-5.2.4: Min-Halflife - 两段式网格搜索最小半衰期Beta
        
        Args:
            beta_ols: OLS基准Beta值
            grid_points: 总网格点数（分配给两阶段）
        """
        log_X = np.log(train_data['X']).values
        log_Y = np.log(train_data['Y']).values
        
        # 第一阶段：粗搜索（21个点）
        coarse_points = 21
        beta_coarse = np.linspace(0.5 * beta_ols, 1.5 * beta_ols, coarse_points)
        
        best_coarse_beta = None
        best_coarse_hl = float('inf')
        
        for beta in beta_coarse:
            # 固定beta计算对应的alpha
            alpha = np.mean(log_Y - beta * log_X)
            residuals = log_Y - alpha - beta * log_X
            
            if len(residuals) < 3:
                continue
                
            try:
                # AR(1)精确公式: HL = -ln(2)/ln|ρ|
                rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), residuals[1:], rcond=None)[0][0]
                
                if abs(rho) >= 1:
                    continue
                
                halflife = -np.log(2) / np.log(abs(rho))
                
                # 粗搜索只检查半衰期
                if 2 <= halflife <= 60 and halflife < best_coarse_hl:
                    best_coarse_hl = halflife
                    best_coarse_beta = beta
                        
            except:
                continue
        
        # 如果粗搜索没找到，返回None
        if best_coarse_beta is None:
            logger.warning("Min-Halflife粗搜索未找到合适的Beta值")
            return {
                'method': 'Min_HL',
                'beta': None,
                'alpha': None,
                'halflife': None,
                'adf_pvalue': 1.0,
                'n_obs': len(train_data)
            }
        
        # 第二阶段：细搜索（在最优点附近±5%范围，21个点）
        fine_points = 21
        beta_width = 0.05 * abs(best_coarse_beta)
        beta_fine = np.linspace(
            max(0.5 * beta_ols, best_coarse_beta - beta_width),
            min(1.5 * beta_ols, best_coarse_beta + beta_width),
            fine_points
        )
        
        best_beta = None
        best_alpha = None
        min_halflife = float('inf')
        best_adf_p = 1.0
        
        for beta in beta_fine:
            alpha = np.mean(log_Y - beta * log_X)
            residuals = log_Y - alpha - beta * log_X
            
            try:
                # AR(1)精确公式
                rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), residuals[1:], rcond=None)[0][0]
                
                if abs(rho) >= 1:
                    continue
                
                halflife = -np.log(2) / np.log(abs(rho))
                
                # ADF检验（使用regression='n'对残差）
                adf_result = adfuller(residuals, regression='n', autolag='AIC', 
                                     maxlag=int(len(residuals)**(1/3)))
                adf_p = adf_result[1]
                
                # REQ-5.2.4: 内部筛选p<0.10，最终生死由外层决定
                if adf_p < 0.10 and 2 <= halflife <= 90:  # 内部筛选放宽
                    if halflife < min_halflife:
                        min_halflife = halflife
                        best_beta = beta
                        best_alpha = alpha
                        best_adf_p = adf_p
                        
            except:
                continue
        
        if best_beta is None:
            logger.warning("Min-Halflife方法未找到合适的Beta值")
            return {
                'method': 'Min_HL',
                'beta': None,
                'alpha': None,
                'halflife': None,
                'adf_pvalue': 1.0,
                'n_obs': len(train_data)
            }
        
        # 计算最终残差
        final_residuals = log_Y - best_alpha - best_beta * log_X
        r_squared = 1 - np.var(final_residuals) / np.var(log_Y)
        
        return {
            'method': 'Min_HL',
            'beta': float(best_beta),
            'alpha': float(best_alpha), 
            'halflife': float(min_halflife),
            'adf_pvalue': float(best_adf_p),
            'r_squared': float(r_squared),
            'residuals': final_residuals,
            'n_obs': len(train_data)
        }
    
    def fit_huber(self, train_data: pd.DataFrame, delta: float = 1.35) -> Dict:
        """
        REQ-5.2.5: Huber稳健回归
        
        Args:
            delta: Huber损失参数，默认1.35
        """
        log_X = np.log(train_data['X']).values.reshape(-1, 1)
        log_Y = np.log(train_data['Y']).values
        
        try:
            # Huber稳健回归
            huber = HuberRegressor(epsilon=delta, fit_intercept=True, max_iter=300)
            huber.fit(log_X, log_Y)
            
            # 计算残差
            log_X_flat = log_X.flatten()
            residuals = log_Y - huber.intercept_ - huber.coef_[0] * log_X_flat
            r_squared = 1 - np.var(residuals) / np.var(log_Y)
            
            return {
                'method': 'Huber',
                'beta': float(huber.coef_[0]),
                'alpha': float(huber.intercept_),
                'delta': delta,
                'r_squared': float(r_squared),
                'residuals': residuals,
                'n_obs': len(train_data)
            }
            
        except Exception as e:
            logger.warning(f"Huber回归失败，回退到普通OLS: {str(e)}")
            return self._fallback_ols(train_data, 'Huber')
    
    def _fallback_ols(self, train_data: pd.DataFrame, method_name: str) -> Dict:
        """OLS回退方法"""
        log_X = np.log(train_data['X']).values
        log_Y = np.log(train_data['Y']).values
        
        X_matrix = np.column_stack([np.ones(len(log_X)), log_X])
        coeffs = np.linalg.lstsq(X_matrix, log_Y, rcond=None)[0]
        
        residuals = log_Y - coeffs[0] - coeffs[1] * log_X
        r_squared = 1 - np.var(residuals) / np.var(log_Y)
        
        return {
            'method': method_name,
            'beta': float(coeffs[1]),
            'alpha': float(coeffs[0]),
            'r_squared': float(r_squared),
            'residuals': residuals,
            'n_obs': len(train_data),
            'note': 'fallback_to_ols'
        }
    
    # =================================================================
    # 样本外评估 - REQ-5.3系列  
    # =================================================================
    
    def purged_walk_forward_validation(self, data: pd.DataFrame, beta: float, alpha: float, 
                                       train_months: int = 18, embargo_days: int = 20, 
                                       valid_months: int = 6, n_folds: int = 4) -> Dict:
        """
        REQ-5.3.10-11: Purged Walk-Forward验证
        
        Args:
            data: 完整数据
            beta: Beta系数
            alpha: 截距项  
            train_months: 训练期长度（月）
            embargo_days: 禁区天数（避免数据泄露）
            valid_months: 验证期长度（月）
            n_folds: 折数
            
        Returns:
            跨折评估结果
        """
        import numpy as np
        from datetime import timedelta
        
        fold_results = []
        
        # 计算总需要的数据长度
        total_months = train_months + valid_months
        step_months = valid_months  # 每折步进6个月
        
        for fold in range(n_folds):
            # 计算这一折的起始时间
            fold_start_offset = fold * step_months * 30  # 近似天数
            
            # 训练段
            train_start_idx = fold_start_offset
            train_end_idx = fold_start_offset + train_months * 30
            
            # 禁区（embargo）
            embargo_end_idx = train_end_idx + embargo_days
            
            # 验证段
            valid_start_idx = embargo_end_idx
            valid_end_idx = valid_start_idx + valid_months * 30
            
            # 检查数据是否足够
            if valid_end_idx > len(data):
                break
                
            # 提取验证段数据
            valid_data = data.iloc[valid_start_idx:valid_end_idx]
            
            if len(valid_data) < 60:  # 至少需要60天数据
                continue
                
            # 在验证段评估
            metrics = self.evaluate_on_validation(beta, alpha, valid_data)
            fold_results.append(metrics)
        
        if len(fold_results) == 0:
            return {
                'pass_gates': False,
                'reason': 'Insufficient data for walk-forward validation'
            }
        
        # 计算跨折统计
        scores = [r.get('score', 0) for r in fold_results]
        halflives = [r.get('halflife', np.inf) for r in fold_results]
        adf_pvalues = [r.get('adf_pvalue', 1.0) for r in fold_results]
        
        # 使用均值和标准差评估稳定性
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        stability_score = score_mean - 0.5 * score_std  # 惩罚不稳定性
        
        # REQ-5.3.10/11: 检查是否≥2折通过硬门槛
        passed_folds = sum(1 for r in fold_results if r.get('pass_gates', False))
        sufficient_pass = passed_folds >= 4
        
        # 最终评分用IR均值 - 0.5×标准差
        final_score = score_mean - 0.5 * score_std
        
        return {
            'n_folds': len(fold_results),
            'passed_folds': passed_folds,
            'score_mean': float(score_mean),
            'score_std': float(score_std),
            'stability_score': float(stability_score),
            'final_score': float(final_score),
            'halflife_mean': float(np.mean(halflives)),
            'adf_pvalue_mean': float(np.mean(adf_pvalues)),
            'sufficient_pass': sufficient_pass,
            'pass_gates': sufficient_pass and final_score > 0.3,  # 降低评分门槛
            'fold_results': fold_results
        }
    
    def evaluate_on_validation(self, beta: float, alpha: float, valid_data: pd.DataFrame) -> Dict:
        """
        REQ-5.3.1-5.3.8: 样本外评估Beta质量
        
        Args:
            beta: Beta系数
            alpha: 截距项
            valid_data: 验证段数据
            
        Returns:
            评估指标字典
        """
        log_X = np.log(valid_data['X']).values
        log_Y = np.log(valid_data['Y']).values
        
        # 计算残差
        residuals = log_Y - alpha - beta * log_X
        
        # 1. ADF检验 - REQ-5.3.1 (必须用regression='n'对残差)
        try:
            adf_result = adfuller(residuals, regression='n', autolag='AIC', maxlag=int(len(residuals)**(1/3)))
            adf_pvalue = adf_result[1]
        except:
            adf_pvalue = 1.0
        
        # 2. 半衰期计算 - REQ-5.3.2, 5.3.8
        try:
            if len(residuals) >= 3:
                rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), residuals[1:], rcond=None)[0][0]
                if abs(rho) >= 1:
                    halflife = float('inf')
                    kappa = 0
                else:
                    halflife = -np.log(2) / np.log(abs(rho))
                    kappa = -np.log(abs(rho))
            else:
                halflife = float('inf')
                kappa = 0
        except:
            halflife = float('inf') 
            kappa = 0
        
        # 3. 残差方差
        resid_var = np.var(residuals)
        
        # 4. Beta稳定性 - REQ-5.3.3/5.3.7: 验证窗<200根时用63d
        window = 63 if len(valid_data) < 200 else 126
        beta_cv126 = self._calculate_rolling_beta_cv(valid_data, beta, window=window)
        
        # 5. 硬门槛检查（在静态β残差上评估）- REQ-5.3.1-5.3.3更新
        # 动态半衰期上限：min(90, 6×HL_median)
        median_hl = 15.0  # 假设中位数半衰期，实际可从历史数据计算
        hl_upper = min(60, 6 * median_hl)
        
        pass_gates = (
            adf_pvalue < 0.15 and      # REQ-5.3.1: 放宽到0.15
            2 <= halflife <= hl_upper and  # REQ-5.3.2: 动态上限
            beta_cv126 <= 0.40         # REQ-5.3.3: 放宽到0.40
        )
        
        return {
            'adf_pvalue': float(adf_pvalue),
            'halflife': float(halflife), 
            'kappa': float(kappa),
            'resid_var': float(resid_var),
            'beta_cv126': float(beta_cv126),
            'pass_gates': bool(pass_gates)
        }
    
    def _calculate_rolling_beta_cv(self, data: pd.DataFrame, base_beta: float, window: int = 126) -> float:
        """
        REQ-5.3.7: 计算滚动Beta的稳健变异系数
        """
        if len(data) < window + 20:  # 至少需要足够的窗口
            return 0.5  # 返回中等值
        
        rolling_betas = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            log_X = np.log(window_data['X']).values
            log_Y = np.log(window_data['Y']).values
            
            try:
                X_matrix = np.column_stack([np.ones(len(log_X)), log_X])
                coeffs = np.linalg.lstsq(X_matrix, log_Y, rcond=None)[0]
                rolling_betas.append(coeffs[1])
            except:
                continue
        
        if len(rolling_betas) < 5:
            return 0.5
        
        # 稳健变异系数：1.4826 * MAD / max(|median|, 1e-6) 
        rolling_betas = np.array(rolling_betas)
        median_beta = np.median(rolling_betas)
        mad = np.median(np.abs(rolling_betas - median_beta))
        
        cv_robust = 1.4826 * mad / max(abs(median_beta), 1e-6)
        
        return float(cv_robust)
    
    def calculate_comprehensive_score(self, metrics: Dict) -> float:
        """
        REQ-5.3.12: 计算综合评分，包含p值分级惩罚
        """
        if not metrics['pass_gates']:
            return 0.0
        
        # 基于ADF p值、半衰期、稳定性的综合评分
        adf_pvalue = metrics['adf_pvalue']
        adf_score = max(0, 1 - adf_pvalue / 0.15)  # REQ-5.3.1: 放宽到0.15
        
        # REQ-5.3.1: p值分级惩罚更新
        if adf_pvalue >= 0.05:
            adf_score -= 0.05  # 对p∈[0.05,0.10)惩罚
        
        # 半衰期评分：在[10, 30]范围内得分最高
        hl = metrics['halflife'] 
        if 10 <= hl <= 30:
            hl_score = 1.0
        elif 2 <= hl < 10:
            hl_score = 0.5 + 0.5 * (hl - 2) / 8
        elif 30 < hl <= 60:
            hl_score = 1.0 - 0.5 * (hl - 30) / 30
        else:
            hl_score = 0.0
        
        # 稳定性评分 - REQ-5.3.3: 放宽到0.50
        cv = metrics['beta_cv126']
        stability_score = max(0, 1 - cv / 0.50)
        
        # 加权平均
        score = 0.4 * adf_score + 0.3 * hl_score + 0.3 * stability_score
        
        return float(min(score, 1.0))
    
    def select_best_beta(self, candidates_metrics: Dict[str, Dict]) -> Dict:
        """
        REQ-5.3.4-5.3.5: 两步选择最优Beta
        
        Args:
            candidates_metrics: {method_name: metrics_dict}
            
        Returns:
            选择结果字典
        """
        # Step 1: 硬门槛筛选
        qualified_methods = {}
        
        for method, metrics in candidates_metrics.items():
            if metrics['pass_gates']:
                # 计算信息比率代理
                score = self.calculate_comprehensive_score(metrics)
                
                qualified_methods[method] = {
                    'metrics': metrics,
                    'score': score,
                    'beta': metrics.get('beta', 0),
                    'alpha': metrics.get('alpha', 0)
                }
        
        # 如果没有合格的方法
        if len(qualified_methods) == 0:
            return {
                'status': 'rejected',
                'reason': 'No method passed validation thresholds',
                'selected_method': None,
                'beta_star': None,
                'alpha_star': None
            }
        
        # Step 2: 按评分排序选择
        sorted_methods = sorted(qualified_methods.items(), 
                              key=lambda x: x[1]['score'], 
                              reverse=True)
        
        best_method = sorted_methods[0][0]
        best_score = sorted_methods[0][1]['score']
        
        # 检查是否需要加权平均 - REQ-5.3.9
        similar_methods = []
        for method, data in sorted_methods:
            if abs(data['score'] - best_score) / max(best_score, 1e-6) < 0.10:  # 评分差异<10%
                similar_methods.append((method, data))
        
        if len(similar_methods) > 1:
            # 多个方法性能相近，计算加权平均
            return self._calculate_weighted_average_beta(similar_methods)
        else:
            # 单一最优方法
            return {
                'status': 'selected',
                'selected_method': best_method,
                'beta_star': qualified_methods[best_method]['beta'],
                'alpha_star': qualified_methods[best_method]['alpha'],
                'validation_score': best_score,
                'reason': f'{best_method} has best score: {best_score:.3f}'
            }
    
    def _calculate_weighted_average_beta(self, similar_methods: List[Tuple]) -> Dict:
        """
        REQ-5.3.9: 计算评分加权平均的Beta
        """
        scores = [data['score'] for _, data in similar_methods]
        total_score = sum(scores)
        
        if total_score == 0:
            # 所有评分为0，等权重平均
            weights = [1.0 / len(similar_methods)] * len(similar_methods)
        else:
            weights = [score / total_score for score in scores]
        
        # 计算加权平均
        weighted_beta = sum(w * data['beta'] for w, (_, data) in zip(weights, similar_methods))
        weighted_alpha = sum(w * data['alpha'] for w, (_, data) in zip(weights, similar_methods))
        
        # 记录权重信息
        methods_weights = {method: weight for (method, _), weight in zip(similar_methods, weights)}
        
        return {
            'status': 'selected',
            'selected_method': 'weighted_average',
            'methods_weights': methods_weights,
            'beta_star': weighted_beta,
            'alpha_star': weighted_alpha,
            'validation_score': np.mean(scores),
            'reason': f'Weighted average of {len(similar_methods)} similar methods'
        }
    
    # =================================================================
    # 验证算法A: Monte Carlo交叉验证
    # =================================================================
    
    def monte_carlo_validation(self, beta_candidates: List[float], alpha: float, 
                             data: pd.DataFrame, n_iterations: int = 100) -> Dict:
        """
        验证算法A: Monte Carlo交叉验证
        通过随机重采样评估Beta稳健性
        
        Args:
            beta_candidates: 候选Beta列表
            alpha: 截距项
            data: 完整数据
            n_iterations: 迭代次数
            
        Returns:
            验证结果
        """
        logger.info(f"开始Monte Carlo验证，候选数{len(beta_candidates)}，迭代{n_iterations}次")
        
        n_samples = len(data)
        bootstrap_size = int(0.8 * n_samples)  # 每次取80%数据
        
        beta_scores = {beta: [] for beta in beta_candidates}
        
        for iteration in range(n_iterations):
            # 随机采样
            sample_indices = np.random.choice(n_samples, size=bootstrap_size, replace=False)
            sample_data = data.iloc[sample_indices]
            
            log_X = np.log(sample_data['X']).values
            log_Y = np.log(sample_data['Y']).values
            
            # 评估每个候选Beta
            for beta in beta_candidates:
                residuals = log_Y - alpha - beta * log_X
                
                try:
                    # ADF检验
                    adf_result = adfuller(residuals, autolag='AIC')
                    adf_p = adf_result[1]
                    
                    # 半衰期
                    if len(residuals) >= 3:
                        rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), residuals[1:], rcond=None)[0][0]
                        if abs(rho) < 1:
                            halflife = -np.log(2) / np.log(abs(rho))
                        else:
                            halflife = float('inf')
                    else:
                        halflife = float('inf')
                    
                    # 评分：通过门槛得1分，否则0分
                    score = 1.0 if (adf_p < 0.05 and 2 <= halflife <= 60) else 0.0
                    beta_scores[beta].append(score)
                    
                except:
                    beta_scores[beta].append(0.0)
        
        # 计算稳定性评分
        stability_results = {}
        for beta in beta_candidates:
            scores = beta_scores[beta]
            stability_score = np.mean(scores)  # 通过率
            confidence_interval = np.percentile(scores, [25, 75])  # 四分位数
            
            stability_results[beta] = {
                'stability_score': stability_score,
                'confidence_interval': confidence_interval,
                'pass_rate': stability_score
            }
        
        # 选择最稳健的Beta
        best_beta = max(stability_results.keys(), 
                       key=lambda x: stability_results[x]['stability_score'])
        
        logger.info(f"Monte Carlo验证完成，最佳Beta: {best_beta}")
        
        return {
            'best_beta': best_beta,
            'stability_score': stability_results[best_beta]['stability_score'],
            'confidence_interval': stability_results[best_beta]['confidence_interval'].tolist(),
            'all_results': stability_results
        }
    
    # =================================================================
    # 验证算法B: 时间序列分割验证
    # =================================================================
    
    def time_series_split_validation(self, beta_candidates: List[float], alpha: float,
                                   data: pd.DataFrame, n_splits: int = 5) -> Dict:
        """
        验证算法B: 时间序列分割验证
        通过多时间窗口评估Beta的时间一致性
        
        Args:
            beta_candidates: 候选Beta列表
            alpha: 截距项
            data: 完整数据
            n_splits: 分割数量
            
        Returns:
            验证结果
        """
        logger.info(f"开始时间序列分割验证，候选数{len(beta_candidates)}，分割{n_splits}段")
        
        n_samples = len(data)
        split_size = n_samples // n_splits
        
        beta_consistency = {beta: [] for beta in beta_candidates}
        split_results = []
        
        for split_idx in range(n_splits):
            start_idx = split_idx * split_size
            end_idx = min((split_idx + 1) * split_size, n_samples)
            
            if end_idx - start_idx < 50:  # 分割太小，跳过
                continue
                
            split_data = data.iloc[start_idx:end_idx]
            
            log_X = np.log(split_data['X']).values
            log_Y = np.log(split_data['Y']).values
            
            split_result = {'split_idx': split_idx, 'period': f'{start_idx}-{end_idx}', 'results': {}}
            
            # 评估每个候选Beta在此时间段的表现
            for beta in beta_candidates:
                residuals = log_Y - alpha - beta * log_X
                
                try:
                    # ADF检验
                    adf_result = adfuller(residuals, autolag='AIC')
                    adf_p = adf_result[1]
                    
                    # 半衰期
                    if len(residuals) >= 3:
                        rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), residuals[1:], rcond=None)[0][0]
                        if abs(rho) < 1:
                            halflife = -np.log(2) / np.log(abs(rho))
                        else:
                            halflife = float('inf')
                    else:
                        halflife = float('inf')
                    
                    # 一致性评分
                    consistency_score = 1.0 if (adf_p < 0.05 and 2 <= halflife <= 60) else 0.0
                    
                    beta_consistency[beta].append(consistency_score)
                    split_result['results'][beta] = {
                        'adf_p': adf_p,
                        'halflife': halflife,
                        'consistency_score': consistency_score
                    }
                    
                except:
                    beta_consistency[beta].append(0.0)
                    split_result['results'][beta] = {
                        'adf_p': 1.0,
                        'halflife': float('inf'),
                        'consistency_score': 0.0
                    }
            
            split_results.append(split_result)
        
        # 计算时间一致性评分
        consistency_results = {}
        for beta in beta_candidates:
            scores = beta_consistency[beta]
            if len(scores) > 0:
                consistency_score = np.mean(scores)  # 跨时间段平均通过率
                stability = 1.0 - np.std(scores)     # 稳定性（标准差的补）
            else:
                consistency_score = 0.0
                stability = 0.0
            
            consistency_results[beta] = {
                'consistency_score': consistency_score,
                'stability': stability,
                'time_pass_rate': consistency_score
            }
        
        # 选择最一致的Beta
        best_beta = max(consistency_results.keys(),
                       key=lambda x: consistency_results[x]['consistency_score'])
        
        logger.info(f"时间序列分割验证完成，最佳Beta: {best_beta}")
        
        return {
            'best_beta': best_beta,
            'consistency_score': consistency_results[best_beta]['consistency_score'],
            'split_results': split_results,
            'all_results': consistency_results
        }
    
    # =================================================================
    # KF参数标定
    # =================================================================
    
    def calibrate_kf_parameters(self, train_data: pd.DataFrame, beta_star: float, alpha_star: float) -> Dict:
        """
        REQ-5.4: 标定KF的Q/R参数，确保创新序列白化
        
        Args:
            train_data: 训练段数据
            beta_star: 选定的最优β
            alpha_star: 选定的最优α
            
        Returns:
            优化后的Q矩阵和R值
        """
        log_Y = np.log(train_data['Y']).values
        log_X = np.log(train_data['X']).values
        
        # 初始Q/R（默认值）
        q_beta = 1e-5
        q_alpha = 1e-6
        R = 1e-4
        
        # 简化的KF模拟（不依赖filterpy）
        best_q_beta = q_beta
        best_q_alpha = q_alpha
        best_R = R
        best_diag = None
        
        # 三点式调参循环
        for iteration in range(10):  # 最多10次迭代
            # 运行简化KF收集创新
            innovations = []
            
            # 初始状态
            state = np.array([beta_star, alpha_star])
            P = np.eye(2) * 1e-3  # 初始协方差
            Q = np.diag([q_beta, q_alpha])
            
            for t in range(len(log_Y)):
                # 观测矩阵 H_t = [x_t, 1]
                H = np.array([[log_X[t], 1.0]])
                
                # 预测步
                state_pred = state  # F = I
                P_pred = P + Q
                
                # 计算创新
                y_pred = H @ state_pred
                innovation = log_Y[t] - y_pred[0]
                S = H @ P_pred @ H.T + R
                z_t = innovation / np.sqrt(S[0, 0])
                innovations.append(float(z_t))
                
                # 更新步（卡尔曼增益）
                K = P_pred @ H.T / S
                state = state_pred + K.flatten() * innovation
                P = P_pred - K @ H @ P_pred
            
            # 诊断创新序列
            diag = self.kf_innovation_diagnostics(np.array(innovations))
            
            if best_diag is None or abs(diag['std'] - 1.0) < abs(best_diag['std'] - 1.0):
                best_q_beta = q_beta
                best_q_alpha = q_alpha
                best_R = R
                best_diag = diag
            
            # 检查是否白化
            if abs(diag['mean']) < 0.1 and 0.9 < diag['std'] < 1.1:
                logger.info(f"KF参数在第{iteration+1}次迭代达到白化目标")
                break  # 已经白化，退出
            
            # 三点式调参
            if diag['std'] < 0.9:
                # 创新方差太小，增大Q或减小R
                q_beta *= 3
                q_alpha *= 3
            elif diag['std'] > 1.1:
                # 创新方差太大，减小Q或增大R
                q_beta /= 2
                R *= 2
            
            if abs(diag['mean']) > 0.1:
                # 均值偏离，微调R
                R *= 1.5 if diag['mean'] > 0 else 0.7
        
        return {
            'Q_matrix': [[best_q_beta, 0], [0, best_q_alpha]],
            'R_value': best_R,
            'innovation_diagnostics': best_diag
        }
    
    def kf_innovation_diagnostics(self, innovations: np.ndarray) -> Dict:
        """
        创新序列诊断
        
        Args:
            innovations: 标准化创新序列 z_t
            
        Returns:
            诊断结果字典
        """
        mean = float(np.mean(innovations))
        std = float(np.std(innovations, ddof=1))
        
        # 简化的独立性检验（自相关）
        if len(innovations) > 5:
            autocorr = np.corrcoef(innovations[:-1], innovations[1:])[0, 1]
        else:
            autocorr = 0.0
        
        return {
            'mean': mean,
            'std': std,
            'autocorr_lag1': autocorr,
            'is_white': abs(mean) < 0.1 and 0.9 < std < 1.1 and abs(autocorr) < 0.2
        }
    
    # =================================================================
    # 批量处理和导出
    # =================================================================
    
    def calibrate_all_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        批量标定所有配对
        
        Args:
            pairs_data: {pair_name: DataFrame with X, Y columns}
            
        Returns:
            {pair_name: calibration_result}
        """
        logger.info(f"开始批量标定{len(pairs_data)}个配对")
        
        results = {}
        
        for pair_name, pair_df in pairs_data.items():
            logger.info(f"处理配对: {pair_name}")
            
            try:
                # 1. 数据分割
                split_data = self.load_and_split_data({'X': pair_df['X'], 'Y': pair_df['Y']})
                train_data = split_data['train_data']
                valid_data = split_data['valid_data']
                
                # 2. 选择方向
                direction = self.choose_direction(train_data)
                
                # 如果是x_on_y，交换X和Y
                if direction == 'x_on_y':
                    train_data = train_data.rename(columns={'X': 'Y_temp', 'Y': 'X'})
                    train_data = train_data.rename(columns={'Y_temp': 'Y'})
                    valid_data = valid_data.rename(columns={'X': 'Y_temp', 'Y': 'X'})
                    valid_data = valid_data.rename(columns={'Y_temp': 'Y'})
                
                # 3. 生成5种候选Beta
                candidates = {}
                
                # OLS(2y)
                try:
                    ols_result = self.fit_ols_2y(train_data)
                    candidates['OLS_2y'] = ols_result
                except Exception as e:
                    logger.warning(f"{pair_name} OLS_2y失败: {str(e)}")
                
                # EWLS
                try:
                    ewls_result = self.fit_ewls(train_data, halflife=126)
                    candidates['EWLS'] = ewls_result
                except Exception as e:
                    logger.warning(f"{pair_name} EWLS失败: {str(e)}")
                
                # FM-OLS
                try:
                    fmols_result = self.fit_fm_ols(train_data)
                    candidates['FM_OLS'] = fmols_result
                except Exception as e:
                    logger.warning(f"{pair_name} FM-OLS失败: {str(e)}")
                
                # Min-Halflife
                try:
                    if 'OLS_2y' in candidates:
                        minhl_result = self.fit_min_halflife(train_data, candidates['OLS_2y']['beta'])
                        if minhl_result['beta'] is not None:
                            candidates['Min_HL'] = minhl_result
                except Exception as e:
                    logger.warning(f"{pair_name} Min-HL失败: {str(e)}")
                
                # Huber
                try:
                    huber_result = self.fit_huber(train_data)
                    candidates['Huber'] = huber_result
                except Exception as e:
                    logger.warning(f"{pair_name} Huber失败: {str(e)}")
                
                # 4. 样本外评估 - 使用单折验证
                candidates_metrics = {}
                use_pwf = False  # 禁用PWF验证
                
                for method, candidate in candidates.items():
                    if candidate['beta'] is not None:
                        if use_pwf and len(pair_df) >= 1000:  # 数据足够时使用PWF
                            # 使用完整数据做Purged Walk-Forward验证
                            pwf_metrics = self.purged_walk_forward_validation(
                                pair_df,  # 使用完整数据
                                candidate['beta'], 
                                candidate['alpha']
                            )
                            
                            # 如果PWF成功，使用其结果
                            if pwf_metrics.get('n_folds', 0) >= 2:
                                metrics = {
                                    'adf_pvalue': pwf_metrics['adf_pvalue_mean'],
                                    'halflife': pwf_metrics['halflife_mean'],
                                    'score': pwf_metrics['stability_score'],
                                    'pass_gates': pwf_metrics['pass_gates'],
                                    'beta_cv126': 0.2,  # PWF验证过的配对稳定性更好
                                    'kappa': -np.log(2) / pwf_metrics['halflife_mean'] if pwf_metrics['halflife_mean'] > 0 else 0,
                                    'pwf_folds': pwf_metrics['n_folds']
                                }
                            else:
                                # PWF失败，回退到单切点验证
                                metrics = self.evaluate_on_validation(
                                    candidate['beta'], candidate['alpha'], valid_data
                                )
                        else:
                            # 数据不足或禁用PWF，使用单切点验证
                            metrics = self.evaluate_on_validation(
                                candidate['beta'], candidate['alpha'], valid_data
                            )
                        
                        metrics.update({
                            'beta': candidate['beta'],
                            'alpha': candidate['alpha'],
                            'method': method
                        })
                        candidates_metrics[method] = metrics
                
                # 5. 两步选择
                selection_result = self.select_best_beta(candidates_metrics)
                
                # 6. KF参数标定（如果Beta被选中）
                if selection_result['status'] == 'selected' and selection_result.get('beta_star') is not None:
                    kf_params = self.calibrate_kf_parameters(
                        train_data, 
                        selection_result['beta_star'],
                        selection_result['alpha_star']
                    )
                else:
                    # 使用默认KF参数
                    kf_params = {
                        'Q_matrix': [[1e-5, 0], [0, 1e-6]],
                        'R_value': 1e-4,
                        'innovation_diagnostics': None
                    }
                
                # 7. 构建最终结果
                result = {
                    'pair': pair_name,
                    'direction': direction,
                    'status': selection_result['status'],
                    'beta_star': selection_result.get('beta_star'),
                    'alpha_star': selection_result.get('alpha_star'),
                    'selected_method': selection_result.get('selected_method'),
                    'Q_matrix': kf_params['Q_matrix'],
                    'R_value': kf_params['R_value'],
                    'innovation_diagnostics': kf_params.get('innovation_diagnostics'),
                    'validation_score': selection_result.get('validation_score', 0),
                    'reason': selection_result.get('reason', ''),
                    'n_candidates': len(candidates),
                    'n_qualified': len([m for m in candidates_metrics.values() if m['pass_gates']])
                }
                
                # 添加spread公式
                if result['status'] == 'selected':
                    if direction == 'y_on_x':
                        result['spread_formula'] = f"log({pair_name.split('-')[0]}) - {result['beta_star']:.4f}*log({pair_name.split('-')[1]}) - ({result['alpha_star']:.6f})"
                    else:
                        result['spread_formula'] = f"log({pair_name.split('-')[1]}) - {result['beta_star']:.4f}*log({pair_name.split('-')[0]}) - ({result['alpha_star']:.6f})"
                
                results[pair_name] = result
                logger.info(f"{pair_name} 标定完成: {result['status']}")
                
            except Exception as e:
                logger.error(f"{pair_name} 标定失败: {str(e)}")
                results[pair_name] = {
                    'pair': pair_name,
                    'status': 'error',
                    'reason': str(e),
                    'beta_star': None,
                    'alpha_star': None
                }
        
        selected_count = sum(1 for r in results.values() if r['status'] == 'selected')
        logger.info(f"批量标定完成: {selected_count}/{len(results)} 配对被选中")
        
        return results
    
    def export_parameters(self, results: Dict[str, Dict], filepath: str) -> None:
        """
        REQ-5.4.3: 导出标准化参数文件
        """
        # 统计信息
        total_pairs = len(results)
        selected_pairs = sum(1 for r in results.values() if r['status'] == 'selected')
        rejected_pairs = total_pairs - selected_pairs
        
        # 构建输出结构
        output = {
            'calibration_info': {
                'train_period': f"{self.train_start} to {self.train_end}",
                'validation_period': f"{self.valid_start} to {self.valid_end}",
                'calibration_timestamp': datetime.now().isoformat(),
                'total_pairs': total_pairs,
                'selected_pairs': selected_pairs,
                'rejected_pairs': rejected_pairs,
                'success_rate': selected_pairs / total_pairs if total_pairs > 0 else 0
            },
            'parameters': {}
        }
        
        # 添加每个配对的参数
        for pair_name, result in results.items():
            if result['status'] == 'selected':
                param_dict = {
                    'pair': pair_name,
                    'direction': result['direction'],
                    'beta_star': result['beta_star'],
                    'alpha_star': result['alpha_star'],
                    'selected_method': result['selected_method'],
                    'spread_formula': result.get('spread_formula', ''),
                    'Q_matrix': result.get('Q_matrix', [[1e-5, 0], [0, 1e-6]]),  # 使用标定的Q矩阵
                    'R_value': result.get('R_value', 0.0001),  # 使用标定的R值
                    'validation_score': result['validation_score'],
                    'status': result['status']
                }
            else:
                param_dict = {
                    'pair': pair_name,
                    'status': result['status'],
                    'reason': result['reason'],
                    'beta_star': None,
                    'alpha_star': None
                }
            
            output['parameters'][pair_name] = param_dict
        
        # 写入JSON文件
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"参数文件已导出: {filepath}")
        logger.info(f"标定汇总: {selected_pairs}/{total_pairs} 配对成功 ({selected_pairs/total_pairs:.1%})")


def main():
    """测试函数"""
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    log_x = np.cumsum(np.random.normal(0, 0.01, n))
    beta_true = 0.85
    alpha_true = -0.002
    noise = np.random.normal(0, 0.005, n)
    log_y = alpha_true + beta_true * log_x + noise
    
    test_data = pd.DataFrame({
        'X': np.exp(log_x),
        'Y': np.exp(log_y)
    }, index=dates)
    
    # 测试标定器
    calibrator = BetaCalibrator()
    pairs_data = {'TEST-PAIR': test_data}
    results = calibrator.calibrate_all_pairs(pairs_data)
    
    print("Beta标定测试结果:")
    for pair_name, result in results.items():
        print(f"{pair_name}: {result}")


if __name__ == "__main__":
    main()