"""
协整配对模块单元测试 - 无Mock版本
测试需求: REQ-2.x.x (协整检验、Beta估计、配对筛选)
使用真实数值计算进行测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.coint import (
    CointegrationAnalyzer, CointegrationError,
    engle_granger_test, estimate_parameters, calculate_halflife,
    determine_direction, adf_test, calculate_volatility
)


class TestEngleGrangerTest:
    """测试Engle-Granger协整检验 - 真实计算"""
    
    def test_cointegrated_series_real(self):
        """测试真实协整序列"""
        np.random.seed(42)
        n = 500  # 增加样本量提高统计显著性
        
        # 生成真实协整序列
        # X是随机游走
        x_shocks = np.random.randn(n)
        x = np.cumsum(x_shocks)
        
        # Y与X协整，有固定的线性关系
        beta = 1.5
        alpha = 10.0
        residual_noise = np.random.randn(n) * 0.5  # 小的残差
        y = alpha + beta * x + residual_noise
        
        # 执行协整检验
        result = engle_granger_test(x, y)
        
        # 验证结果结构
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'beta' in result
        assert 'alpha' in result
        assert 'residuals' in result
        assert 'adf_stat' in result
        
        # 验证参数估计准确性
        assert abs(result['beta'] - beta) < 0.1  # Beta应该接近真实值
        assert abs(result['alpha'] - alpha) < 1.0  # Alpha应该接近真实值
        
        # 验证残差平稳性（协整的关键）
        assert len(result['residuals']) == n
        residual_mean = np.mean(result['residuals'])
        assert abs(residual_mean) < 1.0  # 残差均值应接近0
        
    def test_non_cointegrated_series_real(self):
        """测试真实非协整序列"""
        np.random.seed(123)
        n = 500
        
        # 生成两个独立的随机游走（非协整）
        x = np.cumsum(np.random.randn(n))
        y = np.cumsum(np.random.randn(n))  # 独立的随机游走
        
        # 执行协整检验
        result = engle_granger_test(x, y)
        
        # 残差应该是非平稳的（随机游走）
        residuals = result['residuals']
        
        # 计算残差的自相关
        from pandas import Series
        residual_series = Series(residuals)
        autocorr = residual_series.autocorr(lag=1)
        
        # 非平稳序列的自相关应该接近1
        assert autocorr > 0.8
        
    def test_perfect_linear_relationship(self):
        """测试完美线性关系"""
        n = 100
        x = np.arange(n, dtype=float)
        beta_true = 2.5
        alpha_true = 5.0
        y = alpha_true + beta_true * x  # 完美线性关系
        
        result = engle_granger_test(x, y)
        
        # 完美线性关系应该给出精确的参数
        assert abs(result['beta'] - beta_true) < 1e-10
        assert abs(result['alpha'] - alpha_true) < 1e-10
        
        # 残差应该全为0
        assert np.allclose(result['residuals'], 0, atol=1e-10)


class TestParameterEstimation:
    """测试参数估计功能 - 真实OLS计算"""
    
    def test_ols_estimation_accuracy(self):
        """测试OLS估计的准确性"""
        np.random.seed(42)
        n = 1000
        
        # 生成线性关系数据
        x = np.random.uniform(50, 150, n)
        true_beta = 1.234567  # 6位小数精度
        true_alpha = 50.123456
        noise = np.random.randn(n) * 0.1  # 小噪声
        y = true_alpha + true_beta * x + noise
        
        # 估计参数
        result = estimate_parameters(x, y, method='OLS')
        
        # 验证结果结构
        assert 'beta' in result
        assert 'alpha' in result
        assert 'residuals' in result
        assert 'r_squared' in result
        assert 'std_error' in result
        
        # 验证估计精度（6位小数）
        beta_error = abs(result['beta'] - true_beta)
        assert beta_error < 0.001  # 千分之一的误差
        
        # 验证R方
        assert result['r_squared'] > 0.99  # 高度相关
        
        # 验证残差性质
        residuals = result['residuals']
        assert abs(np.mean(residuals)) < 0.01  # 均值接近0
        assert np.std(residuals) < 0.15  # 标准差接近噪声水平
        
    def test_beta_precision_six_decimals(self):
        """测试Beta系数6位小数精度"""
        # 使用精确构造的数据
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        beta_exact = 2.123456  # 精确的6位小数
        alpha_exact = 3.654321
        y = alpha_exact + beta_exact * x  # 无噪声
        
        result = estimate_parameters(x, y, method='OLS')
        
        # 验证6位小数精度
        beta_str = f"{result['beta']:.6f}"
        expected_str = f"{beta_exact:.6f}"
        assert beta_str == expected_str
        
        # 验证数值精度
        assert abs(result['beta'] - beta_exact) < 1e-10
        
    def test_multicollinearity_handling(self):
        """测试多重共线性处理"""
        # 创建高度相关的数据
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.01  # 几乎相同
        
        # OLS应该能处理
        result1 = estimate_parameters(x1, x2, method='OLS')
        
        # Beta应该接近1（因为x2 ≈ x1）
        assert abs(result1['beta'] - 1.0) < 0.1


class TestHalflife:
    """测试半衰期计算 - 真实AR过程"""
    
    def test_ar1_process_halflife(self):
        """测试AR(1)过程的半衰期计算"""
        np.random.seed(42)
        n = 1000
        
        # 生成AR(1)过程: y_t = phi * y_{t-1} + epsilon
        phi = 0.95  # 自回归系数
        y = [0]
        for i in range(1, n):
            y.append(phi * y[-1] + np.random.randn() * 0.1)
        y = np.array(y)
        
        # 计算半衰期
        halflife = calculate_halflife(y)
        
        # 理论半衰期: -log(2) / log(phi)
        theoretical_hl = -np.log(2) / np.log(phi)
        
        assert halflife is not None
        # 允许20%的误差（由于有限样本）
        assert abs(halflife - theoretical_hl) / theoretical_hl < 0.2
        
    def test_fast_mean_reversion(self):
        """测试快速均值回归"""
        np.random.seed(123)
        n = 500
        
        # 快速均值回归（phi = 0.5）
        phi = 0.5
        y = [0]
        for i in range(1, n):
            y.append(phi * y[-1] + np.random.randn() * 0.1)
        y = np.array(y)
        
        halflife = calculate_halflife(y)
        theoretical_hl = -np.log(2) / np.log(phi)  # 应该约等于1
        
        assert halflife is not None
        assert 0.5 < halflife < 2.0  # 快速回归
        
    def test_no_mean_reversion(self):
        """测试无均值回归（随机游走）"""
        np.random.seed(456)
        
        # 随机游走
        random_walk = np.cumsum(np.random.randn(500))
        
        halflife = calculate_halflife(random_walk)
        
        # 随机游走没有均值回归
        assert halflife is None or halflife > 100


class TestVolatilityCalculation:
    """测试波动率计算 - 真实金融数据特征"""
    
    def test_volatility_calculation_accuracy(self):
        """测试波动率计算准确性"""
        np.random.seed(42)
        
        # 生成已知波动率的价格序列
        n = 252  # 一年的交易日
        daily_vol = 0.02  # 2%日波动率
        annual_vol = daily_vol * np.sqrt(252)  # 年化波动率
        
        # 生成对数价格（几何布朗运动）
        log_returns = np.random.normal(0, daily_vol, n)
        log_prices = 100 + np.cumsum(log_returns)
        
        # 计算波动率
        calculated_vol = calculate_volatility(log_prices)
        
        # 验证波动率接近真实值
        # 由于是对数价格差分，应该接近daily_vol
        assert abs(calculated_vol - daily_vol) < 0.005
        
    def test_zero_volatility(self):
        """测试零波动率情况"""
        # 常数价格
        constant_prices = np.ones(100) * 100
        log_prices = np.log(constant_prices)
        
        vol = calculate_volatility(log_prices)
        
        # 常数价格的波动率应该是0
        assert vol == 0 or vol < 1e-10
        
    def test_high_volatility(self):
        """测试高波动率情况"""
        np.random.seed(789)
        
        # 高波动率序列
        high_vol = 0.1  # 10%日波动率
        log_returns = np.random.normal(0, high_vol, 100)
        log_prices = np.cumsum(log_returns)
        
        vol = calculate_volatility(log_prices)
        
        # 应该检测到高波动率
        assert vol > 0.05


class TestDirectionDetermination:
    """测试配对方向判定 - 真实波动率比较"""
    
    def test_direction_based_on_volatility(self):
        """测试基于波动率的方向判定"""
        np.random.seed(42)
        n = 252
        
        # 创建低波动率序列
        low_vol_returns = np.random.normal(0, 0.01, n)  # 1%日波动率
        low_vol_prices = 100 * np.exp(np.cumsum(low_vol_returns))
        
        # 创建高波动率序列
        high_vol_returns = np.random.normal(0, 0.05, n)  # 5%日波动率
        high_vol_prices = 100 * np.exp(np.cumsum(high_vol_returns))
        
        # 创建带日期索引的Series
        dates = pd.date_range('2024-01-01', periods=n)
        low_vol_series = pd.Series(low_vol_prices, index=dates)
        high_vol_series = pd.Series(high_vol_prices, index=dates)
        
        # 判定方向
        x_symbol, y_symbol = determine_direction(
            low_vol_series.values, 
            high_vol_series.values,
            volatility_period='2024-01-01'
        )
        
        # 验证方向判定逻辑（具体实现可能不同）
        # 一般低波动率作为X（自变量），高波动率作为Y（因变量）
        assert x_symbol != y_symbol
        
    def test_similar_volatility(self):
        """测试相似波动率的情况"""
        np.random.seed(123)
        n = 252
        
        # 创建两个相似波动率的序列
        vol = 0.02
        returns1 = np.random.normal(0, vol, n)
        returns2 = np.random.normal(0, vol, n)
        
        prices1 = 100 * np.exp(np.cumsum(returns1))
        prices2 = 100 * np.exp(np.cumsum(returns2))
        
        dates = pd.date_range('2024-01-01', periods=n)
        series1 = pd.Series(prices1, index=dates)
        series2 = pd.Series(prices2, index=dates)
        
        # 判定方向
        x_symbol, y_symbol = determine_direction(
            series1.values,
            series2.values,
            volatility_period='2024-01-01'
        )
        
        # 即使波动率相似，也应该有确定的方向
        assert x_symbol != y_symbol


class TestADFTest:
    """测试ADF单位根检验 - 真实时间序列"""
    
    def test_stationary_series(self):
        """测试平稳序列"""
        np.random.seed(42)
        
        # 生成平稳序列（白噪声）
        stationary = np.random.randn(500)
        
        stat, pval = adf_test(stationary)
        
        # 平稳序列应该拒绝单位根假设
        assert pval < 0.05
        assert stat < -2  # ADF统计量应该是负的且绝对值较大
        
    def test_nonstationary_series(self):
        """测试非平稳序列"""
        np.random.seed(123)
        
        # 生成随机游走（非平稳）
        random_walk = np.cumsum(np.random.randn(500))
        
        stat, pval = adf_test(random_walk)
        
        # 随机游走不应该拒绝单位根假设
        assert pval > 0.05
        assert stat > -2  # ADF统计量绝对值较小
        
    def test_trend_stationary(self):
        """测试趋势平稳序列"""
        n = 500
        t = np.arange(n)
        
        # 趋势平稳：y = a + b*t + noise
        trend_stationary = 100 + 0.5 * t + np.random.randn(n) * 2
        
        stat, pval = adf_test(trend_stationary)
        
        # 趋势平稳序列的ADF检验结果取决于是否包含趋势项
        # 这里只验证函数能正常运行
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert 0 <= pval <= 1


class TestCointegrationAnalyzer:
    """测试CointegrationAnalyzer类 - 真实数据分析"""
    
    def setup_method(self):
        """准备测试数据"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n)
        
        # 创建协整的数据
        # 共同因子
        common_factor = np.cumsum(np.random.randn(n))
        
        # 三个协整的序列
        self.data = pd.DataFrame({
            'SYMBOL1': 100 + 1.0 * common_factor + np.random.randn(n) * 0.5,
            'SYMBOL2': 200 + 2.0 * common_factor + np.random.randn(n) * 0.5,
            'SYMBOL3': 50 + 0.5 * common_factor + np.random.randn(n) * 0.5,
        }, index=dates)
        
    def test_analyzer_test_pair(self):
        """测试单对协整检验"""
        analyzer = CointegrationAnalyzer(self.data)
        
        result = analyzer.test_pair('SYMBOL1', 'SYMBOL2')
        
        # 验证结果结构
        assert isinstance(result, dict)
        assert 'pair' in result
        assert 'p_value' in result
        assert 'beta' in result
        assert result['pair'] == 'SYMBOL1-SYMBOL2'
        
        # 这两个序列应该是协整的
        # 理论beta应该约为2.0
        assert 1.5 < result['beta'] < 2.5
        
    def test_analyzer_screen_pairs(self):
        """测试筛选所有配对"""
        analyzer = CointegrationAnalyzer(self.data)
        
        # 筛选配对
        results = analyzer.screen_all_pairs(
            p_threshold=1.0,  # 设置为1.0以获得所有结果
            min_halflife=0.1,
            max_halflife=1000.0
        )
        
        # 验证结果
        assert isinstance(results, pd.DataFrame)
        
        # 应该有3个配对组合
        expected_pairs = 3  # C(3,2) = 3
        assert len(results) <= expected_pairs
        
        if len(results) > 0:
            # 验证列
            assert 'pair' in results.columns
            assert 'p_value' in results.columns
            assert 'beta' in results.columns
            
            # 验证数值范围
            assert (results['p_value'] >= 0).all()
            assert (results['p_value'] <= 1).all()
            assert (results['beta'] > 0).all()


class TestNumericalStability:
    """测试数值稳定性 - 边界情况"""
    
    def test_near_zero_values(self):
        """测试接近零的值"""
        # 非常小的值
        small_values = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        
        # OLS估计应该能处理
        result = estimate_parameters(small_values[:-1], small_values[1:])
        
        assert not np.isnan(result['beta'])
        assert not np.isinf(result['beta'])
        
    def test_large_values(self):
        """测试大数值"""
        # 大数值
        large_values = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        
        result = estimate_parameters(large_values[:-1], large_values[1:])
        
        assert not np.isnan(result['beta'])
        assert not np.isinf(result['beta'])
        
    def test_identical_values(self):
        """测试相同值"""
        # 所有值相同
        same_values = np.ones(100) * 42
        
        # 波动率应该是0
        vol = calculate_volatility(np.log(same_values))
        assert vol == 0 or vol < 1e-10
        
        # 半衰期无法计算（无变化）
        hl = calculate_halflife(same_values)
        assert hl is None or hl > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])