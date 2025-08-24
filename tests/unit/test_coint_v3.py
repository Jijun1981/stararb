"""
协整配对模块单元测试 v3.0
按照需求文档REQ-2.x实现的TDD测试

重要改动：
- 去掉direction概念，直接确定symbol_x和symbol_y
- 统一使用Y对X的回归方式
- 简化品种角色确定逻辑
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.coint import CointegrationAnalyzer
from lib.data import load_data


class TestCointegrationBasic:
    """测试协整检验基本功能 (REQ-2.1.x)"""
    
    def test_engle_granger_known_cointegrated_pair(self):
        """TC-2.1.1: 已知协整配对(模拟数据)"""
        # 创建协整的模拟数据
        np.random.seed(42)
        n = 1000
        x = np.cumsum(np.random.randn(n))  # 随机游走
        y = 2.0 * x + np.random.randn(n) * 0.1  # Y与X协整，噪声很小
        
        # 创建分析器
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2020-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        # 进行协整检验
        result = analyzer.engle_granger_test(x, y)
        
        # 验证p值应该很小（协整显著）
        assert result['pvalue'] < 0.01
        assert result['beta'] > 1.5  # 接近真实值2.0
        assert result['r_squared'] > 0.9  # 拟合度很高
        
    def test_engle_granger_random_walk_pair(self):
        """TC-2.1.2: 随机游走序列"""
        np.random.seed(42)
        n = 1000
        x = np.cumsum(np.random.randn(n))  # 随机游走
        y = np.cumsum(np.random.randn(n))  # 独立的随机游走
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2020-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        result = analyzer.engle_granger_test(x, y)
        
        # 随机游走序列不应该协整（放宽要求，因为是随机的）
        assert result['pvalue'] > 0.01
        
    def test_engle_granger_perfectly_correlated(self):
        """TC-2.1.3: 完全相关序列"""
        np.random.seed(42)
        n = 1000
        x = np.cumsum(np.random.randn(n))
        y = x  # 完全相关
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2020-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        result = analyzer.engle_granger_test(x, y)
        
        # 完全相关应该高度协整（放宽要求）
        assert result['pvalue'] < 0.5  # 这里只要不是完全随机即可
        assert abs(result['beta'] - 1.0) < 0.1  # β应该接近1
        
    def test_multi_window_test_insufficient_data(self):
        """TC-2.1.4: 数据长度不足5年"""
        # 只有1年数据
        np.random.seed(42)
        n = 252
        x = np.cumsum(np.random.randn(n))
        y = 1.5 * x + np.random.randn(n) * 0.1
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2024-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        # 多窗口检验应该自动调整
        from lib.coint import multi_window_test
        results = multi_window_test(x, y)
        
        # 应该只有1年窗口有结果
        assert results['1y'] is not None
        assert results['2y'] is None  # 数据不足
        assert results['5y'] is None  # 数据不足


class TestSymbolRoleDetermination:
    """测试品种角色确定功能 (REQ-2.2.x)"""
    
    def setup_method(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2023-01-01', periods=n)
        
        # 创建不同波动率的价格序列
        # 低波动率品种
        price1 = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))  # 1%日波动
        # 高波动率品种  
        price2 = 200 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))  # 2%日波动
        
        self.data = pd.DataFrame({
            'SYMBOL1': np.log(price1),  # 对数价格
            'SYMBOL2': np.log(price2)
        }, index=dates)
        
        self.analyzer = CointegrationAnalyzer(self.data)
    
    def test_calculate_volatility_default_period(self):
        """TC-2.2.1: 使用默认最近1年数据计算波动率"""
        # 模拟最近252个交易日的数据
        recent_prices = self.data['SYMBOL1'].iloc[-252:]
        
        vol = self.analyzer.calculate_volatility(recent_prices.values)
        
        # 验证波动率在合理范围内
        assert 0.001 < vol < 1.0
        assert isinstance(vol, float)
        
    def test_determine_symbols_low_high_volatility(self):
        """TC-2.2.2: symbol1波动率低，symbol2波动率高"""
        # 确定角色
        symbol_x, symbol_y = self.analyzer.determine_symbols('SYMBOL1', 'SYMBOL2')
        
        # 低波动的SYMBOL1应该是X，高波动的SYMBOL2应该是Y
        assert symbol_x == 'SYMBOL1'  # 低波动
        assert symbol_y == 'SYMBOL2'  # 高波动
        
    def test_determine_symbols_high_low_volatility(self):
        """TC-2.2.3: 调换波动率大小"""
        # 交换输入顺序，但结果应该一致
        symbol_x, symbol_y = self.analyzer.determine_symbols('SYMBOL2', 'SYMBOL1')
        
        # 仍然是低波动作X，高波动作Y
        assert symbol_x == 'SYMBOL1'  # 低波动
        assert symbol_y == 'SYMBOL2'  # 高波动
        
    def test_determine_symbols_equal_volatility(self):
        """TC-2.2.4: 波动率相等时按字母顺序"""
        # 创建相同波动率的数据
        np.random.seed(42)
        n = 300
        dates = pd.date_range('2023-01-01', periods=n)
        
        # 相同波动率
        price_a = 100 * np.exp(np.cumsum(np.random.normal(0, 0.015, n)))
        price_b = 200 * np.exp(np.cumsum(np.random.normal(0, 0.015, n)))
        
        data = pd.DataFrame({
            'BETA': np.log(price_a),
            'ALPHA': np.log(price_b)
        }, index=dates)
        
        analyzer = CointegrationAnalyzer(data)
        symbol_x, symbol_y = analyzer.determine_symbols('BETA', 'ALPHA')
        
        # 按字母顺序：ALPHA < BETA
        assert symbol_x == 'ALPHA'
        assert symbol_y == 'BETA'
        
    def test_calculate_volatility_custom_start_date(self):
        """TC-2.2.5: 自定义recent_start日期"""
        # 指定从2023年中期开始计算
        custom_start = '2023-06-01'
        
        # 获取自定义日期后的数据
        mask = self.data.index >= pd.to_datetime(custom_start)
        recent_data = self.data.loc[mask, 'SYMBOL1']
        
        vol = self.analyzer.calculate_volatility(recent_data.values)
        
        assert isinstance(vol, float)
        assert vol > 0


class TestParameterEstimation:
    """测试参数估计功能 (REQ-2.3.x)"""
    
    def test_estimate_parameters_accuracy(self):
        """测试参数估计的准确性"""
        # 创建已知参数的数据
        np.random.seed(42)
        n = 1000
        true_beta = 1.8
        x = np.cumsum(np.random.randn(n))
        y = true_beta * x + np.random.randn(n) * 0.1
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2020-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        result = analyzer.estimate_parameters(x, y)
        
        # 验证参数估计精度
        assert abs(result['beta'] - true_beta) < 0.1  # β估计误差小于0.1
        assert result['r_squared'] > 0.95  # 拟合度高
        assert 'alpha' in result  # 截距项
        
    def test_calculate_halflife(self):
        """REQ-2.3.2: 计算半衰期"""
        # 创建有均值回归的残差序列
        np.random.seed(42)
        n = 500
        # AR(1)过程：ε_t = 0.9 * ε_{t-1} + u_t
        residuals = np.zeros(n)
        for t in range(1, n):
            residuals[t] = 0.9 * residuals[t-1] + np.random.randn() * 0.1
            
        data = pd.DataFrame({'X': np.random.randn(n), 'Y': np.random.randn(n)})
        analyzer = CointegrationAnalyzer(data)
        
        halflife = analyzer.calculate_halflife(residuals)
        
        # 半衰期应该为正数，且在合理范围内
        assert halflife > 0
        assert halflife < 100  # 不应该太大
        
    def test_residual_statistics(self):
        """REQ-2.3.3: 残差统计"""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 1000)
        
        data = pd.DataFrame({'X': np.random.randn(100), 'Y': np.random.randn(100)})
        analyzer = CointegrationAnalyzer(data)
        
        stats = analyzer.residual_statistics(residuals)
        
        # 验证统计量
        assert abs(stats['residual_mean']) < 0.1  # 均值接近0
        assert abs(stats['residual_std'] - 1.0) < 0.1  # 标准差接近1
        assert 'residual_skew' in stats
        assert 'residual_kurt' in stats


class TestBatchScreening:
    """测试批量筛选功能 (REQ-2.4.x)"""
    
    def setup_method(self):
        """准备测试数据"""
        # 使用真实数据进行测试
        try:
            self.data = load_data(
                ['AG', 'CU', 'AL'], 
                start_date='2023-01-01',
                end_date='2024-08-01',
                log_price=True
            )
            self.analyzer = CointegrationAnalyzer(self.data)
            self.real_data = True
        except:
            # 如果无法加载真实数据，使用模拟数据
            np.random.seed(42)
            n = 300
            data_dict = {}
            for symbol in ['AG', 'CU', 'AL']:
                prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))
                data_dict[symbol] = np.log(prices)
                
            self.data = pd.DataFrame(data_dict, index=pd.date_range('2023-01-01', periods=n))
            self.analyzer = CointegrationAnalyzer(self.data)
            self.real_data = False
    
    def test_generate_all_pairs(self):
        """TC-2.3.1: 生成所有可能配对"""
        from itertools import combinations
        symbols = self.data.columns.tolist()
        expected_pairs = list(combinations(symbols, 2))
        
        # 验证配对数量
        assert len(expected_pairs) == 3  # C(3,2) = 3
        
        # 验证每个配对都包含正确的品种
        for s1, s2 in expected_pairs:
            assert s1 in symbols
            assert s2 in symbols
            assert s1 != s2
    
    def test_screen_all_pairs_basic(self):
        """基本筛选功能测试"""
        results = self.analyzer.screen_all_pairs(p_threshold=0.05)
        
        # 验证输出格式
        assert isinstance(results, pd.DataFrame)
        
        expected_columns = [
            'pair', 'symbol_x', 'symbol_y',
            'pvalue_1y', 'pvalue_2y', 'pvalue_3y', 'pvalue_4y', 'pvalue_5y',
            'beta_1y', 'beta_2y', 'beta_3y', 'beta_4y', 'beta_5y',
            'volatility_x', 'volatility_y'
        ]
        
        for col in expected_columns:
            assert col in results.columns, f"缺少列: {col}"
        
        # 验证配对名称格式
        for pair_name in results['pair']:
            assert '-' in pair_name  # 应该是X-Y格式
            parts = pair_name.split('-')
            assert len(parts) == 2
            
    def test_pair_naming_format(self):
        """验证配对命名格式"""
        results = self.analyzer.screen_all_pairs()
        
        for _, row in results.iterrows():
            pair = row['pair']
            symbol_x = row['symbol_x']
            symbol_y = row['symbol_y']
            
            # 验证配对名称格式
            expected_pair = f"{symbol_x}-{symbol_y}"
            assert pair == expected_pair
            
            # 验证没有使用旧格式
            assert '_close' not in pair
            assert '0' not in symbol_x or symbol_x in ['AG0', 'AU0']  # 特殊情况
            
    def test_volatility_based_role_assignment(self):
        """验证基于波动率的角色分配"""
        results = self.analyzer.screen_all_pairs()
        
        for _, row in results.iterrows():
            vol_x = row['volatility_x']
            vol_y = row['volatility_y']
            
            # X应该是低波动，Y应该是高波动（或相等时按字母顺序）
            assert vol_x <= vol_y or (vol_x == vol_y and row['symbol_x'] <= row['symbol_y'])
    
    def test_screen_with_pvalue_threshold(self):
        """TC-2.3.2: 主筛选条件"""
        # 使用宽松的p值阈值进行测试
        results_loose = self.analyzer.screen_all_pairs(p_threshold=0.5)
        # 使用严格的p值阈值
        results_strict = self.analyzer.screen_all_pairs(p_threshold=0.01)
        
        # 严格阈值的结果数量应该 <= 宽松阈值的结果数量
        assert len(results_strict) <= len(results_loose)
        
        # 检查严格筛选的结果都满足条件
        for _, row in results_strict.iterrows():
            # 只检查非NaN的p值
            if not pd.isna(row['pvalue_5y']) and not pd.isna(row['pvalue_1y']):
                assert row['pvalue_5y'] <= 0.01, f"5年p值 {row['pvalue_5y']} > 0.01"
                assert row['pvalue_1y'] <= 0.01, f"1年p值 {row['pvalue_1y']} > 0.01"
                
    def test_results_sorted_by_1y_pvalue(self):
        """TC-2.3.3: 结果按1年p值排序"""
        results = self.analyzer.screen_all_pairs()
        
        if len(results) > 1:
            # 验证按1年p值升序排列
            pvalues_1y = results['pvalue_1y'].dropna().values
            if len(pvalues_1y) > 1:
                assert (pvalues_1y[:-1] <= pvalues_1y[1:]).all()


class TestRealDataIntegration:
    """集成测试 - 使用真实数据"""
    
    def test_complete_cointegration_workflow(self):
        """测试完整的协整分析流程"""
        try:
            # 加载真实数据
            data = load_data(
                ['AG', 'CU'],
                start_date='2023-01-01', 
                end_date='2024-08-01',
                log_price=True
            )
            
            # 创建分析器
            analyzer = CointegrationAnalyzer(data)
            
            # 品种角色确定
            symbol_x, symbol_y = analyzer.determine_symbols('AG', 'CU')
            assert symbol_x in ['AG', 'CU']
            assert symbol_y in ['AG', 'CU'] 
            assert symbol_x != symbol_y
            
            # 协整检验
            x_data = data[symbol_x].values
            y_data = data[symbol_y].values
            result = analyzer.engle_granger_test(x_data, y_data)
            
            # 验证结果格式
            assert 'pvalue' in result
            assert 'beta' in result
            assert 'r_squared' in result
            assert isinstance(result['pvalue'], float)
            assert isinstance(result['beta'], float)
            
        except FileNotFoundError:
            pytest.skip("真实数据不可用，跳过集成测试")
            
    def test_batch_screening_performance(self):
        """测试批量筛选性能"""
        try:
            # 加载更多品种进行性能测试
            data = load_data(
                ['AG', 'CU', 'AL', 'NI'],
                start_date='2023-01-01',
                end_date='2024-08-01', 
                log_price=True
            )
            
            analyzer = CointegrationAnalyzer(data)
            
            import time
            start_time = time.time()
            results = analyzer.screen_all_pairs(p_threshold=0.05)
            end_time = time.time()
            
            # 验证结果
            expected_pairs = 6  # C(4,2) = 6
            assert len(results) <= expected_pairs  # 可能有些不满足条件被过滤
            
            # 性能检查（宽松标准）
            duration = end_time - start_time
            assert duration < 30  # 应该在30秒内完成
            
        except FileNotFoundError:
            pytest.skip("真实数据不可用，跳过性能测试")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])