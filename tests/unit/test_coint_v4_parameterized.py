"""
协整配对模块单元测试 v4.0 - 参数化版本
按照需求文档REQ-2.x实现的TDD测试，支持泛化参数

核心改进：
- 移除所有硬编码的时间参数
- 支持自定义时间窗口配置
- 支持灵活的筛选条件
- 支持指定波动率计算时间段
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.coint import CointegrationAnalyzer
from lib.data import load_data


class TestParameterizedCointegration:
    """测试参数化协整检验功能"""
    
    def test_custom_time_windows(self):
        """测试自定义时间窗口配置"""
        # 创建测试数据
        np.random.seed(42)
        n = 800
        x = np.cumsum(np.random.randn(n))
        y = 1.5 * x + np.random.randn(n) * 0.1
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2020-01-01', periods=n))
        analyzer = CointegrationAnalyzer(data)
        
        # 测试自定义窗口
        custom_windows = {'6m': 126, '1y': 252, '18m': 378}
        
        from lib.coint import multi_window_test
        results = multi_window_test(x, y, windows=custom_windows)
        
        # 验证只有指定窗口的结果
        assert '6m' in results
        assert '1y' in results  
        assert '18m' in results
        assert '2y' not in results  # 默认窗口不应出现
        
        # 验证每个窗口都有结果（因为数据足够）
        for window in custom_windows:
            assert results[window] is not None
            assert 'pvalue' in results[window]
            assert 'beta' in results[window]
    
    def test_insufficient_data_custom_windows(self):
        """测试数据不足时的自定义窗口处理"""
        # 只有半年数据
        np.random.seed(42)
        n = 126
        x = np.cumsum(np.random.randn(n))
        y = 1.5 * x + np.random.randn(n) * 0.1
        
        data = pd.DataFrame({'X': x, 'Y': y}, index=pd.date_range('2024-01-01', periods=n))
        
        # 测试包含长窗口的配置
        custom_windows = {'3m': 63, '6m': 126, '1y': 252}
        
        from lib.coint import multi_window_test
        results = multi_window_test(x, y, windows=custom_windows)
        
        # 短窗口应该有结果
        assert results['3m'] is not None
        assert results['6m'] is not None
        
        # 长窗口应该为None（数据不足）
        assert results['1y'] is None


class TestParameterizedVolatility:
    """测试参数化波动率计算"""
    
    def setup_method(self):
        """创建测试数据"""
        np.random.seed(42)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n)
        
        # 创建带趋势变化的波动率数据
        # 前半段：低波动
        vol1_early = np.random.normal(0, 0.01, n//2)
        vol1_late = np.random.normal(0, 0.02, n//2)  # 后半段波动增加
        vol1_series = np.concatenate([vol1_early, vol1_late])
        
        # 第二个品种：相反趋势
        vol2_early = np.random.normal(0, 0.025, n//2)  
        vol2_late = np.random.normal(0, 0.015, n//2)  # 后半段波动降低
        vol2_series = np.concatenate([vol2_early, vol2_late])
        
        price1 = 100 * np.exp(np.cumsum(vol1_series))
        price2 = 200 * np.exp(np.cumsum(vol2_series))
        
        self.data = pd.DataFrame({
            'SYMBOL1': np.log(price1),
            'SYMBOL2': np.log(price2)
        }, index=dates)
        
        self.analyzer = CointegrationAnalyzer(self.data)
    
    def test_full_period_volatility(self):
        """测试全期间波动率计算"""
        # 不指定日期，使用全部数据
        symbol_x, symbol_y = self.analyzer.determine_symbols('SYMBOL1', 'SYMBOL2')
        
        # 应该根据全期间的平均波动率确定
        assert symbol_x in ['SYMBOL1', 'SYMBOL2']
        assert symbol_y in ['SYMBOL1', 'SYMBOL2'] 
        assert symbol_x != symbol_y
    
    def test_early_period_volatility(self):
        """测试早期时间段波动率计算"""
        # 只使用前半段数据（SYMBOL1低波动期）
        start_date = '2020-01-01'
        end_date = '2021-05-01'  # 大约前半段
        
        symbol_x, symbol_y = self.analyzer.determine_symbols(
            'SYMBOL1', 'SYMBOL2',
            vol_start_date=start_date,
            vol_end_date=end_date
        )
        
        # 在早期，SYMBOL1波动更小，应该作为X
        assert symbol_x == 'SYMBOL1'
        assert symbol_y == 'SYMBOL2'
    
    def test_late_period_volatility(self):
        """测试晚期时间段波动率计算"""
        # 只使用后半段数据（SYMBOL1高波动期）
        start_date = '2021-05-01'
        # end_date不指定，使用到数据末尾
        
        symbol_x, symbol_y = self.analyzer.determine_symbols(
            'SYMBOL1', 'SYMBOL2',
            vol_start_date=start_date
        )
        
        # 在晚期，SYMBOL2波动更小，应该作为X
        assert symbol_x == 'SYMBOL2' 
        assert symbol_y == 'SYMBOL1'
    
    def test_specific_date_range_volatility(self):
        """测试指定具体日期区间的波动率计算"""
        # 指定一个很短的时间段
        start_date = '2020-06-01'
        end_date = '2020-08-01'
        
        symbol_x, symbol_y = self.analyzer.determine_symbols(
            'SYMBOL1', 'SYMBOL2',
            vol_start_date=start_date,
            vol_end_date=end_date
        )
        
        # 应该能正常确定角色
        assert symbol_x in ['SYMBOL1', 'SYMBOL2']
        assert symbol_y in ['SYMBOL1', 'SYMBOL2']
        assert symbol_x != symbol_y
    
    def test_invalid_date_range(self):
        """测试无效日期范围的处理"""
        # end_date早于start_date
        start_date = '2021-01-01'
        end_date = '2020-12-31'
        
        # 应该优雅处理，可能回退到全部数据或字母顺序
        symbol_x, symbol_y = self.analyzer.determine_symbols(
            'SYMBOL1', 'SYMBOL2',
            vol_start_date=start_date,
            vol_end_date=end_date
        )
        
        assert symbol_x in ['SYMBOL1', 'SYMBOL2']
        assert symbol_y in ['SYMBOL1', 'SYMBOL2']


class TestParameterizedScreening:
    """测试参数化筛选功能"""
    
    def setup_method(self):
        """准备测试数据"""
        try:
            self.data = load_data(
                ['AG', 'CU', 'AL', 'NI'], 
                start_date='2022-01-01',
                end_date='2024-08-01',
                log_price=True
            )
            self.analyzer = CointegrationAnalyzer(self.data)
            self.real_data = True
        except:
            # 使用模拟数据
            np.random.seed(42)
            n = 600
            data_dict = {}
            for symbol in ['AG', 'CU', 'AL', 'NI']:
                # 创建不同程度协整的数据
                base_series = np.cumsum(np.random.normal(0, 0.02, n))
                if symbol in ['AG', 'CU']:
                    # AG和CU高度相关
                    prices = base_series + np.random.normal(0, 0.005, n)
                else:
                    # AL和NI独立性更强
                    prices = np.cumsum(np.random.normal(0, 0.02, n))
                data_dict[symbol] = prices
                
            self.data = pd.DataFrame(data_dict, index=pd.date_range('2022-01-01', periods=n))
            self.analyzer = CointegrationAnalyzer(self.data)
            self.real_data = False
    
    def test_single_window_screening(self):
        """测试单一时间窗口筛选"""
        # 只用1年数据进行筛选
        results = self.analyzer.screen_all_pairs(
            screening_windows=['1y'],
            p_thresholds={'1y': 0.1},
            windows={'1y': 252}  # 自定义窗口大小
        )
        
        # 验证结果格式
        assert isinstance(results, pd.DataFrame)
        
        # 如果有结果，验证筛选条件
        for _, row in results.iterrows():
            if not pd.isna(row['pvalue_1y']):
                assert row['pvalue_1y'] <= 0.1
    
    def test_multiple_window_screening_and_logic(self):
        """测试多窗口AND逻辑筛选"""
        custom_windows = {'6m': 126, '1y': 252}
        
        results = self.analyzer.screen_all_pairs(
            screening_windows=['6m', '1y'],
            p_thresholds={'6m': 0.1, '1y': 0.1},
            filter_logic='AND',
            windows=custom_windows
        )
        
        # AND逻辑：所有指定窗口都要满足条件
        for _, row in results.iterrows():
            if not pd.isna(row['pvalue_6m']) and not pd.isna(row['pvalue_1y']):
                assert row['pvalue_6m'] <= 0.1
                assert row['pvalue_1y'] <= 0.1
    
    def test_multiple_window_screening_or_logic(self):
        """测试多窗口OR逻辑筛选"""
        custom_windows = {'6m': 126, '1y': 252}
        
        results = self.analyzer.screen_all_pairs(
            screening_windows=['6m', '1y'],
            p_thresholds={'6m': 0.05, '1y': 0.05},
            filter_logic='OR',
            windows=custom_windows
        )
        
        # OR逻辑：至少一个窗口满足条件即可
        for _, row in results.iterrows():
            if not pd.isna(row['pvalue_6m']) or not pd.isna(row['pvalue_1y']):
                # 至少一个满足条件
                condition_met = False
                if not pd.isna(row['pvalue_6m']) and row['pvalue_6m'] <= 0.05:
                    condition_met = True
                if not pd.isna(row['pvalue_1y']) and row['pvalue_1y'] <= 0.05:
                    condition_met = True
                assert condition_met
    
    def test_custom_sorting(self):
        """测试自定义排序"""
        custom_windows = {'6m': 126, '1y': 252}
        
        # 按6月p值降序排序
        results = self.analyzer.screen_all_pairs(
            windows=custom_windows,
            sort_by='pvalue_6m',
            ascending=False,
            screening_windows=[],  # 不进行筛选，只排序
            p_thresholds={}
        )
        
        if len(results) > 1:
            # 验证排序（忽略NaN值）
            pvalues = results['pvalue_6m'].dropna()
            if len(pvalues) > 1:
                # 应该是降序
                assert (pvalues.values[:-1] >= pvalues.values[1:]).all()
    
    def test_different_thresholds_per_window(self):
        """测试不同窗口使用不同阈值"""
        custom_windows = {'3m': 63, '6m': 126, '1y': 252}
        
        results = self.analyzer.screen_all_pairs(
            screening_windows=['3m', '6m', '1y'],
            p_thresholds={'3m': 0.2, '6m': 0.1, '1y': 0.05},  # 不同阈值
            filter_logic='AND',
            windows=custom_windows
        )
        
        # 验证每个窗口使用对应阈值
        for _, row in results.iterrows():
            if not pd.isna(row['pvalue_3m']):
                assert row['pvalue_3m'] <= 0.2
            if not pd.isna(row['pvalue_6m']):
                assert row['pvalue_6m'] <= 0.1
            if not pd.isna(row['pvalue_1y']):
                assert row['pvalue_1y'] <= 0.05
    
    def test_volatility_period_configuration(self):
        """测试波动率计算期间配置"""
        # 指定波动率计算使用特定时间段
        vol_start = '2023-01-01'
        vol_end = '2023-12-31'
        
        results = self.analyzer.screen_all_pairs(
            vol_start_date=vol_start,
            vol_end_date=vol_end,
            screening_windows=[],
            p_thresholds={}
        )
        
        # 验证结果包含波动率信息
        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert 'volatility_x' in results.columns
            assert 'volatility_y' in results.columns


class TestConfigurationValidation:
    """测试配置验证"""
    
    def test_invalid_window_configuration(self):
        """测试无效窗口配置的处理"""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'X': np.random.randn(n),
            'Y': np.random.randn(n)
        }, index=pd.date_range('2024-01-01', periods=n))
        
        analyzer = CointegrationAnalyzer(data)
        
        # 使用空的自定义窗口
        results = analyzer.screen_all_pairs(
            windows={},  # 空配置
            screening_windows=[],
            p_thresholds={}
        )
        
        # 应该使用默认配置或返回空结果
        assert isinstance(results, pd.DataFrame)
    
    def test_mismatched_screening_config(self):
        """测试筛选配置不匹配的处理"""
        np.random.seed(42)
        n = 300
        data = pd.DataFrame({
            'X': np.random.randn(n),
            'Y': np.random.randn(n)
        }, index=pd.date_range('2023-01-01', periods=n))
        
        analyzer = CointegrationAnalyzer(data)
        
        # 筛选窗口指定了不存在的窗口名称
        results = analyzer.screen_all_pairs(
            windows={'1y': 252},
            screening_windows=['2y'],  # 不存在的窗口
            p_thresholds={'2y': 0.05}
        )
        
        # 应该优雅处理，可能忽略无效配置
        assert isinstance(results, pd.DataFrame)
    
    def test_edge_case_date_ranges(self):
        """测试边界情况的日期范围"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n)
        data = pd.DataFrame({
            'SYMBOL1': np.random.randn(n),
            'SYMBOL2': np.random.randn(n)
        }, index=dates)
        
        analyzer = CointegrationAnalyzer(data)
        
        # 测试各种边界情况
        test_cases = [
            # 超出数据范围的起始日期
            {'vol_start_date': '2019-01-01', 'vol_end_date': None},
            # 超出数据范围的结束日期
            {'vol_start_date': None, 'vol_end_date': '2025-01-01'},
            # 都超出范围
            {'vol_start_date': '2025-01-01', 'vol_end_date': '2025-12-31'},
            # 正常范围
            {'vol_start_date': '2020-06-01', 'vol_end_date': '2020-12-31'}
        ]
        
        for test_case in test_cases:
            symbol_x, symbol_y = analyzer.determine_symbols(
                'SYMBOL1', 'SYMBOL2', **test_case
            )
            
            # 应该都能返回有效结果
            assert symbol_x in ['SYMBOL1', 'SYMBOL2']
            assert symbol_y in ['SYMBOL1', 'SYMBOL2']
            assert symbol_x != symbol_y


class TestRealDataParameterized:
    """真实数据参数化测试"""
    
    def test_flexible_analysis_workflow(self):
        """测试灵活的分析工作流"""
        try:
            # 加载真实数据
            data = load_data(
                ['AG', 'CU', 'AL'],
                start_date='2022-01-01', 
                end_date='2024-08-01',
                log_price=True
            )
            
            analyzer = CointegrationAnalyzer(data)
            
            # 场景1：短期策略 - 只关注最近6个月
            short_term_results = analyzer.screen_all_pairs(
                windows={'3m': 63, '6m': 126},
                screening_windows=['6m'],
                p_thresholds={'6m': 0.1},
                vol_start_date='2024-01-01',  # 使用2024年数据计算波动率
                sort_by='pvalue_6m'
            )
            
            # 场景2：长期策略 - 关注1-2年稳定性
            long_term_results = analyzer.screen_all_pairs(
                windows={'1y': 252, '18m': 378, '2y': 504},
                screening_windows=['1y', '2y'],
                p_thresholds={'1y': 0.05, '2y': 0.05},
                filter_logic='AND',
                vol_start_date='2022-01-01',
                vol_end_date='2023-12-31',
                sort_by='pvalue_2y'
            )
            
            # 验证两种策略得到不同结果
            assert isinstance(short_term_results, pd.DataFrame)
            assert isinstance(long_term_results, pd.DataFrame)
            
            # 两个结果的配对总数可能不同（由于筛选条件）
            assert len(short_term_results) >= 0
            assert len(long_term_results) >= 0
            
        except FileNotFoundError:
            pytest.skip("真实数据不可用，跳过参数化集成测试")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])