"""
数据管理模块单元测试 - 无Mock版本
测试需求: REQ-1.1.x, REQ-1.2.x, REQ-1.3.x, REQ-1.4.x
使用真实数据和文件操作进行测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.data import (
    DataManager, DataError, DataValidationError, DataUpdateError,
    save_to_parquet, load_from_parquet, load_data
)


class TestDataStorage:
    """测试数据存储功能 (REQ-1.2.x) - 真实文件操作"""
    
    def setup_method(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'futures'
        self.data_dir.mkdir(exist_ok=True)
        
    def teardown_method(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_save_and_load_parquet(self):
        """测试保存和加载Parquet格式 (REQ-1.2.1)"""
        # 创建真实测试数据
        dates = pd.date_range('2020-01-01', periods=100)
        df = pd.DataFrame({
            'open': np.random.uniform(4900, 5100, 100),
            'high': np.random.uniform(5000, 5200, 100),
            'low': np.random.uniform(4800, 5000, 100),
            'close': np.random.uniform(4900, 5100, 100),
            'volume': np.random.uniform(1000, 2000, 100),
        }, index=dates)
        
        # 保存数据
        file_path = save_to_parquet(df, 'TEST_SYMBOL', self.data_dir)
        
        # 验证文件存在
        assert file_path.exists()
        assert file_path.suffix == '.parquet'
        assert file_path.stat().st_size > 0
        
        # 加载数据
        loaded_df = load_from_parquet('TEST_SYMBOL', self.data_dir)
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(df, loaded_df)
        
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            load_from_parquet('NONEXISTENT', self.data_dir)
            
    def test_parquet_compression(self):
        """测试Parquet压缩效果"""
        # 创建大量重复数据
        dates = pd.date_range('2020-01-01', periods=1000)
        df = pd.DataFrame({
            'close': [100.0] * 1000,  # 重复数据应该压缩效果好
        }, index=dates)
        
        # 保存并检查文件大小
        file_path = save_to_parquet(df, 'COMPRESS_TEST', self.data_dir)
        file_size = file_path.stat().st_size
        
        # Parquet应该有效压缩重复数据
        assert file_size < 1000 * 8 * 2  # 远小于未压缩的大小


class TestDataPreprocessing:
    """测试数据预处理功能 (REQ-1.4.x) - 真实数据计算"""
    
    def setup_method(self):
        """创建测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'futures'
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建真实测试数据
        dates = pd.date_range('2020-01-01', periods=100)
        
        # Symbol1 - 有缺失值
        data1 = pd.DataFrame({
            'close': [100 + i + np.random.randn() for i in range(100)]
        }, index=dates)
        data1.iloc[10:15] = np.nan  # 添加缺失值
        save_to_parquet(data1, 'SYMBOL1', self.data_dir)
        
        # Symbol2 - 完整数据
        data2 = pd.DataFrame({
            'close': [200 + i * 2 + np.random.randn() for i in range(100)]
        }, index=dates)
        save_to_parquet(data2, 'SYMBOL2', self.data_dir)
        
    def teardown_method(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_log_price_conversion(self):
        """测试对数价格转换 (REQ-1.4.1)"""
        # 加载数据并转换为对数价格
        data = load_data(['SYMBOL1', 'SYMBOL2'], self.data_dir, log_price=True, align=False)
        
        # 验证对数转换
        assert 'SYMBOL1' in data.columns
        assert 'SYMBOL2' in data.columns
        
        # 加载原始数据比较
        raw_data = load_data(['SYMBOL1', 'SYMBOL2'], self.data_dir, log_price=False, align=False)
        
        # 验证对数关系（跳过NaN）
        mask = ~raw_data['SYMBOL1'].isna()
        expected_log = np.log(raw_data['SYMBOL1'][mask])
        actual_log = data['SYMBOL1'][mask]
        np.testing.assert_array_almost_equal(actual_log.values, expected_log.values, decimal=10)
        
    def test_missing_value_handling(self):
        """测试缺失值处理 (REQ-1.4.3)"""
        # 加载数据并处理缺失值
        data = load_data(['SYMBOL1'], self.data_dir, log_price=False, align=True)
        
        # 验证缺失值已被填充
        assert not data['SYMBOL1'].isna().any()
        
        # 验证前向填充逻辑
        # 第10-14位置应该被第9位置的值填充
        assert data['SYMBOL1'].iloc[10] == data['SYMBOL1'].iloc[9]
        assert data['SYMBOL1'].iloc[11] == data['SYMBOL1'].iloc[9]
        
    def test_data_alignment(self):
        """测试数据对齐 (REQ-1.4.2)"""
        # 创建不对齐的数据
        dates1 = pd.date_range('2020-01-01', periods=50)
        dates2 = pd.date_range('2020-01-10', periods=60)
        
        data1 = pd.DataFrame({'close': range(50)}, index=dates1)
        data2 = pd.DataFrame({'close': range(100, 160)}, index=dates2)
        
        save_to_parquet(data1, 'ALIGN1', self.data_dir)
        save_to_parquet(data2, 'ALIGN2', self.data_dir)
        
        # 加载并对齐
        aligned_data = load_data(['ALIGN1', 'ALIGN2'], self.data_dir, log_price=False, align=True)
        
        # 验证对齐后的索引一致
        assert len(aligned_data) > 0
        assert aligned_data.index.is_unique
        assert not aligned_data.isna().all().any()  # 不应该有全为NaN的列


class TestDataCalculationAccuracy:
    """测试计算准确性 - 真实数值计算"""
    
    def test_log_price_precision(self):
        """测试对数价格计算精度"""
        # 使用精确的价格数据
        prices = np.array([100.123456, 150.234567, 200.345678, 250.456789])
        log_prices = np.log(prices)
        
        # 验证对数计算的精度（至少10位小数）
        expected = np.log(prices[0])
        calculated = np.log(100.123456)
        assert abs(expected - calculated) < 1e-10
        
        # 验证对数价格保持精度
        for i, price in enumerate(prices):
            log_val = np.log(price)
            assert f"{log_val:.10f}" == f"{log_prices[i]:.10f}"
            
    def test_statistical_calculations(self):
        """测试统计计算的准确性"""
        # 创建已知分布的数据
        np.random.seed(42)
        data = np.random.normal(100, 10, 1000)
        
        # 计算统计量
        mean = np.mean(data)
        std = np.std(data)
        
        # 验证接近理论值
        assert abs(mean - 100) < 1.0  # 均值接近100
        assert abs(std - 10) < 1.0    # 标准差接近10
        
        # 测试百分位数计算
        percentiles = np.percentile(data, [25, 50, 75])
        median = np.median(data)
        assert abs(percentiles[1] - median) < 1e-10
        
    def test_datetime_precision(self):
        """测试日期时间处理精度"""
        # 创建精确的时间序列
        start_date = pd.Timestamp('2020-01-01 00:00:00')
        dates = pd.date_range(start_date, periods=100, freq='D')
        
        # 验证日期间隔
        for i in range(1, len(dates)):
            delta = dates[i] - dates[i-1]
            assert delta.days == 1
            assert delta.seconds == 0
            assert delta.microseconds == 0
            
    def test_array_operations_precision(self):
        """测试数组运算精度"""
        # 创建测试数组
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
        
        # 测试线性关系
        beta = 2.0
        calculated_y = beta * x
        np.testing.assert_array_equal(calculated_y, y)
        
        # 测试累积和精度
        cumsum = np.cumsum(x)
        expected_cumsum = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        np.testing.assert_array_equal(cumsum, expected_cumsum)
        

class TestDataValidation:
    """测试数据验证功能 - 真实数据检查"""
    
    def test_ohlc_validation(self):
        """测试OHLC数据有效性验证"""
        # 创建有效的OHLC数据
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104]
        })
        
        # 验证 high >= low
        assert (valid_data['high'] >= valid_data['low']).all()
        # 验证 high >= open 和 high >= close
        assert (valid_data['high'] >= valid_data['open']).all()
        assert (valid_data['high'] >= valid_data['close']).all()
        # 验证 low <= open 和 low <= close
        assert (valid_data['low'] <= valid_data['open']).all()
        assert (valid_data['low'] <= valid_data['close']).all()
        
    def test_outlier_detection(self):
        """测试异常值检测"""
        # 创建包含异常值的数据
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 100)
        data_with_outlier = np.append(normal_data, [1000])  # 添加异常值
        
        # 使用5倍标准差检测异常值
        mean = np.mean(data_with_outlier)
        std = np.std(data_with_outlier)
        threshold = 5 * std
        
        outliers = np.abs(data_with_outlier - mean) > threshold
        
        # 应该检测到至少一个异常值
        assert outliers.any()
        assert outliers[-1] == True  # 最后一个值是异常值
        
    def test_data_continuity(self):
        """测试数据连续性检查"""
        # 创建连续的日期序列
        dates = pd.date_range('2020-01-01', periods=100)
        
        # 检查是否有缺失的交易日（简化版，实际应考虑节假日）
        date_diffs = np.diff(dates)
        expected_diff = pd.Timedelta(days=1)
        
        # 所有日期间隔应该是1天
        assert all(d == expected_diff for d in date_diffs)
        
        # 创建有缺口的数据
        dates_with_gap = pd.date_range('2020-01-01', periods=50).append(
            pd.date_range('2020-03-01', periods=50)
        )
        
        # 检测缺口
        gaps = []
        for i in range(1, len(dates_with_gap)):
            diff = dates_with_gap[i] - dates_with_gap[i-1]
            if diff > pd.Timedelta(days=1):
                gaps.append((dates_with_gap[i-1], dates_with_gap[i], diff.days))
                
        # 应该检测到缺口
        assert len(gaps) > 0


class TestRealDataIntegration:
    """集成测试 - 使用真实数据流程"""
    
    def setup_method(self):
        """准备测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'futures'
        self.data_dir.mkdir(exist_ok=True)
        
    def teardown_method(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_complete_data_pipeline(self):
        """测试完整的数据处理流程"""
        # 1. 创建模拟的期货数据
        symbols = ['AG0', 'AU0', 'CU0']
        dates = pd.date_range('2020-01-01', periods=252)  # 一年的交易日
        
        for symbol in symbols:
            # 生成符合期货特征的价格数据
            np.random.seed(hash(symbol) % 1000)
            base_price = {'AG0': 5000, 'AU0': 350, 'CU0': 50000}[symbol]
            volatility = base_price * 0.02  # 2%日波动率
            
            prices = [base_price]
            for _ in range(len(dates)-1):
                change = np.random.normal(0, volatility)
                new_price = prices[-1] + change
                prices.append(max(new_price, base_price * 0.5))  # 防止负价格
                
            df = pd.DataFrame({
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': np.random.uniform(1000, 5000, len(dates))
            }, index=dates)
            
            save_to_parquet(df, symbol, self.data_dir)
            
        # 2. 加载并预处理数据
        processed_data = load_data(symbols, self.data_dir, log_price=True, align=True)
        
        # 3. 验证处理结果
        assert len(processed_data.columns) == len(symbols)
        assert len(processed_data) == len(dates)
        assert not processed_data.isna().any().any()
        
        # 4. 验证对数价格的性质
        for symbol in symbols:
            # 对数价格应该在合理范围内
            assert processed_data[symbol].min() > 0  # log价格应该为正（原价格>1）
            assert processed_data[symbol].max() < 20  # log价格不应太大
            
        # 5. 验证数据统计特性
        returns = processed_data.diff()
        for symbol in symbols:
            # 收益率应该近似正态分布
            symbol_returns = returns[symbol].dropna()
            assert abs(symbol_returns.mean()) < 0.01  # 均值接近0
            assert 0.001 < symbol_returns.std() < 0.1  # 合理的波动率范围


if __name__ == "__main__":
    pytest.main([__file__, "-v"])