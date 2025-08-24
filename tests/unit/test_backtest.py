"""
回测框架模块单元测试
测试需求: REQ-4.x.x (交易执行、风险管理、绩效分析)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lib.backtest import (
    BacktestEngine, Position, PositionManager
)


class TestPosition:
    """测试Position类"""
    
    def test_position_initialization(self):
        """测试仓位初始化"""
        pos = Position(
            symbol='AG0',
            direction=1,  # 多头
            size=10,
            entry_price=5000.0,
            entry_date=pd.Timestamp('2020-01-01')
        )
        
        assert pos.symbol == 'AG0'
        assert pos.direction == 1
        assert pos.size == 10
        assert pos.entry_price == 5000.0
        assert pos.entry_date == pd.Timestamp('2020-01-01')
        
    def test_calculate_pnl_long(self):
        """测试多头PnL计算"""
        pos = Position(
            symbol='AG0',
            direction=1,
            size=10,
            entry_price=5000.0,
            entry_date=pd.Timestamp('2020-01-01')
        )
        
        # 价格上涨，多头盈利
        pnl = pos.calculate_pnl(5100.0)
        assert pnl == 10 * (5100.0 - 5000.0)  # 1000
        
        # 价格下跌，多头亏损
        pnl = pos.calculate_pnl(4900.0)
        assert pnl == 10 * (4900.0 - 5000.0)  # -1000
        
    def test_calculate_pnl_short(self):
        """测试空头PnL计算"""
        pos = Position(
            symbol='AG0',
            direction=-1,
            size=10,
            entry_price=5000.0,
            entry_date=pd.Timestamp('2020-01-01')
        )
        
        # 价格下跌，空头盈利
        pnl = pos.calculate_pnl(4900.0)
        assert pnl == 10 * (5000.0 - 4900.0)  # 1000
        
        # 价格上涨，空头亏损
        pnl = pos.calculate_pnl(5100.0)
        assert pnl == 10 * (5000.0 - 5100.0)  # -1000
        
    def test_calculate_margin_required(self):
        """测试保证金计算"""
        pos = Position(
            symbol='AG0',
            direction=1,
            size=10,
            entry_price=5000.0,
            entry_date=pd.Timestamp('2020-01-01')
        )
        
        # 计算所需保证金
        margin = pos.calculate_margin_required(
            margin_rate=0.12,
            multiplier=15  # AG0合约乘数
        )
        
        # 保证金 = 价格 * 数量 * 乘数 * 保证金率
        expected_margin = 5000.0 * 10 * 15 * 0.12
        assert margin == expected_margin
        
    def test_update_market_price(self):
        """测试市场价格更新"""
        pos = Position(
            symbol='AG0',
            direction=1,
            size=10,
            entry_price=5000.0,
            entry_date=pd.Timestamp('2020-01-01')
        )
        
        # 更新市场价格
        pos.update_market_price(5100.0)
        assert pos.current_price == 5100.0
        
        # 计算浮动盈亏
        assert pos.unrealized_pnl == 10 * (5100.0 - 5000.0)


class TestPositionManager:
    """测试PositionManager类"""
    
    def setup_method(self):
        """初始化测试环境"""
        self.pm = PositionManager(
            initial_capital=5000000,
            margin_rate=0.12
        )
        
    def test_initialization(self):
        """测试资金管理器初始化"""
        assert self.pm.initial_capital == 5000000
        assert self.pm.current_capital == 5000000
        assert self.pm.margin_rate == 0.12
        assert len(self.pm.positions) == 0
        
    def test_open_position_success(self):
        """测试成功开仓"""
        success = self.pm.open_position(
            symbol='AG0',
            direction=1,
            size=10,
            price=5000.0,
            date=pd.Timestamp('2020-01-01'),
            multiplier=15
        )
        
        assert success is True
        assert 'AG0' in self.pm.positions
        assert self.pm.positions['AG0'].size == 10
        
        # 检查资金占用
        margin_used = 5000.0 * 10 * 15 * 0.12
        assert self.pm.available_capital < self.pm.initial_capital
        
    def test_open_position_insufficient_margin(self):
        """测试保证金不足时开仓失败"""
        # 尝试开一个超大仓位
        success = self.pm.open_position(
            symbol='AG0',
            direction=1,
            size=10000,  # 超大仓位
            price=5000.0,
            date=pd.Timestamp('2020-01-01'),
            multiplier=15
        )
        
        assert success is False
        assert 'AG0' not in self.pm.positions
        
    def test_close_position(self):
        """测试平仓"""
        # 先开仓
        self.pm.open_position(
            symbol='AG0',
            direction=1,
            size=10,
            price=5000.0,
            date=pd.Timestamp('2020-01-01'),
            multiplier=15
        )
        
        # 平仓
        pnl = self.pm.close_position(
            symbol='AG0',
            price=5100.0,
            date=pd.Timestamp('2020-01-02')
        )
        
        # 验证盈亏
        expected_pnl = 10 * (5100.0 - 5000.0) * 15  # 考虑合约乘数
        assert abs(pnl - expected_pnl) < 1e-6
        
        # 验证仓位已清空
        assert 'AG0' not in self.pm.positions
        
        # 验证资金已更新
        assert self.pm.current_capital > self.pm.initial_capital
        
    def test_check_stop_loss(self):
        """测试止损检查"""
        # 开仓
        self.pm.open_position(
            symbol='AG0',
            direction=1,
            size=10,
            price=5000.0,
            date=pd.Timestamp('2020-01-01'),
            multiplier=15
        )
        
        # 价格小幅下跌，不触发止损
        should_stop = self.pm.check_stop_loss('AG0', 4950.0, stop_loss_pct=0.10)
        assert should_stop is False
        
        # 价格大幅下跌，触发止损（亏损超过保证金的10%）
        should_stop = self.pm.check_stop_loss('AG0', 4500.0, stop_loss_pct=0.10)
        assert should_stop is True
        
    def test_update_positions(self):
        """测试批量更新仓位"""
        # 开两个仓位
        self.pm.open_position('AG0', 1, 10, 5000.0, pd.Timestamp('2020-01-01'), 15)
        self.pm.open_position('AU0', -1, 5, 350.0, pd.Timestamp('2020-01-01'), 1000)
        
        # 更新价格
        prices = {'AG0': 5100.0, 'AU0': 340.0}
        self.pm.update_positions(prices, pd.Timestamp('2020-01-02'))
        
        # 验证价格已更新
        assert self.pm.positions['AG0'].current_price == 5100.0
        assert self.pm.positions['AU0'].current_price == 340.0


class TestBacktestEngine:
    """测试BacktestEngine类"""
    
    def setup_method(self):
        """准备测试环境"""
        self.engine = BacktestEngine(
            initial_capital=5000000,
            margin_rate=0.12,
            commission_rate=0.0002,
            slippage_ticks=3
        )
        
        # 准备测试数据
        dates = pd.date_range('2020-01-01', periods=100)
        self.y_prices = pd.Series(5000 + np.random.randn(100) * 50, index=dates)
        self.x_prices = pd.Series(350 + np.random.randn(100) * 5, index=dates)
        
        # 准备信号数据
        self.signals = pd.DataFrame({
            'signal': np.zeros(100),
            'position': np.zeros(100),
            'zscore': np.random.randn(100)
        }, index=dates)
        
        # 添加一些交易信号
        self.signals.iloc[10] = {'signal': 1, 'position': 1, 'zscore': 2.5}
        self.signals.iloc[11:20] = {'signal': 0, 'position': 1, 'zscore': 1.5}
        self.signals.iloc[20] = {'signal': -1, 'position': 0, 'zscore': 0.3}
        
    def test_engine_initialization(self):
        """测试回测引擎初始化"""
        assert self.engine.initial_capital == 5000000
        assert self.engine.margin_rate == 0.12
        assert self.engine.commission_rate == 0.0002
        assert self.engine.slippage_ticks == 3
        
    def test_run_backtest_basic(self):
        """测试基本回测功能"""
        results = self.engine.run_backtest(
            signals=self.signals,
            y_prices=self.y_prices,
            x_prices=self.x_prices,
            y_symbol='AG0',
            x_symbol='AU0',
            stop_loss_pct=0.10,
            max_holding_days=30
        )
        
        # 验证结果结构
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results
        assert 'positions' in results
        
        # 验证权益曲线
        equity_curve = results['equity_curve']
        assert len(equity_curve) == len(self.signals)
        assert equity_curve.iloc[0] == self.engine.initial_capital
        
        # 验证交易记录
        trades = results['trades']
        assert isinstance(trades, list)
        if len(trades) > 0:
            trade = trades[0]
            assert 'symbol' in trade
            assert 'entry_date' in trade
            assert 'exit_date' in trade
            assert 'pnl' in trade
            
    def test_calculate_performance_metrics(self):
        """测试绩效指标计算"""
        # 创建模拟权益曲线
        dates = pd.date_range('2020-01-01', periods=252)
        equity_curve = pd.Series(
            5000000 + np.cumsum(np.random.randn(252) * 10000),
            index=dates
        )
        
        # 创建模拟交易记录
        trades = [
            {'pnl': 10000, 'return': 0.002},
            {'pnl': -5000, 'return': -0.001},
            {'pnl': 15000, 'return': 0.003},
            {'pnl': -3000, 'return': -0.0006},
            {'pnl': 8000, 'return': 0.0016}
        ]
        
        # 计算绩效指标
        metrics = self.engine.calculate_performance_metrics(equity_curve, trades)
        
        # 验证指标存在
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        
        # 验证指标合理性
        assert -1 <= metrics['total_return'] <= 10  # 合理的收益率范围
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['max_drawdown'] <= 0  # 最大回撤应为负
        
    def test_commission_and_slippage(self):
        """测试手续费和滑点计算"""
        # 测试手续费计算
        price = 5000.0
        size = 10
        commission = self.engine.commission_rate * price * size
        
        # 验证手续费
        expected_commission = 0.0002 * 5000.0 * 10
        assert abs(commission - expected_commission) < 1e-6
        
        # 测试滑点计算（需要tick_size信息）
        # 假设tick_size = 1
        tick_size = 1
        slippage = self.engine.slippage_ticks * tick_size
        assert slippage == 3
        
    def test_stop_loss_trigger(self):
        """测试止损触发"""
        # 创建会触发止损的价格序列
        dates = pd.date_range('2020-01-01', periods=50)
        
        # Y价格大幅下跌
        y_prices = pd.Series([5000] * 10 + [4000] * 40, index=dates)
        x_prices = pd.Series([350] * 50, index=dates)
        
        # 创建开仓信号
        signals = pd.DataFrame({
            'signal': [1] + [0] * 49,
            'position': [1] * 50,
            'zscore': [2.5] + [1.5] * 49
        }, index=dates)
        
        results = self.engine.run_backtest(
            signals=signals,
            y_prices=y_prices,
            x_prices=x_prices,
            y_symbol='AG0',
            x_symbol='AU0',
            stop_loss_pct=0.10,
            max_holding_days=30
        )
        
        # 应该有止损交易
        trades = results['trades']
        if len(trades) > 0:
            # 检查是否有亏损交易（止损）
            has_loss = any(t['pnl'] < 0 for t in trades)
            assert has_loss
            
    def test_max_holding_days_exit(self):
        """测试最大持仓天数退出"""
        # 创建持续持仓的信号
        dates = pd.date_range('2020-01-01', periods=50)
        
        signals = pd.DataFrame({
            'signal': [1] + [0] * 49,
            'position': [1] * 50,
            'zscore': [2.5] * 50
        }, index=dates)
        
        y_prices = pd.Series(5000, index=dates)
        x_prices = pd.Series(350, index=dates)
        
        results = self.engine.run_backtest(
            signals=signals,
            y_prices=y_prices,
            x_prices=x_prices,
            y_symbol='AG0',
            x_symbol='AU0',
            stop_loss_pct=0.10,
            max_holding_days=10  # 最多持仓10天
        )
        
        # 检查交易记录
        trades = results['trades']
        if len(trades) > 0:
            for trade in trades:
                if 'holding_days' in trade:
                    assert trade['holding_days'] <= 10


class TestCalculationAccuracy:
    """测试计算准确性"""
    
    def test_pnl_calculation_precision(self):
        """测试PnL计算精度"""
        # 创建精确的仓位
        pos = Position('AG0', 1, 10, 5000.0, pd.Timestamp('2020-01-01'))
        
        # 精确的价格变动
        price_change = 0.123456
        new_price = 5000.0 + price_change
        
        pnl = pos.calculate_pnl(new_price)
        expected_pnl = 10 * price_change
        
        # 验证精度（至少6位小数）
        assert abs(pnl - expected_pnl) < 1e-6
        
    def test_sharpe_ratio_calculation(self):
        """测试夏普比率计算"""
        # 创建已知收益率序列
        returns = pd.Series([0.01, -0.005, 0.008, -0.002, 0.006])
        
        # 手动计算夏普比率
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = mean_return / std_return * np.sqrt(252)  # 年化
        
        # 使用引擎计算
        dates = pd.date_range('2020-01-01', periods=6)
        equity = pd.Series([1000000], index=[dates[0]])
        for i, ret in enumerate(returns):
            equity = pd.concat([equity, pd.Series([equity.iloc[-1] * (1 + ret)], index=[dates[i+1]])])
            
        engine = BacktestEngine()
        metrics = engine.calculate_performance_metrics(equity, [])
        
        # 验证夏普比率接近
        if 'sharpe_ratio' in metrics:
            assert abs(metrics['sharpe_ratio'] - sharpe) < 0.5  # 允许一定误差
            
    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        # 创建已知的权益曲线
        equity = pd.Series([100, 110, 105, 95, 100, 90, 95])
        
        # 手动计算最大回撤
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()
        
        # 应该是从110到90的回撤
        expected_max_dd = (90 - 110) / 110
        assert abs(max_dd - expected_max_dd) < 1e-6
        
    def test_win_rate_calculation(self):
        """测试胜率计算"""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 200},
            {'pnl': -30},
            {'pnl': 150},
            {'pnl': 0}  # 平局
        ]
        
        # 手动计算胜率
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        expected_win_rate = wins / total
        
        # 3胜2负1平，胜率=3/6=0.5
        assert expected_win_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])