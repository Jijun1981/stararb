"""
风险管理模块测试用例
严格对应需求文档 REQ-4.3
"""

import unittest
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lib.backtest.risk_manager import RiskManager, RiskConfig


@dataclass
class MockPosition:
    """模拟持仓对象"""
    position_id: str
    pair: str
    symbol_x: str
    symbol_y: str
    lots_x: int
    lots_y: int
    direction: str
    open_date: datetime
    open_price_x: float
    open_price_y: float
    margin: float
    beta: float
    allocated_capital: float  # 分配的资金


class TestRiskManager(unittest.TestCase):
    """测试风险管理模块"""
    
    def setUp(self):
        """初始化测试配置"""
        self.config = RiskConfig(
            stop_loss_pct=0.10,      # 10%止损
            max_holding_days=30,     # 30天最大持仓
            max_positions=20,        # 最大20个持仓
            margin_buffer=0.8        # 80%保证金缓冲
        )
        self.manager = RiskManager(self.config)
        
        # 创建测试用持仓
        self.test_position = MockPosition(
            position_id="TEST001",
            pair="CU-SN",
            symbol_x="CU",
            symbol_y="SN",
            lots_x=2,
            lots_y=5,
            direction="long",
            open_date=datetime(2024, 1, 1),
            open_price_x=60000,
            open_price_y=140000,
            margin=156000,
            beta=0.85,
            allocated_capital=250000  # 分配的25万资金
        )
    
    # ========== REQ-4.3.1: 止损管理 ==========
    
    def test_req_4_3_1_1_unrealized_pnl_calculation(self):
        """REQ-4.3.1.1: 计算浮动盈亏"""
        # 测试盈利情况
        current_prices = {'x': 62000, 'y': 142000}
        pnl = self.manager.calculate_unrealized_pnl(
            self.test_position,
            current_prices
        )
        
        # 验证PnL计算
        # X腿: (62000 - 60000) * 2 * 5(合约乘数) = 20000
        # Y腿（做空）: (140000 - 142000) * 5 * 1 = -10000
        # 总PnL = 20000 - 10000 = 10000（还要减去成本）
        self.assertIsNotNone(pnl)
        self.assertIn('gross_pnl', pnl)
        self.assertIn('net_pnl', pnl)
        self.assertIn('costs', pnl)
    
    def test_req_4_3_1_2_pnl_percentage_based_on_allocated(self):
        """REQ-4.3.1.2: 盈亏百分比基于分配资金"""
        # 亏损26000元
        current_pnl = -26000
        allocated_capital = 250000
        
        pnl_pct = self.manager.calculate_pnl_percentage(
            current_pnl,
            allocated_capital
        )
        
        # -26000 / 250000 = -10.4%
        self.assertAlmostEqual(pnl_pct, -0.104, places=3)
    
    def test_req_4_3_1_3_stop_loss_trigger(self):
        """REQ-4.3.1.3: 触发条件 pnl_pct <= -stop_loss_pct"""
        # 测试触发止损（亏损超过10%）
        current_pnl = -26000  # 亏损26000
        allocated_capital = 250000
        
        should_stop, reason = self.manager.check_stop_loss(
            self.test_position,
            current_pnl,
            allocated_capital
        )
        
        # -26000/250000 = -10.4% < -10%，应该触发
        self.assertTrue(should_stop)
        self.assertIn('Stop loss', reason)
        
        # 测试不触发止损（亏损未超过10%）
        current_pnl = -20000  # 亏损20000
        should_stop, reason = self.manager.check_stop_loss(
            self.test_position,
            current_pnl,
            allocated_capital
        )
        
        # -20000/250000 = -8% > -10%，不应该触发
        self.assertFalse(should_stop)
    
    def test_req_4_3_1_4_immediate_close_on_stop(self):
        """REQ-4.3.1.4: 触发后立即平仓，记录止损原因"""
        current_pnl = -30000
        allocated_capital = 250000
        
        should_stop, reason = self.manager.check_stop_loss(
            self.test_position,
            current_pnl,
            allocated_capital
        )
        
        self.assertTrue(should_stop)
        self.assertIn('stop loss', reason.lower())
        self.assertIn('-12.0%', reason)  # 应包含具体亏损百分比
    
    def test_req_4_3_1_5_stop_loss_statistics(self):
        """REQ-4.3.1.5: 统计止损次数和损失金额"""
        # 记录多次止损
        self.manager.record_stop_loss(self.test_position, -26000)
        self.manager.record_stop_loss(self.test_position, -30000)
        
        stats = self.manager.get_stop_loss_statistics()
        
        self.assertEqual(stats['stop_loss_count'], 2)
        self.assertEqual(stats['total_stop_loss'], -56000)
        self.assertEqual(stats['avg_stop_loss'], -28000)
    
    # ========== REQ-4.3.2: 时间止损 ==========
    
    def test_req_4_3_2_1_holding_days_calculation(self):
        """REQ-4.3.2.1: 计算持仓天数"""
        current_date = datetime(2024, 1, 15)
        
        holding_days = self.manager.calculate_holding_days(
            self.test_position,
            current_date
        )
        
        # 从1月1日到1月15日 = 14天
        self.assertEqual(holding_days, 14)
    
    def test_req_4_3_2_2_time_stop_trigger(self):
        """REQ-4.3.2.2: 触发条件 days >= max_holding_days"""
        # 测试触发时间止损（超过30天）
        current_date = datetime(2024, 2, 1)  # 31天后
        
        should_stop, reason = self.manager.check_time_stop(
            self.test_position,
            current_date
        )
        
        self.assertTrue(should_stop)
        self.assertIn('Time stop', reason)
        
        # 测试不触发（未超过30天）
        current_date = datetime(2024, 1, 29)  # 28天后
        
        should_stop, reason = self.manager.check_time_stop(
            self.test_position,
            current_date
        )
        
        self.assertFalse(should_stop)
    
    def test_req_4_3_2_3_forced_close_reason(self):
        """REQ-4.3.2.3: 触发后强制平仓，记录时间止损原因"""
        current_date = datetime(2024, 2, 5)  # 35天后
        
        should_stop, reason = self.manager.check_time_stop(
            self.test_position,
            current_date
        )
        
        self.assertTrue(should_stop)
        self.assertIn('time stop', reason.lower())
        self.assertIn('35', reason)  # 应包含具体天数
        self.assertIn('30', reason)  # 应包含限制天数
    
    def test_req_4_3_2_4_time_stop_statistics(self):
        """REQ-4.3.2.4: 统计时间止损次数"""
        # 记录多次时间止损
        self.manager.record_time_stop(self.test_position)
        self.manager.record_time_stop(self.test_position)
        self.manager.record_time_stop(self.test_position)
        
        stats = self.manager.get_time_stop_statistics()
        
        self.assertEqual(stats['time_stop_count'], 3)
        self.assertIn('CU-SN', stats['pairs_time_stopped'])
    
    # ========== REQ-4.3.3: 保证金监控 ==========
    
    def test_req_4_3_3_1_used_margin_calculation(self):
        """REQ-4.3.3.1: 实时计算已用保证金"""
        positions = {
            'pos1': MockPosition(
                position_id="pos1",
                pair="CU-SN",
                symbol_x="CU",
                symbol_y="SN",
                lots_x=2,
                lots_y=5,
                direction="long",
                open_date=datetime(2024, 1, 1),
                open_price_x=60000,
                open_price_y=140000,
                margin=156000,
                beta=0.85,
                allocated_capital=250000
            ),
            'pos2': MockPosition(
                position_id="pos2",
                pair="AL-ZN",
                symbol_x="AL",
                symbol_y="ZN",
                lots_x=3,
                lots_y=4,
                direction="short",
                open_date=datetime(2024, 1, 2),
                open_price_x=20000,
                open_price_y=25000,
                margin=80000,
                beta=0.75,
                allocated_capital=250000
            )
        }
        
        used_margin = self.manager.calculate_used_margin(positions)
        
        # 156000 + 80000 = 236000
        self.assertEqual(used_margin, 236000)
    
    def test_req_4_3_3_2_available_margin(self):
        """REQ-4.3.3.2: 计算可用保证金"""
        total_capital = 5000000
        used_margin = 236000
        
        available_margin = self.manager.calculate_available_margin(
            total_capital,
            used_margin
        )
        
        # 5000000 - 236000 = 4764000
        self.assertEqual(available_margin, 4764000)
    
    def test_req_4_3_3_3_margin_check_with_buffer(self):
        """REQ-4.3.3.3: 新开仓前检查 required <= available × buffer"""
        available_margin = 100000
        required_margin = 75000
        
        # 75000 <= 100000 * 0.8 = 80000，应该通过
        can_open = self.manager.check_margin_adequacy(
            available_margin,
            required_margin
        )
        self.assertTrue(can_open)
        
        # 85000 > 100000 * 0.8 = 80000，不应该通过
        can_open = self.manager.check_margin_adequacy(
            available_margin,
            85000
        )
        self.assertFalse(can_open)
    
    def test_req_4_3_3_4_buffer_configuration(self):
        """REQ-4.3.3.4: buffer默认0.8（留20%缓冲）"""
        # 测试默认buffer
        self.assertEqual(self.manager.margin_buffer, 0.8)
        
        # 测试自定义buffer
        custom_config = RiskConfig(margin_buffer=0.7)
        custom_manager = RiskManager(custom_config)
        self.assertEqual(custom_manager.margin_buffer, 0.7)
    
    def test_req_4_3_3_5_reject_when_insufficient(self):
        """REQ-4.3.3.5: 保证金不足时拒绝开仓"""
        available_margin = 50000
        required_margin = 60000
        
        can_open = self.manager.check_margin_adequacy(
            available_margin,
            required_margin
        )
        
        self.assertFalse(can_open)
        
        # 获取拒绝原因
        result = self.manager.check_margin_with_reason(
            available_margin,
            required_margin
        )
        
        self.assertFalse(result['can_open'])
        self.assertIn('Insufficient margin', result['reason'])
        # shortfall = 60000 - 50000*0.8 = 20000
        self.assertEqual(result['shortfall'], 20000)
    
    # ========== 综合风险检查 ==========
    
    def test_comprehensive_risk_check(self):
        """综合测试：同时检查所有风险条件"""
        current_date = datetime(2024, 1, 15)
        current_prices = {'x': 58000, 'y': 145000}
        current_pnl = -22000
        allocated_capital = 250000
        
        # 执行综合风险检查
        risk_status = self.manager.check_all_risks(
            position=self.test_position,
            current_date=current_date,
            current_prices=current_prices,
            current_pnl=current_pnl,
            allocated_capital=allocated_capital
        )
        
        # 验证返回结果结构
        self.assertIn('should_close', risk_status)
        self.assertIn('close_reason', risk_status)
        self.assertIn('risk_metrics', risk_status)
        
        # 验证风险指标
        metrics = risk_status['risk_metrics']
        self.assertIn('pnl_pct', metrics)
        self.assertIn('holding_days', metrics)
        self.assertIn('stop_loss_triggered', metrics)
        self.assertIn('time_stop_triggered', metrics)
    
    def test_position_limit_check(self):
        """测试最大持仓数量限制"""
        # 创建20个持仓（达到限制）
        positions = {
            f'pos{i}': MockPosition(
                position_id=f'pos{i}',
                pair=f'pair{i}',
                symbol_x='X',
                symbol_y='Y',
                lots_x=1,
                lots_y=1,
                direction='long',
                open_date=datetime(2024, 1, 1),
                open_price_x=10000,
                open_price_y=10000,
                margin=10000,
                beta=1.0,
                allocated_capital=50000
            )
            for i in range(20)
        }
        
        # 应该达到限制
        can_add = self.manager.check_position_limit(positions)
        self.assertFalse(can_add)
        
        # 删除一个持仓
        del positions['pos19']
        
        # 应该可以添加
        can_add = self.manager.check_position_limit(positions)
        self.assertTrue(can_add)
    
    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        positions = {
            'pos1': self.test_position
        }
        current_prices = {'x': 59000, 'y': 141000}
        
        metrics = self.manager.calculate_portfolio_risk_metrics(
            positions,
            current_prices,
            total_capital=5000000
        )
        
        # 验证返回的风险指标
        self.assertIn('total_positions', metrics)
        self.assertIn('total_margin_used', metrics)
        self.assertIn('margin_utilization', metrics)
        self.assertIn('positions_at_risk', metrics)
        self.assertIn('max_position_loss', metrics)
        self.assertIn('portfolio_var_95', metrics)


if __name__ == '__main__':
    unittest.main()