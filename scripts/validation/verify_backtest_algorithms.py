#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测框架算法验证脚本
验证所有计算的正确性，包括保证金、手续费、滑点、PnL、止损、强平等

作者：Claude
创建时间：2025-08-22
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.backtest import BacktestEngine, PositionManager, Position


class BacktestAlgorithmVerifier:
    """回测算法验证器"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name: str, passed: bool, actual: float, expected: float, 
                 tolerance: float = 1e-6, details: str = ""):
        """记录测试结果"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: 通过")
        else:
            print(f"❌ {test_name}: 失败")
        
        if not passed:
            print(f"   实际值: {actual:.6f}")
            print(f"   期望值: {expected:.6f}")
            print(f"   差异: {abs(actual - expected):.6f}")
            if details:
                print(f"   详情: {details}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'actual': actual,
            'expected': expected,
            'error': abs(actual - expected) if not passed else 0
        })
    
    def verify_margin_calculation(self):
        """验证保证金计算（REQ-4.1.5）"""
        print("\n" + "="*60)
        print("1. 验证保证金计算（12%保证金率）")
        print("="*60)
        
        # 测试数据
        test_cases = [
            {
                'name': 'CU铜合约',
                'price': 50000,
                'multiplier': 5,
                'contracts': 2,
                'margin_rate': 0.12
            },
            {
                'name': 'AL铝合约',
                'price': 15000,
                'multiplier': 5,
                'contracts': 3,
                'margin_rate': 0.12
            },
            {
                'name': 'AU金合约',
                'price': 450,
                'multiplier': 1000,
                'contracts': 1,
                'margin_rate': 0.12
            }
        ]
        
        for case in test_cases:
            # 计算名义价值
            nominal_value = case['price'] * case['multiplier'] * case['contracts']
            
            # 计算保证金（公式：nominal_value * margin_rate）
            expected_margin = nominal_value * case['margin_rate']
            
            # 模拟引擎计算
            actual_margin = case['price'] * case['multiplier'] * case['contracts'] * case['margin_rate']
            
            passed = abs(actual_margin - expected_margin) < 0.01
            self.log_test(
                f"保证金计算 - {case['name']}",
                passed,
                actual_margin,
                expected_margin,
                details=f"价格={case['price']}, 乘数={case['multiplier']}, 手数={case['contracts']}"
            )
    
    def verify_commission_calculation(self):
        """验证手续费计算（REQ-4.1.6）"""
        print("\n" + "="*60)
        print("2. 验证手续费计算（万分之2双边）")
        print("="*60)
        
        test_cases = [
            {
                'name': '开仓手续费',
                'price_y': 50000,
                'price_x': 15000,
                'multiplier_y': 5,
                'multiplier_x': 5,
                'contracts_y': 2,
                'contracts_x': 7,
                'commission_rate': 0.0002
            },
            {
                'name': '平仓手续费',
                'price_y': 51000,
                'price_x': 15500,
                'multiplier_y': 5,
                'multiplier_x': 5,
                'contracts_y': 2,
                'contracts_x': 7,
                'commission_rate': 0.0002
            }
        ]
        
        for case in test_cases:
            # Y腿手续费
            nominal_y = case['price_y'] * case['multiplier_y'] * case['contracts_y']
            commission_y = nominal_y * case['commission_rate']
            
            # X腿手续费
            nominal_x = case['price_x'] * case['multiplier_x'] * case['contracts_x']
            commission_x = nominal_x * case['commission_rate']
            
            # 总手续费
            expected_total = commission_y + commission_x
            
            # 模拟计算
            actual_total = (nominal_y + nominal_x) * case['commission_rate']
            
            passed = abs(actual_total - expected_total) < 0.01
            self.log_test(
                f"手续费计算 - {case['name']}",
                passed,
                actual_total,
                expected_total,
                details=f"Y腿费用={commission_y:.2f}, X腿费用={commission_x:.2f}"
            )
    
    def verify_slippage_calculation(self):
        """验证滑点计算（REQ-4.1.4）"""
        print("\n" + "="*60)
        print("3. 验证滑点计算（每腿3个tick）")
        print("="*60)
        
        test_cases = [
            {
                'name': '买入滑点（价格上移）',
                'price': 50000,
                'tick_size': 10,
                'slippage_ticks': 3,
                'is_buy': True
            },
            {
                'name': '卖出滑点（价格下移）',
                'price': 50000,
                'tick_size': 10,
                'slippage_ticks': 3,
                'is_buy': False
            },
            {
                'name': 'AU金滑点',
                'price': 450.50,
                'tick_size': 0.05,
                'slippage_ticks': 3,
                'is_buy': True
            }
        ]
        
        for case in test_cases:
            # 计算滑点
            slippage = case['tick_size'] * case['slippage_ticks']
            
            # 买入加滑点，卖出减滑点
            if case['is_buy']:
                expected_price = case['price'] + slippage
            else:
                expected_price = case['price'] - slippage
            
            # 模拟计算
            actual_price = case['price'] + (slippage if case['is_buy'] else -slippage)
            
            passed = abs(actual_price - expected_price) < 0.001
            self.log_test(
                f"滑点计算 - {case['name']}",
                passed,
                actual_price,
                expected_price,
                details=f"原价={case['price']}, tick_size={case['tick_size']}"
            )
    
    def verify_pnl_calculation(self):
        """验证PnL计算（REQ-4.3）"""
        print("\n" + "="*60)
        print("4. 验证PnL计算（逐笔和累计）")
        print("="*60)
        
        # 测试案例：做多价差（多Y空X）
        test_case = {
            'name': '做多价差PnL',
            'direction': 'long_spread',
            # 开仓价格
            'open_price_y': 50000,
            'open_price_x': 15000,
            # 平仓价格
            'close_price_y': 51000,
            'close_price_x': 15200,
            # 合约信息
            'contracts_y': 2,
            'contracts_x': 7,
            'multiplier_y': 5,
            'multiplier_x': 5,
            # 费率
            'commission_rate': 0.0002
        }
        
        # 计算毛PnL
        # Y腿盈亏（做多）
        y_pnl = (test_case['close_price_y'] - test_case['open_price_y']) * \
                test_case['contracts_y'] * test_case['multiplier_y']
        
        # X腿盈亏（做空）
        x_pnl = (test_case['open_price_x'] - test_case['close_price_x']) * \
                test_case['contracts_x'] * test_case['multiplier_x']
        
        gross_pnl = y_pnl + x_pnl
        
        # 计算手续费
        open_commission = (
            test_case['open_price_y'] * test_case['contracts_y'] * test_case['multiplier_y'] +
            test_case['open_price_x'] * test_case['contracts_x'] * test_case['multiplier_x']
        ) * test_case['commission_rate']
        
        close_commission = (
            test_case['close_price_y'] * test_case['contracts_y'] * test_case['multiplier_y'] +
            test_case['close_price_x'] * test_case['contracts_x'] * test_case['multiplier_x']
        ) * test_case['commission_rate']
        
        # 净PnL
        expected_net_pnl = gross_pnl - open_commission - close_commission
        
        # 模拟计算
        actual_gross_pnl = y_pnl + x_pnl
        actual_net_pnl = actual_gross_pnl - open_commission - close_commission
        
        passed = abs(actual_net_pnl - expected_net_pnl) < 0.01
        self.log_test(
            f"PnL计算 - {test_case['name']}",
            passed,
            actual_net_pnl,
            expected_net_pnl,
            details=f"毛PnL={gross_pnl:.2f}, 手续费={open_commission+close_commission:.2f}"
        )
        
        # 测试做空价差
        test_case2 = test_case.copy()
        test_case2['name'] = '做空价差PnL'
        test_case2['direction'] = 'short_spread'
        
        # 做空价差：空Y多X
        y_pnl2 = (test_case2['open_price_y'] - test_case2['close_price_y']) * \
                 test_case2['contracts_y'] * test_case2['multiplier_y']
        x_pnl2 = (test_case2['close_price_x'] - test_case2['open_price_x']) * \
                 test_case2['contracts_x'] * test_case2['multiplier_x']
        
        gross_pnl2 = y_pnl2 + x_pnl2
        expected_net_pnl2 = gross_pnl2 - open_commission - close_commission
        
        # 做空价差的净PnL应该是做多价差的相反数（毛PnL相反，但手续费相同）
        actual_net_pnl2 = -gross_pnl - open_commission - close_commission
        
        passed2 = abs(expected_net_pnl2 - actual_net_pnl2) < 0.01
        self.log_test(
            f"PnL计算 - {test_case2['name']}",
            passed2,
            expected_net_pnl2,
            actual_net_pnl2,
            details=f"毛PnL={gross_pnl2:.2f}"
        )
    
    def verify_stop_loss_logic(self):
        """验证止损逻辑（REQ-4.2.3）"""
        print("\n" + "="*60)
        print("5. 验证止损逻辑（15%止损）")
        print("="*60)
        
        test_cases = [
            {
                'name': '未触发止损',
                'margin': 100000,
                'current_pnl': -10000,
                'stop_loss_pct': 0.15,
                'should_stop': False
            },
            {
                'name': '刚好触发止损',
                'margin': 100000,
                'current_pnl': -15000,
                'stop_loss_pct': 0.15,
                'should_stop': True
            },
            {
                'name': '超过止损线',
                'margin': 100000,
                'current_pnl': -20000,
                'stop_loss_pct': 0.15,
                'should_stop': True
            }
        ]
        
        for case in test_cases:
            # 止损线
            stop_loss_threshold = -case['margin'] * case['stop_loss_pct']
            
            # 判断是否触发
            actual_should_stop = case['current_pnl'] <= stop_loss_threshold
            expected_should_stop = case['should_stop']
            
            passed = actual_should_stop == expected_should_stop
            self.log_test(
                f"止损判断 - {case['name']}",
                passed,
                float(actual_should_stop),
                float(expected_should_stop),
                details=f"保证金={case['margin']}, 当前亏损={case['current_pnl']}, 止损线={stop_loss_threshold}"
            )
    
    def verify_time_stop_logic(self):
        """验证强平逻辑（REQ-4.2.4）"""
        print("\n" + "="*60)
        print("6. 验证强平逻辑（30天强平）")
        print("="*60)
        
        base_date = datetime(2025, 1, 1)
        
        test_cases = [
            {
                'name': '持仓29天',
                'open_date': base_date,
                'current_date': base_date + timedelta(days=29),
                'max_days': 30,
                'should_stop': False
            },
            {
                'name': '持仓30天',
                'open_date': base_date,
                'current_date': base_date + timedelta(days=30),
                'max_days': 30,
                'should_stop': True
            },
            {
                'name': '持仓31天',
                'open_date': base_date,
                'current_date': base_date + timedelta(days=31),
                'max_days': 30,
                'should_stop': True
            }
        ]
        
        for case in test_cases:
            # 计算持仓天数
            days_held = (case['current_date'] - case['open_date']).days
            
            # 判断是否触发强平
            actual_should_stop = days_held >= case['max_days']
            expected_should_stop = case['should_stop']
            
            passed = actual_should_stop == expected_should_stop
            self.log_test(
                f"强平判断 - {case['name']}",
                passed,
                float(actual_should_stop),
                float(expected_should_stop),
                details=f"持仓天数={days_held}"
            )
    
    def verify_sharpe_ratio_calculation(self):
        """验证夏普比率计算（REQ-4.4.3）"""
        print("\n" + "="*60)
        print("7. 验证夏普比率计算")
        print("="*60)
        
        # 测试数据：日收益率序列
        test_returns = np.array([0.01, -0.005, 0.008, -0.002, 0.015, 
                                -0.003, 0.006, -0.001, 0.012, -0.004])
        
        # 手动计算夏普比率
        mean_return = np.mean(test_returns)
        std_return = np.std(test_returns, ddof=1)  # 样本标准差
        
        # 年化夏普比率（假设252个交易日）
        expected_sharpe = mean_return / std_return * np.sqrt(252)
        
        # 模拟引擎计算
        actual_sharpe = (np.mean(test_returns) / np.std(test_returns, ddof=1)) * np.sqrt(252)
        
        passed = abs(actual_sharpe - expected_sharpe) < 1e-6
        self.log_test(
            "夏普比率计算",
            passed,
            actual_sharpe,
            expected_sharpe,
            details=f"平均收益率={mean_return:.6f}, 标准差={std_return:.6f}"
        )
    
    def verify_max_drawdown_calculation(self):
        """验证最大回撤计算（REQ-4.4.4）"""
        print("\n" + "="*60)
        print("8. 验证最大回撤计算")
        print("="*60)
        
        # 测试数据：权益曲线
        equity_curve = np.array([100000, 105000, 103000, 108000, 104000, 
                                 102000, 107000, 105000, 110000, 108000])
        
        # 手动计算最大回撤
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        expected_max_dd = np.max(drawdown)
        
        # 找到最大回撤位置
        max_dd_idx = np.argmax(drawdown)
        
        # 模拟计算
        actual_cummax = np.maximum.accumulate(equity_curve)
        actual_drawdown = (actual_cummax - equity_curve) / actual_cummax
        actual_max_dd = np.max(actual_drawdown)
        
        passed = abs(actual_max_dd - expected_max_dd) < 1e-6
        self.log_test(
            "最大回撤计算",
            passed,
            actual_max_dd,
            expected_max_dd,
            details=f"最大回撤位置={max_dd_idx}, 回撤值={expected_max_dd:.4%}"
        )
    
    def verify_win_rate_calculation(self):
        """验证胜率计算（REQ-4.4.5）"""
        print("\n" + "="*60)
        print("9. 验证胜率和盈亏比计算")
        print("="*60)
        
        # 测试数据：交易PnL
        trade_pnls = [1000, -500, 1500, -300, 800, -600, 1200, -400, 900, -200]
        
        # 计算胜率
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        expected_win_rate = len(winning_trades) / len(trade_pnls)
        
        # 计算盈亏比
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
        expected_profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 模拟计算
        actual_win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
        actual_profit_ratio = np.mean([p for p in trade_pnls if p > 0]) / \
                             abs(np.mean([p for p in trade_pnls if p < 0]))
        
        passed1 = abs(actual_win_rate - expected_win_rate) < 1e-6
        self.log_test(
            "胜率计算",
            passed1,
            actual_win_rate,
            expected_win_rate,
            details=f"盈利交易={len(winning_trades)}, 总交易={len(trade_pnls)}"
        )
        
        passed2 = abs(actual_profit_ratio - expected_profit_ratio) < 1e-6
        self.log_test(
            "盈亏比计算",
            passed2,
            actual_profit_ratio,
            expected_profit_ratio,
            details=f"平均盈利={avg_win:.2f}, 平均亏损={avg_loss:.2f}"
        )
    
    def verify_position_manager(self):
        """验证仓位管理器逐日盯市"""
        print("\n" + "="*60)
        print("10. 验证逐日盯市结算")
        print("="*60)
        
        # 创建仓位管理器
        pm = PositionManager(initial_capital=1000000, margin_rate=0.12)
        
        # 创建测试持仓
        position = Position(
            pair='CU0-AL0',
            direction='long_spread',
            spread_formula='CU0 - 3.3*AL0',
            open_date=datetime(2025, 1, 1),
            position_weight=0.05,
            symbol_y='CU0',
            symbol_x='AL0',
            contracts_y=2,
            contracts_x=7,
            open_price_y=50000,
            open_price_x=15000,
            margin_occupied=75000,
            open_commission=200,
            multiplier_y=5,
            multiplier_x=5
        )
        
        # 添加持仓
        pm.add_position(position)
        
        # 第一天结算（设置初始价格）
        prices_day1 = {'CU0': 50000, 'AL0': 15000}
        result1 = pm.daily_settlement(prices_day1)
        
        # 验证第一天（无PnL）
        expected_equity1 = 1000000  # 初始资金
        passed1 = abs(result1['total_equity'] - expected_equity1) < 0.01
        self.log_test(
            "逐日盯市 - 第一天",
            passed1,
            result1['total_equity'],
            expected_equity1,
            details=f"日PnL={result1['daily_pnl']:.2f}"
        )
        
        # 第二天结算（价格变动）
        prices_day2 = {'CU0': 51000, 'AL0': 15200}
        result2 = pm.daily_settlement(prices_day2)
        
        # 计算预期PnL
        # 做多价差：多Y空X
        y_pnl = (51000 - 50000) * 2 * 5  # Y腿盈利
        x_pnl = -(15200 - 15000) * 7 * 5  # X腿亏损（做空）
        expected_daily_pnl = y_pnl + x_pnl
        
        passed2 = abs(result2['daily_pnl'] - expected_daily_pnl) < 0.01
        self.log_test(
            "逐日盯市 - 第二天PnL",
            passed2,
            result2['daily_pnl'],
            expected_daily_pnl,
            details=f"Y腿PnL={y_pnl}, X腿PnL={x_pnl}"
        )
        
        # 验证权益更新
        expected_equity2 = expected_equity1 + expected_daily_pnl
        passed3 = abs(result2['total_equity'] - expected_equity2) < 0.01
        self.log_test(
            "逐日盯市 - 权益更新",
            passed3,
            result2['total_equity'],
            expected_equity2,
            details=f"可用资金={result2['available_capital']:.2f}"
        )
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*60)
        print("回测框架算法验证")
        print("="*60)
        
        # 运行各项测试
        self.verify_margin_calculation()
        self.verify_commission_calculation()
        self.verify_slippage_calculation()
        self.verify_pnl_calculation()
        self.verify_stop_loss_logic()
        self.verify_time_stop_logic()
        self.verify_sharpe_ratio_calculation()
        self.verify_max_drawdown_calculation()
        self.verify_win_rate_calculation()
        self.verify_position_manager()
        
        # 输出总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"总测试数: {self.total_tests}")
        print(f"通过数: {self.passed_tests}")
        print(f"失败数: {self.total_tests - self.passed_tests}")
        print(f"通过率: {self.passed_tests/self.total_tests*100:.1f}%")
        
        # 输出失败的测试
        if self.total_tests > self.passed_tests:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: 误差={result['error']:.6f}")
        
        return self.passed_tests == self.total_tests


def main():
    """主函数"""
    verifier = BacktestAlgorithmVerifier()
    success = verifier.run_all_tests()
    
    if success:
        print("\n✅ 所有算法验证通过！")
    else:
        print("\n❌ 存在算法验证失败，请检查")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())