"""
增强版回测引擎
包含止损逻辑和多重验证
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from configs.contract_specs import CONTRACT_SPECS, get_multiplier

logger = logging.getLogger(__name__)


class EnhancedBacktestEngine:
    """增强版回测引擎，包含止损和验证"""
    
    def __init__(self, initial_capital: float = 5000000, 
                 stop_loss_rate: float = 0.10,
                 transaction_cost_rate: float = 0.0002,
                 margin_rate: float = 0.12):
        """
        初始化
        
        Args:
            initial_capital: 初始资金
            stop_loss_rate: 止损比例（保证金的10%）
            transaction_cost_rate: 交易成本率
            margin_rate: 保证金率（12%）
        """
        self.initial_capital = initial_capital
        self.stop_loss_rate = stop_loss_rate
        self.transaction_cost_rate = transaction_cost_rate
        self.margin_rate = margin_rate
        self.positions = {}  # 当前持仓
        self.trades = []  # 完成的交易
        
    def calculate_pnl_with_validation(self, trade_data: Dict) -> Dict:
        """
        计算PnL并进行多重验证
        
        Returns:
            包含详细验证信息的PnL字典
        """
        # 提取数据
        pair = trade_data['pair']
        y_symbol, x_symbol = pair.split('-')
        position_ratio = trade_data['position_ratio']
        y_ratio, x_ratio = map(int, position_ratio.split(':'))
        
        # 获取价格
        y_open = trade_data['y_open']
        x_open = trade_data['x_open']
        y_close = trade_data['y_close']
        x_close = trade_data['x_close']
        
        # 获取真实合约乘数
        y_multiplier = get_multiplier(y_symbol)
        x_multiplier = get_multiplier(x_symbol)
        
        # 价格变化
        y_change = y_close - y_open
        x_change = x_close - x_open
        
        # 方法1：分腿计算
        y_pnl = y_change * y_ratio * y_multiplier  # Y腿做多
        x_pnl = -x_change * x_ratio * x_multiplier  # X腿做空（注意负号）
        pnl_method1 = y_pnl + x_pnl
        
        # 方法2：价差计算验证
        open_spread = y_open - trade_data.get('beta', 1.0) * x_open
        close_spread = y_close - trade_data.get('beta', 1.0) * x_close
        spread_change = close_spread - open_spread
        
        # 对于标准配对交易，价差扩大时Y腿相对X腿上涨
        pnl_method2 = spread_change * y_ratio * y_multiplier
        
        # 方法3：逐步验证
        validation = {
            'y_symbol': y_symbol,
            'x_symbol': x_symbol,
            'y_multiplier': y_multiplier,
            'x_multiplier': x_multiplier,
            'y_ratio': y_ratio,
            'x_ratio': x_ratio,
            'y_open': y_open,
            'x_open': x_open,
            'y_close': y_close,
            'x_close': x_close,
            'y_change': y_change,
            'x_change': x_change,
            'y_change_pct': y_change / y_open * 100,
            'x_change_pct': x_change / x_open * 100,
            'y_pnl': y_pnl,
            'x_pnl': x_pnl,
            'pnl_method1': pnl_method1,
            'pnl_method2': pnl_method2,
            'pnl_diff': abs(pnl_method1 - pnl_method2) if abs(pnl_method2) > 0 else 0,
            'direction_check': 'Y做多X做空' if y_pnl * y_change >= 0 else '方向异常',
            'final_pnl': pnl_method1  # 使用方法1作为最终结果
        }
        
        # 计算名义价值和成本
        y_notional = y_open * y_ratio * y_multiplier
        x_notional = x_open * x_ratio * x_multiplier
        total_notional = y_notional + x_notional
        transaction_cost = total_notional * self.transaction_cost_rate * 2
        
        validation.update({
            'y_notional': y_notional,
            'x_notional': x_notional,
            'total_notional': total_notional,
            'transaction_cost': transaction_cost,
            'net_pnl': pnl_method1 - transaction_cost,
            'return_pct': (pnl_method1 - transaction_cost) / total_notional * 100
        })
        
        return validation
    
    def check_stop_loss(self, position: Dict, current_prices: Dict) -> Tuple[bool, float]:
        """
        检查是否触发止损（基于保证金的10%）
        
        Returns:
            (是否止损, 当前浮动盈亏)
        """
        # 计算当前价值
        y_symbol = position['y_symbol']
        x_symbol = position['x_symbol']
        y_current = current_prices.get(y_symbol, position['y_open'])
        x_current = current_prices.get(x_symbol, position['x_open'])
        
        # 计算浮动盈亏
        temp_data = {
            'pair': position['pair'],
            'position_ratio': position['position_ratio'],
            'y_open': position['y_open'],
            'x_open': position['x_open'],
            'y_close': y_current,
            'x_close': x_current,
            'beta': position.get('beta', 1.0)
        }
        
        pnl_info = self.calculate_pnl_with_validation(temp_data)
        current_pnl = pnl_info['final_pnl']
        
        # 计算保证金占用（两腿的保证金总和）
        total_margin = position['total_notional'] * self.margin_rate
        
        # 检查是否达到止损线（亏损达到保证金的10%）
        max_loss = total_margin * self.stop_loss_rate
        should_stop = current_pnl <= -max_loss
        
        return should_stop, current_pnl
    
    def execute_trade_with_stop_loss(self, open_signal: Dict, close_signal: Dict, 
                                    price_data: Dict) -> Dict:
        """
        执行交易，包含止损检查
        """
        pair = open_signal['pair']
        y_symbol, x_symbol = pair.split('-')
        
        open_date = pd.to_datetime(open_signal['date'])
        close_date = pd.to_datetime(close_signal['date'])
        
        # 获取开仓价格
        y_open = price_data[y_symbol].loc[open_date, 'close']
        x_open = price_data[x_symbol].loc[open_date, 'close']
        
        # 创建持仓
        position = {
            'pair': pair,
            'y_symbol': y_symbol,
            'x_symbol': x_symbol,
            'open_date': open_date,
            'position_ratio': open_signal.get('position_ratio', '1:1'),
            'y_open': y_open,
            'x_open': x_open,
            'beta': open_signal.get('beta', 1.0),
            'open_z_score': open_signal.get('z_score', 0),
            'total_notional': self._calculate_notional(y_symbol, x_symbol, y_open, x_open, 
                                                       open_signal.get('position_ratio', '1:1')),
            'total_margin': self._calculate_notional(y_symbol, x_symbol, y_open, x_open, 
                                                     open_signal.get('position_ratio', '1:1')) * self.margin_rate
        }
        
        # 检查每日是否触发止损
        actual_close_date = close_date
        stop_loss_triggered = False
        dates_to_check = pd.date_range(open_date + timedelta(days=1), close_date)
        
        for check_date in dates_to_check:
            if check_date in price_data[y_symbol].index and check_date in price_data[x_symbol].index:
                current_prices = {
                    y_symbol: price_data[y_symbol].loc[check_date, 'close'],
                    x_symbol: price_data[x_symbol].loc[check_date, 'close']
                }
                
                should_stop, current_pnl = self.check_stop_loss(position, current_prices)
                
                if should_stop:
                    actual_close_date = check_date
                    stop_loss_triggered = True
                    break
        
        # 获取实际平仓价格
        y_close = price_data[y_symbol].loc[actual_close_date, 'close']
        x_close = price_data[x_symbol].loc[actual_close_date, 'close']
        
        # 计算最终PnL
        trade_data = {
            'pair': pair,
            'position_ratio': position['position_ratio'],
            'y_open': y_open,
            'x_open': x_open,
            'y_close': y_close,
            'x_close': x_close,
            'beta': position['beta']
        }
        
        pnl_result = self.calculate_pnl_with_validation(trade_data)
        
        # 构建完整交易记录
        trade_record = {
            'pair': pair,
            'y_symbol': y_symbol,
            'x_symbol': x_symbol,
            'open_date': open_date,
            'close_date': actual_close_date,
            'original_close_date': close_date,
            'holding_days': (actual_close_date - open_date).days,
            'stop_loss_triggered': stop_loss_triggered,
            'position_ratio': position['position_ratio'],
            'open_z_score': position['open_z_score'],
            'close_z_score': close_signal.get('z_score', 0) if not stop_loss_triggered else None,
            **pnl_result  # 包含所有PnL验证信息
        }
        
        return trade_record
    
    def _calculate_notional(self, y_symbol: str, x_symbol: str, 
                           y_price: float, x_price: float, 
                           position_ratio: str) -> float:
        """计算名义价值"""
        y_ratio, x_ratio = map(int, position_ratio.split(':'))
        y_multiplier = get_multiplier(y_symbol)
        x_multiplier = get_multiplier(x_symbol)
        
        y_notional = y_price * y_ratio * y_multiplier
        x_notional = x_price * x_ratio * x_multiplier
        
        return y_notional + x_notional


def validate_pnl_calculation(trade: Dict) -> Dict:
    """
    验证PnL计算的正确性
    
    Returns:
        验证报告
    """
    validation_report = {
        'trade_id': f"{trade['pair']}_{trade['open_date']}",
        'checks': []
    }
    
    # 检查1：方向验证
    if trade['y_change'] > 0 and trade['y_pnl'] > 0:
        validation_report['checks'].append('✓ Y腿方向正确（做多）')
    elif trade['y_change'] < 0 and trade['y_pnl'] < 0:
        validation_report['checks'].append('✓ Y腿方向正确（做多）')
    else:
        validation_report['checks'].append('✗ Y腿方向异常')
    
    if trade['x_change'] > 0 and trade['x_pnl'] < 0:
        validation_report['checks'].append('✓ X腿方向正确（做空）')
    elif trade['x_change'] < 0 and trade['x_pnl'] > 0:
        validation_report['checks'].append('✓ X腿方向正确（做空）')
    else:
        validation_report['checks'].append('✗ X腿方向异常')
    
    # 检查2：乘数验证
    expected_y_mult = CONTRACT_SPECS.get(trade['y_symbol'], {}).get('multiplier')
    expected_x_mult = CONTRACT_SPECS.get(trade['x_symbol'], {}).get('multiplier')
    
    if trade['y_multiplier'] == expected_y_mult:
        validation_report['checks'].append(f'✓ Y乘数正确: {expected_y_mult}')
    else:
        validation_report['checks'].append(f'✗ Y乘数错误: {trade["y_multiplier"]} vs {expected_y_mult}')
    
    if trade['x_multiplier'] == expected_x_mult:
        validation_report['checks'].append(f'✓ X乘数正确: {expected_x_mult}')
    else:
        validation_report['checks'].append(f'✗ X乘数错误: {trade["x_multiplier"]} vs {expected_x_mult}')
    
    # 检查3：计算方法一致性
    if 'pnl_diff' in trade and trade['pnl_diff'] < 100:
        validation_report['checks'].append('✓ 两种计算方法结果一致')
    else:
        validation_report['checks'].append(f'⚠ 计算方法差异: {trade.get("pnl_diff", 0):.2f}')
    
    # 检查4：价格合理性
    if 'y_open' in trade and 'x_open' in trade:
        if 0 < trade['y_open'] < 1000000 and 0 < trade['x_open'] < 1000000:
            validation_report['checks'].append('✓ 价格在合理范围')
        else:
            validation_report['checks'].append('✗ 价格异常')
    else:
        validation_report['checks'].append('⚠ 价格数据缺失')
    
    # 汇总
    passed = sum('✓' in check for check in validation_report['checks'])
    total = len(validation_report['checks'])
    validation_report['passed'] = passed
    validation_report['total'] = total
    validation_report['valid'] = passed == total
    
    return validation_report