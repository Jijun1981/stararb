"""
回测引擎模块
协调各子模块，执行完整回测流程
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .position_sizing import PositionSizer, PositionSizingConfig
from .trade_executor import TradeExecutor, ExecutionConfig, Position, Trade
from .risk_manager import RiskManager, RiskConfig
from .performance import PerformanceAnalyzer


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 5000000
    sizing_config: Optional[PositionSizingConfig] = None
    execution_config: Optional[ExecutionConfig] = None
    risk_config: Optional[RiskConfig] = None
    
    def __post_init__(self):
        """初始化默认配置"""
        if self.sizing_config is None:
            self.sizing_config = PositionSizingConfig()
        if self.execution_config is None:
            self.execution_config = ExecutionConfig()
        if self.risk_config is None:
            self.risk_config = RiskConfig()


class BacktestEngine:
    """
    回测引擎 - 协调器
    执行完整的回测流程
    """
    
    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        self.initial_capital = config.initial_capital
        
        # 初始化各子模块
        self.position_sizer = PositionSizer(config.sizing_config)
        self.executor = TradeExecutor(config.execution_config)
        self.risk_manager = RiskManager(config.risk_config)
        self.analyzer = PerformanceAnalyzer()
        
        # 状态管理
        self.positions = {}  # 当前持仓
        self.trades = []     # 完成的交易
        self.equity_curve = []  # 权益曲线
        self.current_capital = self.initial_capital
        self.available_capital = self.initial_capital
    
    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        contract_specs: Dict
    ) -> Dict[str, Any]:
        """
        执行完整回测
        
        Args:
            signals: 信号DataFrame（包含date, pair, trade_signal, beta等）
            prices: 价格DataFrame（包含各品种的日线价格）
            contract_specs: 合约规格字典
            
        Returns:
            回测结果字典
        """
        # 设置合约规格
        self.executor.set_contract_specs(contract_specs)
        
        # 初始化权益曲线
        self.equity_curve = [self.initial_capital]
        
        # 按日期遍历
        dates = signals['date'].unique()
        dates = sorted(dates)
        
        for date in dates:
            # 获取当天信号
            day_signals = signals[signals['date'] == date]
            
            # 获取当天价格
            day_prices = self._get_day_prices(prices, date)
            
            # 处理当天
            self.process_date(date, day_signals, day_prices, contract_specs)
            
            # 更新权益曲线
            self._update_equity(date, day_prices, contract_specs)
        
        # 强制平掉所有剩余持仓
        self._close_all_positions(dates[-1] if len(dates) > 0 else datetime.now(), prices, contract_specs)
        
        # 生成报告
        return self._generate_report()
    
    def process_date(
        self,
        date: datetime,
        day_signals: pd.DataFrame,
        prices: Dict[str, float],
        contract_specs: Dict
    ):
        """
        处理单个交易日
        
        Args:
            date: 当前日期
            day_signals: 当天信号
            prices: 当天价格
            contract_specs: 合约规格
        """
        # 1. 风险检查（止损、时间止损）
        self._check_positions_risk(date, prices, contract_specs)
        
        # 2. 处理信号
        for _, signal in day_signals.iterrows():
            self._process_signal(signal, prices, contract_specs, date)
    
    def _check_positions_risk(
        self,
        date: datetime,
        prices: Dict[str, float],
        contract_specs: Dict
    ):
        """
        检查所有持仓的风险
        
        Args:
            date: 当前日期
            prices: 当前价格
            contract_specs: 合约规格
        """
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            # 获取该配对的当前价格
            pair_prices = {
                'x': prices.get(position.symbol_x, position.open_price_x),
                'y': prices.get(position.symbol_y, position.open_price_y)
            }
            
            # 计算浮动盈亏
            pnl_data = self.risk_manager.calculate_unrealized_pnl(position, pair_prices)
            current_pnl = pnl_data['net_pnl']
            
            # 使用持仓实际分配的资金
            allocated_capital = getattr(position, 'allocated_capital', 
                                      self.initial_capital * self.config.sizing_config.position_weight)
            
            # 检查止损
            should_stop, stop_reason = self.risk_manager.check_stop_loss(
                position, current_pnl, allocated_capital
            )
            
            if should_stop:
                positions_to_close.append((position_id, 'stop_loss'))
                self.risk_manager.record_stop_loss(position, current_pnl)
                continue
            
            # 检查时间止损
            should_stop, time_reason = self.risk_manager.check_time_stop(position, date)
            
            if should_stop:
                positions_to_close.append((position_id, 'time_stop'))
                self.risk_manager.record_time_stop(position)
        
        # 执行平仓
        for position_id, reason in positions_to_close:
            self._close_position(position_id, prices, reason, date)
    
    def _process_signal(
        self,
        signal: pd.Series,
        prices: Dict[str, float],
        contract_specs: Dict,
        date: datetime
    ):
        """
        处理交易信号
        
        Args:
            signal: 信号Series
            prices: 当前价格
            contract_specs: 合约规格
            date: 当前日期
        """
        pair = signal['pair']
        trade_signal = signal.get('trade_signal', None)
        
        if trade_signal is None:
            return
        
        # 检查是否已有该配对的持仓
        existing_position = self._find_position_by_pair(pair)
        
        # 平仓信号
        if trade_signal == 'close' and existing_position:
            self._close_position(existing_position.position_id, prices, 'signal', date)
        
        # 开仓信号
        elif trade_signal in ['open_long', 'open_short'] and not existing_position:
            # 检查持仓数量限制
            if not self.risk_manager.check_position_limit(self.positions):
                return
            
            # 计算手数
            self._open_position(signal, prices, contract_specs, date)
    
    def _open_position(
        self,
        signal: pd.Series,
        prices: Dict[str, float],
        contract_specs: Dict,
        date: datetime
    ):
        """
        开仓
        
        Args:
            signal: 信号
            prices: 当前价格
            contract_specs: 合约规格
            date: 当前日期
        """
        pair = signal['pair']
        symbol_x = signal['symbol_x']
        symbol_y = signal['symbol_y']
        beta = signal.get('beta', 1.0)
        
        # 获取价格和合约规格
        price_x = prices.get(symbol_x)
        price_y = prices.get(symbol_y)
        
        if price_x is None or price_y is None:
            return
        
        spec_x = contract_specs.get(symbol_x, {'multiplier': 1, 'tick_size': 1})
        spec_y = contract_specs.get(symbol_y, {'multiplier': 1, 'tick_size': 1})
        
        # 第一步：计算最小整数比
        ratio_result = self.position_sizer.calculate_min_integer_ratio(
            beta=beta,
            price_x=price_x,
            price_y=price_y,
            multiplier_x=spec_x['multiplier'],
            multiplier_y=spec_y['multiplier']
        )
        
        # 第二步：应用资金约束
        position_result = self.position_sizer.calculate_position_size(
            min_lots={
                'lots_x': ratio_result['lots_x'],
                'lots_y': ratio_result['lots_y']
            },
            prices={'x': price_x, 'y': price_y},
            multipliers={
                'x': spec_x['multiplier'],
                'y': spec_y['multiplier']
            },
            total_capital=self.current_capital,
            position_weight=self.config.sizing_config.position_weight
        )
        
        # 检查是否可以交易
        if not position_result['can_trade']:
            return
        
        # 检查保证金充足性
        if not self.risk_manager.check_margin_adequacy(
            self.available_capital,
            position_result['margin_required']
        ):
            return
        
        # 执行开仓
        position = self.executor.execute_open(
            pair_info={
                'pair': pair,
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                'beta': beta
            },
            lots={
                'x': position_result['final_lots_x'],
                'y': position_result['final_lots_y']
            },
            prices={'x': price_x, 'y': price_y},
            signal_type=signal['trade_signal'],
            open_date=date
        )
        
        # 记录持仓
        position.allocated_capital = position_result['allocated_capital']
        self.positions[position.position_id] = position
        
        # 更新可用资金
        self.available_capital -= position.margin
    
    def _close_position(
        self,
        position_id: str,
        prices: Dict[str, float],
        reason: str,
        date: datetime
    ):
        """
        平仓
        
        Args:
            position_id: 持仓ID
            prices: 当前价格
            reason: 平仓原因
            date: 当前日期
        """
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # 获取平仓价格
        price_x = prices.get(position.symbol_x, position.open_price_x)
        price_y = prices.get(position.symbol_y, position.open_price_y)
        
        # 执行平仓
        trade = self.executor.execute_close(
            position=position,
            prices={'x': price_x, 'y': price_y},
            reason=reason,
            close_date=date
        )
        
        # 记录交易
        self.trades.append(trade)
        
        # 更新资金
        self.available_capital += trade.margin_released
        self.current_capital += trade.net_pnl
        
        # 删除持仓
        del self.positions[position_id]
    
    def _close_all_positions(
        self,
        date: datetime,
        prices: pd.DataFrame,
        contract_specs: Dict
    ):
        """
        强制平掉所有持仓
        
        Args:
            date: 当前日期
            prices: 价格数据
            contract_specs: 合约规格
        """
        positions_to_close = list(self.positions.keys())
        
        for position_id in positions_to_close:
            # 获取最后的价格
            position = self.positions[position_id]
            last_prices = self._get_last_prices(prices, position.symbol_x, position.symbol_y)
            
            self._close_position(position_id, last_prices, 'forced', date)
    
    def _update_equity(
        self,
        date: datetime,
        prices: Dict[str, float],
        contract_specs: Dict
    ):
        """
        更新权益曲线
        
        Args:
            date: 当前日期
            prices: 当前价格
            contract_specs: 合约规格
        """
        # 计算所有持仓的浮动盈亏
        total_unrealized_pnl = 0
        
        for position in self.positions.values():
            pair_prices = {
                'x': prices.get(position.symbol_x, position.open_price_x),
                'y': prices.get(position.symbol_y, position.open_price_y)
            }
            
            pnl_data = self.risk_manager.calculate_unrealized_pnl(position, pair_prices)
            total_unrealized_pnl += pnl_data['net_pnl']
        
        # 当前权益 = 现金 + 浮动盈亏
        current_equity = self.current_capital + total_unrealized_pnl
        self.equity_curve.append(current_equity)
    
    def _find_position_by_pair(self, pair: str) -> Optional[Position]:
        """
        根据配对名称查找持仓
        
        Args:
            pair: 配对名称
            
        Returns:
            Position对象或None
        """
        for position in self.positions.values():
            if position.pair == pair:
                return position
        return None
    
    def _get_day_prices(self, prices: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """
        获取指定日期的价格
        
        Args:
            prices: 价格DataFrame
            date: 日期
            
        Returns:
            价格字典
        """
        if isinstance(prices, pd.DataFrame):
            if date in prices.index:
                return prices.loc[date].to_dict()
            else:
                # 使用最近的价格
                closest_date = prices.index[prices.index <= date][-1] if any(prices.index <= date) else prices.index[0]
                return prices.loc[closest_date].to_dict()
        return {}
    
    def _get_last_prices(self, prices: pd.DataFrame, symbol_x: str, symbol_y: str) -> Dict[str, float]:
        """
        获取最后的价格
        
        Args:
            prices: 价格DataFrame
            symbol_x: X品种代码
            symbol_y: Y品种代码
            
        Returns:
            价格字典
        """
        if isinstance(prices, pd.DataFrame):
            last_row = prices.iloc[-1] if len(prices) > 0 else pd.Series()
            return {
                'x': last_row.get(symbol_x, 0),
                'y': last_row.get(symbol_y, 0)
            }
        return {'x': 0, 'y': 0}
    
    def _generate_report(self) -> Dict[str, Any]:
        """
        生成回测报告
        
        Returns:
            回测结果字典
        """
        # 创建权益曲线Series
        equity_series = pd.Series(self.equity_curve)
        
        # 生成完整报告
        report = self.analyzer.generate_report(
            trades=self.trades,
            equity_curve=equity_series,
            initial_capital=self.initial_capital
        )
        
        # 添加额外信息
        report['initial_capital'] = self.initial_capital
        report['final_capital'] = self.current_capital
        report['total_positions'] = len(self.trades)
        report['risk_statistics'] = {
            'stop_loss_stats': self.risk_manager.get_stop_loss_statistics(),
            'time_stop_stats': self.risk_manager.get_time_stop_statistics()
        }
        
        return report