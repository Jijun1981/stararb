"""
交易执行模块
负责处理所有交易相关的执行逻辑
对应需求：REQ-4.2
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid


@dataclass
class ExecutionConfig:
    """交易执行配置"""
    commission_rate: float = 0.0002  # 手续费率（万分之2）
    slippage_ticks: int = 3          # 滑点tick数
    margin_rate: float = 0.12         # 保证金率


@dataclass
class Position:
    """持仓记录"""
    position_id: str
    pair: str
    symbol_x: str
    symbol_y: str
    lots_x: int
    lots_y: int
    direction: str  # 'long' or 'short'
    open_date: datetime
    open_price_x: float
    open_price_y: float
    margin: float
    beta: float
    open_commission: float = 0
    allocated_capital: float = 0  # 分配的资金


@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    position_id: str
    pair: str
    symbol_x: str
    symbol_y: str
    lots_x: int
    lots_y: int
    direction: str
    # 开仓信息
    open_date: datetime
    open_price_x: float
    open_price_y: float
    open_commission: float
    # 平仓信息
    close_date: datetime
    close_price_x: float
    close_price_y: float
    close_commission: float
    # 盈亏信息
    gross_pnl: float
    net_pnl: float
    return_pct: float
    # 其他信息
    holding_days: int
    close_reason: str  # 'signal', 'stop_loss', 'time_stop'
    margin_released: float
    capital_change: float


class TradeExecutor:
    """
    交易执行器
    处理开仓、平仓、成本计算等交易逻辑
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        初始化交易执行器
        
        Args:
            config: 交易执行配置
        """
        self.config = config
        self.commission_rate = config.commission_rate
        self.slippage_ticks = config.slippage_ticks
        self.margin_rate = config.margin_rate
        
        # 合约规格（需要设置）
        self.contract_specs = {}
    
    def set_contract_specs(self, specs: Dict[str, Dict]):
        """
        设置合约规格
        
        Args:
            specs: 合约规格字典
                {
                    'CU': {'multiplier': 5, 'tick_size': 10},
                    'SN': {'multiplier': 1, 'tick_size': 10},
                    ...
                }
        """
        self.contract_specs = specs
    
    def execute_open(
        self,
        pair_info: Dict,
        lots: Dict[str, int],
        prices: Dict[str, float],
        signal_type: str,
        open_date: Optional[datetime] = None
    ) -> Position:
        """
        REQ-4.2.1: 执行开仓
        
        Args:
            pair_info: 配对信息 {'pair', 'symbol_x', 'symbol_y', 'beta'}
            lots: 手数 {'x': lots_x, 'y': lots_y}
            prices: 价格 {'x': price_x, 'y': price_y}
            signal_type: 信号类型 'open_long' or 'open_short'
            open_date: 开仓日期（可选）
            
        Returns:
            Position对象
        """
        # REQ-4.2.1.6: 生成唯一position_id
        position_id = str(uuid.uuid4())[:8]
        
        # REQ-4.2.1.1: 记录开仓时间、配对名称、方向
        if open_date is None:
            open_date = datetime.now()
        
        pair = pair_info['pair']
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        beta = pair_info.get('beta', 1.0)
        
        direction = 'long' if 'long' in signal_type else 'short'
        
        # REQ-4.2.1.2: 记录手数
        lots_x = lots['x']
        lots_y = lots['y']
        
        # 获取合约规格
        spec_x = self.contract_specs.get(symbol_x, {'multiplier': 1, 'tick_size': 1})
        spec_y = self.contract_specs.get(symbol_y, {'multiplier': 1, 'tick_size': 1})
        
        # REQ-4.2.1.3: 计算开仓滑点
        price_x = prices['x']
        price_y = prices['y']
        
        if direction == 'long':
            # 做多配对：买X卖Y
            open_price_x = price_x + self.slippage_ticks * spec_x['tick_size']  # 买入加滑点
            open_price_y = price_y - self.slippage_ticks * spec_y['tick_size']  # 卖出减滑点
        else:
            # 做空配对：卖X买Y
            open_price_x = price_x - self.slippage_ticks * spec_x['tick_size']  # 卖出减滑点
            open_price_y = price_y + self.slippage_ticks * spec_y['tick_size']  # 买入加滑点
        
        # REQ-4.2.1.4: 计算开仓手续费
        value_x = lots_x * price_x * spec_x['multiplier']
        value_y = lots_y * price_y * spec_y['multiplier']
        total_value = value_x + value_y
        open_commission = total_value * self.commission_rate
        
        # REQ-4.2.1.5: 计算保证金占用
        margin = total_value * self.margin_rate
        
        # 创建持仓对象
        position = Position(
            position_id=position_id,
            pair=pair,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            lots_x=lots_x,
            lots_y=lots_y,
            direction=direction,
            open_date=open_date,
            open_price_x=open_price_x,
            open_price_y=open_price_y,
            margin=margin,
            beta=beta,
            open_commission=open_commission
        )
        
        return position
    
    def execute_close(
        self,
        position: Position,
        prices: Dict[str, float],
        reason: str,
        close_date: Optional[datetime] = None
    ) -> Trade:
        """
        REQ-4.2.2: 执行平仓
        
        Args:
            position: 持仓对象
            prices: 平仓价格 {'x': price_x, 'y': price_y}
            reason: 平仓原因 'signal', 'stop_loss', 'time_stop'
            close_date: 平仓日期（可选）
            
        Returns:
            Trade对象
        """
        # REQ-4.2.2.1: 记录平仓时间
        if close_date is None:
            close_date = datetime.now()
        
        # 获取合约规格
        spec_x = self.contract_specs.get(position.symbol_x, {'multiplier': 1, 'tick_size': 1})
        spec_y = self.contract_specs.get(position.symbol_y, {'multiplier': 1, 'tick_size': 1})
        
        # REQ-4.2.2.2: 计算平仓滑点（方向相反）
        price_x = prices['x']
        price_y = prices['y']
        
        if position.direction == 'long':
            # 平多：卖X买Y
            close_price_x = price_x - self.slippage_ticks * spec_x['tick_size']  # 卖出减滑点
            close_price_y = price_y + self.slippage_ticks * spec_y['tick_size']  # 买入加滑点
        else:
            # 平空：买X卖Y
            close_price_x = price_x + self.slippage_ticks * spec_x['tick_size']  # 买入加滑点
            close_price_y = price_y - self.slippage_ticks * spec_y['tick_size']  # 卖出减滑点
        
        # REQ-4.2.2.3: 计算平仓手续费
        close_value_x = position.lots_x * price_x * spec_x['multiplier']
        close_value_y = position.lots_y * price_y * spec_y['multiplier']
        close_total_value = close_value_x + close_value_y
        close_commission = close_total_value * self.commission_rate
        
        # REQ-4.2.2.4: 计算实现盈亏
        if position.direction == 'long':
            # 做多盈亏
            pnl_x = (close_price_x - position.open_price_x) * position.lots_x * spec_x['multiplier']
            pnl_y = (position.open_price_y - close_price_y) * position.lots_y * spec_y['multiplier']
        else:
            # 做空盈亏
            pnl_x = (position.open_price_x - close_price_x) * position.lots_x * spec_x['multiplier']
            pnl_y = (close_price_y - position.open_price_y) * position.lots_y * spec_y['multiplier']
        
        gross_pnl = pnl_x + pnl_y
        total_commission = position.open_commission + close_commission
        net_pnl = gross_pnl - total_commission
        
        # 计算收益率
        if position.margin > 0:
            return_pct = net_pnl / position.margin
        else:
            return_pct = 0
        
        # 计算持仓天数
        holding_days = (close_date - position.open_date).days
        if holding_days == 0:
            holding_days = 1  # 至少算1天
        
        # REQ-4.2.2.5: 释放保证金
        margin_released = position.margin
        
        # REQ-4.2.2.6: 计算资金变动
        capital_change = net_pnl + margin_released
        
        # 创建交易记录
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            position_id=position.position_id,
            pair=position.pair,
            symbol_x=position.symbol_x,
            symbol_y=position.symbol_y,
            lots_x=position.lots_x,
            lots_y=position.lots_y,
            direction=position.direction,
            # 开仓信息
            open_date=position.open_date,
            open_price_x=position.open_price_x,
            open_price_y=position.open_price_y,
            open_commission=position.open_commission,
            # 平仓信息
            close_date=close_date,
            close_price_x=close_price_x,
            close_price_y=close_price_y,
            close_commission=close_commission,
            # 盈亏信息
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            # 其他信息
            holding_days=holding_days,
            close_reason=reason,
            margin_released=margin_released,
            capital_change=capital_change
        )
        
        return trade
    
    def calculate_slippage(
        self,
        price: float,
        side: str,
        tick_size: float
    ) -> float:
        """
        计算滑点后的价格
        
        Args:
            price: 原始价格
            side: 'buy' or 'sell'
            tick_size: tick大小
            
        Returns:
            滑点后的价格
        """
        if side == 'buy':
            return price + self.slippage_ticks * tick_size
        else:
            return price - self.slippage_ticks * tick_size
    
    def calculate_commission(
        self,
        position_value: float
    ) -> float:
        """
        计算手续费
        
        Args:
            position_value: 持仓价值
            
        Returns:
            手续费
        """
        return position_value * self.commission_rate
    
    def calculate_margin(
        self,
        lots: Dict[str, int],
        prices: Dict[str, float],
        multipliers: Dict[str, float]
    ) -> float:
        """
        计算保证金需求
        
        Args:
            lots: 手数
            prices: 价格
            multipliers: 合约乘数
            
        Returns:
            保证金需求
        """
        value_x = lots['x'] * prices['x'] * multipliers['x']
        value_y = lots['y'] * prices['y'] * multipliers['y']
        return (value_x + value_y) * self.margin_rate