"""
风险管理模块
负责监控和管理所有风险相关逻辑
对应需求：REQ-4.3
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import math


@dataclass
class RiskConfig:
    """风险管理配置"""
    stop_loss_pct: float = 0.10      # 止损百分比（基于分配资金）
    max_holding_days: int = 30       # 最大持仓天数
    max_positions: int = 20          # 最大同时持仓数
    margin_buffer: float = 0.8       # 保证金缓冲率
    beta_filter_enabled: bool = False  # 是否启用beta过滤
    beta_min: float = 0.3            # beta绝对值最小阈值
    beta_max: float = 3.0            # beta绝对值最大阈值


@dataclass
class RiskStatistics:
    """风险统计信息"""
    stop_loss_count: int = 0
    total_stop_loss: float = 0
    time_stop_count: int = 0
    pairs_time_stopped: List[str] = field(default_factory=list)


class RiskManager:
    """
    风险管理器
    实现止损、时间止损、保证金监控等风险管理功能
    """
    
    def __init__(self, config: RiskConfig):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.stop_loss_pct = config.stop_loss_pct
        self.max_holding_days = config.max_holding_days
        self.max_positions = config.max_positions
        self.margin_buffer = config.margin_buffer
        
        # 统计信息
        self.stats = RiskStatistics()
    
    # ========== REQ-4.3.1: 止损管理 ==========
    
    def calculate_unrealized_pnl(
        self,
        position: Any,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        REQ-4.3.1.1: 计算浮动盈亏
        
        Args:
            position: 持仓对象
            current_prices: 当前价格 {'x': price_x, 'y': price_y}
            
        Returns:
            {
                'gross_pnl': float,   # 毛利润
                'costs': float,       # 成本（手续费等）
                'net_pnl': float      # 净利润
            }
        """
        # 获取合约乘数（这里假设标准值，实际应从配置获取）
        multipliers = self._get_multipliers(position.symbol_x, position.symbol_y)
        
        # 计算价格变动
        price_change_x = current_prices['x'] - position.open_price_x
        price_change_y = current_prices['y'] - position.open_price_y
        
        # 计算各腿盈亏 - 根据Beta符号和方向确定
        # REQ-4.2.2.4: 核心：正确计算PnL，考虑Beta符号
        
        # 需要从position获取beta值
        beta = getattr(position, 'beta', 1.0)  # 默认正Beta
        
        if beta > 0:  # 正Beta
            if position.direction == 'long':
                # REQ-4.2.1.4: 正Beta + Long = 买Y卖X
                # REQ-4.2.2.5: pnl_x = -lots_x × (close_x - open_x) × multiplier_x
                # REQ-4.2.2.6: pnl_y = lots_y × (close_y - open_y) × multiplier_y
                pnl_x = -price_change_x * position.lots_x * multipliers['x']  # 卖X
                pnl_y = price_change_y * position.lots_y * multipliers['y']   # 买Y
            else:  # short
                # REQ-4.2.1.5: 正Beta + Short = 卖Y买X
                pnl_x = price_change_x * position.lots_x * multipliers['x']   # 买X
                pnl_y = -price_change_y * position.lots_y * multipliers['y']  # 卖Y
        else:  # 负Beta
            if position.direction == 'long':
                # REQ-4.2.1.6: 负Beta + Long = 买Y买X（同向）
                # REQ-4.2.2.7: pnl_x = lots_x × (close_x - open_x) × multiplier_x
                # REQ-4.2.2.8: pnl_y = lots_y × (close_y - open_y) × multiplier_y
                pnl_x = price_change_x * position.lots_x * multipliers['x']   # 买X
                pnl_y = price_change_y * position.lots_y * multipliers['y']   # 买Y
            else:  # short
                # REQ-4.2.1.7: 负Beta + Short = 卖Y卖X（同向）
                pnl_x = -price_change_x * position.lots_x * multipliers['x']  # 卖X
                pnl_y = -price_change_y * position.lots_y * multipliers['y']  # 卖Y
        
        gross_pnl = pnl_x + pnl_y
        
        # 估算成本（手续费等，这里简化处理）
        position_value = (
            position.lots_x * position.open_price_x * multipliers['x'] +
            position.lots_y * position.open_price_y * multipliers['y']
        )
        costs = position_value * 0.0002 * 2  # 双边手续费
        
        return {
            'gross_pnl': gross_pnl,
            'costs': costs,
            'net_pnl': gross_pnl - costs
        }
    
    def calculate_pnl_percentage(
        self,
        current_pnl: float,
        allocated_capital: float
    ) -> float:
        """
        REQ-4.3.1.2: 盈亏百分比基于分配资金
        
        Args:
            current_pnl: 当前盈亏
            allocated_capital: 分配的资金
            
        Returns:
            盈亏百分比
        """
        if allocated_capital <= 0:
            return 0
        return current_pnl / allocated_capital
    
    def check_stop_loss(
        self,
        position: Any,
        current_pnl: float,
        allocated_capital: float
    ) -> Tuple[bool, str]:
        """
        REQ-4.3.1.3: 检查止损条件
        
        Args:
            position: 持仓对象
            current_pnl: 当前盈亏
            allocated_capital: 分配的资金
            
        Returns:
            (是否触发止损, 原因描述)
        """
        pnl_pct = self.calculate_pnl_percentage(current_pnl, allocated_capital)
        
        # REQ-4.3.1.3: 触发条件 pnl_pct <= -stop_loss_pct
        if pnl_pct <= -self.stop_loss_pct:
            # REQ-4.3.1.4: 记录止损原因
            reason = f"Stop loss triggered: {pnl_pct:.1%} loss (threshold: -{self.stop_loss_pct:.0%})"
            return True, reason
        
        return False, ""
    
    def record_stop_loss(self, position: Any, loss_amount: float):
        """
        REQ-4.3.1.5: 统计止损次数和损失金额
        
        Args:
            position: 止损的持仓
            loss_amount: 损失金额
        """
        self.stats.stop_loss_count += 1
        self.stats.total_stop_loss += loss_amount
    
    def get_stop_loss_statistics(self) -> Dict[str, Any]:
        """
        获取止损统计信息
        
        Returns:
            止损统计
        """
        avg_loss = (
            self.stats.total_stop_loss / self.stats.stop_loss_count
            if self.stats.stop_loss_count > 0
            else 0
        )
        
        return {
            'stop_loss_count': self.stats.stop_loss_count,
            'total_stop_loss': self.stats.total_stop_loss,
            'avg_stop_loss': avg_loss
        }
    
    # ========== REQ-4.3.2: 时间止损 ==========
    
    def calculate_holding_days(
        self,
        position: Any,
        current_date: datetime,
        price_data: Any = None
    ) -> int:
        """
        REQ-4.3.2.1: 计算持仓天数（基于观测点数量，即交易日）
        
        Args:
            position: 持仓对象
            current_date: 当前日期
            price_data: 价格数据（用于计算交易日数量）
            
        Returns:
            持仓天数（交易日数量）
        """
        # 如果有价格数据，使用观测点数量计算交易日
        if price_data is not None:
            try:
                # 获取开仓日期到当前日期之间的交易日数量
                date_range = price_data.loc[position.open_date:current_date].index
                return len(date_range) - 1  # 减1是因为开仓当天不算持仓天数
            except (KeyError, AttributeError):
                # 如果价格数据不可用，回退到自然日计算
                pass
        
        # 回退方案：使用自然日
        delta = current_date - position.open_date
        return delta.days
    
    def check_time_stop(
        self,
        position: Any,
        current_date: datetime,
        price_data: Any = None
    ) -> Tuple[bool, str]:
        """
        REQ-4.3.2.2: 检查时间止损
        
        Args:
            position: 持仓对象
            current_date: 当前日期
            price_data: 价格数据（用于计算交易日数量）
            
        Returns:
            (是否触发时间止损, 原因描述)
        """
        holding_days = self.calculate_holding_days(position, current_date, price_data)
        
        # REQ-4.3.2.2: 触发条件 days >= max_holding_days
        if holding_days >= self.max_holding_days:
            # REQ-4.3.2.3: 记录时间止损原因
            reason = f"Time stop triggered: held for {holding_days} trading days (max: {self.max_holding_days} trading days)"
            return True, reason
        
        return False, ""
    
    def record_time_stop(self, position: Any):
        """
        REQ-4.3.2.4: 统计时间止损次数
        
        Args:
            position: 时间止损的持仓
        """
        self.stats.time_stop_count += 1
        if position.pair not in self.stats.pairs_time_stopped:
            self.stats.pairs_time_stopped.append(position.pair)
    
    def get_time_stop_statistics(self) -> Dict[str, Any]:
        """
        获取时间止损统计信息
        
        Returns:
            时间止损统计
        """
        return {
            'time_stop_count': self.stats.time_stop_count,
            'pairs_time_stopped': self.stats.pairs_time_stopped.copy()
        }
    
    # ========== REQ-4.3.3: 保证金监控 ==========
    
    def calculate_used_margin(self, positions: Dict[str, Any]) -> float:
        """
        REQ-4.3.3.1: 实时计算已用保证金
        
        Args:
            positions: 持仓字典
            
        Returns:
            已用保证金总额
        """
        total_margin = 0
        for position in positions.values():
            total_margin += position.margin
        return total_margin
    
    def calculate_available_margin(
        self,
        total_capital: float,
        used_margin: float
    ) -> float:
        """
        REQ-4.3.3.2: 计算可用保证金
        
        Args:
            total_capital: 总资金
            used_margin: 已用保证金
            
        Returns:
            可用保证金
        """
        return total_capital - used_margin
    
    def check_margin_adequacy(
        self,
        available_margin: float,
        required_margin: float
    ) -> bool:
        """
        REQ-4.3.3.3: 新开仓前检查保证金是否充足
        
        Args:
            available_margin: 可用保证金
            required_margin: 所需保证金
            
        Returns:
            是否有足够保证金
        """
        # REQ-4.3.3.3: required <= available × buffer
        return required_margin <= available_margin * self.margin_buffer
    
    def check_margin_with_reason(
        self,
        available_margin: float,
        required_margin: float
    ) -> Dict[str, Any]:
        """
        REQ-4.3.3.5: 保证金不足时返回详细原因
        
        Args:
            available_margin: 可用保证金
            required_margin: 所需保证金
            
        Returns:
            检查结果和原因
        """
        can_open = self.check_margin_adequacy(available_margin, required_margin)
        
        if not can_open:
            shortfall = required_margin - available_margin * self.margin_buffer
            return {
                'can_open': False,
                'reason': f'Insufficient margin: need {required_margin:.0f}, available {available_margin * self.margin_buffer:.0f}',
                'shortfall': shortfall
            }
        
        return {
            'can_open': True,
            'reason': '',
            'shortfall': 0
        }
    
    def check_position_limit(self, positions: Dict[str, Any]) -> bool:
        """
        检查是否达到最大持仓数量限制
        
        Args:
            positions: 当前持仓字典
            
        Returns:
            是否可以添加新持仓
        """
        if self.max_positions is None:
            return True  # 无限制
        return len(positions) < self.max_positions
    
    # ========== 综合风险管理 ==========
    
    def check_all_risks(
        self,
        position: Any,
        current_date: datetime,
        current_prices: Dict[str, float],
        current_pnl: float,
        allocated_capital: float,
        price_data: Any = None
    ) -> Dict[str, Any]:
        """
        综合风险检查
        
        Args:
            position: 持仓对象
            current_date: 当前日期
            current_prices: 当前价格
            current_pnl: 当前盈亏
            allocated_capital: 分配的资金
            price_data: 价格数据（用于计算交易日数量）
            
        Returns:
            综合风险状态
        """
        # 检查止损
        stop_loss_triggered, stop_loss_reason = self.check_stop_loss(
            position, current_pnl, allocated_capital
        )
        
        # 检查时间止损
        time_stop_triggered, time_stop_reason = self.check_time_stop(
            position, current_date, price_data
        )
        
        # 计算风险指标
        pnl_pct = self.calculate_pnl_percentage(current_pnl, allocated_capital)
        holding_days = self.calculate_holding_days(position, current_date, price_data)
        
        # 确定是否需要平仓
        should_close = stop_loss_triggered or time_stop_triggered
        close_reason = stop_loss_reason if stop_loss_triggered else (
            time_stop_reason if time_stop_triggered else ""
        )
        
        return {
            'should_close': should_close,
            'close_reason': close_reason,
            'risk_metrics': {
                'pnl_pct': pnl_pct,
                'holding_days': holding_days,
                'stop_loss_triggered': stop_loss_triggered,
                'time_stop_triggered': time_stop_triggered
            }
        }
    
    def calculate_portfolio_risk_metrics(
        self,
        positions: Dict[str, Any],
        current_prices: Dict[str, float],
        total_capital: float
    ) -> Dict[str, Any]:
        """
        计算组合风险指标
        
        Args:
            positions: 持仓字典
            current_prices: 当前价格
            total_capital: 总资金
            
        Returns:
            组合风险指标
        """
        total_margin = self.calculate_used_margin(positions)
        margin_utilization = total_margin / total_capital if total_capital > 0 else 0
        
        # 计算各持仓盈亏
        position_pnls = []
        positions_at_risk = 0
        
        for position in positions.values():
            pnl_data = self.calculate_unrealized_pnl(position, current_prices)
            pnl = pnl_data['net_pnl']
            position_pnls.append(pnl)
            
            # 检查是否处于风险状态
            if hasattr(position, 'allocated_capital'):
                pnl_pct = self.calculate_pnl_percentage(pnl, position.allocated_capital)
                if pnl_pct <= -self.stop_loss_pct * 0.8:  # 接近止损
                    positions_at_risk += 1
        
        # 计算最大损失和VaR
        max_position_loss = min(position_pnls) if position_pnls else 0
        
        # 简化的VaR计算（95%置信度）
        if position_pnls:
            sorted_pnls = sorted(position_pnls)
            var_index = int(len(sorted_pnls) * 0.05)
            portfolio_var_95 = sorted_pnls[var_index] if var_index < len(sorted_pnls) else sorted_pnls[0]
        else:
            portfolio_var_95 = 0
        
        return {
            'total_positions': len(positions),
            'total_margin_used': total_margin,
            'margin_utilization': margin_utilization,
            'positions_at_risk': positions_at_risk,
            'max_position_loss': max_position_loss,
            'portfolio_var_95': portfolio_var_95
        }
    
    def _get_multipliers(self, symbol_x: str, symbol_y: str) -> Dict[str, float]:
        """
        获取合约乘数（简化版本）
        
        Args:
            symbol_x: X品种代码
            symbol_y: Y品种代码
            
        Returns:
            合约乘数字典
        """
        # 这里使用默认值，实际应从配置获取
        multiplier_map = {
            'CU': 5,    # 铜 5吨/手
            'AL': 5,    # 铝 5吨/手
            'ZN': 5,    # 锌 5吨/手
            'PB': 5,    # 铅 5吨/手
            'NI': 1,    # 镍 1吨/手
            'SN': 1,    # 锡 1吨/手
            'AU': 1000, # 金 1000克/手
            'AG': 15,   # 银 15千克/手
            'RB': 10,   # 螺纹 10吨/手
            'HC': 10,   # 热卷 10吨/手
            'I': 100,   # 铁矿 100吨/手
            'SF': 5,    # 硅铁 5吨/手
            'SM': 5,    # 锰硅 5吨/手
            'SS': 5,    # 不锈钢 5吨/手
        }
        
        return {
            'x': multiplier_map.get(symbol_x, 1),
            'y': multiplier_map.get(symbol_y, 1)
        }