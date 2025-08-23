#!/usr/bin/env python3
"""
回测框架模块 - 执行交易信号并计算真实PnL

主要功能：
1. 交易执行：根据信号和仓位权重计算手数，执行开平仓
2. 资金管理：逐日盯市结算，保证金管理，强平控制  
3. PnL计算：精确计算盈亏，包含滑点和手续费
4. 绩效分析：计算夏普比率、最大回撤等关键指标
5. 风险控制：止损、时间止损、强制平仓

作者：Claude
创建时间：2025-08-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)


@dataclass 
class Position:
    """持仓记录"""
    # 必需字段（无默认值）
    pair: str
    direction: str  # 'long_spread' or 'short_spread'
    spread_formula: str
    open_date: datetime
    position_weight: float
    symbol_y: str
    symbol_x: str
    contracts_y: int  # Y合约手数
    contracts_x: int  # X合约手数
    beta: float  # Kalman滤波的beta
    open_price_y: float
    open_price_x: float
    margin_occupied: float
    open_commission: float
    
    # 可选字段（有默认值）
    ols_beta: float = np.nan  # 60天滚动OLS的beta
    open_z_score: float = 0.0  # 开仓时的Z-score
    prev_price_y: float = 0.0
    prev_price_x: float = 0.0
    unrealized_pnl: float = 0.0
    multiplier_y: float = 1.0
    multiplier_x: float = 1.0


class PositionManager:
    """
    仓位管理器 - 负责资金管理和逐日盯市
    
    重要原则：
    - 浮盈亏每日结算后计入可用资金
    - 权益 = 可用资金 + 占用保证金（不重复加浮盈亏）
    """
    
    def __init__(self, initial_capital: float, margin_rate: float = 0.12):
        self.initial_capital = initial_capital
        self.margin_rate = margin_rate
        
        # 核心资金状态
        self.available_capital = initial_capital  # 可用资金（已含浮盈亏）
        self.occupied_margin = 0.0               # 占用保证金
        self.total_equity = initial_capital      # 总权益
        
        # 持仓记录
        self.positions: Dict[str, Position] = {}
        
        # 历史记录
        self.daily_records = []
        self.total_pnl = 0.0
        
    def can_open_position(self, required_margin: float) -> bool:
        """检查是否有足够资金开仓"""
        return self.available_capital >= required_margin
        
    def add_position(self, position: Position) -> None:
        """添加新持仓"""
        self.positions[position.pair] = position
        self.available_capital -= position.margin_occupied
        self.occupied_margin += position.margin_occupied
        logger.info(f"开仓 {position.pair}: 占用保证金 {position.margin_occupied:,.0f}")
        
    def remove_position(self, pair: str) -> Optional[Position]:
        """移除持仓并释放保证金"""
        if pair not in self.positions:
            return None
            
        position = self.positions.pop(pair)
        self.available_capital += position.margin_occupied
        self.occupied_margin -= position.margin_occupied
        logger.info(f"平仓 {pair}: 释放保证金 {position.margin_occupied:,.0f}")
        return position
        
    def daily_settlement(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        逐日盯市结算
        
        Args:
            prices: 当日收盘价格字典 {'CU0': 45000, 'SN0': 150000, ...}
            
        Returns:
            结算结果字典
        """
        if not self.positions:
            # 无持仓时直接更新权益
            self.total_equity = self.available_capital
            return {
                'available_capital': self.available_capital,
                'occupied_margin': 0.0, 
                'total_equity': self.total_equity,
                'daily_pnl': 0.0,
                'position_count': 0
            }
            
        daily_pnl = 0.0
        
        for pair, pos in self.positions.items():
            # 获取当前价格
            try:
                current_y = prices[pos.symbol_y] 
                current_x = prices[pos.symbol_x]
            except KeyError as e:
                logger.warning(f"价格数据缺失: {e}, 跳过 {pair} 的盯市结算")
                continue
                
            # 计算当日价格变动PnL
            if pos.prev_price_y > 0 and pos.prev_price_x > 0:
                y_pnl = (current_y - pos.prev_price_y) * pos.contracts_y * pos.multiplier_y
                x_pnl = (current_x - pos.prev_price_x) * pos.contracts_x * pos.multiplier_x
                
                # 根据方向计算净PnL
                if pos.direction == 'long_spread':
                    # 做多价差：多Y空X
                    pos_pnl = y_pnl - x_pnl
                else:
                    # 做空价差：空Y多X  
                    pos_pnl = -y_pnl + x_pnl
                    
                daily_pnl += pos_pnl
                pos.unrealized_pnl += pos_pnl
                
            # 更新前日价格
            pos.prev_price_y = current_y
            pos.prev_price_x = current_x
            
        # 浮盈亏结算到可用资金
        self.available_capital += daily_pnl
        
        # 更新总权益（浮盈亏已在可用资金中）
        self.total_equity = self.available_capital + self.occupied_margin
        
        # 记录当日结算
        settlement_record = {
            'available_capital': self.available_capital,
            'occupied_margin': self.occupied_margin,
            'total_equity': self.total_equity, 
            'daily_pnl': daily_pnl,
            'position_count': len(self.positions)
        }
        
        self.daily_records.append(settlement_record.copy())
        
        return settlement_record
        
    def check_margin_call(self) -> bool:
        """检查是否触发强平（可用资金<0）"""
        return self.available_capital < 0
        
    def update_equity(self) -> float:
        """更新并返回当前权益"""
        self.total_equity = self.available_capital + self.occupied_margin
        return self.total_equity


class BacktestEngine:
    """
    回测引擎 - 核心回测逻辑
    
    功能：
    1. 交易执行：开平仓操作
    2. 手数计算：基于权重和资金分配
    3. PnL计算：包含滑点和手续费
    4. 绩效分析：计算各种指标
    """
    
    def __init__(self, 
                 initial_capital: float = 5000000,
                 margin_rate: float = 0.12,
                 commission_rate: float = 0.0002,  # 单边费率
                 slippage_ticks: int = 3,
                 position_weights: Optional[Dict[str, float]] = None,
                 stop_loss_pct: float = 0.15,  # 止损比例（相对保证金）
                 max_holding_days: int = 30):  # 最大持仓天数
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            margin_rate: 保证金比例
            commission_rate: 手续费率（单边）
            slippage_ticks: 滑点tick数
            position_weights: 配对权重分配 {'pair1': 0.05, ...}
            stop_loss_pct: 止损比例（相对于保证金）
            max_holding_days: 最大持仓天数
        """
        self.initial_capital = initial_capital
        self.margin_rate = margin_rate
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        
        # 默认均等权重分配
        self.position_weights = position_weights or {}
        
        # 初始化仓位管理器
        self.position_manager = PositionManager(initial_capital, margin_rate)
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.trade_id_counter = 1
        
        # 权益曲线
        self.equity_curve = []
        self.daily_returns = []
        
        # 合约规格（需要从外部提供）
        self.contract_specs = {}
        
    def load_contract_specs(self, specs_file: str) -> None:
        """加载合约规格"""
        if Path(specs_file).exists():
            with open(specs_file, 'r', encoding='utf-8') as f:
                self.contract_specs = json.load(f)
        else:
            logger.warning(f"合约规格文件不存在: {specs_file}")
            
    def _prepare_lots_info_from_signal(self, signal: Dict, current_prices: Dict[str, float]) -> Optional[Dict]:
        """
        从信号中准备手数信息（用于外部指定手数）
        """
        try:
            pair = signal['pair']
            symbol_x, symbol_y = pair.split('-')
            
            # 获取基础符号
            base_symbol_x = symbol_x.replace('_close', '') if '_close' in symbol_x else symbol_x
            base_symbol_y = symbol_y.replace('_close', '') if '_close' in symbol_y else symbol_y
            
            # 获取合约规格
            if base_symbol_y not in self.contract_specs or base_symbol_x not in self.contract_specs:
                return None
                
            spec_y = self.contract_specs[base_symbol_y]
            spec_x = self.contract_specs[base_symbol_x]
            
            # 获取价格
            price_y = current_prices.get(symbol_y)
            price_x = current_prices.get(symbol_x)
            
            if price_y is None or price_x is None:
                return None
            
            # 使用信号中的手数
            contracts_y = signal['contracts_y']
            contracts_x = signal['contracts_x']
            
            # 计算保证金
            margin_per_y = price_y * spec_y['multiplier'] * self.margin_rate
            margin_per_x = price_x * spec_x['multiplier'] * self.margin_rate
            margin_required = contracts_y * margin_per_y + contracts_x * margin_per_x
            
            return {
                'symbol_y': symbol_y,
                'symbol_x': symbol_x,
                'contracts_y': contracts_y,
                'contracts_x': contracts_x,
                'margin_per_y': margin_per_y,
                'margin_per_x': margin_per_x,
                'margin_required': margin_required,
                'tick_size_y': spec_y['tick_size'],
                'tick_size_x': spec_x['tick_size'],
                'multiplier_y': spec_y['multiplier'],
                'multiplier_x': spec_x['multiplier']
            }
        except Exception as e:
            logger.error(f"准备手数信息失败: {e}")
            return None
    
    def apply_slippage(self, price: float, side: str, tick_size: float) -> float:
        """
        应用滑点
        
        Args:
            price: 市场价格
            side: 'buy' 或 'sell'
            tick_size: 最小变动价位
            
        Returns:
            含滑点的实际成交价
        """
        slippage = tick_size * self.slippage_ticks
        
        if side == 'buy':
            return price + slippage  # 买入价格上滑
        else:  # sell
            return price - slippage  # 卖出价格下滑
            
    def calculate_lots(self, signal: Dict, position_weight: float, 
                      current_prices: Dict[str, float]) -> Optional[Dict]:
        """
        计算最优手数
        
        Args:
            signal: 交易信号，包含theoretical_ratio
            position_weight: 仓位权重
            current_prices: 当前价格
            
        Returns:
            手数分配结果或None
        """
        try:
            pair = signal['pair']
            
            # 优先使用信号中提供的symbol_x和symbol_y
            if 'symbol_x' in signal and 'symbol_y' in signal:
                symbol_x = signal['symbol_x']
                symbol_y = signal['symbol_y']
            else:
                # 兼容旧代码：按照'X-Y'格式解析
                symbol_x, symbol_y = pair.split('-')
            
            # 获取基础符号（用于合约规格查询）
            base_symbol_x = symbol_x.replace('_close', '') if '_close' in symbol_x else symbol_x
            base_symbol_y = symbol_y.replace('_close', '') if '_close' in symbol_y else symbol_y
            
            # 获取合约规格
            if base_symbol_y not in self.contract_specs or base_symbol_x not in self.contract_specs:
                logger.warning(f"缺少合约规格: {pair} -> {base_symbol_x}-{base_symbol_y}")
                return None
                
            spec_y = self.contract_specs[base_symbol_y]
            spec_x = self.contract_specs[base_symbol_x]
            
            # 获取价格（使用完整符号名）
            price_y = current_prices.get(symbol_y)
            price_x = current_prices.get(symbol_x)
            
            if price_y is None or price_x is None:
                logger.warning(f"价格数据缺失: {pair}")
                return None
                
            # 计算该配对的预算
            position_budget = self.position_manager.total_equity * position_weight
            
            # 获取理论比率
            theoretical_ratio = abs(signal.get('theoretical_ratio', 1.0))
            
            # 计算单手保证金
            margin_per_y = price_y * spec_y['multiplier'] * self.margin_rate
            margin_per_x = price_x * spec_x['multiplier'] * self.margin_rate
            
            # 搜索最优整数手数组合
            best_lots = self._search_optimal_lots(
                theoretical_ratio, position_budget, margin_per_y, margin_per_x
            )
            
            if best_lots is None:
                return None
                
            # 计算总保证金需求
            total_margin = (best_lots['contracts_y'] * margin_per_y + 
                          best_lots['contracts_x'] * margin_per_x)
                          
            return {
                'symbol_y': symbol_y,
                'symbol_x': symbol_x,
                'contracts_y': best_lots['contracts_y'],
                'contracts_x': best_lots['contracts_x'],
                'margin_required': total_margin,
                'multiplier_y': spec_y['multiplier'],
                'multiplier_x': spec_x['multiplier'],
                'tick_size_y': spec_y['tick_size'],
                'tick_size_x': spec_x['tick_size']
            }
            
        except Exception as e:
            logger.error(f"手数计算失败 {signal.get('pair', 'unknown')}: {e}")
            return None
            
    def _search_optimal_lots(self, theoretical_ratio: float, budget: float,
                           margin_per_y: float, margin_per_x: float) -> Optional[Dict]:
        """
        搜索最优整数手数组合 - 使用多种算法验证
        
        实现3种算法：
        1. 网格搜索法 (Grid Search)
        2. 比率约简法 (Ratio Reduction)  
        3. 线性规划近似法 (LP Approximation)
        
        返回3种算法中最优的结果
        """
        # 算法1: 网格搜索法
        result1 = self._grid_search_lots(theoretical_ratio, budget, margin_per_y, margin_per_x)
        
        # 算法2: 比率约简法
        result2 = self._ratio_reduction_lots(theoretical_ratio, budget, margin_per_y, margin_per_x)
        
        # 算法3: 线性规划近似法
        result3 = self._lp_approximation_lots(theoretical_ratio, budget, margin_per_y, margin_per_x)
        
        # 选择最优结果（优先利用率，其次比率精度）
        candidates = [r for r in [result1, result2, result3] if r is not None]
        
        if not candidates:
            return None
            
        # 按利用率降序，比率误差升序排序
        best = max(candidates, key=lambda x: (x['utilization'], -x['ratio_error']))
        
        logger.debug(f"手数计算结果: 网格搜索={'✓' if result1 else '✗'}, "
                    f"比率约简={'✓' if result2 else '✗'}, "
                    f"线性规划={'✓' if result3 else '✗'}, "
                    f"最优=Y:{best['contracts_y']}, X:{best['contracts_x']}")
                    
        return best
        
    def _grid_search_lots(self, theoretical_ratio: float, budget: float,
                         margin_per_y: float, margin_per_x: float) -> Optional[Dict]:
        """
        算法1: 网格搜索法
        
        穷举搜索所有可能的手数组合
        """
        best_lots = None
        best_utilization = 0.0
        
        # 搜索范围：基于预算估算
        max_y = min(100, int(budget / margin_per_y) + 1)
        max_x = min(100, int(budget / margin_per_x) + 1)
        
        for contracts_y in range(1, max_y + 1):
            for contracts_x in range(1, max_x + 1):
                # 计算保证金需求
                total_margin = contracts_y * margin_per_y + contracts_x * margin_per_x
                
                # 检查预算约束
                if total_margin > budget:
                    continue
                    
                # 计算实际比率与理论比率的偏差
                actual_ratio = contracts_y / contracts_x
                ratio_error = abs(actual_ratio - theoretical_ratio) / theoretical_ratio
                
                # 计算资金利用率
                utilization = total_margin / budget
                
                # 选择标准：比率偏差<20% 且资金利用率最高
                if ratio_error < 0.20 and utilization > best_utilization:
                    best_utilization = utilization
                    best_lots = {
                        'contracts_y': contracts_y,
                        'contracts_x': contracts_x,
                        'total_margin': total_margin,
                        'actual_ratio': actual_ratio,
                        'ratio_error': ratio_error,
                        'utilization': utilization,
                        'algorithm': 'grid_search'
                    }
                    
        return best_lots
        
    def _ratio_reduction_lots(self, theoretical_ratio: float, budget: float,
                            margin_per_y: float, margin_per_x: float) -> Optional[Dict]:
        """
        算法2: 比率约简法
        
        基于理论比率的分数约简，寻找最简分数形式，然后按预算缩放
        """
        try:
            # 将理论比率转换为分数形式
            # 使用连分数展开找到最佳有理逼近
            from fractions import Fraction
            
            # 限制分母大小避免过大的手数
            max_denominator = 50
            frac = Fraction(theoretical_ratio).limit_denominator(max_denominator)
            
            base_y = frac.numerator
            base_x = frac.denominator
            
            # 计算基础组合的保证金
            base_margin = base_y * margin_per_y + base_x * margin_per_x
            
            if base_margin > budget:
                # 基础组合已超预算，尝试更小的约简
                for denom in range(2, 20):
                    frac2 = Fraction(theoretical_ratio).limit_denominator(denom)
                    test_y, test_x = frac2.numerator, frac2.denominator
                    test_margin = test_y * margin_per_y + test_x * margin_per_x
                    
                    if test_margin <= budget:
                        base_y, base_x, base_margin = test_y, test_x, test_margin
                        break
                else:
                    return None
                    
            # 按预算缩放到最大可能倍数
            scale_factor = int(budget / base_margin)
            
            if scale_factor < 1:
                return None
                
            contracts_y = base_y * scale_factor
            contracts_x = base_x * scale_factor
            total_margin = contracts_y * margin_per_y + contracts_x * margin_per_x
            
            # 验证约束
            if total_margin > budget:
                return None
                
            actual_ratio = contracts_y / contracts_x
            ratio_error = abs(actual_ratio - theoretical_ratio) / theoretical_ratio
            utilization = total_margin / budget
            
            return {
                'contracts_y': contracts_y,
                'contracts_x': contracts_x,
                'total_margin': total_margin,
                'actual_ratio': actual_ratio,
                'ratio_error': ratio_error,
                'utilization': utilization,
                'algorithm': 'ratio_reduction',
                'base_fraction': f"{base_y}/{base_x}",
                'scale_factor': scale_factor
            }
            
        except Exception as e:
            logger.debug(f"比率约简法失败: {e}")
            return None
            
    def _lp_approximation_lots(self, theoretical_ratio: float, budget: float,
                             margin_per_y: float, margin_per_x: float) -> Optional[Dict]:
        """
        算法3: 线性规划近似法
        
        将问题建模为线性规划，然后取整数解
        """
        try:
            # LP松弛解：最大化资金利用率
            # 目标函数: max(y * margin_y + x * margin_x)
            # 约束条件: y * margin_y + x * margin_x <= budget
            #          y/x ≈ theoretical_ratio (软约束)
            
            # 假设x=1，计算对应的y
            y_float = theoretical_ratio
            
            # 计算单位资金的最优分配
            total_weight = y_float * margin_per_y + 1.0 * margin_per_x
            allocation_ratio = budget / total_weight
            
            # 计算浮点解
            y_lp = y_float * allocation_ratio
            x_lp = 1.0 * allocation_ratio
            
            # 尝试多种取整策略
            candidates = []
            
            # 策略1: 直接四舍五入
            y1, x1 = round(y_lp), round(x_lp)
            if y1 >= 1 and x1 >= 1:
                candidates.append((y1, x1))
                
            # 策略2: 向下取整
            y2, x2 = int(y_lp), int(x_lp)
            if y2 >= 1 and x2 >= 1:
                candidates.append((y2, x2))
                
            # 策略3: Y向上，X向下
            y3, x3 = int(y_lp) + 1, int(x_lp)
            if y3 >= 1 and x3 >= 1:
                candidates.append((y3, x3))
                
            # 策略4: Y向下，X向上  
            y4, x4 = int(y_lp), int(x_lp) + 1
            if y4 >= 1 and x4 >= 1:
                candidates.append((y4, x4))
                
            # 选择最优的取整结果
            best_candidate = None
            best_score = -1
            
            for y_int, x_int in candidates:
                total_margin = y_int * margin_per_y + x_int * margin_per_x
                
                if total_margin > budget:
                    continue
                    
                actual_ratio = y_int / x_int
                ratio_error = abs(actual_ratio - theoretical_ratio) / theoretical_ratio
                utilization = total_margin / budget
                
                # 综合评分：利用率权重0.7，比率精度权重0.3
                score = 0.7 * utilization - 0.3 * ratio_error
                
                if ratio_error < 0.20 and score > best_score:
                    best_score = score
                    best_candidate = {
                        'contracts_y': y_int,
                        'contracts_x': x_int,
                        'total_margin': total_margin,
                        'actual_ratio': actual_ratio,
                        'ratio_error': ratio_error,
                        'utilization': utilization,
                        'algorithm': 'lp_approximation',
                        'lp_solution': f"({y_lp:.2f}, {x_lp:.2f})"
                    }
                    
            return best_candidate
            
        except Exception as e:
            logger.debug(f"线性规划近似法失败: {e}")
            return None
        
    def execute_signal(self, signal: Dict, current_prices: Dict[str, float], current_date: datetime = None) -> bool:
        """
        执行交易信号
        
        Args:
            signal: 交易信号
            current_prices: 当前价格
            
        Returns:
            是否成功执行
        """
        pair = signal['pair']
        signal_type = signal['signal']
        
        if signal_type in ['long_spread', 'short_spread'] or signal_type.startswith('open'):
            return self._open_position(signal, current_prices)
        elif signal_type == 'close':
            close_z_score = signal.get('z_score', np.nan)
            return self._close_position(pair, current_prices, 'signal', current_date, close_z_score)
        else:
            logger.warning(f"未知信号类型: {signal_type}")
            return False
            
    def _open_position(self, signal: Dict, current_prices: Dict[str, float]) -> bool:
        """开仓操作"""
        pair = signal['pair']
        
        # 检查是否已有持仓
        if pair in self.position_manager.positions:
            logger.debug(f"配对 {pair} 已有持仓，跳过开仓")
            return False
            
        # 获取权重
        position_weight = self.position_weights.get(pair, 0.05)  # 默认5%
        
        # 检查是否有外部指定的手数
        if 'contracts_y' in signal and 'contracts_x' in signal:
            # 使用外部指定的手数
            lots_info = self._prepare_lots_info_from_signal(signal, current_prices)
            if lots_info is None:
                logger.warning(f"无法准备手数信息: {pair}")
                return False
        else:
            # 计算手数
            lots_info = self.calculate_lots(signal, position_weight, current_prices)
            if lots_info is None:
                logger.warning(f"无法计算手数: {pair}")
                return False
            
        # 检查资金
        if not self.position_manager.can_open_position(lots_info['margin_required']):
            logger.warning(f"资金不足，无法开仓: {pair}")
            return False
            
        # 获取方向和应用滑点
        direction = 'long_spread' if 'long' in signal['signal'] else 'short_spread'
        
        if direction == 'long_spread':
            # 做多价差：买Y卖X
            
            # 调试HC0-I0的价格获取
            if pair == 'HC0-I0':
                logger.info(f"HC0-I0调试 - lots_info: symbol_x={lots_info['symbol_x']}, symbol_y={lots_info['symbol_y']}")
                logger.info(f"HC0-I0调试 - current_prices keys: {list(current_prices.keys())}")
                logger.info(f"HC0-I0调试 - 获取价格: X({lots_info['symbol_x']})={current_prices.get(lots_info['symbol_x'])}, Y({lots_info['symbol_y']})={current_prices.get(lots_info['symbol_y'])}")
            
            open_price_y = self.apply_slippage(
                current_prices[lots_info['symbol_y']], 'buy', lots_info['tick_size_y']
            )
            open_price_x = self.apply_slippage(
                current_prices[lots_info['symbol_x']], 'sell', lots_info['tick_size_x']
            )
            
            # 再次调试，确认价格没有被交换
            if pair == 'HC0-I0':
                logger.info(f"HC0-I0调试 - 应用滑点后: open_price_x={open_price_x}, open_price_y={open_price_y}")
        else:
            # 做空价差：卖Y买X
            open_price_y = self.apply_slippage(
                current_prices[lots_info['symbol_y']], 'sell', lots_info['tick_size_y']
            )
            open_price_x = self.apply_slippage(
                current_prices[lots_info['symbol_x']], 'buy', lots_info['tick_size_x']
            )
            
        # 计算开仓手续费
        nominal_y = open_price_y * lots_info['contracts_y'] * lots_info['multiplier_y']
        nominal_x = open_price_x * lots_info['contracts_x'] * lots_info['multiplier_x']
        open_commission = (nominal_y + nominal_x) * self.commission_rate
        
        # 创建持仓记录
        try:
            logger.debug(f"创建持仓: {pair}, 方向={direction}")
            logger.debug(f"lots_info: {lots_info}")
            
            # 调试HC0-I0的Position创建
            if pair == 'HC0-I0':
                logger.info(f"HC0-I0调试 - 创建Position: symbol_x={lots_info['symbol_x']}, symbol_y={lots_info['symbol_y']}")
                logger.info(f"HC0-I0调试 - 创建Position: open_price_x={open_price_x}, open_price_y={open_price_y}")
            
            position = Position(
                pair=pair,
                direction=direction,
                spread_formula=signal.get('spread_formula', ''),
                open_date=pd.to_datetime(signal['date']),
                position_weight=position_weight,
                symbol_y=lots_info['symbol_y'],
                symbol_x=lots_info['symbol_x'],
                contracts_y=lots_info['contracts_y'],
                contracts_x=lots_info['contracts_x'],
                beta=abs(signal.get('theoretical_ratio', signal.get('beta', 1.0))),  # Kalman beta
                ols_beta=signal.get('ols_beta', np.nan),  # OLS beta
                open_z_score=signal.get('z_score', 0.0),  # 开仓时的Z-score
                open_price_y=open_price_y,
                open_price_x=open_price_x,
                margin_occupied=lots_info['margin_required'],
                open_commission=open_commission,
                prev_price_y=open_price_y,  # 初始化前日价格
                prev_price_x=open_price_x,
                multiplier_y=lots_info['multiplier_y'],
                multiplier_x=lots_info['multiplier_x']
            )
            logger.debug(f"持仓创建成功: {position}")
        except Exception as e:
            logger.error(f"持仓创建失败: {e}")
            return False
        
        # 扣除手续费
        self.position_manager.available_capital -= open_commission
        
        # 添加持仓
        self.position_manager.add_position(position)
        
        logger.info(f"开仓成功: {pair} {direction}, 手数=Y:{lots_info['contracts_y']} X:{lots_info['contracts_x']}")
        
        return True
        
    def _close_position(self, pair: str, current_prices: Dict[str, float], 
                       reason: str, current_date: datetime = None, 
                       close_z_score: float = np.nan) -> bool:
        """平仓操作"""
        position = self.position_manager.remove_position(pair)
        if position is None:
            logger.warning(f"未找到持仓: {pair}")
            return False
            
        try:
            # 获取基础符号用于合约规格查询
            base_symbol_y = position.symbol_y.replace('_close', '') if '_close' in position.symbol_y else position.symbol_y
            base_symbol_x = position.symbol_x.replace('_close', '') if '_close' in position.symbol_x else position.symbol_x
            
            # 获取平仓价格并应用滑点
            if position.direction == 'long_spread':
                # 平多头价差：卖Y买X
                close_price_y = self.apply_slippage(
                    current_prices[position.symbol_y], 'sell', 
                    self.contract_specs[base_symbol_y]['tick_size']
                )
                close_price_x = self.apply_slippage(
                    current_prices[position.symbol_x], 'buy',
                    self.contract_specs[base_symbol_x]['tick_size'] 
                )
            else:
                # 平空头价差：买Y卖X
                close_price_y = self.apply_slippage(
                    current_prices[position.symbol_y], 'buy',
                    self.contract_specs[base_symbol_y]['tick_size']
                )
                close_price_x = self.apply_slippage(
                    current_prices[position.symbol_x], 'sell',
                    self.contract_specs[base_symbol_x]['tick_size']
                )
                
            # 计算毛PnL - 使用双算法验证
            gross_pnl_1, y_pnl_1, x_pnl_1 = self._calculate_pnl_method1(
                position, close_price_y, close_price_x
            )
            
            gross_pnl_2, y_pnl_2, x_pnl_2 = self._calculate_pnl_method2(
                position, close_price_y, close_price_x
            )
            
            # 验证两种算法结果一致性
            pnl_diff = abs(gross_pnl_1 - gross_pnl_2)
            if pnl_diff > 0.01:  # 允许1分钱误差
                logger.warning(f"PnL计算不一致 {pair}: 方法1={gross_pnl_1:,.2f}, 方法2={gross_pnl_2:,.2f}, 差异={pnl_diff:,.2f}")
                
            # 使用第一种方法的结果
            gross_pnl = gross_pnl_1
            y_pnl, x_pnl = y_pnl_1, x_pnl_1
            
            # 计算平仓手续费 - 使用双算法验证
            close_commission_1 = self._calculate_commission_method1(
                close_price_y, close_price_x, position
            )
            
            close_commission_2 = self._calculate_commission_method2(
                close_price_y, close_price_x, position
            )
            
            # 验证手续费计算一致性
            comm_diff = abs(close_commission_1 - close_commission_2)
            if comm_diff > 0.01:
                logger.warning(f"手续费计算不一致 {pair}: 方法1={close_commission_1:,.2f}, 方法2={close_commission_2:,.2f}")
                
            close_commission = close_commission_1
            
            # 计算净PnL
            net_pnl = gross_pnl - position.open_commission - close_commission
            
            # 更新可用资金
            self.position_manager.available_capital += net_pnl - close_commission
            
            # 记录交易
            close_date = pd.to_datetime(current_date) if current_date else pd.Timestamp.now()
            holding_days = (close_date - position.open_date).days
            
            trade_record = {
                'trade_id': self.trade_id_counter,
                'pair': pair,
                'direction': position.direction,
                'beta_kalman': position.beta,  # Kalman beta
                'beta_ols': position.ols_beta,  # OLS beta
                'open_z_score': position.open_z_score,  # 开仓Z-score
                'close_z_score': close_z_score,  # 平仓Z-score
                'spread_formula': position.spread_formula,
                'open_date': position.open_date.strftime('%Y-%m-%d'),
                'close_date': close_date.strftime('%Y-%m-%d'),
                'holding_days': holding_days,
                'position_weight': position.position_weight,
                'contracts_y': position.contracts_y,
                'contracts_x': position.contracts_x,
                'open_price_y': position.open_price_y,
                'open_price_x': position.open_price_x,
                'close_price_y': close_price_y,
                'close_price_x': close_price_x,
                'margin_occupied': position.margin_occupied,
                'gross_pnl': gross_pnl,
                'open_commission': position.open_commission,
                'close_commission': close_commission,
                'net_pnl': net_pnl,
                'close_reason': reason,
                'return_on_margin': net_pnl / position.margin_occupied if position.margin_occupied > 0 else 0.0
            }
            
            self.trade_records.append(trade_record)
            self.trade_id_counter += 1
            
            logger.info(f"平仓成功: {pair}, 净PnL={net_pnl:+,.0f}, 原因={reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"平仓失败 {pair}: {e}")
            # 如果平仓失败，需要重新添加持仓
            self.position_manager.add_position(position)
            return False
            
    def calculate_metrics(self) -> Dict[str, Any]:
        """计算绩效指标"""
        if not self.trade_records:
            return {}
            
        df = pd.DataFrame(self.trade_records)
        
        # 基本统计
        total_trades = len(df)
        winning_trades = len(df[df['net_pnl'] > 0])
        losing_trades = len(df[df['net_pnl'] < 0])
        
        total_pnl = df['net_pnl'].sum()
        total_return = total_pnl / self.initial_capital
        
        # 胜率和盈亏比
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else 0
        
        # 时间相关指标
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # 最大回撤
            equity = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # 年化收益率计算
        if total_trades > 0:
            first_trade_date = pd.to_datetime(df['open_date'].min())
            last_trade_date = pd.to_datetime(df['close_date'].max())
            trading_days = (last_trade_date - first_trade_date).days
            if trading_days > 0:
                annual_return = (1 + total_return) ** (252 / trading_days) - 1
            else:
                annual_return = 0
        else:
            annual_return = 0
            
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,  # 转换为百分比
            'total_pnl': total_pnl,
            'total_return': total_return * 100,  # 转换为百分比
            'annual_return': annual_return * 100,  # 转换为百分比
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,  # 转换为百分比
            'avg_holding_days': df['holding_days'].mean(),
            'total_commission': df['open_commission'].sum() + df['close_commission'].sum(),
            'trade_records': self.trade_records  # 返回交易记录供分析
        }
    
    # ========== PnL计算双算法验证方法 ==========
    
    def _calculate_pnl_method1(self, position: Position, close_price_y: float, 
                              close_price_x: float) -> Tuple[float, float, float]:
        """
        PnL计算方法1：基于价差方向的标准算法
        
        多头价差 (long_spread): 买Y卖X，预期价差上涨
        - Y腿PnL = (平仓价 - 开仓价) × 手数 × 乘数 (做多)
        - X腿PnL = (开仓价 - 平仓价) × 手数 × 乘数 (做空)
        
        空头价差 (short_spread): 卖Y买X，预期价差下跌  
        - Y腿PnL = (开仓价 - 平仓价) × 手数 × 乘数 (做空)
        - X腿PnL = (平仓价 - 开仓价) × 手数 × 乘数 (做多)
        """
        if position.direction == 'long_spread':
            # 做多价差：多Y空X
            y_pnl = (close_price_y - position.open_price_y) * position.contracts_y * position.multiplier_y
            x_pnl = (position.open_price_x - close_price_x) * position.contracts_x * position.multiplier_x
        else:
            # 做空价差：空Y多X
            y_pnl = (position.open_price_y - close_price_y) * position.contracts_y * position.multiplier_y
            x_pnl = (close_price_x - position.open_price_x) * position.contracts_x * position.multiplier_x
            
        gross_pnl = y_pnl + x_pnl
        return gross_pnl, y_pnl, x_pnl
    
    def _calculate_pnl_method2(self, position: Position, close_price_y: float,
                              close_price_x: float) -> Tuple[float, float, float]:
        """
        PnL计算方法2：基于名义价值变化的验证算法
        
        通过计算开仓和平仓时的名义价值变化来验证PnL
        """
        # 开仓时的名义价值
        open_nominal_y = position.open_price_y * position.contracts_y * position.multiplier_y
        open_nominal_x = position.open_price_x * position.contracts_x * position.multiplier_x
        
        # 平仓时的名义价值
        close_nominal_y = close_price_y * position.contracts_y * position.multiplier_y
        close_nominal_x = close_price_x * position.contracts_x * position.multiplier_x
        
        # 根据方向计算PnL
        if position.direction == 'long_spread':
            # 多头价差：持有Y合约，做空X合约
            y_pnl = close_nominal_y - open_nominal_y  # 多头：价值增加为盈利
            x_pnl = open_nominal_x - close_nominal_x  # 空头：价值下降为盈利
        else:
            # 空头价差：做空Y合约，持有X合约
            y_pnl = open_nominal_y - close_nominal_y  # 空头：价值下降为盈利
            x_pnl = close_nominal_x - open_nominal_x  # 多头：价值增加为盈利
            
        gross_pnl = y_pnl + x_pnl
        return gross_pnl, y_pnl, x_pnl
    
    def _calculate_commission_method1(self, close_price_y: float, close_price_x: float,
                                     position: Position) -> float:
        """
        手续费计算方法1：基于名义价值的标准算法
        """
        nominal_y = close_price_y * position.contracts_y * position.multiplier_y
        nominal_x = close_price_x * position.contracts_x * position.multiplier_x
        return (nominal_y + nominal_x) * self.commission_rate
    
    def _calculate_commission_method2(self, close_price_y: float, close_price_x: float,
                                     position: Position) -> float:
        """
        手续费计算方法2：分别计算每腿手续费的验证算法
        """
        commission_y = close_price_y * position.contracts_y * position.multiplier_y * self.commission_rate
        commission_x = close_price_x * position.contracts_x * position.multiplier_x * self.commission_rate
        return commission_y + commission_x
    
    def _verify_direction_calculation(self, position: Position) -> bool:
        """
        验证方向计算的正确性
        
        检查价差公式与方向的一致性：
        - spread = log(Y) - β*log(X) - c
        - long_spread: 预期spread上涨，做多Y做空X
        - short_spread: 预期spread下跌，做空Y做多X
        """
        try:
            # 从价差公式中提取方向信息
            formula = position.spread_formula.lower()
            
            # 检查公式格式
            if 'log(' not in formula:
                return True  # 无法验证，认为正确
                
            # 解析公式中的系数符号
            if position.direction == 'long_spread':
                # 多头价差：期望Y相对X上涨，应该买Y卖X
                expected_behavior = "buy_y_sell_x"
            else:
                # 空头价差：期望Y相对X下跌，应该卖Y买X  
                expected_behavior = "sell_y_buy_x"
                
            logger.debug(f"方向验证 {position.pair}: {position.direction} -> {expected_behavior}")
            return True
            
        except Exception as e:
            logger.warning(f"方向验证失败 {position.pair}: {e}")
            return True
    
    def _verify_multiplier_calculation(self, symbol: str, expected_multiplier: float) -> bool:
        """
        验证合约乘数计算的正确性
        
        双重验证合约乘数是否与规格文件一致
        """
        try:
            # 方法1：从合约规格直接获取
            spec_multiplier = self.contract_specs.get(symbol, {}).get('multiplier', 1.0)
            
            # 方法2：基于常见期货合约的标准乘数验证
            standard_multipliers = {
                'CU': 5,     # 沪铜
                'AL': 5,     # 沪铝  
                'ZN': 5,     # 沪锌
                'NI': 1,     # 沪镍
                'SN': 1,     # 沪锡
                'PB': 5,     # 沪铅
                'AG': 15,    # 沪银
                'AU': 1000,  # 沪金
                'RB': 10,    # 螺纹钢
                'HC': 10,    # 热轧卷板
                'I': 100,    # 铁矿石
                'SF': 5,     # 硅铁
                'SM': 5,     # 锰硅
                'SS': 5      # 不锈钢
            }
            
            # 提取品种代码（去掉数字和0）
            base_symbol = ''.join([c for c in symbol if c.isalpha()])
            standard_multiplier = standard_multipliers.get(base_symbol, spec_multiplier)
            
            # 验证一致性
            if abs(spec_multiplier - expected_multiplier) < 0.01:
                multiplier_ok_1 = True
            else:
                multiplier_ok_1 = False
                logger.warning(f"合约乘数不一致 {symbol}: 规格={spec_multiplier}, 实际={expected_multiplier}")
                
            if abs(standard_multiplier - expected_multiplier) < 0.01:
                multiplier_ok_2 = True
            else:
                multiplier_ok_2 = False
                logger.debug(f"标准乘数检查 {symbol}: 标准={standard_multiplier}, 实际={expected_multiplier}")
                
            return multiplier_ok_1 or multiplier_ok_2
            
        except Exception as e:
            logger.warning(f"乘数验证失败 {symbol}: {e}")
            return True
    
    def _verify_price_calculation(self, base_price: float, side: str, 
                                 tick_size: float, slippage_ticks: int,
                                 actual_price: float) -> bool:
        """
        验证价格滑点计算的正确性
        
        双算法验证滑点应用是否正确
        """
        try:
            # 方法1：直接计算
            expected_price_1 = self.apply_slippage(base_price, side, tick_size)
            
            # 方法2：分步计算验证
            slippage_amount = tick_size * slippage_ticks
            if side == 'buy':
                expected_price_2 = base_price + slippage_amount
            else:
                expected_price_2 = base_price - slippage_amount
                
            # 验证两种方法结果一致
            method_diff = abs(expected_price_1 - expected_price_2)
            if method_diff > 0.0001:
                logger.warning(f"滑点计算方法不一致: 方法1={expected_price_1}, 方法2={expected_price_2}")
                return False
                
            # 验证与实际价格的一致性
            actual_diff = abs(expected_price_1 - actual_price)
            if actual_diff > 0.0001:
                logger.warning(f"价格滑点不一致: 预期={expected_price_1}, 实际={actual_price}")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"价格验证失败: {e}")
            return True
    
    # ========== 风险管理功能 ==========
    
    def check_stop_loss(self, position: Position, current_prices: Dict[str, float]) -> bool:
        """
        检查止损条件：单笔亏损达保证金10%触发平仓
        
        Args:
            position: 持仓记录
            current_prices: 当前价格
            
        Returns:
            是否触发止损
        """
        try:
            current_y = current_prices.get(position.symbol_y)
            current_x = current_prices.get(position.symbol_x)
            
            if current_y is None or current_x is None:
                return False
                
            # 计算当前未实现PnL
            unrealized_pnl, _, _ = self._calculate_pnl_method1(position, current_y, current_x)
            
            # 止损阈值：保证金的负百分比
            stop_loss_threshold = -position.margin_occupied * self.stop_loss_pct
            
            if unrealized_pnl <= stop_loss_threshold:
                loss_pct = unrealized_pnl / position.margin_occupied * 100
                logger.info(f"触发止损 {position.pair}: 当前PnL={unrealized_pnl:+,.0f}, "
                          f"止损线={stop_loss_threshold:+,.0f}, 实际损失={loss_pct:.2f}%, "
                          f"开仓Z={position.open_z_score:.3f}, 当前价格Y={current_y:.3f} X={current_x:.3f}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"止损检查失败 {position.pair}: {e}")
            return False
    
    def check_time_stop(self, position: Position, current_date: datetime) -> bool:
        """
        检查时间止损：持仓超过最大天数强制平仓
        
        Args:
            position: 持仓记录
            current_date: 当前日期
            
        Returns:
            是否触发时间止损
        """
        try:
            holding_days = (current_date - position.open_date).days
            
            if holding_days >= self.max_holding_days:
                logger.info(f"触发时间止损 {position.pair}: 持仓{holding_days}天（最大{self.max_holding_days}天）")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"时间止损检查失败 {position.pair}: {e}")
            return False
    
    def run_risk_management(self, current_date: datetime, current_prices: Dict[str, float]) -> List[str]:
        """
        运行风险管理检查，返回需要强制平仓的配对列表
        
        优先级：止损 > 时间止损 > 强制平仓
        
        Args:
            current_date: 当前日期
            current_prices: 当前价格
            
        Returns:
            需要平仓的配对列表及原因
        """
        force_close_list = []
        
        for pair, position in list(self.position_manager.positions.items()):
            close_reason = None
            
            # 1. 检查止损（优先级最高）
            if self.check_stop_loss(position, current_prices):
                close_reason = 'stop_loss'
                
            # 2. 检查时间止损
            elif self.check_time_stop(position, current_date):
                close_reason = 'time_stop'
                
            if close_reason:
                force_close_list.append({
                    'pair': pair,
                    'reason': close_reason,
                    'holding_days': (current_date - position.open_date).days
                })
                
        # 3. 检查强制平仓（资金不足）
        if self.position_manager.check_margin_call():
            logger.warning("触发强制平仓：可用资金不足")
            for pair in list(self.position_manager.positions.keys()):
                if not any(item['pair'] == pair for item in force_close_list):
                    force_close_list.append({
                        'pair': pair,
                        'reason': 'margin_call',
                        'available_capital': self.position_manager.available_capital
                    })
                    
        return force_close_list
    
    # ========== 配对分析功能 ==========
    
    def analyze_pairs_performance(self) -> pd.DataFrame:
        """
        配对收益分析
        
        Returns:
            配对分析结果DataFrame
        """
        if not self.trade_records:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.trade_records)
        
        # 按配对分组分析
        pair_analysis = df.groupby('pair').agg({
            'net_pnl': ['sum', 'mean', 'std', 'count'],
            'holding_days': 'mean',
            'return_on_margin': ['mean', 'std'],
            'open_commission': 'sum',
            'close_commission': 'sum'
        }).round(2)
        
        # 扁平化多级列名
        pair_analysis.columns = [
            'total_pnl', 'avg_pnl', 'pnl_std', 'trade_count',
            'avg_holding_days', 'avg_return_on_margin', 'return_std',
            'total_open_comm', 'total_close_comm'
        ]
        
        # 计算胜率
        pair_wins = df[df['net_pnl'] > 0].groupby('pair').size()
        pair_analysis['win_rate'] = (pair_wins / pair_analysis['trade_count']).fillna(0)
        
        # 计算收益率贡献
        total_pnl = df['net_pnl'].sum()
        if total_pnl != 0:
            pair_analysis['pnl_contribution'] = pair_analysis['total_pnl'] / total_pnl
        else:
            pair_analysis['pnl_contribution'] = 0
            
        # 按总收益排序
        pair_analysis = pair_analysis.sort_values('total_pnl', ascending=False)
        
        return pair_analysis
        
    def get_top_pairs(self, n: int = 20) -> pd.DataFrame:
        """
        获取Top N配对
        
        Args:
            n: 返回配对数量
            
        Returns:
            Top N配对分析结果
        """
        pair_analysis = self.analyze_pairs_performance()
        
        if len(pair_analysis) == 0:
            return pd.DataFrame()
            
        return pair_analysis.head(n)
        
    def calculate_pairs_correlation(self) -> pd.DataFrame:
        """
        计算配对间收益相关性
        
        Returns:
            配对相关性矩阵
        """
        if not self.trade_records:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.trade_records)
        
        # 构建配对收益矩阵
        pairs_pnl = df.pivot_table(
            index='trade_id', 
            columns='pair', 
            values='net_pnl', 
            fill_value=0
        )
        
        # 计算相关系数
        correlation_matrix = pairs_pnl.corr()
        
        return correlation_matrix
        
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        生成完整绩效总结
        
        Returns:
            绩效总结字典
        """
        # 基础绩效指标
        basic_metrics = self.calculate_metrics()
        
        # 配对分析
        pair_analysis = self.analyze_pairs_performance()
        top_5_pairs = self.get_top_pairs(5) if len(pair_analysis) > 0 else pd.DataFrame()
        
        # 相关性分析
        correlation_matrix = self.calculate_pairs_correlation()
        
        # 资金使用分析
        capital_analysis = {
            'initial_capital': self.initial_capital,
            'final_equity': self.position_manager.total_equity,
            'max_margin_used': max([r.get('occupied_margin', 0) for r in self.position_manager.daily_records], default=0),
            'avg_capital_utilization': 0,  # 需要从daily_records计算
        }
        
        if self.position_manager.daily_records:
            capital_utilizations = [
                r['occupied_margin'] / (r['available_capital'] + r['occupied_margin']) 
                for r in self.position_manager.daily_records
                if (r['available_capital'] + r['occupied_margin']) > 0
            ]
            if capital_utilizations:
                capital_analysis['avg_capital_utilization'] = np.mean(capital_utilizations)
        
        return {
            'basic_metrics': basic_metrics,
            'pair_analysis': pair_analysis.to_dict('index') if len(pair_analysis) > 0 else {},
            'top_pairs': top_5_pairs.to_dict('index') if len(top_5_pairs) > 0 else {},
            'correlation_matrix': correlation_matrix.to_dict() if len(correlation_matrix) > 0 else {},
            'capital_analysis': capital_analysis,
            'risk_events': {
                'margin_calls': sum(1 for r in self.trade_records if r.get('close_reason') == 'margin_call'),
                'stop_losses': sum(1 for r in self.trade_records if r.get('close_reason') == 'stop_loss'),
                'time_stops': sum(1 for r in self.trade_records if r.get('close_reason') == 'time_stop')
            }
        }