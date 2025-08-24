#!/usr/bin/env python3
"""
回测框架v4 - 参数化版本
与需求文档完全对齐，支持灵活的参数配置

主要功能：
1. 参数化配置：所有关键参数可配置
2. 基于动态β值的最小整数比手数计算
3. 精确的PnL计算（含手续费和滑点）
4. 风险控制：15%止损和30天强制平仓
5. 与信号生成模块输出格式完全对齐

作者：Claude
创建时间：2025-08-23
"""

import pandas as pd
import numpy as np
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置参数（所有可调参数）"""
    
    # ========== 资金管理参数 ==========
    initial_capital: float = 5000000      # 初始资金
    margin_rate: float = 0.12             # 保证金率
    position_weight: float = 0.05         # 默认仓位权重（每个配对）
    position_weights: Dict[str, float] = field(default_factory=dict)  # 配对个性化权重
    
    # ========== 交易成本参数 ==========
    commission_rate: float = 0.0002       # 手续费率（单边）
    slippage_ticks: int = 3              # 滑点tick数
    
    # ========== 风险控制参数 ==========
    stop_loss_pct: float = 0.15          # 止损比例（相对保证金）
    max_holding_days: int = 30           # 最大持仓天数
    enable_stop_loss: bool = True        # 是否启用止损
    enable_time_stop: bool = True        # 是否启用时间止损
    
    # ========== 信号参数 ==========
    z_open_threshold: float = 2.0        # 开仓Z-score阈值
    z_close_threshold: float = 0.5       # 平仓Z-score阈值
    
    # ========== 手数计算参数 ==========
    max_denominator: int = 10            # 最大分母（用于Fraction）
    min_lots: int = 1                    # 最小手数
    max_lots_per_leg: int = 100          # 每腿最大手数限制
    
    # ========== 执行控制参数 ==========
    allow_multiple_positions: bool = False  # 是否允许同一配对多个持仓
    force_close_at_end: bool = True        # 回测结束时是否强制平仓
    
    # ========== 输出参数 ==========
    save_trades: bool = True             # 是否保存交易记录
    save_daily_pnl: bool = True          # 是否保存每日PnL
    output_dir: str = "output/backtest"  # 输出目录


@dataclass
class Position:
    """持仓记录"""
    pair: str
    symbol_x: str
    symbol_y: str
    direction: str              # 'open_long' or 'open_short'
    open_date: datetime
    
    # 手数信息（基于β计算）
    beta: float                 # 动态β值
    lots_x: int                # X品种手数
    lots_y: int                # Y品种手数
    theoretical_ratio: float    # 理论比例
    actual_ratio: float        # 实际比例
    
    # 价格信息（含滑点）
    open_price_x: float
    open_price_y: float
    
    # 保证金和成本
    margin_occupied: float
    open_commission: float
    
    # 合约规格
    multiplier_x: float
    multiplier_y: float
    tick_size_x: float
    tick_size_y: float
    
    # 浮动盈亏
    unrealized_pnl: float = 0.0
    max_loss: float = 0.0      # 记录最大浮亏（用于止损）
    
    # 其他信息
    z_score_open: float = 0.0  # 开仓时的Z-score
    holding_days: int = 0       # 持仓天数


class BacktestEngine:
    """
    参数化回测引擎
    
    核心原则：
    1. 所有关键参数可配置
    2. 与信号生成模块输出格式对齐
    3. 基于动态β值计算手数
    4. 精确的PnL和风险管理
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置，None则使用默认配置
        """
        self.config = config or BacktestConfig()
        
        # 资金状态
        self.available_capital = self.config.initial_capital
        self.occupied_margin = 0.0
        self.total_equity = self.config.initial_capital
        
        # 持仓管理
        self.positions: Dict[str, Position] = {}
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.trade_id = 1
        
        # 绩效记录
        self.equity_curve = []
        self.daily_pnl = []
        
        # 合约规格（需要外部加载）
        self.contract_specs = {}
        
        logger.info(f"回测引擎初始化完成")
        logger.info(f"  初始资金: {self.config.initial_capital:,.0f}")
        logger.info(f"  保证金率: {self.config.margin_rate:.1%}")
        logger.info(f"  手续费率: {self.config.commission_rate:.4%}")
        logger.info(f"  止损线: {self.config.stop_loss_pct:.1%}")
        
    def load_contract_specs(self, specs_file: str) -> None:
        """加载合约规格"""
        specs_path = Path(specs_file)
        if specs_path.exists():
            with open(specs_path, 'r', encoding='utf-8') as f:
                self.contract_specs = json.load(f)
            logger.info(f"加载合约规格: {len(self.contract_specs)}个品种")
        else:
            logger.error(f"合约规格文件不存在: {specs_file}")
    
    def calculate_min_lots(self, beta: float) -> Dict:
        """
        根据β值计算最小整数比手数（REQ-4.1.1）
        
        Args:
            beta: 动态β值（来自信号）
            
        Returns:
            手数分配结果
        """
        if beta <= 0:
            logger.warning(f"β值异常: {beta}, 使用1:1")
            return {
                'lots_y': self.config.min_lots,
                'lots_x': self.config.min_lots,
                'theoretical_ratio': abs(beta),
                'actual_ratio': 1.0,
                'error': abs(1.0 - abs(beta))
            }
        
        # 使用Fraction找最简分数
        # Y:X = 1:β
        frac = Fraction(beta).limit_denominator(self.config.max_denominator)
        
        lots_y = frac.denominator
        lots_x = frac.numerator
        
        # 确保满足最小手数要求
        if lots_y < self.config.min_lots:
            scale = self.config.min_lots / lots_y
            lots_y = self.config.min_lots
            lots_x = max(self.config.min_lots, int(lots_x * scale))
        
        if lots_x < self.config.min_lots:
            if lots_x == 0:
                lots_x = self.config.min_lots
                lots_y = max(self.config.min_lots, lots_y)
            else:
                scale = self.config.min_lots / lots_x
                lots_x = self.config.min_lots
                lots_y = max(self.config.min_lots, int(lots_y * scale))
        
        # 限制最大手数
        if lots_x > self.config.max_lots_per_leg:
            scale = self.config.max_lots_per_leg / lots_x
            lots_x = self.config.max_lots_per_leg
            lots_y = max(self.config.min_lots, int(lots_y * scale))
        
        if lots_y > self.config.max_lots_per_leg:
            scale = self.config.max_lots_per_leg / lots_y
            lots_y = self.config.max_lots_per_leg
            lots_x = max(self.config.min_lots, int(lots_x * scale))
        
        actual_ratio = lots_x / lots_y if lots_y > 0 else 0
        error = abs(actual_ratio - beta) / beta if beta != 0 else 0
        
        return {
            'lots_y': int(lots_y),
            'lots_x': int(lots_x),
            'theoretical_ratio': beta,
            'actual_ratio': actual_ratio,
            'error': error
        }
    
    def apply_slippage(self, price: float, side: str, tick_size: float) -> float:
        """
        应用滑点（REQ-4.1.4）
        
        Args:
            price: 市场价格
            side: 'buy' 或 'sell'
            tick_size: 最小变动价位
        """
        slippage = tick_size * self.config.slippage_ticks
        
        if side == 'buy':
            return price + slippage
        else:  # sell
            return price - slippage
    
    def process_signal(self, signal: Dict, current_prices: Dict[str, float]) -> bool:
        """
        处理交易信号
        
        Args:
            signal: 来自信号生成模块的信号（13个字段）
            current_prices: 当前价格字典
            
        Returns:
            是否成功处理
        """
        pair = signal['pair']
        signal_type = signal['signal']
        z_score = signal.get('z_score', 0)
        
        # 跳过非交易信号
        if signal_type in ['converging', 'hold']:
            return True
        
        # 处理平仓信号
        if signal_type == 'close' or abs(z_score) < self.config.z_close_threshold:
            if pair in self.positions:
                return self._close_position(pair, current_prices, 'signal', signal['date'])
            return True
        
        # 处理开仓信号
        if signal_type in ['open_long', 'open_short']:
            # 检查Z-score阈值
            if abs(z_score) < self.config.z_open_threshold:
                logger.debug(f"Z-score {z_score:.2f} 未达到开仓阈值")
                return False
            
            # 检查是否已有持仓
            if pair in self.positions and not self.config.allow_multiple_positions:
                logger.debug(f"{pair} 已有持仓，跳过")
                return False
            
            return self._open_position(signal, current_prices)
        
        return False
    
    def _open_position(self, signal: Dict, current_prices: Dict[str, float]) -> bool:
        """
        开仓操作
        
        Args:
            signal: 交易信号
            current_prices: 当前价格
        """
        pair = signal['pair']
        symbol_x = signal['symbol_x']
        symbol_y = signal['symbol_y']
        beta = abs(signal['beta'])  # 使用动态β值
        
        # 检查合约规格
        if symbol_x not in self.contract_specs or symbol_y not in self.contract_specs:
            logger.warning(f"缺少合约规格: {symbol_x} 或 {symbol_y}")
            return False
        
        spec_x = self.contract_specs[symbol_x]
        spec_y = self.contract_specs[symbol_y]
        
        # 计算手数
        lots_result = self.calculate_min_lots(beta)
        
        # 获取价格
        price_x = current_prices.get(symbol_x)
        price_y = current_prices.get(symbol_y)
        
        if price_x is None or price_y is None:
            logger.warning(f"缺少价格: {symbol_x}={price_x}, {symbol_y}={price_y}")
            return False
        
        # 计算保证金
        margin_x = price_x * lots_result['lots_x'] * spec_x['multiplier'] * self.config.margin_rate
        margin_y = price_y * lots_result['lots_y'] * spec_y['multiplier'] * self.config.margin_rate
        margin_required = margin_x + margin_y
        
        # 检查资金
        if self.available_capital < margin_required:
            logger.warning(f"资金不足: 需要{margin_required:,.0f}, 可用{self.available_capital:,.0f}")
            return False
        
        # 应用滑点
        if signal['signal'] == 'open_long':
            # 做多价差：买Y卖X
            open_price_y = self.apply_slippage(price_y, 'buy', spec_y['tick_size'])
            open_price_x = self.apply_slippage(price_x, 'sell', spec_x['tick_size'])
        else:  # open_short
            # 做空价差：卖Y买X
            open_price_y = self.apply_slippage(price_y, 'sell', spec_y['tick_size'])
            open_price_x = self.apply_slippage(price_x, 'buy', spec_x['tick_size'])
        
        # 计算手续费
        nominal_x = open_price_x * lots_result['lots_x'] * spec_x['multiplier']
        nominal_y = open_price_y * lots_result['lots_y'] * spec_y['multiplier']
        open_commission = (nominal_x + nominal_y) * self.config.commission_rate
        
        # 创建持仓
        position = Position(
            pair=pair,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            direction=signal['signal'],
            open_date=pd.to_datetime(signal['date']),
            beta=beta,
            lots_x=lots_result['lots_x'],
            lots_y=lots_result['lots_y'],
            theoretical_ratio=lots_result['theoretical_ratio'],
            actual_ratio=lots_result['actual_ratio'],
            open_price_x=open_price_x,
            open_price_y=open_price_y,
            margin_occupied=margin_required,
            open_commission=open_commission,
            multiplier_x=spec_x['multiplier'],
            multiplier_y=spec_y['multiplier'],
            tick_size_x=spec_x['tick_size'],
            tick_size_y=spec_y['tick_size'],
            z_score_open=signal.get('z_score', 0)
        )
        
        # 更新资金
        self.available_capital -= (margin_required + open_commission)
        self.occupied_margin += margin_required
        
        # 添加持仓
        self.positions[pair] = position
        
        logger.info(f"开仓: {pair} {signal['signal']}")
        logger.info(f"  β={beta:.4f}, 手数Y:{lots_result['lots_y']} X:{lots_result['lots_x']}")
        logger.info(f"  保证金: {margin_required:,.0f}, 手续费: {open_commission:,.0f}")
        
        return True
    
    def _close_position(self, pair: str, current_prices: Dict[str, float], 
                       reason: str, current_date: str) -> bool:
        """
        平仓操作
        
        Args:
            pair: 配对名称
            current_prices: 当前价格
            reason: 平仓原因
            current_date: 当前日期
        """
        if pair not in self.positions:
            logger.warning(f"未找到持仓: {pair}")
            return False
        
        position = self.positions[pair]
        
        # 获取价格
        price_x = current_prices.get(position.symbol_x)
        price_y = current_prices.get(position.symbol_y)
        
        if price_x is None or price_y is None:
            logger.warning(f"缺少平仓价格: {position.symbol_x}={price_x}, {position.symbol_y}={price_y}")
            return False
        
        # 应用滑点
        if position.direction == 'open_long':
            # 平多头价差：卖Y买X
            close_price_y = self.apply_slippage(price_y, 'sell', position.tick_size_y)
            close_price_x = self.apply_slippage(price_x, 'buy', position.tick_size_x)
        else:  # open_short
            # 平空头价差：买Y卖X
            close_price_y = self.apply_slippage(price_y, 'buy', position.tick_size_y)
            close_price_x = self.apply_slippage(price_x, 'sell', position.tick_size_x)
        
        # 计算PnL
        if position.direction == 'open_long':
            y_pnl = (close_price_y - position.open_price_y) * position.lots_y * position.multiplier_y
            x_pnl = (position.open_price_x - close_price_x) * position.lots_x * position.multiplier_x
        else:  # open_short
            y_pnl = (position.open_price_y - close_price_y) * position.lots_y * position.multiplier_y
            x_pnl = (close_price_x - position.open_price_x) * position.lots_x * position.multiplier_x
        
        gross_pnl = y_pnl + x_pnl
        
        # 计算平仓手续费
        nominal_x = close_price_x * position.lots_x * position.multiplier_x
        nominal_y = close_price_y * position.lots_y * position.multiplier_y
        close_commission = (nominal_x + nominal_y) * self.config.commission_rate
        
        # 净PnL
        net_pnl = gross_pnl - position.open_commission - close_commission
        
        # 更新资金
        self.available_capital += position.margin_occupied + net_pnl - close_commission
        self.occupied_margin -= position.margin_occupied
        
        # 计算持仓天数
        close_date = pd.to_datetime(current_date)
        holding_days = (close_date - position.open_date).days
        
        # 记录交易
        trade_record = {
            'trade_id': self.trade_id,
            'pair': pair,
            'symbol_x': position.symbol_x,
            'symbol_y': position.symbol_y,
            'direction': position.direction,
            'beta': position.beta,
            'z_score_open': position.z_score_open,
            'open_date': position.open_date.strftime('%Y-%m-%d'),
            'close_date': close_date.strftime('%Y-%m-%d'),
            'holding_days': holding_days,
            'lots_x': position.lots_x,
            'lots_y': position.lots_y,
            'theoretical_ratio': position.theoretical_ratio,
            'actual_ratio': position.actual_ratio,
            'open_price_x': position.open_price_x,
            'open_price_y': position.open_price_y,
            'close_price_x': close_price_x,
            'close_price_y': close_price_y,
            'margin_occupied': position.margin_occupied,
            'gross_pnl': gross_pnl,
            'y_pnl': y_pnl,
            'x_pnl': x_pnl,
            'open_commission': position.open_commission,
            'close_commission': close_commission,
            'net_pnl': net_pnl,
            'return_on_margin': net_pnl / position.margin_occupied if position.margin_occupied > 0 else 0,
            'close_reason': reason
        }
        
        self.trade_records.append(trade_record)
        self.trade_id += 1
        
        # 移除持仓
        del self.positions[pair]
        
        logger.info(f"平仓: {pair}, 原因={reason}")
        logger.info(f"  持仓{holding_days}天, 净PnL={net_pnl:,.0f}")
        logger.info(f"  收益率: {net_pnl/position.margin_occupied:.2%}")
        
        return True
    
    def check_risk_control(self, current_date: str, current_prices: Dict[str, float]) -> List[str]:
        """
        检查风险控制（止损和时间止损）
        
        Returns:
            需要平仓的配对列表
        """
        pairs_to_close = []
        current_dt = pd.to_datetime(current_date)
        
        for pair, position in self.positions.items():
            # 1. 检查时间止损
            if self.config.enable_time_stop:
                holding_days = (current_dt - position.open_date).days
                if holding_days >= self.config.max_holding_days:
                    logger.info(f"{pair} 持仓{holding_days}天，触发时间止损")
                    pairs_to_close.append((pair, 'time_stop'))
                    continue
            
            # 2. 检查止损
            if self.config.enable_stop_loss:
                # 获取当前价格
                price_x = current_prices.get(position.symbol_x)
                price_y = current_prices.get(position.symbol_y)
                
                if price_x is not None and price_y is not None:
                    # 计算当前PnL
                    if position.direction == 'open_long':
                        y_pnl = (price_y - position.open_price_y) * position.lots_y * position.multiplier_y
                        x_pnl = (position.open_price_x - price_x) * position.lots_x * position.multiplier_x
                    else:
                        y_pnl = (position.open_price_y - price_y) * position.lots_y * position.multiplier_y
                        x_pnl = (price_x - position.open_price_x) * position.lots_x * position.multiplier_x
                    
                    current_pnl = y_pnl + x_pnl - position.open_commission
                    
                    # 检查是否触发止损
                    if current_pnl < 0:
                        loss_pct = abs(current_pnl) / position.margin_occupied
                        if loss_pct >= self.config.stop_loss_pct:
                            logger.info(f"{pair} 亏损{loss_pct:.1%}，触发止损")
                            pairs_to_close.append((pair, 'stop_loss'))
        
        return pairs_to_close
    
    def run_backtest(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            signals: 信号数据（来自信号生成模块）
            prices: 价格数据
            
        Returns:
            回测结果
        """
        logger.info("开始回测...")
        logger.info(f"  信号数量: {len(signals)}")
        logger.info(f"  日期范围: {signals['date'].min()} 至 {signals['date'].max()}")
        
        # 按日期分组处理
        for date in sorted(signals['date'].unique()):
            # 获取当天价格
            if date in prices.index:
                current_prices = prices.loc[date].to_dict()
            else:
                continue
            
            # 1. 风险控制检查
            pairs_to_close = self.check_risk_control(date, current_prices)
            for pair, reason in pairs_to_close:
                self._close_position(pair, current_prices, reason, date)
            
            # 2. 处理当天信号
            day_signals = signals[signals['date'] == date]
            for _, signal in day_signals.iterrows():
                self.process_signal(signal.to_dict(), current_prices)
            
            # 3. 记录每日权益
            self.total_equity = self.available_capital + self.occupied_margin
            self.equity_curve.append({
                'date': date,
                'equity': self.total_equity,
                'available': self.available_capital,
                'occupied': self.occupied_margin,
                'positions': len(self.positions)
            })
        
        # 强制平仓所有未平仓持仓
        if self.config.force_close_at_end:
            logger.info("回测结束，强制平仓所有持仓")
            last_date = signals['date'].max()
            if last_date in prices.index:
                last_prices = prices.loc[last_date].to_dict()
                for pair in list(self.positions.keys()):
                    self._close_position(pair, last_prices, 'forced', last_date)
        
        # 计算绩效指标
        results = self.calculate_metrics()
        
        # 保存结果
        if self.config.save_trades:
            self.save_trades()
        
        logger.info("回测完成")
        logger.info(f"  总交易数: {len(self.trade_records)}")
        logger.info(f"  总收益: {results['total_pnl']:,.0f}")
        logger.info(f"  收益率: {results['total_return']:.2%}")
        logger.info(f"  夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.trade_records:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # 基础统计
        trades_df = pd.DataFrame(self.trade_records)
        total_pnl = trades_df['net_pnl'].sum()
        total_return = total_pnl / self.config.initial_capital
        
        # 胜率
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df)
        
        # 夏普比率（如果有日收益数据）
        sharpe_ratio = 0
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['return'] = equity_df['equity'].pct_change()
            if equity_df['return'].std() > 0:
                sharpe_ratio = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252)
        
        # 最大回撤
        max_drawdown = 0
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax
            max_drawdown = abs(drawdown.min())
        
        return {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (252 / len(self.equity_curve)) - 1 if self.equity_curve else 0,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_pnl': trades_df['net_pnl'].mean(),
            'avg_holding_days': trades_df['holding_days'].mean(),
            'stop_loss_count': len(trades_df[trades_df['close_reason'] == 'stop_loss']),
            'time_stop_count': len(trades_df[trades_df['close_reason'] == 'time_stop'])
        }
    
    def save_trades(self) -> None:
        """保存交易记录"""
        if not self.trade_records:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存交易记录
        trades_df = pd.DataFrame(self.trade_records)
        trades_file = output_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"交易记录已保存: {trades_file}")
        
        # 保存权益曲线
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_dir / f"equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"权益曲线已保存: {equity_file}")


def create_backtest_engine(custom_config: Dict = None) -> BacktestEngine:
    """
    创建回测引擎的便捷函数
    
    Args:
        custom_config: 自定义配置字典
        
    Example:
        engine = create_backtest_engine({
            'initial_capital': 10000000,
            'stop_loss_pct': 0.10,
            'max_holding_days': 20
        })
    """
    config = BacktestConfig()
    
    # 应用自定义配置
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return BacktestEngine(config)


if __name__ == "__main__":
    # 示例：创建一个自定义配置的回测引擎
    engine = create_backtest_engine({
        'initial_capital': 10000000,    # 1000万初始资金
        'margin_rate': 0.15,            # 15%保证金率
        'stop_loss_pct': 0.10,          # 10%止损
        'max_holding_days': 20,         # 20天强制平仓
        'commission_rate': 0.0001,      # 万分之1手续费
        'slippage_ticks': 2             # 2个tick滑点
    })
    
    print("回测引擎创建成功")
    print(f"配置参数:")
    print(f"  初始资金: {engine.config.initial_capital:,.0f}")
    print(f"  保证金率: {engine.config.margin_rate:.1%}")
    print(f"  止损线: {engine.config.stop_loss_pct:.1%}")
    print(f"  最大持仓: {engine.config.max_holding_days}天")