# 模块4: 回测框架需求文档

## 1. 模块概述
接收信号生成模块的交易信号（包含动态β值），根据β值计算最小整数比手数，执行交易，计算PnL，生成绩效报告。

**简化版核心原则**：
- **无资金池限制**：不考虑总资金，有信号就交易
- **单一持仓**：每个配对同时只能有一个持仓
- **最小整数比**：根据β值计算Y:X的最小整数比
- **15%止损**：基于保证金的止损线

## 2. 用户故事 (User Stories)

### Story 4.1: 交易执行模拟
**作为**研究员  
**我希望**根据信号模拟真实期货交易  
**以便**评估策略在实际市场中的表现

**验收标准:**
- 按信号执行开仓和平仓
- 计算保证金占用(12%)
- 扣除交易成本(万分之2+3tick滑点)
- 管理资金和持仓状态

### Story 4.2: 风险管理
**作为**研究员  
**我希望**实施止损和风险控制  
**以便**控制下行风险

**验收标准:**
- 逐日检查浮动盈亏
- 15%保证金止损线
- 30天强制平仓
- 记录止损交易

### Story 4.3: PnL计算
**作为**研究员  
**我希望**准确计算每笔交易的盈亏  
**以便**真实反映策略收益

**验收标准:**
- 计算毛收益和净收益
- 考虑所有交易成本
- 记录逐笔交易明细
- 生成累计收益曲线

### Story 4.4: 绩效分析
**作为**研究员  
**我希望**全面评估策略绩效  
**以便**判断策略的可行性

**验收标准:**
- 计算收益率、夏普比率
- 分析最大回撤
- 统计胜率和盈亏比
- 配对贡献分析

### Story 4.5: 结果可视化
**作为**研究员  
**我希望**直观展示回测结果  
**以便**快速理解策略表现

**验收标准:**
- 累计PnL曲线图
- 配对收益对比图
- 交易分布统计图
- 风险指标仪表板

## 3. 功能需求 (Requirements)

### REQ-4.1: 交易执行与手数计算
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.1.1 | 根据β值计算最小整数比手数（使用Fraction类） | P0 |
| REQ-4.1.2 | 开仓条件：\|z_score\| > z_open_threshold且无持仓（z_open_threshold可配置，默认2.0） | P0 |
| REQ-4.1.3 | 平仓条件：\|z_score\| < z_close_threshold（z_close_threshold可配置，默认0.5） | P0 |
| REQ-4.1.4 | 滑点：每腿各加减tick_size * slippage_ticks（slippage_ticks可配置，默认3） | P0 |
| REQ-4.1.5 | 保证金计算：nominal_value * margin_rate（margin_rate可配置，默认0.12） | P0 |
| REQ-4.1.6 | 交易费用：每次成交收nominal_value * commission_rate（commission_rate可配置，默认0.0002） | P0 |
| REQ-4.1.7 | 持仓记录：包含β值、信号类型和配对信息 | P0 |

### REQ-4.2: 风险控制
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.2.1 | 每个配对同时只能有一个持仓（allow_multiple_positions可配置，默认False） | P0 |
| REQ-4.2.2 | 逐日检查所有持仓的浮动盈亏 | P0 |
| REQ-4.2.3 | 止损条件：单笔亏损达保证金stop_loss_pct触发平仓（stop_loss_pct可配置，默认0.15） | P0 |
| REQ-4.2.4 | 时间止损：持仓超过max_holding_days强制平仓（max_holding_days可配置，默认30） | P0 |
| REQ-4.2.5 | 回测结束时强制平仓所有未平仓持仓（force_close_at_end可配置，默认True） | P0 |

### REQ-4.3: PnL计算
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.3.1 | 毛PnL：严格按direction计算两腿盈亏 | P0 |
| REQ-4.3.2 | 手续费：开仓时各腿nominal*0.0002，平仓时同样 | P0 |
| REQ-4.3.3 | 滑点成本：买入价+3tick，卖出价-3tick | P0 |
| REQ-4.3.4 | 净PnL：毛PnL - 手续费 | P0 |
| REQ-4.3.5 | 收益率：基于保证金计算 (net_pnl/margin*100) | P0 |

### REQ-4.4: 绩效指标
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.4.1 | 总收益率：total_pnl / initial_capital | P0 |
| REQ-4.4.2 | 年化收益率：(1 + total_return)^(252/days) - 1 | P0 |
| REQ-4.4.3 | 夏普比率：mean(returns)/std(returns)*sqrt(252) | P0 |
| REQ-4.4.4 | 最大回撤：max(cummax - cumulative) / cummax | P0 |
| REQ-4.4.5 | 胜率：winning_trades / total_trades | P0 |
| REQ-4.4.6 | 盈亏比：avg(winning) / abs(avg(losing)) | P0 |

### REQ-4.5: 配对分析
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.5.1 | 每个配对的总PnL、平均PnL、标准差 | P0 |
| REQ-4.5.2 | 每个配对的交易次数、胜率、平均持仓天数 | P0 |
| REQ-4.5.3 | 每个配对的止损次数和止损损失 | P0 |
| REQ-4.5.4 | 配对排名：按总PnL排序输出 | P0 |

### REQ-4.6: 结果输出
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.6.1 | 交易明细表：每笔交易的完整记录 | P0 |
| REQ-4.6.2 | 绩效报告：所有关键指标的汇总表 | P0 |
| REQ-4.6.3 | PnL曲线图：累计收益的时间序列图 | P0 |
| REQ-4.6.4 | 配对对比图：Top20配对的收益柱状图 | P0 |
| REQ-4.6.5 | 导出功能：CSV/Excel格式的结果导出 | P1 |

### REQ-4.7: 参数配置
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.7.1 | 资金管理参数可配置：initial_capital（初始资金）、position_weight（仓位权重） | P0 |
| REQ-4.7.2 | 交易成本参数可配置：commission_rate、slippage_ticks | P0 |
| REQ-4.7.3 | 风险控制参数可配置：stop_loss_pct、max_holding_days、enable_stop_loss、enable_time_stop | P0 |
| REQ-4.7.4 | 信号参数可配置：z_open_threshold、z_close_threshold | P0 |
| REQ-4.7.5 | 手数计算参数可配置：max_denominator（最大分母）、min_lots（最小手数）、max_lots_per_leg（每腿最大手数） | P0 |
| REQ-4.7.6 | 提供默认配置类BacktestConfig，支持参数覆盖 | P0 |
| REQ-4.7.7 | 提供create_backtest_engine便捷函数，支持字典形式的配置传入 | P0 |

## 4. 接口定义

### 4.0 输入信号格式
```python
# 来自信号生成模块的信号DataFrame格式
signals_df = pd.DataFrame({
    'date': datetime,           # 交易日期
    'pair': str,                # 配对名称，如'AU0-ZN0'
    'signal': str,              # 信号类型: open_long, open_short, close, hold, converging
    'z_score': float,           # Z-score值
    'beta': float,              # 动态β值（来自Kalman滤波）
    'residual': float,          # 价差残差
    'days_held': int,           # 持仓天数
    'phase': str,               # 阶段: convergence_period, signal_period
    'converged': bool,          # 是否已收敛
})

# 配对信息DataFrame格式
pair_info_df = pd.DataFrame({
    'pair': str,                # 配对名称
    'symbol_x': str,            # X品种代码
    'symbol_y': str,            # Y品种代码
    'direction': str,           # 协整方向: y_on_x
    'beta_1y': float,           # 1年期初始β（参考值）
})
```

### 4.1 BacktestEngine类接口
```python
class BacktestEngine:
    def __init__(self, initial_capital: float = 5000000, 
                 margin_rate: float = 0.12,
                 commission_rate: float = 0.0002,  # 单边费率
                 slippage_ticks: int = 3,
                 position_weights: Dict[str, float] = None,
                 pair_configs: Dict[str, Dict] = None)  # 配对信息
    
    # 回测执行
    def run_backtest(self, signals: pd.DataFrame, 
                     prices: pd.DataFrame, 
                     contract_specs: Dict,
                     pair_info: pd.DataFrame) -> Dict  # 包含symbol_x, symbol_y映射
    
    # 手数计算与交易执行
    def calculate_lots(self, signal: Dict, position_weight: float,
                      available_capital: float) -> Dict
    def execute_signal(self, signal: Dict, 
                       current_prices: Dict) -> Optional[Dict]
    def open_position(self, signal: Dict, lots: Dict,
                     prices: Dict) -> bool
    def close_position(self, pair: str, prices: Dict, 
                      reason: str) -> Dict
    
    # 风险管理
    def check_stop_loss(self, position: Dict, 
                       current_pnl: float) -> bool
    def check_time_stop(self, position: Dict, 
                       current_date: datetime) -> bool
    def update_available_capital(self, amount: float) -> None
    
    # 绩效计算
    def calculate_metrics(self) -> Dict
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int]
    
    # 结果输出
    def generate_report(self) -> pd.DataFrame
    def plot_equity_curve(self) -> None
    def export_results(self, filepath: str) -> None
```

### 4.2 PositionManager类接口
```python
class PositionManager:
    def __init__(self, initial_capital: float, margin_rate: float)
    
    # 属性
    self.positions: Dict[str, Position]
    self.available_capital: float  # 可用资金（已含浮盈亏）
    self.occupied_margin: float   # 占用保证金
    self.daily_pnl: float         # 当日盈亏（已结算到可用资金）
    self.total_equity: float      # 总权益 = 可用 + 保证金
    
    # 方法
    def daily_settlement(self, prices: Dict) -> None  # 逐日盯市
    def update_equity(self) -> float  # 更新权益
    def check_margin_call(self) -> bool  # 检查强平
    def can_open_position(self, required_margin: float) -> bool
    def add_position(self, position: Position) -> None
    def remove_position(self, pair: str) -> None
```

### 4.3 交易记录格式
```python
{
    'trade_id': 1001,
    'pair': 'CU0-SN0',
    'signal_type': 'open_long',  # open_long, open_short, close
    'beta': 0.8523,  # 来自信号的动态β值
    'z_score': -2.41,  # 来自信号的Z-score
    'open_date': '2024-04-10',
    'close_date': '2024-04-15',
    'holding_days': 5,
    'position_weight': 0.05,  # 仓位权重
    'contracts_x': 17,  # 实际手数（基于β计算）
    'contracts_y': 3,   # Y手数（基准）
    'theoretical_ratio': 0.8523,  # Y:X = 1:β
    'open_price_x': 45000.0,  # 含滑点的实际成交价
    'open_price_y': 150000.0,
    'close_price_x': 45200.0,
    'close_price_y': 151000.0,
    'margin_occupied': 162000.0,  # 占用保证金
    'gross_pnl': 2500.0,
    'open_commission': 324.0,  # 开仓手续费
    'close_commission': 328.0,  # 平仓手续费
    'slippage_cost': 150.0,
    'net_pnl': 1698.0,
    'close_reason': 'signal',  # signal, stop_loss, time_stop, margin_call
    'return_on_margin': 1.05  # 基于保证金的收益率
}
```

## 5. 算法细节

### 5.1 基于β值的最小整数比手数计算
```python
def calculate_min_lots(beta: float, max_denominator: int = 10) -> Dict:
    """
    根据β值计算最小整数比手数（无资金限制版本）
    
    Args:
        beta: β值（Y/X的比例）
        max_denominator: 最大分母限制（默认10）
    
    核心原理：
    - 价差公式：spread = log(Y) - β*log(X)
    - 手数比例：Y:X = β:1，需要转换为最简整数比
    - 使用Fraction类找到最简分数表示
    
    示例：
    - β=0.5 -> Y:X = 1:2
    - β=1.5 -> Y:X = 3:2
    - β=0.85 -> Y:X = 6:7
    - β=3.5 -> Y:X = 7:2
    """
    from fractions import Fraction
    
    # 处理特殊情况
    if beta <= 0:
        return {
            'lots_y': 1,
            'lots_x': 1,
            'theoretical_ratio': abs(beta),
            'actual_ratio': 1.0
        }
    
    beta_abs = abs(beta)
    
    # 使用分数类找最简分数
    frac = Fraction(beta_abs).limit_denominator(max_denominator)
    
    # β = lots_y / lots_x
    lots_y = frac.numerator
    lots_x = frac.denominator
    
    # 确保至少1手
    if lots_y == 0:
        lots_y = 1
    if lots_x == 0:
        lots_x = 1
    
    return {
        'lots_y': lots_y,
        'lots_x': lots_x,
        'theoretical_ratio': beta_abs,
        'actual_ratio': lots_y / lots_x
    }
```

### 5.2 精确PnL计算
```
方向定义（基于信号类型）:
- spread = log(Y) - β*log(X)
- open_long: Z < -2.0，价差偏低，做多价差（买Y卖X）
- open_short: Z > 2.0，价差偏高，做空价差（卖Y买X）

开多价差(signal='open_long'):
- Y腿PnL = (Y_close - Y_open) * n_Y * mult_Y  # 做多Y
- X腿PnL = (X_open - X_close) * n_X * mult_X  # 做空X
- 毛PnL = Y腿PnL + X腿PnL

开空价差(signal='open_short'):
- Y腿PnL = (Y_open - Y_close) * n_Y * mult_Y  # 做空Y
- X腿PnL = (X_close - X_open) * n_X * mult_X  # 做多X
- 毛PnL = Y腿PnL + X腿PnL

手续费（每次成交）:
- 开仓费 = (Y_open*n_Y*mult_Y + X_open*n_X*mult_X) * 0.0002
- 平仓费 = (Y_close*n_Y*mult_Y + X_close*n_X*mult_X) * 0.0002

净PnL = 毛PnL - 开仓费 - 平仓费
```

### 5.3 滑点计算（每腿独立）
```python
def apply_slippage(price, side, tick_size, slippage_ticks=3):
    """
    每腿独立计算滑点
    
    Args:
        price: 市场价格
        side: 'buy' 或 'sell'
        tick_size: 最小变动价位
        slippage_ticks: 滑点tick数
    """
    if side == 'buy':
        return price + tick_size * slippage_ticks
    else:  # sell
        return price - tick_size * slippage_ticks

# 示例：开多头价差
# Y腿：买入，价格上滑
Y_open_actual = apply_slippage(Y_market, 'buy', tick_Y, 3)
# X腿：卖出，价格下滑
X_open_actual = apply_slippage(X_market, 'sell', tick_X, 3)

# 平仓时反向
Y_close_actual = apply_slippage(Y_market, 'sell', tick_Y, 3)
X_close_actual = apply_slippage(X_market, 'buy', tick_X, 3)
```

### 5.4 逐日止损检查
```python
def check_daily_stop_loss(position, current_prices, multipliers, tick_sizes):
    """
    逐日检查止损（15%保证金）
    
    每日检查所有持仓的浮动盈亏，
    如果亏损达到保证金的15%，触发止损
    """
    # 获取当前价格
    current_price_y = current_prices[position.symbol_y]
    current_price_x = current_prices[position.symbol_x]
    
    # 计算当前PnL（含滑点）
    if position.direction == 'long':
        exit_price_y = apply_slippage(current_price_y, 'sell', tick_sizes[symbol_y], 3)
        exit_price_x = apply_slippage(current_price_x, 'buy', tick_sizes[symbol_x], 3)
    else:
        exit_price_y = apply_slippage(current_price_y, 'buy', tick_sizes[symbol_y], 3)
        exit_price_x = apply_slippage(current_price_x, 'sell', tick_sizes[symbol_x], 3)
    
    # 计算PnL
    pnl_result = calculate_pnl(
        position=position,
        exit_price_y=exit_price_y,
        exit_price_x=exit_price_x,
        mult_y=multipliers[symbol_y],
        mult_x=multipliers[symbol_x]
    )
    
    # 检查是否超过15%止损线
    if pnl_result['net_pnl'] < 0:
        loss_pct = abs(pnl_result['net_pnl']) / position['margin']
        if loss_pct >= 0.15:
            return True, pnl_result['net_pnl']
    
    return False, pnl_result['net_pnl']
```

### 5.5 简化版回测主流程
```python
def run_simple_backtest(signals, prices):
    """
    简化版回测主流程
    
    核心逻辑：
    1. 按日期顺序处理
    2. 先检查所有持仓的止损
    3. 再处理当天的信号
    4. 记录所有交易细节
    """
    positions = {}  # 当前持仓
    all_trades = []  # 所有交易记录
    
    # 按日期处理
    for current_date in all_dates:
        # 1. 检查止损
        for pair, pos in positions.items():
            should_stop, current_pnl = check_daily_stop_loss(pos, current_date)
            if should_stop:
                # 执行止损平仓
                close_position(pair, 'stop_loss')
                del positions[pair]
        
        # 2. 处理信号
        day_signals = signals[signals['date'] == current_date]
        for signal in day_signals:
            pair = signal['pair']
            z_score = signal['z_score']
            beta = abs(signal['beta'])
            
            # 检查平仓信号
            if pair in positions and abs(z_score) < 0.5:
                close_position(pair, 'signal')
                del positions[pair]
            
            # 检查开仓信号
            if pair not in positions and abs(z_score) > 2.0:
                direction = 'long' if z_score < -2.0 else 'short'
                
                # 计算最小整数比手数
                lots_result = calculate_min_lots(beta)
                
                # 开仓
                positions[pair] = open_position(
                    pair, direction, lots_result, beta
                )
    
    # 3. 强制平仓未平仓持仓
    for pair in positions:
        close_position(pair, 'forced')
    
    return all_trades
```

### 5.6 手数计算验证算法
```python
def verify_lots_calculation(beta, lots_Y, lots_X):
    """
    验证手数计算的正确性
    
    核心验证点：
    1. 手数必须为正整数
    2. 实际比例应接近理论比例β
    3. 比例偏差应在合理范围内
    
    Args:
        beta: 信号中的动态β值
        lots_Y: 计算出的Y品种手数
        lots_X: 计算出的X品种手数
    
    Returns:
        Dict: 验证结果和偏差分析
    """
    # 1. 基础验证
    assert lots_Y > 0 and isinstance(lots_Y, int), "Y手数必须为正整数"
    assert lots_X > 0 and isinstance(lots_X, int), "X手数必须为正整数"
    
    # 2. 计算实际比例
    actual_ratio = lots_X / lots_Y
    
    # 3. 计算偏差
    ratio_error = abs(actual_ratio - beta) / beta
    
    # 4. 评估偏差是否可接受
    # 由于整数约束，允许一定偏差
    max_acceptable_error = 1.0 / lots_Y  # 偏差上限约为1/Y手数
    
    return {
        'valid': ratio_error <= max_acceptable_error * 2,  # 允许2倍理论偏差
        'theoretical_ratio': beta,
        'actual_ratio': actual_ratio,
        'ratio_error': ratio_error,
        'error_percentage': ratio_error * 100,
        'lots_Y': lots_Y,
        'lots_X': lots_X,
        'message': f"Y:X = {lots_Y}:{lots_X}, 理论比例1:{beta:.4f}, 实际比例1:{actual_ratio:.4f}"
    }

# 示例验证
# β=0.85, Y=10手, X=9手 (round(10*0.85)=9)
# 实际比例=0.9, 偏差=5.88%, 可接受

# β=0.85, Y=3手, X=3手 (round(3*0.85)=3)  
# 实际比例=1.0, 偏差=17.6%, 但由于手数较小，仍可接受

# β=0.85, Y=20手, X=17手 (round(20*0.85)=17)
# 实际比例=0.85, 偏差=0%, 完美匹配
```

## 6. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 性能 | 2年数据回测总耗时 | < 60秒 |
| 性能 | 内存占用 | < 4GB |
| 准确性 | PnL计算精度 | 精确到分 |
| 可靠性 | 异常处理覆盖率 | 100% |
| 可维护性 | 代码测试覆盖率 | > 80% |

## 7. 简化版示例

### 7.1 手数计算示例
```
信号：AU0-ZN0，β=0.85，Z-score=-2.5（开多）

最小整数比计算：
1. β=0.85 ≈ 6/7（使用Fraction类）
2. Y(ZN0)=6手，X(AU0)=7手
3. 理论比例：6:7=0.857，误差：0.7%

保证金计算：
- AU0: 500×1000×7×12% = 420,000
- ZN0: 25000×5×6×12% = 90,000
- 总保证金：510,000

PnL计算（5天后平仓）：
- AU0: 500→520，盈利=20×1000×7=140,000
- ZN0: 25000→24500，亏损=500×5×6=15,000
- 毛利润：140,000-15,000=125,000
- 手续费：开仓+平仓≈4,000
- 净利润：121,000
- 收益率：121,000/510,000=23.7%
```

### 7.2 止损示例
```
持仓：CU0-SN0，保证金=300,000

Day 1: 浮亏-20,000（6.7%），继续持有
Day 2: 浮亏-35,000（11.7%），继续持有
Day 3: 浮亏-48,000（16%），触发15%止损
- 立即平仓
- 实际亏损：-48,000
- 记录为stop_loss
```

## 7. 测试用例

### TC-4.1: 交易执行测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-4.1.1 | 资金充足时开仓 | 成功开仓，扣除保证金和费用 |
| TC-4.1.2 | 资金不足时开仓 | 拒绝开仓，保持原状态 |
| TC-4.1.3 | 正常信号平仓 | 成功平仓，释放保证金 |
| TC-4.1.4 | 滑点计算 | 正确计算3tick滑点成本 |

### TC-4.2: 风险控制测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-4.2.1 | 浮亏达到10% | 触发止损，强制平仓 |
| TC-4.2.2 | 持仓30天 | 触发时间止损 |
| TC-4.2.3 | 多重风险触发 | 优先级：止损>时间>信号 |

### TC-4.3: PnL计算测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-4.3.1 | 多头盈利场景 | 正确计算正收益 |
| TC-4.3.2 | 空头亏损场景 | 正确计算负收益 |
| TC-4.3.3 | 费用扣除 | 准确扣除所有成本 |

### TC-4.4: 绩效指标测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-4.4.1 | 计算夏普比率 | 与标准公式结果一致 |
| TC-4.4.2 | 计算最大回撤 | 正确识别最大跌幅 |
| TC-4.4.3 | 统计胜率 | 准确统计盈利比例 |

## 8. 可视化需求

### 8.1 累计PnL曲线
- X轴：日期
- Y轴：累计净收益（万元）
- 显示：每个配对的曲线 + 总体曲线
- 标注：重要事件点（大额盈亏、止损）

### 8.2 配对收益柱状图
- 显示Top20配对
- 绿色：盈利配对
- 红色：亏损配对
- 标注：具体金额

### 8.3 交易分布图
- 散点图：X轴持仓天数，Y轴收益率
- 颜色：区分不同平仓原因
- 大小：代表交易金额

### 8.4 风险指标仪表板
- 夏普比率趋势
- 滚动最大回撤
- 胜率变化
- 资金使用率

## 9. 依赖关系
- **上游模块**: 信号生成模块（提供交易信号）
- **数据需求**: 实时价格数据、合约规格
- **Python包**: pandas, numpy, matplotlib, plotly
- **输出格式**: CSV, Excel, PNG图表