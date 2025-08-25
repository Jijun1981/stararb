# 模块4: 回测框架需求文档 V2.0

## 1. 模块概述
负责期货配对交易的完整回测，包含手数计算、交易执行、风险管理和绩效分析四个子模块。**特别强调正确处理负Beta配对**，确保交易方向、PnL计算的正确性，模拟真实交易环境并生成详细的分析报告。

**重要说明**: Beta可以为正或负，决定了配对交易的对冲方向：
- **正Beta (β > 0)**: X和Y同向变动，做多配对时买Y卖X，做空配对时卖Y买X
- **负Beta (β < 0)**: X和Y反向变动，做多配对时买Y买X，做空配对时卖Y卖X

## 2. 用户故事 (User Stories)

### Story 4.1: 精确手数计算
**作为**量化交易员  
**我希望**系统能根据动态β值和资金约束计算最优手数  
**以便**在保持对冲比例的同时充分利用分配资金

**验收标准:**
- 使用连分数算法找到最接近β的整数比
- 每个配对独立分配资金（默认5%）
- 在分配资金内等比例缩放手数
- 缩放后仍保持原有手数比例
- 资金不足时返回明确的不可交易标识

### Story 4.2: 真实交易执行
**作为**量化交易员  
**我希望**系统能模拟真实的期货交易执行  
**以便**准确评估策略的实际可执行性

**验收标准:**
- 计算双边手续费（默认万分之2）
- 应用滑点成本（默认3个tick）
- 准确计算保证金占用（12%）
- 记录开仓时间、价格、手数
- 支持多种平仓原因（信号/止损/时间）

### Story 4.3: 严格风险控制
**作为**风险管理员  
**我希望**系统能实时监控并控制交易风险  
**以便**避免超出风险承受能力

**验收标准:**
- 实时监控浮动盈亏
- 触发15%止损自动平仓
- 30天强制平仓规则
- 监控保证金充足率
- 限制最大同时持仓数量

### Story 4.4: 全面绩效分析
**作为**基金经理  
**我希望**获得组合和配对级别的详细绩效报告  
**以便**评估策略表现和各配对贡献

**验收标准:**
- 计算组合级别所有指标（Sharpe、Sortino、最大回撤等）
- 计算每个配对的独立绩效指标
- 生成配对贡献度分析
- 计算配对间相关性矩阵
- 输出完整的交易明细

## 3. 功能需求 (Requirements)

### REQ-4.1: 手数计算模块 (position_sizing.py)

#### REQ-4.1.1: 资金分配机制
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.1.1.1 | 每个配对独立分配资金：allocated = total_capital × position_weight | P0 |
| REQ-4.1.1.2 | 默认position_weight = 0.05（5%） | P0 |
| REQ-4.1.1.3 | 配对间资金完全隔离，不可挪用 | P0 |
| REQ-4.1.1.4 | 支持自定义每个配对的权重 | P1 |

#### REQ-4.1.2: 最小整数比计算（支持负Beta）
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.1.2.1 | 计算有效对冲比：h* = abs(β) × (Py × My) / (Px × Mx) | P0 |
| REQ-4.1.2.2 | 使用Fraction类对h*进行连分数逼近：Fraction(h*).limit_denominator(max_denominator) | P0 |
| REQ-4.1.2.3 | max_denominator默认值10，可配置 | P0 |
| REQ-4.1.2.4 | 得到最小整数对(nx, ny)，确保每腿至少min_lots手（默认1） | P0 |
| REQ-4.1.2.5 | 计算名义价值匹配误差：error = abs(nx×Px×Mx - ny×Py×My×abs(β)) / (ny×Py×My×abs(β)) | P0 |
| REQ-4.1.2.6 | 返回lots_x、lots_y、effective_ratio、nominal_error_pct、beta_sign | P0 |
| REQ-4.1.2.7 | **重要**: 手数计算时使用abs(β)，但保留β的符号信息用于交易方向判断 | P0 |

#### REQ-4.1.3: 资金约束应用
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.1.3.1 | 计算最小整数对(nx, ny)所需保证金：margin = nx×Px×Mx×margin_rate + ny×Py×My×margin_rate | P0 |
| REQ-4.1.3.2 | 计算整数倍缩放系数：k = floor(allocated_capital × 0.95 / margin) | P0 |
| REQ-4.1.3.3 | 最终手数：final_nx = nx × k, final_ny = ny × k | P0 |
| REQ-4.1.3.4 | 如果k=0（资金不足），返回can_trade=False | P0 |
| REQ-4.1.3.5 | 检查最大手数限制：确保final_nx ≤ max_lots_per_leg 且 final_ny ≤ max_lots_per_leg | P0 |
| REQ-4.1.3.6 | 计算资金利用率：utilization = actual_margin / allocated_capital | P0 |

### REQ-4.2: 交易执行模块 (trade_executor.py)

#### REQ-4.2.1: 开仓执行（考虑Beta符号）
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.2.1.1 | 记录开仓时间、配对名称、方向（long/short） | P0 |
| REQ-4.2.1.2 | 记录X和Y的开仓价格、手数、Beta值 | P0 |
| REQ-4.2.1.3 | **核心逻辑**: 根据Beta符号确定交易方向 | P0 |
| REQ-4.2.1.4 | 正Beta (β>0) + Long信号: 买Y卖X | P0 |
| REQ-4.2.1.5 | 正Beta (β>0) + Short信号: 卖Y买X | P0 |
| REQ-4.2.1.6 | 负Beta (β<0) + Long信号: 买Y买X | P0 |
| REQ-4.2.1.7 | 负Beta (β<0) + Short信号: 卖Y卖X | P0 |
| REQ-4.2.1.8 | 计算开仓滑点：buy加tick，sell减tick | P0 |
| REQ-4.2.1.9 | 计算开仓手续费：commission = value × rate | P0 |
| REQ-4.2.1.10 | 计算保证金占用：margin = value × margin_rate | P0 |
| REQ-4.2.1.11 | 生成唯一position_id | P0 |

#### REQ-4.2.2: 平仓执行（考虑Beta符号的PnL计算）
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.2.2.1 | 记录平仓时间、价格、原因 | P0 |
| REQ-4.2.2.2 | 计算平仓滑点（方向与开仓相反） | P0 |
| REQ-4.2.2.3 | 计算平仓手续费 | P0 |
| REQ-4.2.2.4 | **核心**: 正确计算PnL，考虑Beta符号 | P0 |
| REQ-4.2.2.5 | 正Beta情况：pnl_x = -lots_x × (close_x - open_x) × multiplier_x | P0 |
| REQ-4.2.2.6 | 正Beta情况：pnl_y = lots_y × (close_y - open_y) × multiplier_y × direction | P0 |
| REQ-4.2.2.7 | 负Beta情况：pnl_x = lots_x × (close_x - open_x) × multiplier_x × direction | P0 |
| REQ-4.2.2.8 | 负Beta情况：pnl_y = lots_y × (close_y - open_y) × multiplier_y × direction | P0 |
| REQ-4.2.2.9 | 总PnL = pnl_x + pnl_y - 总成本 | P0 |
| REQ-4.2.2.10 | 释放保证金 | P0 |
| REQ-4.2.2.11 | 更新可用资金 | P0 |

#### REQ-4.2.3: 成本计算
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.2.3.1 | 手续费率默认0.0002（万分之2） | P0 |
| REQ-4.2.3.2 | 滑点默认3个tick | P0 |
| REQ-4.2.3.3 | 保证金率默认12% | P0 |
| REQ-4.2.3.4 | 所有参数可配置 | P0 |

### REQ-4.3: 风险管理模块 (risk_manager.py)

#### REQ-4.3.1: 止损管理
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.3.1.1 | 计算浮动盈亏：unrealized_pnl = current_value - open_value - costs | P0 |
| REQ-4.3.1.2 | 盈亏百分比基于分配资金：pnl_pct = unrealized_pnl / allocated_capital | P0 |
| REQ-4.3.1.3 | 触发条件：pnl_pct <= -stop_loss_pct（默认-10%） | P0 |
| REQ-4.3.1.4 | 触发后立即平仓，记录止损原因 | P0 |
| REQ-4.3.1.5 | 统计止损次数和损失金额 | P0 |

#### REQ-4.3.2: 时间止损
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.3.2.1 | 计算持仓天数：days = current_date - open_date | P0 |
| REQ-4.3.2.2 | 触发条件：days >= max_holding_days（默认30） | P0 |
| REQ-4.3.2.3 | 触发后强制平仓，记录时间止损原因 | P0 |
| REQ-4.3.2.4 | 统计时间止损次数 | P0 |

#### REQ-4.3.3: 保证金监控
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.3.3.1 | 实时计算已用保证金：used = Σ(position_margin) | P0 |
| REQ-4.3.3.2 | 计算可用保证金：available = capital - used | P0 |
| REQ-4.3.3.3 | 新开仓前检查：required <= available × buffer | P0 |
| REQ-4.3.3.4 | buffer默认0.8（留20%缓冲） | P1 |
| REQ-4.3.3.5 | 保证金不足时拒绝开仓 | P0 |

### REQ-4.4: 绩效分析模块 (performance.py)

#### REQ-4.4.1: 组合级别指标
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.4.1.1 | 总收益率：total_return = (final - initial) / initial | P0 |
| REQ-4.4.1.2 | 年化收益：annual_return = (1 + total_return)^(252/days) - 1 | P0 |
| REQ-4.4.1.3 | 夏普比率：sharpe = mean(returns) / std(returns) × sqrt(252) | P0 |
| REQ-4.4.1.4 | Sortino比率：sortino = mean(returns) / downside_std × sqrt(252) | P0 |
| REQ-4.4.1.5 | 最大回撤：max_dd = max(peak - trough) / peak | P0 |
| REQ-4.4.1.6 | 胜率：win_rate = winning_trades / total_trades | P0 |
| REQ-4.4.1.7 | 盈亏比：profit_factor = sum(wins) / abs(sum(losses)) | P0 |
| REQ-4.4.1.8 | Calmar比率：calmar = annual_return / max_drawdown | P1 |

#### REQ-4.4.2: 配对级别指标
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.4.2.1 | 每个配对独立计算所有组合级别指标 | P0 |
| REQ-4.4.2.2 | 统计配对交易次数、平均持仓天数 | P0 |
| REQ-4.4.2.3 | 计算配对贡献度：contribution = pair_pnl / total_pnl | P0 |
| REQ-4.4.2.4 | 统计止损次数、止损损失 | P0 |
| REQ-4.4.2.5 | 记录平均手数、平均β值 | P0 |
| REQ-4.4.2.6 | 生成配对权益曲线 | P0 |

#### REQ-4.4.3: 交易明细
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-4.4.3.1 | 记录每笔交易完整信息 | P0 |
| REQ-4.4.3.2 | 包含开仓时间、平仓时间、持仓天数 | P0 |
| REQ-4.4.3.3 | 包含配对名称、方向、手数 | P0 |
| REQ-4.4.3.4 | 包含开仓价、平仓价、滑点、手续费 | P0 |
| REQ-4.4.3.5 | 包含毛利润、净利润、收益率 | P0 |
| REQ-4.4.3.6 | 包含平仓原因（信号/止损/时间） | P0 |

## 4. 接口定义

### 4.1 手数计算模块接口

```python
class PositionSizer:
    def __init__(self, config: PositionSizingConfig):
        """
        Args:
            config: 包含以下参数
                - max_denominator: 最大分母（默认10）
                - min_lots: 最小手数（默认1）
                - max_lots_per_leg: 每腿最大手数（默认100）
                - margin_rate: 保证金率（默认0.12）
        """
        
    def calculate_min_integer_ratio(
        self, 
        beta: float,
        price_x: float,
        price_y: float,
        multiplier_x: float,
        multiplier_y: float
    ) -> Dict[str, Any]:
        """
        计算考虑价格和乘数的最小整数比
        
        Args:
            beta: 动态β值
            price_x: X品种当前价格
            price_y: Y品种当前价格
            multiplier_x: X品种合约乘数
            multiplier_y: Y品种合约乘数
            
        Returns:
            {
                'lots_x': int,          # 最小手数X
                'lots_y': int,          # 最小手数Y
                'effective_ratio': float,  # 有效对冲比h*
                'nominal_error_pct': float  # 名义价值误差
            }
        """
        
    def calculate_position_size(
        self,
        min_lots: Dict[str, int],  # 最小整数对(nx, ny)
        prices: Dict[str, float],
        multipliers: Dict[str, float],
        total_capital: float,
        position_weight: float = 0.05
    ) -> Dict[str, Any]:
        """
        应用资金约束，整数倍缩放手数
        
        Returns:
            {
                'final_lots_x': int,        # nx × k
                'final_lots_y': int,        # ny × k  
                'scaling_factor': int,      # k值
                'allocated_capital': float,
                'margin_required': float,
                'position_value': float,
                'utilization_rate': float,
                'can_trade': bool,
                'reason': str  # 如果can_trade=False
            }
        """
```

### 4.2 交易执行模块接口

```python
class TradeExecutor:
    def execute_open(
        self,
        pair_info: Dict,
        lots: Dict[str, int],
        prices: Dict[str, float],
        signal_type: str
    ) -> Position:
        """执行开仓"""
        
    def execute_close(
        self,
        position: Position,
        prices: Dict[str, float],
        reason: str
    ) -> Trade:
        """执行平仓"""
```

### 4.3 风险管理模块接口

```python
class RiskManager:
    def check_stop_loss(
        self,
        position: Position,
        current_pnl: float,
        allocated_capital: float
    ) -> Tuple[bool, str]:
        """检查止损（基于分配资金）"""
        
    def check_time_stop(
        self,
        position: Position,
        current_date: datetime
    ) -> Tuple[bool, str]:
        """检查时间止损"""
```

### 4.4 绩效分析模块接口

```python
class PerformanceAnalyzer:
    def calculate_portfolio_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """计算组合级别指标"""
        
    def calculate_pair_metrics(
        self,
        pair: str,
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """计算配对级别指标"""
```

## 5. 数据结构

### 5.1 Position（持仓）
```python
@dataclass
class Position:
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
```

### 5.2 Trade（交易记录）
```python
@dataclass
class Trade:
    trade_id: str
    position_id: str
    pair: str
    # ... 开仓信息
    close_date: datetime
    close_price_x: float
    close_price_y: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    holding_days: int
    close_reason: str  # 'signal', 'stop_loss', 'time_stop'
```

## 6. 性能要求

| 指标 | 要求 |
|---|---|
| 2年回测完成时间 | < 60秒 |
| 内存占用 | < 2GB |
| 手数计算精度 | 100%整数 |
| PnL计算精度 | 精确到分 |

## 7. 测试用例

### Test Case 4.1: 手数计算测试（包含负Beta）
```python
def test_position_sizing():
    sizer = PositionSizer()
    
    # 测试1：正Beta情况
    result = sizer.calculate_min_integer_ratio(
        beta=0.85,
        price_x=60000,
        price_y=140000,
        multiplier_x=5,
        multiplier_y=1
    )
    # h* = 0.85 × (140000×1)/(60000×5) = 0.397
    # Fraction(0.397).limit_denominator(10) ≈ 2/5
    assert result['lots_x'] == 2
    assert result['lots_y'] == 5
    assert result['beta_sign'] == 1
    assert abs(result['nominal_error_pct']) < 5
    
    # 测试2：负Beta情况
    result = sizer.calculate_min_integer_ratio(
        beta=-0.5,  # 负Beta
        price_x=60000,
        price_y=140000,
        multiplier_x=5,
        multiplier_y=1
    )
    # h* = abs(-0.5) × (140000×1)/(60000×5) = 0.233
    # Fraction(0.233).limit_denominator(10) ≈ 1/4
    assert result['lots_x'] == 1
    assert result['lots_y'] == 4
    assert result['beta_sign'] == -1  # 保留负号信息
    
    # 测试2：整数倍缩放
    position = sizer.calculate_position_size(
        min_lots={'lots_x': 2, 'lots_y': 5},
        prices={'x': 60000, 'y': 140000},
        multipliers={'x': 5, 'y': 1},
        total_capital=5000000,
        position_weight=0.05  # 250000分配资金
    )
    # 最小对所需保证金 = 2×60000×5×0.12 + 5×140000×1×0.12 = 156000
    # k = floor(250000×0.95/156000) = 1
    assert position['scaling_factor'] == 1
    assert position['final_lots_x'] == 2
    assert position['final_lots_y'] == 5
    assert position['can_trade'] == True
```

### Test Case 4.2: 交易执行测试（负Beta）
```python
def test_trade_execution_negative_beta():
    executor = TradeExecutor()
    
    # 测试1：负Beta + Long信号 = 买Y买X
    position = executor.execute_open(
        pair_info={'pair': 'AG-NI', 'beta': -0.5},
        lots={'x': 2, 'y': 5},
        prices={'x': 60000, 'y': 140000},
        signal_type='long'
    )
    # 验证：两边都是买入
    assert position.action_x == 'buy'  # 买X
    assert position.action_y == 'buy'  # 买Y
    
    # 测试2：负Beta + Short信号 = 卖Y卖X
    position = executor.execute_open(
        pair_info={'pair': 'AG-NI', 'beta': -0.5},
        lots={'x': 2, 'y': 5},
        prices={'x': 60000, 'y': 140000},
        signal_type='short'
    )
    # 验证：两边都是卖出
    assert position.action_x == 'sell'  # 卖X
    assert position.action_y == 'sell'  # 卖Y
    
    # 测试3：PnL计算（负Beta情况）
    # 开仓：买Y买X，价格上涨后平仓
    trade = executor.execute_close(
        position=position,
        prices={'x': 61000, 'y': 142000},  # 都上涨
        reason='signal'
    )
    # 负Beta + Long：X和Y都买入
    # X盈利 = 2 × (61000-60000) × 5 = 10000
    # Y盈利 = 5 × (142000-140000) × 1 = 10000
    # 总毛利 = 20000
    assert trade.gross_pnl == 20000
```

### Test Case 4.3: 止损测试
```python
def test_stop_loss():
    manager = RiskManager(stop_loss_pct=0.10)  # 10%止损
    position = Position(pair='CU-SN')
    allocated_capital = 250000  # 分配的资金
    
    # 测试：触发止损（亏损超过10%）
    current_pnl = -26000  # 亏损26000
    should_stop, reason = manager.check_stop_loss(
        position, current_pnl, allocated_capital
    )
    assert should_stop == True  # -26000/250000 = -10.4% < -10%
    assert 'stop_loss' in reason
    
    # 测试：不触发（亏损未超过10%）
    current_pnl = -20000  # 亏损20000
    should_stop, _ = manager.check_stop_loss(
        position, current_pnl, allocated_capital
    )
    assert should_stop == False  # -20000/250000 = -8% > -10%
```

### Test Case 4.3: 绩效计算测试
```python
def test_performance_metrics():
    analyzer = PerformanceAnalyzer()
    trades = load_test_trades()
    
    # 组合指标
    portfolio_metrics = analyzer.calculate_portfolio_metrics(trades, equity_curve)
    assert 'sharpe_ratio' in portfolio_metrics
    assert 'sortino_ratio' in portfolio_metrics
    assert portfolio_metrics['total_trades'] == len(trades)
    
    # 配对指标
    pair_metrics = analyzer.calculate_pair_metrics('CU-SN', trades)
    assert pair_metrics['pair'] == 'CU-SN'
    assert 'sharpe_ratio' in pair_metrics
```

## 8. 验收标准总结

### 功能验收
- [ ] 手数计算保持精确的β比例
- [ ] 资金约束正确应用
- [ ] 交易成本准确计算
- [ ] 止损规则正确触发
- [ ] 绩效指标计算准确

### 性能验收
- [ ] 2年回测在60秒内完成
- [ ] 内存占用不超过2GB
- [ ] 支持至少100个配对并行

### 质量验收
- [ ] 单元测试覆盖率>90%
- [ ] 所有接口有完整文档
- [ ] 代码通过lint检查
- [ ] 有完整的错误处理

## 9. 依赖关系

- 依赖信号生成模块的输出格式
- 依赖数据管理模块的价格数据
- 需要合约规格配置文件

## 10. 风险和限制

- 整数手数可能导致对冲比例偏差
- 资金分配可能导致部分配对无法交易
- 滑点模型可能与实际不符