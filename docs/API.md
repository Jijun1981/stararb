# API文档 - 配对交易系统

## 数据流程
```
1. 数据加载 → 2. 协整分析 → 3. 信号生成 → 4. 回测执行
```

## 标准数据接口定义

### 1. 价格数据 (PriceData)
```python
# DataFrame格式
# index: pd.DatetimeIndex  # 日期索引
# columns: List[str]       # 品种代码 ['CU', 'ZN', 'AL', ...]
# values: float            # 收盘价

price_data = pd.DataFrame(
    index=pd.date_range('2020-01-01', '2024-12-31'),
    columns=['CU', 'ZN', 'AL'],
    data=[[77000, 26000, 18500], ...]  # 每日收盘价
)
```

### 2. 配对信息 (PairInfo)
```python
# DataFrame格式，每行一个配对
pair_info = pd.DataFrame({
    'pair': str,          # 配对名称，格式: 'X-Y' (如 'CU-ZN')
    'symbol_x': str,      # X品种代码
    'symbol_y': str,      # Y品种代码
    'beta_5y': float,     # 5年窗口Beta系数
    'beta_1y': float,     # 1年窗口Beta系数（可选）
    'pvalue_5y': float,   # 5年窗口p值
    'pvalue_1y': float,   # 1年窗口p值
    'halflife': float,    # 半衰期（可选）
    'r_squared': float    # R²值（可选）
})
```

### 3. 交易信号 (TradingSignals)
```python
# DataFrame格式，每行一个时间点
trading_signals = pd.DataFrame({
    'date': pd.Timestamp,      # 日期（可作为index或列）
    'pair': str,                # 配对名称 'CU-ZN'
    'symbol_x': str,            # X品种代码
    'symbol_y': str,            # Y品种代码
    'trade_signal': str|None,   # 交易信号: 'open_long', 'open_short', 'close', None
    'z_score': float,           # Z分数
    'beta': float,              # 动态Beta（Kalman估计）
    'alpha': float,             # 动态Alpha（Kalman估计）
    'position': str|None,       # 当前持仓: 'long', 'short', None
    'days_held': int           # 持仓天数
})
```

### 4. 回测结果 (BacktestResults)
```python
# 字典格式
backtest_results = {
    'trades': List[Trade],      # 交易记录列表
    'equity_curve': List[float], # 权益曲线
    'metrics': Dict,            # 绩效指标
    'pair_metrics': Dict        # 配对级别指标
}

# Trade对象包含
Trade = {
    'pair': str,                # 配对名称
    'direction': str,           # 方向: 'long', 'short'
    'open_date': datetime,      # 开仓日期
    'close_date': datetime,     # 平仓日期
    'net_pnl': float,          # 净盈亏
    'return_pct': float,       # 收益率
    'close_reason': str        # 平仓原因: 'signal', 'stop_loss', 'time_stop'
}
```

## 1. 数据模块 (lib/data.py)

### load_all_symbols_data
```python
data = load_all_symbols_data(
    symbols=['CU', 'ZN', 'AL'],  # 品种列表
    start_date='2020-01-01',     # 开始日期
    end_date='2024-12-31'        # 结束日期
) 
# 返回: pd.DataFrame, index=日期, columns=品种
```

## 2. 协整分析模块 (lib/coint.py)

### CointegrationAnalyzer + screen_all_pairs
```python
# 创建分析器
analyzer = CointegrationAnalyzer(data)  # data: DataFrame

# 筛选配对
from lib.coint import screen_all_pairs
pairs_df = screen_all_pairs(analyzer)
# 返回: pd.DataFrame，包含以下列:
#   - pair: 配对名称 'CU-ZN'
#   - symbol_x: X品种
#   - symbol_y: Y品种  
#   - beta_5y: 5年Beta系数
#   - pvalue_5y: 5年p值
#   - pvalue_1y: 1年p值
```

## 3. 信号生成模块 (lib/signal_generation.py)

### SignalGeneratorV3
```python
# 创建生成器
generator = SignalGeneratorV3(
    signal_start_date='2024-07-01',  # 信号开始日期
    kalman_warmup_days=30,           # Kalman预热天数
    ols_training_days=60,            # OLS训练天数
    z_open=2.0,                      # 开仓阈值
    z_close=0.5,                     # 平仓阈值
    max_holding_days=30,             # 最大持仓天数
    Q_beta=5e-6,                     # Beta过程噪声
    Q_alpha=1e-5,                    # Alpha过程噪声
    R_init=0.005,                    # 初始测量噪声
    R_adapt=True                     # 自适应R
)

# 处理单个配对
signals = generator.process_pair(
    pair_name='CU-ZN',               # 配对名称
    x_data=data['CU'],               # X品种数据 (Series)
    y_data=data['ZN'],               # Y品种数据 (Series)
    initial_beta=0.77                # 初始Beta（可选）
)
# 返回: pd.DataFrame，包含以下列:
#   - date: 日期 (index)
#   - z_score: Z分数
#   - trade_signal: 交易信号 ('open_long', 'open_short', 'close', None)
#   - beta: 动态Beta
#   - alpha: 动态Alpha
```

## 4. 回测模块 (lib/backtest/)

### 配置类
```python
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.trade_executor import ExecutionConfig  
from lib.backtest.risk_manager import RiskConfig
from lib.backtest.engine import BacktestConfig

# 仓位配置
sizing_config = PositionSizingConfig(
    max_denominator=10,      # 最大分母
    min_lots=1,              # 最小手数
    max_lots_per_leg=100,    # 每腿最大手数
    margin_rate=0.12,        # 保证金率
    position_weight=0.05     # 仓位权重
)

# 执行配置
exec_config = ExecutionConfig(
    commission_rate=0.0002,  # 手续费率
    slippage_ticks=3,        # 滑点tick数
    margin_rate=0.12         # 保证金率
)

# 风险配置
risk_config = RiskConfig(
    stop_loss_pct=0.10,      # 止损百分比
    max_holding_days=30,     # 最大持仓天数
    max_positions=20         # 最大持仓数（注意：不是max_concurrent_positions）
)

# 回测配置
config = BacktestConfig(
    initial_capital=5000000,         # 初始资金
    sizing_config=sizing_config,     # 仓位配置
    execution_config=exec_config,    # 执行配置（注意：不是exec_config）
    risk_config=risk_config          # 风险配置
)
```

### BacktestEngine
```python
from lib.backtest.engine import BacktestEngine

# 创建引擎
engine = BacktestEngine(config)

# 设置合约规格（通过executor）
engine.executor.set_contract_specs({
    'CU': {'multiplier': 5, 'tick_size': 10},
    'ZN': {'multiplier': 5, 'tick_size': 5}
})

# 准备信号数据（必须包含以下列）
signals_df = pd.DataFrame({
    'date': [...],           # 日期
    'pair': 'CU-ZN',        # 配对名称
    'symbol_x': 'CU',       # X品种
    'symbol_y': 'ZN',       # Y品种
    'trade_signal': [...],   # 信号：'open_long', 'open_short', 'close', None
    'beta': 0.77            # Beta系数
})

# 运行回测
results = engine.run(
    signals=signals_df,      # 信号DataFrame
    prices=data             # 价格数据DataFrame
)

# 返回结果
results = {
    'trades': [...],         # Trade对象列表
    'equity_curve': [...],   # 权益曲线列表
    'metrics': {...},        # 绩效指标字典
    'pair_metrics': {...}    # 配对级别指标
}
```

## 完整示例

```python
# 1. 加载数据
from lib.data import load_all_symbols_data
data = load_all_symbols_data(['CU', 'ZN'], '2020-01-01', '2024-12-31')

# 2. 协整分析
from lib.coint import CointegrationAnalyzer, screen_all_pairs
analyzer = CointegrationAnalyzer(data)
pairs = screen_all_pairs(analyzer)

# 3. 生成信号
from lib.signal_generation import SignalGeneratorV3
generator = SignalGeneratorV3('2024-07-01')
signals = generator.process_pair(
    pair_name='CU-ZN',
    x_data=data['CU'],
    y_data=data['ZN']
)

# 4. 准备回测数据
signals['date'] = signals.index
signals['pair'] = 'CU-ZN'
signals['symbol_x'] = 'CU'
signals['symbol_y'] = 'ZN'

# 5. 运行回测
from lib.backtest.engine import BacktestEngine, BacktestConfig
config = BacktestConfig()
engine = BacktestEngine(config)
engine.executor.set_contract_specs({...})
results = engine.run(signals, data)
```

## 常见错误

1. **参数名错误**
   - ❌ `exec_config` → ✅ `execution_config`
   - ❌ `max_concurrent_positions` → ✅ `max_positions`
   - ❌ `px/py` → ✅ `x_data/y_data`

2. **方法名错误**
   - ❌ `analyzer.analyze_all_pairs()` → ✅ `screen_all_pairs(analyzer)`
   - ❌ `engine.set_contract_specs()` → ✅ `engine.executor.set_contract_specs()`

3. **数据格式错误**
   - 信号DataFrame必须包含: date, pair, symbol_x, symbol_y, trade_signal, beta
   - 价格数据必须是DataFrame，index为日期，columns为品种代码