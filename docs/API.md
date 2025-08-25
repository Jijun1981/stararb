# API 文档

版本: 5.0  
更新日期: 2025-08-24

## 目录
- [数据管理模块](#数据管理模块)
- [协整分析模块](#协整分析模块)
- [信号生成模块](#信号生成模块)
- [回测引擎模块](#回测引擎模块)

---

## 数据管理模块

### `lib.data.load_data`

加载期货数据并进行预处理。

```python
def load_data(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: List[str] = ['close'],
    log_price: bool = False,
    fill_method: str = 'ffill',
    data_dir: Optional[Path] = None
) -> pd.DataFrame
```

**参数:**
- `symbols`: 品种列表，如 `['AG', 'AL', 'CU']`
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `columns`: 需要的数据列，默认 `['close']`
- `log_price`: 是否返回对数价格
- `fill_method`: 缺失值填充方法 ('ffill', 'bfill', None)
- `data_dir`: 数据目录（忽略，强制使用data-joint）

**返回:**
- `pd.DataFrame`: 按日期索引对齐的宽表，列名格式为 `{symbol}`

### `lib.data.load_all_symbols_data`

加载所有14个品种的数据。

```python
def load_all_symbols_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame
```

### `lib.data.update_symbol_data`

增量更新期货数据（当前为占位实现）。

```python
def update_symbol_data(symbol: str) -> bool
```

**参数:**
- `symbol`: 品种代码

**返回:**
- `bool`: 是否更新成功

### `lib.data.check_data_quality`

数据质量检查。

```python
def check_data_quality(symbol: str = None) -> Dict
```

**参数:**
- `symbol`: 品种代码，None表示检查所有品种

**返回:**
- `Dict`: 数据质量报告，包含：
  - `symbol`: 品种代码
  - `missing_ratio`: 缺失值比例
  - `outliers`: 异常值列表
  - `data_range`: 数据时间范围
  - `total_records`: 总记录数
  - `status`: 状态 (OK/WARNING/ERROR)

**示例:**
```python
from lib.data import load_data, check_data_quality

# 加载3个品种的收盘价
data = load_data(
    symbols=['AG', 'AL', 'CU'],
    start_date='2020-01-01',
    log_price=True
)

# 检查数据质量
report = check_data_quality('AG')
```

---

## 协整分析模块

### `lib.coint.CointegrationAnalyzer`

协整分析器，用于筛选和分析配对。

```python
class CointegrationAnalyzer:
    def __init__(self, data: pd.DataFrame)
    
    def test_pair_cointegration(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> Dict
    
    def calculate_beta(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float
    
    def screen_all_pairs(
        self,
        screening_windows: Optional[List[str]] = None,
        p_thresholds: Optional[Dict[str, float]] = None,
        filter_logic: str = 'AND',
        sort_by: str = 'pvalue_1y',
        ascending: bool = True,
        vol_start_date: Optional[str] = None,
        vol_end_date: Optional[str] = None,
        windows: Optional[Dict[str, int]] = None,
        p_threshold: float = 0.05,
        halflife_min: Optional[float] = None,
        halflife_max: Optional[float] = None,
        use_halflife_filter: bool = False,
        volatility_start_date: Optional[str] = None
    ) -> pd.DataFrame
```

**初始化参数:**
- `data`: 对数价格数据，index为日期，columns为品种代码

**test_pair_cointegration方法:**
- 执行Engle-Granger协整检验
- 返回包含p值、beta等信息的字典

**calculate_beta方法:**
- 计算配对的Beta系数
- 使用OLS回归估计

**screen_all_pairs参数:**
- `screening_windows`: 筛选使用的时间窗口列表
- `p_thresholds`: 各窗口的p值阈值字典
- `filter_logic`: 多窗口筛选逻辑 ('AND' 或 'OR')
- `sort_by`: 排序字段
- `ascending`: 是否升序排序
- `vol_start_date`: 波动率计算开始日期
- `vol_end_date`: 波动率计算结束日期
- `windows`: 自定义时间窗口配置
- `p_threshold`: p值筛选阈值（默认0.05）
- `halflife_min`: 最小半衰期阈值（可选）
- `halflife_max`: 最大半衰期阈值（可选）
- `use_halflife_filter`: 是否启用半衰期筛选
- `volatility_start_date`: 波动率计算起始日期（兼容旧参数）

**返回:**
- `pd.DataFrame`: 配对结果，包含以下列：
  - `pair`: 配对名称（格式：`X-Y`）
  - `symbol_x`: X品种代码（低波动）
  - `symbol_y`: Y品种代码（高波动）
  - `pvalue_1y` ~ `pvalue_5y`: 各窗口p值
  - `beta_1y` ~ `beta_5y`: 各窗口Beta系数
  - `halflife_1y` ~ `halflife_5y`: 各窗口半衰期
  - `volatility_x`, `volatility_y`: 波动率

**示例:**
```python
from lib.coint import CointegrationAnalyzer
from lib.data import load_data

# 加载数据
data = load_data(
    symbols=['AG', 'AL', 'CU', 'NI'],
    start_date='2020-01-01',
    log_price=True
)

# 协整分析
analyzer = CointegrationAnalyzer(data)

# 基础筛选
pairs = analyzer.screen_all_pairs(p_threshold=0.05)

# 参数化筛选
pairs = analyzer.screen_all_pairs(
    screening_windows=['1y', '2y'],
    p_thresholds={'1y': 0.05, '2y': 0.1},
    filter_logic='AND',
    vol_start_date='2024-01-01'
)
```

---

## 信号生成模块

### `lib.signal_generation.AdaptiveKalmanFilter`

自适应Kalman滤波器，使用折扣因子和EWMA自适应测量噪声。

```python
class AdaptiveKalmanFilter:
    def __init__(
        self, 
        pair_name: str,
        delta: float = 0.96,      # 优化后默认值
        lambda_r: float = 0.92,    # 优化后默认值
        beta_bounds: Optional[Tuple[float, float]] = None
    )
    
    def warm_up_ols(
        self, 
        x_data: np.ndarray, 
        y_data: np.ndarray, 
        window: int = 60
    ) -> Dict
    
    def update(self, y_t: float, x_t: float) -> Dict
    
    def calibrate_delta(self, window: int = 60) -> bool
    
    def get_quality_metrics(self, window: int = 60) -> Dict
```

**初始化参数:**
- `pair_name`: 配对名称
- `delta`: 折扣因子（优化后默认0.96，原为0.98）
- `lambda_r`: R的EWMA参数（优化后默认0.92，原为0.96）
- `beta_bounds`: β边界限制（已移除限制）

**核心参数优化（2025-08-24突破）:**
- `delta_init`: 0.98 → 0.96 (关键突破)
- `lambda_r`: 0.96 → 0.92 (EWMA参数优化)
- `delta_min`: 0.95 → 0.90 (下界放宽)

**warm_up_ols方法:**
- 使用OLS预热获得初始参数
- 采用去中心化处理避免截距问题
- 返回初始β、R、P等参数

**update方法:**
- 执行折扣Kalman更新
- 返回包含beta、创新v、创新方差S、标准化创新z等

**calibrate_delta方法:**
- 基于z-score方差自动校准δ
- 目标：保持z_var在[0.8, 1.3]区间
- δ调整范围：[0.90, 0.995]

**get_quality_metrics方法:**
- 返回质量指标：z_var、z_mean、z_std、质量状态等

### `lib.signal_generation.AdaptiveSignalGenerator`

基于自适应Kalman滤波的信号生成器，集成了双旋钮Kalman和状态机。

```python
class AdaptiveSignalGenerator:
    def __init__(
        self,
        z_open: float = 2.0,
        z_close: float = 0.5,
        max_holding_days: int = 30,
        calibration_freq: int = 5,
        ols_window: int = 60,
        warm_up_days: int = 30
    )
    
    def process_all_pairs(
        self,
        pairs_df: pd.DataFrame,
        price_data: pd.DataFrame,
        beta_window: str = '1y'
    ) -> pd.DataFrame
    
    def get_quality_report(self) -> pd.DataFrame
```

**初始化参数:**
- `z_open`: 开仓Z-score阈值（默认2.0）
- `z_close`: 平仓Z-score阈值（默认0.5）
- `max_holding_days`: 最大持仓天数（默认30）
- `calibration_freq`: δ校准频率（默认每5天）
- `ols_window`: OLS预热窗口（默认60天）
- `warm_up_days`: Kalman预热天数（默认30天）

**process_all_pairs参数:**
- `pairs_df`: 协整配对信息DataFrame
- `price_data`: 原始价格数据（会自动转对数）
- `beta_window`: 使用的Beta时间窗口

**返回:**
- `pd.DataFrame`: 信号数据，包含以下列：
  - `date`: 日期
  - `pair`: 配对名称
  - `symbol_x`: X品种代码
  - `symbol_y`: Y品种代码
  - `x_price`: X原始价格
  - `y_price`: Y原始价格
  - `beta`: 当前Beta值（Kalman滤波估计）
  - `spread`: 价差（对数空间）
  - `z_score`: 标准化创新（z = v/√S）
  - `position`: 当前持仓状态
  - `trade_signal`: 交易信号
  - `innovation`: 创新值v
  - `S`: 创新方差
  - `phase`: 阶段 (warm_up_period/signal_period)
  - `delta`: 当前折扣因子
  - `R`: 当前测量噪声
  - `quality_status`: 质量状态

### `lib.signal_generation.generate_signal`

信号生成状态机函数。

```python
def generate_signal(
    z_score: float,
    position: Optional[str],
    days_held: int,
    z_open: float,
    z_close: float,
    max_holding_days: int
) -> Tuple[Optional[str], str]
```

**参数:**
- `z_score`: 当前Z-score值
- `position`: 当前持仓 (None/'long'/'short')
- `days_held`: 已持仓天数
- `z_open`: 开仓阈值
- `z_close`: 平仓阈值
- `max_holding_days`: 最大持仓天数

**返回:**
- `(position, signal)`: 新持仓状态和交易信号

**信号类型:**
- `open_long`: Z < -z_open时做多
- `open_short`: Z > z_open时做空
- `close`: 平仓（到达平仓条件）
- `force_close`: 强制平仓（超过最大持仓天数）
- `None`: 无信号

**示例:**
```python
from lib.signal_generation import AdaptiveSignalGenerator
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer

# 1. 加载数据
price_data = load_all_symbols_data(start_date='2020-01-01')

# 2. 协整分析
log_data = np.log(price_data)
analyzer = CointegrationAnalyzer(log_data)
pairs = analyzer.screen_all_pairs(p_threshold=0.05)

# 3. 生成信号（使用优化后的参数）
sg = AdaptiveSignalGenerator(
    z_open=2.0,
    z_close=0.5,
    max_holding_days=30,
    calibration_freq=5
)

# 处理所有配对
signals = sg.process_all_pairs(
    pairs_df=pairs,
    price_data=price_data,
    beta_window='1y'
)

# 获取质量报告
quality_report = sg.get_quality_report()
print(f"Good质量配对: {(quality_report['quality_status'] == 'good').sum()}")
```

---

## 回测引擎模块

### 模块结构
```
lib/backtest/
├── __init__.py           # 模块初始化
├── position_sizing.py    # 手数计算模块
├── trade_executor.py     # 交易执行模块
├── risk_manager.py       # 风险管理模块
├── performance.py        # 绩效分析模块
└── engine.py            # 回测引擎
```

### `lib.backtest.PositionSizer`

手数计算器，负责根据β值计算最优手数配比。

```python
class PositionSizer:
    def __init__(self, config: PositionSizingConfig)
    
    def calculate_min_integer_ratio(
        self, 
        beta: float
    ) -> Dict[str, Any]
    
    def calculate_position_size(
        self,
        lots: Dict[str, int],
        prices: Dict[str, float],
        multipliers: Dict[str, float],
        available_capital: float,
        position_weight: float = 0.05
    ) -> Dict[str, Any]
```

### `lib.backtest.position_sizing.PositionSizer`

手数计算器，负责根据β值计算最优手数配比并应用资金约束。

```python
class PositionSizer:
    def __init__(self, config: PositionSizingConfig)
    
    def calculate_min_integer_ratio(
        self, 
        beta: float
    ) -> Dict[str, Any]
    
    def calculate_position_size(
        self,
        lots: Dict[str, int],
        prices: Dict[str, float],
        multipliers: Dict[str, float],
        total_capital: float,
        position_weight: float = 0.05
    ) -> Dict[str, Any]
```

**calculate_min_integer_ratio返回:**
```python
{
    'lots_x': int,              # X品种最小手数
    'lots_y': int,              # Y品种最小手数
    'theoretical_ratio': float,  # 理论比例(beta)
    'actual_ratio': float,       # 实际比例
    'error_pct': float          # 误差百分比
}
```

**calculate_position_size返回:**
```python
{
    'final_lots_x': int,         # X品种实际手数
    'final_lots_y': int,         # Y品种实际手数
    'allocated_capital': float,  # 分配的资金
    'margin_required': float,    # 实际占用保证金
    'position_value': float,     # 名义价值
    'scaling_factor': float,     # 缩放因子
    'utilization_rate': float,   # 资金利用率
    'can_trade': bool,          # 是否可交易
    'reason': str               # 不可交易原因(如有)
}
```

### `lib.backtest.BacktestEngine`

期货配对交易回测引擎（协调器）。

```python
class BacktestEngine:
    def __init__(self, config: BacktestConfig)
    
    def run(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame,
        contract_specs: Optional[Dict] = None
    ) -> Dict[str, Any]
```

**配置参数（BacktestConfig）:**
```python
@dataclass
class BacktestConfig:
    # 资金管理
    initial_capital: float = 5000000
    
    # 子模块配置
    sizing: PositionSizingConfig
    execution: ExecutionConfig  
    risk: RiskConfig
```

**返回结果结构:**
```python
{
    'portfolio_metrics': {       # 组合级别指标
        'total_return': float,
        'annual_return': float,
        'sharpe_ratio': float,
        'sortino_ratio': float,
        'max_drawdown': float,
        'win_rate': float,
        'profit_factor': float,
        # ... 更多指标
    },
    'pair_metrics': pd.DataFrame({  # 配对级别指标
        'pair': str,
        'total_pnl': float,
        'sharpe_ratio': float,
        'sortino_ratio': float,
        'max_drawdown': float,
        # ... 每个配对的完整指标
    }),
    'trades': pd.DataFrame,      # 交易明细
    'equity_curve': pd.Series,   # 组合权益曲线
    'pair_equity_curves': Dict[str, pd.Series], # 各配对权益曲线
    'contributions': pd.DataFrame,  # 配对贡献分析
    'correlations': pd.DataFrame   # 配对相关性矩阵
}
```

### `lib.backtest.performance.PerformanceAnalyzer`

绩效分析器，计算组合和配对级别的全面指标。

```python
class PerformanceAnalyzer:
    def calculate_trade_pnl(
        self, 
        trade: Trade
    ) -> Dict[str, float]
    
    def calculate_portfolio_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        daily_returns: pd.Series
    ) -> Dict[str, float]
    
    def calculate_pair_metrics(
        self,
        pair: str,
        trades: List[Trade]
    ) -> Dict[str, Any]
    
    def analyze_all_pairs(
        self,
        trades: List[Trade]
    ) -> pd.DataFrame
    
    def calculate_contribution_analysis(
        self,
        trades: List[Trade],
        portfolio_metrics: Dict
    ) -> pd.DataFrame
```

**组合级别指标（calculate_portfolio_metrics）:**
- 收益指标：total_return, annual_return, monthly_return
- 风险指标：volatility, sharpe_ratio, sortino_ratio, calmar_ratio
- 回撤指标：max_drawdown, max_dd_duration, recovery_time
- 交易统计：total_trades, win_rate, profit_factor, avg_win, avg_loss
- 风险度量：skewness, kurtosis, var_95, cvar_95

**配对级别指标（calculate_pair_metrics）:**
- 基本信息：pair, symbol_x, symbol_y
- 收益指标：total_pnl, total_return, annual_return
- 风险指标：volatility, sharpe_ratio, sortino_ratio, max_drawdown
- 交易统计：num_trades, win_rate, avg_pnl, avg_holding_days
- 止损统计：stop_loss_count, stop_loss_pnl, time_stop_count
- 手数统计：avg_lots_x, avg_lots_y, avg_beta
- 时间序列：equity_curve, daily_returns

**示例:**
```python
from lib.backtest import BacktestEngine
from lib.signal_generation import AdaptiveSignalGenerator
from lib.data import load_all_symbols_data

# 1. 准备数据
price_data = load_all_symbols_data()
signals = sg.process_all_pairs(pairs, price_data)

# 2. 创建回测引擎
engine = BacktestEngine(
    initial_capital=5000000,
    margin_rate=0.12,
    commission_rate=0.0002,
    stop_loss_pct=0.10
)

# 3. 运行回测
results = engine.run(
    signals_df=signals,
    price_data=price_data
)

# 4. 查看结果
print(f"年化收益: {results['metrics']['annual_return']:.2%}")
print(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")
print(f"胜率: {results['metrics']['win_rate']:.2%}")
```

---

## 完整工作流示例

```python
# 1. 数据管理
from lib.data import load_all_symbols_data, check_data_quality

price_data = load_all_symbols_data(start_date='2020-01-01')
quality = check_data_quality()
print(f"数据质量: {quality['status']}")

# 2. 协整分析
from lib.coint import CointegrationAnalyzer
import numpy as np

log_data = np.log(price_data)
analyzer = CointegrationAnalyzer(log_data)
pairs = analyzer.screen_all_pairs(
    screening_windows=['1y', '2y'],
    p_thresholds={'1y': 0.05, '2y': 0.1},
    vol_start_date='2024-01-01'
)
print(f"筛选出 {len(pairs)} 个配对")

# 3. 信号生成（使用优化后的Kalman参数）
from lib.signal_generation import AdaptiveSignalGenerator

sg = AdaptiveSignalGenerator(
    z_open=2.0,
    z_close=0.5,
    max_holding_days=30,
    calibration_freq=5,
    ols_window=60,
    warm_up_days=30
)

# 处理所有配对
signals = sg.process_all_pairs(
    pairs_df=pairs,
    price_data=price_data,
    beta_window='1y'
)

# 查看质量报告
quality_report = sg.get_quality_report()
good_pairs = quality_report[quality_report['quality_status'] == 'good']
print(f"高质量配对: {len(good_pairs)}")

# 4. 回测（模块化版本）
from lib.backtest import BacktestEngine, BacktestConfig
from lib.backtest.position_sizing import PositionSizingConfig
from lib.backtest.risk_manager import RiskConfig
from lib.backtest.trade_executor import ExecutionConfig

# 配置各模块
sizing_config = PositionSizingConfig(
    max_denominator=10,
    min_lots=1,
    max_lots_per_leg=50,
    position_weight=0.05  # 每配对5%资金
)

risk_config = RiskConfig(
    stop_loss_pct=0.15,
    max_holding_days=30,
    max_positions=20
)

execution_config = ExecutionConfig(
    commission_rate=0.0002,
    slippage_ticks=3,
    margin_rate=0.12
)

# 创建回测引擎
config = BacktestConfig(
    initial_capital=5000000,
    sizing_config=sizing_config,
    risk_config=risk_config,
    execution_config=execution_config
)

engine = BacktestEngine(config)

results = engine.run(
    signals_df=signals,
    price_data=price_data
)

# 5. 结果分析
print(f"\n回测结果:")
print(f"  年化收益: {results['metrics']['annual_return']:.2%}")
print(f"  夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
print(f"  最大回撤: {results['metrics']['max_drawdown']:.2%}")
print(f"  胜率: {results['metrics']['win_rate']:.2%}")
print(f"  总交易次数: {len(results['trades'])}")
```

---

## 重要更新说明

### Kalman参数优化突破（2025-08-24）

通过81个参数组合的网格搜索，发现了显著提升系统性能的最优参数：

**参数调整:**
- `delta_init`: 0.98 → 0.96（关键突破）
- `lambda_r`: 0.96 → 0.92（EWMA优化）
- `delta_min`: 0.95 → 0.90（下界放宽）

**性能提升:**
- 交易信号数量：112 → 266（+137.5%）
- |z|≥2比例：0.98% → 1.80%（接近理想区间）
- 残差平稳性：Kalman 83.3% vs OLS 80%
- 系统动态性：6个配对保持δ<0.995（原仅1个）

**技术原理:**
- 较低的初始δ为系统提供更多"动态空间"
- 避免过早收敛到δ=0.995上界
- 增强的P矩阵折扣使Kalman增益更敏感
- 更好地捕捉配对关系的时变特性

详见：`docs/kalman_parameter_optimization_breakthrough.md`

---

## 更新历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2024-01-15 | 初始版本 |
| 2.0 | 2024-08-01 | 更新数据源为data-joint |
| 3.0 | 2024-08-23 | 统一配对命名规范 |
| 4.0 | 2025-08-23 | 更新所有模块接口，添加参数化支持 |
| 5.0 | 2025-08-24 | 实现自适应Kalman滤波，参数优化突破，性能大幅提升 |