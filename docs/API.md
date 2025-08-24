# API 文档

版本: 4.0  
更新日期: 2025-08-23

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

### `lib.signal_generation.KalmanFilter1D`

一维Kalman滤波器，用于动态估计Beta系数。所有Kalman参数已写死，不可配置。

```python
class KalmanFilter1D:
    def __init__(self, initial_beta: float)
    
    def update(self, y_t: float, x_t: float) -> Dict
```

**初始化参数:**
- `initial_beta`: 初始Beta值（从协整模块获取）

**固定参数（写死）:**
- `Q = 1e-4`: 过程噪声
- `R = 1.0`: 观测噪声初始值
- `P = 0.1`: 初始不确定性

### `lib.signal_generation.SignalGenerator`

基于Kalman滤波的动态信号生成器。

```python
class SignalGenerator:
    def __init__(
        self,
        window: int = 60,
        z_open: float = 2.0,
        z_close: float = 0.5,
        convergence_days: int = 30,
        convergence_threshold: float = 0.02,
        max_holding_days: int = 30
    )
    
    def process_pair_signals(
        self,
        pair_data: pd.DataFrame,
        initial_beta: float,
        convergence_end: str,
        signal_start: str,
        pair_info: Dict,
        beta_window: str = '1y'
    ) -> pd.DataFrame
```

**初始化参数（可配置）:**
- `window`: 滚动窗口大小（默认60）
- `z_open`: 开仓Z-score阈值（默认2.0）
- `z_close`: 平仓Z-score阈值（默认0.5）
- `convergence_days`: 收敛期天数（默认30）
- `convergence_threshold`: 收敛判断阈值（默认0.02）
- `max_holding_days`: 最大持仓天数（默认30）

**process_pair_signals参数:**
- `pair_data`: 配对价格数据，包含'date', 'x', 'y'列（对数价格）
- `initial_beta`: 初始Beta值（从协整模块获取）
- `convergence_end`: 收敛期结束日期
- `signal_start`: 信号期开始日期
- `pair_info`: 配对信息字典（从协整模块获取）
- `beta_window`: 使用的Beta时间窗口（如'1y', '2y'等）

**返回:**
- `pd.DataFrame`: 信号数据，包含以下列（REQ-4.3规范）：
  - `date`: 日期
  - `pair`: 配对名称
  - `symbol_x`: X品种代码
  - `symbol_y`: Y品种代码
  - `signal`: 信号类型 (converging/open_long/open_short/close/hold)
  - `z_score`: 当前Z-score值
  - `residual`: 当前残差值
  - `beta`: 当前Beta值（Kalman滤波估计）
  - `beta_initial`: 初始Beta值
  - `days_held`: 持仓天数
  - `reason`: 信号原因
  - `phase`: 阶段 (convergence_period/signal_period)
  - `beta_window_used`: 使用的Beta窗口

**示例:**
```python
from lib.signal_generation import SignalGenerator
from lib.data import load_data
import numpy as np

# 准备配对数据
pair_data = pd.DataFrame({
    'date': dates,
    'x': np.log(x_prices),  # 对数价格
    'y': np.log(y_prices)
})

# 配对信息（从协整模块获取）
pair_info = {
    'pair': 'AG-NI',
    'symbol_x': 'AG',
    'symbol_y': 'NI',
    'beta_1y': 0.8234
}

# 生成信号
sg = SignalGenerator(
    window=60,
    z_open=2.0,
    z_close=0.5,
    max_holding_days=30
)

signals = sg.process_pair_signals(
    pair_data=pair_data,
    initial_beta=pair_info['beta_1y'],
    convergence_end='2023-12-31',
    signal_start='2024-01-01',
    pair_info=pair_info,
    beta_window='1y'
)
```

---

## 回测引擎模块

### `lib.backtest.run_backtest`

执行回测分析。

```python
def run_backtest(
    signal_df: pd.DataFrame,
    initial_capital: float = 5000000,
    position_size: float = 0.05,
    commission: float = 0.0002,
    slippage: float = 3,
    stop_loss: float = 0.15,
    max_holding_days: int = 30
) -> Dict
```

**参数:**
- `signal_df`: 信号数据（从信号生成模块获取）
- `initial_capital`: 初始资金（默认500万）
- `position_size`: 仓位比例（默认5%）
- `commission`: 手续费率（默认万分之2）
- `slippage`: 滑点（tick数，默认3）
- `stop_loss`: 止损比例（默认15%）
- `max_holding_days`: 最大持仓天数（默认30）

**返回:**
- `Dict`: 回测结果，包含：
  - `total_return`: 总收益率
  - `annual_return`: 年化收益率
  - `sharpe_ratio`: 夏普比率
  - `max_drawdown`: 最大回撤
  - `trades`: 交易记录
  - `pnl_curve`: PnL曲线

**示例:**
```python
from lib.backtest import run_backtest

# 执行回测
results = run_backtest(
    signal_df=signals,
    initial_capital=5000000,
    position_size=0.05,
    stop_loss=0.15
)

print(f"年化收益: {results['annual_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

---

## 完整工作流示例

```python
# 1. 数据管理
from lib.data import load_all_symbols_data, check_data_quality

data = load_all_symbols_data(start_date='2020-01-01')
quality = check_data_quality()
print(f"数据质量: {quality['status']}")

# 2. 协整分析
from lib.coint import CointegrationAnalyzer

analyzer = CointegrationAnalyzer(data)
pairs = analyzer.screen_all_pairs(
    screening_windows=['1y', '2y'],
    p_thresholds={'1y': 0.05, '2y': 0.1},
    vol_start_date='2024-01-01'
)
print(f"筛选出 {len(pairs)} 个配对")

# 3. 信号生成
from lib.signal_generation import SignalGenerator
import pandas as pd

sg = SignalGenerator(window=60, z_open=2.0, z_close=0.5)
all_signals = []

for _, pair_info in pairs.iterrows():
    # 准备配对数据
    pair_data = pd.DataFrame({
        'date': data.index,
        'x': data[pair_info['symbol_x']].values,
        'y': data[pair_info['symbol_y']].values
    })
    
    # 生成信号
    signals = sg.process_pair_signals(
        pair_data=pair_data,
        initial_beta=pair_info['beta_1y'],
        convergence_end='2023-12-31',
        signal_start='2024-01-01',
        pair_info=pair_info.to_dict(),
        beta_window='1y'
    )
    all_signals.append(signals)

# 合并所有信号
all_signals_df = pd.concat(all_signals, ignore_index=True)

# 4. 回测
from lib.backtest import run_backtest

results = run_backtest(
    signal_df=all_signals_df,
    initial_capital=5000000,
    position_size=0.05
)

print(f"回测结果:")
print(f"  年化收益: {results['annual_return']:.2%}")
print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
print(f"  最大回撤: {results['max_drawdown']:.2%}")
```

---

## 更新历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2024-01-15 | 初始版本 |
| 2.0 | 2024-08-01 | 更新数据源为data-joint |
| 3.0 | 2024-08-23 | 统一配对命名规范 |
| 4.0 | 2025-08-23 | 更新所有模块接口，添加参数化支持，Kalman参数写死 |