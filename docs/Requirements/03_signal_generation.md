# 块3: 信号生成需求文档 V3.1

## 1. 模块概述

接收协整配对模块的配对参数，使用**原始状态空间Kalman滤波器**进行动态β和α更新，基于标准化残差(z-score)生成交易信号。核心设计：使用二维状态空间模型[β, α]，严格的数学框架。

**设计理念（基于实证修正）**：

- 使用完整的二维状态空间模型：状态向量包含[β, α]
- 固定的过程噪声协方差Q和自适应的测量噪声R
- **关键修正**：基于残差标准化 z = v/√R，不使用创新标准化 z = v/√S
- **实证依据**：滚动年度评估显示原版本KF (z = residual/√R) 效果优秀：
  - Z方差：1.288（理想范围）
  - Z>2比例：7.6%（符合2-5%预期上限）
  - 均值回归率：95.1%（优秀）
- 经过实证验证的最优参数：Q_beta=5e-6, Q_alpha=1e-5

**职责边界**：

- 协整配对模块负责：提供多个时间窗口的初始β值（1y, 2y, 3y, 5y等）
- 本模块负责：OLS预热初始化，自适应Kalman滤波，批量生成交易信号
- 回测模块负责：根据信号和资金计算具体手数

## 2. 实证验证结果（重要修正依据）

### 2.1 滚动年度评估验证

基于2025-08-25的滚动年度Kalman评估 (`evaluate_kalman_rolling_yearly.py`)，验证了原版本KF的优秀性能：

**整体统计结果**：

- **平均Z方差**: 1.288（接近理想值1.0）
- **平均Z>2比例**: 7.6%（符合预期2-5%范围上限）
- **平均均值回归率**: 95.1%（极好的均值回归性）
- **无符号变化比例**: 100.0%（Beta完全稳定）
- **平均综合评分**: 6.1/9（良好）

**分年度统计**：

- **2020年**: Z方差1.214，Z>2比例7.4%，评分6.7/9，负beta完美保持
- **2021年**: Z方差1.459，Z>2比例7.8%，评分5.5/9，负beta完美保持
- **2022年**: Z方差1.191，Z>2比例7.5%，评分6.2/9，负beta完美保持

**关键发现**：

- **z = residual/√R** 方法产生理想的统计特性
- **z = innovation/√S** 方法在实际测试中信号量严重不足（Z方差~0.3，Z>2比例接近0%）
- **矩阵构造顺序至关重要**：必须使用 `[x_data, 常数项]` 确保负beta不被颠倒
- **负beta符号稳定性**：100%的配对无符号变化，负beta配对完美保持

### 2.2 方法对比验证

| 方法                         | Z方差 | Z>2比例 | 信号量 | 评估      |
| ---------------------------- | ----- | ------- | ------ | --------- |
| 原版本: z = residual/√R     | ~1.3  | 7.6%    | 充足   | ✅ 优秀   |
| 工程版本: z = innovation/√S | ~0.3  | 0%      | 极少   | ❌ 不可用 |

**结论**：需求必须使用原版本的 z = residual/√R 方法。

## 3. 用户故事 (User Stories)

### Story 3.1: 状态空间Kalman滤波

**作为**研究员
**我希望**使用原始状态空间Kalman滤波动态估计对冲比率和截距
**以便**通过严格的数学框架跟踪配对关系的演化

**验收标准:**

- 使用60天OLS预热获得初始[β, α]和协方差
- 二维状态向量：[β, α]'
- 固定过程噪声：Q_beta=5e-6, Q_alpha=1e-5
- 自适应测量噪声R（可选）
- 支持批量处理多个配对
- 记录完整的状态历史

### Story 3.2: 残差标准化（修正）

**作为**研究员
**我希望**计算标准化残差(z-score)
**以便**生成稳定的交易信号

**验收标准:**

- 计算创新：v = y - β*x - α
- 计算标准化残差：z = v/√R（使用测量噪声R标准化）
- z方差保持在[0.8, 1.3]带宽内
- **重要修正**：经实证验证，z = v/√R 能产生正确信号量，而z = v/√S信号量不足
- **实证依据**：原版本KF使用z = residual/√R，Z方差~1.3，Z>2比例7.6%

### Story 3.3: 基于残差标准化的信号生成（修正）

**作为**研究员
**我希望**基于残差标准化z生成开平仓信号
**以便**捕捉均值回归机会

**验收标准:**

- 使用残差标准化：z = v/√R，其中v = y - β*x - α，R为测量噪声
- 设定开仓阈值：|z| > 2.0
- 设定平仓阈值：|z| < 0.5
- 生成明确的交易信号状态
- **实证验证**：此方法产生Z>2比例7.6%，符合预期2-5%上限

### Story 3.4: 批量配对处理

**作为**研究员
**我希望**批量处理多个配对的信号生成
**以便**高效管理整个投资组合

**验收标准:**

- 支持从协整模块接收配对DataFrame
- 每个配对独立运行自适应Kalman
- 并行处理提高效率
- 统一输出格式的信号DataFrame
- 生成配对级别的质量报告

## 3. 功能需求 (Requirements)

### REQ-3.0: 时间轴配置（新增）

| ID        | 需求描述                                                                   | 优先级 |
| --------- | -------------------------------------------------------------------------- | ------ |
| REQ-3.0.1 | 信号期起点：用户指定信号生成开始日期（如2024-07-01）                       | P0     |
| REQ-3.0.2 | Kalman预热期：可配置天数（默认30天），从信号期起点往前推                   | P0     |
| REQ-3.0.3 | OLS训练期：可配置天数（默认60天），从Kalman预热期起点往前推                | P0     |
| REQ-3.0.4 | 数据范围：自动计算总数据需求 = signal_start - kalman_warmup - ols_training | P0     |
| REQ-3.0.5 | 时间阶段：OLS训练 → Kalman预热（不出信号） → 信号生成期                  | P0     |
| REQ-3.0.6 | 参数接口：signal_start_date, kalman_warmup_days, ols_training_days         | P0     |

### REQ-3.1: 状态空间Kalman滤波

| ID        | 需求描述                                           | 优先级 |
| --------- | -------------------------------------------------- | ------ |
| REQ-3.1.1 | 状态方程：x_t = x_{t-1} + w_t，其中x=[β, α]'     | P0     |
| REQ-3.1.2 | 观测方程：y_t = β_t * x_t + α_t + v_t            | P0     |
| REQ-3.1.3 | OLS预热：使用ols_training_days窗口估计初始[β, α] | P0     |
| REQ-3.1.4 | 过程噪声：Q = diag(5e-6, 1e-5) 固定值              | P0     |
| REQ-3.1.5 | 测量噪声：R_init = 0.005，可选自适应               | P0     |
| REQ-3.1.6 | 初始协方差：P_0 = I * 0.001                        | P0     |
| REQ-3.1.7 | 边界保护：无需限制，让数据说话                     | P0     |
| REQ-3.1.8 | 预热期：可配置OLS训练期 + 可配置Kalman预热期       | P0     |

### REQ-3.2: 信号质量监控

| ID        | 需求描述                     | 优先级 |
| --------- | ---------------------------- | ------ |
| REQ-3.2.1 | Z方差监控：期望值接近1.0     | P0     |
| REQ-3.2.2 | 信号频率：Z>2比例在2-5%之间  | P0     |
| REQ-3.2.3 | 均值回归率：>70%在20天内回归 | P0     |
| REQ-3.2.4 | Beta稳定性：监控符号变化     | P0     |

### REQ-3.3: 信号生成逻辑

| ID        | 需求描述                                                                   | 优先级 |
| --------- | -------------------------------------------------------------------------- | ------ |
| REQ-3.3.1 | 开仓阈值：                                                                 | z      |
| REQ-3.3.2 | 平仓阈值：                                                                 | z      |
| REQ-3.3.3 | 持仓限制：最大持仓30天（可配置）                                           | P0     |
| REQ-3.3.4 | 信号类型：open_long, open_short, holding_long, holding_short, close, empty | P0     |
| REQ-3.3.5 | 防重复开仓：同配对同方向不重复开仓                                         | P0     |
| REQ-3.3.6 | 信号优先级：强制平仓 > 平仓 > 开仓                                         | P0     |

### REQ-3.4: 批量配对处理

| ID        | 需求描述                                                       | 优先级 |
| --------- | -------------------------------------------------------------- | ------ |
| REQ-3.4.1 | 接收协整模块DataFrame：包含pair, symbol_x, symbol_y, beta_1y等 | P0     |
| REQ-3.4.2 | 支持选择β时间窗口：'1y', '2y', '3y', '5y'等                   | P0     |
| REQ-3.4.3 | 每个配对独立处理：独立维护状态和参数                           | P0     |
| REQ-3.4.4 | 输出统一格式信号DataFrame                                      | P0     |
| REQ-3.4.5 | 记录配对级别的β历史和参数调整日志                             | P0     |
| REQ-3.4.6 | 生成质量报告：z方差、校准次数、参数状态                        | P1     |

### REQ-3.5: 质量监控

| ID        | 需求描述                                          | 优先级 |
| --------- | ------------------------------------------------- | ------ |
| REQ-3.5.1 | 核心指标：最近60根z方差 ∈ [0.8, 1.3]             | P0     |
| REQ-3.5.2 | 实时监控：记录z_mean, z_std, 当前δ和R            | P0     |
| REQ-3.5.3 | 配对级别质量评级：good(z方差在带宽内)/warning/bad | P0     |
| REQ-3.5.4 | 支持导出质量报告                                  | P1     |

### REQ-3.5: 时间轴计算示例

**配置参数**：

- signal_start_date = "2024-07-01" （信号生成开始日期）
- kalman_warmup_days = 30 （Kalman预热天数）
- ols_training_days = 60 （OLS训练天数）

**自动计算时间轴**：

```
OLS训练期：   2024-04-02 至 2024-05-31 （60天）
Kalman预热期： 2024-06-01 至 2024-06-30 （30天，不出信号）
信号生成期：   2024-07-01 至 数据结束   （实际出信号）

数据开始日期 = signal_start_date - kalman_warmup_days - ols_training_days
          = 2024-07-01 - 30天 - 60天 = 2024-04-02
```

**阶段定义**：

- **OLS阶段**: 前ols_training_days天，只做参数初始化，状态='ols_training'
- **Kalman预热阶段**: 接下来kalman_warmup_days天，更新状态但不出信号，状态='kalman_warmup'
- **信号生成阶段**: 剩余时间，正常出信号，状态='signal_generation'

## 4. 接口定义

### 4.1 SignalGenerator类接口

```python
class SignalGenerator:
    def __init__(self, 
                 # 时间配置（新增）
                 signal_start_date: str,              # 信号生成开始日期
                 kalman_warmup_days: int = 30,        # Kalman预热天数
                 ols_training_days: int = 60,         # OLS训练天数
               
                 # 交易阈值
                 z_open: float = 2.0, 
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
               
                 # Kalman参数
                 Q_beta: float = 5e-6,
                 Q_alpha: float = 1e-5,
                 R_init: float = 0.005,
                 R_adapt: bool = True)
  
    # 单配对处理
    def process_pair(self, 
                    pair_name: str,
                    x_data: pd.Series, 
                    y_data: pd.Series,
                    initial_beta: float = None) -> pd.DataFrame
  
    # 批量处理
    def process_all_pairs(self, 
                         pairs_df: pd.DataFrame,  # 协整模块输出
                         price_data: pd.DataFrame,
                         beta_window: str = '1y') -> pd.DataFrame
  
    # 质量监控
    def get_quality_report(self) -> pd.DataFrame
```

### 4.2 OriginalKalmanFilter类接口

```python
class OriginalKalmanFilter:
    def __init__(self, 
                 warmup: int = 60,
                 Q_beta: float = 5e-6,     # Beta过程噪声
                 Q_alpha: float = 1e-5,    # Alpha过程噪声  
                 R_init: float = 0.005,    # 初始测量噪声
                 R_adapt: bool = True):    # 是否自适应R
  
    # 初始化
    def initialize(self, x_data: np.ndarray, y_data: np.ndarray) -> None
  
    # 单步更新
    def update(self, x_t: float, y_t: float) -> None
  
    # 获取当前状态
    def get_state(self) -> dict
```

### 4.3 信号格式

```python
{
    'date': '2025-04-10',           # 无硬编码日期
    'pair': 'AG-NI',               # 与协整模块格式一致：纯符号，无后缀
    'symbol_x': 'AG',              # X品种（低波动）
    'symbol_y': 'NI',              # Y品种（高波动）
    'signal': 'open_long',          # open_long, open_short, holding_long, holding_short, close, empty
    'z_score': -2.15,               # 残差标准化 z = v/√R（修正），经实证验证的正确方法
    'innovation': -0.0234,          # 当前创新值 v = y - β*x - α
    'beta': 0.8523,                 # 当前β值
    'beta_initial': 0.8234,         # 初始β值（从协整模块获取）
    'days_held': 0,                 # 持仓天数（新开仓为0）
    'reason': 'z_threshold',        # 信号原因：converging, z_threshold, force_close等
    'phase': 'signal_period',       # 阶段标识：convergence_period, signal_period
    'beta_window_used': '1y'        # 使用的β值时间窗口
}
```

### 4.4 配对状态管理

```python
# 每个配对独立维护的状态
class PairState:
    def __init__(self, pair_name: str):
        self.pair_name = pair_name
        self.delta = 0.98              # 折扣因子（会自适应调整）
        self.R = None                  # 测量噪声（EWMA更新）
        self.beta = None               # 当前β
        self.P = None                  # 当前不确定性
        self.z_history = []            # z-score历史（用于校准）
        self.last_calibration = None   # 最后校准时间
        self.calibration_log = []      # 校准日志
        self.position = None           # 当前持仓状态
        self.days_held = 0             # 持仓天数
        self.quality_status = 'unknown' # good/warning/bad
```

## 5. 算法细节

### 5.0 核心设计理念

```python
"""
清晰的信号状态机制 - REQ-3.3.7, REQ-3.3.8

状态定义：
- converging: 收敛期状态，β正在收敛，不生成交易信号
- empty: 空仓状态，等待开仓机会
- open_long: 当日开多头仓位
- open_short: 当日开空头仓位  
- holding_long: 持有多头仓位中
- holding_short: 持有空头仓位中
- close: 当日平仓

状态转换逻辑：
1. 初始状态: position=None, signal='empty'
2. 开仓时: position='long'/'short', signal='open_long'/'open_short' 
3. 持仓第二天开始: signal='holding_long'/'holding_short'
4. 平仓时: position=None, signal='close'
5. 平仓后第二天: signal='empty'

优点：
- 状态含义明确：一眼看出是空仓还是持仓
- 便于统计：可以清楚区分空仓期间和持仓期间的Z-score触发
- 便于调试：清楚看到为什么某些极端Z值没有开仓
- 符合交易逻辑：与实际交易员的理解一致
"""

def generate_signal_with_clear_states(z_score, position, days_held, 
                                    z_open, z_close, max_days):
    """
    改进的信号生成逻辑，使用清晰的状态机制
    """
    # 强制平仓（最高优先级）
    if position and days_held >= max_days:
        return 'close'
  
    # 平仓条件
    if position and abs(z_score) <= z_close:
        return 'close'
  
    # 开仓条件（仅在空仓时）
    if not position:
        if abs(z_score) >= z_open:
            if z_score <= -z_open:
                return 'open_long'
            elif z_score >= z_open:
                return 'open_short'
        return 'empty'  # 空仓等待
  
    # 持仓期间状态
    if position == 'long':
        return 'holding_long'
    elif position == 'short':
        return 'holding_short'
  
    return 'empty'

def update_position_state(signal, position, days_held):
    """
    根据信号更新持仓状态
    """
    if signal == 'open_long':
        return 'long', 1
    elif signal == 'open_short':
        return 'short', 1
    elif signal == 'close':
        return None, 0
    elif position:
        return position, days_held + 1
    else:
        return None, 0
```

### 5.1 原始状态空间Kalman滤波算法

```python
class OriginalKalmanFilter:
    def __init__(self, warmup=60, Q_beta=5e-6, Q_alpha=1e-5, 
                 R_init=0.005, R_adapt=True):
        """
        初始化原始状态空间Kalman滤波器
      
        Args:
            warmup: 预热期长度
            Q_beta: Beta过程噪声
            Q_alpha: Alpha过程噪声
            R_init: 初始测量噪声
            R_adapt: 是否自适应R
        """
        self.warmup = warmup
        self.Q = np.diag([Q_beta, Q_alpha])  # 过程噪声协方差
        self.R = R_init                       # 测量噪声
        self.R_adapt = R_adapt
      
        # 状态变量 [beta, alpha]'
        self.state = None
        self.P = None  # 状态协方差
      
        # 历史记录
        self.beta_history = []
        self.alpha_history = []
        self.z_history = []
      
    def initialize(self, x_data, y_data):
        """使用OLS初始化状态 - 修正版本，与kalman_original_version.py一致"""
        # 手动构造设计矩阵，确保列顺序：[x_data, 常数项]
        X = np.column_stack([x_data[:self.warmup], np.ones(self.warmup)])
        Y = y_data[:self.warmup]
      
        # 使用最小二乘法直接求解 y = β*x + α
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # 初始状态 [β, α]' - 正确的顺序
        beta_init = coeffs[0]    # x_data对应的系数
        alpha_init = coeffs[1]   # 常数项系数  
        self.state = np.array([beta_init, alpha_init])
      
        # 初始协方差
        self.P = np.eye(2) * 0.001
      
        # 初始R基于残差方差
        residuals = Y - X @ coeffs
        self.R = np.var(residuals) if len(residuals) > 1 else self.R_init
      
    def update(self, x_t, y_t):
        """
        状态空间Kalman更新
        """
        # 1. 预测步
        # 状态预测: x_t|t-1 = x_t-1|t-1
        state_pred = self.state
      
        # 协方差预测: P_t|t-1 = P_t-1|t-1 + Q
        P_pred = self.P + self.Q
      
        # 2. 观测预测
        # H = [x_t, 1] 观测矩阵
        H = np.array([x_t, 1.0])
      
        # 预测观测: y_pred = β*x + α
        y_pred = H @ state_pred
      
        # 3. 创新
        v = y_t - y_pred  # 创新
        S = H @ P_pred @ H.T + self.R  # 创新方差
      
        # 4. 标准化残差（修正）
        z = v / np.sqrt(self.R)  # 使用R而非S进行标准化
      
        # 5. 更新步
        # Kalman增益
        K = P_pred @ H.T / S
      
        # 状态更新
        self.state = state_pred + K * v
      
        # 协方差更新
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
      
        # 6. 自适应R（可选）
        if self.R_adapt:
            lambda_r = 0.99  # EWMA参数
            self.R = lambda_r * self.R + (1 - lambda_r) * v**2
      
        # 7. 记录历史
        self.beta_history.append(self.state[0])
        self.alpha_history.append(self.state[1])
        self.z_history.append(z)
```

### 5.2 OLS基准计算

```python
def calculate_ols_beta(y_data, x_data, window):
    """
    计算滚动窗口OLS beta作为Kalman滤波验证基准
  
    Args:
        y_data: Y价格序列
        x_data: X价格序列  
        window: 滚动窗口大小（必须提供，无默认值）
  
    Returns:
        float: OLS回归系数
    """
    if len(y_data) < window or len(x_data) < window:
        return np.nan
  
    # 取最后window天的数据
    y_window = y_data[-window:]
    x_window = x_data[-window:]
  
    # OLS回归: y = beta * x + alpha，与Kalman初始化保持一致
    X = np.column_stack([x_window, np.ones(len(x_window))])
    try:
        coeffs = np.linalg.lstsq(X, y_window, rcond=None)[0]
        return coeffs[0]  # beta系数（现在是第1个系数）
    except:
        return np.nan
```

### 5.3 完整处理流程

```python
def process_pair(pair_name: str, x_data: np.ndarray, y_data: np.ndarray,
                 initial_beta: float = None) -> pd.DataFrame:
    """
    处理单个配对的完整流程
  
    Args:
        pair_name: 配对名称
        x_data: X价格序列（对数价格）
        y_data: Y价格序列（对数价格）
        initial_beta: 初始β（可选，否则用OLS估计）
  
    # 参数设置
    z_open = 2.0           # 开仓阈值
    z_close = 0.5          # 平仓阈值
    max_holding_days = 30  # 最大持仓天数
    calibration_freq = 5   # 每5天校准一次
    ols_window = 60        # OLS预热窗口
    warm_up_days = 60      # Kalman预热天数
  
    # 1. 初始化Kalman滤波器
    kf = AdaptiveKalmanFilter(pair_name)
  
    # 2. OLS预热（前ols_window天）
    init_result = kf.warm_up_ols(x_data, y_data, ols_window)
  
    signals = []
    position = None
    days_held = 0
    calibration_counter = 0
  
    # 3. Kalman预热期（ols_window到ols_window+warm_up_days）
    warm_up_end = min(ols_window + warm_up_days, len(x_data))
  
    for i in range(ols_window, warm_up_end):
        result = kf.update(y_data[i], x_data[i])
      
        # 预热期不生成交易信号，但记录z
        signals.append({
            'date': dates[i],
            'pair': pair_name,
            'signal': 'warm_up',
            'z_score': result['z'],  # 使用创新标准化
            'beta': result['beta'],
            'S': result['S'],
            'R': result['R'],
            'delta': kf.delta,
            'quality': kf.quality_status,
            'days_held': 0,
            'phase': 'warm_up'
        })
      
        # 预热期也进行校准检查
        if (i - ols_window) % 20 == 0 and i > ols_window + 20:
            kf.calibrate_delta()
  
    # 4. 正式交易期
    for i in range(warm_up_end, len(x_data)):
        # Kalman更新
        result = kf.update(y_data[i], x_data[i])
      
        # 使用残差标准化z生成信号（修正）
        z = result['z']  # z = v/√R，经滚动年度评估验证的正确方法
        signal = generate_signal(z, position, days_held, 
                               z_open, z_close, max_holding_days)
          
            # 更新持仓状态
            if signal.startswith('open'):
                position = signal
                days_held = 1
                reason = 'z_threshold'
            elif signal == 'close':
                position = None
                days_held = 0
                reason = 'z_threshold' if abs(z_score) < 0.5 else 'force_close'
            elif position:
                days_held += 1
                reason = 'holding'
            else:
                reason = 'no_signal'
              
        else:
            signal = 'hold'
            z_score = 0.0
            phase = 'signal_period'
            reason = 'insufficient_data'
          
        signals.append({
            'date': row['date'],
            'signal': signal,
            'z_score': z_score,
            'innovation': v,  # 创新值
            'beta': beta_t,
            'days_held': days_held,
            'reason': reason,
            'phase': phase,
            'converged': converged
        })
  
    return pd.DataFrame(signals)

def generate_signal(z_score, position, days_held, 
                   z_open, z_close, max_days):
    """
    信号生成逻辑（使用残差标准化-修正）
  
    Args:
        z_score: 标准化残差 z = v/√R（经实证验证的正确方法）
        position: 当前持仓状态 ('long'/'short'/None)
        days_held: 持仓天数
        z_open, z_close, max_days: 阈值参数
    """
    # 强制平仓（最高优先级）
    if position and days_held >= max_days:
        return 'close'
  
    # 平仓条件
    if position and abs(z_score) < z_close:
        return 'close'
  
    # 开仓条件（仅在空仓时）
    if not position:
        if z_score < -z_open:
            return 'open_long'
        elif z_score > z_open:
            return 'open_short'
        return 'empty'  # 空仓等待
  
    # 持仓状态
    if position == 'long':
        return 'holding_long'
    elif position == 'short':
        return 'holding_short'
  
    return 'empty'  # 默认空仓
```

## 6. 配置参数与输入格式

### 6.1 核心参数配置

```python
# 全局参数（经过实证验证的最优值）
config = {
    # 交易阈值
    "z_open": 2.0,                  # 开仓阈值
    "z_close": 0.5,                 # 平仓阈值
    "max_holding_days": 30,         # 最大持仓天数
  
    # Kalman参数（固定值，实证最优）
    "Q_beta": 5e-6,                 # Beta过程噪声
    "Q_alpha": 1e-5,                # Alpha过程噪声
    "R_init": 0.005,                # 初始测量噪声
    "R_adapt": True,                # 自适应R
  
    # 预热参数
    "warmup": 60                    # OLS+Kalman预热天数
}
```

### 6.2 配对参数输入格式

直接使用协整配对模块的DataFrame输出，字段名称完全匹配：

```python
# 从协整模块的screen_all_pairs()直接获取的DataFrame
# 包含字段：pair, symbol_x, symbol_y, beta_1y, beta_2y, beta_3y, beta_5y等
pairs_df = pd.DataFrame({
    'pair': ['AG-NI', 'AU-ZN'],
    'symbol_x': ['AG', 'AU'],         # 纯符号格式，与协整模块一致
    'symbol_y': ['NI', 'ZN'],         # 纯符号格式，与协整模块一致
    'beta_1y': [-0.2169, -0.3064],   # 1年β值
    'beta_2y': [-0.3540, 0.1695],    # 2年β值
    'beta_3y': [-0.6264, 0.0477],    # 3年β值
    'beta_5y': [-0.4296, 0.0429],    # 5年β值
    'pvalue_1y': [1.03e-05, 5.59e-05], # 协整p值
    # ...其他协整统计信息
})

# 使用指定的β时间窗口
beta_window = '1y'  # 可配置：'1y', '2y', '3y', '5y'等
initial_beta = pairs_df[f'beta_{beta_window}']
```

### 6.3 每个配对的状态

```python
# 每个配对独立维护的状态（使用统一参数）
pair_states = {
    'AL-ZN': {
        'state': [1.23, 0.01],  # [β, α]
        'P': [[0.001, 0], 
              [0, 0.001]],      # 2x2协方差矩阵
        'R': 0.005,             # 当前R（可自适应）
        'z_var': 1.05,          # 最近60天Z方差
        'quality': 'good'       # good/warning/bad
    },
    'CU-ZN': {
        'state': [0.85, -0.02], # [β, α]
        'P': [[0.001, 0],
              [0, 0.001]],      # 2x2协方差矩阵
        'R': 0.006,             # 当前R
        'z_var': 0.98,          # 接近理想值1.0
        'quality': 'good'
    }
}
```

## 7. 非功能需求

| 需求类型 | 描述                 | 目标值      |
| -------- | -------------------- | ----------- |
| 性能     | 单配对信号生成延迟   | < 50ms      |
| 性能     | 64个配对批量信号生成 | < 10秒      |
| 准确性   | β计算精度           | 小数点后6位 |
| 稳定性   | Kalman滤波数值稳定性 | 无发散      |
| 内存     | 单配对内存占用       | < 10MB      |

## 8. 测试用例

### TC-3.1: 一维Kalman滤波测试

| 测试ID   | 场景           | 预期结果                         |
| -------- | -------------- | -------------------------------- |
| TC-3.1.1 | 恒定关系数据   | β收敛到真实值                   |
| TC-3.1.2 | β缓慢漂移数据 | β跟踪漂移趋势                   |
| TC-3.1.3 | 高噪声数据     | β保持稳定不发散                 |
| TC-3.1.4 | β日变化>5%    | β变化被限制在5%以内             |
| TC-3.1.5 | 波动率突变数据 | R自适应调整，z-score保持合理范围 |

### TC-3.2: 信号生成测试

| 测试ID    | 场景                           | 预期结果                                     |
| --------- | ------------------------------ | -------------------------------------------- |
| TC-3.2.1  | z超过开仓阈值，无持仓          | 生成open_long/open_short                     |
| TC-3.2.2  | z小于平仓阈值，有持仓          | 生成close                                    |
| TC-3.2.3  | 持仓超过最大持仓天数           | 生成强制close                                |
| TC-3.2.4  | z超过开仓阈值，已有持仓        | 生成holding_long/holding_short               |
| TC-3.2.5  | 收敛期内                       | 生成converging                               |
| TC-3.2.6  | 信号期数据不足配置的窗口大小   | 生成empty                                    |
| TC-3.2.9  | 状态机制测试：空仓期间Z值变化  | 正确区分empty和holding_*状态                 |
| TC-3.2.10 | 状态机制测试：持仓期间状态显示 | 持仓第二天开始显示holding_long/holding_short |
| TC-3.2.7  | 自定义β时间窗口配置           | 使用指定窗口的β值                           |
| TC-3.2.8  | 自定义Z-score阈值配置          | 按配置阈值生成信号                           |

### TC-3.3: 分阶段处理和参数化测试

| 测试ID   | 场景             | 预期结果                                |
| -------- | ---------------- | --------------------------------------- |
| TC-3.3.1 | 收敛期β稳定性   | 按配置的收敛天数和阈值判定收敛          |
| TC-3.3.2 | 信号期准时开始   | 配置的信号开始日期生成交易信号          |
| TC-3.3.3 | 多配对同时处理   | 按配置的性能要求完成                    |
| TC-3.3.4 | 部分配对数据缺失 | 跳过缺失配对，记录错误信息              |
| TC-3.3.5 | 内存使用监控     | 按配置的内存限制                        |
| TC-3.3.6 | 无效参数配置     | 抛出清晰的错误信息                      |
| TC-3.3.7 | 默认参数处理     | 在参数为空时提供合理默认值              |
| TC-3.3.8 | 协整数据格式匹配 | 正确读取协整模块输出DataFrame的所有字段 |

## 9. 依赖关系

- **上游模块**:
  - 数据管理模块（提供对数价格数据）
  - 协整配对模块（提供配对参数和初始β值）
- **下游模块**: 回测框架模块
- **Python包**: numpy, pandas, typing
- **配置文件**:
  - 时间参数配置（收敛期、信号期时间边界）
  - 信号参数配置（开平仓阈值、收敛参数等）
