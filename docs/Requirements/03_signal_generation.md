# 模块3: 信号生成需求文档 V2.0

## 1. 模块概述
接收协整配对模块的配对参数，使用**自适应双旋钮Kalman滤波**进行动态β更新，基于标准化创新(z-score)生成交易信号。核心设计：每个配对独立自适应，使用统一规则。

**设计理念**：
- 只用两个控制变量：测量噪声R_t（EWMA自适应）和折扣因子δ（控制模型灵活性）
- 通过z方差带宽[0.8, 1.3]作为质量反馈信号自动调整参数
- 每个配对独立维护自己的参数状态，根据自身特性自适应

**职责边界**：
- 协整配对模块负责：提供多个时间窗口的初始β值（1y, 2y, 3y, 5y等）
- 本模块负责：OLS预热初始化，自适应Kalman滤波，批量生成交易信号
- 回测模块负责：根据信号和资金计算具体手数

## 2. 用户故事 (User Stories)

### Story 3.1: 自适应β估计
**作为**研究员  
**我希望**使用自适应Kalman滤波动态估计对冲比率  
**以便**每个配对能根据自身特性找到合适的参数

**验收标准:**
- OLS预热获得稳定初始参数
- 每个配对独立维护R_t和δ参数
- 通过z方差带宽自动校准
- 支持批量处理多个配对
- 记录完整β历史和参数调整日志

### Story 3.2: 创新标准化
**作为**研究员  
**我希望**计算标准化创新(z-score)  
**以便**生成稳定的交易信号

**验收标准:**
- 计算创新：v = y - β*x
- 计算创新方差：S = x²*P + R
- 计算标准化创新：z = v/√S（不使用滚动窗口）
- z方差保持在[0.8, 1.3]带宽内
- **重要**：z必须是创新标准化，不是残差滚动窗口标准化

### Story 3.3: 基于创新标准化的信号生成
**作为**研究员  
**我希望**基于创新标准化z生成开平仓信号  
**以便**捕捉均值回归机会

**验收标准:**
- 使用创新标准化：z = v/√S，其中v = y - β*x，S = x²*P + R
- 设定开仓阈值：|z| > 2.0
- 设定平仓阈值：|z| < 0.5
- 生成明确的交易信号状态

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

### REQ-3.1: 自适应Kalman滤波
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.1.1 | 状态方程：β_t = β_{t-1} + w_t，使用折扣因子实现 | P0 |
| REQ-3.1.2 | 观测方程：y_t = β_t * x_t + v_t | P0 |
| REQ-3.1.3 | OLS预热：60日窗口估计初始β和残差方差 | P0 |
| REQ-3.1.4 | R自适应：R_t = λ*R_{t-1} + (1-λ)*ε_t²，λ=0.96（日频） | P0 |
| REQ-3.1.5 | 折扣实现：P^- = P/δ（只暴露δ，不配置Q） | P0 |
| REQ-3.1.6 | 初始δ：全局δ_β=0.98，每个配对可独立调整 | P0 |
| REQ-3.1.7 | 边界保护：β ∈ [-4, 4]，超出则限制 | P0 |
| REQ-3.1.8 | 异常日处理：换月/公告日"只预测不更新" | P1 |
| REQ-3.1.9 | 参数状态维护：每个配对独立维护δ、R、β、P | P0 |
| REQ-3.1.10 | 预热期：60-120根K线，不交易但检查质量 | P0 |

### REQ-3.2: 参数自动校准
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.2.1 | 校准周期：每周或双周执行一次 | P0 |
| REQ-3.2.2 | 校准窗口：最近60根K线 | P0 |
| REQ-3.2.3 | z方差计算：v = Var(z_scores[-60:])，z为创新标准化 | P0 |
| REQ-3.2.4 | δ调整规则：v>1.3则δ-=0.01，v<0.8则δ+=0.01 | P0 |
| REQ-3.2.5 | δ边界：δ ∈ [0.95, 0.995] | P0 |
| REQ-3.2.6 | 调整步长：固定0.01，温和调整 | P0 |
| REQ-3.2.7 | 校准日志：记录每次校准的时间、原因、调整结果 | P0 |

### REQ-3.3: 信号生成逻辑
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.3.1 | 开仓阈值：|z| > 2.0，z为创新标准化 | P0 |
| REQ-3.3.2 | 平仓阈值：|z| < 0.5，z为创新标准化 | P0 |
| REQ-3.3.3 | 持仓限制：最大持仓30天（可配置） | P0 |
| REQ-3.3.4 | 信号类型：open_long, open_short, holding_long, holding_short, close, empty | P0 |
| REQ-3.3.5 | 防重复开仓：同配对同方向不重复开仓 | P0 |
| REQ-3.3.6 | 信号优先级：强制平仓 > 平仓 > 开仓 | P0 |

### REQ-3.4: 批量配对处理
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.4.1 | 接收协整模块DataFrame：包含pair, symbol_x, symbol_y, beta_1y等 | P0 |
| REQ-3.4.2 | 支持选择β时间窗口：'1y', '2y', '3y', '5y'等 | P0 |
| REQ-3.4.3 | 每个配对独立处理：独立维护状态和参数 | P0 |
| REQ-3.4.4 | 输出统一格式信号DataFrame | P0 |
| REQ-3.4.5 | 记录配对级别的β历史和参数调整日志 | P0 |
| REQ-3.4.6 | 生成质量报告：z方差、校准次数、参数状态 | P1 |

### REQ-3.5: 质量监控
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.5.1 | 核心指标：最近60根z方差 ∈ [0.8, 1.3] | P0 |
| REQ-3.5.2 | 实时监控：记录z_mean, z_std, 当前δ和R | P0 |
| REQ-3.5.3 | 配对级别质量评级：good(z方差在带宽内)/warning/bad | P0 |
| REQ-3.5.4 | 支持导出质量报告 | P1 |

## 4. 接口定义

### 4.1 AdaptiveSignalGenerator类接口
```python
class AdaptiveSignalGenerator:
    def __init__(self, 
                 # 交易阈值 - 全局统一
                 z_open: float = 2.0, 
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
                 
                 # 校准参数
                 calibration_freq: int = 5,  # 每5天校准一次
                 
                 # OLS预热参数
                 ols_window: int = 60,
                 warm_up_days: int = 60)
    
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

### 4.2 AdaptiveKalmanFilter类接口
```python
class AdaptiveKalmanFilter:
    def __init__(self, 
                 pair_name: str,
                 delta: float = 0.98,      # 折扣因子初始值
                 lambda_r: float = 0.96):   # R的EWMA参数
    
    # OLS预热
    def warm_up_ols(self, x_data: np.ndarray, y_data: np.ndarray, 
                     window: int = 60) -> dict
    
    # 单步更新
    def update(self, y_t: float, x_t: float) -> dict
    
    # 自动校准
    def calibrate_delta(self) -> bool
    
    # 获取质量指标
    def get_quality_metrics(self, window: int = 60) -> dict
```

### 4.3 信号格式
```python
{
    'date': '2025-04-10',           # 无硬编码日期
    'pair': 'AG-NI',               # 与协整模块格式一致：纯符号，无后缀
    'symbol_x': 'AG',              # X品种（低波动）
    'symbol_y': 'NI',              # Y品种（高波动）
    'signal': 'open_long',          # open_long, open_short, holding_long, holding_short, close, empty
    'z_score': -2.15,               # 创新标准化 z = v/√S，其中S = x²*P + R
    'innovation': -0.0234,          # 当前创新值 v = y - β*x
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

### 5.1 折扣Kalman滤波算法
```python
class AdaptiveKalmanFilter:
    def __init__(self, pair_name, delta=0.98, lambda_r=0.96, 
                 beta_bounds=(-4, 4)):
        """
        初始化自适应Kalman滤波器
        
        Args:
            pair_name: 配对名称
            delta: 折扣因子（默认0.98）
            lambda_r: R的EWMA参数（默认0.96，日频）
            beta_bounds: β边界限制
        """
        self.pair_name = pair_name
        self.delta = delta
        self.lambda_r = lambda_r
        self.beta_bounds = beta_bounds
        
        # 状态变量（通过OLS预热初始化）
        self.beta = None
        self.P = None
        self.R = None
        
        # 历史记录
        self.z_history = []
        self.beta_history = []
        
    def warm_up_ols(self, x_data, y_data, window=60):
        """OLS预热获得初始参数（使用去中心化数据）"""
        # 去中心化处理（关键：避免截距问题导致R膨胀）
        mu_x = np.mean(x_data[:window])
        mu_y = np.mean(y_data[:window])
        x_use = x_data[:window] - mu_x
        y_use = y_data[:window] - mu_y
        
        # OLS回归估计初始β（基于去中心化数据）
        reg = LinearRegression(fit_intercept=False)  # 不需要截距
        reg.fit(x_use.reshape(-1, 1), y_use)
        
        self.beta = reg.coef_[0]
        innovations = y_use - reg.predict(x_use.reshape(-1, 1))
        self.R = np.var(innovations, ddof=1)  # 初始R为创新方差
        
        # P0初始化：使用Var(x)标定，不再乘0.1
        x_var = np.var(x_use, ddof=1)
        self.P = self.R / max(x_var, 1e-12)  # 让x²*P与R同量级
        
        # 保存均值用于后续去中心化
        self.mu_x = mu_x
        self.mu_y = mu_y
        
        return {'beta': self.beta, 'R': self.R, 'P': self.P, 
                'mu_x': mu_x, 'mu_y': mu_y}
        
    def update(self, y_t, x_t):
        """
        折扣Kalman更新（核心算法）
        
        Returns:
            dict: 包含β、标准化创新z等
        """
        # 1. 折扣先验协方差（等价于加入过程噪声）
        P_prior = self.P / self.delta
        
        # 2. 预测
        beta_pred = self.beta  # 随机游走
        y_pred = beta_pred * x_t
        
        # 3. 创新
        v = y_t - y_pred
        S = x_t * P_prior * x_t + self.R
        S = max(S, 1e-12)  # 数值稳定性
        
        # 4. 创新标准化 z = v/√S（关键：不是滚动窗口标准化）
        z = v / np.sqrt(S)
        
        # 5. Kalman增益
        K = P_prior * x_t / S
        
        # 6. 状态更新
        beta_new = beta_pred + K * v
        
        # 7. β边界保护
        beta_new = np.clip(beta_new, self.beta_bounds[0], self.beta_bounds[1])
        
        # 8. 后验协方差
        self.P = (1 - K * x_t) * P_prior
        
        # 9. R自适应（EWMA，λ=0.96）
        self.R = self.lambda_r * self.R + (1 - self.lambda_r) * (v ** 2)
        
        # 10. 更新状态
        self.beta = beta_new
        self.z_history.append(z)
        self.beta_history.append(self.beta)
        
        return {
            'beta': self.beta,
            'v': v,      # 创新 v = y - β*x
            'S': S,      # 创新方差 S = x²*P + R
            'z': z,      # 创新标准化 z = v/√S
            'R': self.R,
            'K': K
        }
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
    
    # OLS回归: y = alpha + beta * x
    X = np.column_stack([np.ones(len(x_window)), x_window])
    try:
        coeffs = np.linalg.lstsq(X, y_window, rcond=None)[0]
        return coeffs[1]  # beta系数
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
        
        # 使用创新标准化z生成信号
        z = result['z']  # z = v/√S，不使用滚动窗口
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
    信号生成逻辑（使用创新标准化）
    
    Args:
        z_score: 标准化创新 z = v/√S
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
# 全局参数（经验值）
config = {
    # 交易阈值
    "z_open": 2.0,                  # 开仓阈值
    "z_close": 0.5,                 # 平仓阈值
    "max_holding_days": 30,         # 最大持仓天数
    
    # 校准参数
    "calibration_freq": 5,          # 每5天校准一次
    "delta_step": 0.01,             # δ调整步长
    
    # 预热参数
    "ols_window": 60,               # OLS预热窗口
    "warm_up_days": 60              # Kalman预热天数
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
# 每个配对独立维护的参数
pair_states = {
    'AL-ZN': {
        'delta': 0.98,      # 当前δ（会自适应调整）
        'R': 0.001,         # 当前R（EWMA更新）
        'beta': 1.23,       # 当前β
        'P': 0.001,         # 当前P
        'last_calibration': '2025-01-01',
        'quality': 'good'   # good/warning/bad
    },
    'CU-ZN': {
        'delta': 0.97,      # CU-ZN可能需要更灵活
        'R': 0.002,         # 噪声更大
        'beta': 0.85,       
        'P': 0.002,         
        'last_calibration': '2025-01-01',
        'quality': 'good'
    }
}
```

## 7. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 性能 | 单配对信号生成延迟 | < 50ms |
| 性能 | 64个配对批量信号生成 | < 10秒 |
| 准确性 | β计算精度 | 小数点后6位 |
| 稳定性 | Kalman滤波数值稳定性 | 无发散 |
| 内存 | 单配对内存占用 | < 10MB |

## 8. 测试用例

### TC-3.1: 一维Kalman滤波测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.1.1 | 恒定关系数据 | β收敛到真实值 |
| TC-3.1.2 | β缓慢漂移数据 | β跟踪漂移趋势 |
| TC-3.1.3 | 高噪声数据 | β保持稳定不发散 |
| TC-3.1.4 | β日变化>5% | β变化被限制在5%以内 |
| TC-3.1.5 | 波动率突变数据 | R自适应调整，z-score保持合理范围 |

### TC-3.2: 信号生成测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.2.1 | z超过开仓阈值，无持仓 | 生成open_long/open_short |
| TC-3.2.2 | z小于平仓阈值，有持仓 | 生成close |
| TC-3.2.3 | 持仓超过最大持仓天数 | 生成强制close |
| TC-3.2.4 | z超过开仓阈值，已有持仓 | 生成holding_long/holding_short |
| TC-3.2.5 | 收敛期内 | 生成converging |
| TC-3.2.6 | 信号期数据不足配置的窗口大小 | 生成empty |
| TC-3.2.9 | 状态机制测试：空仓期间Z值变化 | 正确区分empty和holding_*状态 |
| TC-3.2.10 | 状态机制测试：持仓期间状态显示 | 持仓第二天开始显示holding_long/holding_short |
| TC-3.2.7 | 自定义β时间窗口配置 | 使用指定窗口的β值 |
| TC-3.2.8 | 自定义Z-score阈值配置 | 按配置阈值生成信号 |

### TC-3.3: 分阶段处理和参数化测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.3.1 | 收敛期β稳定性 | 按配置的收敛天数和阈值判定收敛 |
| TC-3.3.2 | 信号期准时开始 | 配置的信号开始日期生成交易信号 |
| TC-3.3.3 | 多配对同时处理 | 按配置的性能要求完成 |
| TC-3.3.4 | 部分配对数据缺失 | 跳过缺失配对，记录错误信息 |
| TC-3.3.5 | 内存使用监控 | 按配置的内存限制 |
| TC-3.3.6 | 无效参数配置 | 抛出清晰的错误信息 |
| TC-3.3.7 | 默认参数处理 | 在参数为空时提供合理默认值 |
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