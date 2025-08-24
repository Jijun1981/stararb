# 模块3: 信号生成需求文档 V2.0 (极简自适应版)

## 1. 模块概述
接收协整配对模块的配对参数，使用**极简双旋钮Kalman滤波**进行动态β更新，基于标准化创新(z-score)生成交易信号。核心原则：**最少参数，最大适应性**。

**设计哲学**：
- 不追求完美参数，而是让参数自适应市场
- 只用两个控制变量：测量噪声R_t（EWMA自适应）和折扣因子δ（控制模型灵活性）
- 通过z方差带宽[0.8, 1.3]作为唯一的质量反馈信号

**职责边界**：
- 协整配对模块负责：提供初始β值（OLS估计）
- 本模块负责：自适应Kalman滤波，动态β更新，信号生成
- 回测模块负责：根据信号和资金计算具体手数

## 2. 用户故事 (User Stories)

### Story 3.1: 极简参数自适应
**作为**研究员  
**我希望**使用最少的参数实现自适应Kalman滤波  
**以便**避免过拟合和参数调优地狱

**验收标准:**
- 只调两个旋钮：R_t（EWMA自适应）和δ（折扣因子）
- 自动校准：通过z方差带宽反馈调整δ
- 无需网格搜索或复杂优化
- 每个配对独立适应，但用统一规则

### Story 3.2: OLS预热启动
**作为**研究员  
**我希望**使用OLS预热获得稳定的初始参数  
**以便**Kalman滤波有良好的起点

**验收标准:**
- 60日OLS窗口估计初始β和残差方差
- 用残差方差初始化R_0
- 设置合理的初始P_0

### Story 3.3: 工程化质量控制
**作为**研究员  
**我希望**有简单明确的质量判断标准  
**以便**快速判断参数是否合格

**验收标准:**
- 核心指标：z方差 ∈ [0.8, 1.3]
- 辅助验证：强信号有效性
- 边界保护：β ∈ [-4, 4]
- 异常处理：极端日"只预测不更新"

## 3. 功能需求 (Requirements)

### REQ-3.1: 极简Kalman滤波实现
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.1.1 | 状态方程：β_t = β_{t-1} + w_t，使用折扣因子实现 | P0 |
| REQ-3.1.2 | 观测方程：y_t = β_t * x_t + v_t | P0 |
| REQ-3.1.3 | R自适应：R_t = λ*R_{t-1} + (1-λ)*ε_t²，λ=0.96（日频） | P0 |
| REQ-3.1.4 | 折扣实现：P^- = P/δ，等价于Q=(1/δ-1)*P | P0 |
| REQ-3.1.5 | 初始δ：全局δ_β=0.98（可微调） | P0 |
| REQ-3.1.6 | 边界保护：β ∈ [-4, 4]，超出则限制 | P0 |
| REQ-3.1.7 | 异常日处理：换月/公告日"只预测不更新" | P1 |

### REQ-3.2: 参数自动校准
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.2.1 | 校准周期：每周或双周执行一次 | P0 |
| REQ-3.2.2 | 校准窗口：最近60根K线 | P0 |
| REQ-3.2.3 | z方差计算：v = Var(z_scores[-60:]) | P0 |
| REQ-3.2.4 | δ调整规则：v>1.3则δ-=0.01，v<0.8则δ+=0.01 | P0 |
| REQ-3.2.5 | δ边界：δ ∈ [0.95, 0.995] | P0 |
| REQ-3.2.6 | 调整步长：固定0.01，温和调整 | P0 |

### REQ-3.3: OLS预热初始化
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.3.1 | OLS窗口：60日（可配置） | P0 |
| REQ-3.3.2 | 初始β：OLS回归系数 | P0 |
| REQ-3.3.3 | 初始R：残差方差s² | P0 |
| REQ-3.3.4 | 初始P：diag(s², 0.1)（如果是2D）或s²（如果是1D） | P0 |
| REQ-3.3.5 | 预热期：60-120根K线，不交易 | P0 |
| REQ-3.3.6 | 预热验证：检查z方差是否在[0.8, 1.3]内 | P0 |

### REQ-3.4: 信号生成
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.4.1 | 开仓阈值：|z| > 2.0（全局统一） | P0 |
| REQ-3.4.2 | 平仓阈值：|z| < 0.5（全局统一） | P0 |
| REQ-3.4.3 | 风险配比：按√S_t调整仓位 | P0 |
| REQ-3.4.4 | 对冲比率：β仅用于计算腿数比例 | P0 |
| REQ-3.4.5 | 持仓限制：最大持仓30天（可配置） | P0 |
| REQ-3.4.6 | 信号类型：open_long, open_short, close, hold | P0 |

### REQ-3.5: 质量监控
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-3.5.1 | 红线1：最近60根z方差 ∈ [0.8, 1.3] | P0 |
| REQ-3.5.2 | 红线2：|z|>2的样本外均值回归收益为正 | P0 |
| REQ-3.5.3 | 实时监控：记录z_mean, z_std, r_ratio | P0 |
| REQ-3.5.4 | 警告机制：连续3次校准失败则告警 | P1 |

## 4. 接口定义

### 4.1 MinimalKalmanFilter类接口
```python
class MinimalKalmanFilter:
    def __init__(self, 
                 delta: float = 0.98,      # 折扣因子
                 lambda_r: float = 0.96,    # R的EWMA参数
                 beta_bounds: tuple = (-4, 4)):  # β边界
        """极简Kalman滤波器"""
        
    def warm_up_ols(self, x_data: np.ndarray, y_data: np.ndarray, 
                     window: int = 60) -> dict:
        """OLS预热初始化"""
        # 返回: {'beta': β_0, 'R': s², 'P': P_0}
        
    def update(self, y_t: float, x_t: float) -> dict:
        """单步更新"""
        # 返回: {'beta': β_t, 'z': z_t, 'S': S_t, 'v': v_t}
        
    def calibrate_delta(self, z_scores: np.ndarray) -> bool:
        """自动校准δ"""
        # 基于z方差调整δ，返回是否调整
        
    def get_quality_metrics(self, window: int = 60) -> dict:
        """获取质量指标"""
        # 返回: {'z_var': v, 'z_mean': μ, 'in_band': bool}
```

### 4.2 AdaptiveSignalGenerator类接口
```python
class AdaptiveSignalGenerator:
    def __init__(self,
                 z_open: float = 2.0,
                 z_close: float = 0.5,
                 max_holding_days: int = 30,
                 calibration_freq: int = 5):  # 每5天校准一次
        """自适应信号生成器"""
        
    def process_pair(self, 
                     x_data: pd.Series, 
                     y_data: pd.Series,
                     warm_up_days: int = 60,
                     min_signal_days: int = 60) -> pd.DataFrame:
        """处理单个配对"""
        # 返回信号DataFrame
        
    def generate_signal(self, z_score: float, position: str, 
                       days_held: int) -> str:
        """生成交易信号"""
        
    def risk_sizing(self, S_t: float, base_size: float) -> float:
        """基于S_t的风险配比"""
```

### 4.3 信号格式（简化版）
```python
{
    'date': '2025-04-10',
    'pair': 'AL-ZN',
    'signal': 'open_long',      # open_long, open_short, close, hold
    'z_score': -2.15,           # 标准化创新
    'beta': 0.8523,             # 当前β
    'S': 0.0012,                # 创新方差（用于风险配比）
    'delta': 0.98,              # 当前折扣因子
    'R': 0.0008,                # 当前测量噪声
    'quality': 'good',          # good(v∈[0.8,1.3]), warning, bad
    'days_held': 0              # 持仓天数
}
```

## 5. 算法细节

### 5.1 折扣Kalman滤波核心算法
```python
def discount_kalman_update(self, y_t, x_t):
    """折扣因子实现的Kalman更新"""
    # 1. 折扣先验协方差（等价于加入过程噪声）
    P_prior = self.P / self.delta  # 这是核心！
    
    # 2. 预测
    beta_pred = self.beta  # 随机游走
    y_pred = beta_pred * x_t
    
    # 3. 创新
    v = y_t - y_pred
    S = x_t * P_prior * x_t + self.R
    
    # 4. 更新
    K = P_prior * x_t / S
    self.beta = beta_pred + K * v
    
    # 5. 边界保护
    self.beta = np.clip(self.beta, self.beta_bounds[0], self.beta_bounds[1])
    
    # 6. 后验协方差
    self.P = (1 - K * x_t) * P_prior
    
    # 7. R自适应（EWMA）
    self.R = self.lambda_r * self.R + (1 - self.lambda_r) * (v ** 2)
    
    # 8. 记录z-score
    z = v / np.sqrt(S)
    self.z_history.append(z)
    
    return {'beta': self.beta, 'z': z, 'S': S, 'v': v, 'R': self.R}
```

### 5.2 自动校准律
```python
def calibrate_delta(self):
    """简易校准律 - 每周执行"""
    if len(self.z_history) < 60:
        return False
        
    # 计算最近60根的z方差
    recent_z = self.z_history[-60:]
    v = np.var(recent_z)
    
    # 校准规则（核心逻辑）
    if v > 1.3:  # 模型太慢，低估风险
        self.delta = max(0.95, self.delta - 0.01)
        self.calibration_log.append(f"v={v:.2f}>1.3, δ→{self.delta:.3f}")
        return True
        
    elif v < 0.8:  # 模型太快，过拟合
        self.delta = min(0.995, self.delta + 0.01)
        self.calibration_log.append(f"v={v:.2f}<0.8, δ→{self.delta:.3f}")
        return True
        
    else:  # 在目标带宽内，不调整
        return False
```

### 5.3 完整流程
```python
def run_adaptive_kalman(x_data, y_data):
    """完整的自适应Kalman流程"""
    
    # 1. OLS预热（60日）
    kf = MinimalKalmanFilter()
    init_params = kf.warm_up_ols(x_data[:60], y_data[:60])
    
    # 2. KF预热（60-120日，不交易）
    for i in range(60, 120):
        result = kf.update(y_data[i], x_data[i])
        
        # 每20天检查一次质量
        if i % 20 == 0:
            metrics = kf.get_quality_metrics()
            if not metrics['in_band']:
                kf.calibrate_delta(kf.z_history[-60:])
    
    # 3. 上线交易（120日后）
    signals = []
    position = None
    days_held = 0
    calibration_counter = 0
    
    for i in range(120, len(x_data)):
        # Kalman更新
        result = kf.update(y_data[i], x_data[i])
        
        # 生成信号
        signal = generate_signal(result['z'], position, days_held)
        
        # 风险配比
        risk_size = base_size / np.sqrt(result['S'])
        
        # 记录
        signals.append({
            'date': dates[i],
            'signal': signal,
            'z_score': result['z'],
            'beta': result['beta'],
            'S': result['S'],
            'risk_size': risk_size
        })
        
        # 更新持仓
        if signal.startswith('open'):
            position = signal.split('_')[1]
            days_held = 1
        elif signal == 'close':
            position = None
            days_held = 0
        elif position:
            days_held += 1
            
        # 定期校准（每周）
        calibration_counter += 1
        if calibration_counter >= 5:  # 5个交易日
            kf.calibrate_delta(kf.z_history[-60:])
            calibration_counter = 0
    
    return pd.DataFrame(signals)
```

## 6. 配置参数（极简版）

```python
# 全局固定参数（经验值，不需要调优）
GLOBAL_CONFIG = {
    # EWMA参数（根据频率选择）
    'lambda_daily': 0.96,      # 日频
    'lambda_30min': 0.985,     # 30分钟
    'lambda_1min': 0.995,      # 1分钟
    
    # 折扣因子初始值
    'delta_initial': 0.98,     # β的折扣因子
    'delta_bounds': (0.95, 0.995),  # δ的调整范围
    
    # 交易阈值（全局统一）
    'z_open': 2.0,
    'z_close': 0.5,
    
    # 质量带宽
    'z_var_band': (0.8, 1.3),
    
    # 边界保护
    'beta_bounds': (-4, 4),
    
    # 时间参数
    'ols_window': 60,
    'warm_up_days': 60,
    'calibration_freq': 5,  # 每5天校准一次
}

# 每个配对只需要存储的参数
PAIR_STATE = {
    'AL-ZN': {
        'delta': 0.98,      # 当前δ（会自适应调整）
        'R': 0.001,         # 当前R（EWMA更新）
        'beta': 1.23,       # 当前β
        'P': 0.001,         # 当前P
        'last_calibration': '2025-01-01'
    }
}
```

## 7. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 简洁性 | 核心参数数量 | ≤ 2个（R_t, δ） |
| 自适应 | 参数调整频率 | 每周1次 |
| 稳定性 | z方差带宽保持率 | > 90% |
| 性能 | 单配对更新延迟 | < 1ms |
| 内存 | 单配对状态存储 | < 1KB |

## 8. 测试用例

### TC-3.1: 自适应效果测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.1.1 | 平稳市场数据 | z方差稳定在[0.9, 1.1] |
| TC-3.1.2 | 波动率突变 | R自适应调整，z方差恢复到带宽内 |
| TC-3.1.3 | β漂移 | δ自动调整，跟踪β变化 |
| TC-3.1.4 | 极端值出现 | β限制在[-4, 4]内 |

### TC-3.2: 校准机制测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.2.1 | z方差>1.3持续2周 | δ减小0.02 |
| TC-3.2.2 | z方差<0.8持续2周 | δ增大0.02 |
| TC-3.2.3 | z方差在带宽内 | δ不变 |
| TC-3.2.4 | δ达到边界 | 停止调整，发出警告 |

### TC-3.3: 质量红线测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-3.3.1 | 60根z方差检查 | 准确计算并判断是否在带宽内 |
| TC-3.3.2 | 强信号回测 | |z|>2信号的收益统计正确 |
| TC-3.3.3 | 连续校准失败 | 3次失败后发出警告 |

## 9. 与原需求的主要变化

1. **删除了复杂的三阶段设计**：不需要收敛期，直接OLS预热后开始
2. **删除了所有硬编码的Kalman参数**：Q通过折扣因子自动生成，R自适应
3. **简化了参数配置**：从几十个参数减少到2个核心参数
4. **统一了交易阈值**：全局使用相同的z阈值，不需要每对单独配置
5. **去掉了复杂的收敛评估**：用z方差带宽作为唯一质量指标
6. **简化了信号类型**：只保留核心的4种信号

## 10. 实施建议

1. **先实现核心算法**：折扣Kalman + EWMA的R
2. **再加入自动校准**：基于z方差的δ调整
3. **最后加入边界保护**：异常日处理、β限制等
4. **持续监控质量**：重点关注z方差带宽指标

这个极简方案的核心优势：
- **可解释性强**：每个参数都有明确的物理意义
- **避免过拟合**：不需要历史数据调参
- **工程友好**：实现简单，调试方便
- **自适应强**：市场变化时参数自动调整