# 自适应Kalman滤波器使用指南

## 1. 核心类介绍

### AdaptiveKalmanFilter
自适应Kalman滤波器，用于动态估计β（对冲比率）

### AdaptiveSignalGenerator  
信号生成器，管理多个配对的信号生成流程

## 2. 使用流程

### 2.1 单个配对的使用流程

```python
import numpy as np
import pandas as pd
from lib.signal_generation import AdaptiveKalmanFilter, AdaptiveSignalGenerator

# 步骤1: 准备数据（对数价格）
x_data = np.log(x_prices)  # X品种的对数价格
y_data = np.log(y_prices)  # Y品种的对数价格

# 步骤2: 创建Kalman滤波器
kf = AdaptiveKalmanFilter(
    pair_name="CU-ZN",
    delta=0.92,           # 折扣因子（越小跟踪越快）
    lambda_r=0.96,        # EWMA参数（测量噪声自适应）
    beta_bounds=(-4, 4),  # β的上下界
    z_var_band=(0.8, 1.3) # z方差目标带宽
)

# 步骤3: OLS预热（必须！）
# 使用前60个数据点进行OLS回归，获取初始参数
init_result = kf.warm_up_ols(
    x_data=x_data,      # numpy数组
    y_data=y_data,      # numpy数组  
    window=60           # OLS窗口大小
)

# 预热返回的结果：
print(f"初始β: {init_result['beta']}")      # OLS估计的β
print(f"初始截距: {init_result['c']}")      # OLS截距（未使用）
print(f"初始R: {init_result['R']}")         # 残差方差
print(f"初始P: {init_result['P']}")         # 初始不确定性

# 步骤4: Kalman滤波更新（从第61个点开始）
for i in range(60, len(x_data)):
    # 单步更新
    result = kf.update(
        y_t=y_data[i],   # 当前Y值
        x_t=x_data[i]    # 当前X值
    )
    
    # 更新结果包含：
    # result['beta']   - 更新后的β
    # result['z']      - 标准化创新 (z-score)
    # result['v']      - 创新值 (y - β*x)
    # result['S']      - 创新方差
    # result['R']      - 当前测量噪声
    
    # 每5步校准一次δ（可选）
    if (i - 60) % 5 == 0 and i > 120:
        kf.calibrate_delta()

# 步骤5: 获取质量指标
metrics = kf.get_quality_metrics()
print(f"z方差: {metrics['z_var']}")         # 应在[0.8, 1.3]范围内
print(f"质量状态: {metrics['quality_status']}") # good/warning/bad
```

### 2.2 使用信号生成器（推荐方式）

```python
from lib.signal_generation import AdaptiveSignalGenerator

# 步骤1: 创建信号生成器
sg = AdaptiveSignalGenerator(
    z_open=2.0,              # 开仓阈值
    z_close=0.5,             # 平仓阈值  
    max_holding_days=30,     # 最大持仓天数
    calibration_freq=5,      # 校准频率（天）
    ols_window=60,           # OLS预热窗口
    warm_up_days=60          # Kalman预热天数
)

# 步骤2: 准备数据（带日期索引的Series）
dates = pd.date_range('2020-01-01', periods=500)
x_series = pd.Series(np.log(x_prices), index=dates)
y_series = pd.Series(np.log(y_prices), index=dates)

# 步骤3: 处理单个配对
signals_df = sg.process_pair(
    pair_name="CU-ZN",
    x_data=x_series,         # 带索引的Series
    y_data=y_series,         # 带索引的Series
    initial_beta=1.2         # 可选：指定初始β（否则用OLS估计）
)

# 返回的DataFrame包含：
# - date: 日期
# - pair: 配对名称
# - signal: 信号类型 (warm_up/open_long/open_short/close/hold)
# - z_score: 标准化创新
# - beta: 当前β值
# - S: 创新方差
# - R: 测量噪声
# - delta: 当前折扣因子
# - quality: 质量状态
# - days_held: 持仓天数
# - phase: 阶段 (warm_up/trading)
```

### 2.3 批量处理多个配对

```python
# 步骤1: 准备配对信息（来自协整模块）
pairs_df = pd.DataFrame({
    'pair': ['AL-ZN', 'CU-ZN', 'RB-HC'],
    'symbol_x': ['AL', 'CU', 'RB'],
    'symbol_y': ['ZN', 'ZN', 'HC'],
    'beta_1y': [1.2, 0.8, 1.5],    # 1年期β
    'beta_2y': [1.3, 0.9, 1.6],    # 2年期β
    'beta_3y': [1.4, 1.0, 1.7],    # 3年期β
})

# 步骤2: 准备价格数据（所有品种）
price_data = pd.DataFrame(index=dates)
price_data['AL'] = np.log(al_prices)
price_data['CU'] = np.log(cu_prices)
price_data['RB'] = np.log(rb_prices)
price_data['ZN'] = np.log(zn_prices)
price_data['HC'] = np.log(hc_prices)

# 步骤3: 批量处理
all_signals = sg.process_all_pairs(
    pairs_df=pairs_df,
    price_data=price_data,
    beta_window='1y'         # 使用哪个时间窗口的β作为初始值
)

# 步骤4: 获取质量报告
quality_report = sg.get_quality_report()
print(quality_report)
# 包含每个配对的：z_var, quality, delta, R, beta等
```

## 3. 关键参数说明

### 3.1 OLS预热参数
- **window**: OLS回归窗口，默认60天
- **作用**: 
  - 估计初始β（如果没有提供）
  - 估计初始R（测量噪声）
  - 计算初始P（不确定性）

### 3.2 Kalman滤波参数
- **delta** (δ): 折扣因子，控制β的变化速度
  - 范围: [0.90, 0.995]
  - 越小: β跟踪越快，但可能过拟合
  - 越大: β更稳定，但反应较慢
  - 自动校准: 根据z方差调整

- **lambda_r** (λ): EWMA参数，控制R的自适应速度
  - 固定值: 0.96（日频数据）
  - R更新: R_new = λ*R_old + (1-λ)*v²

- **beta_bounds**: β的边界保护
  - 默认: [-4, 4]
  - 防止β发散

- **z_var_band**: z方差目标带宽
  - 目标: [0.8, 1.3]
  - 用于自动校准δ

### 3.3 信号生成参数
- **z_open**: 开仓阈值（默认2.0）
- **z_close**: 平仓阈值（默认0.5）
- **max_holding_days**: 最大持仓天数（默认30）

## 4. 完整示例：从协整结果到信号生成

```python
# 假设已经从协整模块获得结果
from lib.coint import screen_all_pairs
from lib.data import load_all_symbols_data
from lib.signal_generation import AdaptiveSignalGenerator

# 1. 获取数据
price_data = load_all_symbols_data()

# 2. 协整筛选
coint_results = screen_all_pairs(price_data)

# 3. 生成信号
sg = AdaptiveSignalGenerator()

# 对每个通过协整检验的配对
for _, pair_info in coint_results.iterrows():
    pair_name = pair_info['pair']
    symbol_x = pair_info['symbol_x']
    symbol_y = pair_info['symbol_y']
    
    # 获取价格数据
    x_data = price_data[symbol_x]
    y_data = price_data[symbol_y]
    
    # 使用协整模块提供的β作为初始值
    initial_beta = pair_info['beta_1y']  # 使用1年期β
    
    # 生成信号
    signals = sg.process_pair(
        pair_name=pair_name,
        x_data=x_data,
        y_data=y_data,
        initial_beta=initial_beta
    )
    
    print(f"{pair_name}: 生成{len(signals)}个信号")
```

## 5. 预热阶段详解

### 5.1 为什么需要预热？
1. **OLS预热（前60天）**: 估计初始参数
2. **Kalman预热（60-120天）**: 让滤波器收敛

### 5.2 预热流程
```python
# 数据分段
# [0:60]     - OLS预热，估计初始β、R、P
# [60:120]   - Kalman预热，不生成交易信号
# [120:]     - 正式交易期，生成信号

kf = AdaptiveKalmanFilter("TEST")

# OLS预热（必须）
kf.warm_up_ols(x_data[:60], y_data[:60], window=60)

# Kalman预热期
for i in range(60, 120):
    kf.update(y_data[i], x_data[i])
    # 预热期也进行校准
    if (i - 60) % 20 == 0:
        kf.calibrate_delta()

# 正式交易期
for i in range(120, len(data)):
    result = kf.update(y_data[i], x_data[i])
    z_score = result['z']
    # 生成交易信号...
```

## 6. 质量监控

### 6.1 红线检查
```python
red_lines = kf.check_red_lines()
# red_line_1: z方差是否在[0.8, 1.3]范围内
# red_line_2: 需要回测验证（未实现）
```

### 6.2 诊断指标
```python
metrics = kf.get_quality_metrics(window=60)
# z_var: z方差（目标0.8-1.3）
# z_mean: z均值（应接近0）
# quality_status: good/warning/bad
# extreme_3_ratio: |z|>3的比例
# extreme_4_ratio: |z|>4的比例
```

## 7. 注意事项

1. **数据要求**：
   - 必须是对数价格
   - 至少需要120个数据点（60 OLS + 60预热）

2. **参数调优**：
   - 不同配对可能需要不同的初始δ
   - 自动校准会逐步优化参数

3. **性能考虑**：
   - 批量处理多个配对时，每个配对独立维护状态
   - 校准频率不要太高（建议5-20天）

4. **与协整模块配合**：
   - 使用协整模块提供的初始β
   - 可以选择不同时间窗口的β（1y/2y/3y/5y）