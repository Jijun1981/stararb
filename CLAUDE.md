# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述
这是一个基于协整理论的金属期货配对交易量化研究平台，采用模块化架构设计，支持多种对冲算法和完整的回测分析。系统使用Python实现，主要分析14个金属期货品种的配对交易机会。

**重要说明**：本项目必须严格按照需求文档实现，禁止任何简化、降级或fallback方案。所有功能都必须基于真实数据和严格的数学模型实现。

## 常用命令

### 数据获取和更新
```bash
# 使用AkShare获取期货数据（从notebooks运行）
cd notebooks
jupyter notebook 01_data_management.ipynb

# Python直接运行数据更新
python -c "from lib.data import load_all_symbols_data; df = load_all_symbols_data()"
```

### 运行完整流程
```bash
# 运行V2.1版本的完整流程（推荐）
python scripts/pipeline/run_complete_pipeline_v2_1.py

# 运行V2版本流程
python scripts/pipeline/run_complete_pipeline_v2.py

# 运行V1版本流程（兼容）
python scripts/pipeline/run_complete_pipeline.py
```

### 协整分析
```bash
# 运行协整分析notebook
cd notebooks
jupyter notebook 02_cointegration_complete.ipynb

# 直接运行协整检验
python -c "from lib.coint import run_cointegration_analysis; results = run_cointegration_analysis()"
```

### 策略回测
```bash
# 运行策略演示notebook
cd notebooks
jupyter notebook 04_strategy_demo.ipynb

# 运行完整回测分析
jupyter notebook 05_backtest_analysis.ipynb
```

### 测试运行
```bash
# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 运行所有测试
pytest tests/

# 手动运行Python语法检查
python -m py_compile lib/*.py scripts/*/*.py
```

## 代码架构

### 核心模块结构 (V2.1)
```
lib/                          # 核心函数库
├── __init__.py              # 模块导入配置
├── data.py                  # 数据管理模块 - AkShare数据获取、Parquet存储
├── coint.py                 # 协整配对模块 - Engle-Granger检验、配对筛选、Beta估计
├── signal_generation.py    # 信号生成模块 - Kalman滤波、动态beta估计、Z-score信号
└── backtest.py              # 回测引擎模块 - 交易执行模拟、绩效分析
```

### 系统架构（4模块设计）
```
数据管理模块 → 协整配对模块 → 信号生成模块 → 回测框架模块
     ↓              ↓              ↓              ↓
AkShare获取    Engle-Granger    Kalman滤波    真实交易模拟
Parquet存储    Beta估计(OLS)    双Z-score     12%保证金管理
```

### 模块依赖关系
1. **数据管理模块** (lib/data.py): AkShare API → Parquet存储 → 统一数据接口
2. **协整配对模块** (lib/coint.py): 协整检验 → Beta估计 → 配对筛选
3. **信号生成模块** (lib/signal_generation.py): Kalman滤波 → 动态β估计 → 双Z-score信号
4. **回测框架模块** (lib/backtest.py): 交易执行 → 风险管理 → 绩效分析

### 关键设计原则
- **严格按需求实现**: 绝对禁止简化、降级或fallback，必须完整实现所有功能
- **真实数据驱动**: 所有分析基于AkShare真实数据，API失败时直接报错
- **原子操作**: 数据更新使用临时文件和备份机制确保完整性
- **时间序列处理**: 统一的日期索引对齐和缺失值处理
- **模块化设计**: 每个模块专注单一职责，接口清晰

## 开发指南

### 核心业务参数

#### 期货品种配置（14个）
- **贵金属**: AG0(银), AU0(金)  
- **有色金属**: AL0(铝), CU0(铜), NI0(镍), PB0(铅), SN0(锡), ZN0(锌)
- **黑色系**: HC0(热卷), I0(铁矿), RB0(螺纹), SF0(硅铁), SM0(锰硅), SS0(不锈钢)

#### 数据参数
- **数据范围**: 2020-01-01至今（用于分析）
- **数据频率**: 日线数据
- **数据源**: AkShare API的futures_zh_daily_sina接口
- **存储格式**: Parquet格式，支持增量更新

#### 协整检验参数
- **检验方法**: Engle-Granger两步法
- **时间窗口**: 5年、4年、3年、2年、1年多窗口验证
- **p值阈值**: 0.05（主要依据5年p值）
- **方向判定**: 基于2024年至今的波动率，低波动作X，高波动作Y

#### Beta估计参数
- **估计方法**: OLS（普通最小二乘法）
- **估计窗口**: 使用全部可用历史数据
- **更新频率**: 初始化时计算，之后保持固定
- **精度要求**: 6位小数

#### 信号生成参数
- **Z-score模式**: 创新z-score（主）+ 滚动z-score（验证）
- **开仓阈值**: |Z| > 2.0
- **平仓阈值**: |Z| < 0.5
- **滚动窗口**: 60个交易日
- **最大持仓**: 30天强制平仓

#### 回测参数
- **初始资金**: 500万元
- **保证金率**: 12%
- **交易费率**: 万分之2（双边）
- **滑点设置**: 每腿3个tick
- **止损条件**: 保证金的10%
- **仓位权重**: 每配对约5%资金（可配置）

### 关键技术要求

#### 数据质量控制
- 缺失值处理: 前向填充(ffill)
- 异常值检测: 5倍标准差阈值
- 数据验证: 每次更新后完整性验证
- 增量更新: 使用原子操作和备份机制

#### 算法精度要求
- β估计精度: 6位小数
- p值计算: 6位小数
- PnL计算: 精确到分
- 时间序列对齐: 统一交易日历

#### 性能要求
- 数据加载(5年14品种): < 5秒
- 协整检验(91配对): < 30秒
- 信号生成(70配对): < 10秒
- 完整回测(2年): < 60秒
- 内存占用峰值: < 4GB

### 工作流程指南

#### 主要执行脚本
1. **完整流程脚本** (scripts/pipeline/):
   - `run_complete_pipeline_v2_1.py`: V2.1版本，最新优化版本
   - `run_complete_pipeline_v2.py`: V2版本，稳定版本
   - `run_complete_pipeline.py`: V1版本，兼容版本

2. **分析脚本** (scripts/analysis/):
   - 协整分析、Beta对比、算法验证等

3. **回测脚本** (scripts/backtest/):
   - 各种策略的回测执行脚本

4. **报告生成** (scripts/reports/):
   - 生成详细分析报告和交易报告

#### Jupyter Notebook分析流程
1. **01_data_management.ipynb**: 
   - 使用AkShare获取14个品种历史数据
   - Parquet格式存储和增量更新
   - 数据质量检查和统计报告

2. **02_cointegration_complete.ipynb**: 
   - 多时间窗口协整检验（5年-1年）
   - OLS估计Beta系数
   - 生成配对分析报告

3. **04_strategy_demo.ipynb**: 
   - Kalman滤波动态β估计演示
   - 双Z-score信号生成逻辑
   - 单配对策略效果展示

4. **05_backtest_analysis.ipynb**: 
   - 完整回测框架运行
   - 真实交易模拟和绩效分析
   - 多配对组合效果评估

#### 核心依赖包
```python
# 数据处理
pandas >= 1.5.0
numpy >= 1.23.0
pyarrow >= 10.0.0

# 数据获取  
akshare >= 1.10.0

# 统计分析
statsmodels >= 0.13.0
scipy >= 1.9.0
scikit-learn >= 1.1.0
linearmodels >= 4.25  # FM-OLS回归

# 可视化
matplotlib >= 3.6.0
plotly >= 5.11.0

# 工具
pyyaml >= 6.0
```

### 严格执行原则

#### 数据处理
- **绝对禁止模拟数据**: 所有数据必须来自AkShare真实API
- **API失败处理**: 获取失败时直接抛出异常，不使用任何fallback
- **数据完整性**: 每次更新后验证数据质量，异常时回滚
- **时间对齐**: 严格按交易日历对齐，处理不同品种上市时间差异

#### 算法实现
- **协整检验**: 严格按Engle-Granger两步法，不简化任何步骤
- **Beta估计**: 使用OLS方法估计配对的Beta系数
- **信号生成**: Kalman滤波必须是二维状态空间模型
- **回测执行**: 必须模拟真实期货交易，包含保证金、滑点、手续费

#### 质量控制
- **精度要求**: β系数6位小数，PnL精确到分
- **性能标准**: 必须满足需求文档中的性能目标
- **错误处理**: 异常情况下记录详细日志，但不降低算法标准
- **代码质量**: 遵循PEP 8规范，100%类型注解覆盖

## 开发建议

### 开发新功能前必读
1. **详读需求文档**: 优先阅读`Requirements/`目录下的对应模块需求
2. **理解接口定义**: 每个模块都有明确的输入输出格式要求
3. **遵循测试用例**: 按需求文档中的测试用例验证功能正确性
4. **保持架构一致**: 不允许改变5模块的架构设计

### 代码实现指南
1. **算法优先**: 先在notebook中验证算法的数学正确性
2. **模块提取**: 算法稳定后提取到对应的lib模块中
3. **接口统一**: 严格按照需求文档的接口定义实现
4. **异常处理**: 数值异常、API失败等都要有完整的错误处理

### 修改现有代码时
1. **需求驱动**: 任何修改都必须有明确的需求依据
2. **向后兼容**: 确保不破坏现有模块的接口和功能
3. **全面测试**: 在notebook中测试修改对整个流程的影响
4. **性能验证**: 确保修改后仍满足性能要求

### 调试和优化策略
1. **数据验证**: 每个模块的输出都要验证数据的合理性
2. **向量化计算**: 优先使用numpy/pandas的向量化操作
3. **内存管理**: 大数据量时使用分批处理避免内存溢出
4. **并行计算**: 协整检验等独立计算可以并行处理

### 常见陷阱避免
1. **时间对齐**: 不同品种上市时间不同，注意处理缺失数据
2. **浮点精度**: 金融计算要注意浮点数精度问题
3. **数据泄露**: 严格区分训练段和验证段，避免未来信息泄露
4. **过度拟合**: Beta估计必须通过样本外验证，不能只看训练段表现

### 文件组织规范

详细的项目结构规范请参考 `PROJECT_STRUCTURE.md` 文件。以下是核心原则：

#### 目录职责划分
- **lib/**: 核心库 - 稳定的、经过测试的生产代码
  - 只存放被多处复用的核心功能模块
  - 必须有完整的文档字符串和类型注解
  - 需要对应的单元测试

- **scripts/**: 可执行脚本 - 按功能分类存放
  - `analysis/`: 分析脚本（协整分析、beta对比等）
  - `backtest/`: 回测脚本（各种策略回测）
  - `pipeline/`: 完整流程脚本
  - `reports/`: 报告生成脚本
  - `signals/`: 信号生成脚本
  - `validation/`: 验证和比较脚本

- **notebooks/**: Jupyter Notebooks - 交互式分析
  - 编号命名：`01_data_management.ipynb`
  - 用于算法原型验证和数据探索
  - 不存放生产代码

- **tests/**: 测试代码
  - `unit/`: 单元测试
  - `integration/`: 集成测试
  - `acceptance/`: 验收测试

- **data/**: 数据文件
  - `futures/`: 原始期货数据
  - `signals/`: 生成的信号文件
  - `cache/`: 缓存文件

- **configs/**: 配置文件
  - 业务参数、合约规格、策略配置

- **output/**: 输出结果（不入版本控制）
  - `backtests/`: 回测结果
  - `reports/`: 生成的报告
  - `plots/`: 图表文件

#### 文件命名规范
- **脚本命名**: 动词开头 `run_`, `generate_`, `compare_`, `verify_`
- **测试命名**: `test_` 开头
- **数据文件**: `{symbol}.parquet`, `signals_{strategy}_{date}.csv`
- **报告文件**: `report_{type}_{timestamp}.{ext}`

#### 记住文件位置的规则
1. **核心算法** → `lib/`
2. **执行脚本** → `scripts/{category}/`
3. **探索分析** → `notebooks/`
4. **测试代码** → `tests/{level}/`
5. **配置文件** → `configs/`
6. **数据文件** → `data/{type}/`
7. **输出结果** → `output/{type}/`