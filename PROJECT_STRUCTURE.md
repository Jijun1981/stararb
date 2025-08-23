# 项目结构规范

## 更新日期
2025-08-22

## 目录结构最佳实践

```
Star-arb/
├── lib/                      # 核心库 - 稳定的、经过测试的生产代码
│   ├── __init__.py
│   ├── data.py              # 数据管理模块
│   ├── coint.py             # 协整分析模块
│   ├── signal_generation.py # 信号生成模块
│   └── backtest.py          # 回测引擎模块
│
├── scripts/                  # 脚本目录 - 可执行的Python脚本
│   ├── data/                # 数据管理脚本
│   │   └── update_futures_data.py  # 期货数据增量更新工具
│   │
│   ├── analysis/            # 分析脚本
│   │   ├── run_cointegration_analysis.py
│   │   ├── compare_beta_constraints.py
│   │   └── negative_beta_analysis.py
│   │
│   ├── backtest/            # 回测相关脚本
│   │   ├── run_backtest_ols.py
│   │   ├── run_backtest_kalman.py
│   │   └── run_backtest_with_stoploss.py
│   │
│   ├── pipeline/            # 完整流程脚本
│   │   ├── run_complete_pipeline.py
│   │   ├── run_complete_pipeline_v2.py
│   │   └── run_complete_pipeline_v2_1.py
│   │
│   ├── reports/             # 报告生成脚本
│   │   ├── generate_detailed_report.py
│   │   ├── generate_complete_backtest_report.py
│   │   └── print_all_trades_detail.py
│   │
│   ├── signals/             # 信号生成脚本
│   │   ├── generate_ols_signals.py
│   │   ├── run_signal_generation.py
│   │   └── test_kalman_fix.py
│   │
│   └── validation/          # 验证和测试脚本
│       ├── compare_filtered_sets.py
│       ├── validate_coint_algorithms.py
│       ├── verify_backtest_algorithms.py
│       ├── verify_cointegration_algorithms.py
│       ├── verify_coint_requirements.py  # 协整需求验证
│       ├── verify_data_status.py          # 数据状态验证
│       ├── verify_signal_generation_algorithms.py
│       └── verify_signal_real_data.py
│
├── notebooks/               # Jupyter Notebooks - 交互式分析和探索
│   ├── 00_test_akshare.ipynb
│   ├── 01_data_management.ipynb
│   ├── 02_cointegration_complete.ipynb
│   ├── 04_strategy_demo.ipynb
│   ├── 05_backtest_analysis.ipynb
│   ├── 05_backtest_real_multipliers.ipynb
│   ├── 05_detailed_backtest_analysis.ipynb
│   ├── 05_enhanced_backtest_analysis.ipynb
│   └── 06_new_beta_backtest.ipynb
│
├── tests/                   # 测试代码
│   ├── unit/               # 单元测试
│   │   ├── test_backtest.py
│   │   ├── test_backtest_core.py
│   │   ├── test_coint.py
│   │   ├── test_coint_final.py     # 协整最终测试
│   │   ├── test_data.py
│   │   ├── test_min_ratio.py
│   │   ├── test_signal_generation.py
│   │   └── verify_calculations.py
│   │
│   ├── integration/        # 集成测试
│   │   ├── test_incremental_update.py  # 增量更新测试
│   │   └── test_pipeline.py
│   │
│   └── acceptance/         # 验收测试
│       └── test_complete_system.py
│
├── data/                   # 数据文件
│   ├── futures/           # 期货原始数据
│   │   └── *.parquet      # 14个品种的parquet文件
│   │
│   ├── signals/           # 生成的信号文件
│   │   └── *.csv
│   │
│   ├── cache/             # 缓存文件
│   │   └── *.pkl
│   │
│   └── update_log.csv     # 数据更新日志
│
├── configs/               # 配置文件
│   ├── business.yaml      # 业务参数配置
│   └── contract_specs.json # 合约规格
│
├── output/                # 输出结果（不入版本控制）
│   ├── cointegration/    # 协整配对模块输出
│   │   ├── verification/  # 算法验证结果
│   │   └── results/      # 协整分析结果
│   │
│   ├── signals/          # 信号生成模块输出
│   ├── backtest/         # 回测模块输出
│   └── tests/            # 测试输出
│
├── docs/                  # 文档
│   ├── API.md            # API文档
│   ├── ARCHITECTURE.md   # 架构文档
│   ├── MODULE_INTERFACES.md # 模块间接口规范
│   ├── OUTPUT_SPECIFICATION.md # 输出格式规范
│   ├── README.md         # 文档说明
│   ├── backtest_verification_report.md    # 回测验证报告
│   ├── signal_generation_api_summary.md   # 信号生成API总结
│   ├── signal_generation_verification_report.md  # 信号生成验证报告
│   ├── traceability.yaml   # 需求追踪矩阵
│   └── Requrements/       # 需求文档
│       ├── 00_requirements_overview.md
│       ├── 01_data_management.md
│       ├── 02_cointegration_pairing.md
│       ├── 03_signal_generation.md
│       └── 04_backtest_framework.md
│
├── backup/               # 备份文件（不入版本控制）
│   ├── beta_estimation/  # Beta估计备份
│   ├── lib_cleanup/      # 清理的库文件
│   ├── lib_original/     # 原始库文件
│   └── notebooks_original/ # 原始notebook文件
│
├── logs/                 # 日志文件（不入版本控制）
├── models/               # 模型文件
├── reports/              # 报告输出
├── results/              # 结果文件
├── test_reports/         # 测试报告
│
├── CLAUDE.md            # Claude AI的记忆文件
├── PROJECT_STRUCTURE.md # 本文件 - 项目结构说明
├── README.md            # 项目说明
├── requirements.txt     # 依赖管理
└── .gitignore          # Git忽略配置
```

## 新增文件说明

### 数据更新工具
- **scripts/data/update_futures_data.py**: 期货数据增量更新工具
  - 支持检查数据状态 (--check)
  - 支持批量更新所有品种
  - 支持指定品种更新
  - 支持模拟运行 (--dry-run)
  - 实现REQ-1.3.x增量更新需求

### 测试脚本
- **test_incremental_update.py**: 测试增量更新功能
  - 单品种增量更新测试
  - 批量更新测试
  - 错误恢复机制测试
  - 更新日志验证

- **verify_data_status.py**: 验证数据状态和性能
  - 检查所有品种数据状态
  - 性能测试（加载时间）
  - 数据质量检查
  - OHLC关系验证

### 接口文档
- **docs/MODULE_INTERFACES.md**: 模块间接口规范文档
  - 定义模块间数据流
  - 规范接口输入输出格式
  - Beta精度规范（6位小数）
  - 异常处理规范

### 验证脚本
- **scripts/validation/verify_cointegration_algorithms.py**: 协整算法验证
  - 使用NumPy、Statsmodels、Sklearn三种方法验证OLS Beta（误差<3%）
  - ADF检验与Phillips-Perron检验对比
  - 多种半衰期计算方法对比（AR(1)、OLS、ACF）
  - 波动率计算验证（标准差、EWMA、滚动窗口）
  - 输出MD和CSV格式验证报告到output/cointegration/verification/

### 追踪文件
- **traceability.yaml**: 需求追踪矩阵
  - 记录30个需求的实现状态
  - 记录新增脚本和功能
  - 记录数据更新指标
  - 100%测试覆盖率

## 文件命名规范

### Python文件
- **库模块** (`lib/`): 使用小写字母和下划线，如 `signal_generation.py`
- **脚本** (`scripts/`): 使用动词开头，如 `run_`, `generate_`, `compare_`, `verify_`
- **测试** (`tests/`): 以 `test_` 开头，如 `test_backtest.py`
- **Notebooks**: 编号+描述，如 `01_data_management.ipynb`

### 数据文件
- **原始数据**: `{symbol}.parquet` 如 `AG0.parquet`
- **信号文件**: `signals_{strategy}_{date}.csv`
- **报告文件**: `report_{type}_{timestamp}.{ext}`

## 文件放置原则

### 什么放在 lib/
- ✅ 经过充分测试的稳定代码
- ✅ 被多个脚本复用的功能模块
- ✅ 核心业务逻辑
- ❌ 实验性代码
- ❌ 一次性脚本
- ❌ 特定分析的代码

### 什么放在 scripts/
- ✅ 可执行的Python脚本
- ✅ 特定任务的实现
- ✅ 数据处理和分析脚本
- ✅ 报告生成脚本
- ❌ 核心算法实现
- ❌ 可复用的功能模块

### 什么放在 notebooks/
- ✅ 交互式数据探索
- ✅ 算法原型验证
- ✅ 可视化分析
- ✅ 教学和演示
- ❌ 生产代码
- ❌ 批量处理脚本

### 什么放在 tests/
- ✅ 单元测试
- ✅ 集成测试
- ✅ 性能测试
- ✅ 验收测试
- ❌ 临时调试代码
- ❌ 手动测试脚本

## 版本控制原则

### 应该入库的文件
- ✅ 所有源代码 (`lib/`, `scripts/`, `tests/`)
- ✅ Notebooks (`notebooks/`)
- ✅ 配置文件 (`configs/`)
- ✅ 文档 (`docs/`, `*.md`)
- ✅ 依赖说明 (`requirements.txt`)

### 不应该入库的文件
- ❌ 数据文件 (`data/` - 除了小型示例数据)
- ❌ 输出结果 (`output/`)
- ❌ 日志文件 (`logs/`)
- ❌ 缓存文件 (`__pycache__/`, `*.pyc`)
- ❌ 虚拟环境 (`.venv/`)
- ❌ IDE配置 (`.vscode/`, `.idea/`)
- ❌ 临时文件 (`*.tmp`, `*.bak`)

## 开发工作流

### 1. 新功能开发
```bash
# 1. 在notebook中原型验证
notebooks/xx_feature_prototype.ipynb

# 2. 提取稳定代码到lib
lib/feature_module.py

# 3. 编写测试
tests/unit/test_feature_module.py

# 4. 创建使用脚本
scripts/category/run_feature.py
```

### 2. 分析任务
```bash
# 1. 创建分析脚本
scripts/analysis/analyze_xxx.py

# 2. 生成报告
output/reports/analysis_xxx_report.html

# 3. 如果需要复用，提取到lib
lib/analysis_utils.py
```

### 3. 回测验证
```bash
# 1. 使用标准回测脚本
scripts/backtest/run_backtest_xxx.py

# 2. 输出结果到
output/backtests/backtest_xxx_20250822.csv

# 3. 生成报告
scripts/reports/generate_backtest_report.py
```

## 命令示例

### 数据更新
```bash
python scripts/pipeline/update_data.py
```

### 运行完整流程
```bash
python scripts/pipeline/run_complete_pipeline.py
```

### 生成报告
```bash
python scripts/reports/generate_complete_report.py
```

### 运行测试
```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 全部测试
pytest tests/
```

## 维护指南

### 定期清理
- 每周清理 `output/` 目录中的旧文件
- 每月备份重要的分析结果到 `backup/`
- 定期更新 `requirements.txt`

### 代码迁移
当脚本代码稳定后，考虑迁移到 `lib/`:
1. 提取可复用的函数
2. 添加完整的文档字符串
3. 编写对应的单元测试
4. 更新 `lib/__init__.py`

### 文档更新
- 新增模块时更新 `API.md`
- 架构变更时更新 `ARCHITECTURE.md`
- 重要决策记录在 `docs/decisions/`