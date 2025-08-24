# 项目结构规范

## 更新日期
2025-08-23

## 目录结构详细说明

根据实际项目结构整理，明确每个目录的用途和文件放置原则。

```
Star-arb/                    # 项目根目录
├── lib/                     # ✅ 核心库 - 稳定的生产代码
│   ├── __init__.py          # 模块初始化
│   ├── data.py              # 数据管理模块（支持data-joint源）
│   ├── coint.py             # 协整分析模块（Engle-Granger）
│   ├── signal_generation.py # 信号生成模块（Kalman滤波+状态机）
│   └── backtest.py          # 回测引擎模块
│
├── scripts/                 # ✅ 可执行脚本 - 按功能分类
│   ├── data/               # 数据管理脚本
│   │   ├── convert_jq_to_parquet.py  # 数据格式转换
│   │   └── update_futures_data.py   # 期货数据更新
│   │
│   ├── analysis/           # 分析脚本
│   │   ├── check_complete_calculation_logic.py
│   │   ├── compare_screening_modes.py
│   │   ├── debug_beta_calculation.py
│   │   ├── run_parameterized_cointegration_screening.py
│   │   ├── verify_beta_direction.py
│   │   └── verify_ols_beta_calculation.py
│   │
│   ├── backtest/           # 回测执行脚本
│   │   ├── run_backtest_v4.py
│   │   └── test_backtest_v4.py
│   │
│   ├── pipeline/           # 完整流程脚本
│   │   ├── run_complete_pipeline.py        # V1版本（兼容）
│   │   ├── run_complete_pipeline_v2.py     # V2版本（稳定）
│   │   ├── run_complete_pipeline_v2_1.py   # V2.1版本（最新）
│   │   ├── run_complete_pipeline_shifted.py
│   │   ├── run_ols_rolling_pipeline.py
│   │   ├── e2e_pipeline.py
│   │   └── e2e_pipeline_clean.py
│   │
│   ├── signals/            # 信号生成脚本
│   │   ├── generate_ols_rolling_signals.py
│   │   ├── signal_generation_test.py
│   │   └── test_with_real_data.py
│   │
│   ├── validation/         # 验证和比较脚本
│   │   ├── compare_backtest_versions.py
│   │   ├── compare_filtered_sets.py
│   │   ├── consistency_check.py
│   │   ├── test_pipeline_v21_calculations.py
│   │   ├── validate_coint_algorithms.py
│   │   ├── verify_backtest_algorithms.py
│   │   ├── verify_cointegration_algorithms.py
│   │   ├── verify_data_status.py
│   │   ├── verify_lots_direction.py
│   │   └── verify_signal_generation_algorithms.py
│   │
│   ├── joinquant/          # 聚宽数据获取脚本
│   │   ├── fetch_8888_data.py
│   │   └── fetch_jq_data.py
│   │
│   └── 其他调试脚本/        # 临时调试和测试脚本
│       ├── debug_signal_systematic.py
│       ├── test_complete_pipeline_clear_states.py
│       ├── debug_backtest_execution.py
│       └── debug_signal_generation.py
│
├── notebooks/              # ✅ Jupyter Notebooks - 交互式分析
│   ├── 00_test_akshare.ipynb           # AkShare接口测试
│   ├── 01_data_management.ipynb        # 数据管理演示
│   ├── 02_cointegration_complete.ipynb # 协整分析演示
│   ├── 04_strategy_demo.ipynb          # 策略演示
│   ├── 05_backtest_analysis.ipynb      # 回测分析
│   ├── 05_backtest_real_multipliers.ipynb
│   ├── 05_detailed_backtest_analysis.ipynb
│   ├── 05_enhanced_backtest_analysis.ipynb
│   └── 06_new_beta_backtest.ipynb
│
├── tests/                  # ✅ 测试代码 - 分层次测试
│   ├── unit/              # 单元测试
│   │   ├── test_backtest.py              # 回测模块测试
│   │   ├── test_backtest_core.py
│   │   ├── test_backtest_tdd.py
│   │   ├── test_coint.py                 # 协整模块测试
│   │   ├── test_coint_final.py
│   │   ├── test_coint_v3.py
│   │   ├── test_coint_v4_parameterized.py
│   │   ├── test_data.py                  # 数据模块测试
│   │   ├── test_data_v3.py
│   │   ├── test_signal_generation.py     # 信号生成测试
│   │   ├── test_signal_state_machine.py  # 状态机测试
│   │   ├── test_signal_tdd.py
│   │   └── verify_calculations.py
│   │
│   ├── integration/       # 集成测试
│   │   └── test_incremental_update.py
│   │
│   ├── acceptance/        # 验收测试
│   │   └── test_e2e_pipeline.py
│   │
│   └── test_xy_consistency.py  # 配对一致性测试
│
├── data/                   # ✅ 数据文件 - 分类存储
│   ├── data-joint/        # 聚宽8888主力连续合约数据（源数据）
│   │   ├── jq_8888_AG.csv      # 白银数据
│   │   ├── jq_8888_AL.csv      # 铝数据
│   │   ├── jq_8888_AU.csv      # 黄金数据
│   │   ├── jq_8888_CU.csv      # 铜数据
│   │   ├── jq_8888_HC.csv      # 热卷数据
│   │   ├── jq_8888_I.csv       # 铁矿石数据
│   │   ├── jq_8888_NI.csv      # 镍数据
│   │   ├── jq_8888_PB.csv      # 铅数据
│   │   ├── jq_8888_RB.csv      # 螺纹钢数据
│   │   ├── jq_8888_SF.csv      # 硅铁数据
│   │   ├── jq_8888_SM.csv      # 锰硅数据
│   │   ├── jq_8888_SN.csv      # 锡数据
│   │   ├── jq_8888_SS.csv      # 不锈钢数据
│   │   └── jq_8888_ZN.csv      # 锌数据
│   │
│   ├── cache/             # 缓存文件（可选）
│   └── update_log.csv     # 数据更新日志
│
├── configs/               # ✅ 配置文件
│   ├── business.yaml      # 业务参数配置
│   └── contract_specs.json # 合约规格配置
│
├── output/                # ❌ 输出结果（不入版本控制）
│   ├── cointegration/    # 协整分析结果
│   │   ├── verification/  # 算法验证结果
│   │   └── results/      # 配对筛选结果
│   │
│   ├── backtest/         # 回测结果
│   │   ├── equity_*.csv        # 权益曲线
│   │   └── trades_*.csv        # 交易记录
│   │
│   ├── pipeline_v21/     # V2.1流程输出
│   │   ├── cointegrated_pairs_*.csv # 协整配对
│   │   ├── signals_*.csv           # 信号记录  
│   │   ├── trades_*.csv            # 交易记录
│   │   └── pipeline_report_*.json  # 流程报告
│   │
│   ├── pipeline_shifted/ # 移动窗口流程输出
│   ├── ols_rolling_*/    # OLS滚动窗口输出
│   ├── e2e/              # 端到端测试输出
│   ├── beta_analysis/    # Beta分析结果
│   ├── kalman_analysis/  # Kalman滤波分析
│   └── tests/            # 测试输出结果
│
├── docs/                  # ✅ 文档 - 需求和架构文档
│   ├── API.md            # API接口文档
│   ├── ARCHITECTURE.md   # 系统架构文档
│   ├── backtest_framework_summary.md # 回测框架总结
│   ├── traceability.yaml # 需求追踪矩阵
│   └── Requrements/      # 需求文档（核心）
│       ├── 00_requirements_overview.md    # 需求总览
│       ├── 01_data_management.md          # 数据管理需求
│       ├── 02_cointegration_pairing.md    # 协整配对需求
│       ├── 03_signal_generation.md        # 信号生成需求
│       └── 04_backtest_framework.md       # 回测框架需求
│
├── backup/               # ❌ 备份文件（不入版本控制）
│   ├── README.md             # 备份说明
│   ├── beta_estimation/      # Beta估计方法备份
│   ├── lib_cleanup/          # 清理掉的旧库文件
│   ├── lib_original/         # 原始库文件版本
│   └── notebooks_original/   # 原始notebook备份
│
├── logs/                 # ❌ 日志文件（不入版本控制）
├── models/               # 模型文件（如有）
├── reports/              # 报告输出目录
├── results/              # 结果文件
├── test_reports/         # ✅ 测试报告
│   └── data_management_test_*.json  # 数据管理测试报告
│
├── cointegration_results.csv        # ✅ 协整分析结果（根目录）
├── signals_complete_pipeline_*.csv # ✅ 完整流程信号输出（根目录）
├── backtest_trades_*.csv          # 回测交易记录（根目录）
├── debug_*.csv                    # 调试输出文件（根目录）
├── signals_test.csv               # 测试信号文件（根目录）
│
├── CLAUDE.md                      # ✅ Claude AI项目指令
├── PROJECT_STRUCTURE.md           # ✅ 本文件 - 项目结构说明
└── requirements.txt               # Python依赖管理（如有）
```

## 📁 文件放置指南

### ✅ 什么放在 `lib/` （核心库）
**用途：** 稳定、经过测试、被多处复用的生产代码

**应该放入：**
- ✅ 经过充分测试的算法实现
- ✅ 被多个脚本复用的功能模块  
- ✅ 核心业务逻辑（数据管理、协整分析、信号生成、回测引擎）
- ✅ 有完整文档和类型注解的代码

**不应该放入：**
- ❌ 实验性代码或原型
- ❌ 一次性使用的脚本
- ❌ 调试和测试代码
- ❌ 特定分析任务的代码

**现有模块：**
- `data.py` - 数据管理（支持data-joint CSV源）
- `coint.py` - 协整分析（Engle-Granger检验）
- `signal_generation.py` - 信号生成（Kalman滤波+状态机）
- `backtest.py` - 回测引擎

### ✅ 什么放在 `scripts/` （可执行脚本）
**用途：** 按功能分类的可执行Python脚本

**分类原则：**
- `data/` - 数据获取、转换、更新脚本
- `analysis/` - 各种分析任务（协整、Beta、算法验证等）
- `backtest/` - 回测执行脚本
- `pipeline/` - 端到端完整流程脚本
- `signals/` - 信号生成相关脚本
- `validation/` - 验证、比较、测试脚本
- `joinquant/` - 数据源特定脚本

**命名规范：**
- 以动词开头：`run_`, `generate_`, `compare_`, `verify_`, `debug_`
- 描述具体功能：`run_complete_pipeline_v2_1.py`

### ✅ 什么放在 `notebooks/` （交互式分析）
**用途：** Jupyter Notebooks用于交互式数据探索和演示

**应该放入：**
- ✅ 数据探索和可视化分析
- ✅ 算法原型验证和演示
- ✅ 教学和说明性内容
- ✅ 交互式参数调试

**不应该放入：**
- ❌ 生产环境代码
- ❌ 批量处理脚本
- ❌ 自动化流程代码

**编号规范：** `编号_功能描述.ipynb`

### ✅ 什么放在 `tests/` （测试代码）
**用途：** 分层次的测试代码

**分层结构：**
- `unit/` - 单元测试（测试单个模块函数）
- `integration/` - 集成测试（测试模块间协作）
- `acceptance/` - 验收测试（测试完整业务流程）

**命名规范：** `test_` 开头

### 📊 什么放在 `data/` （数据文件）
**用途：** 各类数据文件的分类存储

**结构说明：**
- `data-joint/` - 聚宽源数据（14个品种CSV）
- `cache/` - 临时缓存文件
- `update_log.csv` - 数据更新日志

**注意：** 大数据文件不入版本控制

### 📄 什么放在 `docs/` （文档）
**用途：** 项目文档和需求规范

**核心文档：**
- `Requrements/` - 4个核心需求文档（数据管理、协整配对、信号生成、回测框架）
- `API.md` - 接口文档
- `ARCHITECTURE.md` - 架构文档
- `traceability.yaml` - 需求追踪

## 🏃‍♂️ 常用命令和工作流

### 快速开始命令

```bash
# 运行最新完整流程（推荐）
python3 scripts/pipeline/run_complete_pipeline_v2_1.py

# 运行稳定版本流程
python3 scripts/pipeline/run_complete_pipeline_v2.py

# 运行兼容版本流程
python3 scripts/pipeline/run_complete_pipeline.py

# 测试状态机清晰版本
python3 scripts/test_complete_pipeline_clear_states.py

# 运行单元测试
pytest tests/unit/

# 运行状态机测试
pytest tests/unit/test_signal_state_machine.py -v
```

### 文件寻找指南

**🔍 我要找...**

| 需求 | 位置 | 具体文件 |
|------|------|----------|
| 核心算法实现 | `lib/` | `data.py`, `coint.py`, `signal_generation.py`, `backtest.py` |
| 运行完整流程 | `scripts/pipeline/` | `run_complete_pipeline_v2_1.py` |
| 协整分析脚本 | `scripts/analysis/` | `run_parameterized_cointegration_screening.py` |
| 信号生成脚本 | `scripts/signals/` | `generate_ols_rolling_signals.py` |
| 调试信号问题 | `scripts/` | `debug_signal_systematic.py` |
| 验证算法正确性 | `scripts/validation/` | `verify_*.py` |
| 数据管理演示 | `notebooks/` | `01_data_management.ipynb` |
| 协整分析演示 | `notebooks/` | `02_cointegration_complete.ipynb` |
| 需求文档 | `docs/Requrements/` | `03_signal_generation.md` |
| 测试状态机 | `tests/unit/` | `test_signal_state_machine.py` |
| 输出结果查看 | `output/pipeline_v21/` | `signals_*.csv`, `trades_*.csv` |
| 协整结果 | 根目录 | `cointegration_results.csv` |
| 最新信号文件 | 根目录 | `signals_complete_pipeline_*.csv` |

### 版本控制原则

**✅ 应该提交到Git：**
- 所有源代码（`lib/`, `scripts/`, `tests/`）
- 配置文件（`configs/`）
- 文档（`docs/`, `*.md`）
- Notebooks（`notebooks/`）

**❌ 不应该提交到Git：**
- 大数据文件（`data/`中的CSV）
- 输出结果（`output/`）
- 日志文件（`logs/`）
- 缓存文件（`__pycache__/`, `*.pyc`）
- 根目录下的结果CSV文件

## 📝 开发最佳实践

### 新功能开发流程

```bash
# 1. 在notebook中验证算法原型
notebooks/xx_new_feature_prototype.ipynb

# 2. 算法稳定后，提取到lib模块
lib/new_feature_module.py

# 3. 编写单元测试验证
tests/unit/test_new_feature_module.py

# 4. 创建执行脚本
scripts/category/run_new_feature.py

# 5. 更新文档
docs/API.md  # 如果有新的接口
```

### 文件命名规范

**Python文件：**
- **库模块** (`lib/`)：小写+下划线，如 `signal_generation.py`
- **脚本** (`scripts/`)：动词开头，如 `run_complete_pipeline_v2_1.py`
- **测试** (`tests/`)：`test_` 开头，如 `test_signal_state_machine.py`
- **Notebooks**：编号+功能，如 `01_data_management.ipynb`

**数据文件：**
- **信号文件**：`signals_complete_pipeline_20250823_212252.csv`
- **协整结果**：`cointegration_results.csv`
- **交易记录**：`backtest_trades_20250823_195509.csv`
- **调试文件**：`debug_AU_ZN_signals.csv`

### 代码迁移指南

**何时将脚本代码迁移到lib：**
1. 代码在多个地方被复用
2. 算法经过充分测试验证
3. 代码稳定且不再频繁修改
4. 有完整的文档和类型注解

**迁移步骤：**
1. 提取可复用函数到lib模块
2. 添加完整的docstring文档
3. 编写对应单元测试
4. 更新`lib/__init__.py`导入
5. 更新使用该功能的脚本

## 🎯 项目维护指南

### 定期清理任务
```bash
# 每周清理输出目录
rm -rf output/*/  # 清理旧的输出结果
find . -name "*.csv" -path "./signals_*" -mtime +7 -delete  # 清理一周前的信号文件

# 每月备份重要结果  
cp cointegration_results.csv backup/
cp signals_complete_pipeline_latest.csv backup/
```

### 常见问题排查

**Q: 找不到某个功能在哪里实现？**
A: 查看上面的"文件寻找指南"表格，根据功能类型找到对应位置

**Q: 要添加新的分析脚本应该放在哪？**
A: 
- 如果是协整相关 → `scripts/analysis/`
- 如果是信号生成相关 → `scripts/signals/`
- 如果是回测相关 → `scripts/backtest/`
- 如果是验证比较 → `scripts/validation/`

**Q: 什么时候应该把代码从scripts移到lib？**
A: 满足以下条件时考虑迁移：
1. 被3个以上脚本复用
2. 代码稳定运行1个月以上
3. 有完整的测试覆盖
4. 属于核心业务逻辑

**Q: output目录下文件太多怎么办？**
A: 定期清理，只保留最近的重要结果。大部分output内容都可以重新生成。

### 项目结构演进记录
- **2025-08-23**: 添加状态机测试，整理项目结构文档
- **2025-08-22**: 实现V2.1流程，优化Kalman参数
- **更早版本**: 基础4模块架构建立

---

**总结**: 这份文档提供了清晰的文件放置指南，帮助快速定位所需功能和添加新内容。遵循这些原则可以保持项目结构整洁且易于维护。