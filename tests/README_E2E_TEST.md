# 端到端测试脚本使用指南

## 概述

`test_e2e_pipeline.py` 是一个完整的端到端测试脚本，用于验证配对交易系统的所有核心功能模块。该脚本基于 `configs/e2e_pipeline_config.yaml` 配置文件，严格按照API文档规范进行测试验证。

## 测试模块

### 1. 数据管理模块测试
- 验证数据加载功能
- 检查数据格式和完整性
- 验证时间序列对齐
- 检查数据质量指标

### 2. 协整配对分析测试
- 创建协整分析器
- 测试单配对协整检验
- 验证统计结果格式
- 检查分析结果的合理性

### 3. 信号生成模块测试
- 创建信号生成器
- 测试Kalman滤波器
- 验证Z-score信号计算
- 检查交易信号格式

### 4. 回测框架模块测试
- 配置回测引擎
- 设置合约规格
- 运行回测流程
- 验证回测结果格式

### 5. 完整流程集成测试
- 端到端流程验证
- 数据流完整性检查
- 集成结果验证
- 性能指标检查

### 6. 详细结果验证
- 生成测试报告
- 保存详细结果
- 验证测试完整性
- 提供可视化分析

## 使用方法

### 基本运行
```bash
# 使用默认配置运行
python tests/test_e2e_pipeline.py

# 或者直接执行（已添加可执行权限）
./tests/test_e2e_pipeline.py
```

### 指定配置文件
```bash
python tests/test_e2e_pipeline.py --config configs/e2e_pipeline_config.yaml
```

### 详细日志输出
```bash
python tests/test_e2e_pipeline.py --verbose
```

### 指定输出目录
```bash
python tests/test_e2e_pipeline.py --output output/my_test_results
```

### 完整参数示例
```bash
python tests/test_e2e_pipeline.py \
    --config configs/e2e_pipeline_config.yaml \
    --output output/test_results \
    --verbose
```

## 输出结果

### 测试报告
- **文件位置**: `output/test_results/e2e_test_report_YYYYMMDD_HHMMSS.md`
- **格式**: Markdown格式的测试报告
- **内容**: 
  - 测试概要统计
  - 每个模块的详细测试结果
  - 失败测试列表
  - 配置参数总结

### 详细结果
- **文件位置**: `output/test_results/e2e_test_results_YYYYMMDD_HHMMSS.json`
- **格式**: JSON格式的详细结果
- **内容**:
  - 每个测试模块的完整结果
  - 错误信息和异常详情
  - 性能指标和统计数据
  - 配置参数和测试数据

### 日志文件
- **文件位置**: `test_e2e_pipeline.log`
- **格式**: 文本格式的详细日志
- **内容**: 所有测试过程的详细记录

## 测试配置

测试脚本使用 `configs/e2e_pipeline_config.yaml` 中的以下主要配置：

### 时间配置
```yaml
time_config:
  data_start_date: "2020-01-02"
  data_end_date: "2025-08-20"
  signal_generation_start: "2024-07-01"
  backtest_start: "2024-07-01"
```

### 品种配置
```yaml
symbols:
  metals:
    precious: ["AG", "AU"]
    nonferrous: ["AL", "CU", "NI", "PB", "SN", "ZN"]
    ferrous: ["HC", "I", "RB", "SF", "SM", "SS"]
```

### 信号生成配置
```yaml
signal_generation:
  ols_window: 60
  kalman_warmup: 30
  kalman_params:
    Q_beta: 5.0e-6
    Q_alpha: 1.0e-5
    R_init: 0.005
  signal_thresholds:
    z_open: 2.0
    z_close: 0.5
```

### 回测配置
```yaml
backtest:
  capital_management:
    initial_capital: 5000000
    position_weight: 0.05
    margin_rate: 0.12
  trading_costs:
    commission_rate: 0.0003
    slippage_ticks: 3
```

## 期望结果

### 成功标准
所有测试模块都应该通过，包括：
- ✅ 数据模块: 成功加载14个品种的数据
- ✅ 协整模块: 成功创建分析器并运行协整检验
- ✅ 信号模块: 成功生成交易信号
- ✅ 回测模块: 成功运行回测并生成结果
- ✅ 集成测试: 完整流程端到端验证
- ✅ 结果验证: 生成完整测试报告

### 输出示例
```
🚀 开始端到端测试
测试配置: 端到端配对交易流水线

==================================================
🧪 测试1: 数据管理模块
==================================================
✅ 数据类型检查: PASSED 数据应该是DataFrame
✅ 品种数量检查: PASSED 应该有14个品种，实际14
✅ 索引类型检查: PASSED 索引应该是DatetimeIndex
✅ 数据量检查: PASSED 数据量: 1234 行
✅ 数据完整性检查: PASSED 缺失率: 0.02%

📋 测试总结
============================================================
⏱️  测试耗时: 0:02:15
✅ 通过测试: 25
❌ 失败测试: 0
📊 成功率: 100.0%
🎉 所有测试通过！端到端流程验证成功
```

## 故障排除

### 常见问题

1. **模块导入失败**
   ```
   ❌ 模块导入失败: No module named 'lib.data'
   ```
   - **解决**: 确保在项目根目录运行脚本
   - **检查**: `sys.path` 是否包含项目根目录

2. **配置文件不存在**
   ```
   ❌ 配置文件不存在: configs/e2e_pipeline_config.yaml
   ```
   - **解决**: 检查配置文件路径是否正确
   - **检查**: 使用 `--config` 参数指定正确路径

3. **数据加载失败**
   ```
   ❌ 数据模块测试异常: No such file or directory: 'data-joint'
   ```
   - **解决**: 确保数据目录存在
   - **检查**: `data-joint/` 目录是否包含所需的CSV文件

4. **权限问题**
   ```
   Permission denied: ./tests/test_e2e_pipeline.py
   ```
   - **解决**: 添加可执行权限 `chmod +x tests/test_e2e_pipeline.py`

5. **内存不足**
   ```
   MemoryError: Unable to allocate array
   ```
   - **解决**: 减少测试数据量或使用更小的时间窗口
   - **修改**: 在集成测试中使用更少的品种

### 调试技巧

1. **使用详细日志**
   ```bash
   python tests/test_e2e_pipeline.py --verbose
   ```

2. **单独运行特定测试**
   - 修改 `run_all_tests()` 方法，注释掉不需要的测试

3. **检查中间结果**
   - 测试结果保存在 `self.test_results` 中
   - 可以在测试过程中打印中间变量

4. **减少数据量**
   - 修改配置文件中的日期范围
   - 减少测试品种数量

## 扩展功能

### 添加新测试模块
1. 在 `E2ETestRunner` 类中添加新的测试方法
2. 方法名格式: `test_N_module_name`
3. 在 `run_all_tests` 中添加到测试列表

### 自定义验证逻辑
1. 修改 `_assert_test` 方法添加自定义断言
2. 在各测试模块中添加特定的验证规则

### 扩展报告格式
1. 修改 `_generate_test_report` 方法
2. 支持HTML、PDF等其他格式输出

## 注意事项

1. **测试数据**: 脚本使用真实的期货数据进行测试，确保数据完整性
2. **性能考虑**: 完整测试可能需要几分钟时间，请耐心等待
3. **环境要求**: 需要安装所有必需的Python依赖包
4. **资源占用**: 测试过程可能占用较多内存，建议在高配置机器上运行
5. **日志管理**: 注意日志文件大小，定期清理旧的测试结果

## 版本信息

- **脚本版本**: 1.0.0
- **支持配置**: e2e_pipeline_config.yaml v1.0
- **兼容系统**: lib v2.1.0
- **创建日期**: 2025-08-25
- **作者**: Claude Code

---

*更多详细信息请参考项目文档和API说明*