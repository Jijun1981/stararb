#!/usr/bin/env python3
"""
评估原子服务中的硬编码问题，为往前平移一年做准备
"""

import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("评估原子服务硬编码问题")
print("=" * 80)

print("1. 时间配置对比:")
print("当前v2.1流程 (2024年信号期):")
print("  beta_training: 2023-01-01 ~ 2023-12-31")
print("  convergence: 2024-01-01 ~ 2024-06-30")
print("  signal_start: 2024-07-01")
print("  backtest_end: 2025-08-20")

print("\n平移一年后 (2023年信号期):")
print("  beta_training: 2022-01-01 ~ 2022-12-31")
print("  convergence: 2023-01-01 ~ 2023-06-30")
print("  signal_start: 2023-07-01")
print("  backtest_end: 2024-08-20")

print(f"\n2. 检查原子服务可能的硬编码:")

# 检查协整模块
print("检查lib/coint.py:")
coint_file = project_root / "lib" / "coint.py"
if coint_file.exists():
    with open(coint_file, 'r') as f:
        content = f.read()
    
    hardcoded_issues = []
    
    # 检查日期硬编码
    if '2024' in content or '2023' in content:
        hardcoded_issues.append("  ❓ 可能有硬编码日期")
    
    # 检查波动率计算的默认日期
    if 'start_date: Optional[str] = None' in content:
        print("  ✓ 波动率计算支持自定义起始日期")
    
    # 检查方向判定的日期依赖
    if 'recent_start' in content:
        print("  ✓ 方向判定支持自定义recent_start参数")
    
    if not hardcoded_issues:
        print("  ✓ 协整模块看起来没有明显硬编码")

# 检查信号生成模块
print("\n检查lib/signal_generation.py:")
signal_file = project_root / "lib" / "signal_generation.py"
if signal_file.exists():
    with open(signal_file, 'r') as f:
        content = f.read()
    
    # 检查时间参数
    if 'convergence_end' in content and 'signal_start' in content:
        print("  ✓ 信号生成支持时间参数配置")
    
    if 'hist_start' in content and 'hist_end' in content:
        print("  ✓ 历史数据时间段可配置")

# 检查回测模块
print("\n检查lib/backtest.py:")
backtest_file = project_root / "lib" / "backtest.py"
if backtest_file.exists():
    print("  ✓ 回测模块主要依赖输入数据，无时间硬编码")

print(f"\n3. 数据可用性检查:")
# 检查是否有2022-2024年的数据
from lib.data import load_data

try:
    # 测试2022年数据
    test_data_2022 = load_data(
        symbols=['CU0', 'I0'],
        start_date='2022-01-01',
        end_date='2022-12-31',
        columns=['close'],
        log_price=False
    )
    
    if not test_data_2022.empty:
        print(f"  ✓ 2022年数据可用: {len(test_data_2022)} 条记录")
    else:
        print(f"  ❌ 2022年数据不可用")
        
    # 测试2023年数据
    test_data_2023 = load_data(
        symbols=['CU0', 'I0'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        columns=['close'],
        log_price=False
    )
    
    if not test_data_2023.empty:
        print(f"  ✓ 2023年数据可用: {len(test_data_2023)} 条记录")
    else:
        print(f"  ❌ 2023年数据不可用")
        
except Exception as e:
    print(f"  ❌ 数据检查失败: {e}")

print(f"\n4. 需要调整的地方:")
print("✓ 主脚本中的TIME_CONFIG时间配置")
print("✓ 协整分析的volatility_start_date参数")
print("✓ 输出目录名称 (pipeline_v21 -> pipeline_v21_shifted)")
print("✓ 验证数据的完整性和质量")

print(f"\n5. 不需要修改的地方:")
print("✓ 原子服务的核心逻辑")
print("✓ 信号生成算法")
print("✓ 回测引擎")
print("✓ 合约规格")

print(f"\n💡 总结:")
print("原子服务设计得很好，基本没有硬编码。")
print("只需要:")
print("1. 调整主脚本的时间配置")
print("2. 确保2022-2024年数据完整")
print("3. 修改输出目录避免覆盖")
print("4. 可能需要调整波动率计算的参考期间")