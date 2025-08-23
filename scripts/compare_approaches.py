#!/usr/bin/env python3
"""
对比我们的回测脚本与完整流程脚本的关键差异
"""

print("=" * 80)
print("对比两个脚本的关键差异")
print("=" * 80)

print("\n1. 时间配置差异:")
print("我们的脚本:")
print("  data_start: 2019-01-01")
print("  convergence_end: 2023-06-30")
print("  signal_start: 2023-07-01")
print("  backtest_start: 2023-07-01")

print("\n完整流程脚本:")
print("  data_start: 2020-01-01")
print("  beta_training_start: 2023-01-01")
print("  beta_training_end: 2023-12-31")
print("  convergence_start: 2024-01-01")
print("  convergence_end: 2024-06-30")
print("  signal_start: 2024-07-01")

print("\n2. 信号参数差异:")
print("我们的脚本:")
print("  z_open: 2.0")
print("  z_close: 0.5")

print("\n完整流程脚本:")
print("  z_open: 2.2")  
print("  z_close: 0.3")

print("\n3. 合约规格差异:")
print("我们的脚本: 使用JSON格式（CU0乘数=5, I0乘数=100）")
print("完整流程脚本: 内置规格（CU0乘数=5, I0乘数=100）")

print("\n4. 价格数据处理差异:")
print("我们的脚本:")
print("  信号生成: log_price=True")
print("  回测执行: log_price=False")
print("  列名处理: 手动替换'_close'后缀")

print("\n完整流程脚本:")  
print("  信号生成: log_price=False (原始价格!)")
print("  回测执行: log_price=False")
print("  列名处理: 使用'f\"{symbol}_close\"'格式")

print("\n5. 信号转换逻辑差异:")
print("我们的脚本:")
print("  直接使用原子服务生成的信号格式")

print("\n完整流程脚本:")
print("  转换信号格式:")
print("    open_long -> long_spread")
print("    open_short -> short_spread")
print("    close -> close")

print("\n🔍 关键问题发现:")
print("1. 信号生成使用的价格数据类型不同！")
print("   - 我们用对数价格生成信号")
print("   - 完整流程用原始价格生成信号")
print("2. 时间周期不同")
print("3. Z-score阈值不同")

print("\n💡 建议:")
print("1. 检查信号生成应该用什么价格数据")
print("2. 统一时间配置")
print("3. 验证Z-score阈值设置")