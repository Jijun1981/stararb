"""验证手数计算逻辑是否正确"""

from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig
import pandas as pd

# 配置
config = PositionSizingConfig(
    max_denominator=10,
    min_lots=1,
    max_lots_per_leg=100,
    margin_rate=0.12,
    position_weight=0.05
)

sizer = PositionSizer(config)

# 测试SN-SF配对
print("=== SN-SF配对手数计算验证 ===\n")

# 从数据读取实际价格
data_sn = pd.read_parquet('data/SN.parquet')
data_sf = pd.read_parquet('data/SF.parquet')

date = '2024-08-08'
price_sn = data_sn.loc[date, 'close']
price_sf = data_sf.loc[date, 'close']

print(f"日期: {date}")
print(f"SN价格: {price_sn:,.0f} 元/吨")
print(f"SF价格: {price_sf:,.0f} 元/吨")

# 合约规格
multiplier_sn = 1  # SN合约乘数
multiplier_sf = 5  # SF合约乘数

# Beta值（从协整分析得到，这里用绝对值）
beta = 0.0086

print(f"\n合约规格:")
print(f"SN乘数: {multiplier_sn}")
print(f"SF乘数: {multiplier_sf}")
print(f"Beta: {beta}")

# 第一步：计算最小整数比
print("\n=== 第一步：计算最小整数比 ===")

# 计算有效对冲比 h* = β × (Py × My) / (Px × Mx)
h_star = beta * (price_sf * multiplier_sf) / (price_sn * multiplier_sn)
print(f"有效对冲比 h* = {beta} × ({price_sf} × {multiplier_sf}) / ({price_sn} × {multiplier_sn})")
print(f"h* = {h_star:.6f}")

# 使用Fraction逼近
from fractions import Fraction
frac = Fraction(h_star).limit_denominator(10)
print(f"连分数逼近: {frac.numerator}/{frac.denominator}")

# 最小整数比
ratio_result = sizer.calculate_min_integer_ratio(
    beta=beta,
    price_x=price_sn,
    price_y=price_sf,
    multiplier_x=multiplier_sn,
    multiplier_y=multiplier_sf
)

print(f"\n最小整数手数:")
print(f"SN: {ratio_result['lots_x']} 手")
print(f"SF: {ratio_result['lots_y']} 手")

# 验证名义价值匹配
nominal_sn = ratio_result['lots_x'] * price_sn * multiplier_sn
nominal_sf = ratio_result['lots_y'] * price_sf * multiplier_sf
print(f"\n名义价值:")
print(f"SN: {nominal_sn:,.0f} 元")
print(f"SF: {nominal_sf:,.0f} 元")
print(f"比例: {nominal_sn/nominal_sf if nominal_sf > 0 else 0:.4f}")

# 第二步：应用资金约束
print("\n=== 第二步：应用资金约束 ===")

total_capital = 5000000
allocated = total_capital * 0.05
print(f"总资金: {total_capital:,.0f}")
print(f"分配资金 (5%): {allocated:,.0f}")

# 计算最小保证金
min_margin_sn = ratio_result['lots_x'] * price_sn * multiplier_sn * 0.12
min_margin_sf = ratio_result['lots_y'] * price_sf * multiplier_sf * 0.12
min_margin_total = min_margin_sn + min_margin_sf

print(f"\n最小整数对的保证金需求:")
print(f"SN: {min_margin_sn:,.0f}")
print(f"SF: {min_margin_sf:,.0f}")
print(f"合计: {min_margin_total:,.0f}")

# 计算缩放系数k
k = int(allocated * 0.95 / min_margin_total)
print(f"\n缩放系数计算:")
print(f"k = floor({allocated} × 0.95 / {min_margin_total:.0f})")
print(f"k = floor({allocated * 0.95:.0f} / {min_margin_total:.0f})")
print(f"k = {k}")

# 最终手数
final_lots_sn = ratio_result['lots_x'] * k
final_lots_sf = ratio_result['lots_y'] * k

print(f"\n最终手数:")
print(f"SN: {ratio_result['lots_x']} × {k} = {final_lots_sn} 手")
print(f"SF: {ratio_result['lots_y']} × {k} = {final_lots_sf} 手")

# 最终合约价值
final_value_sn = final_lots_sn * price_sn * multiplier_sn
final_value_sf = final_lots_sf * price_sf * multiplier_sf
final_value_total = final_value_sn + final_value_sf

print(f"\n最终合约价值:")
print(f"SN: {final_value_sn:,.0f}")
print(f"SF: {final_value_sf:,.0f}")
print(f"合计: {final_value_total:,.0f}")

# 最终保证金
final_margin = final_value_total * 0.12
print(f"\n最终保证金: {final_margin:,.0f}")
print(f"资金利用率: {final_margin/allocated:.1%}")

# 杠杆率
leverage = final_value_total / allocated
print(f"\n杠杆率: {leverage:.1f}倍")

print("\n=== 结论 ===")
if final_margin <= allocated * 0.95:
    print("✓ 保证金在分配资金95%以内，计算正确")
else:
    print("✗ 保证金超过分配资金95%，计算有误")
    
print(f"实际保证金占比: {final_margin/allocated:.1%}")