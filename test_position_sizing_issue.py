"""测试仓位计算问题"""

from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig

# 创建配置
config = PositionSizingConfig(
    max_denominator=10,
    min_lots=1,
    max_lots_per_leg=100,
    margin_rate=0.12,
    position_weight=0.05
)

sizer = PositionSizer(config)

# 模拟一个真实的例子
# CU-ZN配对，beta=0.77
beta = 0.77
price_x = 77000  # CU价格
price_y = 26000  # ZN价格
multiplier_x = 5
multiplier_y = 5

# 第一步：计算最小整数比
ratio_result = sizer.calculate_min_integer_ratio(
    beta=beta,
    price_x=price_x,
    price_y=price_y,
    multiplier_x=multiplier_x,
    multiplier_y=multiplier_y
)

print("=== 第一步：最小整数比 ===")
print(f"Beta: {beta}")
print(f"CU手数: {ratio_result['lots_x']}")
print(f"ZN手数: {ratio_result['lots_y']}")
print(f"有效对冲比: {ratio_result['effective_ratio']:.4f}")

# 第二步：应用资金约束
total_capital = 5000000  # 500万
position_result = sizer.calculate_position_size(
    min_lots={
        'lots_x': ratio_result['lots_x'],
        'lots_y': ratio_result['lots_y']
    },
    prices={'x': price_x, 'y': price_y},
    multipliers={'x': multiplier_x, 'y': multiplier_y},
    total_capital=total_capital,
    position_weight=0.05
)

print("\n=== 第二步：应用资金约束 ===")
print(f"总资金: {total_capital:,.0f}")
print(f"分配资金 (5%): {position_result['allocated_capital']:,.0f}")
print(f"最终CU手数: {position_result['final_lots_x']}")
print(f"最终ZN手数: {position_result['final_lots_y']}")
print(f"缩放系数k: {position_result['scaling_factor']}")
print(f"保证金需求: {position_result['margin_required']:,.0f}")
print(f"持仓价值: {position_result['position_value']:,.0f}")
print(f"资金利用率: {position_result['utilization_rate']:.1%}")

# 计算止损金额
print("\n=== 止损分析 ===")
print(f"分配资金: {position_result['allocated_capital']:,.0f}")
print(f"10%止损金额: {position_result['allocated_capital'] * 0.1:,.0f}")
print(f"保证金: {position_result['margin_required']:,.0f}")
print(f"保证金的10%: {position_result['margin_required'] * 0.1:,.0f}")

# 问题分析
print("\n=== 问题分析 ===")
print(f"如果按保证金计算止损: {position_result['margin_required'] * 0.1:,.0f}")
print(f"如果按分配资金计算止损: {position_result['allocated_capital'] * 0.1:,.0f}")
print(f"差异: {(position_result['margin_required'] - position_result['allocated_capital']) * 0.1:,.0f}")