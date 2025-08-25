"""测试极端情况"""

from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig
from lib.backtest.trade_executor import TradeExecutor, ExecutionConfig

# 创建配置
sizing_config = PositionSizingConfig(
    max_denominator=10,
    min_lots=1,
    max_lots_per_leg=100,
    margin_rate=0.12,
    position_weight=0.05
)

exec_config = ExecutionConfig(
    commission_rate=0.0002,
    slippage_ticks=3,
    margin_rate=0.12
)

sizer = PositionSizer(sizing_config)
executor = TradeExecutor(exec_config)

# 设置合约规格
contract_specs = {
    'AU': {'multiplier': 1000, 'tick_size': 0.02},
    'AG': {'multiplier': 15, 'tick_size': 1}
}
executor.set_contract_specs(contract_specs)

# AU-AG配对，高价值品种
beta = 0.08  # 金银比价
price_x = 550  # AU价格（元/克）
price_y = 7000  # AG价格（元/千克）
multiplier_x = 1000
multiplier_y = 15

# 第一步：计算最小整数比
ratio_result = sizer.calculate_min_integer_ratio(
    beta=beta,
    price_x=price_x,
    price_y=price_y,
    multiplier_x=multiplier_x,
    multiplier_y=multiplier_y
)

print("=== AU-AG配对分析 ===")
print(f"Beta: {beta}")
print(f"AU价格: {price_x} 元/克, 乘数: {multiplier_x}")
print(f"AG价格: {price_y} 元/千克, 乘数: {multiplier_y}")
print(f"\n最小整数比:")
print(f"AU手数: {ratio_result['lots_x']}")
print(f"AG手数: {ratio_result['lots_y']}")

# 第二步：应用资金约束
total_capital = 5000000
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

print(f"\n=== 资金分配 ===")
print(f"总资金: {total_capital:,.0f}")
print(f"分配资金 (5%): {position_result['allocated_capital']:,.0f}")
print(f"最终AU手数: {position_result['final_lots_x']}")
print(f"最终AG手数: {position_result['final_lots_y']}")
print(f"缩放系数k: {position_result['scaling_factor']}")

# 计算实际价值
au_value = position_result['final_lots_x'] * price_x * multiplier_x
ag_value = position_result['final_lots_y'] * price_y * multiplier_y
total_value = au_value + ag_value

print(f"\n=== 持仓价值 ===")
print(f"AU合约价值: {au_value:,.0f}")
print(f"AG合约价值: {ag_value:,.0f}")
print(f"总合约价值: {total_value:,.0f}")
print(f"保证金需求: {position_result['margin_required']:,.0f}")

# 模拟执行开仓
position = executor.execute_open(
    pair_info={
        'pair': 'AU-AG',
        'symbol_x': 'AU',
        'symbol_y': 'AG',
        'beta': beta
    },
    lots={
        'x': position_result['final_lots_x'],
        'y': position_result['final_lots_y']
    },
    prices={'x': price_x, 'y': price_y},
    signal_type='open_long'
)

print(f"\n=== 执行结果 ===")
print(f"开仓保证金: {position.margin:,.0f}")
print(f"开仓手续费: {position.open_commission:,.0f}")

print(f"\n=== 止损分析 ===")
print(f"分配资金: {position_result['allocated_capital']:,.0f}")
print(f"止损阈值 (分配资金的10%): {position_result['allocated_capital'] * 0.1:,.0f}")
print(f"执行保证金: {position.margin:,.0f}")
print(f"如果按保证金10%算止损: {position.margin * 0.1:,.0f}")

# 分析问题
if position.margin > position_result['allocated_capital']:
    print(f"\n⚠️ 问题：保证金({position.margin:,.0f})超过了分配资金({position_result['allocated_capital']:,.0f})!")
    print(f"超出金额: {position.margin - position_result['allocated_capital']:,.0f}")