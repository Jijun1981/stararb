"""调试SN-SF配对的仓位问题"""

from lib.backtest.position_sizing import PositionSizer, PositionSizingConfig
from lib.backtest.trade_executor import TradeExecutor, ExecutionConfig
import pandas as pd

# 读取历史数据找到SN和SF的价格
data_sn = pd.read_parquet('data/SN.parquet')
data_sf = pd.read_parquet('data/SF.parquet')

# 找到2024-08-08的价格（开仓日期）
date = '2024-08-08'
price_sn = data_sn.loc[date, 'close']
price_sf = data_sf.loc[date, 'close']

print(f"SN-SF配对分析 (2024-08-08)")
print(f"SN价格: {price_sn:,.0f}")
print(f"SF价格: {price_sf:,.0f}")

# 配置
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
    'SN': {'multiplier': 1, 'tick_size': 10},
    'SF': {'multiplier': 5, 'tick_size': 2}
}
executor.set_contract_specs(contract_specs)

# 假设beta（从协整分析获得）
beta = -0.0086  # 负beta

# 计算最小整数比
ratio_result = sizer.calculate_min_integer_ratio(
    beta=abs(beta),  # 使用绝对值
    price_x=price_sn,
    price_y=price_sf,
    multiplier_x=1,
    multiplier_y=5
)

print(f"\n最小整数比:")
print(f"SN手数: {ratio_result['lots_x']}")
print(f"SF手数: {ratio_result['lots_y']}")

# 应用资金约束
total_capital = 5000000
position_result = sizer.calculate_position_size(
    min_lots={
        'lots_x': ratio_result['lots_x'],
        'lots_y': ratio_result['lots_y']
    },
    prices={'x': price_sn, 'y': price_sf},
    multipliers={'x': 1, 'y': 5},
    total_capital=total_capital,
    position_weight=0.05
)

print(f"\n资金分配:")
print(f"分配资金: {position_result['allocated_capital']:,.0f}")
print(f"最终SN手数: {position_result['final_lots_x']}")
print(f"最终SF手数: {position_result['final_lots_y']}")
print(f"缩放系数k: {position_result['scaling_factor']}")

# 计算合约价值
sn_value = position_result['final_lots_x'] * price_sn * 1
sf_value = position_result['final_lots_y'] * price_sf * 5
total_value = sn_value + sf_value

print(f"\n持仓价值:")
print(f"SN合约价值: {sn_value:,.0f}")
print(f"SF合约价值: {sf_value:,.0f}")
print(f"总合约价值: {total_value:,.0f}")
print(f"保证金需求: {position_result['margin_required']:,.0f}")

# 模拟10%的价格不利变动
price_change = 0.1
sn_loss = position_result['final_lots_x'] * price_sn * 1 * price_change
sf_loss = position_result['final_lots_y'] * price_sf * 5 * price_change

print(f"\n如果价格不利变动10%:")
print(f"SN亏损: {sn_loss:,.0f}")
print(f"SF亏损: {sf_loss:,.0f}")
print(f"总亏损: {(sn_loss + sf_loss):,.0f}")
print(f"占分配资金比例: {(sn_loss + sf_loss) / position_result['allocated_capital'] * 100:.1f}%")

# 反推实际亏损
actual_loss = 84022  # 从交易记录中看到的亏损
implied_price_change = actual_loss / total_value
print(f"\n实际亏损 {actual_loss:,.0f} 元")
print(f"意味着价格变动: {implied_price_change * 100:.1f}%")