"""检查止损详情"""

import pandas as pd

# 读取交易记录
trades_df = pd.read_csv('backtest_trades_20250825_133726.csv')

# 筛选止损交易
stop_loss_trades = trades_df[trades_df['close_reason'] == 'stop_loss'].copy()

print(f"止损交易数量: {len(stop_loss_trades)}")
print(f"平均止损金额: {stop_loss_trades['net_pnl'].mean():,.0f}")

# 假设5%的仓位，500万总资金
allocated_capital = 5000000 * 0.05
print(f"\n分配资金: {allocated_capital:,.0f}")
print(f"10%止损阈值: {allocated_capital * 0.1:,.0f}")

# 计算止损百分比
stop_loss_trades['loss_pct'] = stop_loss_trades['net_pnl'] / allocated_capital * 100

print("\n止损交易详情:")
for idx, trade in stop_loss_trades.iterrows():
    print(f"{trade['pair']:8} {trade['direction']:5} 亏损: {trade['net_pnl']:,.0f} ({trade['loss_pct']:.1f}%)")
    
print(f"\n止损百分比统计:")
print(f"最小: {stop_loss_trades['loss_pct'].min():.1f}%")
print(f"最大: {stop_loss_trades['loss_pct'].max():.1f}%")
print(f"平均: {stop_loss_trades['loss_pct'].mean():.1f}%")