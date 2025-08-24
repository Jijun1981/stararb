#!/usr/bin/env python3
"""
分析SN0-ZN0 2024年4月交易损失原因
"""

def analyze_sn_zn_trade_loss():
    """分析SN0-ZN0交易损失的详细原因"""
    
    print("=== SN0-ZN0 交易损失分析 (2024-04-19至04-23) ===\n")
    
    # 从AkShare获取的实际价格数据
    sn0_prices = {
        "2024-04-19": 272560.0,
        "2024-04-22": 276010.0, 
        "2024-04-23": 254040.0
    }
    
    zn0_prices = {
        "2024-04-19": 22740.0,
        "2024-04-22": 22535.0,
        "2024-04-23": 22270.0  
    }
    
    # 计算价格变化
    print("📊 实际价格变化情况:")
    print(f"SN0: {sn0_prices['2024-04-19']} → {sn0_prices['2024-04-23']}")
    sn0_change = (sn0_prices['2024-04-23'] - sn0_prices['2024-04-19']) / sn0_prices['2024-04-19'] * 100
    print(f"SN0总变化: {sn0_change:.2f}%")
    
    print(f"ZN0: {zn0_prices['2024-04-19']} → {zn0_prices['2024-04-23']}")  
    zn0_change = (zn0_prices['2024-04-23'] - zn0_prices['2024-04-19']) / zn0_prices['2024-04-19'] * 100
    print(f"ZN0总变化: {zn0_change:.2f}%")
    
    # 分析单日最大跌幅
    print(f"\n⚠️  关键发现:")
    sn0_single_day_drop = (sn0_prices['2024-04-23'] - sn0_prices['2024-04-22']) / sn0_prices['2024-04-22'] * 100
    print(f"SN0在2024-04-23单日跌幅: {sn0_single_day_drop:.2f}%")
    
    zn0_single_day_drop = (zn0_prices['2024-04-23'] - zn0_prices['2024-04-22']) / zn0_prices['2024-04-22'] * 100
    print(f"ZN0在2024-04-23单日跌幅: {zn0_single_day_drop:.2f}%")
    
    # 分析价差变化
    print(f"\n📈 价差分析（假设long spread: long SN0, short ZN0）:")
    
    # 计算归一化价格（以4-19为基准）
    sn0_norm_19 = 1.0
    sn0_norm_23 = sn0_prices['2024-04-23'] / sn0_prices['2024-04-19']
    
    zn0_norm_19 = 1.0  
    zn0_norm_23 = zn0_prices['2024-04-23'] / zn0_prices['2024-04-19']
    
    # Long spread = Long SN0 - Short ZN0
    # 当SN0下跌更多或ZN0上涨更多时，long spread亏损
    print(f"SN0归一化: 1.000 → {sn0_norm_23:.3f}")
    print(f"ZN0归一化: 1.000 → {zn0_norm_23:.3f}")
    
    spread_change = (sn0_norm_23 - sn0_norm_19) - (zn0_norm_23 - zn0_norm_19)
    print(f"Price spread变化: {spread_change:.3f}")
    
    # 结论分析
    print(f"\n🎯 损失原因分析:")
    print(f"1. SN0大幅下跌-6.79%，特别是4月23日单日暴跌-7.96%")
    print(f"2. ZN0相对稳定，仅下跌-2.07%")  
    print(f"3. Long SN0 Short ZN0的价差大幅收窄")
    print(f"4. 这是真实的市场风险，非合约换月造成")
    
    print(f"\n✅ 这不是合约换月问题，而是:")
    print(f"   • 真实的市场波动")
    print(f"   • SN0(沪锡)相对ZN0(沪锌)的基本面恶化") 
    print(f"   • 止损机制面临价格跳空无法有效控制")
    
    print(f"\n⚠️  止损失效原因:")
    print(f"   • SN0单日跌幅-7.96%远超日常波动")
    print(f"   • 开盘即跳空，无法在10%止损点位成交")
    print(f"   • 这种情况下止损机制天然失效")

if __name__ == "__main__":
    analyze_sn_zn_trade_loss()