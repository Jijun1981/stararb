#!/usr/bin/env python3
"""
检查2024年4月SN0和ZN0合约换月情况
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def check_sn_zn_contracts_apr2024():
    """检查2024年4月SN0和ZN0的合约情况"""
    
    print("=== 2024年4月SN0/ZN0合约换月检查 ===\n")
    
    # 检查日期范围
    start_date = "20240415"
    end_date = "20240430"
    
    try:
        print("📊 检查SN0主力合约数据...")
        # 获取SN0主力合约数据
        sn_data = ak.futures_zh_daily_sina(symbol="SN0")
        print(f"SN0数据获取成功: {len(sn_data)}条记录")
        print("SN0价格变化:")
        print(sn_data[['date', 'close']].head(10))
        
        print("\n" + "="*50)
        
        print("\n📊 检查ZN0主力合约数据...")
        # 获取ZN0主力合约数据  
        zn_data = ak.futures_zh_daily_sina(symbol="ZN0")
        print(f"ZN0数据获取成功: {len(zn_data)}条记录")
        print("ZN0价格变化:")
        print(zn_data[['date', 'close']].head(10))
        
        # 检查价格跳跃
        print("\n🔍 价格跳跃分析:")
        for symbol, data in [("SN0", sn_data), ("ZN0", zn_data)]:
            print(f"\n{symbol} 每日价格变化:")
            data['price_change'] = data['close'].pct_change() * 100
            data['date_str'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            
            # 找出异常波动的日期
            large_changes = data[abs(data['price_change']) > 3.0]
            if not large_changes.empty:
                print(f"⚠️  {symbol} 异常波动日期:")
                for idx, row in large_changes.iterrows():
                    print(f"  {row['date_str']}: {row['price_change']:.2f}% (价格: {row['close']})")
            
            # 检查4月19-23日的具体价格
            target_data = data[
                (pd.to_datetime(data['date']) >= pd.Timestamp('2024-04-19')) &
                (pd.to_datetime(data['date']) <= pd.Timestamp('2024-04-23'))
            ]
            
            if not target_data.empty:
                print(f"\n{symbol} 交易期间 (2024-04-19 至 04-23):")
                for idx, row in target_data.iterrows():
                    change = row['price_change'] if pd.notna(row['price_change']) else 0
                    print(f"  {row['date_str']}: 收盘{row['close']} (变化{change:+.2f}%)")
                
                first_price = target_data.iloc[0]['close']
                last_price = target_data.iloc[-1]['close']
                total_change = (last_price - first_price) / first_price * 100
                print(f"  总变化: {first_price} → {last_price} ({total_change:+.2f}%)")
        
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        print("尝试备用方法...")
        
    print("\n" + "="*60)
    
    # 尝试获取具体合约信息
    try:
        print("\n📋 检查可用的SN和ZN合约...")
        
        # 获取期货品种信息
        try:
            futures_info = ak.futures_zh_spot()
            print(f"期货信息获取成功: {len(futures_info)}条记录")
            print("列名:", futures_info.columns.tolist())
            
            # 根据实际列名查找
            if '代码' in futures_info.columns:
                sn_contracts = futures_info[futures_info['代码'].str.contains('SN', na=False)]
                zn_contracts = futures_info[futures_info['代码'].str.contains('ZN', na=False)]
            else:
                print("可用列:", futures_info.columns.tolist())
                sn_contracts = pd.DataFrame()
                zn_contracts = pd.DataFrame()
                
        except Exception as e:
            print(f"期货信息获取失败: {e}")
            sn_contracts = pd.DataFrame()
            zn_contracts = pd.DataFrame()
        
        print("SN相关合约:")
        if not sn_contracts.empty:
            print(sn_contracts[['品种代码', '品种名称']].head())
        
        print("\nZN相关合约:")
        if not zn_contracts.empty:
            print(zn_contracts[['品种代码', '品种名称']].head())
            
    except Exception as e:
        print(f"合约信息获取失败: {e}")
    
    # 检查是否是合约换月导致的价格跳跃
    print("\n🎯 结论分析:")
    print("如果在2024-04-19到04-23期间出现:")
    print("1. 单日跌幅超过5%")  
    print("2. 价格出现不连续跳跃")
    print("3. 成交量异常")
    print("则很可能是合约换月导致的人为跳跃，而非真实市场风险")

if __name__ == "__main__":
    check_sn_zn_contracts_apr2024()