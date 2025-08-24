#!/usr/bin/env python3
"""
调试Z-score计算
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.data import load_symbol_data
from lib.signal_generation import SignalGenerator

def main():
    print("=" * 60)
    print("调试Z-score计算")
    print("=" * 60)
    
    # 加载数据
    au_data = load_symbol_data('AU')
    zn_data = load_symbol_data('ZN')
    
    # 只看一小段数据
    start_idx = 1100  # 大概2024年7月的数据
    end_idx = 1150   # 50天的数据
    
    au_prices = au_data['close'].iloc[start_idx:end_idx].values
    zn_prices = zn_data['close'].iloc[start_idx:end_idx].values
    dates = au_data.index[start_idx:end_idx]
    
    print(f"数据范围: {dates[0]} 至 {dates[-1]}")
    print(f"AU价格范围: {au_prices.min():.2f} - {au_prices.max():.2f}")
    print(f"ZN价格范围: {zn_prices.min():.2f} - {zn_prices.max():.2f}")
    
    # 使用协整结果中的beta
    beta = -0.306369  # AU-ZN的beta
    print(f"使用beta: {beta}")
    
    # 计算残差
    residuals = []
    for i in range(len(au_prices)):
        # 残差 = log(y) - beta * log(x)
        residual = np.log(zn_prices[i]) - beta * np.log(au_prices[i])
        residuals.append(residual)
    
    residuals = np.array(residuals)
    print(f"残差范围: {residuals.min():.4f} - {residuals.max():.4f}")
    print(f"残差均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")
    
    # 测试SignalGenerator的calculate_zscore方法
    sg = SignalGenerator()
    window = 30
    
    print(f"\n逐日计算Z-score (窗口={window}):")
    print("日期            残差      Z-score")
    print("-" * 35)
    
    for i in range(window, len(residuals)):
        z_score = sg.calculate_zscore(residuals[:i+1], window)
        print(f"{dates[i].strftime('%Y-%m-%d')} {residuals[i]:8.4f} {z_score:8.4f}")
        
        # 手动验证计算
        window_residuals = residuals[i+1-window:i+1]
        manual_mean = np.mean(window_residuals)
        manual_std = np.std(window_residuals, ddof=1)
        manual_zscore = (residuals[i] - manual_mean) / manual_std
        
        if i == len(residuals) - 1:  # 最后一个点详细验证
            print(f"\n最后一个点详细验证:")
            print(f"  窗口数据: {window_residuals}")
            print(f"  窗口均值: {manual_mean:.6f}")
            print(f"  窗口标准差: {manual_std:.6f}") 
            print(f"  当前残差: {residuals[i]:.6f}")
            print(f"  手动计算Z-score: {manual_zscore:.6f}")
            print(f"  方法计算Z-score: {z_score:.6f}")
            
            # 检查是否达到开仓阈值
            if abs(z_score) >= 2.2:
                signal_type = 'open_long' if z_score <= -2.2 else 'open_short'
                print(f"  应该产生信号: {signal_type}")
            else:
                print(f"  未达到开仓阈值2.2")

if __name__ == '__main__':
    main()