#!/usr/bin/env python3
"""
端到端配对交易流程 - 清晰版本
严格按照各模块的实际接口定义
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入核心模块 - 使用实际存在的接口
from lib.data import load_symbol_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator
from lib.backtest_v4 import BacktestEngine, BacktestConfig


def main():
    """主函数 - 清晰的端到端流程"""
    
    print("=" * 80)
    print("端到端配对交易流程 - 清晰版本")
    print("=" * 80)
    
    # ========== 1. 参数配置 ==========
    config = {
        'symbols': ['AG0', 'AU0', 'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0', 
                   'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'],
        'beta_train_start': '2023-01-01',
        'beta_train_end': '2023-12-31',
        'signal_start': '2024-07-01',
        'signal_end': '2025-08-20',
        'z_open': 2.0,  # 降低阈值以获得更多信号
        'z_close': 0.5,
        'beta_min': 0.3,
        'beta_max': 3.0
    }
    
    # ========== 2. 数据加载 ==========
    print("\n步骤1: 加载数据")
    print("-" * 40)
    
    data = {}
    for symbol in config['symbols']:
        try:
            # 去掉末尾的0
            file_symbol = symbol.rstrip('0')
            df = load_symbol_data(file_symbol)
            if df is not None and len(df) > 0:
                data[symbol] = df
                print(f"✓ {symbol}: {len(df)}条")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
    
    print(f"成功加载: {len(data)}/{len(config['symbols'])}个品种")
    
    if len(data) < 2:
        print("数据不足，退出")
        return
    
    # ========== 3. 协整筛选 ==========
    print("\n步骤2: 协整筛选")
    print("-" * 40)
    
    # 准备价格数据
    prices = pd.DataFrame()
    for symbol, df in data.items():
        prices[symbol] = df['close']
    
    # 对齐时间
    prices = prices.dropna(how='all').fillna(method='ffill')
    
    # 创建协整分析器
    log_prices = np.log(prices)
    analyzer = CointegrationAnalyzer(log_prices)
    
    # 筛选配对 - 简单直接的方法
    pairs = []
    symbols = list(data.keys())
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            x_symbol = symbols[i]
            y_symbol = symbols[j]
            
            # 获取5年数据
            end_date = prices.index[-1]
            start_5y = end_date - pd.DateOffset(years=5)
            
            data_5y = prices.loc[start_5y:end_date]
            if x_symbol in data_5y.columns and y_symbol in data_5y.columns:
                x_5y = data_5y[x_symbol].dropna().values
                y_5y = data_5y[y_symbol].dropna().values
                
                if len(x_5y) > 252 and len(y_5y) > 252:
                    # 协整检验
                    result = analyzer.engle_granger_test(x_5y, y_5y)
                    
                    if result['pvalue'] < 0.05:
                        # 确定方向（低波动作X）
                        vol_x = np.std(x_5y / x_5y[0])
                        vol_y = np.std(y_5y / y_5y[0])
                        
                        if vol_x < vol_y:
                            pair_name = f"{x_symbol}-{y_symbol}"
                            beta = result['beta']
                        else:
                            pair_name = f"{y_symbol}-{x_symbol}"
                            beta = 1.0 / result['beta'] if result['beta'] != 0 else 1.0
                        
                        pairs.append({
                            'pair': pair_name,
                            'x': pair_name.split('-')[0],
                            'y': pair_name.split('-')[1],
                            'pvalue': result['pvalue'],
                            'beta_ols': beta
                        })
    
    print(f"筛选出: {len(pairs)}对协整配对")
    for i, p in enumerate(pairs[:5]):
        print(f"  {i+1}. {p['pair']}: p={p['pvalue']:.4f}, β={p['beta_ols']:.4f}")
    
    if len(pairs) == 0:
        print("没有协整配对，退出")
        return
    
    # ========== 4. Beta估计（使用训练期） ==========
    print("\n步骤3: Beta估计")
    print("-" * 40)
    
    pairs_with_beta = []
    for pair in pairs:
        x_symbol = pair['x']
        y_symbol = pair['y']
        
        # 获取训练期数据
        x_train = data[x_symbol]['close'].loc[config['beta_train_start']:config['beta_train_end']]
        y_train = data[y_symbol]['close'].loc[config['beta_train_start']:config['beta_train_end']]
        
        # 对齐
        train_data = pd.DataFrame({'x': x_train, 'y': y_train}).dropna()
        
        if len(train_data) > 100:
            # 重新估计Beta
            from scipy import stats
            slope, _, _, _, _ = stats.linregress(
                np.log(train_data['x']), 
                np.log(train_data['y'])
            )
            
            # Beta约束检查
            if config['beta_min'] <= abs(slope) <= config['beta_max']:
                pair['beta'] = slope
                pairs_with_beta.append(pair)
                print(f"  {pair['pair']}: β={slope:.4f} ✓")
            else:
                print(f"  {pair['pair']}: β={slope:.4f} (超出范围)")
    
    print(f"通过Beta约束: {len(pairs_with_beta)}/{len(pairs)}")
    
    if len(pairs_with_beta) == 0:
        print("没有配对通过Beta约束，退出")
        return
    
    # ========== 5. 信号生成 ==========
    print("\n步骤4: 信号生成")
    print("-" * 40)
    
    # 准备价格数据（需要date列）
    all_prices = prices.reset_index()
    all_prices.rename(columns={'index': 'date'}, inplace=True)
    
    # 准备配对参数
    pairs_params = {}
    for pair in pairs_with_beta:
        pairs_params[pair['pair']] = {
            'x': pair['x'],
            'y': pair['y'],
            'beta': pair['beta'],
            'beta_initial': pair['beta']  # SignalGenerator需要这个
        }
    
    # 创建信号生成器
    generator = SignalGenerator(
        window=60,
        z_open=config['z_open'],
        z_close=config['z_close']
    )
    
    # 生成信号
    try:
        signals = generator.generate_all_signals(
            pairs_params=pairs_params,
            price_data=all_prices,
            convergence_end=config['signal_start'],
            signal_start=config['signal_start'],
            hist_start='2024-01-01',
            hist_end=config['signal_end']
        )
        
        if signals is not None and len(signals) > 0:
            print(f"生成信号: {len(signals)}个")
            print("信号统计:")
            for sig, count in signals['signal'].value_counts().items():
                print(f"  {sig}: {count}")
            
            # 确保必需字段
            if 'x_symbol' not in signals.columns:
                signals['x_symbol'] = signals['pair'].apply(lambda x: x.split('-')[0])
            if 'y_symbol' not in signals.columns:
                signals['y_symbol'] = signals['pair'].apply(lambda x: x.split('-')[1])
            if 'position' not in signals.columns:
                signals['position'] = 0
            
            # 添加缺失的必需字段
            required_fields = ['zscore', 'spread', 'x_price', 'y_price', 'spread_mean', 'spread_std']
            for field in required_fields:
                if field not in signals.columns:
                    if field == 'zscore' and 'z_score' in signals.columns:
                        signals['zscore'] = signals['z_score']
                    elif field == 'spread' and 'residual' in signals.columns:
                        signals['spread'] = signals['residual']
                    else:
                        signals[field] = 0  # 临时填充
        else:
            print("未生成信号")
            return
            
    except Exception as e:
        print(f"信号生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 6. 回测 ==========
    print("\n步骤5: 回测")
    print("-" * 40)
    
    # 创建回测引擎
    backtest_config = BacktestConfig(
        initial_capital=5000000,
        margin_rate=0.12,
        commission_rate=0.0002,
        slippage_ticks=3,
        stop_loss_pct=0.15,
        max_holding_days=30
    )
    
    engine = BacktestEngine(backtest_config)
    
    # 加载合约规格
    contract_specs_path = project_root / 'configs' / 'contract_specs.json'
    if contract_specs_path.exists():
        engine.load_contract_specs(str(contract_specs_path))
    
    # 准备价格数据
    price_df = pd.DataFrame()
    for symbol in data.keys():
        price_df[symbol] = data[symbol]['close']
    
    # 运行回测
    try:
        metrics = engine.run_backtest(signals, price_df)
        
        print(f"总交易: {metrics.get('total_trades', 0)}笔")
        print(f"总收益: {metrics.get('total_pnl', 0):,.0f}")
        print(f"收益率: {metrics.get('total_return', 0):.2%}")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("流程完成")
    print("=" * 80)


if __name__ == '__main__':
    main()