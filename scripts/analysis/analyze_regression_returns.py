#!/usr/bin/env python3
"""
分析样本外Z>2信号的回归收益
验证：Z>2信号 → 回归后收益 > 0
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_regression_returns():
    """分析样本外Z>2信号的回归收益"""
    
    print("📊 样本外Z>2信号回归收益分析")
    print("=" * 70)
    
    # 加载最新的优化参数信号
    try:
        # 找到最新的信号文件
        signal_files = [f for f in os.listdir('.') if f.startswith('signals_e2e_') and f.endswith('.csv')]
        latest_signal_file = max(signal_files) if signal_files else None
        
        if not latest_signal_file:
            print("❌ 未找到信号文件")
            return
            
        signals_df = pd.read_csv(latest_signal_file)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        print(f"使用信号文件: {latest_signal_file}")
    except Exception as e:
        print(f"❌ 加载信号文件失败: {e}")
        return
    
    # 加载价格数据
    data = load_all_symbols_data()
    
    print(f"信号数据: {len(signals_df)}条")
    print(f"分析期间: {signals_df['date'].min()} 至 {signals_df['date'].max()}")
    
    # 筛选信号期数据
    signal_period_df = signals_df[signals_df['phase'] == 'signal_period'].copy()
    print(f"信号期数据: {len(signal_period_df)}条")
    
    # 分析各配对
    pairs = signal_period_df['pair'].unique()
    results = []
    
    print(f"\n🔍 分配对分析 ({len(pairs)}个配对):")
    
    for pair in pairs[:10]:  # 分析前10个配对
        pair_data = signal_period_df[signal_period_df['pair'] == pair].copy()
        pair_data = pair_data.sort_values('date')
        
        if len(pair_data) < 100:
            continue
            
        # 获取Z>2的信号点
        z_gt2_signals = pair_data[np.abs(pair_data['z_score']) > 2.0].copy()
        
        if len(z_gt2_signals) < 5:
            print(f"  {pair}: Z>2信号太少({len(z_gt2_signals)}个)，跳过")
            continue
        
        print(f"\n=== {pair} ===")
        print(f"总信号点: {len(pair_data)}, Z>2信号: {len(z_gt2_signals)}个 ({len(z_gt2_signals)/len(pair_data)*100:.1f}%)")
        
        # 获取价格数据
        symbol_x = pair_data['symbol_x'].iloc[0]
        symbol_y = pair_data['symbol_y'].iloc[0]
        
        # 计算各Z>2信号点的后续回归收益
        forward_returns = []
        signal_types = []
        z_values = []
        
        for _, signal_row in z_gt2_signals.iterrows():
            signal_date = signal_row['date']
            z_score = signal_row['z_score']
            
            # 找到信号日期后的5个交易日的数据
            future_dates = pair_data[pair_data['date'] > signal_date]['date'].head(5)
            
            if len(future_dates) < 3:  # 至少需要3个后续点
                continue
                
            # 获取信号点和后续点的价格
            try:
                signal_x = data.loc[signal_date, symbol_x]
                signal_y = data.loc[signal_date, symbol_y]
                
                # 计算后续几天的回归收益
                daily_returns = []
                for future_date in future_dates:
                    try:
                        future_x = data.loc[future_date, symbol_x]
                        future_y = data.loc[future_date, symbol_y]
                        
                        # 计算价格变化
                        delta_x = future_x - signal_x
                        delta_y = future_y - signal_y
                        
                        # 根据Z-score方向计算回归收益
                        # Z<-2.0: 预期价差回归(收敛), long Y short X
                        # Z>+2.0: 预期价差回归(收敛), short Y long X
                        if z_score < -2.0:
                            # Long Y, Short X: 收益 = delta_y - beta*delta_x
                            regression_return = delta_y - signal_row['beta'] * delta_x
                        else:  # z_score > +2.0
                            # Short Y, Long X: 收益 = beta*delta_x - delta_y  
                            regression_return = signal_row['beta'] * delta_x - delta_y
                        
                        daily_returns.append(regression_return)
                        
                    except:
                        continue
                
                if daily_returns:
                    # 计算平均回归收益
                    avg_return = np.mean(daily_returns)
                    forward_returns.append(avg_return)
                    signal_types.append('long' if z_score < -2.0 else 'short')
                    z_values.append(abs(z_score))
                    
            except:
                continue
        
        if len(forward_returns) < 5:
            print(f"  可分析信号不足({len(forward_returns)}个)")
            continue
        
        forward_returns = np.array(forward_returns)
        z_values = np.array(z_values)
        
        # 统计分析
        positive_returns = np.sum(forward_returns > 0)
        negative_returns = np.sum(forward_returns < 0)
        zero_returns = len(forward_returns) - positive_returns - negative_returns
        
        mean_return = np.mean(forward_returns)
        std_return = np.std(forward_returns)
        
        # t检验：收益是否显著大于0
        t_stat, p_value = stats.ttest_1samp(forward_returns, 0)
        is_significant = p_value < 0.05 and mean_return > 0
        
        print(f"  Z>1.5信号分析:")
        print(f"    有效信号: {len(forward_returns)}个")
        print(f"    正收益: {positive_returns}个 ({positive_returns/len(forward_returns)*100:.1f}%)")
        print(f"    负收益: {negative_returns}个 ({negative_returns/len(forward_returns)*100:.1f}%)")
        print(f"    零收益: {zero_returns}个")
        print(f"  收益统计:")
        print(f"    平均收益: {mean_return:.4f}")
        print(f"    收益标准差: {std_return:.4f}")
        print(f"    收益t检验: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"    显著性: {'✅ 显著>0' if is_significant else '❌ 不显著' if mean_return > 0 else '❌ 负收益'}")
        
        # IR计算
        ir = mean_return / (std_return + 1e-8)
        print(f"    信息比率: {ir:.3f}")
        
        results.append({
            'pair': pair,
            'total_signals': len(z_gt2_signals),
            'valid_signals': len(forward_returns),
            'positive_ratio': positive_returns / len(forward_returns),
            'mean_return': mean_return,
            'std_return': std_return,
            't_stat': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'ir': ir
        })
    
    # 总结分析
    if results:
        print(f"\n📊 总体分析结果:")
        print("=" * 70)
        
        results_df = pd.DataFrame(results)
        
        # 整体统计
        total_valid_signals = results_df['valid_signals'].sum()
        positive_pairs = len(results_df[results_df['mean_return'] > 0])
        significant_pairs = len(results_df[results_df['is_significant']])
        
        print(f"分析配对数: {len(results_df)}个")
        print(f"总有效Z>1.5信号: {total_valid_signals}个")
        print(f"平均收益>0的配对: {positive_pairs}/{len(results_df)} ({positive_pairs/len(results_df)*100:.1f}%)")
        print(f"收益显著>0的配对: {significant_pairs}/{len(results_df)} ({significant_pairs/len(results_df)*100:.1f}%)")
        
        # 加权平均收益
        weights = results_df['valid_signals'].values
        weighted_avg_return = np.average(results_df['mean_return'].values, weights=weights)
        
        print(f"\n整体表现:")
        print(f"加权平均收益: {weighted_avg_return:.4f}")
        print(f"平均IR: {results_df['ir'].mean():.3f}")
        print(f"最佳配对IR: {results_df['ir'].max():.3f}")
        
        # 显示最好和最差的配对
        print(f"\n🏆 表现最佳配对:")
        best_pairs = results_df.nlargest(3, 'ir')
        for _, row in best_pairs.iterrows():
            print(f"  {row['pair']}: 平均收益={row['mean_return']:.4f}, IR={row['ir']:.3f}, "
                  f"胜率={row['positive_ratio']*100:.1f}%")
        
        print(f"\n⚠️ 需要关注配对:")
        worst_pairs = results_df.nsmallest(3, 'mean_return')
        for _, row in worst_pairs.iterrows():
            print(f"  {row['pair']}: 平均收益={row['mean_return']:.4f}, IR={row['ir']:.3f}, "
                  f"胜率={row['positive_ratio']*100:.1f}%")
        
        # 核心结论
        print(f"\n🎯 核心结论:")
        if weighted_avg_return > 0:
            print(f"✅ Z>1.5信号整体回归收益为正: {weighted_avg_return:.4f}")
        else:
            print(f"❌ Z>1.5信号整体回归收益为负: {weighted_avg_return:.4f}")
            
        if significant_pairs >= len(results_df) * 0.5:
            print(f"✅ 过半配对({significant_pairs}/{len(results_df)})收益显著大于0")
        else:
            print(f"⚠️ 仅{significant_pairs}/{len(results_df)}个配对收益显著大于0")
        
        return results_df
    else:
        print("❌ 没有足够数据进行分析")
        return None

if __name__ == "__main__":
    results = analyze_regression_returns()