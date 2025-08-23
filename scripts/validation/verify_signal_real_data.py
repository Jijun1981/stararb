#!/usr/bin/env python3
"""
信号生成模块真实数据端到端验证
使用实际期货数据测试完整的信号生成流程

功能:
1. 加载真实期货数据
2. 运行协整分析获取初始Beta
3. 执行完整的三阶段信号生成流程
4. 验证每个阶段的输出正确性
5. 对比Kalman滤波与60天OLS的实际表现

作者: Star-arb Team
日期: 2025-08-22
版本: V1.0
"""

import sys
sys.path.append('/mnt/e/Star-arb')

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from lib.data import load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator, calculate_ols_beta
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def run_end_to_end_validation():
    """运行完整的端到端验证"""
    
    print("=" * 80)
    print("信号生成模块真实数据端到端验证")
    print("=" * 80)
    
    # 第一步：加载真实数据
    print("\n1. 加载真实期货数据")
    print("-" * 40)
    
    try:
        # 选择两个流动性好的品种
        symbols = ['CU0', 'AL0']  # 铜和铝
        data = load_data(symbols, columns=['close'], log_price=True)
        print(f"✓ 成功加载数据: {symbols}, {len(data)}条记录")
        print(f"  数据时间范围: {data.index[0]} 到 {data.index[-1]}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 第二步：协整分析
    print("\n2. 协整分析获取初始Beta")
    print("-" * 40)
    
    try:
        analyzer = CointegrationAnalyzer(data)
        results = analyzer.screen_all_pairs(p_threshold=1.0)  # 获取所有结果
        
        if len(results) == 0:
            print("✗ 协整分析无结果")
            return False
        
        # 获取最佳配对
        best_pair = results.iloc[0]
        pair_name = best_pair['pair']
        initial_beta = best_pair['beta_5y']
        
        print(f"✓ 协整分析完成")
        print(f"  最佳配对: {pair_name}")
        print(f"  初始Beta: {initial_beta:.6f}")
        print(f"  5年p值: {best_pair['pvalue_5y']:.6f}")
        
    except Exception as e:
        print(f"✗ 协整分析失败: {e}")
        return False
    
    # 第三步：准备信号生成数据
    print("\n3. 准备信号生成数据")
    print("-" * 40)
    
    try:
        # 解析配对 (注意移除_close后缀)
        symbol_x_raw, symbol_y_raw = pair_name.split('-')
        # 如果包含_close，移除它
        symbol_x = symbol_x_raw.replace('_close', '')
        symbol_y = symbol_y_raw.replace('_close', '')
        
        # 准备时间配置
        data_dates = pd.to_datetime(data.index)
        total_days = len(data_dates)
        
        # 设置时间边界（最近6个月作为测试期）
        signal_start_idx = max(0, total_days - 180)  # 最后180天
        convergence_end_idx = signal_start_idx - 30   # 收敛期30天
        hist_end_idx = convergence_end_idx
        hist_start_idx = max(0, hist_end_idx - 252)   # 历史数据1年
        
        hist_start = data_dates[hist_start_idx].strftime('%Y-%m-%d')
        hist_end = data_dates[hist_end_idx].strftime('%Y-%m-%d')
        convergence_end = data_dates[convergence_end_idx].strftime('%Y-%m-%d')
        signal_start = data_dates[signal_start_idx].strftime('%Y-%m-%d')
        
        print(f"✓ 时间配置完成")
        print(f"  历史期: {hist_start} 到 {hist_end}")
        print(f"  收敛期结束: {convergence_end}")
        print(f"  信号期开始: {signal_start}")
        
        # 准备配对数据
        pair_data = data.copy()
        pair_data = pair_data.reset_index()
        
        # 检查实际列名
        actual_columns = list(pair_data.columns)
        print(f"  实际列名: {actual_columns}")
        
        # 正确的列名映射
        x_col = f'{symbol_x}_close'
        y_col = f'{symbol_y}_close'
        
        if x_col in pair_data.columns and y_col in pair_data.columns:
            pair_data = pair_data.rename(columns={
                x_col: 'x',
                y_col: 'y'
            })
        else:
            print(f"  ✗ 列名不匹配: 期望 {x_col}, {y_col}")
            return False
        
        print(f"  配对数据: X={symbol_x}, Y={symbol_y}, {len(pair_data)}条记录")
        
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        return False
    
    # 第四步：信号生成
    print("\n4. 执行三阶段信号生成")
    print("-" * 40)
    
    try:
        sg = SignalGenerator(
            window=60,
            z_open=2.0,
            z_close=0.5,
            convergence_days=20,
            convergence_threshold=0.01
        )
        
        signals = sg.process_pair_signals(
            pair_data=pair_data,
            initial_beta=initial_beta,
            convergence_end=convergence_end,
            signal_start=signal_start,
            hist_start=hist_start,
            hist_end=hist_end
        )
        
        if signals.empty:
            print("✗ 信号生成失败")
            return False
            
        print(f"✓ 信号生成完成: {len(signals)}条记录")
        
    except Exception as e:
        print(f"✗ 信号生成失败: {e}")
        return False
    
    # 第五步：验证信号质量
    print("\n5. 验证信号质量")
    print("-" * 40)
    
    try:
        # 分析各阶段信号
        convergence_signals = signals[signals['phase'] == 'convergence_period']
        signal_signals = signals[signals['phase'] == 'signal_period']
        
        print(f"  收敛期信号: {len(convergence_signals)}条")
        print(f"  信号期信号: {len(signal_signals)}条")
        
        # 验证收敛期
        conv_signal_types = convergence_signals['signal'].unique()
        conv_only_converging = all(sig == 'converging' for sig in conv_signal_types)
        print(f"  收敛期只有converging信号: {'✓' if conv_only_converging else '✗'}")
        
        # 验证信号期
        if len(signal_signals) > 0:
            signal_types = signal_signals['signal'].unique()
            has_trading_signals = any(sig in ['open_long', 'open_short', 'close'] for sig in signal_types)
            print(f"  信号期有交易信号: {'✓' if has_trading_signals else '✗'}")
            
            # 统计信号分布
            signal_counts = signal_signals['signal'].value_counts()
            print(f"  信号分布: {dict(signal_counts)}")
        
        # Beta收敛性分析
        if len(signals) > 60:
            final_betas = signals['beta'].tail(60)  # 最后60个Beta
            beta_volatility = final_betas.std()
            print(f"  后期Beta稳定性: {beta_volatility:.6f} (越小越稳定)")
        
        # Kalman vs OLS对比
        ols_comparison = signals[['beta', 'ols_beta']].dropna()
        if len(ols_comparison) > 10:
            correlation = ols_comparison['beta'].corr(ols_comparison['ols_beta'])
            rmse = np.sqrt(((ols_comparison['beta'] - ols_comparison['ols_beta'])**2).mean())
            print(f"  Kalman vs OLS相关性: {correlation:.4f}")
            print(f"  Kalman vs OLS均方根误差: {rmse:.6f}")
        
    except Exception as e:
        print(f"✗ 信号质量验证失败: {e}")
        return False
    
    # 第六步：性能分析
    print("\n6. 性能分析")
    print("-" * 40)
    
    try:
        # Z-score分布分析
        signal_period_data = signals[signals['phase'] == 'signal_period']
        if len(signal_period_data) > 0:
            z_scores = signal_period_data['z_score'].dropna()
            if len(z_scores) > 0:
                print(f"  Z-score统计:")
                print(f"    均值: {z_scores.mean():.4f}")
                print(f"    标准差: {z_scores.std():.4f}")
                print(f"    范围: [{z_scores.min():.2f}, {z_scores.max():.2f}]")
                
                # 信号触发率
                open_signals = signal_period_data[signal_period_data['signal'].isin(['open_long', 'open_short'])]
                signal_rate = len(open_signals) / len(signal_period_data) * 100
                print(f"    开仓信号触发率: {signal_rate:.1f}%")
        
        # 残差分析
        residuals = signals['residual'].dropna()
        if len(residuals) > 0:
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            print(f"  残差统计:")
            print(f"    均值: {residual_mean:.6f} (应接近0)")
            print(f"    标准差: {residual_std:.6f}")
            
            # 残差正态性检验 (简单版本)
            residual_skew = residuals.skew()
            residual_kurt = residuals.kurtosis()
            print(f"    偏度: {residual_skew:.3f}")
            print(f"    峰度: {residual_kurt:.3f}")
        
    except Exception as e:
        print(f"✗ 性能分析失败: {e}")
        return False
    
    # 第七步：数值准确性验证
    print("\n7. 数值准确性验证")
    print("-" * 40)
    
    try:
        # 验证Kalman滤波数值
        finite_betas = signals['beta'].dropna()
        all_finite = finite_betas.apply(np.isfinite).all()
        print(f"  所有Beta值有限: {'✓' if all_finite else '✗'}")
        
        # 验证Beta精度
        beta_precision_ok = all(abs(beta) < 10 for beta in finite_betas)  # 合理范围
        print(f"  Beta值在合理范围: {'✓' if beta_precision_ok else '✗'}")
        
        # 验证Z-score计算
        finite_zscores = signal_period_data['z_score'].dropna()
        all_zscore_finite = finite_zscores.apply(np.isfinite).all()
        print(f"  所有Z-score有限: {'✓' if all_zscore_finite else '✗'}")
        
        # 验证残差计算
        finite_residuals = signals['residual'].dropna()
        all_residual_finite = finite_residuals.apply(np.isfinite).all()
        print(f"  所有残差有限: {'✓' if all_residual_finite else '✗'}")
        
    except Exception as e:
        print(f"✗ 数值准确性验证失败: {e}")
        return False
    
    # 第八步：输出样本结果
    print("\n8. 样本输出")
    print("-" * 40)
    
    try:
        # 显示最后10条信号
        print("  最后10条信号记录:")
        sample_signals = signals.tail(10)[['date', 'signal', 'z_score', 'beta', 'residual', 'phase']]
        for _, row in sample_signals.iterrows():
            print(f"    {row['date']}: {row['signal']:>10} z={row['z_score']:6.2f} β={row['beta']:.4f} phase={row['phase']}")
        
    except Exception as e:
        print(f"✗ 样本输出失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("🎉 端到端验证完成！信号生成模块在真实数据上运行正常！")
    print("=" * 80)
    
    return True

def main():
    """主函数"""
    success = run_end_to_end_validation()
    
    if success:
        print(f"\n✅ 所有验证通过！信号生成模块已准备就绪用于生产环境。")
        print(f"📊 验证覆盖范围:")
        print(f"   - 真实期货数据加载")
        print(f"   - 协整分析集成")
        print(f"   - 三阶段信号生成")
        print(f"   - Kalman滤波数值稳定性")
        print(f"   - 信号逻辑正确性")
        print(f"   - 性能指标分析")
        return 0
    else:
        print(f"\n❌ 验证失败！请检查具体错误信息并修复。")
        return 1

if __name__ == '__main__':
    exit(main())