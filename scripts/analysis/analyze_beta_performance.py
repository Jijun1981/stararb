#!/usr/bin/env python3
"""
分析Beta值与交易表现的关系

统计不同Beta范围的交易表现，验证手数计算的合理性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 配置
plt.style.use('default')

def analyze_beta_performance(trades_file: str = None):
    """分析Beta值与交易表现"""
    
    # 如果没有指定文件，使用最新的交易记录
    if trades_file is None:
        output_dir = Path("output/pipeline_v21")
        trades_files = list(output_dir.glob("trades_*.csv"))
        if not trades_files:
            print("未找到交易记录文件")
            return
        trades_file = max(trades_files, key=lambda x: x.stat().st_mtime)
    
    print(f"分析文件: {trades_file}")
    
    # 加载交易数据
    trades_df = pd.read_csv(trades_file)
    
    print("\n" + "=" * 80)
    print("Beta值分析报告")
    print("=" * 80)
    
    # 1. Beta值分布
    print("\n1. Beta值分布统计:")
    print("-" * 40)
    
    beta_stats = trades_df['beta'].describe()
    print(f"交易数量: {len(trades_df)}")
    print(f"Beta均值: {beta_stats['mean']:.4f}")
    print(f"Beta中位数: {beta_stats['50%']:.4f}")
    print(f"Beta标准差: {beta_stats['std']:.4f}")
    print(f"Beta范围: [{beta_stats['min']:.4f}, {beta_stats['max']:.4f}]")
    
    # 2. Beta分组分析
    print("\n2. Beta分组表现:")
    print("-" * 40)
    
    # 将Beta分为几个区间
    beta_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, float('inf')]
    beta_labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '>3.0']
    trades_df['beta_group'] = pd.cut(trades_df['beta'].abs(), bins=beta_bins, labels=beta_labels)
    
    # 统计每个Beta组的表现
    group_stats = []
    for group in beta_labels:
        group_data = trades_df[trades_df['beta_group'] == group]
        if len(group_data) > 0:
            stats = {
                'Beta范围': group,
                '交易数': len(group_data),
                '胜率%': (group_data['net_pnl'] > 0).mean() * 100,
                '平均盈亏': group_data['net_pnl'].mean(),
                '总盈亏': group_data['net_pnl'].sum(),
                '平均持仓天数': group_data['holding_days'].mean()
            }
            group_stats.append(stats)
    
    group_df = pd.DataFrame(group_stats)
    print(group_df.to_string(index=False))
    
    # 3. 手数比率准确性分析
    print("\n3. 手数比率准确性:")
    print("-" * 40)
    
    # 计算实际手数比率
    trades_df['actual_ratio'] = trades_df['contracts_y'] / trades_df['contracts_x']
    trades_df['ratio_error'] = abs(trades_df['actual_ratio'] - trades_df['beta'].abs()) / trades_df['beta'].abs() * 100
    
    print(f"平均比率误差: {trades_df['ratio_error'].mean():.2f}%")
    print(f"中位数比率误差: {trades_df['ratio_error'].median():.2f}%")
    print(f"最大比率误差: {trades_df['ratio_error'].max():.2f}%")
    print(f"误差<10%的交易占比: {(trades_df['ratio_error'] < 10).mean() * 100:.1f}%")
    print(f"误差<20%的交易占比: {(trades_df['ratio_error'] < 20).mean() * 100:.1f}%")
    
    # 4. Beta与盈亏的相关性
    print("\n4. Beta与盈亏相关性:")
    print("-" * 40)
    
    correlation = trades_df[['beta', 'net_pnl', 'holding_days', 'ratio_error']].corr()
    print("相关性矩阵:")
    print(correlation.round(3))
    
    # 5. 配对统计
    print("\n5. 配对Beta统计:")
    print("-" * 40)
    
    pair_stats = trades_df.groupby('pair').agg({
        'beta': ['mean', 'std', 'count'],
        'net_pnl': ['sum', 'mean'],
        'holding_days': 'mean'
    }).round(2)
    
    # 展平列名
    pair_stats.columns = ['_'.join(col).strip() for col in pair_stats.columns.values]
    pair_stats = pair_stats.sort_values('net_pnl_sum', ascending=False)
    
    print("前10个盈利配对:")
    print(pair_stats.head(10))
    
    print("\n后10个亏损配对:")
    print(pair_stats.tail(10))
    
    # 6. 生成可视化报告
    output_dir = Path("output/beta_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Beta分布直方图
    axes[0, 0].hist(trades_df['beta'].abs(), bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('|Beta|')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('Beta值分布')
    
    # Beta vs 盈亏散点图
    colors = ['green' if x > 0 else 'red' for x in trades_df['net_pnl']]
    axes[0, 1].scatter(trades_df['beta'].abs(), trades_df['net_pnl'], c=colors, alpha=0.6)
    axes[0, 1].set_xlabel('|Beta|')
    axes[0, 1].set_ylabel('净盈亏')
    axes[0, 1].set_title('Beta vs 盈亏')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 比率误差分布
    axes[0, 2].hist(trades_df['ratio_error'], bins=30, edgecolor='black')
    axes[0, 2].set_xlabel('比率误差(%)')
    axes[0, 2].set_ylabel('频数')
    axes[0, 2].set_title('手数比率误差分布')
    
    # Beta分组胜率
    if len(group_df) > 0:
        axes[1, 0].bar(group_df['Beta范围'], group_df['胜率%'])
        axes[1, 0].set_xlabel('Beta范围')
        axes[1, 0].set_ylabel('胜率(%)')
        axes[1, 0].set_title('不同Beta范围的胜率')
        axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.3)
    
    # Beta分组平均盈亏
    if len(group_df) > 0:
        colors = ['green' if x > 0 else 'red' for x in group_df['平均盈亏']]
        axes[1, 1].bar(group_df['Beta范围'], group_df['平均盈亏'], color=colors)
        axes[1, 1].set_xlabel('Beta范围')
        axes[1, 1].set_ylabel('平均盈亏')
        axes[1, 1].set_title('不同Beta范围的平均盈亏')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 实际比率 vs 理论Beta
    axes[1, 2].scatter(trades_df['beta'].abs(), trades_df['actual_ratio'], alpha=0.6)
    axes[1, 2].plot([0, 3], [0, 3], 'r--', label='理想线')
    axes[1, 2].set_xlabel('理论|Beta|')
    axes[1, 2].set_ylabel('实际手数比率')
    axes[1, 2].set_title('理论Beta vs 实际手数比率')
    axes[1, 2].legend()
    axes[1, 2].set_xlim(0, 3)
    axes[1, 2].set_ylim(0, 3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_file = output_dir / f"beta_analysis_{timestamp}.png"
    plt.savefig(chart_file, dpi=100, bbox_inches='tight')
    print(f"\n图表已保存至: {chart_file}")
    
    # 7. 保存详细报告
    report = {
        'analysis_time': timestamp,
        'total_trades': len(trades_df),
        'beta_statistics': {
            'mean': float(beta_stats['mean']),
            'median': float(beta_stats['50%']),
            'std': float(beta_stats['std']),
            'min': float(beta_stats['min']),
            'max': float(beta_stats['max'])
        },
        'ratio_accuracy': {
            'mean_error': float(trades_df['ratio_error'].mean()),
            'median_error': float(trades_df['ratio_error'].median()),
            'max_error': float(trades_df['ratio_error'].max()),
            'error_below_10pct': float((trades_df['ratio_error'] < 10).mean()),
            'error_below_20pct': float((trades_df['ratio_error'] < 20).mean())
        },
        'group_performance': group_df.to_dict('records') if len(group_df) > 0 else [],
        'top_pairs': pair_stats.head(10).to_dict() if len(pair_stats) > 0 else {}
    }
    
    report_file = output_dir / f"beta_analysis_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"详细报告已保存至: {report_file}")
    
    return trades_df, group_df

if __name__ == "__main__":
    trades_df, group_df = analyze_beta_performance()
    print("\n分析完成!")