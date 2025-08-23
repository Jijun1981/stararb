#!/usr/bin/env python3
"""
比较不同筛选模式的效果

演示如何使用灵活的协整筛选参数：
1. 仅使用p值筛选（默认）
2. p值 + 半衰期筛选
3. 不同的p值阈值
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 品种列表
SYMBOLS = [
    'AG0', 'AU0', 'AL0', 'CU0', 'NI0', 'PB0', 'SN0', 'ZN0',
    'HC0', 'I0', 'RB0', 'SF0', 'SM0', 'SS0'
]

def compare_screening_modes():
    """比较不同筛选模式的结果"""
    
    logger.info("=" * 80)
    logger.info("协整筛选模式对比分析")
    logger.info("=" * 80)
    
    # 1. 加载数据
    logger.info("\n1. 加载历史数据...")
    data = load_data(
        symbols=SYMBOLS,
        start_date='2020-01-01',
        columns=['close'],
        log_price=True
    )
    
    # 2. 创建分析器
    logger.info("\n2. 创建协整分析器...")
    analyzer = CointegrationAnalyzer(data)
    
    # 3. 模式1：仅p值筛选（默认模式）
    logger.info("\n" + "=" * 60)
    logger.info("模式1: 仅P值筛选 (p < 0.05)")
    logger.info("=" * 60)
    
    mode1_results = analyzer.screen_all_pairs(
        p_threshold=0.05,
        use_halflife_filter=False  # 不使用半衰期筛选
    )
    
    logger.info(f"通过筛选的配对数: {len(mode1_results)}")
    if len(mode1_results) > 0:
        logger.info("前5个配对:")
        for i, row in mode1_results.head(5).iterrows():
            logger.info(f"  {row['pair']}: 1年p={row['pvalue_1y']:.4f}, "
                       f"5年p={row['pvalue_5y']:.4f}, "
                       f"半衰期={row.get('halflife_5y', np.nan):.1f}天")
    
    # 4. 模式2：p值 + 半衰期筛选
    logger.info("\n" + "=" * 60)
    logger.info("模式2: P值 + 半衰期筛选 (p < 0.05, 半衰期 ∈ [5, 60])")
    logger.info("=" * 60)
    
    mode2_results = analyzer.screen_all_pairs(
        p_threshold=0.05,
        halflife_min=5,
        halflife_max=60,
        use_halflife_filter=True  # 启用半衰期筛选
    )
    
    logger.info(f"通过筛选的配对数: {len(mode2_results)}")
    if len(mode2_results) > 0:
        logger.info("前5个配对:")
        for i, row in mode2_results.head(5).iterrows():
            logger.info(f"  {row['pair']}: 1年p={row['pvalue_1y']:.4f}, "
                       f"5年p={row['pvalue_5y']:.4f}, "
                       f"半衰期={row.get('halflife_5y', np.nan):.1f}天")
    
    # 5. 模式3：更严格的p值筛选
    logger.info("\n" + "=" * 60)
    logger.info("模式3: 更严格的P值筛选 (p < 0.01)")
    logger.info("=" * 60)
    
    mode3_results = analyzer.screen_all_pairs(
        p_threshold=0.01,  # 更严格的p值
        use_halflife_filter=False
    )
    
    logger.info(f"通过筛选的配对数: {len(mode3_results)}")
    if len(mode3_results) > 0:
        logger.info("前5个配对:")
        for i, row in mode3_results.head(5).iterrows():
            logger.info(f"  {row['pair']}: 1年p={row['pvalue_1y']:.4f}, "
                       f"5年p={row['pvalue_5y']:.4f}, "
                       f"半衰期={row.get('halflife_5y', np.nan):.1f}天")
    
    # 6. 模式4：仅半衰期筛选（宽松p值）
    logger.info("\n" + "=" * 60)
    logger.info("模式4: 重点关注半衰期 (p < 0.1, 半衰期 ∈ [10, 30])")
    logger.info("=" * 60)
    
    mode4_results = analyzer.screen_all_pairs(
        p_threshold=0.1,  # 较宽松的p值
        halflife_min=10,
        halflife_max=30,  # 较严格的半衰期范围
        use_halflife_filter=True
    )
    
    logger.info(f"通过筛选的配对数: {len(mode4_results)}")
    if len(mode4_results) > 0:
        logger.info("前5个配对:")
        for i, row in mode4_results.head(5).iterrows():
            logger.info(f"  {row['pair']}: 1年p={row['pvalue_1y']:.4f}, "
                       f"5年p={row['pvalue_5y']:.4f}, "
                       f"半衰期={row.get('halflife_5y', np.nan):.1f}天")
    
    # 7. 汇总对比
    logger.info("\n" + "=" * 80)
    logger.info("筛选结果对比汇总")
    logger.info("=" * 80)
    
    summary = pd.DataFrame({
        '筛选模式': [
            '仅P值 (p<0.05)',
            'P值+半衰期 (p<0.05, HL∈[5,60])',
            '严格P值 (p<0.01)',
            '半衰期优先 (p<0.1, HL∈[10,30])'
        ],
        '配对数量': [
            len(mode1_results),
            len(mode2_results),
            len(mode3_results),
            len(mode4_results)
        ]
    })
    
    logger.info("\n" + summary.to_string(index=False))
    
    # 8. 保存结果
    output_dir = Path("output/screening_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存各模式结果
    mode1_results.to_csv(output_dir / f"mode1_pvalue_only_{timestamp}.csv", index=False)
    mode2_results.to_csv(output_dir / f"mode2_pvalue_halflife_{timestamp}.csv", index=False)
    mode3_results.to_csv(output_dir / f"mode3_strict_pvalue_{timestamp}.csv", index=False)
    mode4_results.to_csv(output_dir / f"mode4_halflife_focus_{timestamp}.csv", index=False)
    
    # 保存汇总
    summary.to_csv(output_dir / f"screening_summary_{timestamp}.csv", index=False)
    
    logger.info(f"\n结果已保存至: {output_dir}")
    
    # 9. 分析配对重叠
    logger.info("\n" + "=" * 60)
    logger.info("配对重叠分析")
    logger.info("=" * 60)
    
    if len(mode1_results) > 0 and len(mode2_results) > 0:
        mode1_pairs = set(mode1_results['pair'].values)
        mode2_pairs = set(mode2_results['pair'].values)
        
        overlap = mode1_pairs.intersection(mode2_pairs)
        mode1_only = mode1_pairs - mode2_pairs
        mode2_only = mode2_pairs - mode1_pairs
        
        logger.info(f"模式1和模式2的重叠配对: {len(overlap)}个")
        logger.info(f"仅模式1通过的配对: {len(mode1_only)}个")
        logger.info(f"仅模式2通过的配对: {len(mode2_only)}个")
        
        if len(mode1_only) > 0:
            logger.info("\n被半衰期筛选过滤掉的配对（前5个）:")
            filtered_out = mode1_results[mode1_results['pair'].isin(mode1_only)].head(5)
            for _, row in filtered_out.iterrows():
                logger.info(f"  {row['pair']}: 半衰期={row.get('halflife_5y', np.nan):.1f}天")
    
    return summary

if __name__ == "__main__":
    try:
        summary = compare_screening_modes()
        logger.info("\n分析完成!")
    except Exception as e:
        logger.error(f"分析失败: {str(e)}", exc_info=True)
        sys.exit(1)