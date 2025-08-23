#!/usr/bin/env python3
"""
期货数据增量更新脚本
实现REQ-1.3.x: 数据更新需求

使用方法:
    python update_futures_data.py              # 更新所有品种
    python update_futures_data.py AG0 AU0      # 更新指定品种
    python update_futures_data.py --check      # 只检查不更新
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.data import DataManager, detect_last_date, load_config


def check_data_status(symbols=None):
    """检查数据更新状态"""
    config = load_config()
    
    if symbols is None:
        symbols = config['symbols']['all']
    
    data_dir = Path("./data/futures")
    
    print("\n" + "=" * 70)
    print("期货数据更新状态检查")
    print("=" * 70)
    print(f"{'品种':<8} {'本地最后日期':<15} {'天数':<8} {'状态':<10}")
    print("-" * 70)
    
    today = datetime.now().date()
    
    for symbol in symbols:
        last_date = detect_last_date(symbol, data_dir)
        
        if last_date is None:
            print(f"{symbol:<8} {'无数据':<15} {'-':<8} {'需要初始化':<10}")
        else:
            last_date_str = last_date.strftime('%Y-%m-%d')
            days_diff = (today - last_date.date()).days
            
            if days_diff == 0:
                status = "最新"
            elif days_diff == 1:
                status = "待更新"
            elif days_diff <= 3:
                status = f"落后{days_diff}天"
            else:
                status = f"严重落后{days_diff}天"
            
            print(f"{symbol:<8} {last_date_str:<15} {days_diff:<8} {status:<10}")
    
    print("-" * 70)
    
    # 检查更新日志
    log_file = data_dir / "update_log.csv"
    if log_file.exists():
        log_df = pd.read_csv(log_file)
        if len(log_df) > 0:
            print(f"\n最近更新记录:")
            recent_logs = log_df.tail(5)
            for _, row in recent_logs.iterrows():
                timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                print(f"  {timestamp}: {row['symbol']} - {row['status']} (新增{row['new_records']}条)")


def update_data(symbols=None, dry_run=False):
    """更新期货数据"""
    config = load_config()
    
    if symbols is None:
        symbols = config['symbols']['all']
    
    manager = DataManager()
    
    print("\n" + "=" * 70)
    print(f"开始{'检查' if dry_run else '更新'}期货数据")
    print("=" * 70)
    print(f"品种列表: {', '.join(symbols)}")
    print(f"品种数量: {len(symbols)}")
    print(f"模式: {'模拟运行（不实际更新）' if dry_run else '实际更新'}")
    print("-" * 70)
    
    if dry_run:
        # 只检查状态，不实际更新
        check_data_status(symbols)
        return
    
    # 执行更新
    print("\n开始更新...")
    start_time = datetime.now()
    
    update_results = manager.update_all(symbols)
    
    # 统计结果
    total_new = sum(count for count in update_results.values() if count >= 0)
    success_count = sum(1 for count in update_results.values() if count >= 0)
    fail_count = sum(1 for count in update_results.values() if count < 0)
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("更新完成")
    print("-" * 70)
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"成功: {success_count} 个品种")
    print(f"失败: {fail_count} 个品种")
    print(f"新增记录: {total_new} 条")
    
    # 显示详细结果
    if len(update_results) > 0:
        print("\n详细结果:")
        for symbol, count in sorted(update_results.items()):
            if count >= 0:
                status = f"✓ 新增 {count} 条"
            else:
                status = "✗ 更新失败"
            print(f"  {symbol}: {status}")
    
    # 再次检查状态
    print("\n更新后状态:")
    check_data_status(symbols)
    
    print("\n" + "=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='期货数据增量更新工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    # 更新所有品种
  %(prog)s AG0 AU0           # 更新指定品种
  %(prog)s --check           # 检查数据状态
  %(prog)s --dry-run         # 模拟运行
        """
    )
    
    parser.add_argument(
        'symbols',
        nargs='*',
        help='要更新的品种代码，不指定则更新所有品种'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='只检查数据状态，不执行更新'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际更新数据'
    )
    
    args = parser.parse_args()
    
    # 确定要处理的品种
    symbols = args.symbols if args.symbols else None
    
    try:
        if args.check:
            check_data_status(symbols)
        else:
            update_data(symbols, dry_run=args.dry_run)
            
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()