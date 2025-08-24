#!/usr/bin/env python3
"""
测试数据管理模块的增量更新功能
验证REQ-1.3.x需求的实现
"""

import sys
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from lib.data import (
    fetch_single_symbol,
    save_to_parquet,
    load_from_parquet,
    atomic_update,
    detect_last_date,
    DataManager
)


def test_incremental_update():
    """测试增量更新功能"""
    
    print("=" * 60)
    print("测试数据管理模块的增量更新功能")
    print("=" * 60)
    
    # 创建临时测试目录
    test_dir = Path("./test_data_update")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 测试单个品种的增量更新
        print("\n1. 测试单个品种的增量更新")
        print("-" * 40)
        
        symbol = "AG0"  # 使用银作为测试品种
        
        # 首次获取数据
        print(f"首次获取 {symbol} 数据...")
        initial_df = fetch_single_symbol(symbol)
        print(f"获取到 {len(initial_df)} 条记录")
        print(f"日期范围: {initial_df['date'].min()} 至 {initial_df['date'].max()}")
        
        # 保存初始数据（模拟只保存一部分作为旧数据）
        # 只保存到30天前的数据
        cutoff_date = datetime.now() - timedelta(days=30)
        old_data = initial_df[initial_df['date'] < cutoff_date].copy()
        
        print(f"\n模拟旧数据: 保存到 {old_data['date'].max()} 的数据")
        save_path = save_to_parquet(old_data, symbol, test_dir)
        print(f"保存路径: {save_path}")
        
        # 检测最后日期
        last_date = detect_last_date(symbol, test_dir)
        print(f"检测到本地最后日期: {last_date}")
        
        # 执行增量更新
        print(f"\n执行增量更新...")
        success, new_count = atomic_update(symbol, test_dir)
        
        if success:
            print(f"✓ 增量更新成功！新增 {new_count} 条记录")
            
            # 验证更新后的数据
            updated_df = load_from_parquet(symbol, test_dir)
            print(f"更新后总记录数: {len(updated_df)}")
            print(f"更新后日期范围: {updated_df['date'].min()} 至 {updated_df['date'].max()}")
            
            # 验证数据完整性
            print("\n验证数据完整性:")
            print(f"- 原始数据记录数: {len(initial_df)}")
            print(f"- 更新后记录数: {len(updated_df)}")
            print(f"- 数据是否一致: {len(updated_df) >= len(old_data)}")
            
            # 再次执行更新（应该无新数据）
            print("\n再次执行更新（测试无新数据情况）...")
            success2, new_count2 = atomic_update(symbol, test_dir)
            print(f"第二次更新: 成功={success2}, 新增={new_count2} 条")
            
        else:
            print(f"✗ 增量更新失败")
            
        # 2. 测试DataManager的批量更新
        print("\n2. 测试DataManager批量更新")
        print("-" * 40)
        
        manager = DataManager(data_dir=str(test_dir))
        
        # 测试多个品种
        test_symbols = ["AG0", "AU0"]
        print(f"测试品种: {test_symbols}")
        
        # 首次获取和保存（模拟旧数据）
        print("\n准备测试数据...")
        for sym in test_symbols:
            try:
                df = fetch_single_symbol(sym)
                # 只保存60天前的数据
                cutoff = datetime.now() - timedelta(days=60)
                old_df = df[df['date'] < cutoff].copy()
                save_to_parquet(old_df, sym, test_dir)
                print(f"  {sym}: 准备了 {len(old_df)} 条旧数据")
            except Exception as e:
                print(f"  {sym}: 准备失败 - {e}")
        
        # 执行批量更新
        print("\n执行批量更新...")
        update_results = manager.update_all(test_symbols)
        
        print("\n更新结果:")
        for sym, count in update_results.items():
            status = "✓" if count >= 0 else "✗"
            print(f"  {status} {sym}: 新增 {count} 条记录")
            
        # 3. 测试错误恢复机制
        print("\n3. 测试错误恢复机制")
        print("-" * 40)
        
        # 创建一个损坏的临时文件
        symbol = "AG0"
        temp_file = test_dir / f"{symbol}.parquet.tmp"
        temp_file.write_text("corrupted data")
        
        print(f"创建损坏的临时文件: {temp_file}")
        
        # 尝试更新（应该能够处理错误）
        success, new_count = atomic_update(symbol, test_dir)
        
        if not temp_file.exists():
            print("✓ 临时文件已被正确清理")
        else:
            print("✗ 临时文件未被清理")
            
        # 验证原数据是否完好
        try:
            df = load_from_parquet(symbol, test_dir)
            print(f"✓ 原数据完好，记录数: {len(df)}")
        except Exception as e:
            print(f"✗ 原数据损坏: {e}")
            
        # 4. 测试更新日志
        print("\n4. 检查更新日志")
        print("-" * 40)
        
        log_file = test_dir / "update_log.csv"
        if log_file.exists():
            log_df = pd.read_csv(log_file)
            print(f"更新日志记录数: {len(log_df)}")
            print("\n最近5条更新记录:")
            print(log_df.tail(5).to_string())
        else:
            print("未找到更新日志文件")
            
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理测试目录
        if test_dir.exists():
            print(f"\n清理测试目录: {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_incremental_update()