#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目结构清理脚本
移动散落在根目录的文件到合适的位置
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def create_directories():
    """创建必要的目录结构"""
    directories = [
        'output/signals',
        'output/quality_reports', 
        'output/residual_validation',
        'output/kalman_analysis',
        'output/comparison',
        'output/debug',
        'test_data',
        'scripts/experimental',
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 确保目录存在: {dir_path}")

def move_files():
    """移动文件到适当的目录"""
    
    # 定义文件移动规则
    move_rules = {
        # 信号文件 → output/signals/
        'signals_*.csv': 'output/signals/',
        
        # 质量报告 → output/quality_reports/
        'quality_report_*.csv': 'output/quality_reports/',
        
        # 残差验证 → output/residual_validation/
        'residual_validation_*.csv': 'output/residual_validation/',
        
        # Kalman分析相关 → output/kalman_analysis/
        'kalman_*.csv': 'output/kalman_analysis/',
        'kalman_*.png': 'output/kalman_analysis/',
        'kalman_*.py': 'scripts/experimental/',  # 实验性脚本
        
        # 比较文件 → output/comparison/
        'ols_vs_kalman_*.csv': 'output/comparison/',
        '*_comparison.csv': 'output/comparison/',
        '*_comparison.png': 'output/comparison/',
        
        # 调试文件 → output/debug/
        'debug_*.csv': 'output/debug/',
        'debug_*.py': 'scripts/experimental/',
        
        # 测试相关 → test_data/
        'test_*.py': 'scripts/experimental/',
        'adf_stationarity_test_*.csv': 'test_data/',
        
        # 交易和回测结果
        'backtest_trades_*.csv': 'output/backtest/',
        'actual_trades.csv': 'output/backtest/',
        'trading_signals_summary.csv': 'output/signals/',
        'pairs_trading_summary.csv': 'output/backtest/',
    }
    
    # 执行文件移动
    for pattern, target_dir in move_rules.items():
        files = list(Path('.').glob(pattern))
        if files:
            print(f"\n移动 {pattern} 到 {target_dir}:")
            for file_path in files:
                if file_path.is_file():
                    target_path = Path(target_dir) / file_path.name
                    try:
                        shutil.move(str(file_path), str(target_path))
                        print(f"  ✓ {file_path.name}")
                    except Exception as e:
                        print(f"  ✗ {file_path.name}: {e}")

def organize_scripts():
    """整理scripts目录下的脚本"""
    
    # 根目录下的独立脚本
    standalone_scripts = [
        'final_kalman_test.py',
        'test_improvements.py',
        'test_sm_rb_kalman.py',
        'test_adaptive_kalman_integration.py',
        'test_final_kalman.py',
    ]
    
    print("\n整理独立脚本:")
    for script in standalone_scripts:
        if Path(script).exists():
            target = Path('scripts/experimental') / script
            try:
                shutil.move(script, str(target))
                print(f"  ✓ {script} → scripts/experimental/")
            except Exception as e:
                print(f"  ✗ {script}: {e}")

def cleanup_root():
    """清理根目录的临时文件"""
    
    # 需要保留在根目录的文件
    keep_in_root = {
        'CLAUDE.md',
        'PROJECT_STRUCTURE.md', 
        'requirements.txt',
        'cointegration_results.csv',  # 主要协整结果
    }
    
    print("\n根目录文件状态:")
    for file in Path('.').glob('*.csv'):
        if file.name not in keep_in_root:
            print(f"  ⚠ {file.name} 应该移动到output/目录")

def generate_cleanup_report():
    """生成清理报告"""
    
    report = []
    report.append("# 项目结构清理报告")
    report.append(f"清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 统计各目录文件数量
    report.append("## 目录文件统计\n")
    
    dirs_to_check = [
        'lib',
        'scripts/analysis',
        'scripts/backtest', 
        'scripts/pipeline',
        'scripts/signals',
        'scripts/validation',
        'scripts/experimental',
        'output/signals',
        'output/quality_reports',
        'output/residual_validation',
        'output/kalman_analysis',
        'output/comparison',
        'output/debug',
        'data/data-joint',
        'tests/unit',
        'notebooks',
    ]
    
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            file_count = len(list(Path(dir_path).glob('*.*')))
            report.append(f"- {dir_path}: {file_count} 个文件")
    
    # 根目录遗留文件
    report.append("\n## 根目录遗留文件\n")
    root_files = [f for f in Path('.').glob('*.*') if f.is_file()]
    csv_files = [f for f in root_files if f.suffix == '.csv']
    py_files = [f for f in root_files if f.suffix == '.py']
    
    report.append(f"- CSV文件: {len(csv_files)} 个")
    report.append(f"- Python脚本: {len(py_files)} 个")
    report.append(f"- 其他文件: {len(root_files) - len(csv_files) - len(py_files)} 个")
    
    # 写入报告
    report_path = 'docs/cleanup_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✓ 清理报告已生成: {report_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("开始清理项目结构...")
    print("=" * 60)
    
    # 1. 创建目录
    create_directories()
    
    # 2. 移动文件
    move_files()
    
    # 3. 整理脚本
    organize_scripts()
    
    # 4. 检查根目录
    cleanup_root()
    
    # 5. 生成报告
    generate_cleanup_report()
    
    print("\n" + "=" * 60)
    print("项目结构清理完成！")
    print("请查看 docs/cleanup_report.md 了解详情")
    print("=" * 60)

if __name__ == "__main__":
    main()