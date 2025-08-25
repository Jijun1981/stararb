#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目结构清理脚本
整理根目录下散乱的文件，按照PROJECT_STRUCTURE.md规范重新组织
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectCleaner:
    def __init__(self, project_root: str = "/mnt/e/Star-arb"):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup" / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def create_directories(self):
        """创建标准目录结构"""
        dirs_to_create = [
            "scripts/maintenance",
            "scripts/analysis", 
            "scripts/validation",
            "scripts/debug",
            "output/analysis",
            "output/kalman_analysis",
            "output/debug",
            "output/backtest",
            "backup/root_files",
            "logs"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {dir_path}")
    
    def move_analysis_scripts(self):
        """移动分析类Python脚本"""
        analysis_patterns = [
            'analyze_',
            'verify_',
            'check_',
            'compare_',
            'debug_beta',
            'debug_kalman',
            'evaluate_'
        ]
        
        moved_files = []
        for py_file in self.project_root.glob("*.py"):
            if py_file.name in ['setup.py', 'main.py']:  # 保留这些文件在根目录
                continue
                
            filename = py_file.name
            
            # 判断文件类型并移动
            if any(pattern in filename for pattern in analysis_patterns):
                if 'debug' in filename:
                    target_dir = self.project_root / "scripts" / "debug"
                else:
                    target_dir = self.project_root / "scripts" / "analysis"
                    
                target_path = target_dir / filename
                shutil.move(str(py_file), str(target_path))
                moved_files.append(f"{filename} → scripts/{'debug' if 'debug' in filename else 'analysis'}/")
                logger.info(f"移动分析脚本: {filename}")
        
        return moved_files
    
    def move_test_scripts(self):
        """移动测试相关脚本"""
        test_patterns = [
            'test_',
            'validate_',
            'verification_'
        ]
        
        moved_files = []
        for py_file in self.project_root.glob("*.py"):
            filename = py_file.name
            
            if any(pattern in filename for pattern in test_patterns):
                target_dir = self.project_root / "scripts" / "validation"
                target_path = target_dir / filename
                shutil.move(str(py_file), str(target_path))
                moved_files.append(f"{filename} → scripts/validation/")
                logger.info(f"移动测试脚本: {filename}")
        
        return moved_files
    
    def organize_output_files(self):
        """整理输出文件"""
        file_mappings = {
            # CSV文件分类
            'backtest_*': 'output/backtest/',
            'signals_*': 'output/signals/', 
            'quality_report_*': 'output/analysis/',
            'kalman_*': 'output/kalman_analysis/',
            'beta_evolution_*': 'output/analysis/',
            'all_stable_pairs_*': 'output/analysis/',
            'cointegration_*': 'output/cointegration/',
            
            # PNG文件分类  
            '*.png': 'output/plots/',
        }
        
        moved_files = []
        
        # 确保目标目录存在
        for target_path in file_mappings.values():
            (self.project_root / target_path).mkdir(parents=True, exist_ok=True)
        
        # 移动CSV文件
        for csv_file in self.project_root.glob("*.csv"):
            filename = csv_file.name
            moved = False
            
            for pattern, target_dir in file_mappings.items():
                if pattern.endswith('*') and filename.startswith(pattern[:-1]):
                    target_path = self.project_root / target_dir / filename
                    shutil.move(str(csv_file), str(target_path))
                    moved_files.append(f"{filename} → {target_dir}")
                    moved = True
                    break
            
            if not moved:
                # 默认移动到output/misc/
                misc_dir = self.project_root / "output" / "misc"
                misc_dir.mkdir(exist_ok=True)
                target_path = misc_dir / filename
                shutil.move(str(csv_file), str(target_path))
                moved_files.append(f"{filename} → output/misc/")
        
        # 移动PNG文件
        plots_dir = self.project_root / "output" / "plots"
        for png_file in self.project_root.glob("*.png"):
            target_path = plots_dir / png_file.name
            shutil.move(str(png_file), str(target_path))
            moved_files.append(f"{png_file.name} → output/plots/")
        
        return moved_files
    
    def backup_remaining_files(self):
        """备份根目录下的其他文件"""
        backup_dir = self.project_root / "backup" / "root_files"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 不应该在根目录的文件类型
        file_patterns = ['*.log', '*.tmp', '*.bak']
        
        backed_up = []
        for pattern in file_patterns:
            for file in self.project_root.glob(pattern):
                target_path = backup_dir / file.name
                shutil.move(str(file), str(target_path))
                backed_up.append(f"{file.name} → backup/root_files/")
        
        return backed_up
    
    def update_gitignore(self):
        """更新.gitignore文件"""
        gitignore_path = self.project_root / ".gitignore"
        
        additional_ignores = [
            "\n# 输出和结果文件",
            "output/",
            "logs/",
            "*.csv",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "\n# 缓存和临时文件",
            "__pycache__/",
            "*.pyc", 
            "*.pyo",
            "*.tmp",
            "*.log",
            "\n# 数据文件", 
            "data/*.csv",
            "data/*.parquet",
            "\n# 备份文件",
            "backup/cleanup_*",
        ]
        
        # 读取现有内容
        existing_content = ""
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # 添加新规则
        new_rules = []
        for rule in additional_ignores:
            if rule not in existing_content:
                new_rules.append(rule)
        
        if new_rules:
            with open(gitignore_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(new_rules))
            logger.info("更新了.gitignore文件")
    
    def generate_cleanup_report(self, moved_scripts, moved_outputs, backed_up):
        """生成清理报告"""
        report = f"""
# 项目结构清理报告
清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 移动的脚本文件 ({len(moved_scripts)}个)
"""
        for item in moved_scripts:
            report += f"- {item}\n"
        
        report += f"\n## 移动的输出文件 ({len(moved_outputs)}个)\n"
        for item in moved_outputs:
            report += f"- {item}\n"
        
        if backed_up:
            report += f"\n## 备份的文件 ({len(backed_up)}个)\n"
            for item in backed_up:
                report += f"- {item}\n"
        
        report += """
## 清理后的目录结构
```
Star-arb/
├── scripts/
│   ├── analysis/        # 分析脚本
│   ├── debug/          # 调试脚本  
│   ├── validation/     # 验证脚本
│   └── maintenance/    # 维护脚本
├── output/
│   ├── analysis/       # 分析结果
│   ├── backtest/       # 回测结果
│   ├── kalman_analysis/ # Kalman分析结果
│   ├── plots/          # 图表文件
│   └── signals/        # 信号文件
└── backup/
    └── root_files/     # 备份文件
```

## 注意事项
1. 如果有脚本路径问题，检查新位置的import路径
2. 输出文件已分类存放，便于管理
3. 根目录现在更清洁，只保留核心配置文件
"""
        
        report_path = self.project_root / "scripts" / "maintenance" / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"生成清理报告: {report_path}")
        return report_path

def main():
    """执行清理"""
    cleaner = ProjectCleaner()
    
    logger.info("开始清理项目结构...")
    
    # 1. 创建目录结构
    cleaner.create_directories()
    
    # 2. 移动脚本文件
    moved_scripts = cleaner.move_analysis_scripts()
    moved_scripts.extend(cleaner.move_test_scripts())
    
    # 3. 整理输出文件
    moved_outputs = cleaner.organize_output_files()
    
    # 4. 备份其他文件
    backed_up = cleaner.backup_remaining_files()
    
    # 5. 更新gitignore
    cleaner.update_gitignore()
    
    # 6. 生成报告
    report_path = cleaner.generate_cleanup_report(moved_scripts, moved_outputs, backed_up)
    
    logger.info("项目结构清理完成!")
    logger.info(f"总共移动了 {len(moved_scripts)} 个脚本文件")
    logger.info(f"总共移动了 {len(moved_outputs)} 个输出文件")
    logger.info(f"清理报告: {report_path}")
    
    # 显示清理后的根目录
    logger.info("\n清理后的根目录文件:")
    root_files = [f for f in os.listdir(cleaner.project_root) if os.path.isfile(os.path.join(cleaner.project_root, f))]
    for f in sorted(root_files):
        logger.info(f"  {f}")

if __name__ == "__main__":
    main()