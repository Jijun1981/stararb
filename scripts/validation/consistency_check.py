#!/usr/bin/env python3
"""
需求文档与代码实现一致性检查
检查三个模块的需求文档与实际代码是否对齐
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def check_data_management_module():
    """检查数据管理模块的一致性"""
    print("\n" + "="*60)
    print("1. 数据管理模块一致性检查")
    print("="*60)
    
    issues = []
    
    # 检查lib/data.py的接口
    from lib import data
    
    # 需求要求的功能
    required_functions = {
        'load_all_symbols_data': '批量获取14个品种的数据',
        'update_symbol_data': '增量更新期货数据',
        'check_data_quality': '数据质量检查',
    }
    
    # 检查函数是否存在
    for func_name, desc in required_functions.items():
        if hasattr(data, func_name):
            print(f"✅ {func_name}: {desc}")
        else:
            print(f"❌ {func_name}: {desc} - 未找到")
            issues.append(f"数据管理模块缺少函数: {func_name}")
    
    # 检查数据源配置
    if hasattr(data, 'SYMBOLS'):
        symbols = data.SYMBOLS
        print(f"\n✅ 支持的品种数量: {len(symbols)}")
        if len(symbols) == 14:
            print("   符合需求（14个金属期货品种）")
        else:
            issues.append(f"品种数量不符: 需求14个，实际{len(symbols)}个")
    else:
        issues.append("未找到SYMBOLS配置")
    
    return issues

def check_cointegration_module():
    """检查协整配对模块的一致性"""
    print("\n" + "="*60)
    print("2. 协整配对模块一致性检查")
    print("="*60)
    
    issues = []
    
    from lib import coint
    
    # 检查CointegrationAnalyzer类
    if hasattr(coint, 'CointegrationAnalyzer'):
        analyzer_class = coint.CointegrationAnalyzer
        print("✅ CointegrationAnalyzer类存在")
        
        # 检查必需的方法
        required_methods = {
            'test_pair_cointegration': '协整检验',
            'screen_all_pairs': '批量配对筛选',
            'calculate_beta': 'Beta系数估计',
        }
        
        for method_name, desc in required_methods.items():
            if hasattr(analyzer_class, method_name):
                print(f"✅ {method_name}: {desc}")
            else:
                print(f"❌ {method_name}: {desc} - 未找到")
                issues.append(f"CointegrationAnalyzer缺少方法: {method_name}")
    else:
        issues.append("未找到CointegrationAnalyzer类")
    
    # 检查参数化支持
    if hasattr(coint, 'CointegrationAnalyzer'):
        # 检查构造函数参数
        sig = inspect.signature(coint.CointegrationAnalyzer.__init__)
        params = list(sig.parameters.keys())
        print(f"\n构造函数参数: {params}")
        
        # 检查screen_all_pairs的参数化
        if hasattr(coint.CointegrationAnalyzer, 'screen_all_pairs'):
            sig = inspect.signature(coint.CointegrationAnalyzer.screen_all_pairs)
            params = list(sig.parameters.keys())
            print(f"screen_all_pairs参数: {params}")
            
            # 验证参数化要求（根据需求文档的实际参数）
            required_params = ['windows', 'p_thresholds', 'screening_windows']
            for param in required_params:
                if param in params:
                    print(f"  ✅ 支持{param}参数")
                else:
                    print(f"  ❌ 缺少{param}参数")
                    issues.append(f"screen_all_pairs缺少参数: {param}")
    
    return issues

def check_signal_generation_module():
    """检查信号生成模块的一致性"""
    print("\n" + "="*60)
    print("3. 信号生成模块一致性检查")
    print("="*60)
    
    issues = []
    
    from lib import signal_generation
    
    # 检查Kalman滤波器
    if hasattr(signal_generation, 'KalmanFilter1D'):
        kf_class = signal_generation.KalmanFilter1D
        print("✅ KalmanFilter1D类存在")
        
        # 检查Kalman参数是否写死
        sig = inspect.signature(kf_class.__init__)
        params = list(sig.parameters.keys())
        
        # 按需求，应该只有self和initial_beta
        expected_params = ['self', 'initial_beta']
        if params == expected_params:
            print("✅ Kalman参数全部写死（只接受initial_beta）")
        else:
            extra_params = [p for p in params if p not in expected_params]
            if extra_params:
                print(f"❌ Kalman参数未完全写死，多余参数: {extra_params}")
                issues.append(f"KalmanFilter1D有多余参数: {extra_params}")
        
        # 验证固定值
        kf_test = kf_class(initial_beta=1.0)
        expected_values = {
            'Q': 1e-4,
            'P': 0.1,
            'R': 1.0
        }
        
        for attr, expected in expected_values.items():
            actual = getattr(kf_test, attr, None)
            if actual == expected:
                print(f"  ✅ {attr} = {expected} (固定值正确)")
            else:
                print(f"  ❌ {attr} = {actual} (期望{expected})")
                issues.append(f"Kalman参数{attr}值不正确: 实际{actual}，期望{expected}")
    else:
        issues.append("未找到KalmanFilter1D类")
    
    # 检查SignalGenerator类
    if hasattr(signal_generation, 'SignalGenerator'):
        sg_class = signal_generation.SignalGenerator
        print("\n✅ SignalGenerator类存在")
        
        # 检查构造函数参数
        sig = inspect.signature(sg_class.__init__)
        params = list(sig.parameters.keys())
        
        # 验证非Kalman参数可配置
        expected_params = ['window', 'z_open', 'z_close', 'convergence_days', 
                          'convergence_threshold', 'max_holding_days']
        
        for param in expected_params:
            if param in params:
                print(f"  ✅ {param}参数可配置")
            else:
                print(f"  ❌ 缺少{param}参数")
                issues.append(f"SignalGenerator缺少参数: {param}")
        
        # 检查process_pair_signals方法
        if hasattr(sg_class, 'process_pair_signals'):
            sig = inspect.signature(sg_class.process_pair_signals)
            params = list(sig.parameters.keys())
            
            # 验证输出格式参数
            if 'pair_info' in params:
                print("  ✅ 支持pair_info参数（配对信息）")
            else:
                print("  ❌ 缺少pair_info参数")
                issues.append("process_pair_signals缺少pair_info参数")
            
            if 'beta_window' in params:
                print("  ✅ 支持beta_window参数（β窗口选择）")
            else:
                print("  ❌ 缺少beta_window参数")
                issues.append("process_pair_signals缺少beta_window参数")
    else:
        issues.append("未找到SignalGenerator类")
    
    return issues

def check_output_format():
    """检查信号输出格式是否符合需求"""
    print("\n" + "="*60)
    print("4. 输出格式一致性检查")
    print("="*60)
    
    issues = []
    
    # 需求文档要求的字段（REQ-4.3）
    required_fields = [
        'date', 'pair', 'symbol_x', 'symbol_y', 'signal',
        'z_score', 'residual', 'beta', 'beta_initial',
        'days_held', 'reason', 'phase', 'beta_window_used'
    ]
    
    print("需求文档要求的输出字段:")
    for field in required_fields:
        print(f"  - {field}")
    
    # 这里可以加载实际输出文件进行验证
    output_file = project_root / "output/signals_test/real_data_test.csv"
    if output_file.exists():
        import pandas as pd
        df = pd.read_csv(output_file)
        actual_fields = list(df.columns)
        
        print("\n实际输出字段:")
        missing = []
        for field in required_fields:
            if field in actual_fields:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field} - 缺失")
                missing.append(field)
                issues.append(f"输出缺少字段: {field}")
        
        extra = [f for f in actual_fields if f not in required_fields]
        if extra:
            print(f"\n额外字段（需求未要求）: {extra}")
    else:
        print("未找到测试输出文件")
    
    return issues

def main():
    """运行完整的一致性检查"""
    print("="*60)
    print("需求文档与代码实现一致性检查")
    print("="*60)
    
    all_issues = []
    
    # 1. 数据管理模块
    issues = check_data_management_module()
    all_issues.extend(issues)
    
    # 2. 协整配对模块
    issues = check_cointegration_module()
    all_issues.extend(issues)
    
    # 3. 信号生成模块
    issues = check_signal_generation_module()
    all_issues.extend(issues)
    
    # 4. 输出格式
    issues = check_output_format()
    all_issues.extend(issues)
    
    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    
    if all_issues:
        print(f"\n发现 {len(all_issues)} 个不一致问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✅ 所有模块与需求文档完全一致！")
    
    return len(all_issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)