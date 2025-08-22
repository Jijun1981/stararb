"""
期货配对交易系统 - 核心模块 V2.1

模块架构:
- data: 数据管理模块 (REQ-1.x.x)
- coint: 协整配对模块 (REQ-2.x.x, 包含β估计功能)  
- signal_generation: 信号生成模块 (REQ-3.x.x)
- backtest: 回测框架模块 (REQ-4.x.x)
"""

__version__ = "2.1.0"
__author__ = "Star-arb Team"

# 模块导入
try:
    from . import data
except ImportError as e:
    print(f"Warning: data module import failed: {e}")
    data = None

try:
    from . import coint
except ImportError as e:
    print(f"Warning: coint module import failed: {e}")
    coint = None

try:
    from . import signal_generation
except ImportError as e:
    print(f"Warning: signal_generation module import failed: {e}")
    signal_generation = None

try:
    from . import backtest
except ImportError as e:
    print(f"Warning: backtest module import failed: {e}")
    backtest = None

__all__ = [
    'data',
    'coint',
    'signal_generation',
    'backtest'
]