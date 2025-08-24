"""
回测框架模块
"""

from .position_sizing import PositionSizer, PositionSizingConfig

__all__ = [
    'PositionSizer',
    'PositionSizingConfig',
]

# 其他模块将在实现后添加
try:
    from .trade_executor import TradeExecutor, ExecutionConfig
    __all__.extend(['TradeExecutor', 'ExecutionConfig'])
except ImportError:
    pass

try:
    from .risk_manager import RiskManager, RiskConfig
    __all__.extend(['RiskManager', 'RiskConfig'])
except ImportError:
    pass

try:
    from .performance import PerformanceAnalyzer
    __all__.append('PerformanceAnalyzer')
except ImportError:
    pass