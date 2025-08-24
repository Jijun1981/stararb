"""
数据管理模块
处理数据加载和访问

从 notebooks/01_data_management.ipynb 提取的核心功能
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import json
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: List[str] = ['close'],
    log_price: bool = False,
    fill_method: str = 'ffill',
    data_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    统一的数据加载接口
    
    Args:
        symbols: 品种列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        columns: 需要的数据列，默认['close']
        log_price: 是否返回对数价格
        fill_method: 缺失值填充方法 ('ffill', 'bfill', None)
        data_dir: 数据目录，默认为 data/futures
        
    Returns:
        按日期索引对齐的宽表DataFrame，列名格式: {symbol}_{column}
    """
    if data_dir is None:
        # 获取项目根目录下的data/futures路径
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    else:
        data_dir = Path(data_dir)
    
    dfs = []
    
    for symbol in symbols:
        file_path = data_dir / f"{symbol}.parquet"
        
        if not file_path.exists():
            logger.warning(f"{symbol} 数据文件不存在: {file_path}")
            continue
        
        # 读取数据
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 筛选日期范围
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # 选择需要的列
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"{symbol} 缺少列: {missing_cols}")
            continue
        
        df = df[columns]
        
        # 重命名列
        df.columns = [f"{symbol}_{col}" for col in columns]
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("没有加载任何数据")
    
    # 合并所有DataFrame（外连接，保留所有日期）
    result = pd.concat(dfs, axis=1, join='outer')
    
    # 处理缺失值
    if fill_method == 'ffill':
        result = result.fillna(method='ffill')
    elif fill_method == 'bfill':
        result = result.fillna(method='bfill')
    
    # 应用对数变换
    if log_price:
        price_cols = [col for col in result.columns if any(p in col for p in ['open', 'high', 'low', 'close'])]
        for col in price_cols:
            if (result[col] <= 0).any():
                raise ValueError(f"{col} 包含非正数，无法进行对数变换")
            result[col] = np.log(result[col])
    
    # 按日期排序
    result = result.sort_index()
    
    logger.info(f"加载数据完成: {len(symbols)} 个品种, {len(result)} 条记录")
    return result


def load_single_symbol(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    加载单个品种的数据
    
    Args:
        symbol: 品种代码
        start_date: 开始日期
        end_date: 结束日期
        data_dir: 数据目录
        
    Returns:
        DataFrame with columns: date(index), open, high, low, close, volume, open_interest
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    else:
        data_dir = Path(data_dir)
    
    file_path = data_dir / f"{symbol}.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"{symbol} 数据文件不存在: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 筛选日期范围
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    return df


def get_available_symbols(data_dir: Optional[str] = None) -> List[str]:
    """
    获取所有可用的品种列表
    
    Args:
        data_dir: 数据目录
        
    Returns:
        品种代码列表
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    else:
        data_dir = Path(data_dir)
    
    symbols = []
    for file_path in data_dir.glob("*.parquet"):
        symbol = file_path.stem  # 获取文件名（不含扩展名）
        symbols.append(symbol)
    
    return sorted(symbols)


def load_metadata(data_dir: Optional[str] = None) -> Dict:
    """
    加载元信息
    
    Args:
        data_dir: 数据目录
        
    Returns:
        元信息字典
    """
    if data_dir is None:
        metadata_file = Path(__file__).parent.parent / "data" / "metadata.json"
    else:
        metadata_file = Path(data_dir).parent / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# 保持向后兼容
def fetch_symbol(symbol: str) -> pd.DataFrame:
    """向后兼容的函数"""
    return load_single_symbol(symbol)


def fetch_all(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """向后兼容的函数"""
    if symbols is None:
        symbols = get_available_symbols()
    
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = load_single_symbol(symbol)
        except Exception as e:
            logger.warning(f"加载 {symbol} 失败: {e}")
    
    return result


def save_parquet(df: pd.DataFrame, symbol: str, data_dir: str = "./data/processed") -> Path:
    """向后兼容的函数"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / f"{symbol}.parquet"
    df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
    return file_path


def update_data(symbol: str) -> int:
    """向后兼容的函数"""
    # 暂时返回0，表示没有新数据
    return 0