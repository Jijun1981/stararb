"""
数据管理模块
处理期货数据的获取、存储和预处理

数据源: data-joint目录下的聚宽8888主力连续合约数据
文件格式: jq_8888_{SYMBOL}.csv
品种格式: 纯符号格式 AG, AL, CU 等

重要：只使用data-joint数据源，纯符号格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# 设置日志
logger = logging.getLogger(__name__)


# 品种列表 - 14个金属期货
SYMBOLS = ['AG', 'AL', 'AU', 'CU', 'HC', 'I', 'NI', 
           'PB', 'RB', 'SF', 'SM', 'SN', 'SS', 'ZN']


def load_data(symbols: List[str],
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              columns: List[str] = ['close'],
              log_price: bool = False,
              fill_method: str = 'ffill',
              data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    从data-joint目录加载期货数据
    
    Args:
        symbols: 品种列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        columns: 需要的数据列，默认['close']
        log_price: 是否返回对数价格
        fill_method: 缺失值填充方法 ('ffill', 'bfill', None)
        data_dir: 忽略此参数，使用data-joint数据源
        
    Returns:
        按日期索引对齐的宽表DataFrame
        - 单列(close)时: 列名为纯符号 {symbol}
        - 多列时: 列名格式 {symbol}_{column}
    """
    # 使用data-joint数据源
    data_dir = Path(__file__).parent.parent / "data" / "data-joint"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"data-joint目录不存在: {data_dir}")
    
    dfs = []
    
    for symbol in symbols:
        # 使用纯符号格式访问数据文件
        csv_file = data_dir / f"jq_8888_{symbol}.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {csv_file}")
        
        # 读取CSV数据
        df = pd.read_csv(csv_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        
        # 筛选日期范围
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # 选择需要的列
        selected_df = pd.DataFrame(index=df.index)
        for col in columns:
            if col in df.columns:
                if log_price and col in ['open', 'high', 'low', 'close']:
                    # 使用纯符号作为列名（当只有close列时）
                    if len(columns) == 1 and col == 'close':
                        selected_df[symbol] = np.log(df[col])
                    else:
                        selected_df[f"{symbol}_{col}"] = np.log(df[col])
                else:
                    # 使用纯符号作为列名（当只有close列时）
                    if len(columns) == 1 and col == 'close':
                        selected_df[symbol] = df[col]
                    else:
                        selected_df[f"{symbol}_{col}"] = df[col]
            else:
                raise ValueError(f"{symbol} 缺少列: {col}")
        
        dfs.append(selected_df)
    
    if not dfs:
        raise ValueError("没有加载任何数据")
    
    # 合并所有DataFrame
    result = pd.concat(dfs, axis=1, join='outer')
    
    # 处理缺失值
    if fill_method == 'ffill':
        result = result.ffill()
    elif fill_method == 'bfill':
        result = result.bfill()
    
    # 确保按日期排序
    result = result.sort_index()
    
    logger.info(f"从data-joint加载了 {len(symbols)} 个品种的数据")
    logger.info(f"数据范围: {result.index[0]} 至 {result.index[-1]}")
    logger.info(f"数据形状: {result.shape}")
    
    return result


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """
    从CSV文件加载单个品种数据
    
    Args:
        symbol: 品种代码 (如 'AG', 'CU' 等，不带0后缀)
        
    Returns:
        DataFrame with datetime index and columns: open, high, low, close, volume, open_interest
    """
    # 使用data-joint数据源
    data_dir = Path(__file__).parent.parent / "data" / "data-joint"
    csv_file = data_dir / f"jq_8888_{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_file}")
    
    # 读取CSV，第一列是日期
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # 确保索引是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # 确保有必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    logger.info(f"加载 {symbol}: {len(df)} 条记录")
    
    return df


def load_all_symbols_data(symbols: Optional[List[str]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    加载所有品种的收盘价数据
    
    Args:
        symbols: 品种列表，默认为所有14个品种
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        DataFrame with columns: {symbol} (纯符号，不带后缀)
    """
    if symbols is None:
        symbols = SYMBOLS
    
    return load_data(symbols, start_date, end_date, columns=['close'], log_price=False)


def update_symbol_data(symbol: str) -> bool:
    """
    增量更新期货数据（占位实现）
    
    Args:
        symbol: 品种代码
        
    Returns:
        是否更新成功
    """
    # 由于当前使用静态data-joint数据，暂不实现增量更新
    print(f"[INFO] 增量更新功能暂未实现（使用静态data-joint数据）")
    return True


def check_data_quality(symbol: str = None) -> Dict:
    """
    数据质量检查
    
    Args:
        symbol: 品种代码，None表示检查所有品种
        
    Returns:
        数据质量报告
    """
    report = {
        'symbol': symbol if symbol else 'ALL',
        'missing_ratio': 0.0,
        'outliers': [],
        'data_range': None,
        'total_records': 0,
        'status': 'OK'
    }
    
    try:
        if symbol:
            # 检查单个品种
            df = load_from_parquet(symbol)
            report['total_records'] = len(df)
            report['data_range'] = f"{df.index.min()} to {df.index.max()}"
            
            # 缺失值检查
            missing = df['close'].isna().sum()
            report['missing_ratio'] = missing / len(df)
            
            # 异常值检查（5倍标准差）
            returns = df['close'].pct_change()
            std = returns.std()
            outliers = returns[abs(returns) > 5 * std]
            if len(outliers) > 0:
                report['outliers'] = outliers.index.tolist()
        else:
            # 检查所有品种
            all_data = load_all_symbols_data()
            report['total_records'] = len(all_data)
            report['data_range'] = f"{all_data.index.min()} to {all_data.index.max()}"
            
            # 缺失值检查
            total_missing = all_data.isna().sum().sum()
            total_cells = all_data.size
            report['missing_ratio'] = total_missing / total_cells
            
        if report['missing_ratio'] > 0.01:
            report['status'] = 'WARNING'
        if report['missing_ratio'] > 0.05:
            report['status'] = 'ERROR'
            
    except Exception as e:
        report['status'] = 'ERROR'
        report['error'] = str(e)
    
    return report


def check_data_availability():
    """
    检查data-joint目录中的数据可用性
    """
    data_dir = Path(__file__).parent.parent / "data" / "data-joint"
    
    print(f"检查数据目录: {data_dir}")
    print("=" * 60)
    
    available = []
    missing = []
    
    for symbol in SYMBOLS:
        csv_file = data_dir / f"jq_8888_{symbol}.csv"
        
        if csv_file.exists():
            # 读取文件获取信息
            df = pd.read_csv(csv_file, index_col=0, nrows=5)
            df.index = pd.to_datetime(df.index)
            
            # 获取完整数据统计
            full_df = pd.read_csv(csv_file, index_col=0)
            full_df.index = pd.to_datetime(full_df.index)
            
            available.append({
                'symbol': symbol,
                'file': csv_file.name,
                'records': len(full_df),
                'start': full_df.index[0],
                'end': full_df.index[-1]
            })
            print(f"✓ {symbol}: {csv_file.name} ({len(full_df)} 条记录)")
            print(f"  日期范围: {full_df.index[0].strftime('%Y-%m-%d')} 至 {full_df.index[-1].strftime('%Y-%m-%d')}")
        else:
            missing.append(symbol)
            print(f"✗ {symbol}: 文件不存在 ({csv_file.name})")
    
    print("\n" + "=" * 60)
    print(f"可用: {len(available)}/{len(SYMBOLS)} 个品种")
    if missing:
        print(f"缺失: {', '.join(missing)}")
    
    return available, missing


# 清理旧的parquet文件
def clean_old_parquet_files():
    """
    清理旧的parquet文件，确保使用新数据
    """
    parquet_dir = Path(__file__).parent.parent / "data" / "futures"
    
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        
        if parquet_files:
            print(f"发现 {len(parquet_files)} 个旧的parquet文件")
            
            for file in parquet_files:
                print(f"删除: {file.name}")
                file.unlink()
            
            print("旧文件清理完成")
        else:
            print("没有发现旧的parquet文件")
    else:
        print("futures目录不存在")


if __name__ == "__main__":
    # 检查数据可用性
    print("检查data-joint数据源...")
    available, missing = check_data_availability()
    
    if available:
        print("\n测试数据加载...")
        # 测试加载一个品种
        test_symbol = available[0]['symbol']
        df = load_from_parquet(test_symbol)
        print(f"\n成功加载 {test_symbol}: {len(df)} 条记录")
        print(f"列: {list(df.columns)}")
        print(f"\n前5行:")
        print(df.head())