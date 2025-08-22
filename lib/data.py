"""
数据管理模块
处理期货数据的获取、存储、更新和预处理

实现需求:
- REQ-1.1.x: 数据获取
- REQ-1.2.x: 数据存储  
- REQ-1.3.x: 数据更新
- REQ-1.4.x: 数据访问

基于: notebooks/01_data_management.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
from datetime import datetime, timedelta
import time
import logging
import akshare as ak
import yaml

# 设置日志
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """加载业务配置"""
    config_path = Path(__file__).parent.parent / "configs" / "business.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class DataError(Exception):
    """数据相关异常"""
    pass


class DataValidationError(DataError):
    """数据验证失败"""
    pass


class DataUpdateError(DataError):
    """数据更新失败"""
    pass


def fetch_single_symbol(symbol: str, retries: int = 3) -> pd.DataFrame:
    """
    获取单个品种的期货连续合约历史数据
    
    实现: REQ-1.1.1, REQ-1.1.3
    
    Args:
        symbol: 品种代码，如'RB0'
        retries: 重试次数
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, open_interest
        
    Raises:
        DataError: 数据获取失败
    """
    config = load_config()
    symbol_names = config['symbols']['names']
    
    for attempt in range(retries):
        try:
            logger.info(f"正在获取 {symbol} ({symbol_names.get(symbol, symbol)}) 的数据...")
            
            # 使用AkShare API获取数据
            df = ak.futures_zh_daily_sina(symbol=symbol)
            
            if df is None or df.empty:
                raise DataError(f"{symbol} 返回空数据")
            
            # 重命名列：hold -> open_interest
            df = df.rename(columns={'hold': 'open_interest'})
            
            # 选择需要的列
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataValidationError(f"{symbol} 缺少必需字段: {missing_cols}")
            
            df = df[required_cols]
            
            # 数据类型转换
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'open_interest']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除包含NaN的行
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            if len(df) == 0:
                raise DataValidationError(f"{symbol} 清理后无有效数据")
            
            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"成功获取 {symbol} 数据: {len(df)} 条记录, "
                       f"日期范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败 (尝试 {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(5)  # 等待5秒后重试
            else:
                raise DataError(f"无法获取 {symbol} 数据: {str(e)}")


def fetch_multiple_symbols(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    批量获取多个品种的数据
    
    实现: REQ-1.1.1
    
    Args:
        symbols: 品种代码列表
        
    Returns:
        字典 {symbol: DataFrame}
        
    Raises:
        DataError: 没有成功获取任何数据
    """
    data_dict = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            df = fetch_single_symbol(symbol)
            data_dict[symbol] = df
            time.sleep(1)  # 避免请求过快
        except Exception as e:
            logger.error(f"获取 {symbol} 失败: {str(e)}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        logger.warning(f"失败的品种: {failed_symbols}")
    
    logger.info(f"批量获取完成: 成功 {len(data_dict)}/{len(symbols)} 个品种")
    
    if not data_dict:
        raise DataError("没有成功获取任何数据")
    
    return data_dict


def save_to_parquet(df: pd.DataFrame, symbol: str, data_dir: Optional[Path] = None) -> Path:
    """
    将DataFrame保存为Parquet格式
    
    实现: REQ-1.2.1
    
    Args:
        df: 数据DataFrame
        symbol: 品种代码
        data_dir: 保存目录，None时使用默认目录
        
    Returns:
        保存的文件路径
        
    Raises:
        DataValidationError: 数据验证失败
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    
    # 确保目录存在
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / f"{symbol}.parquet"
    
    # 保存为Parquet格式
    df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"成功保存 {symbol} 数据到 {file_path}")
    
    # 验证保存的文件
    try:
        test_df = pd.read_parquet(file_path)
        if len(test_df) != len(df):
            raise DataValidationError("保存的数据行数不匹配")
        if list(test_df.columns) != list(df.columns):
            raise DataValidationError("保存的数据列不匹配")
    except Exception as e:
        raise DataValidationError(f"验证保存的文件失败: {e}")
    
    return file_path


def load_from_parquet(symbol: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    从Parquet文件加载数据
    
    Args:
        symbol: 品种代码
        data_dir: 数据目录
        
    Returns:
        DataFrame
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    
    file_path = data_dir / f"{symbol}.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"{symbol} 数据文件不存在: {file_path}")
    
    return pd.read_parquet(file_path)


def save_metadata(data_dict: Dict[str, pd.DataFrame], data_dir: Optional[Path] = None) -> Path:
    """
    保存数据元信息到metadata.json
    
    实现: REQ-1.2.2
    
    Args:
        data_dict: {symbol: DataFrame} 字典
        data_dir: 数据目录
        
    Returns:
        metadata文件路径
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = data_dir / "metadata.json"
    
    config = load_config()
    symbol_names = config['symbols']['names']
    
    metadata = {
        "update_time": datetime.now().isoformat(),
        "symbols": {}
    }
    
    for symbol, df in data_dict.items():
        metadata["symbols"][symbol] = {
            "name": symbol_names.get(symbol, symbol),
            "fetch_time": datetime.now().isoformat(),
            "start_date": df['date'].min().isoformat(),
            "end_date": df['date'].max().isoformat(),
            "record_count": len(df),
            "columns": list(df.columns),
            "file": f"{symbol}.parquet"
        }
    
    # 保存metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"元信息已保存到 {metadata_file}")
    return metadata_file


def load_metadata(data_dir: Optional[Path] = None) -> Dict:
    """
    加载元信息
    
    Args:
        data_dir: 数据目录
        
    Returns:
        元信息字典
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    metadata_file = data_dir / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def detect_last_date(symbol: str, data_dir: Optional[Path] = None) -> Optional[datetime]:
    """
    检测本地数据的最后日期
    
    实现: REQ-1.3.1
    
    Args:
        symbol: 品种代码
        data_dir: 数据目录
        
    Returns:
        最后日期，如果文件不存在返回None
        
    Raises:
        DataError: 读取文件失败
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    
    file_path = data_dir / f"{symbol}.parquet"
    
    if not file_path.exists():
        logger.info(f"{symbol} 本地数据不存在")
        return None
    
    try:
        df = pd.read_parquet(file_path)
        last_date = df['date'].max()
        logger.info(f"{symbol} 本地数据最后日期: {last_date.date()}")
        return last_date
    except Exception as e:
        raise DataError(f"读取 {symbol} 本地数据失败: {str(e)}")


def atomic_update(symbol: str, data_dir: Optional[Path] = None) -> Tuple[bool, int]:
    """
    原子性更新单个品种的数据
    
    实现: REQ-1.3.1, REQ-1.3.2, REQ-1.3.3
    
    Args:
        symbol: 品种代码
        data_dir: 数据目录
        
    Returns:
        (是否成功, 新增记录数)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = data_dir / f"{symbol}.parquet"
    temp_path = data_dir / f"{symbol}.parquet.tmp"
    backup_path = data_dir / f"{symbol}.parquet.bak"
    
    try:
        # 检测本地最后日期
        last_date = detect_last_date(symbol, data_dir)
        
        # 获取最新数据
        new_df = fetch_single_symbol(symbol)
        
        if last_date is not None:
            # 加载旧数据
            old_df = pd.read_parquet(file_path)
            
            # 只保留新增的数据
            new_records = new_df[new_df['date'] > last_date]
            
            if len(new_records) == 0:
                logger.info(f"{symbol} 无新数据")
                return True, 0
            
            # 合并数据并去重
            combined_df = pd.concat([old_df, new_records], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            new_count = len(new_records)
        else:
            combined_df = new_df
            new_count = len(new_df)
        
        # 保存到临时文件
        combined_df.to_parquet(temp_path, engine='pyarrow', compression='snappy', index=False)
        
        # 验证临时文件
        test_df = pd.read_parquet(temp_path)
        if len(test_df) != len(combined_df):
            raise DataValidationError("临时文件验证失败")
        
        # 备份原文件（如果存在）
        if file_path.exists():
            file_path.rename(backup_path)
        
        # 将临时文件重命名为正式文件
        temp_path.rename(file_path)
        
        # 删除备份文件
        if backup_path.exists():
            backup_path.unlink()
        
        logger.info(f"{symbol} 原子更新成功: 新增 {new_count} 条记录")
        return True, new_count
        
    except Exception as e:
        logger.error(f"{symbol} 更新失败: {str(e)}")
        
        # 回滚：恢复备份文件
        if backup_path.exists() and not file_path.exists():
            backup_path.rename(file_path)
            logger.info(f"{symbol} 已回滚到原数据")
        
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        
        return False, 0


def log_update(symbol: str, new_count: int, status: str, error_msg: str = "", 
               data_dir: Optional[Path] = None) -> None:
    """
    记录更新日志
    
    实现: REQ-1.3.4
    
    Args:
        symbol: 品种代码
        new_count: 新增记录数
        status: 状态（success/failed）
        error_msg: 错误信息（如果失败）
        data_dir: 数据目录
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / "update_log.csv"
    
    log_entry = {
        'timestamp': datetime.now(),
        'symbol': symbol,
        'new_records': new_count,
        'status': status,
        'error': error_msg if error_msg else None
    }
    
    # 读取现有日志（如果存在）
    if log_file.exists():
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame()
    
    # 添加新日志
    new_log_df = pd.DataFrame([log_entry])
    log_df = pd.concat([log_df, new_log_df], ignore_index=True)
    
    # 保存日志
    log_df.to_csv(log_file, index=False)
    logger.info(f"更新日志已记录: {symbol} - {status}")


def load_data(symbols: List[str],
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              columns: List[str] = ['close'],
              log_price: bool = False,
              fill_method: str = 'ffill',
              data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    统一的数据加载接口
    
    实现: REQ-1.4.1, REQ-1.4.2, REQ-1.4.3, REQ-1.4.4
    
    Args:
        symbols: 品种列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        columns: 需要的数据列，默认['close']
        log_price: 是否返回对数价格
        fill_method: 缺失值填充方法 ('ffill', 'bfill', None)
        data_dir: 数据目录
        
    Returns:
        按日期索引对齐的宽表DataFrame，列名格式: {symbol}_{column}
        
    Raises:
        FileNotFoundError: 数据文件不存在
        ValueError: 参数无效或缺少列
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "futures"
    
    dfs = []
    
    for symbol in symbols:
        file_path = data_dir / f"{symbol}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"{symbol} 数据文件不存在: {file_path}")
        
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
            raise ValueError(f"{symbol} 缺少列: {missing_cols}")
        
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
        price_cols = [col for col in result.columns 
                     if any(p in col for p in ['open', 'high', 'low', 'close'])]
        for col in price_cols:
            if (result[col] <= 0).any():
                raise ValueError(f"{col} 包含非正数，无法进行对数变换")
            result[col] = np.log(result[col])
    
    # 按日期排序
    result = result.sort_index()
    
    logger.info(f"加载数据完成: {len(symbols)} 个品种, {len(result)} 条记录")
    return result


class DataManager:
    """
    数据管理器类
    
    统一的数据管理接口，封装所有数据操作
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.futures_dir = self.data_dir / "futures"
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.futures_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_symbol(self, symbol: str, **kwargs) -> pd.DataFrame:
        """获取单个品种数据"""
        return fetch_single_symbol(symbol, **kwargs)
    
    def fetch_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """批量获取多个品种数据"""
        return fetch_multiple_symbols(symbols)
    
    def save_to_parquet(self, data: pd.DataFrame, symbol: str) -> Path:
        """保存数据到Parquet文件"""
        return save_to_parquet(data, symbol, self.futures_dir)
    
    def load_from_parquet(self, symbol: str) -> pd.DataFrame:
        """从Parquet文件加载数据"""
        return load_from_parquet(symbol, self.futures_dir)
    
    def update_symbol(self, symbol: str) -> int:
        """更新单个品种数据，返回新增记录数"""
        success, new_count = atomic_update(symbol, self.futures_dir)
        
        # 记录日志
        if success:
            log_update(symbol, new_count, 'success', data_dir=self.data_dir)
        else:
            log_update(symbol, 0, 'failed', 'Update failed', data_dir=self.data_dir)
        
        return new_count if success else 0
    
    def update_all(self, symbols: List[str]) -> Dict[str, int]:
        """批量更新多个品种数据"""
        results = {}
        
        for symbol in symbols:
            try:
                new_count = self.update_symbol(symbol)
                results[symbol] = new_count
            except Exception as e:
                logger.error(f"更新 {symbol} 失败: {e}")
                results[symbol] = 0
                log_update(symbol, 0, 'failed', str(e), data_dir=self.data_dir)
        
        return results
    
    def get_log_prices(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """获取对数价格数据"""
        return load_data(symbols, log_price=True, data_dir=self.futures_dir, **kwargs)
    
    def get_aligned_data(self, symbols: List[str], log_price: bool = True, **kwargs) -> pd.DataFrame:
        """获取对齐的数据"""
        return load_data(symbols, log_price=log_price, data_dir=self.futures_dir, **kwargs)
    
    def save_metadata(self, data_dict: Dict[str, pd.DataFrame]) -> Path:
        """保存元数据"""
        return save_metadata(data_dict, self.data_dir)
    
    def load_metadata(self) -> Dict:
        """加载元数据"""
        return load_metadata(self.data_dir)
    
    def check_data_quality(self, symbol: str) -> Dict:
        """检查数据质量"""
        try:
            df = self.load_from_parquet(symbol)
            
            return {
                'symbol': symbol,
                'record_count': len(df),
                'date_range': [df['date'].min(), df['date'].max()],
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e)
            }