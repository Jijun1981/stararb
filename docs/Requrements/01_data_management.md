# 数据管理模块需求文档 (Data Management Requirements)

版本: 3.0  
更新日期: 2024-08-23  
状态: 已实施

## 1. 模块概述

数据管理模块负责期货数据的获取、存储、更新和预处理，为策略分析提供高质量的时序数据。

**关键特性:**
- 支持14个金属期货品种的数据管理
- 使用data-joint数据源（聚宽8888主力连续合约）
- 统一的数据接口和命名规范
- 数据质量控制和异常处理

## 2. 用户故事 (User Stories)

### Story 1.1: 数据获取
**作为**量化研究员  
**我希望**能够方便地获取14个金属期货品种的历史数据  
**以便**进行协整分析和策略回测

**验收标准:**
- 批量获取14个品种的数据
- 数据包含OHLCV和持仓量
- 支持指定时间范围
- 处理数据缺失和异常

### Story 1.2: 数据更新  
**作为**策略开发者  
**我希望**系统能够增量更新期货数据  
**以便**保持数据的时效性

**验收标准:**
- 检测本地数据的最新日期
- 仅获取缺失的新数据
- 原子性操作确保数据完整性
- 生成更新日志

### Story 1.3: 数据预处理
**作为**研究员  
**我希望**对原始数据进行标准化预处理  
**以便**为协整分析准备合适的数据格式

**验收标准:**
- 支持对数价格转换
- 处理缺失值和异常值
- 多品种数据对齐
- 生成数据质量报告

## 3. 功能需求 (Requirements)

### REQ-1.1: 数据获取
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-1.1.1 | 支持批量获取14个品种的主力连续合约日线数据(AG,AL,AU,CU,HC,I,NI,PB,RB,SF,SM,SN,SS,ZN) | P0 |
| REQ-1.1.2 | 从data-joint目录读取jq_8888_{SYMBOL}.csv文件获取数据 | P0 |
| REQ-1.1.3 | 数据字段包含：date, open, high, low, close, volume, open_interest | P0 |
| REQ-1.1.4 | 支持指定时间范围(2020-01-01至今) | P0 |
| REQ-1.1.5 | 异常处理：文件不存在时报错、数据验证 | P1 |

### REQ-1.2: 数据存储
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-1.2.1 | 数据存储路径为./data/data-joint/jq_8888_{symbol}.csv | P0 |
| REQ-1.2.2 | 记录元信息（获取时间、数据起止日期、记录数）到metadata.json | P1 |
| REQ-1.2.3 | 支持数据压缩以节省存储空间 | P2 |
| REQ-1.2.4 | 实现数据版本管理，保留历史版本 | P2 |

### REQ-1.3: 数据更新（暂不实现）
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-1.3.1 | 检测本地数据最新日期，仅获取缺失数据 | P0 |
| REQ-1.3.2 | 使用临时文件和原子操作确保更新安全性 | P0 |
| REQ-1.3.3 | 更新失败时回滚，保持原数据不变 | P0 |
| REQ-1.3.4 | 生成更新日志update_log.csv（时间、品种、新增条数、状态） | P1 |
| REQ-1.3.5 | 支持定时自动更新（暂不实现，数据源为静态CSV） | P2 |

### REQ-1.4: 数据预处理
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-1.4.1 | 对数价格转换：log_price = np.log(price) | P0 |
| REQ-1.4.2 | 数据对齐：处理不同品种的交易日差异，生成对齐的DataFrame | P0 |
| REQ-1.4.3 | 缺失值处理：前向填充(ffill)处理停牌等情况 | P0 |
| REQ-1.4.4 | 异常值检测：标记超过5倍标准差的异常价格变动 | P1 |
| REQ-1.4.5 | 生成数据质量报告：缺失率、异常值统计 | P1 |

### REQ-1.5: 配对命名规范
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-1.5.1 | 所有配对使用纯符号命名，格式：`{symbol_x}-{symbol_y}` | P0 |
| REQ-1.5.2 | 符号使用正式期货代码，不带后缀：AG, AL, AU, CU等 | P0 |
| REQ-1.5.3 | 示例：`AL-SN`、`HC-I`、`RB-SF` | P0 |
| REQ-1.5.4 | 禁止在配对名称中包含列名后缀（如_close、_open等） | P0 |

## 4. 接口定义

### 4.1 DataManager类接口
```python
class DataManager:
    def __init__(self, data_dir: str = "./data/data-joint")
    
    # 数据获取
    def load_data(
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: List[str] = ['close'],
        log_price: bool = False,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame
    
    def load_from_csv(self, symbol: str) -> pd.DataFrame
    
    # 数据预处理
    def get_log_prices(self, symbols: List[str]) -> pd.DataFrame
    def get_aligned_data(self, symbols: List[str], log_price: bool = True) -> pd.DataFrame
    def check_data_quality(self, symbol: str) -> Dict
```

### 4.2 数据格式规范
```python
# 原始数据格式
{
    'date': pd.DatetimeIndex,
    'open': float64,
    'high': float64, 
    'low': float64,
    'close': float64,
    'volume': float64,
    'open_interest': float64
}

# 对齐后数据格式（多品种）
# 列名使用纯符号格式
{
    'AG': float64,  # 银
    'AU': float64,  # 金
    'AL': float64,  # 铝
    'CU': float64,  # 铜
    'NI': float64,  # 镍
    'PB': float64,  # 铅
    'SN': float64,  # 锡
    'ZN': float64,  # 锌
    'HC': float64,  # 热卷
    'I': float64,   # 铁矿
    'RB': float64,  # 螺纹钢
    'SF': float64,  # 硅铁
    'SM': float64,  # 锰硅
    'SS': float64,  # 不锈钢
}
```

### 4.3 品种代码映射
```python
SYMBOL_MAPPING = {
    'AG': '银',
    'AU': '金',
    'AL': '铝',
    'CU': '铜',
    'NI': '镍',
    'PB': '铅',
    'SN': '锡',
    'ZN': '锌',
    'HC': '热卷',
    'I': '铁矿',
    'RB': '螺纹钢',
    'SF': '硅铁',
    'SM': '锰硅',
    'SS': '不锈钢'
}

# 文件名格式：jq_8888_{symbol}.csv
# 示例：jq_8888_AG.csv, jq_8888_CU.csv
```

## 5. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 性能 | 单品种5年数据加载时间 | < 1秒 |
| 性能 | 14品种批量加载时间 | < 5秒 |
| 可靠性 | 文件读取异常处理 | 100% |
| 存储 | 单品种5年数据文件大小 | < 5MB |
| 兼容性 | Python版本 | >= 3.8 |

## 6. 测试用例

### TC-1.1: 数据获取测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-1.1.1 | 获取单个品种(RB)历史数据 | 成功返回DataFrame，包含所有必需字段 |
| TC-1.1.2 | 批量获取3个品种数据 | 返回3个品种的对齐DataFrame |
| TC-1.1.3 | 文件不存在情况 | 抛出明确的FileNotFoundError |
| TC-1.1.4 | 无效品种代码 | 抛出明确的错误信息 |

### TC-1.2: 数据预处理测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-1.2.1 | 对数价格转换 | 所有价格正确转换为对数形式 |
| TC-1.2.2 | 多品种数据对齐 | 按交易日正确对齐，缺失值填充 |
| TC-1.2.3 | 异常值检测 | 标记出超过5倍标准差的异常值 |
| TC-1.2.4 | 缺失值处理 | 使用前向填充方法处理缺失值 |

### TC-1.3: 配对命名测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-1.3.1 | 生成配对名称 | 格式为`AL-SN`，不带后缀 |
| TC-1.3.2 | 解析配对名称 | 正确拆分为symbol_x和symbol_y |
| TC-1.3.3 | 验证禁用格式 | 拒绝带_close后缀的配对名称 |

## 7. 数据源说明

**数据源**: data-joint目录（聚宽8888主力连续合约数据）
- 路径：`./data/data-joint/`
- 文件格式：CSV
- 文件命名：`jq_8888_{symbol}.csv`
- 更新频率：静态数据，不自动更新
- 数据范围：2020-01-01至2024-08-20

## 8. 变更历史

| 版本 | 日期 | 变更内容 | 作者 |
|---|---|---|---|
| 1.0 | 2024-01-15 | 初始版本，基于AkShare | 系统 |
| 2.0 | 2024-08-01 | 改为使用data-joint数据源 | 系统 |
| 3.0 | 2024-08-23 | 统一配对命名规范，使用纯符号格式 | 系统 |