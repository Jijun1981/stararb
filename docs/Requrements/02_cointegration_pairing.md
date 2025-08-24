# 模块2: 协整配对需求文档

## 1. 模块概述
负责协整检验、配对筛选、方向判定和参数估计，生成可交易的配对列表及其统计特征。

## 2. 用户故事 (User Stories)

### Story 2.1: 多时间窗口协整检验
**作为**研究员  
**我希望**对所有可能的品种配对进行可配置的多时间窗口协整检验  
**以便**灵活评估不同时间尺度的配对关系

**验收标准:**
- 支持Engle-Granger两步法检验
- 支持自定义时间窗口列表（如1年、2年、3年等）
- 评估协整关系在不同时间尺度的稳定性
- 生成多时间窗口协整报告

### Story 2.2: 配对品种确定
**作为**研究员  
**我希望**基于指定时间段数据的波动率直接确定配对中的X和Y品种  
**以便**灵活控制分析时间范围

**验收标准:**
- 支持指定任意时间段计算波动率
- 低波动品种直接确定为symbol_x
- 高波动品种直接确定为symbol_y  
- 记录波动率信息和计算时间段

### Story 2.3: 配对参数估计
**作为**研究员  
**我希望**估计配对的关键参数  
**以便**为后续交易策略提供基础

**验收标准:**
- 计算协整系数β
- 估计均值回归半衰期
- 计算残差统计特征
- 评估配对稳定性

### Story 2.4: 批量配对筛选
**作为**研究员  
**我希望**使用灵活的筛选条件批量筛选优质配对  
**以便**根据不同策略需求构建配对池

**验收标准:**
- 测试所有可能配对组合
- 支持自定义p值阈值筛选
- 支持指定筛选的时间窗口（如仅用1年数据筛选）
- 支持多重筛选条件组合
- 生成配对排名列表
- 导出配对参数表

## 3. 功能需求 (Requirements)

### REQ-2.1: 协整检验
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-2.1.1 | 实现Engle-Granger两步法：第一步OLS回归，第二步ADF检验残差 | P0 |
| REQ-2.1.2 | 支持自定义时间窗口列表进行多窗口检验 | P0 |
| REQ-2.1.3 | 默认时间窗口：[252, 504, 756, 1008, 1260]个交易日 | P0 |
| REQ-2.1.4 | 支持指定任意交易日数量作为时间窗口 | P0 |
| REQ-2.1.5 | 当数据不足时自动跳过对应窗口 | P0 |
| REQ-2.1.6 | p值阈值可配置，默认0.05 | P0 |
| REQ-2.1.7 | 支持自定义检验参数（滞后阶数、趋势项） | P1 |

### REQ-2.2: 品种角色确定
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-2.2.1 | 支持指定任意时间段计算波动率 | P0 |
| REQ-2.2.2 | 计算对数收益率：returns = np.diff(np.log(prices)) | P0 |
| REQ-2.2.3 | 计算年化波动率：vol = std(returns) * sqrt(252) | P0 |
| REQ-2.2.4 | 波动率比较：低波动直接作为symbol_x，高波动作为symbol_y | P0 |
| REQ-2.2.5 | 支持start_date和end_date参数指定计算时间窗口 | P0 |
| REQ-2.2.6 | 默认使用全部可用数据，支持灵活时间段指定 | P0 |

### REQ-2.3: 参数估计
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-2.3.1 | 对每个时间窗口分别进行OLS回归估计β系数 | P0 |
| REQ-2.3.2 | 计算每个窗口的半衰期：halflife = -log(2) / λ (λ为AR(1)系数) | P0 |
| REQ-2.3.3 | 残差统计：均值、标准差、偏度、峰度 | P0 |
| REQ-2.3.4 | 记录不同时间窗口β的稳定性和变化趋势 | P0 |
| REQ-2.3.5 | 计算R²和调整R²评估拟合优度 | P1 |
| REQ-2.3.6 | 残差正态性检验(Jarque-Bera test) | P1 |

### REQ-2.4: 批量筛选
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-2.4.1 | 生成所有可能配对：C(n,2)个组合 | P0 |
| REQ-2.4.2 | 并行处理提高效率（使用multiprocessing） | P1 |
| REQ-2.4.3 | 支持灵活的筛选条件配置 | P0 |
| REQ-2.4.4 | 支持指定筛选用的时间窗口（如仅用1年数据） | P0 |
| REQ-2.4.5 | 支持多个时间窗口的组合筛选条件 | P0 |
| REQ-2.4.6 | p值阈值可配置，默认0.05 | P0 |
| REQ-2.4.7 | 支持自定义排序字段和排序方向 | P0 |
| REQ-2.4.8 | 记录所有计算的时间窗口的p值供分析 | P0 |
| REQ-2.4.9 | 导出包含多时间窗口结果的配对参数文件 | P0 |

## 4. 接口定义

### 4.1 CointegrationAnalyzer类接口
```python
class CointegrationAnalyzer:
    def __init__(self, data: pd.DataFrame)
    
    # 协整检验
    def engle_granger_test(self, x: np.ndarray, y: np.ndarray) -> Dict
    def multi_window_test(self, x: np.ndarray, y: np.ndarray, 
                         windows: Optional[Dict[str, int]] = None) -> Dict
    def adf_test(self, series: np.ndarray) -> Tuple[float, float]
    
    # 品种角色确定
    def calculate_volatility(self, log_prices: np.ndarray, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> float
    def determine_symbols(self, symbol1: str, symbol2: str,
                         vol_start_date: Optional[str] = None,
                         vol_end_date: Optional[str] = None) -> Tuple[str, str]
    
    # 参数估计
    def estimate_parameters(self, x: np.ndarray, y: np.ndarray) -> Dict
    def calculate_halflife(self, residuals: np.ndarray) -> float
    def residual_statistics(self, residuals: np.ndarray) -> Dict
    
    # 批量筛选
    def screen_all_pairs(self,
                        screening_windows: Optional[List[str]] = None,
                        p_thresholds: Optional[Dict[str, float]] = None,
                        filter_logic: str = 'AND',
                        sort_by: str = 'pvalue_1y',
                        ascending: bool = True,
                        vol_start_date: Optional[str] = None,
                        vol_end_date: Optional[str] = None,
                        windows: Optional[Dict[str, int]] = None) -> pd.DataFrame
    def get_top_pairs(self, n: int = 20, **kwargs) -> pd.DataFrame
    def export_results(self, filepath: str) -> None
```

### 4.2 配对参数格式
```python
{
    'pair': 'AG-AU',             # 配对名称
    'symbol_x': 'AG',            # 低波动品种
    'symbol_y': 'AU',            # 高波动品种
    'pvalue_1y': 0.0113,         # 1年p值
    'pvalue_2y': 0.0156,         # 2年p值
    'pvalue_3y': 0.0089,         # 3年p值
    'pvalue_4y': 0.0025,         # 4年p值
    'pvalue_5y': 0.0018,         # 5年p值
    'beta_1y': 0.5210,           # 1年β系数
    'beta_2y': 0.6234,           # 2年β系数
    'beta_3y': 0.7856,           # 3年β系数
    'beta_4y': 0.8123,           # 4年β系数
    'beta_5y': 0.8498,           # 5年β系数
    'halflife_1y': 12.6,         # 1年半衰期
    'halflife_5y': 45.3,         # 5年半衰期
    'volatility_x': 0.0019,      # X波动率(最近1年)
    'volatility_y': 0.0025,      # Y波动率(最近1年)
}
```

## 5. 算法细节

### 5.1 Engle-Granger检验流程
```
1. 数据准备：获取两个品种的对数价格序列
2. 品种角色确定：基于指定时间段数据的波动率，低波动作X，高波动作Y
3. 多时间窗口检验：
   对于每个配置的窗口（可自定义，如252d, 504d, 756d等）：
   a. 截取对应长度的历史数据
   b. OLS回归：Y = α + β*X + ε (统一Y对X回归)
   c. 提取残差：ε = Y - (α + β*X)
   d. ADF检验：检验残差平稳性
   e. 记录p值和β系数
4. 筛选判定：
   - 支持指定筛选窗口（如仅用1年数据筛选）
   - 支持多窗口组合筛选（AND/OR逻辑）
   - p值阈值可配置（默认0.05）
   - 提供所有计算窗口的结果供分析
   - 支持自定义排序字段和方向
```

### 5.2 半衰期计算方法
```
1. 对残差进行AR(1)回归：
   Δε_t = λ * ε_{t-1} + u_t
2. 估计λ参数
3. 计算半衰期：
   halflife = -log(2) / λ
4. 半衰期表示偏离均值后回归到一半偏离值所需的时间
```

## 6. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 性能 | 91个配对×5个时间窗口协整检验总耗时 | < 60秒 |
| 准确性 | p值计算精度 | 小数点后6位 |
| 稳定性 | 数值计算异常处理 | 100%覆盖 |
| 内存 | 多窗口计算峰值内存占用 | < 2GB |
| 可扩展 | 支持自定义检验方法 | 插件式架构 |

## 7. 测试用例

### TC-2.1: 协整检验测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-2.1.1 | 已知协整配对(模拟数据) | 所有窗口p值 < 0.01 |
| TC-2.1.2 | 随机游走序列 | 所有窗口p值 > 0.5 |
| TC-2.1.3 | 完全相关序列 | 所有窗口p值接近0 |
| TC-2.1.4 | 数据长度不足5年 | 自动调整可用窗口 |
| TC-2.1.5 | 多窗口β稳定性 | β变化幅度在合理范围 |

### TC-2.2: 品种角色确定测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-2.2.1 | 使用全部数据计算波动率 | 使用所有可用交易日 |
| TC-2.2.2 | symbol1波动率0.01, symbol2波动率0.02 | symbol_x=symbol1, symbol_y=symbol2 |
| TC-2.2.3 | symbol1波动率0.02, symbol2波动率0.01 | symbol_x=symbol2, symbol_y=symbol1 |
| TC-2.2.4 | 波动率相等 | 按字母顺序确定X和Y |
| TC-2.2.5 | 指定start_date='2023-01-01' | 使用2023年至最新数据计算波动率 |
| TC-2.2.6 | 指定start_date和end_date | 使用指定时间段数据计算波动率 |
| TC-2.2.7 | start_date早于数据起始 | 从数据起始日期开始计算 |

### TC-2.3: 批量筛选测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-2.3.1 | n个品种批量筛选 | 生成C(n,2)个配对×配置窗口数结果 |
| TC-2.3.2 | 自定义筛选条件 | 按指定窗口和阈值筛选 |
| TC-2.3.3 | 多窗口组合筛选 | AND/OR逻辑正确应用 |
| TC-2.3.4 | 结果排序 | 按指定字段和方向排序 |
| TC-2.3.5 | 自定义时间窗口 | 使用自定义窗口配置进行计算 |
| TC-2.3.6 | 多窗口数据完整性 | 每个配对包含所有配置窗口的数据 |

## 8. 依赖关系
- **上游模块**: 数据管理模块（提供对数价格数据）
- **下游模块**: 信号生成模块（使用配对参数）
- **Python包**: statsmodels, scipy, numpy, pandas
- **统计方法**: OLS回归、ADF检验、AR模型