# 模块3: Beta估计需求文档

## 1. 模块概述
负责配对β系数的一次性标定（one-shot calibration），通过训练段多候选方法生成、验证段样本外评分，为每个配对选出最优的初始β*和Q/R参数，供信号生成模块使用。

**重要原则**：
- **评估与交易分离**：硬门槛评估必须在静态β的残差上进行，而非KF动态残差
- **超参数优化**：通过创新诊断优化Q/R参数，确保创新序列白化（均值≈0，标准差≈1）
- **二维状态空间**：KF必须包含截距项c_t，避免均值偏置被β_t背锅

## 2. 用户故事 (User Stories)

### Story 5.1: 一次性β标定
**作为**研究员  
**我希望**对所有配对进行一次性β标定  
**以便**获得稳健的初始参数用于整段回测

**验收标准:**
- 使用训练段(2020-2023)生成5种候选β
- 在验证段(2024-2025)进行样本外评分
- 为每个配对选出最优β*和对应参数
- 输出标准化的参数文件

### Story 5.2: 多候选方法生成
**作为**研究员  
**我希望**使用多种方法估计β  
**以便**覆盖不同的估计思路和适用场景

**验收标准:**
- 实现5种β估计方法：OLS(2y)、EWLS、FM-OLS、Min-HL、Huber
- 每种方法都包含截距项
- 记录每种方法的详细参数和诊断信息
- 支持方向自动选择（Y~X vs X~Y）

### Story 5.3: 样本外验证
**作为**研究员  
**我希望**在样本外数据上评估β质量  
**以便**避免过拟合并选择真正稳健的参数

**验收标准:**
- 基于协整性、稳定性、可交易性的综合评分
- 硬性门槛筛选（ADF p值、半衰期、变异系数）**仅在静态β残差上评估**
- 生成详细的评估报告
- 对不合格配对给出明确的弃用建议

### Story 5.4: Kalman Filter参数标定
**作为**研究员  
**我希望**优化KF的Q/R超参数  
**以便**获得白化的创新序列用于交易信号

**验收标准:**
- 实现创新诊断功能（均值、标准差、Ljung-Box检验）
- 自动调整Q/R使创新序列白化（mean≈0, std≈1）
- 支持二维状态空间[β_t, c_t]
- 输出优化后的Q矩阵和R值

## 3. 功能需求 (Requirements)

### REQ-5.1: 数据分割
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-5.1.1 | 训练段：2020-01-01至2023-12-31，用于生成候选β | P0 |
| REQ-5.1.2 | 验证段：2024-01-01至2025-08-20，用于样本外评分 | P0 |
| REQ-5.1.3 | 确保数据完整性，处理缺失值和异常值 | P0 |
| REQ-5.1.4 | 统一交易日历，支持Asia/Shanghai时区 | P1 |

### REQ-5.2: 候选β生成
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-5.2.1 | OLS(2y)：使用训练段末尾2年数据（2022-2023） | P0 |
| REQ-5.2.2 | EWLS：整个训练段，半衰期可配置(63/126/252日) | P0 |
| REQ-5.2.3 | FM-OLS：使用linearmodels包，默认Newey-West带宽 | P0 |
| REQ-5.2.4 | Min-HL：在[0.5β_OLS, 1.5β_OLS]网格搜索，ADF p<0.05门槛 | P0 |
| REQ-5.2.5 | Huber：稳健回归，δ=1.35或MAD自适应 | P0 |
| REQ-5.2.6 | 方向选择：比较Y~X和X~Y的ADF p值，选择更小者 | P0 |
| REQ-5.2.7 | 方向一致性：确定后所有回归、信号、手数必须沿用同一方向 | P0 |

### REQ-5.3: 样本外评估（静态β）
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-5.3.1 | 硬门槛1：**静态β残差**ADF p值 < 0.15（必须regression='n'）【2025-08-20优化】 | P0 |
| REQ-5.3.2 | 硬门槛2：**静态β残差**半衰期在[2, 60]交易日范围内【2025-08-20优化】 | P0 |
| REQ-5.3.3 | 硬门槛3：验证段**静态β**稳健CV ≤ 0.40【2025-08-20优化】 | P0 |
| REQ-5.3.10 | Purged Walk-Forward验证：禁用PWF，使用单折验证【2025-08-20优化】 | P0 |
| REQ-5.3.11 | 跨折评分：使用均值±标准差衡量稳定性 | P0 |
| REQ-5.3.12 | p值分级惩罚：p∈[0.01,0.05)轻惩罚-0.05分 | P1 |
| REQ-5.3.4 | 两步选择：先筛选（淘汰不合格），再排序（按IR/稳定性） | P0 |
| REQ-5.3.5 | 最终选择：IR最优选单一方法，性能相近取加权平均 | P0 |
| REQ-5.3.6 | ADF检验参数：autolag='AIC', maxlag=int(T**(1/3)) | P0 |
| REQ-5.3.7 | 稳健CV计算：cv_robust = 1.4826*MAD / max(abs(median), 1e-6) | P0 |
| REQ-5.3.8 | 半衰期计算：HL = -ln(2)/ln|ρ|，拒绝|ρ|≥1的情况 | P0 |
| REQ-5.3.9 | 加权平均参数：当IR差异<10%时，按IR比例加权 | P1 |

### REQ-5.4: KF参数标定
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-5.4.1 | 使用训练段数据初始化KF，状态向量[β*, c*] | P0 |
| REQ-5.4.2 | 优化Q/R使创新白化：|mean(z)|<0.1, 0.9<std(z)<1.1 | P0 |
| REQ-5.4.3 | 三点式调参：std(z)<0.9则Q×3；std(z)>1.1则Q÷2或R×2 | P0 |
| REQ-5.4.4 | Ljung-Box检验创新序列独立性，p_LB>0.2 | P1 |
| REQ-5.4.5 | 输出优化后的Q矩阵（2×2）和R值 | P0 |

### REQ-5.5: 参数输出
| ID | 需求描述 | 优先级 |
|---|---|---|
| REQ-5.5.1 | 输出每个配对的最优β*和c*初值 | P0 |
| REQ-5.5.2 | 输出优化后的Q/R参数用于KF | P0 |
| REQ-5.4.3 | 生成标准化的参数文件（JSON/CSV格式） | P0 |
| REQ-5.4.4 | 记录详细的评估指标和选择理由 | P1 |
| REQ-5.4.5 | 识别并标记需要弃用的配对 | P0 |
| REQ-5.4.6 | 输出方向一致性：包含direction和spread_formula字段 | P0 |

## 4. 接口定义

### 4.1 BetaCalibrator类接口
```python
class BetaCalibrator:
    def __init__(self, train_start: str = "2020-01-01", 
                 train_end: str = "2023-12-31",
                 valid_start: str = "2024-01-01",
                 valid_end: str = "2025-08-20")
    
    # 数据准备
    def load_and_split_data(self, pair_data: Dict[str, pd.DataFrame]) -> Dict
    def choose_direction(self, train_data: pd.DataFrame) -> str  # 'y_on_x' or 'x_on_y'
    
    # 候选β生成
    def fit_ols_2y(self, train_data: pd.DataFrame) -> Dict
    def fit_ewls(self, train_data: pd.DataFrame, halflife: int = 126) -> Dict
    def fit_fm_ols(self, train_data: pd.DataFrame) -> Dict
    def fit_min_halflife(self, train_data: pd.DataFrame, grid_points: int = 201) -> Dict
    def fit_huber(self, train_data: pd.DataFrame, delta: float = 1.35) -> Dict
    
    # 评估验证（静态β）
    def evaluate_on_validation(self, beta: float, alpha: float, 
                              valid_data: pd.DataFrame) -> Dict
    def calculate_comprehensive_score(self, metrics: Dict) -> float
    def select_best_beta(self, candidates: List[Dict], metrics: List[Dict]) -> Dict
    
    # KF参数标定
    def calibrate_kf_parameters(self, train_data: pd.DataFrame, 
                               beta_star: float, alpha_star: float) -> Dict
    def kf_innovation_diagnostics(self, innovations: np.ndarray) -> Dict
    def optimize_qr_for_whitening(self, train_data: pd.DataFrame,
                                 beta_star: float, alpha_star: float) -> Tuple[np.ndarray, float]
    
    # 批量处理
    def calibrate_all_pairs(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict
    def export_parameters(self, results: Dict, filepath: str) -> None
```

### 4.2 候选β结果格式
```python
{
    'method': 'EWLS',           # 方法名称
    'beta': 0.8523,             # β系数
    'alpha': -0.0012,           # 截距
    'window': 'full_train',     # 使用的数据窗口
    'r_squared': 0.85,          # 拟合优度
    'residuals': np.array,      # 残差序列
    'diagnostics': {            # 诊断信息
        'halflife': 23.5,
        'adf_pvalue': 0.008,
        'durbin_watson': 2.01
    }
}
```

### 4.3 验证评估结果格式
```python
{
    'adf_pvalue': 0.008,        # ADF检验p值
    'halflife': 23.5,           # 半衰期(交易日)
    'kappa': 0.029,             # OU速度参数
    'resid_var': 0.0001,        # 残差方差
    'beta_cv126': 0.12,         # 126日滚动β变异系数
    'beta_drift': 0.05,         # 训练到验证的β漂移
    'score': 0.78,              # 综合评分
    'pass_gates': True,         # 是否通过硬门槛
    'selected': True            # 是否被选中
}
```

### 4.4 最终参数输出格式
```python
{
    'pair': 'CU0-SN0',
    'direction': 'y_on_x',      # CU对SN回归
    'spread_formula': 'log(CU0) - 0.8523*log(SN0) - (-0.0012)',  # 明确公式
    'beta_star': 0.8523,        # 选定的β*
    'alpha_star': -0.0012,      # 选定的α*
    'selected_method': 'EWLS',  # 获胜方法
    'Q_matrix': [[1e-5, 0], [0, 1e-6]],  # 状态噪声
    'R_value': 0.0001,          # 观测噪声
    'validation_score': 0.78,   # 验证段得分
    'status': 'selected',       # selected/rejected
    'reason': 'Best score in validation'
}
```

## 5. 算法细节

### 5.1 五种候选β方法

#### 方法1: OLS(2y)
```python
def fit_ols_2y(train_data):
    # 使用训练段末尾2年数据 (约504个交易日)
    recent_data = train_data.tail(504)
    log_Y = np.log(recent_data['Y'])
    log_X = np.log(recent_data['X'])
    
    # 带截距的OLS回归: log(Y) = α + β*log(X) + ε
    X_matrix = np.column_stack([np.ones(len(log_X)), log_X])
    coeffs = np.linalg.lstsq(X_matrix, log_Y, rcond=None)[0]
    
    return {'beta': coeffs[1], 'alpha': coeffs[0]}
```

#### 方法2: EWLS (指数加权最小二乘)
```python
def fit_ewls(train_data, halflife=126):
    # 数值优化版本，避免大矩阵构造
    log_Y = np.log(train_data['Y']).values
    log_X = np.log(train_data['X']).values
    
    T = len(log_Y)
    lambda_param = np.exp(-np.log(2) / halflife)
    # 远期更小，近期更大
    weights = lambda_param ** np.arange(T-1, -1, -1)
    
    # 使用加权正规方程避免构造对角矩阵
    X_matrix = np.column_stack([np.ones(T), log_X])
    Xw = X_matrix * np.sqrt(weights[:, None])
    yw = log_Y * np.sqrt(weights)
    
    coeffs = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    
    return {
        'alpha': float(coeffs[0]), 
        'beta': float(coeffs[1]),
        'halflife_used': halflife
    }
```

#### 方法3: FM-OLS (Phillips-Hansen)
```python
def fit_fm_ols(train_data):
    # 使用linearmodels包的FM-OLS实现
    from linearmodels.cointegration import FMOLS
    
    log_Y = np.log(train_data['Y']).values
    log_X = np.log(train_data['X']).values
    
    # FM-OLS估计，默认Newey-West长期方差
    res = FMOLS(log_Y, log_X).fit()
    
    return {
        'beta': float(res.params[0]), 
        'alpha': float(res.params.const)
    }
```

#### 方法4: Min-Halflife网格搜索
```python
def fit_min_halflife(train_data, beta_ols_2y, grid_points=201):
    log_Y = np.log(train_data['Y'])
    log_X = np.log(train_data['X'])
    
    # 搜索范围：[0.5*β_OLS, 1.5*β_OLS]
    beta_range = np.linspace(0.5 * beta_ols_2y, 1.5 * beta_ols_2y, grid_points)
    
    best_beta = None
    min_halflife = float('inf')
    
    for beta in beta_range:
        # 固定β，计算对应的α
        alpha = np.mean(log_Y - beta * log_X)
        residuals = log_Y - alpha - beta * log_X
        
        # AR(1)回归估计半衰期（无截距）
        rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), 
                              residuals[1:], rcond=None)[0][0]
        
        # 拒绝非平稳情况
        if abs(rho) >= 1:
            continue  # 跳过该β值
            
        # 使用严格的AR(1)半衰期公式
        halflife = -np.log(2) / np.log(abs(rho))
            
            # ADF检验
            adf_pvalue = adf_test(residuals)
            
            if adf_pvalue < 0.05 and 2 <= halflife <= 60:
                if halflife < min_halflife:
                    min_halflife = halflife
                    best_beta = beta
    
    return {'beta': best_beta, 'halflife': min_halflife}
```

#### 方法5: Huber稳健回归
```python
def fit_huber(train_data, delta=1.35):
    from sklearn.linear_model import HuberRegressor
    
    log_Y = np.log(train_data['Y'])
    log_X = np.log(train_data['X']).reshape(-1, 1)
    
    # Huber回归
    huber = HuberRegressor(epsilon=delta, fit_intercept=True)
    huber.fit(log_X, log_Y)
    
    return {'beta': huber.coef_[0], 'alpha': huber.intercept_}
```

### 5.2 样本外评估流程
```python
def evaluate_on_validation(beta, alpha, valid_data):
    """在静态β残差上评估（不使用KF）"""
    log_Y = np.log(valid_data['Y'])
    log_X = np.log(valid_data['X'])
    
    # 计算静态残差（固定β）
    residuals = log_Y - alpha - beta * log_X
    
    # 1. ADF检验（必须regression='n'）
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(residuals, regression='n', autolag='AIC')
    adf_pvalue = adf_result[1]
    
    # 2. 半衰期计算（严格公式）
    rho = np.linalg.lstsq(residuals[:-1].reshape(-1, 1), 
                          residuals[1:], rcond=None)[0][0]
    if abs(rho) >= 1:
        halflife = float('inf')  # 非平稳，设为无穷大
    else:
        halflife = -np.log(2) / np.log(abs(rho))
    
    # 3. OU速度
    kappa = -np.log(abs(rho))
    
    # 4. 残差方差
    resid_var = np.var(residuals)
    
    # 5. β稳定性（126日滚动窗口）
    rolling_betas = []
    for i in range(126, len(valid_data)):
        window_data = valid_data.iloc[i-126:i]
        window_log_Y = np.log(window_data['Y'])
        window_log_X = np.log(window_data['X'])
        X_matrix = np.column_stack([np.ones(126), window_log_X])
        beta_rolling = np.linalg.lstsq(X_matrix, window_log_Y, rcond=None)[0][1]
        rolling_betas.append(beta_rolling)
    
    beta_cv126 = np.std(rolling_betas) / np.mean(rolling_betas)
    
    # 6. 训练到验证的漂移
    # 需要传入训练段的β进行比较
    
    return {
        'adf_pvalue': adf_pvalue,
        'halflife': halflife,
        'kappa': kappa,
        'resid_var': resid_var,
        'beta_cv126': beta_cv126
    }
```

### 5.3 两步选择算法
```python
def select_best_beta_two_step(candidates_metrics):
    """
    两步选择最优β：先筛选，再排序
    
    Args:
        candidates_metrics: Dict[method_name, metrics_dict]
            包含5种方法的验证段评估指标
    
    Returns:
        selected_result: Dict 包含选中的方法和参数
    """
    
    # Step 1: 筛选 - 淘汰不合格的方法
    qualified_methods = {}
    
    for method, metrics in candidates_metrics.items():
        # 硬性门槛检查（统一使用p<0.05）
        if (metrics['adf_pvalue'] < 0.05 and           # 协整性
            2 <= metrics['halflife'] <= 60 and          # 可交易性
            metrics['beta_cv126'] <= 0.35):             # 稳定性
            
            # 计算验证段的Information Ratio
            validation_returns = metrics['validation_returns']  # 日收益率序列
            ir = np.mean(validation_returns) / np.std(validation_returns) * np.sqrt(252)
            
            qualified_methods[method] = {
                'metrics': metrics,
                'ir': ir,
                'beta': metrics['beta'],
                'alpha': metrics['alpha']
            }
    
    # 如果没有合格的方法，标记配对为弃用
    if len(qualified_methods) == 0:
        return {
            'status': 'rejected',
            'reason': 'No method passed validation thresholds',
            'selected_method': None
        }
    
    # Step 2: 排序 - 按IR选择最优
    sorted_methods = sorted(qualified_methods.items(), 
                          key=lambda x: x[1]['ir'], 
                          reverse=True)
    
    best_method = sorted_methods[0][0]
    best_ir = sorted_methods[0][1]['ir']
    
    # 检查是否需要加权平均（多个方法性能相近）
    similar_methods = []
    for method, data in sorted_methods:
        # IR差异在10%以内认为性能相近
        if abs(data['ir'] - best_ir) / abs(best_ir) < 0.10:
            similar_methods.append((method, data))
    
    if len(similar_methods) > 1:
        # 多个方法性能相近，计算加权平均β
        return calculate_weighted_average_beta(similar_methods)
    else:
        # 单一方法明显优于其他
        return {
            'status': 'selected',
            'selected_method': best_method,
            'beta_star': qualified_methods[best_method]['beta'],
            'alpha_star': qualified_methods[best_method]['alpha'],
            'validation_ir': best_ir,
            'reason': f'{best_method} has best IR: {best_ir:.3f}'
        }

def calculate_weighted_average_beta(similar_methods):
    """
    计算IR加权平均的β
    
    Args:
        similar_methods: List[(method_name, method_data)]
    
    Returns:
        weighted_result: Dict
    """
    # 提取IR值用于计算权重
    irs = [data['ir'] for _, data in similar_methods]
    total_ir = sum(irs)
    
    # 计算IR比例权重
    weights = [ir / total_ir for ir in irs]
    
    # 计算加权平均β和α
    weighted_beta = sum(w * data['beta'] 
                       for w, (_, data) in zip(weights, similar_methods))
    weighted_alpha = sum(w * data['alpha'] 
                        for w, (_, data) in zip(weights, similar_methods))
    
    # 记录参与平均的方法和权重
    methods_weights = {method: weight 
                      for (method, _), weight in zip(similar_methods, weights)}
    
    return {
        'status': 'selected',
        'selected_method': 'weighted_average',
        'methods_weights': methods_weights,
        'beta_star': weighted_beta,
        'alpha_star': weighted_alpha,
        'validation_ir': np.mean(irs),
        'reason': f'Weighted average of {len(similar_methods)} similar methods'
    }
```

### 5.4 KF参数标定算法
```python
def calibrate_kf_parameters(train_data, beta_star, alpha_star):
    """
    标定KF的Q/R参数，确保创新序列白化
    
    Args:
        train_data: 训练段数据
        beta_star: 选定的最优β
        alpha_star: 选定的最优α
        
    Returns:
        优化后的Q矩阵和R值
    """
    from filterpy.kalman import KalmanFilter
    
    log_Y = np.log(train_data['Y']).values
    log_X = np.log(train_data['X']).values
    
    # 初始化二维KF
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([beta_star, alpha_star])  # 初始状态 [β*, c*]
    
    # 初始Q/R（默认值）
    q_beta = 1e-5
    q_alpha = 1e-6
    R = 1e-4
    
    # 三点式调参循环
    for iteration in range(10):  # 最多10次迭代
        # 设置当前Q/R
        kf.Q = np.diag([q_beta, q_alpha])
        kf.R = np.array([[R]])
        
        # 运行KF收集创新
        innovations = []
        for t in range(len(log_Y)):
            # 观测矩阵 H_t = [x_t, 1]
            kf.H = np.array([[log_X[t], 1.0]])
            
            # 预测
            kf.predict()
            
            # 计算创新和标准化创新
            y_pred = kf.H @ kf.x_prior
            innovation = log_Y[t] - y_pred
            S_t = kf.H @ kf.P_prior @ kf.H.T + kf.R
            z_t = innovation / np.sqrt(S_t)
            innovations.append(float(z_t))
            
            # 更新
            kf.update(log_Y[t])
        
        # 诊断创新序列
        diag = kf_innovation_diagnostics(np.array(innovations))
        
        # 检查是否白化
        if abs(diag['mean']) < 0.1 and 0.9 < diag['std'] < 1.1:
            break  # 已经白化，退出
        
        # 三点式调参
        if diag['std'] < 0.9:
            # 创新方差太小，增大Q或减小R
            q_beta *= 3
            q_alpha *= 3
        elif diag['std'] > 1.1:
            # 创新方差太大，减小Q或增大R
            q_beta /= 2
            R *= 2
        
        if abs(diag['mean']) > 0.1:
            # 均值偏离，微调R
            R *= 1.5 if diag['mean'] > 0 else 0.7
    
    return {
        'Q_matrix': [[q_beta, 0], [0, q_alpha]],
        'R_value': R,
        'innovation_diagnostics': diag
    }

def kf_innovation_diagnostics(innovations):
    """
    创新序列诊断
    
    Args:
        innovations: 标准化创新序列 z_t
        
    Returns:
        诊断结果字典
    """
    import numpy as np
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    mean = float(np.mean(innovations))
    std = float(np.std(innovations, ddof=1))
    
    # Ljung-Box检验（lag=5）
    try:
        lb_result = acorr_ljungbox(innovations, lags=[5], return_df=False)
        p_lb = float(lb_result[1][0])
    except:
        p_lb = np.nan
    
    return {
        'mean': mean,
        'std': std,
        'p_lb': p_lb,
        'is_white': abs(mean) < 0.1 and 0.9 < std < 1.1 and p_lb > 0.2
    }
```

### 5.5 选择逻辑示例
```python
# 示例：CU0-SN0配对的β选择过程
candidates_metrics = {
    'OLS_2y': {
        'beta': 0.8451, 'alpha': -0.0012,
        'adf_pvalue': 0.012,    # 通过
        'halflife': 25.3,        # 通过
        'beta_cv126': 0.28,      # 通过
        'validation_returns': [...],  # 验证段日收益率
        # 计算得IR = 1.45
    },
    'EWLS': {
        'beta': 0.8523, 'alpha': -0.0015,
        'adf_pvalue': 0.008,    # 通过
        'halflife': 23.5,        # 通过 
        'beta_cv126': 0.25,      # 通过
        'validation_returns': [...],
        # 计算得IR = 1.52  (最高)
    },
    'FM_OLS': {
        'beta': 0.8602, 'alpha': -0.0018,
        'adf_pvalue': 0.065,    # 不通过(>0.05)
        'halflife': 28.1,
        'beta_cv126': 0.31,
        'validation_returns': [...],
        # 被筛选掉，不计算IR
    },
    'Min_HL': {
        'beta': 0.8234, 'alpha': -0.0008,
        'adf_pvalue': 0.009,    # 通过
        'halflife': 18.2,        # 通过
        'beta_cv126': 0.42,      # 不通过(>0.35)
        'validation_returns': [...],
        # 被筛选掉，不计算IR
    },
    'Huber': {
        'beta': 0.8489, 'alpha': -0.0014,
        'adf_pvalue': 0.011,    # 通过
        'halflife': 24.8,        # 通过
        'beta_cv126': 0.26,      # 通过
        'validation_returns': [...],
        # 计算得IR = 1.48
    }
}

# 执行两步选择
result = select_best_beta_two_step(candidates_metrics)

# 结果解析：
# Step 1筛选后：OLS_2y(IR=1.45), EWLS(IR=1.52), Huber(IR=1.48)通过
# Step 2排序后：EWLS最优，IR=1.52
# IR差异检查：1.48和1.45相对1.52的差异都>10%
# 最终选择：单一方法EWLS

print(result)
# {
#     'status': 'selected',
#     'selected_method': 'EWLS',
#     'beta_star': 0.8523,
#     'alpha_star': -0.0015,
#     'validation_ir': 1.52,
#     'reason': 'EWLS has best IR: 1.520'
# }

# 如果EWLS的IR=1.52, Huber的IR=1.50（差异<10%）
# 则会触发加权平均：
# {
#     'status': 'selected',
#     'selected_method': 'weighted_average',
#     'methods_weights': {'EWLS': 0.503, 'Huber': 0.497},
#     'beta_star': 0.8506,  # 加权平均
#     'alpha_star': -0.00145,
#     'validation_ir': 1.51,
#     'reason': 'Weighted average of 2 similar methods'
# }
```

## 6. 非功能需求

| 需求类型 | 描述 | 目标值 |
|---|---|---|
| 性能 | 单配对5种方法标定 | < 30秒 |
| 性能 | 70个配对批量标定 | < 30分钟 |
| 准确性 | β估计精度 | 小数点后6位 |
| 可靠性 | 异常配对处理覆盖率 | 100% |
| 可扩展性 | 支持新增β估计方法 | 插件式架构 |

## 7. 测试用例

### TC-5.1: 候选β生成测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-5.1.1 | 正常配对数据 | 生成5个有效候选β |
| TC-5.1.2 | 数据长度不足5年 | 自动调整可用方法 |
| TC-5.1.3 | 极端价格比例 | Huber方法更稳健 |
| TC-5.1.4 | 无协整关系 | 所有方法都标记为不合格 |

### TC-5.2: 样本外评估测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-5.2.1 | 稳定协整关系 | 通过所有硬门槛 |
| TC-5.2.2 | 不稳定关系 | ADF检验失败 |
| TC-5.2.3 | 半衰期过长 | 超出60日限制 |
| TC-5.2.4 | β漂移严重 | 变异系数超过0.35 |

### TC-5.3: 两步选择测试
| 测试ID | 场景 | 预期结果 |
|---|---|---|
| TC-5.3.1 | 3个方法通过筛选，IR差异明显 | 选择IR最高的单一方法 |
| TC-5.3.2 | 所有方法都不通过硬门槛 | 标记配对为rejected |
| TC-5.3.3 | 2个方法IR差异<10% | 计算IR加权平均β |
| TC-5.3.4 | 只有1个方法通过筛选 | 直接选择该方法 |
| TC-5.3.5 | ADF p值边界情况(0.049 vs 0.051) | 严格按0.05阈值筛选 |

## 8. 输出文件格式

### 8.1 参数文件 (beta_parameters.json)
```json
{
    "calibration_info": {
        "train_period": "2020-01-01 to 2023-12-31",
        "validation_period": "2024-01-01 to 2025-08-20",
        "total_pairs": 91,
        "selected_pairs": 67,
        "rejected_pairs": 24
    },
    "parameters": {
        "CU0-SN0": {
            "pair": "CU0-SN0",
            "direction": "y_on_x",
            "beta_star": 0.8523,
            "alpha_star": -0.0012,
            "selected_method": "EWLS",
            "spread_formula": "log(CU0) - 0.8523*log(SN0) - (-0.0012)",
            "Q_matrix": [[1e-5, 0], [0, 1e-6]],
            "R_value": 0.0001,
            "validation_score": 0.78,
            "status": "selected"
        }
    }
}
```

### 8.2 评估报告 (calibration_report.csv)
| pair | method | beta | alpha | adf_p | halflife | score | selected | reason |
|------|--------|------|-------|-------|----------|-------|----------|--------|
| CU0-SN0 | OLS_2y | 0.845 | -0.001 | 0.012 | 25.3 | 0.72 | False | ADF p > 0.01 |
| CU0-SN0 | EWLS | 0.852 | -0.001 | 0.008 | 23.5 | 0.78 | True | Best score |

## 9. 依赖关系
- **上游模块**: 数据管理模块、协整配对模块
- **下游模块**: 信号生成模块
- **Python包**: numpy, pandas, scipy, statsmodels, sklearn, linearmodels>=4.25
- **输出**: β*参数文件、Q/R矩阵、评估报告

## 10. 常见问题诊断表

| 症状 | 很可能原因 | 处理方法 |
|---|---|---|
| KF后半衰期很长、ADF p大 | 在KF动态残差上做ADF检验 | 只在静态β残差上做ADF检验 |
| 创新std≪1（0.6-0.8） | Q太小或R太大 | 增大Q（q_β×3），或减少R（÷2） |
| 创新均值偏离0 | 没有截距c_t，或R/Q失衡 | 使用二维KF[β_t,c_t]；微调R |
| 训练能过、OOS不过 | 方向/口径变了；或结构突变 | 锁死方向与spread口径 |
| ADF p一直大 | 残差用了regression='c'/'ct' | 改为regression='n' |

## 11. 部署考虑
- 标定过程计算密集，建议在配置较高的机器上运行
- 支持并行计算，可同时处理多个配对
- 生成的参数文件需要版本控制和备份
- 建议定期重新标定（如季度或半年）以适应市场变化
- **重要**：评估与交易必须分离，硬门槛只在静态β上评估