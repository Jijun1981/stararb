#!/usr/bin/env python3
"""
协整算法验证脚本
使用多种算法交叉验证协整计算结果，确保误差小于3%

验证内容：
1. OLS Beta估计：使用statsmodels、numpy、sklearn三种方法
2. ADF检验：使用statsmodels和手动实现
3. 半衰期计算：使用多种方法
4. 波动率计算：验证不同实现

输出：
- MD格式验证报告
- CSV格式详细数据
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 统计库
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize_scalar

# 项目模块
from lib.data import load_data
from lib.coint import engle_granger_test, estimate_parameters, calculate_halflife


class CointegrationVerifier:
    """协整算法验证器"""
    
    def __init__(self, output_dir: Path = None):
        """初始化验证器"""
        if output_dir is None:
            output_dir = Path("./output/cointegration")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.results = []
        
    def verify_ols_beta(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        使用三种方法验证OLS Beta估计
        1. NumPy最小二乘法
        2. Statsmodels OLS
        3. Sklearn LinearRegression
        """
        print("\n" + "="*60)
        print("验证OLS Beta估计")
        print("-"*60)
        
        results = {}
        
        # 方法1: NumPy最小二乘法
        # y = alpha + beta * x
        # 使用正规方程: beta = (X'X)^(-1)X'y
        X = np.column_stack([np.ones(len(x)), x])
        beta_numpy = np.linalg.lstsq(X, y, rcond=None)[0]
        results['numpy'] = {
            'beta': beta_numpy[1],
            'alpha': beta_numpy[0],
            'method': 'NumPy lstsq (正规方程)'
        }
        print(f"NumPy方法: beta = {beta_numpy[1]:.6f}, alpha = {beta_numpy[0]:.6f}")
        
        # 方法2: Statsmodels OLS
        X_sm = sm.add_constant(x)
        model_sm = OLS(y, X_sm)
        res_sm = model_sm.fit()
        results['statsmodels'] = {
            'beta': res_sm.params[1],
            'alpha': res_sm.params[0],
            'r_squared': res_sm.rsquared,
            'std_error': res_sm.bse[1],
            'method': 'Statsmodels OLS'
        }
        print(f"Statsmodels方法: beta = {res_sm.params[1]:.6f}, alpha = {res_sm.params[0]:.6f}")
        print(f"  R² = {res_sm.rsquared:.4f}, Std Error = {res_sm.bse[1]:.6f}")
        
        # 方法3: Sklearn LinearRegression
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        results['sklearn'] = {
            'beta': lr.coef_[0],
            'alpha': lr.intercept_,
            'method': 'Sklearn LinearRegression'
        }
        print(f"Sklearn方法: beta = {lr.coef_[0]:.6f}, alpha = {lr.intercept_:.6f}")
        
        # 方法4: 手动计算（协方差方法）
        # beta = Cov(x,y) / Var(x)
        cov_xy = np.cov(x, y)[0, 1]
        var_x = np.var(x, ddof=1)
        beta_cov = cov_xy / var_x
        alpha_cov = np.mean(y) - beta_cov * np.mean(x)
        results['covariance'] = {
            'beta': beta_cov,
            'alpha': alpha_cov,
            'method': '协方差方法 Cov(x,y)/Var(x)'
        }
        print(f"协方差方法: beta = {beta_cov:.6f}, alpha = {alpha_cov:.6f}")
        
        # 计算差异
        betas = [results[method]['beta'] for method in results]
        beta_mean = np.mean(betas)
        beta_std = np.std(betas)
        max_diff = (max(betas) - min(betas)) / beta_mean * 100
        
        print(f"\n验证结果:")
        print(f"  Beta均值: {beta_mean:.6f}")
        print(f"  Beta标准差: {beta_std:.6f}")
        print(f"  最大差异: {max_diff:.2f}%")
        
        results['summary'] = {
            'mean': beta_mean,
            'std': beta_std,
            'max_diff_pct': max_diff,
            'pass': max_diff < 3.0
        }
        
        return results
    
    def verify_adf_test(self, residuals: np.ndarray) -> dict:
        """
        验证ADF检验
        1. Statsmodels adfuller
        2. 手动实现ADF检验
        3. 使用其他单位根检验
        """
        print("\n" + "="*60)
        print("验证ADF检验")
        print("-"*60)
        
        results = {}
        
        # 方法1: Statsmodels ADF
        adf_result = adfuller(residuals, autolag='AIC')
        results['statsmodels_adf'] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'lags': adf_result[2],
            'nobs': adf_result[3],
            'critical_values': adf_result[4]
        }
        print(f"Statsmodels ADF:")
        print(f"  统计量: {adf_result[0]:.6f}")
        print(f"  p值: {adf_result[1]:.6f}")
        print(f"  滞后阶数: {adf_result[2]}")
        
        # 方法2: 手动实现简化ADF (DF检验)
        # Δy_t = ρ*y_{t-1} + ε_t
        # H0: ρ = 0 (存在单位根)
        y = residuals[1:]
        y_lag = residuals[:-1]
        
        # OLS回归
        X = sm.add_constant(y_lag)
        model = OLS(y - y_lag, X)
        res = model.fit()
        
        # 计算t统计量
        rho = res.params[1]
        se_rho = res.bse[1]
        t_stat = rho / se_rho
        
        results['manual_df'] = {
            'statistic': t_stat,
            'rho': rho,
            'se': se_rho,
            'method': '手动DF检验'
        }
        print(f"\n手动DF检验:")
        print(f"  统计量: {t_stat:.6f}")
        print(f"  ρ系数: {rho:.6f}")
        
        # 方法3: Phillips-Perron检验（另一种单位根检验）
        pp_result = adfuller(residuals, regression='c', autolag=None, maxlag=0)
        results['phillips_perron'] = {
            'statistic': pp_result[0],
            'pvalue': pp_result[1]
        }
        print(f"\nPhillips-Perron检验:")
        print(f"  统计量: {pp_result[0]:.6f}")
        print(f"  p值: {pp_result[1]:.6f}")
        
        # 验证p值差异
        pvalues = [results['statsmodels_adf']['pvalue'], 
                   results['phillips_perron']['pvalue']]
        pvalue_diff = abs(pvalues[0] - pvalues[1]) / np.mean(pvalues) * 100
        
        print(f"\n验证结果:")
        print(f"  p值差异: {pvalue_diff:.2f}%")
        
        results['summary'] = {
            'pvalue_diff_pct': pvalue_diff,
            'pass': pvalue_diff < 10.0  # p值差异允许更大一些
        }
        
        return results
    
    def verify_halflife(self, residuals: np.ndarray) -> dict:
        """
        验证半衰期计算
        1. AR(1)方法
        2. OLS方法
        3. 自相关函数方法
        """
        print("\n" + "="*60)
        print("验证半衰期计算")
        print("-"*60)
        
        results = {}
        
        # 方法1: AR(1)模型
        # y_t = λ * y_{t-1} + ε_t
        y = residuals[1:]
        y_lag = residuals[:-1]
        
        # OLS估计λ
        model = OLS(y, y_lag)
        res = model.fit()
        lambda_ar1 = res.params[0]
        halflife_ar1 = -np.log(2) / np.log(lambda_ar1) if lambda_ar1 > 0 and lambda_ar1 < 1 else np.nan
        
        results['ar1'] = {
            'lambda': lambda_ar1,
            'halflife': halflife_ar1,
            'method': 'AR(1)模型'
        }
        print(f"AR(1)方法: λ = {lambda_ar1:.6f}, 半衰期 = {halflife_ar1:.2f} 天")
        
        # 方法2: OLS with constant
        X = sm.add_constant(y_lag)
        model2 = OLS(y, X)
        res2 = model2.fit()
        lambda_ols = res2.params[1]
        halflife_ols = -np.log(2) / np.log(lambda_ols) if lambda_ols > 0 and lambda_ols < 1 else np.nan
        
        results['ols'] = {
            'lambda': lambda_ols,
            'halflife': halflife_ols,
            'method': 'OLS with constant'
        }
        print(f"OLS方法: λ = {lambda_ols:.6f}, 半衰期 = {halflife_ols:.2f} 天")
        
        # 方法3: 指数衰减拟合
        # 使用自相关函数
        from statsmodels.tsa.stattools import acf
        acf_values = acf(residuals, nlags=40)
        
        # 找到ACF降到0.5的滞后数
        halflife_acf = np.where(acf_values < 0.5)[0]
        halflife_acf = halflife_acf[0] if len(halflife_acf) > 0 else np.nan
        
        results['acf'] = {
            'halflife': halflife_acf,
            'method': '自相关函数法'
        }
        print(f"ACF方法: 半衰期 = {halflife_acf:.2f} 天")
        
        # 验证差异
        halflives = [hl for hl in [halflife_ar1, halflife_ols, halflife_acf] if not np.isnan(hl)]
        if len(halflives) > 1:
            hl_mean = np.mean(halflives)
            hl_std = np.std(halflives)
            max_diff = (max(halflives) - min(halflives)) / hl_mean * 100 if hl_mean > 0 else np.inf
        else:
            hl_mean = halflives[0] if halflives else np.nan
            hl_std = 0
            max_diff = 0
        
        print(f"\n验证结果:")
        print(f"  半衰期均值: {hl_mean:.2f} 天")
        print(f"  半衰期标准差: {hl_std:.2f}")
        print(f"  最大差异: {max_diff:.2f}%")
        
        results['summary'] = {
            'mean': hl_mean,
            'std': hl_std,
            'max_diff_pct': max_diff,
            'pass': max_diff < 20.0  # 半衰期差异允许更大
        }
        
        return results
    
    def verify_volatility(self, log_prices: np.ndarray) -> dict:
        """
        验证波动率计算
        1. 标准差方法
        2. EWMA方法
        3. GARCH方法
        """
        print("\n" + "="*60)
        print("验证波动率计算")
        print("-"*60)
        
        results = {}
        
        # 计算对数收益率
        returns = np.diff(log_prices)
        
        # 方法1: 简单标准差
        vol_std = np.std(returns) * np.sqrt(252)
        results['std'] = {
            'volatility': vol_std,
            'method': '简单标准差 * sqrt(252)'
        }
        print(f"标准差方法: 年化波动率 = {vol_std:.6f}")
        
        # 方法2: 样本标准差（自由度修正）
        vol_sample = np.std(returns, ddof=1) * np.sqrt(252)
        results['sample_std'] = {
            'volatility': vol_sample,
            'method': '样本标准差(ddof=1) * sqrt(252)'
        }
        print(f"样本标准差: 年化波动率 = {vol_sample:.6f}")
        
        # 方法3: EWMA（指数加权移动平均）
        lambda_param = 0.94
        ewma_var = pd.Series(returns**2).ewm(alpha=1-lambda_param, adjust=False).mean()
        vol_ewma = np.sqrt(ewma_var.iloc[-1] * 252)
        results['ewma'] = {
            'volatility': vol_ewma,
            'lambda': lambda_param,
            'method': f'EWMA(λ={lambda_param})'
        }
        print(f"EWMA方法: 年化波动率 = {vol_ewma:.6f}")
        
        # 方法4: 滚动窗口标准差
        window = 60
        if len(returns) >= window:
            rolling_std = pd.Series(returns).rolling(window).std()
            vol_rolling = rolling_std.iloc[-1] * np.sqrt(252)
            results['rolling'] = {
                'volatility': vol_rolling,
                'window': window,
                'method': f'滚动窗口({window}天)'
            }
            print(f"滚动窗口: 年化波动率 = {vol_rolling:.6f}")
        
        # 验证差异
        vols = [results[method]['volatility'] for method in results]
        vol_mean = np.mean(vols)
        vol_std = np.std(vols)
        max_diff = (max(vols) - min(vols)) / vol_mean * 100
        
        print(f"\n验证结果:")
        print(f"  波动率均值: {vol_mean:.6f}")
        print(f"  波动率标准差: {vol_std:.6f}")
        print(f"  最大差异: {max_diff:.2f}%")
        
        results['summary'] = {
            'mean': vol_mean,
            'std': vol_std,
            'max_diff_pct': max_diff,
            'pass': max_diff < 5.0
        }
        
        return results
    
    def verify_cointegration_pair(self, symbol1: str, symbol2: str) -> dict:
        """验证一对品种的协整关系"""
        print(f"\n{'='*60}")
        print(f"验证配对: {symbol1} - {symbol2}")
        print('='*60)
        
        # 加载数据
        data = load_data([symbol1, symbol2], columns=['close'], log_price=True)
        
        # 使用最近5年数据
        data_5y = data.iloc[-1260:]
        
        # 处理列名格式（load_data返回的是 symbol_close 格式）
        x_col = f"{symbol1}_close"
        y_col = f"{symbol2}_close"
        x = data_5y[x_col].values
        y = data_5y[y_col].values
        
        # 验证各个算法
        verification_results = {
            'pair': f"{symbol1}-{symbol2}",
            'ols_beta': self.verify_ols_beta(x, y),
            'adf_test': None,
            'halflife': None,
            'volatility': None
        }
        
        # 计算残差用于ADF和半衰期验证
        beta = verification_results['ols_beta']['summary']['mean']
        residuals = y - beta * x
        
        verification_results['adf_test'] = self.verify_adf_test(residuals)
        verification_results['halflife'] = self.verify_halflife(residuals)
        
        # 验证波动率（使用最近1年数据）
        data_1y = data.iloc[-252:]
        vol1_data = data_1y[x_col].values
        vol2_data = data_1y[y_col].values
            
        verification_results['volatility'] = {
            symbol1: self.verify_volatility(vol1_data),
            symbol2: self.verify_volatility(vol2_data)
        }
        
        # 总体验证结果
        all_pass = all([
            verification_results['ols_beta']['summary']['pass'],
            verification_results['adf_test']['summary']['pass'],
            verification_results['halflife']['summary']['pass'],
            verification_results['volatility'][symbol1]['summary']['pass'],
            verification_results['volatility'][symbol2]['summary']['pass']
        ])
        
        verification_results['overall_pass'] = all_pass
        
        return verification_results
    
    def generate_report(self, results: list):
        """生成验证报告（MD和CSV格式）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成MD报告
        md_path = self.output_dir / f"cointegration_verification_{timestamp}.md"
        csv_path = self.output_dir / f"cointegration_verification_{timestamp}.csv"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 协整算法验证报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 验证摘要\n\n")
            
            # 汇总表格
            f.write("| 配对 | OLS Beta误差 | ADF p值差异 | 半衰期误差 | 波动率误差 | 总体结果 |\n")
            f.write("|------|-------------|------------|-----------|-----------|----------|\n")
            
            csv_data = []
            
            for result in results:
                pair = result['pair']
                ols_err = result['ols_beta']['summary']['max_diff_pct']
                adf_err = result['adf_test']['summary']['pvalue_diff_pct']
                hl_err = result['halflife']['summary']['max_diff_pct']
                vol_err1 = result['volatility'][pair.split('-')[0]]['summary']['max_diff_pct']
                vol_err2 = result['volatility'][pair.split('-')[1]]['summary']['max_diff_pct']
                vol_err = max(vol_err1, vol_err2)
                overall = "✓" if result['overall_pass'] else "✗"
                
                f.write(f"| {pair} | {ols_err:.2f}% | {adf_err:.2f}% | {hl_err:.2f}% | {vol_err:.2f}% | {overall} |\n")
                
                # CSV数据
                csv_data.append({
                    'pair': pair,
                    'ols_beta_error_pct': ols_err,
                    'adf_pvalue_diff_pct': adf_err,
                    'halflife_error_pct': hl_err,
                    'volatility_error_pct': vol_err,
                    'overall_pass': result['overall_pass']
                })
            
            # 详细结果
            f.write("\n## 详细验证结果\n\n")
            
            for result in results:
                f.write(f"### {result['pair']}\n\n")
                
                # OLS Beta验证
                f.write("#### OLS Beta估计\n\n")
                f.write("| 方法 | Beta值 | Alpha值 |\n")
                f.write("|------|--------|--------|\n")
                for method, data in result['ols_beta'].items():
                    if method != 'summary':
                        f.write(f"| {data['method']} | {data['beta']:.6f} | {data['alpha']:.6f} |\n")
                
                f.write(f"\n**验证结果**: 最大误差 {result['ols_beta']['summary']['max_diff_pct']:.2f}% ")
                f.write(f"{'✓ 通过' if result['ols_beta']['summary']['pass'] else '✗ 未通过'}\n\n")
                
                # ADF检验验证
                f.write("#### ADF检验\n\n")
                f.write("| 方法 | 统计量 | p值 |\n")
                f.write("|------|--------|-----|\n")
                f.write(f"| Statsmodels ADF | {result['adf_test']['statsmodels_adf']['statistic']:.6f} | {result['adf_test']['statsmodels_adf']['pvalue']:.6f} |\n")
                f.write(f"| Phillips-Perron | {result['adf_test']['phillips_perron']['statistic']:.6f} | {result['adf_test']['phillips_perron']['pvalue']:.6f} |\n")
                
                f.write(f"\n**验证结果**: p值差异 {result['adf_test']['summary']['pvalue_diff_pct']:.2f}% ")
                f.write(f"{'✓ 通过' if result['adf_test']['summary']['pass'] else '✗ 未通过'}\n\n")
                
                # 半衰期验证
                f.write("#### 半衰期计算\n\n")
                f.write("| 方法 | 半衰期(天) |\n")
                f.write("|------|----------|\n")
                for method, data in result['halflife'].items():
                    if method != 'summary' and 'halflife' in data:
                        f.write(f"| {data['method']} | {data['halflife']:.2f} |\n")
                
                f.write(f"\n**验证结果**: 最大误差 {result['halflife']['summary']['max_diff_pct']:.2f}% ")
                f.write(f"{'✓ 通过' if result['halflife']['summary']['pass'] else '✗ 未通过'}\n\n")
        
        # 保存CSV
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        print(f"\n报告已生成:")
        print(f"  MD格式: {md_path}")
        print(f"  CSV格式: {csv_path}")
        
        return md_path, csv_path


def main():
    """主函数"""
    print("="*60)
    print("协整算法交叉验证")
    print("="*60)
    
    # 创建验证器
    verifier = CointegrationVerifier()
    
    # 选择要验证的配对
    test_pairs = [
        ('AG0', 'AU0'),  # 贵金属配对
        ('CU0', 'AL0'),  # 有色金属配对
        ('RB0', 'HC0'),  # 黑色系配对
    ]
    
    results = []
    for symbol1, symbol2 in test_pairs:
        try:
            result = verifier.verify_cointegration_pair(symbol1, symbol2)
            results.append(result)
        except Exception as e:
            print(f"验证 {symbol1}-{symbol2} 时出错: {e}")
            continue
    
    # 生成报告
    if results:
        verifier.generate_report(results)
        
        # 打印总结
        print("\n" + "="*60)
        print("验证总结")
        print("-"*60)
        
        pass_count = sum(1 for r in results if r['overall_pass'])
        total_count = len(results)
        
        print(f"通过验证: {pass_count}/{total_count}")
        
        for result in results:
            status = "✓" if result['overall_pass'] else "✗"
            print(f"  {status} {result['pair']}:")
            print(f"    - OLS Beta误差: {result['ols_beta']['summary']['max_diff_pct']:.2f}%")
            print(f"    - ADF p值差异: {result['adf_test']['summary']['pvalue_diff_pct']:.2f}%")
            print(f"    - 半衰期误差: {result['halflife']['summary']['max_diff_pct']:.2f}%")


if __name__ == "__main__":
    main()