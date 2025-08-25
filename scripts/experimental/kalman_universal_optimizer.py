#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用Kalman滤波参数优化器
按照严格工程标准，支持多配对验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.data import load_all_symbols_data
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

class KalmanOptimizer:
    """通用Kalman滤波参数优化器"""
    
    def __init__(self):
        self.quality_standards = {
            'mean_strict': 0.10,
            'mean_loose': 0.15,
            'std_strict': (0.95, 1.05),
            'std_loose': (0.90, 1.10),
            'std_hetero': (0.85, 1.20),
            'acf_threshold': 0.1,
            'ljung_box_threshold': 0.20,
            'energy_ratio_strict': (0.90, 1.10),
            'energy_ratio_loose': (0.85, 1.15),
            'extreme_3sigma': 0.01,
            'extreme_4sigma': 0.002
        }
    
    def load_pair_data(self, x_symbol, y_symbol, log_prices=True):
        """加载配对数据"""
        df = load_all_symbols_data()
        
        if log_prices:
            x_data = np.log(df[x_symbol].dropna())
            y_data = np.log(df[y_symbol].dropna())
        else:
            x_data = df[x_symbol].dropna()
            y_data = df[y_symbol].dropna()
        
        common_dates = x_data.index.intersection(y_data.index)
        x_aligned = x_data.loc[common_dates]
        y_aligned = y_data.loc[common_dates]
        
        return x_aligned, y_aligned
    
    def estimate_initial_params(self, x_data, y_data, window=300):
        """估计初始参数（通用方法，减少过拟合）"""
        # 使用较长窗口估计稳定的β
        reg = LinearRegression()
        reg.fit(x_data[:window].values.reshape(-1, 1), y_data[:window].values)
        beta0 = reg.coef_[0]
        c0 = reg.intercept_
        
        # 计算创新统计（使用样本外数据避免过拟合）
        test_start = window
        test_end = min(window + 100, len(x_data))
        
        innovations = []
        for i in range(test_start, test_end):
            v = y_data.iloc[i] - (beta0 * x_data.iloc[i] + c0)
            innovations.append(v)
        
        v_var = np.var(innovations)
        v_mean = np.mean(innovations)
        
        # 系统性偏移调整
        if abs(v_mean) > 0.001:
            c0_adjusted = c0 + v_mean
        else:
            c0_adjusted = c0
            
        return beta0, c0_adjusted, v_var
    
    def calculate_universal_params(self, x_data, y_data, v_var):
        """计算通用参数（减少魔法数字）"""
        avg_x = np.mean(x_data[300:400])  # 使用中段数据
        
        # 核心思路：让S接近创新方差，但用更保守的参数
        # 减少硬编码，增加理论依据
        
        # 基础参数：让期望的S值接近创新方差
        base_S = v_var * 1.0  # 不用1.1，直接用1.0更保守
        
        # R和P的分配：基于系统辨识理论
        # R主导（观测噪声），P提供适应性
        R = base_S * 0.8      # R承担80%（减少到0.8）
        P_contribution = base_S * 0.2 / (avg_x ** 2)
        
        # Q的设定：基于β的预期变化率
        # 假设β每年变化不超过5%，每天变化约0.02%
        daily_beta_change = 0.0002  # 0.02%
        Q_beta = (daily_beta_change * abs(x_data.mean())) ** 2 * P_contribution
        Q_c = R * 1e-6
        
        return {
            'R': R,
            'Q_beta': Q_beta, 
            'Q_c': Q_c,
            'P_target': P_contribution,
            'base_S': base_S
        }
    
    def run_kalman_filter(self, x_data, y_data, beta0, c0, params, start_idx=300):
        """运行Kalman滤波"""
        # 初始化
        beta_kf = beta0
        c_kf = c0
        P = np.diag([params['P_target'], params['P_target'] * 0.1])
        Q = np.diag([params['Q_beta'], params['Q_c']])
        R = params['R']
        
        results = []
        
        for i in range(start_idx, len(x_data)):
            x_t = x_data.iloc[i]
            y_t = y_data.iloc[i]
            
            # 预测
            P_pred = P + Q
            H = np.array([[x_t, 1.0]])
            y_pred = beta_kf * x_t + c_kf
            
            # 创新
            v = y_t - y_pred
            S = float(H @ P_pred @ H.T + R)
            S = max(S, 1e-12)
            z = v / np.sqrt(S)
            
            # 能量比
            r_ratio = (v ** 2) / S
            
            results.append({
                'date': x_data.index[i],
                'v': v,
                'S': S,
                'z': z,
                'r_ratio': r_ratio,
                'beta': beta_kf,
                'c': c_kf
            })
            
            # 更新
            K = (P_pred @ H.T) / S
            update_vec = (K * v).ravel()
            beta_kf += update_vec[0]
            c_kf += update_vec[1]
            
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
        
        return pd.DataFrame(results).set_index('date')
    
    def evaluate_quality(self, results_df, pair_name=""):
        """按照工程标准评估质量"""
        z_scores = results_df['z'].values
        r_ratios = results_df['r_ratio'].values
        
        # 1. 均值检查
        z_mean = np.mean(z_scores)
        z_abs_mean = np.abs(z_mean)
        mean_strict_ok = z_abs_mean <= self.quality_standards['mean_strict']
        mean_loose_ok = z_abs_mean <= self.quality_standards['mean_loose']
        
        # 2. 标准差检查
        z_std = np.std(z_scores)
        std_strict_ok = (self.quality_standards['std_strict'][0] <= z_std <= 
                        self.quality_standards['std_strict'][1])
        std_loose_ok = (self.quality_standards['std_loose'][0] <= z_std <= 
                       self.quality_standards['std_loose'][1])
        std_hetero_ok = (self.quality_standards['std_hetero'][0] <= z_std <= 
                        self.quality_standards['std_hetero'][1])
        
        # 3. 自相关检查
        from statsmodels.tsa.stattools import acf
        try:
            acf_values = acf(z_scores, nlags=5, fft=False)[1:6]  # 排除lag=0
            acf_ok = np.all(np.abs(acf_values) < self.quality_standards['acf_threshold'])
            max_acf = np.max(np.abs(acf_values))
        except:
            acf_ok = False
            max_acf = np.nan
        
        # 4. Ljung-Box检查
        try:
            lb_result = acorr_ljungbox(z_scores, lags=5, return_df=True)
            lb_pvalue = lb_result.iloc[-1]['lb_pvalue']  # 取lag=5的p值
            lb_ok = lb_pvalue > self.quality_standards['ljung_box_threshold']
        except:
            lb_ok = False
            lb_pvalue = np.nan
        
        # 5. 能量比检查
        r_mean = np.mean(r_ratios)
        r_strict_ok = (self.quality_standards['energy_ratio_strict'][0] <= r_mean <= 
                      self.quality_standards['energy_ratio_strict'][1])
        r_loose_ok = (self.quality_standards['energy_ratio_loose'][0] <= r_mean <= 
                     self.quality_standards['energy_ratio_loose'][1])
        
        # 6. 极值频率检查
        extreme_3sigma = np.mean(np.abs(z_scores) > 3)
        extreme_4sigma = np.mean(np.abs(z_scores) > 4)
        extreme_3_ok = extreme_3sigma <= self.quality_standards['extreme_3sigma']
        extreme_4_ok = extreme_4sigma <= self.quality_standards['extreme_4sigma']
        
        # 综合评分
        strict_score = sum([mean_strict_ok, std_strict_ok, acf_ok, lb_ok, r_strict_ok, extreme_3_ok, extreme_4_ok])
        loose_score = sum([mean_loose_ok, std_loose_ok, acf_ok, lb_ok, r_loose_ok, extreme_3_ok, extreme_4_ok])
        hetero_score = sum([mean_loose_ok, std_hetero_ok, acf_ok, lb_ok, r_loose_ok, extreme_3_ok, extreme_4_ok])
        
        quality_report = {
            'pair_name': pair_name,
            'z_mean': z_mean,
            'z_std': z_std,
            'z_abs_mean': z_abs_mean,
            'r_mean': r_mean,
            'max_acf': max_acf,
            'lb_pvalue': lb_pvalue,
            'extreme_3sigma': extreme_3sigma,
            'extreme_4sigma': extreme_4sigma,
            'mean_strict_ok': mean_strict_ok,
            'mean_loose_ok': mean_loose_ok,
            'std_strict_ok': std_strict_ok,
            'std_loose_ok': std_loose_ok,
            'std_hetero_ok': std_hetero_ok,
            'acf_ok': acf_ok,
            'lb_ok': lb_ok,
            'r_strict_ok': r_strict_ok,
            'r_loose_ok': r_loose_ok,
            'extreme_3_ok': extreme_3_ok,
            'extreme_4_ok': extreme_4_ok,
            'strict_score': strict_score,
            'loose_score': loose_score,
            'hetero_score': hetero_score,
            'max_possible_score': 7
        }
        
        return quality_report
    
    def optimize_pair(self, x_symbol, y_symbol, pair_name=None):
        """优化单个配对"""
        if pair_name is None:
            pair_name = f"{x_symbol}-{y_symbol}"
        
        print(f"\n{'='*60}")
        print(f"优化配对: {pair_name}")
        print(f"{'='*60}")
        
        # 1. 加载数据
        x_data, y_data = self.load_pair_data(x_symbol, y_symbol)
        print(f"数据范围: {x_data.index[0].date()} 到 {x_data.index[-1].date()}")
        print(f"样本数: {len(x_data)}")
        
        # 2. 估计初始参数
        beta0, c0, v_var = self.estimate_initial_params(x_data, y_data)
        print(f"初始参数: β={beta0:.6f}, c={c0:.6f}, v_var={v_var:.8f}")
        
        # 3. 计算通用参数
        params = self.calculate_universal_params(x_data, y_data, v_var)
        print(f"Kalman参数: R={params['R']:.6f}, Q_β={params['Q_beta']:.2e}")
        
        # 4. 运行Kalman滤波
        results = self.run_kalman_filter(x_data, y_data, beta0, c0, params)
        print(f"滤波完成: 处理{len(results)}个样本")
        
        # 5. 评估质量
        quality = self.evaluate_quality(results, pair_name)
        
        return {
            'pair_name': pair_name,
            'params': params,
            'results': results,
            'quality': quality,
            'data': {'x_data': x_data, 'y_data': y_data}
        }
    
    def print_quality_report(self, quality):
        """打印质量报告"""
        print(f"\n📊 质量评估报告: {quality['pair_name']}")
        print(f"{'='*50}")
        
        # 核心指标
        print(f"均值: {quality['z_mean']:7.4f} (|z̄|={quality['z_abs_mean']:.4f})")
        mean_status = "✅严格" if quality['mean_strict_ok'] else ("✅宽松" if quality['mean_loose_ok'] else "❌超标")
        print(f"      标准: ≤0.10(严格) / ≤0.15(宽松) → {mean_status}")
        
        print(f"标准差: {quality['z_std']:6.4f}")
        if quality['std_strict_ok']:
            std_status = "✅严格合格"
        elif quality['std_loose_ok']:
            std_status = "✅宽松合格"
        elif quality['std_hetero_ok']:
            std_status = "✅异方差合格"
        else:
            std_status = "❌不合格"
        print(f"      标准: 0.95-1.05(严格) / 0.90-1.10(宽松) / 0.85-1.20(异方差) → {std_status}")
        
        # 自相关和独立性
        print(f"自相关: max|ACF|={quality['max_acf']:.4f} → {'✅' if quality['acf_ok'] else '❌'}")
        print(f"Ljung-Box: p={quality['lb_pvalue']:.4f} → {'✅' if quality['lb_ok'] else '❌'}")
        
        # 能量比
        print(f"能量比: r̄={quality['r_mean']:.4f}")
        r_status = "✅严格" if quality['r_strict_ok'] else ("✅宽松" if quality['r_loose_ok'] else "❌超标")
        print(f"      标准: 1.00±0.10(严格) / 1.00±0.15(宽松) → {r_status}")
        
        # 极值频率
        print(f"极值频率: |z|>3: {quality['extreme_3sigma']*100:.2f}% → {'✅' if quality['extreme_3_ok'] else '❌'}")
        print(f"         |z|>4: {quality['extreme_4sigma']*100:.2f}% → {'✅' if quality['extreme_4_ok'] else '❌'}")
        
        # 综合评分
        print(f"\n🎯 综合评分:")
        print(f"  严格标准: {quality['strict_score']}/{quality['max_possible_score']}")
        print(f"  宽松标准: {quality['loose_score']}/{quality['max_possible_score']}")
        print(f"  异方差容忍: {quality['hetero_score']}/{quality['max_possible_score']}")
        
        # 最终判定
        if quality['strict_score'] >= 6:
            print("🏆 严格标准通过！生产可用")
        elif quality['loose_score'] >= 6:
            print("✅ 宽松标准通过！实用可行")
        elif quality['hetero_score'] >= 6:
            print("⚠️ 异方差容忍下勉强合格")
        else:
            print("❌ 未达标，需要参数调整")

def test_multiple_pairs():
    """测试多个配对的通用性"""
    optimizer = KalmanOptimizer()
    
    # 测试配对列表（选择协整性强的）
    test_pairs = [
        ('AL', 'ZN', 'AL-ZN'),   # 之前测试的配对
        ('CU', 'ZN', 'CU-ZN'),   # 有色金属
        ('RB', 'HC', 'RB-HC'),   # 黑色系
    ]
    
    results = {}
    
    for x_symbol, y_symbol, pair_name in test_pairs:
        try:
            result = optimizer.optimize_pair(x_symbol, y_symbol, pair_name)
            results[pair_name] = result
            optimizer.print_quality_report(result['quality'])
        except Exception as e:
            print(f"❌ {pair_name} 优化失败: {str(e)}")
            continue
    
    # 总结报告
    print(f"\n{'='*80}")
    print("🎯 多配对通用性验证总结")
    print(f"{'='*80}")
    
    summary_data = []
    for pair_name, result in results.items():
        q = result['quality']
        summary_data.append({
            '配对': pair_name,
            'z̄': f"{q['z_mean']:.4f}",
            'σ(z)': f"{q['z_std']:.4f}",
            'r̄': f"{q['r_mean']:.3f}",
            '严格': f"{q['strict_score']}/7",
            '宽松': f"{q['loose_score']}/7",
            '状态': '🏆' if q['strict_score']>=6 else ('✅' if q['loose_score']>=6 else '❌')
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return results

if __name__ == '__main__':
    results = test_multiple_pairs()