#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应Kalman滤波参数优化器
基于目标导向的参数搜索，减少过拟合
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

class AdaptiveKalmanOptimizer:
    """自适应参数优化器"""
    
    def __init__(self):
        self.target_z_std = 1.0  # 目标标准差
        self.target_z_mean = 0.0  # 目标均值
        self.target_r_ratio = 1.0  # 目标能量比
        
    def load_pair_data(self, x_symbol, y_symbol):
        """加载配对数据"""
        df = load_all_symbols_data()
        x_data = np.log(df[x_symbol].dropna())
        y_data = np.log(df[y_symbol].dropna())
        
        common_dates = x_data.index.intersection(y_data.index)
        x_aligned = x_data.loc[common_dates]
        y_aligned = y_data.loc[common_dates]
        
        return x_aligned, y_aligned
    
    def run_kalman_with_params(self, x_data, y_data, R, Q_beta_ratio, start_idx=300):
        """运行Kalman滤波"""
        # 初始OLS
        reg = LinearRegression()
        reg.fit(x_data[:start_idx].values.reshape(-1, 1), y_data[:start_idx].values)
        beta0 = reg.coef_[0]
        c0 = reg.intercept_
        
        # 计算Q_beta
        avg_x = np.mean(x_data[start_idx-100:start_idx])
        P_base = R / (avg_x ** 2) * 0.1  # 基础P值
        Q_beta = P_base * Q_beta_ratio   # Q_beta相对于P的比例
        Q_c = R * 1e-6
        
        # 初始化
        beta_kf = beta0
        c_kf = c0
        P = np.diag([P_base, P_base * 0.1])
        Q = np.diag([Q_beta, Q_c])
        
        z_scores = []
        r_ratios = []
        
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
            r_ratio = (v ** 2) / S
            
            z_scores.append(z)
            r_ratios.append(r_ratio)
            
            # 更新
            K = (P_pred @ H.T) / S
            update_vec = (K * v).ravel()
            beta_kf += update_vec[0]
            c_kf += update_vec[1]
            
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P_pred @ I_KH.T + K @ np.array([[R]]) @ K.T
        
        z_scores = np.array(z_scores)
        r_ratios = np.array(r_ratios)
        
        return {
            'z_mean': np.mean(z_scores),
            'z_std': np.std(z_scores),
            'r_mean': np.mean(r_ratios),
            'z_scores': z_scores,
            'beta_change': abs(beta_kf - beta0) / abs(beta0)
        }
    
    def objective_function(self, results):
        """目标函数：越小越好"""
        z_mean_error = abs(results['z_mean'] - self.target_z_mean)
        z_std_error = abs(results['z_std'] - self.target_z_std)
        r_mean_error = abs(results['r_mean'] - self.target_r_ratio)
        
        # 综合目标函数，优先考虑std(z)接近1
        objective = z_std_error * 10 + z_mean_error * 5 + r_mean_error * 1
        
        return objective
    
    def optimize_single_pair(self, x_symbol, y_symbol, pair_name=None):
        """优化单个配对"""
        if pair_name is None:
            pair_name = f"{x_symbol}-{y_symbol}"
        
        print(f"\n🔍 优化配对: {pair_name}")
        
        # 加载数据
        x_data, y_data = self.load_pair_data(x_symbol, y_symbol)
        
        # 网格搜索参数空间
        R_candidates = np.logspace(-6, -2, 20)  # 从1e-6到1e-2
        Q_beta_ratio_candidates = np.logspace(-4, -1, 15)  # Q相对于P的比例
        
        best_objective = float('inf')
        best_params = None
        best_results = None
        
        print(f"搜索空间: R × Q_ratio = {len(R_candidates)} × {len(Q_beta_ratio_candidates)} = {len(R_candidates) * len(Q_beta_ratio_candidates)}")
        
        search_results = []
        
        for i, R in enumerate(R_candidates):
            for j, Q_ratio in enumerate(Q_beta_ratio_candidates):
                try:
                    results = self.run_kalman_with_params(x_data, y_data, R, Q_ratio)
                    objective = self.objective_function(results)
                    
                    search_results.append({
                        'R': R,
                        'Q_ratio': Q_ratio,
                        'z_mean': results['z_mean'],
                        'z_std': results['z_std'],
                        'r_mean': results['r_mean'],
                        'objective': objective,
                        'beta_change': results['beta_change']
                    })
                    
                    if objective < best_objective:
                        best_objective = objective
                        best_params = {'R': R, 'Q_ratio': Q_ratio}
                        best_results = results
                    
                    # 进度显示
                    if (i * len(Q_beta_ratio_candidates) + j + 1) % 50 == 0:
                        progress = (i * len(Q_beta_ratio_candidates) + j + 1) / (len(R_candidates) * len(Q_beta_ratio_candidates)) * 100
                        print(f"  进度: {progress:.1f}% | 当前最佳: z_std={best_results['z_std']:.3f}, objective={best_objective:.3f}")
                        
                except Exception as e:
                    continue
        
        # 转换为DataFrame进行分析
        search_df = pd.DataFrame(search_results)
        
        print(f"\n📊 {pair_name} 优化结果:")
        print(f"最佳参数: R={best_params['R']:.2e}, Q_ratio={best_params['Q_ratio']:.2e}")
        print(f"最佳结果: z_mean={best_results['z_mean']:.4f}, z_std={best_results['z_std']:.4f}")
        print(f"         r_mean={best_results['r_mean']:.4f}, β变化={best_results['beta_change']*100:.1f}%")
        
        # 质量评估
        quality_score = self.evaluate_quality(best_results, pair_name)
        
        return {
            'pair_name': pair_name,
            'best_params': best_params,
            'best_results': best_results,
            'search_df': search_df,
            'quality_score': quality_score
        }
    
    def evaluate_quality(self, results, pair_name):
        """评估质量（简化版）"""
        z_scores = results['z_scores']
        
        # 核心指标
        z_mean = results['z_mean']
        z_std = results['z_std']
        r_mean = results['r_mean']
        
        # 计算自相关
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(z_scores, nlags=5, fft=False)[1:6]
            max_acf = np.max(np.abs(acf_values))
            acf_ok = max_acf < 0.1
        except:
            max_acf = np.nan
            acf_ok = False
        
        # Ljung-Box检验
        try:
            lb_result = acorr_ljungbox(z_scores, lags=5, return_df=True)
            lb_pvalue = lb_result.iloc[-1]['lb_pvalue']
            lb_ok = lb_pvalue > 0.20
        except:
            lb_pvalue = np.nan
            lb_ok = False
        
        # 极值频率
        extreme_3 = np.mean(np.abs(z_scores) > 3)
        extreme_4 = np.mean(np.abs(z_scores) > 4)
        
        # 评分
        checks = {
            'mean_ok': abs(z_mean) <= 0.15,
            'std_ok': 0.85 <= z_std <= 1.20,
            'r_ok': 0.85 <= r_mean <= 1.15,
            'acf_ok': acf_ok,
            'lb_ok': lb_ok,
            'extreme_3_ok': extreme_3 <= 0.01,
            'extreme_4_ok': extreme_4 <= 0.002
        }
        
        score = sum(checks.values())
        
        print(f"\n✅ {pair_name} 质量评估:")
        print(f"  均值: {z_mean:.4f} → {'✅' if checks['mean_ok'] else '❌'}")
        print(f"  标准差: {z_std:.4f} → {'✅' if checks['std_ok'] else '❌'}")
        print(f"  能量比: {r_mean:.4f} → {'✅' if checks['r_ok'] else '❌'}")
        print(f"  自相关: {max_acf:.4f} → {'✅' if checks['acf_ok'] else '❌'}")
        print(f"  Ljung-Box: {lb_pvalue:.4f} → {'✅' if checks['lb_ok'] else '❌'}")
        print(f"  |z|>3: {extreme_3*100:.2f}% → {'✅' if checks['extreme_3_ok'] else '❌'}")
        print(f"  |z|>4: {extreme_4*100:.2f}% → {'✅' if checks['extreme_4_ok'] else '❌'}")
        print(f"  综合评分: {score}/7")
        
        if score >= 6:
            print("🏆 优秀！达到生产标准")
        elif score >= 5:
            print("✅ 良好！基本可用")
        elif score >= 4:
            print("⚠️ 勉强合格")
        else:
            print("❌ 不达标")
        
        return {**checks, 'score': score, 'max_acf': max_acf, 'lb_pvalue': lb_pvalue, 
                'extreme_3': extreme_3, 'extreme_4': extreme_4}

def test_adaptive_optimization():
    """测试自适应优化"""
    optimizer = AdaptiveKalmanOptimizer()
    
    test_pairs = [
        ('AL', 'ZN'),
        ('CU', 'ZN'), 
        ('RB', 'HC')
    ]
    
    all_results = {}
    
    for x_symbol, y_symbol in test_pairs:
        try:
            result = optimizer.optimize_single_pair(x_symbol, y_symbol)
            all_results[result['pair_name']] = result
        except Exception as e:
            print(f"❌ {x_symbol}-{y_symbol} 优化失败: {str(e)}")
    
    # 总结
    print(f"\n{'='*80}")
    print("🎯 自适应优化总结")
    print(f"{'='*80}")
    
    summary = []
    for pair_name, result in all_results.items():
        r = result['best_results']
        q = result['quality_score']
        summary.append({
            '配对': pair_name,
            'R': f"{result['best_params']['R']:.2e}",
            'Q_ratio': f"{result['best_params']['Q_ratio']:.2e}",
            'z̄': f"{r['z_mean']:.4f}",
            'σ(z)': f"{r['z_std']:.4f}",
            'r̄': f"{r['r_mean']:.3f}",
            'β变化': f"{r['beta_change']*100:.1f}%",
            '评分': f"{q['score']}/7",
            '状态': '🏆' if q['score']>=6 else ('✅' if q['score']>=5 else ('⚠️' if q['score']>=4 else '❌'))
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    return all_results

if __name__ == '__main__':
    results = test_adaptive_optimization()