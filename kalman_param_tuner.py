#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalman滤波器参数快速调优工具
按专家建议实现创新白化诊断和参数迭代调整
"""

import pandas as pd
import numpy as np
from lib.data import load_all_symbols_data
from lib.coint import CointegrationAnalyzer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import copy

class KalmanTuner:
    """
    Kalman参数调优器 - 按专家建议的快速调参法
    """
    
    def __init__(self, x_symbol: str, y_symbol: str):
        self.x_symbol = x_symbol
        self.y_symbol = y_symbol
        
        # 加载数据
        print(f'加载{x_symbol}-{y_symbol}配对数据...')
        df = load_all_symbols_data()
        
        # 数据预处理 - 关键：必须用对数价格！
        x_data = df[x_symbol].dropna()
        y_data = df[y_symbol].dropna()
        common_dates = x_data.index.intersection(y_data.index)
        
        # 转换为对数价格（专家强调的必要步骤）
        self.x_aligned = np.log(x_data.loc[common_dates].values)
        self.y_aligned = np.log(y_data.loc[common_dates].values)
        self.dates = common_dates
        
        print('✅ 已转换为对数价格')
        
        print(f'数据点数量: {len(self.x_aligned)}')
        print(f'时间范围: {self.dates[0].strftime("%Y-%m-%d")} 到 {self.dates[-1].strftime("%Y-%m-%d")}')
        
        # 协整检验确认方向和基础参数
        analyzer = CointegrationAnalyzer(df)
        coint_result = analyzer.test_pair_cointegration(self.x_aligned, self.y_aligned)
        print(f'协整检验: p={coint_result["pvalue"]:.6f}, β={coint_result["beta"]:.6f}')
        
        # 基础OLS参数（用前252天初始化）
        reg = LinearRegression()
        reg.fit(self.x_aligned[:252].reshape(-1, 1), self.y_aligned[:252])
        self.initial_beta = float(reg.coef_[0])
        self.initial_c = float(reg.intercept_)
        self.initial_residuals = self.y_aligned[:252] - (self.initial_beta * self.x_aligned[:252] + self.initial_c)
        
        print(f'初始化: β={self.initial_beta:.6f}, c={self.initial_c:.2f}')
    
    def build_kalman_2d(self, eta_beta: float = 5e-3, eta_c: float = 5e-4, 
                       use_mad: bool = True) -> 'SimpleKalman2D':
        """
        按专家建议构建Kalman滤波器
        """
        # MAD方法计算R
        if use_mad:
            residual_mad = np.median(np.abs(self.initial_residuals - np.median(self.initial_residuals)))
            R = (residual_mad * 1.4826) ** 2
        else:
            R = np.var(self.initial_residuals)
        
        R = max(R, 1e-6)
        
        # X方差
        x_var = np.var(self.x_aligned[:252])
        x_var = max(x_var, 1e-10)
        
        # Q矩阵按专家建议
        q_beta = eta_beta * R / x_var
        q_c = eta_c * R
        Q = np.diag([q_beta, q_c])
        
        # P0适度设定
        P = np.diag([1.0, 0.1])
        
        return SimpleKalman2D(self.initial_beta, self.initial_c, Q, R, P)
    
    def test_innovation_whitening(self, kf: 'SimpleKalman2D', test_steps: int = 250) -> Dict:
        """
        测试创新白化效果（专家的核心诊断指标）
        """
        kf_copy = copy.deepcopy(kf)
        innovations = []
        betas = []
        
        start_idx = 252
        end_idx = min(start_idx + test_steps, len(self.x_aligned))
        
        for i in range(start_idx, end_idx):
            x_t = self.x_aligned[i]
            y_t = self.y_aligned[i]
            
            beta_new, innovation_z = kf_copy.update(y_t, x_t)
            innovations.append(innovation_z)
            betas.append(beta_new)
        
        innovations = np.array(innovations)
        betas = np.array(betas)
        
        # 白化诊断
        z_mean = np.mean(innovations)
        z_std = np.std(innovations)
        beta_std = np.std(betas)
        
        # 判断状态
        mean_ok = abs(z_mean) <= 0.1
        std_ok = 0.9 <= z_std <= 1.1
        
        return {
            'z_mean': z_mean,
            'z_std': z_std,
            'beta_std': beta_std,
            'mean_ok': mean_ok,
            'std_ok': std_ok,
            'whitening_ok': mean_ok and std_ok,
            'betas': betas,
            'innovations': innovations
        }
    
    def compare_with_ols(self, kf: 'SimpleKalman2D', window: int = 60) -> Dict:
        """
        与OLS滚动窗口对比
        """
        kf_copy = copy.deepcopy(kf)
        
        start_idx = 252 + window
        kalman_betas = []
        ols_betas = []
        
        for i in range(start_idx, len(self.x_aligned)):
            # Kalman更新
            x_t = self.x_aligned[i]
            y_t = self.y_aligned[i]
            beta_kf, _ = kf_copy.update(y_t, x_t)
            kalman_betas.append(beta_kf)
            
            # OLS滚动窗口
            x_window = self.x_aligned[i-window+1:i+1]
            y_window = self.y_aligned[i-window+1:i+1]
            reg_window = LinearRegression()
            reg_window.fit(x_window.reshape(-1, 1), y_window)
            ols_betas.append(float(reg_window.coef_[0]))
        
        kalman_betas = np.array(kalman_betas)
        ols_betas = np.array(ols_betas)
        
        # 统计对比
        kf_std = np.std(kalman_betas)
        ols_std = np.std(ols_betas)
        stability_ratio = ols_std / kf_std
        correlation = np.corrcoef(kalman_betas, ols_betas)[0, 1]
        
        return {
            'kalman_betas': kalman_betas,
            'ols_betas': ols_betas,
            'kf_std': kf_std,
            'ols_std': ols_std,
            'stability_ratio': stability_ratio,
            'correlation': correlation
        }
    
    def auto_tune_parameters(self, max_iterations: int = 5) -> Dict:
        """
        按专家建议自动调参 - 核心是达到创新白化
        """
        print("\\n=== 开始参数自动调优 ===")
        
        # 初始参数（均衡档起步）
        eta_beta = 5e-3
        eta_c = 5e-4
        
        results = []
        
        for iteration in range(max_iterations):
            print(f"\\n--- 第{iteration+1}轮调参 ---")
            print(f"当前参数: eta_β={eta_beta:.2e}, eta_c={eta_c:.2e}")
            
            # 构建KF
            kf = self.build_kalman_2d(eta_beta, eta_c)
            
            # 测试创新白化
            whitening_result = self.test_innovation_whitening(kf)
            z_std = whitening_result['z_std']
            z_mean = whitening_result['z_mean']
            
            print(f"创新白化结果: mean(z)={z_mean:.4f}, std(z)={z_std:.4f}")
            
            # 对比OLS
            comparison = self.compare_with_ols(kf)
            
            result = {
                'iteration': iteration + 1,
                'eta_beta': eta_beta,
                'eta_c': eta_c,
                'z_mean': z_mean,
                'z_std': z_std,
                'stability_ratio': comparison['stability_ratio'],
                'correlation': comparison['correlation'],
                'whitening_ok': whitening_result['whitening_ok']
            }
            results.append(result)
            
            print(f"稳定性改善: {comparison['stability_ratio']:.2f}x, 相关性: {comparison['correlation']:.4f}")
            
            # 判断是否达到目标
            if whitening_result['whitening_ok'] and comparison['correlation'] >= 0.6:
                print(f"✅ 第{iteration+1}轮达到目标！")
                break
            
            # 按专家建议调整
            if z_std < 0.9:
                # KF太钝，增大Q
                eta_beta *= 3
                print(f"std(z)={z_std:.3f} < 0.9，增大eta_β × 3")
            elif z_std > 1.1:
                # KF太敏感，减小Q或增大R  
                eta_beta /= 2
                print(f"std(z)={z_std:.3f} > 1.1，减小eta_β ÷ 2")
            else:
                # 白化OK但相关性不够，微调
                if comparison['correlation'] < 0.6:
                    eta_beta *= 1.5
                    print(f"白化正常但相关性{comparison['correlation']:.3f} < 0.6，微调eta_β × 1.5")
        
        return {
            'best_params': results[-1],
            'all_results': results,
            'final_kf': kf,
            'final_comparison': comparison
        }


class SimpleKalman2D:
    """
    简化的2D Kalman滤波器，用于参数调优测试
    """
    
    def __init__(self, beta0: float, c0: float, Q: np.ndarray, R: float, P: np.ndarray):
        self.beta = beta0
        self.c = c0
        self.Q = Q  # 过程噪声
        self.R = R  # 观测噪声
        self.P = P  # 状态协方差
    
    def update(self, y_t: float, x_t: float) -> Tuple[float, float]:
        """
        Kalman更新，返回(新beta, 标准化创新)
        """
        # 观测矩阵 H = [x_t, 1]
        H = np.array([[x_t, 1.0]])
        
        # 1. 预测
        P_pred = self.P + self.Q
        state = np.array([self.beta, self.c])
        y_pred = float(H @ state)
        
        # 2. 创新
        v = y_t - y_pred
        S = float(H @ P_pred @ H.T + self.R)
        S = max(S, 1e-12)
        
        # 3. 标准化创新（白化诊断核心指标）
        z_score = v / np.sqrt(S)
        
        # 4. Kalman增益和状态更新
        K = (P_pred @ H.T) / S
        update_vec = (K * v).ravel()
        
        self.beta += update_vec[0]
        self.c += update_vec[1]
        
        # 5. 协方差更新
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ np.array([[self.R]]) @ K.T
        
        # 6. 简单的R自适应（EWMA）
        R_innov = max(v*v - float(H @ P_pred @ H.T), 1e-8)
        self.R = 0.98 * self.R + 0.02 * R_innov
        self.R = np.clip(self.R, 1e-8, 1e6)
        
        return self.beta, z_score


def main():
    """
    主函数：测试SM-RB配对的Kalman参数调优
    """
    tuner = KalmanTuner('SM', 'RB')
    
    # 自动调参
    tune_results = tuner.auto_tune_parameters()
    
    print("\\n=== 调参结果总结 ===")
    best = tune_results['best_params']
    print(f"最佳参数: eta_β={best['eta_beta']:.2e}, eta_c={best['eta_c']:.2e}")
    print(f"创新白化: mean(z)={best['z_mean']:.4f}, std(z)={best['z_std']:.4f}")
    print(f"稳定性改善: {best['stability_ratio']:.2f}x")
    print(f"相关性: {best['correlation']:.4f}")
    print(f"白化状态: {'✅通过' if best['whitening_ok'] else '❌需调整'}")
    
    # 保存调参历史
    results_df = pd.DataFrame(tune_results['all_results'])
    results_df.to_csv('kalman_tuning_history.csv', index=False)
    print(f"调参历史已保存: kalman_tuning_history.csv")
    
    # 生成最终对比图
    final_comp = tune_results['final_comparison']
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: Beta对比
    plt.subplot(2, 2, 1)
    plt.plot(final_comp['kalman_betas'], label=f'Kalman (std={final_comp["kf_std"]:.4f})', alpha=0.8)
    plt.plot(final_comp['ols_betas'], label=f'OLS-60 (std={final_comp["ols_std"]:.4f})', alpha=0.7)
    plt.title(f'优化后Beta对比 (相关性={final_comp["correlation"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 稳定性对比
    plt.subplot(2, 2, 2)
    methods = ['Kalman (优化后)', 'OLS-60天']
    stds = [final_comp['kf_std'], final_comp['ols_std']]
    bars = plt.bar(methods, stds, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Beta标准差')
    plt.title(f'稳定性改善: {final_comp["stability_ratio"]:.2f}x')
    plt.grid(True, alpha=0.3)
    
    for bar, std in zip(bars, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std*0.01, 
                f'{std:.4f}', ha='center', va='bottom')
    
    # 子图3: 调参历史
    plt.subplot(2, 2, 3)
    iterations = [r['iteration'] for r in tune_results['all_results']]
    z_stds = [r['z_std'] for r in tune_results['all_results']]
    correlations = [r['correlation'] for r in tune_results['all_results']]
    
    plt.plot(iterations, z_stds, 'o-', label='std(z)', color='red')
    plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='目标区间')
    plt.axhline(y=1.1, color='red', linestyle='--', alpha=0.5)
    plt.ylabel('std(z)', color='red')
    plt.xlabel('调参轮次')
    plt.title('调参历史: 创新白化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图4: 相关性历史
    plt.subplot(2, 2, 4)
    plt.plot(iterations, correlations, 'o-', label='KF-OLS相关性', color='green')
    plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='目标≥0.6')
    plt.ylabel('相关性', color='green')
    plt.xlabel('调参轮次')
    plt.title('调参历史: KF-OLS相关性')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kalman_auto_tuning_results.png', dpi=150, bbox_inches='tight')
    print("调参结果图已保存: kalman_auto_tuning_results.png")


if __name__ == '__main__':
    main()