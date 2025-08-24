#!/usr/bin/env python3
"""
Kalman滤波质量改善方案
主要改进：自适应参数、残差监控、多层滤波
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from lib.data import load_all_symbols_data
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class ImprovedKalmanFilter:
    """改进的Kalman滤波器"""
    
    def __init__(self, initial_beta=1.0, initial_P=1.0, 
                 delta=0.96, lambda_r=0.92, 
                 adaptive_delta=True, residual_monitor=True):
        """
        改进的Kalman滤波器初始化
        
        Parameters:
        - adaptive_delta: 自适应δ参数
        - residual_monitor: 残差监控和自动重置
        """
        self.beta = initial_beta
        self.P = initial_P
        self.delta_base = delta  # 基础δ值
        self.delta = delta
        self.lambda_r = lambda_r
        self.adaptive_delta = adaptive_delta
        self.residual_monitor = residual_monitor
        
        # 历史记录
        self.beta_history = [initial_beta]
        self.P_history = [initial_P]
        self.innovations = []
        self.R_history = []
        self.delta_history = [delta]
        
        # 残差监控
        self.residual_window = 30
        self.reset_threshold = 0.10  # ADF p值阈值
        self.last_reset = 0
        
        # 自适应参数
        self.adaptation_window = 20
        self.min_delta = 0.85
        self.max_delta = 0.99
        
    def update(self, y, x, step):
        """更新滤波器状态"""
        
        # 1. 预测步骤
        beta_pred = self.beta  # 状态预测（随机游走模型）
        P_pred = self.P / self.delta  # 预测方差
        
        # 2. 计算创新值
        innovation = y - beta_pred * x
        self.innovations.append(innovation)
        
        # 3. 自适应测量噪声方差R
        if len(self.innovations) >= 5:
            recent_innovations = self.innovations[-5:]
            R = self.lambda_r * self.R_history[-1] + (1-self.lambda_r) * innovation**2 if self.R_history else innovation**2
        else:
            R = innovation**2
        
        # 4. 卡尔曼增益
        S = x**2 * P_pred + R  # 创新协方差
        K = P_pred * x / S     # 卡尔曼增益
        
        # 5. 更新步骤
        self.beta = beta_pred + K * innovation
        self.P = P_pred * (1 - K * x)
        
        # 6. 记录历史
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.R_history.append(R)
        
        # 7. 自适应δ调整
        if self.adaptive_delta and step > self.adaptation_window:
            self._adapt_delta(step)
        
        self.delta_history.append(self.delta)
        
        # 8. 残差监控和重置
        if self.residual_monitor and step > self.residual_window and step - self.last_reset > self.residual_window:
            self._monitor_residuals(step)
        
        return innovation, self.beta, self.P, R
    
    def _adapt_delta(self, step):
        """自适应δ参数调整"""
        if len(self.innovations) < self.adaptation_window:
            return
        
        # 分析最近的创新值质量
        recent_innovations = self.innovations[-self.adaptation_window:]
        
        # 1. 检查趋势 - 如果有显著趋势，降低δ增强适应性
        from scipy import stats
        x_trend = range(len(recent_innovations))
        slope, _, _, p_value, _ = stats.linregress(x_trend, recent_innovations)
        
        # 2. 检查方差稳定性 - 如果方差不稳定，调整δ
        mid = len(recent_innovations) // 2
        var1 = np.var(recent_innovations[:mid])
        var2 = np.var(recent_innovations[mid:])
        var_ratio = max(var1, var2) / (min(var1, var2) + 1e-8)
        
        # 3. 自适应调整规则
        delta_adjustment = 0
        
        # 如果有显著趋势，降低δ
        if p_value < 0.05 and abs(slope) > 1e-4:
            delta_adjustment -= 0.02
        
        # 如果方差不稳定，根据情况调整
        if var_ratio > 2.0:
            delta_adjustment -= 0.01  # 增强适应性
        elif var_ratio < 1.2:
            delta_adjustment += 0.01  # 增强平滑性
        
        # 应用调整
        new_delta = np.clip(self.delta + delta_adjustment, self.min_delta, self.max_delta)
        
        # 如果调整幅度显著，才更新
        if abs(new_delta - self.delta) > 0.005:
            self.delta = new_delta
    
    def _monitor_residuals(self, step):
        """监控残差平稳性，必要时重置"""
        if len(self.innovations) < self.residual_window:
            return
        
        recent_innovations = self.innovations[-self.residual_window:]
        
        # ADF检验
        try:
            adf_result = adfuller(recent_innovations, autolag='AIC')
            p_value = adf_result[1]
            
            # 如果残差非平稳且显著，考虑重置
            if p_value > self.reset_threshold:
                # 软重置：增加状态不确定性，不完全重置参数
                self.P *= 2.0  # 增加不确定性
                self.delta = max(self.delta - 0.02, self.min_delta)  # 暂时增强适应性
                self.last_reset = step
                
        except:
            pass
    
    def get_quality_metrics(self):
        """获取滤波质量指标"""
        if len(self.innovations) < 30:
            return {}
        
        innovations = np.array(self.innovations[-60:])  # 最近60个点
        
        metrics = {
            'innovation_std': np.std(innovations),
            'innovation_mean': np.mean(innovations),
            'beta_stability': np.std(self.beta_history[-30:]) / np.mean(self.beta_history[-30:]) if len(self.beta_history) >= 30 else np.inf,
            'avg_delta': np.mean(self.delta_history[-30:]) if len(self.delta_history) >= 30 else self.delta,
            'avg_R': np.mean(self.R_history[-30:]) if len(self.R_history) >= 30 else 0,
        }
        
        # ADF检验
        try:
            adf_result = adfuller(innovations, autolag='AIC')
            metrics['adf_pvalue'] = adf_result[1]
            metrics['is_stationary'] = adf_result[1] < 0.05
        except:
            metrics['adf_pvalue'] = 1.0
            metrics['is_stationary'] = False
        
        # 趋势检验
        try:
            from scipy import stats
            slope, _, _, p_value, _ = stats.linregress(range(len(innovations)), innovations)
            metrics['trend_pvalue'] = p_value
            metrics['has_trend'] = p_value < 0.05 and abs(slope) > 1e-4
        except:
            metrics['trend_pvalue'] = 1.0
            metrics['has_trend'] = False
        
        return metrics

def test_improved_kalman():
    """测试改进的Kalman滤波器"""
    
    print("🔧 改进Kalman滤波器测试")
    print("="*60)
    
    # 加载数据
    data = load_all_symbols_data()
    signals_df = pd.read_csv('signals_e2e_20250824_182241.csv')
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # 选择几个有代表性的配对测试
    test_pairs = [
        ('CU-SN', '优秀配对'),  # 原本就好的
        ('AU-ZN', '问题配对'),  # 原本有问题的  
        ('ZN-SM', '中等配对')   # 中等质量的
    ]
    
    results = {}
    
    for pair, desc in test_pairs:
        if pair not in signals_df['pair'].unique():
            continue
            
        print(f"\n=== {pair} ({desc}) ===")
        
        # 获取该配对的信息
        pair_info = signals_df[signals_df['pair'] == pair].iloc[0]
        symbol_x = pair_info['symbol_x'] 
        symbol_y = pair_info['symbol_y']
        
        # 获取价格数据
        pair_signals = signals_df[signals_df['pair'] == pair].sort_values('date')
        start_date = pair_signals['date'].iloc[0]
        end_date = pair_signals['date'].iloc[-1]
        
        price_data = data[start_date:end_date]
        x_prices = price_data[symbol_x].dropna()
        y_prices = price_data[symbol_y].dropna()
        common_dates = x_prices.index.intersection(y_prices.index)
        
        x_data = x_prices[common_dates].values
        y_data = y_prices[common_dates].values
        
        if len(x_data) < 100:
            continue
        
        # 分别测试原始和改进的滤波器
        print(f"数据点数: {len(x_data)}")
        
        # 1. 原始Kalman滤波器
        from lib.signal_generation import AdaptiveKalmanFilter
        original_kf = AdaptiveKalmanFilter(pair_name=pair, delta=0.96, lambda_r=0.92)
        original_kf.warm_up_ols(x_data, y_data, 60)
        
        original_innovations = []
        for i in range(90, len(x_data)):  # 跳过预热期
            result = original_kf.update(y_data[i], x_data[i])
            innovation = result['v']  # 创新值
            original_innovations.append(innovation)
        
        # 2. 改进的Kalman滤波器
        improved_kf = ImprovedKalmanFilter(
            initial_beta=original_kf.beta,
            initial_P=original_kf.P,
            delta=0.94,  # 稍微更激进
            lambda_r=0.90,
            adaptive_delta=True,
            residual_monitor=True
        )
        
        improved_innovations = []
        for i in range(90, len(x_data)):
            innovation, _, _, _ = improved_kf.update(y_data[i], x_data[i], i)
            improved_innovations.append(innovation)
        
        # 3. 质量对比
        def test_stationarity(series):
            try:
                adf_result = adfuller(series, autolag='AIC')
                return adf_result[1] < 0.05, adf_result[1]
            except:
                return False, 1.0
        
        orig_stationary, orig_p = test_stationarity(original_innovations)
        impr_stationary, impr_p = test_stationarity(improved_innovations)
        
        print(f"\\n原始滤波器:")
        print(f"  创新值std: {np.std(original_innovations):.4f}")
        print(f"  ADF p值: {orig_p:.4f}")
        print(f"  平稳性: {'✅' if orig_stationary else '❌'}")
        
        print(f"\\n改进滤波器:")
        print(f"  创新值std: {np.std(improved_innovations):.4f}")
        print(f"  ADF p值: {impr_p:.4f}")
        print(f"  平稳性: {'✅' if impr_stationary else '❌'}")
        print(f"  平均δ: {np.mean(improved_kf.delta_history[-50:]):.3f}")
        
        # 改进效果
        std_improvement = (np.std(original_innovations) - np.std(improved_innovations)) / np.std(original_innovations)
        p_improvement = (orig_p - impr_p) / orig_p if orig_p > 0 else 0
        
        print(f"\\n📊 改进效果:")
        print(f"  标准差变化: {std_improvement*100:+.1f}%")
        print(f"  ADF p值变化: {p_improvement*100:+.1f}%")
        
        if impr_stationary and not orig_stationary:
            print(f"  ✅ 成功改善了平稳性！")
        elif impr_stationary and orig_stationary:
            print(f"  ✅ 保持了平稳性，质量提升")
        elif not impr_stationary and not orig_stationary:
            if impr_p < orig_p:
                print(f"  ⚠️ 虽未达到平稳，但有改善趋势")
            else:
                print(f"  ❌ 改进效果不明显")
        else:
            print(f"  ⚠️ 可能过度调整，需要参数微调")
        
        results[pair] = {
            'original_std': np.std(original_innovations),
            'improved_std': np.std(improved_innovations),
            'original_adf_p': orig_p,
            'improved_adf_p': impr_p,
            'original_stationary': orig_stationary,
            'improved_stationary': impr_stationary,
            'std_improvement': std_improvement,
            'p_improvement': p_improvement
        }
    
    # 总结
    if results:
        print(f"\\n🎯 总体改进效果:")
        
        improved_pairs = sum(1 for r in results.values() if r['improved_stationary'] and not r['original_stationary'])
        maintained_pairs = sum(1 for r in results.values() if r['improved_stationary'] and r['original_stationary'])
        degraded_pairs = sum(1 for r in results.values() if not r['improved_stationary'] and r['original_stationary'])
        
        print(f"  成功改善平稳性: {improved_pairs}个配对")
        print(f"  保持优秀质量: {maintained_pairs}个配对")
        print(f"  质量下降: {degraded_pairs}个配对")
        
        avg_std_improvement = np.mean([r['std_improvement'] for r in results.values()])
        avg_p_improvement = np.mean([r['p_improvement'] for r in results.values()])
        
        print(f"  平均标准差改善: {avg_std_improvement*100:+.1f}%")
        print(f"  平均ADF p值改善: {avg_p_improvement*100:+.1f}%")
        
        return results
    
    return None

if __name__ == "__main__":
    results = test_improved_kalman()