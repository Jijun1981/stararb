#!/usr/bin/env python3
"""
信号生成模块测试脚本
基于前面协整模块和数据管理模块的结果生成交易信号
严格按照需求文档实现，卡尔曼滤波所有参数写死
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class KalmanFilter1D:
    """
    一维卡尔曼滤波器 - 所有参数写死，不可配置
    """
    def __init__(self, initial_beta: float, hist_residuals: Optional[np.ndarray] = None):
        """
        初始化一维KF - 所有参数都写死
        
        Args:
            initial_beta: 初始β（来自协整模块）
            hist_residuals: 历史残差（用于估计R）
            
        固定参数：
            Q = 1e-4 (过程噪声)
            P0 = 0.1 (初始不确定性)
            beta_change_limit = 0.05 (5%变化限制)
            R_ewma_alpha = 0.02 (EWMA系数)
        """
        self.beta = initial_beta
        self.P = 0.1  # 固定值
        self.Q = 1e-4  # 固定值
        self.R = self._estimate_initial_R(hist_residuals) if hist_residuals is not None else 1.0
        self.beta_change_limit = 0.05  # 固定值
        self.beta_history = [initial_beta]
        
    def _estimate_initial_R(self, residuals: np.ndarray) -> float:
        """估计初始观测噪声R"""
        if len(residuals) == 0:
            return 1.0
        return max(np.var(residuals), 1e-6)
    
    def update(self, y_t: float, x_t: float) -> Dict:
        """
        一步KF更新
        
        Returns:
            dict: 包含更新后β、残差等
        """
        # 1. 预测步
        beta_pred = self.beta  # 随机游走
        P_pred = self.P + self.Q
        
        # 2. 计算残差（观测-预测）
        y_pred = beta_pred * x_t
        residual = y_t - y_pred
        
        # 3. 创新协方差
        S = x_t * P_pred * x_t + self.R
        
        # 4. Kalman增益
        K = P_pred * x_t / S if S > 1e-12 else 0
        
        # 5. 状态更新
        beta_new = beta_pred + K * residual
        
        # 6. β变化限制（固定参数）
        min_abs_change = 0.001  # 最小允许变化（固定）
        max_change = max(abs(self.beta) * self.beta_change_limit, min_abs_change)
        if abs(beta_new - self.beta) > max_change:
            beta_new = self.beta + np.sign(beta_new - self.beta) * max_change
        
        self.beta = beta_new
        
        # 7. 协方差更新
        self.P = (1 - K * x_t) * P_pred
        
        # 8. 自适应观测噪声更新（EWMA，α=0.02固定）
        innovation_sq = residual * residual
        self.R = 0.98 * self.R + 0.02 * max(innovation_sq, 1e-6)
        
        # 9. 记录历史
        self.beta_history.append(self.beta)
        
        return {
            'beta': self.beta,
            'residual': residual,
            'K': K,
            'P': self.P,
            'innovation': residual
        }

class SignalGenerator:
    """
    信号生成器 - 仅非Kalman参数可配置
    """
    def __init__(self, 
                 # 核心参数 - 可配置
                 window: int = 60, 
                 z_open: float = 2.0, 
                 z_close: float = 0.5,
                 convergence_days: int = 30, 
                 convergence_threshold: float = 0.02,
                 max_holding_days: int = 30):
                 # 注：所有Kalman参数都写死，不提供配置
        self.window = window
        self.z_open = z_open
        self.z_close = z_close
        self.convergence_days = convergence_days
        self.convergence_threshold = convergence_threshold
        self.max_holding_days = max_holding_days
    
    def init_kalman_filter(self, initial_beta: float, hist_residuals: Optional[np.ndarray] = None) -> KalmanFilter1D:
        """初始化Kalman滤波器"""
        return KalmanFilter1D(initial_beta, hist_residuals)
    
    def calculate_zscore(self, residuals: List[float], window: int) -> float:
        """计算Z-score"""
        if len(residuals) < window:
            return 0.0
        
        recent_residuals = residuals[-window:]
        mean_val = np.mean(recent_residuals)
        std_val = np.std(recent_residuals)
        
        if std_val < 1e-12:
            return 0.0
        
        return (residuals[-1] - mean_val) / std_val
    
    def generate_signal(self, z_score: float, position: Optional[str], 
                       days_held: int) -> str:
        """
        信号生成逻辑（仅在信号期使用）
        """
        # 强制平仓
        if position and days_held >= self.max_holding_days:
            return 'close'
        
        # 平仓条件
        if position and abs(z_score) < self.z_close:
            return 'close'
        
        # 开仓条件
        if not position:
            if z_score < -self.z_open:
                return 'open_long'
            elif z_score > self.z_open:
                return 'open_short'
        
        return 'hold'
    
    def is_converged(self, beta_changes: List[float]) -> bool:
        """判断β是否收敛"""
        if len(beta_changes) < self.convergence_days:
            return False
        
        recent_changes = beta_changes[-self.convergence_days:]
        return all(change < self.convergence_threshold for change in recent_changes)
    
    def process_pair_signals(self, 
                           pair_data: pd.DataFrame,
                           pair_info: Dict,
                           convergence_end: str,
                           signal_start: str,
                           beta_window: str = '1y') -> pd.DataFrame:
        """
        单配对信号生成
        """
        # 获取初始β值
        beta_col = f'beta_{beta_window}'
        if beta_col not in pair_info:
            raise ValueError(f"协整数据中缺少{beta_col}字段")
        initial_beta = pair_info[beta_col]
        
        # 初始化Kalman滤波器（参数全部写死）
        kf = self.init_kalman_filter(initial_beta)
        
        signals = []
        residuals = []
        betas = []
        position = None
        days_held = 0
        converged = False
        beta_changes = []
        
        for _, row in pair_data.iterrows():
            # 更新β
            result = kf.update(row['y'], row['x'])
            beta_t = result['beta']
            
            # 计算残差
            residual = row['y'] - beta_t * row['x']
            residuals.append(residual)
            betas.append(beta_t)
            
            # 收敛评估（收敛期内）
            if row['date'] <= convergence_end:
                if len(betas) >= 2:
                    beta_change = abs(betas[-1] - betas[-2]) / abs(betas[-2]) if abs(betas[-2]) > 1e-12 else 0
                    beta_changes.append(beta_change)
                    
                    # 收敛判定
                    if self.is_converged(beta_changes):
                        converged = True
                
                signal = 'converging'
                z_score = 0.0
                phase = 'convergence_period'
                reason = 'converging'
                
            # 信号期（收敛期结束后）
            elif row['date'] >= signal_start and len(residuals) >= self.window:
                z_score = self.calculate_zscore(residuals, self.window)
                
                # 生成交易信号
                signal = self.generate_signal(z_score, position, days_held)
                phase = 'signal_period'
                
                # 更新持仓状态
                if signal.startswith('open'):
                    position = signal
                    days_held = 1
                    reason = 'z_threshold'
                elif signal == 'close':
                    reason = 'z_threshold' if abs(z_score) < self.z_close else 'force_close'
                    position = None
                    days_held = 0
                elif position:
                    days_held += 1
                    reason = 'holding'
                else:
                    reason = 'no_signal'
                    
            else:
                signal = 'hold'
                z_score = 0.0
                phase = 'signal_period'
                reason = 'insufficient_data'
                
            signals.append({
                'date': row['date'],
                'pair': pair_info['pair'],
                'symbol_x': pair_info['symbol_x'],
                'symbol_y': pair_info['symbol_y'],
                'signal': signal,
                'z_score': z_score,
                'residual': residual,
                'beta': beta_t,
                'beta_initial': initial_beta,
                'days_held': days_held,
                'reason': reason,
                'phase': phase,
                'beta_window_used': beta_window,
                'converged': converged,
                'price_x': row['x'],
                'price_y': row['y']
            })
        
        return pd.DataFrame(signals)
    
    def generate_all_signals(self, 
                           pairs_params: pd.DataFrame,
                           price_data: pd.DataFrame,
                           # 时间参数 - 可配置
                           hist_start: Optional[str] = None,
                           hist_end: Optional[str] = None,
                           convergence_end: Optional[str] = None,
                           signal_start: Optional[str] = None,
                           # 选择β值的时间窗口 - 可配置
                           beta_window: str = '1y') -> pd.DataFrame:
        """
        生成所有配对的信号
        """
        all_signals = []
        
        for _, pair_info in pairs_params.iterrows():
            pair = pair_info['pair']
            symbol_x = pair_info['symbol_x']
            symbol_y = pair_info['symbol_y']
            
            print(f"Processing pair: {pair}")
            
            # 获取配对价格数据
            try:
                pair_price_data = price_data[
                    (price_data['symbol_x'] == symbol_x) & 
                    (price_data['symbol_y'] == symbol_y)
                ].copy().sort_values('date')
                
                if len(pair_price_data) == 0:
                    print(f"Warning: No price data found for pair {pair}")
                    continue
                
                # 生成信号
                pair_signals = self.process_pair_signals(
                    pair_price_data, 
                    pair_info.to_dict(),
                    convergence_end,
                    signal_start,
                    beta_window
                )
                
                all_signals.append(pair_signals)
                
            except Exception as e:
                print(f"Error processing pair {pair}: {str(e)}")
                continue
        
        if not all_signals:
            return pd.DataFrame()
        
        return pd.concat(all_signals, ignore_index=True)

def load_cointegration_results() -> pd.DataFrame:
    """加载协整模块结果"""
    results_path = project_root / "output" / "cointegration" / "results" / "filtered_pairs_20250823_164108.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"协整结果文件不存在: {results_path}")
    
    return pd.read_csv(results_path)

def load_price_data() -> pd.DataFrame:
    """加载价格数据"""
    from lib.data import load_all_symbols_data
    
    print("Loading price data...")
    df = load_all_symbols_data()
    
    # 转换为对数价格
    df['log_price'] = np.log(df['close'])
    
    # 创建配对数据
    pairs_data = []
    symbols = df['symbol'].unique()
    
    for i, symbol_x in enumerate(symbols):
        for symbol_y in symbols[i+1:]:
            # 获取两个符号的数据
            data_x = df[df['symbol'] == symbol_x][['date', 'log_price']].rename(columns={'log_price': 'x'})
            data_y = df[df['symbol'] == symbol_y][['date', 'log_price']].rename(columns={'log_price': 'y'})
            
            # 合并数据
            merged = pd.merge(data_x, data_y, on='date', how='inner')
            if len(merged) > 0:
                merged['symbol_x'] = symbol_x
                merged['symbol_y'] = symbol_y
                pairs_data.append(merged)
    
    if not pairs_data:
        raise ValueError("No paired price data found")
    
    return pd.concat(pairs_data, ignore_index=True)

def define_output_format():
    """定义信号输出格式"""
    format_definition = {
        'file_format': 'CSV',
        'required_columns': [
            'date',           # 日期 (YYYY-MM-DD)
            'pair',           # 配对名称 (例如: AG-NI)
            'symbol_x',       # X品种符号 (低波动)
            'symbol_y',       # Y品种符号 (高波动)
            'signal',         # 信号类型: converging, open_long, open_short, close, hold
            'z_score',        # 残差Z-score
            'residual',       # 当前残差值
            'beta',           # 当前β值
            'beta_initial',   # 初始β值（从协整模块获取）
            'days_held',      # 持仓天数（新开仓为0）
            'reason',         # 信号原因: converging, z_threshold, force_close等
            'phase',          # 阶段标识: convergence_period, signal_period
            'beta_window_used', # 使用的β值时间窗口
            'converged',      # 是否已收敛
            'price_x',        # X价格（对数）
            'price_y'         # Y价格（对数）
        ],
        'data_types': {
            'date': 'string (YYYY-MM-DD)',
            'pair': 'string',
            'symbol_x': 'string',
            'symbol_y': 'string', 
            'signal': 'string',
            'z_score': 'float64',
            'residual': 'float64',
            'beta': 'float64',
            'beta_initial': 'float64',
            'days_held': 'int32',
            'reason': 'string',
            'phase': 'string',
            'beta_window_used': 'string',
            'converged': 'boolean',
            'price_x': 'float64',
            'price_y': 'float64'
        },
        'sample_row': {
            'date': '2024-04-10',
            'pair': 'AG-NI',
            'symbol_x': 'AG',
            'symbol_y': 'NI',
            'signal': 'open_long',
            'z_score': -2.15,
            'residual': -0.0234,
            'beta': 0.8523,
            'beta_initial': 0.8234,
            'days_held': 0,
            'reason': 'z_threshold',
            'phase': 'signal_period',
            'beta_window_used': '1y',
            'converged': True,
            'price_x': 4.123,
            'price_y': 7.456
        }
    }
    return format_definition

def main():
    """主函数：运行信号生成测试"""
    print("=" * 50)
    print("信号生成模块测试")
    print("=" * 50)
    
    # 1. 定义输出格式
    print("\n1. 定义信号输出格式...")
    output_format = define_output_format()
    print(f"输出格式定义完成，包含 {len(output_format['required_columns'])} 个字段")
    
    # 2. 加载协整结果
    print("\n2. 加载协整模块结果...")
    try:
        pairs_params = load_cointegration_results()
        print(f"加载了 {len(pairs_params)} 个协整配对")
        print(f"配对列表: {', '.join(pairs_params['pair'].head(10))}")
    except Exception as e:
        print(f"加载协整结果失败: {e}")
        return
    
    # 3. 加载价格数据
    print("\n3. 加载价格数据...")
    try:
        price_data = load_price_data()
        print(f"加载了 {len(price_data)} 条价格记录")
        print(f"时间范围: {price_data['date'].min()} 到 {price_data['date'].max()}")
    except Exception as e:
        print(f"加载价格数据失败: {e}")
        return
    
    # 4. 初始化信号生成器
    print("\n4. 初始化信号生成器（Kalman参数全部写死）...")
    signal_generator = SignalGenerator(
        window=60,
        z_open=2.0,
        z_close=0.5,
        convergence_days=30,
        convergence_threshold=0.02,
        max_holding_days=30
    )
    print("信号生成器初始化完成")
    
    # 5. 配置时间参数
    convergence_end = '2023-12-31'
    signal_start = '2024-01-01'
    beta_window = '1y'
    
    print(f"\n5. 时间参数配置:")
    print(f"   收敛期结束: {convergence_end}")
    print(f"   信号期开始: {signal_start}")
    print(f"   β时间窗口: {beta_window}")
    
    # 6. 生成信号（只处理前5个配对进行测试）
    print(f"\n6. 生成信号（测试前5个配对）...")
    try:
        test_pairs = pairs_params.head(5).copy()  # 只测试前5个配对
        
        signals = signal_generator.generate_all_signals(
            pairs_params=test_pairs,
            price_data=price_data,
            convergence_end=convergence_end,
            signal_start=signal_start,
            beta_window=beta_window
        )
        
        if len(signals) == 0:
            print("警告: 没有生成任何信号")
            return
        
        print(f"成功生成 {len(signals)} 条信号记录")
        
    except Exception as e:
        print(f"信号生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 保存结果
    print("\n7. 保存信号结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    output_dir = project_root / "output" / "signals_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存信号
    signals_file = output_dir / f"signals_kalman_test_{timestamp}.csv"
    signals.to_csv(signals_file, index=False)
    print(f"信号已保存到: {signals_file}")
    
    # 保存格式定义
    format_file = output_dir / f"output_format_definition_{timestamp}.json"
    import json
    with open(format_file, 'w', encoding='utf-8') as f:
        json.dump(output_format, f, indent=2, ensure_ascii=False)
    print(f"格式定义已保存到: {format_file}")
    
    # 8. 信号统计
    print(f"\n8. 信号统计分析:")
    print(f"   总记录数: {len(signals):,}")
    print(f"   配对数量: {signals['pair'].nunique()}")
    print(f"   时间范围: {signals['date'].min()} 到 {signals['date'].max()}")
    
    print(f"\n   信号类型分布:")
    signal_counts = signals['signal'].value_counts()
    for signal_type, count in signal_counts.items():
        percentage = count / len(signals) * 100
        print(f"     {signal_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n   阶段分布:")
    phase_counts = signals['phase'].value_counts()
    for phase, count in phase_counts.items():
        percentage = count / len(signals) * 100
        print(f"     {phase}: {count:,} ({percentage:.1f}%)")
    
    # 9. 显示样本数据
    print(f"\n9. 样本信号数据（前10条）:")
    sample_columns = ['date', 'pair', 'signal', 'z_score', 'beta', 'phase', 'reason']
    print(signals[sample_columns].head(10).to_string(index=False))
    
    print(f"\n=" * 50)
    print("信号生成测试完成!")
    print(f"信号文件: {signals_file}")
    print(f"格式文件: {format_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()