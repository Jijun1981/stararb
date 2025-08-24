#!/usr/bin/env python3
"""
OLS滚动窗口完整Pipeline
与Kalman滤波pipeline并行运行，进行策略对比

=== 计算条件和方法说明 ===

1. 数据准备条件:
   - 数据范围: 2023-03-01至今 (提前4个月确保7月1日有足够历史数据)
   - 最小数据要求: 每个配对至少60天数据
   - 数据对齐: 使用inner join确保交易日对齐
   - 价格转换: 使用对数价格 log(price) 进行协整关系计算

2. OLS滚动Beta计算方法:
   - 滚动窗口: 60个交易日
   - 回归方程根据direction:
     * x_on_y: log(price_x) = α + β * log(price_y) + ε
     * y_on_x: log(price_y) = α + β * log(price_x) + ε  
   - Beta计算: β = Cov(Y,X) / Var(X), 其中Y是因变量，X是自变量
   - 数值稳定性: 要求标准差 > 1e-8, 方差 > 1e-8
   - 计算频率: 每个交易日更新一次Beta值

3. 残差和Z-score计算:
   - 残差计算: residual = Y_actual - β * X_actual
   - Z-score计算: Z = (当前残差 - 60天残差均值) / 60天残差标准差
   - 残差序列: 使用当前Beta重新计算整个60天窗口的残差
   - 数值要求: 残差标准差 > 1e-8

4. 信号生成条件:
   - 信号开始: 2023年7月1日 (确保有足够历史数据训练Beta)
   - 开仓条件: |Z-score| > 2.0 且 |Z-score| <= 3.2
   - 平仓条件: |Z-score| < 0.5
   - 强制平仓: 持仓30天后强制平仓
   - 止损条件: 损失超过保证金的10%

5. 回测参数设置:
   - 初始资金: 500万元人民币
   - 保证金率: 12% (所有品种统一)
   - 交易费率: 万分之2 (双边)
   - 滑点设置: 每腿3个tick
   - 仓位管理: 每配对约5%资金分配

Pipeline步骤:
1. 加载协整配对结果 (从shifted pipeline输出)
2. 使用60天滚动OLS估计Beta
3. 生成Z-score交易信号 
4. 运行回测分析
5. 生成对比报告

确保2023年7月1日开始就有交易信号
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加lib路径
sys.path.insert(0, '/mnt/e/Star-arb/lib')
sys.path.insert(0, '/mnt/e/Star-arb')

from lib.data import load_from_parquet, load_data
from lib.backtest import BacktestEngine, PositionManager

class OLSRollingPipeline:
    """OLS滚动窗口Pipeline类"""
    
    def __init__(self, window=60, start_date='2023-07-01', end_date='2024-12-31'):
        self.window = window
        self.start_date = start_date  
        self.end_date = end_date
        self.data_start = '2023-03-01'  # 提前4个月确保7月1日有足够历史数据
        
        # 创建输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"/mnt/e/Star-arb/output/ols_rolling_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"OLS滚动Pipeline初始化")
        print(f"滚动窗口: {self.window}天")
        print(f"信号日期范围: {self.start_date} 到 {self.end_date}")
        print(f"输出目录: {self.output_dir}")
    
    def load_cointegration_pairs(self):
        """加载协整配对结果"""
        coint_file = "/mnt/e/Star-arb/output/pipeline_shifted/cointegration_results.csv"
        
        if not os.path.exists(coint_file):
            raise FileNotFoundError(f"协整结果文件不存在: {coint_file}")
            
        coint_results = pd.read_csv(coint_file)
        print(f"\n加载协整配对: {len(coint_results)}个")
        
        return coint_results
    
    def calculate_rolling_ols_beta(self, y_data, x_data, window=None):
        """
        计算滚动OLS Beta
        
        Args:
            y_data: 因变量序列
            x_data: 自变量序列  
            window: 滚动窗口大小
        
        Returns:
            pandas.Series: 滚动Beta序列
        """
        if window is None:
            window = self.window
            
        aligned_data = pd.DataFrame({'y': y_data, 'x': x_data}).dropna()
        
        if len(aligned_data) < window:
            return pd.Series(dtype=float)
        
        beta_series = pd.Series(index=aligned_data.index, dtype=float)
        
        for i in range(window-1, len(aligned_data)):
            y_window = aligned_data['y'].iloc[i-window+1:i+1]
            x_window = aligned_data['x'].iloc[i-window+1:i+1]
            
            if len(y_window) == window and y_window.std() > 1e-8 and x_window.std() > 1e-8:
                # OLS回归: y = alpha + beta * x
                covariance = np.cov(y_window, x_window, ddof=1)[0, 1]
                variance_x = np.var(x_window, ddof=1)
                
                if variance_x > 1e-8:
                    beta = covariance / variance_x
                    beta_series.iloc[i] = beta
        
        return beta_series
    
    def generate_pair_signals(self, pair_info):
        """
        生成单个配对的信号
        
        Args:
            pair_info: dict, 包含配对信息
            
        Returns:
            DataFrame: 信号数据
        """
        symbol_x = pair_info['symbol_x']
        symbol_y = pair_info['symbol_y']
        direction = pair_info['direction']
        
        try:
            # 加载数据
            df_x = load_from_parquet(symbol_x)
            df_y = load_from_parquet(symbol_y)
            
            # 筛选日期范围
            start_dt = pd.Timestamp(self.data_start)
            end_dt = pd.Timestamp(self.end_date)
            df_x = df_x[(df_x.index >= start_dt) & (df_x.index <= end_dt)]
            df_y = df_y[(df_y.index >= start_dt) & (df_y.index <= end_dt)]
            
            if df_x.empty or df_y.empty:
                print(f"  ❌ 数据为空: {symbol_x}-{symbol_y}")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"  ❌ 数据加载失败: {symbol_x}-{symbol_y}, {e}")
            return pd.DataFrame()
        
        # 合并数据并对齐
        data = pd.merge(df_x[['close']], df_y[['close']], 
                       left_index=True, right_index=True, 
                       how='inner', suffixes=('_x', '_y'))
        
        if len(data) < self.window:
            print(f"  ❌ 数据不足{self.window}天: {symbol_x}-{symbol_y}, 只有{len(data)}天")
            return pd.DataFrame()
        
        # 计算对数价格
        data['log_x'] = np.log(data['close_x'])
        data['log_y'] = np.log(data['close_y'])
        
        # 根据方向确定回归关系
        if direction == 'x_on_y':
            # X对Y回归: log_x = alpha + beta * log_y
            y_var = data['log_x']  # 因变量
            x_var = data['log_y']  # 自变量
        else:  # y_on_x
            # Y对X回归: log_y = alpha + beta * log_x
            y_var = data['log_y']  # 因变量  
            x_var = data['log_x']  # 自变量
        
        # 计算滚动OLS Beta
        rolling_beta = self.calculate_rolling_ols_beta(y_var, x_var, self.window)
        
        if rolling_beta.empty:
            print(f"  ❌ Beta计算失败: {symbol_x}-{symbol_y}")
            return pd.DataFrame()
        
        # 生成信号
        signals = []
        
        for date in rolling_beta.dropna().index:
            if date < pd.Timestamp(self.start_date):
                continue
                
            beta = rolling_beta[date]
            if pd.isna(beta):
                continue
            
            # Beta约束检查: 绝对值必须在0.3-3之间
            if abs(beta) < 0.3 or abs(beta) > 3.0:
                continue
                
            # 计算当前残差
            current_residual = y_var[date] - beta * x_var[date]
            
            # 计算滚动窗口内的残差序列用于Z-score计算
            end_idx = data.index.get_loc(date)
            start_idx = max(0, end_idx - self.window + 1)
            window_data = data.iloc[start_idx:end_idx+1]
            
            # 使用当前Beta计算窗口内所有残差
            if direction == 'x_on_y':
                residuals_window = window_data['log_x'] - beta * window_data['log_y']
            else:
                residuals_window = window_data['log_y'] - beta * window_data['log_x']
            
            if len(residuals_window) > 1 and residuals_window.std() > 1e-8:
                z_score = (current_residual - residuals_window.mean()) / residuals_window.std()
            else:
                continue
                
            signals.append({
                'date': date,
                'pair': f"{symbol_x}-{symbol_y}",
                'symbol_x': symbol_x,
                'symbol_y': symbol_y,
                'direction': direction,
                'price_x': data.loc[date, 'close_x'],
                'price_y': data.loc[date, 'close_y'],
                'ols_beta': beta,
                'residual': current_residual,
                'z_score': z_score
            })
        
        if not signals:
            print(f"  ❌ 未生成信号: {symbol_x}-{symbol_y}")
            return pd.DataFrame()
            
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('date', inplace=True)
        
        print(f"  ✅ {symbol_x}-{symbol_y}: {len(signals_df)}个信号, 范围: {signals_df.index.min().strftime('%Y-%m-%d')} 到 {signals_df.index.max().strftime('%Y-%m-%d')}")
        
        return signals_df
    
    def generate_all_signals(self, coint_pairs):
        """生成所有配对的信号"""
        print(f"\n{'='*60}")
        print("开始生成OLS滚动信号")
        print(f"{'='*60}")
        
        all_signals = []
        successful_pairs = 0
        
        for idx, row in coint_pairs.iterrows():
            pair_info = {
                'symbol_x': row['symbol_x'],
                'symbol_y': row['symbol_y'], 
                'direction': row['direction'],
                'pvalue_4y': row['pvalue_4y'],
                'beta_1y': row['beta_1y']
            }
            
            print(f"[{idx+1:2d}/{len(coint_pairs)}] 处理: {row['symbol_x']}-{row['symbol_y']} ({row['direction']})")
            
            signals_df = self.generate_pair_signals(pair_info)
            if not signals_df.empty:
                all_signals.append(signals_df)
                successful_pairs += 1
        
        print(f"\n信号生成完成: {successful_pairs}/{len(coint_pairs)} 配对成功")
        
        if not all_signals:
            raise ValueError("未生成任何信号")
            
        # 合并所有信号
        combined_signals = pd.concat(all_signals, ignore_index=False)
        combined_signals.sort_index(inplace=True)
        
        return combined_signals
    
    def save_signals(self, signals_df):
        """保存信号数据"""
        signals_file = f"{self.output_dir}/signals_ols_rolling_{self.timestamp}.csv"
        signals_df.to_csv(signals_file)
        
        print(f"\n信号数据保存: {signals_file}")
        print(f"总信号数: {len(signals_df)}")
        print(f"配对数: {signals_df['pair'].nunique()}")
        print(f"日期范围: {signals_df.index.min()} 到 {signals_df.index.max()}")
        
        return signals_file
    
    def validate_signal_timing(self, signals_df):
        """验证信号时间是否符合要求"""
        first_signal_date = signals_df.index.min()
        target_date = pd.Timestamp(self.start_date)
        
        print(f"\n信号时间验证:")
        print(f"目标开始日期: {target_date.strftime('%Y-%m-%d')}")
        print(f"实际首个信号: {first_signal_date.strftime('%Y-%m-%d')}")
        
        # 允许1周内的差异
        if first_signal_date <= target_date + timedelta(days=7):
            print("✅ 信号时间符合要求")
            return True
        else:
            print("❌ 信号开始时间偏晚")
            return False
    
    def run_backtest(self, signals_df):
        """运行回测分析"""
        print(f"\n{'='*60}")
        print("开始OLS滚动策略回测")
        print(f"{'='*60}")
        
        try:
            # 回测参数
            backtest_params = {
                'initial_capital': 5000000,
                'z_open_threshold': 2.5,
                'z_close_threshold': 0.5, 
                'z_open_max': 3.2,
                'stop_loss_pct': 0.15,  # 15%止损
                'max_hold_days': 30
            }
            
            print("回测参数:")
            for key, value in backtest_params.items():
                print(f"  {key}: {value}")
            
            # 加载价格数据用于回测 (原始价格，非对数)
            print("\n加载价格数据用于回测...")
            SYMBOLS = ['CU0', 'AL0', 'ZN0', 'NI0', 'SN0', 'PB0', 'AG0', 'AU0', 
                      'RB0', 'HC0', 'I0', 'SF0', 'SM0', 'SS0']
            
            price_data = load_data(
                symbols=SYMBOLS,
                start_date=self.start_date,
                end_date=self.end_date,
                columns=['close'],
                log_price=False  # 回测使用原始价格
            )
            
            # 获取所有交易日期
            all_dates = sorted(price_data.index.unique())
            start_date = pd.Timestamp(self.start_date)
            end_date = pd.Timestamp(self.end_date)
            trading_dates = [d for d in all_dates if start_date <= d <= end_date]
            
            # 初始化回测引擎
            backtest_engine = BacktestEngine(
                initial_capital=backtest_params['initial_capital'],
                margin_rate=0.12,
                commission_rate=0.0002,
                slippage_ticks=3,
                stop_loss_pct=backtest_params['stop_loss_pct'],
                max_holding_days=backtest_params['max_hold_days']
            )
            
            # 加载合约规格
            import json
            specs_file = "/mnt/e/Star-arb/configs/contract_specs.json"
            with open(specs_file, 'r', encoding='utf-8') as f:
                contract_specs = json.load(f)
            backtest_engine.contract_specs = contract_specs
            
            # 设置仓位权重（每配对5%）
            position_weights = {}
            for _, signal in signals_df.head(22).iterrows():  # 取22个配对
                pair = signal['pair']
                position_weights[pair] = 0.05
            backtest_engine.position_weights = position_weights
            
            # 按日期分组信号
            signals_by_date = {}
            for _, signal in signals_df.iterrows():
                signal_date = signal.name
                if signal_date not in signals_by_date:
                    signals_by_date[signal_date] = []
                
                # 确定信号类型
                z = signal['z_score']
                if abs(z) >= backtest_params['z_open_threshold'] and abs(z) <= backtest_params['z_open_max']:
                    # 开仓信号
                    if z > 0:
                        signal_type = 'open_short'  # Z-score > 0, 做空价差
                    else:
                        signal_type = 'open_long'   # Z-score < 0, 做多价差
                elif abs(z) <= backtest_params['z_close_threshold']:
                    signal_type = 'close'
                else:
                    signal_type = None  # 无信号
                
                if signal_type is None:
                    continue
                
                # 转换为信号字典格式
                signal_dict = {
                    'pair': signal['pair'],
                    'signal': signal_type,
                    'date': signal_date,  # 添加date字段
                    'symbol_x': signal['symbol_x'],
                    'symbol_y': signal['symbol_y'],
                    'direction': signal['direction'],
                    'beta': signal['ols_beta'],  # 使用OLS beta
                    'theoretical_ratio': abs(signal['ols_beta']),  # 手数计算需要的理论比率
                    'z_score': signal['z_score'],
                    'price_x': signal['price_x'],
                    'price_y': signal['price_y'],
                    'ols_beta': signal['ols_beta']  # 额外记录OLS beta
                }
                signals_by_date[signal_date].append(signal_dict)
            
            print(f"回测日期范围: {start_date} 至 {end_date}")
            print(f"总交易日数: {len(trading_dates)}")
            print(f"有信号日数: {len(signals_by_date)}")
            
            # 执行每日回测
            processed_signals = 0
            for i, current_date in enumerate(trading_dates):
                if i % 50 == 0:
                    print(f"  处理进度: {i+1}/{len(trading_dates)} ({current_date.strftime('%Y-%m-%d')})")
                    
                # 获取当前价格
                current_prices = {}
                for symbol in SYMBOLS:
                    col_name = f"{symbol}_close"
                    if col_name in price_data.columns:
                        current_prices[symbol] = price_data.loc[current_date, col_name]
                
                # 处理当日信号
                if current_date in signals_by_date:
                    for signal in signals_by_date[current_date]:
                        if backtest_engine.execute_signal(signal, current_prices, current_date):
                            processed_signals += 1
                
                # 风险管理 - 检查并执行止损等风险控制
                force_close_list = backtest_engine.run_risk_management(current_date, current_prices)
                
                # 执行强制平仓
                for item in force_close_list:
                    pair = item['pair']
                    reason = item['reason']
                    backtest_engine._close_position(pair, current_prices, reason, current_date)
            
            print(f"处理信号总数: {processed_signals}")
            
            # 生成回测结果
            performance_summary = backtest_engine.generate_performance_summary()
            
            # 获取交易记录和其他数据
            results = {
                'summary': performance_summary,
                'trades': backtest_engine.trade_records,
                'positions': backtest_engine.position_manager.positions,
                'daily_pnl': getattr(backtest_engine, 'daily_pnl', {}),
                'capital_history': getattr(backtest_engine, 'equity_curve', [])
            }
            
            # 保存结果文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存交易记录
            if results.get('trades'):
                trades_file = f"{self.output_dir}/trades_{timestamp}.csv"
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_csv(trades_file, index=False)
                print(f"\n交易记录保存: {trades_file}")
            
            # 保存回测报告
            report_file = f"{self.output_dir}/backtest_report_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"回测报告保存: {report_file}")
            
            return results
            
        except Exception as e:
            print(f"❌ 回测运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_summary_report(self, backtest_result):
        """生成策略总结报告"""
        if not backtest_result or 'summary' not in backtest_result:
            print("❌ 无法生成报告，回测结果缺失")
            return
            
        summary = backtest_result['summary']
        
        print(f"\n{'='*60}")
        print("OLS滚动策略回测结果")
        print(f"{'='*60}")
        
        key_metrics = [
            ('总收益率', 'total_return', '.2%'),
            ('年化收益率', 'annualized_return', '.2%'),
            ('夏普比率', 'sharpe_ratio', '.3f'),
            ('最大回撤', 'max_drawdown', '.2%'),
            ('总交易次数', 'total_trades', 'd'),
            ('胜率', 'win_rate', '.1%'),
            ('平均持仓天数', 'avg_hold_days', '.1f'),
            ('盈亏比', 'profit_loss_ratio', '.2f')
        ]
        
        for name, key, fmt in key_metrics:
            value = summary.get(key, 'N/A')
            if value != 'N/A':
                if fmt.endswith('%'):
                    print(f"{name:12}: {value:{fmt}}")
                elif fmt.endswith('f'):
                    print(f"{name:12}: {value:{fmt}}")
                else:
                    print(f"{name:12}: {value}")
            else:
                print(f"{name:12}: {value}")
        
        # 保存详细报告
        report_data = {
            'pipeline_type': 'OLS_Rolling',
            'window_size': self.window,
            'signal_period': f"{self.start_date} to {self.end_date}",
            'timestamp': self.timestamp,
            'summary': summary,
            'parameters': {
                'z_open_threshold': 2.5,
                'z_close_threshold': 0.5,
                'z_open_max': 3.2,
                'stop_loss_pct': 0.15,  # 15%止损
                'max_hold_days': 30,
                'initial_capital': 5000000
            }
        }
        
        report_file = f"{self.output_dir}/backtest_report_{self.timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            
        print(f"\n详细报告已保存: {report_file}")
        
    def run_complete_pipeline(self):
        """运行完整pipeline流程"""
        print(f"{'='*80}")
        print("OLS滚动窗口完整Pipeline启动")
        print(f"{'='*80}")
        
        try:
            # 1. 加载协整配对
            coint_pairs = self.load_cointegration_pairs()
            
            # 2. 生成信号
            signals_df = self.generate_all_signals(coint_pairs)
            
            # 3. 保存信号
            signals_file = self.save_signals(signals_df)
            
            # 4. 验证信号时间
            self.validate_signal_timing(signals_df)
            
            # 5. 运行回测
            backtest_result = self.run_backtest(signals_df)
            
            # 6. 生成报告
            if backtest_result:
                self.generate_summary_report(backtest_result)
                
            print(f"\n{'='*80}")
            print("OLS滚动Pipeline完成!")
            print(f"输出目录: {self.output_dir}")
            print(f"{'='*80}")
            
            return {
                'signals_file': signals_file,
                'backtest_result': backtest_result,
                'output_dir': self.output_dir
            }
            
        except Exception as e:
            print(f"❌ Pipeline运行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    # 创建并运行OLS滚动Pipeline
    pipeline = OLSRollingPipeline(
        window=60,
        start_date='2023-07-01',
        end_date='2024-12-31'
    )
    
    result = pipeline.run_complete_pipeline()
    
    if result:
        print("\n🎉 Pipeline成功完成!")
        print(f"查看结果: {result['output_dir']}")
    else:
        print("\n❌ Pipeline执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()