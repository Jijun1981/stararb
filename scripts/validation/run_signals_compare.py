#!/usr/bin/env python3
"""
生成并比较两套算法（库实现 vs 独立实现）的开/平仓点：
- 库实现：lib.signal_generation.SignalGenerator.process_pair_signals
- 独立实现：本脚本内按 03_signal_generation.md 重新实现（Kalman + Z-score）

流程：
1) 使用 lib.coint.CointegrationAnalyzer 选取“5年且1年同时满足”的配对集合（可限制Top N）
2) 为每个配对生成两套信号（同样的时间边界与阈值）
3) 比较 open_long/open_short/close 事件的日期与类型，输出差异报告

不修改库代码，仅作为只读验证。
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer
from lib.signal_generation import SignalGenerator


# ----------------------- 独立算法实现（与文档一致） ----------------------- #

class IndependentKalman:
    def __init__(self, initial_beta: float, Q: float = 1e-4, R: Optional[float] = None, P0: float = 0.1):
        self.beta = float(initial_beta)
        self.P = float(P0)
        self.Q = float(Q)
        self.R = float(R) if R is not None else 1e-2
        self.beta_history: List[float] = [self.beta]

    def update(self, y: float, x: float) -> Dict[str, float]:
        if not np.isfinite(y) or not np.isfinite(x):
            raise ValueError("Invalid y/x")
        if abs(x) < 1e-10:
            raise ValueError("x too small")

        beta_pred = self.beta
        P_pred = self.P + self.Q

        y_pred = beta_pred * x
        residual = y - y_pred

        S = x * P_pred * x + self.R
        S = max(S, 1e-15)
        K = P_pred * x / S

        beta_new = beta_pred + K * residual

        # β变化限制：≤5%，且最小绝对变化0.001
        min_abs_change = 0.001
        max_change = max(abs(self.beta) * 0.05, min_abs_change)
        if abs(beta_new - self.beta) > max_change:
            beta_new = self.beta + np.sign(beta_new - self.beta) * max_change

        self.beta = float(beta_new)
        self.P = float((1 - K * x) * P_pred)

        # R自适应（EWMA α=0.02）
        innovation_sq = float(residual * residual)
        self.R = 0.98 * self.R + 0.02 * max(innovation_sq, 1e-6)

        self.beta_history.append(self.beta)
        return {"beta": self.beta, "residual": residual, "P": self.P}


def independent_process_pair_signals(
    pair_data: pd.DataFrame,
    initial_beta: float,
    convergence_end: str,
    signal_start: str,
    window: int = 60,
    z_open: float = 2.0,
    z_close: float = 0.5,
    convergence_days: int = 20,
    convergence_threshold: float = 0.01,
    hist_start: Optional[str] = None,
    hist_end: Optional[str] = None,
) -> pd.DataFrame:
    required_cols = ["date", "x", "y"]
    for c in required_cols:
        if c not in pair_data.columns:
            raise ValueError(f"missing column: {c}")

    data = pair_data.sort_values("date").copy()
    # 估计R（历史区段）
    R0 = 1e-2
    if hist_start and hist_end:
        hist = data[(data["date"] >= hist_start) & (data["date"] <= hist_end)]
        if len(hist) > 0:
            init_resid = hist["y"].values - initial_beta * hist["x"].values
            R0 = float(np.var(init_resid)) if np.isfinite(np.var(init_resid)) else 1e-2

    kf = IndependentKalman(initial_beta=initial_beta, Q=1e-4, R=R0, P0=0.1)

    results: List[Dict] = []
    residuals: List[float] = []
    betas: List[float] = []
    beta_changes: List[float] = []
    position: Optional[str] = None
    days_held = 0
    converged = False

    conv_end_dt = pd.to_datetime(convergence_end)
    sig_start_dt = pd.to_datetime(signal_start)

    for _, row in data.iterrows():
        y_t = float(row["y"])
        x_t = float(row["x"])
        dt = pd.to_datetime(row["date"]) if not isinstance(row["date"], pd.Timestamp) else row["date"]

        # KF更新
        kf_out = kf.update(y_t, x_t)
        beta_t = float(kf_out["beta"])
        resid = float(kf_out["residual"])
        residuals.append(resid)
        betas.append(beta_t)

        # 收敛评估（≥60样本后才评估）
        if dt <= conv_end_dt:
            if len(betas) >= 2:
                prev = betas[-2]
                change = abs(beta_t - prev) / abs(prev) if abs(prev) > 1e-12 else 0.0
                beta_changes.append(change)
                if len(betas) >= 60 and len(beta_changes) >= convergence_days:
                    recent = beta_changes[-convergence_days:]
                    if all(c < convergence_threshold for c in recent):
                        converged = True

            signal = "converging"
            z_score = 0.0
            reason = "converging"

        elif dt >= sig_start_dt:
            # 60日滚动Z-score
            if len(residuals) >= window:
                window_res = np.array(residuals[-window:])
                # 4σ保护：如需要可在此用于β更新门控；此实现遵循库的做法不跳过，仅用于Z-score
                mean = float(np.mean(window_res))
                std = float(np.std(window_res, ddof=1))
                if std < 1e-10:
                    z_score = 0.0
                else:
                    z_score = float((resid - mean) / std)

                # 生成信号（阈值边界与库一致：开仓：>=，平仓：<=）
                if position and days_held >= 30:
                    signal = "close"
                elif position and abs(z_score) <= z_close:
                    signal = "close"
                elif not position:
                    if z_score <= -z_open:
                        signal = "open_long"
                    elif z_score >= z_open:
                        signal = "open_short"
                    else:
                        signal = "hold"
                else:
                    signal = "hold"

                # 更新持仓天数
                if signal.startswith("open"):
                    position = signal
                    days_held = 1
                    reason = "z_threshold"
                elif signal == "close":
                    position = None
                    days_held = 0
                    reason = "z_threshold" if abs(z_score) <= z_close else "force_close"
                elif position:
                    days_held += 1
                    reason = "holding"
                else:
                    reason = "no_signal"
            else:
                signal = "hold"
                z_score = 0.0
                reason = "insufficient_data"

        else:
            signal = "hold"
            z_score = 0.0
            reason = "transition_period"

        results.append({
            "date": dt,
            "signal": signal,
            "z_score": z_score,
            "residual": resid,
            "beta": beta_t,
            "days_held": days_held,
            "reason": reason,
        })

    return pd.DataFrame(results)


# ----------------------------- 对比执行逻辑 ----------------------------- #

def build_price_table_for_pair(data: pd.DataFrame, symbol_x: str, symbol_y: str) -> pd.DataFrame:
    # data 为宽表: 列如 AG0_close, AU0_close
    df = data.copy().reset_index().rename(columns={"index": "date"})
    # 将 AG0_close -> AG0
    col_x = f"{symbol_x}_close"
    col_y = f"{symbol_y}_close"
    if col_x not in df.columns or col_y not in df.columns:
        # 容错：若列已为 symbol 名
        col_x = symbol_x if symbol_x in df.columns else col_x
        col_y = symbol_y if symbol_y in df.columns else col_y
    out = df[["date", col_x, col_y]].copy()
    out = out.rename(columns={col_x: "x", col_y: "y"})
    return out


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "signal" not in df.columns:
        return pd.DataFrame(columns=["date", "signal"])  
    events = df[df["signal"].isin(["open_long", "open_short", "close"])][["date", "signal"]].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    return events.reset_index(drop=True)


def main():
    # 基础配置
    p_threshold = 0.05
    top_n = 10  # 限制对比配对数量，避免运行过久
    window = 60
    z_open, z_close = 2.0, 0.5
    convergence_end = "2024-06-30"
    signal_start = "2024-07-01"
    hist_start, hist_end = "2023-01-01", "2023-12-31"

    # 符号集合
    symbols = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']

    # 加载对数价格宽表
    data = load_data(symbols, start_date='2020-01-01', columns=['close'], log_price=True)

    # 选配对（5年且1年）
    analyzer = CointegrationAnalyzer(data)
    pairs_df = analyzer.screen_all_pairs(p_threshold=p_threshold)
    fallback_pairs: List[Tuple[str, str]] = [('AG0','AU0'), ('CU0','AL0'), ('RB0','HC0')]
    if len(pairs_df) == 0:
        print("No pairs pass filters. Fallback to default demonstration pairs.")
        # 构造一个DataFrame，包含必要字段
        rows = []
        for x, y in fallback_pairs:
            # 估计5年beta作为初始beta（全样本近似）
            try:
                log_x = data[f"{x}_close"] if f"{x}_close" in data.columns else data[x]
                log_y = data[f"{y}_close"] if f"{y}_close" in data.columns else data[y]
                n = len(log_x)
                w = min(1260, n)
                xx = log_x.values[-w:]
                yy = log_y.values[-w:]
                X = np.column_stack([np.ones(len(xx)), xx])
                beta = float(np.linalg.lstsq(X, yy, rcond=None)[0][1]) if len(xx) >= 20 else 1.0
            except Exception:
                beta = 1.0
            rows.append({
                'pair': f'{x}-{y}',
                'symbol_x': x,
                'symbol_y': y,
                'beta_5y': beta,
                'pvalue_5y': np.nan,
                'pvalue_1y': np.nan,
            })
        pairs_df = pd.DataFrame(rows)
    else:
        pairs_df = pairs_df.head(top_n)

    print(f"Selected pairs (Top {top_n}): {len(pairs_df)}")
    try:
        print(pairs_df[['pair','pvalue_5y','pvalue_1y']].to_string(index=False))
    except Exception:
        print(pairs_df['pair'].head(top_n).to_string(index=False))

    # 对比并记录
    out_lines: List[str] = []
    out_lines.append("# 开/平仓点一致性对比报告\n")
    out_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    out_lines.append(f"筛选规则: 5y<={p_threshold} 且 1y<={p_threshold}, Top {top_n}\n")

    total_match = 0
    total_events = 0

    for _, row in pairs_df.iterrows():
        pair = row['pair']
        symbol_x = row['symbol_x'] if 'symbol_x' in row else row.get('x_symbol', '').strip()
        symbol_y = row['symbol_y'] if 'symbol_y' in row else row.get('y_symbol', '').strip()
        beta_initial = row.get('beta_5y', np.nan)

        price_df = build_price_table_for_pair(data, symbol_x, symbol_y)

        # 库实现信号
        sg = SignalGenerator()
        lib_signals = sg.process_pair_signals(
            pair_data=price_df,
            initial_beta=float(beta_initial) if np.isfinite(beta_initial) else 1.0,
            convergence_end=convergence_end,
            signal_start=signal_start,
            hist_start=hist_start,
            hist_end=hist_end,
        )

        # 独立实现信号
        ind_signals = independent_process_pair_signals(
            pair_data=price_df,
            initial_beta=float(beta_initial) if np.isfinite(beta_initial) else 1.0,
            convergence_end=convergence_end,
            signal_start=signal_start,
            window=window,
            z_open=z_open,
            z_close=z_close,
            convergence_days=20,
            convergence_threshold=0.01,
            hist_start=hist_start,
            hist_end=hist_end,
        )

        lib_events = extract_events(lib_signals)
        ind_events = extract_events(ind_signals)

        # 事件集合比较（按日期+类型）
        lib_set = set((pd.to_datetime(d).date(), s) for d, s in lib_events[['date', 'signal']].itertuples(index=False))
        ind_set = set((pd.to_datetime(d).date(), s) for d, s in ind_events[['date', 'signal']].itertuples(index=False))

        inter = lib_set & ind_set
        only_lib = lib_set - ind_set
        only_ind = ind_set - lib_set

        total_match += len(inter)
        total_events += max(len(lib_set), len(ind_set))

        out_lines.append(f"## {pair}")
        out_lines.append(f"- 库事件数: {len(lib_set)}, 独立事件数: {len(ind_set)}, 交集: {len(inter)}")
        if only_lib:
            out_lines.append("- 仅库出现:")
            for d, s in sorted(only_lib):
                out_lines.append(f"  - {d} {s}")
        if only_ind:
            out_lines.append("- 仅独立出现:")
            for d, s in sorted(only_ind):
                out_lines.append(f"  - {d} {s}")
        out_lines.append("")

    overall = (total_match / total_events) if total_events > 0 else 1.0
    out_lines.append(f"\n## 总体一致度\n- 事件一致率: {overall:.2%}")

    out_dir = ROOT / 'output' / 'signals_compare'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = out_dir / f'signals_compare_{ts}.md'
    report_path.write_text("\n".join(out_lines), encoding='utf-8')
    print(f"报告: {report_path}")


if __name__ == '__main__':
    main()


