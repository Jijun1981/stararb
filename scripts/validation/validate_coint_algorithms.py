#!/usr/bin/env python3
"""
独立协整算法验证脚本（只读，不改库代码）

目标：
- 实现需求文档(02_cointegration_pairing.md)版本的EG/多窗口/方向/半衰期算法
- 与现有 lib.coint.CointegrationAnalyzer.screen_all_pairs 结果逐对比
- 生成Markdown报告，量化差异并列出可疑配对

用法：
  python scripts/validation/validate_coint_algorithms.py

依赖：pandas, numpy, statsmodels, scipy
"""

import sys
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional
import logging
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# 项目根目录到路径，确保可以导入lib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("coint_validation")


# ---------------------------- 文档版算法实现 ---------------------------- #

WINDOWS_TRADING_DAYS: Dict[str, int] = {
    '5y': 1260,
    '4y': 1008,
    '3y': 756,
    '2y': 504,
    '1y': 252,
}


def compute_volatility_from_log_prices(
    log_prices: np.ndarray,
    dates: pd.DatetimeIndex,
    start_date: Optional[str] = None,
) -> float:
    """基于对数价格计算年化波动率（文档：returns = diff(log_price), vol = std * sqrt(252)）"""
    if start_date:
        mask = dates >= pd.to_datetime(start_date)
        series = log_prices[mask]
    else:
        series = log_prices

    if series is None or len(series) < 3:
        return np.nan

    returns = np.diff(series)
    if len(returns) == 0:
        return np.nan

    return float(np.std(returns) * math.sqrt(252))


def determine_direction_doc(
    series_1: np.ndarray,
    series_2: np.ndarray,
    dates: pd.DatetimeIndex,
    symbol_1: str,
    symbol_2: str,
    volatility_start: Optional[str] = None,
) -> Tuple[str, str, str]:
    """文档版方向判定：最近一年（或指定起始）波动率，低波动作X，高波动作Y"""
    vol_1 = compute_volatility_from_log_prices(series_1, dates, volatility_start)
    vol_2 = compute_volatility_from_log_prices(series_2, dates, volatility_start)

    if not np.isfinite(vol_1) or not np.isfinite(vol_2):
        return 'y_on_x', symbol_1, symbol_2

    if vol_1 < vol_2:
        return 'y_on_x', symbol_1, symbol_2
    elif vol_1 > vol_2:
        return 'x_on_y', symbol_2, symbol_1
    else:
        return 'y_on_x', symbol_1, symbol_2


def eg_test_doc(
    x: np.ndarray,
    y: np.ndarray,
    direction: str = 'y_on_x',
    adf_regression: str = 'n',
    autolag: str = 'AIC',
    maxlag: Optional[int] = None,
) -> Dict:
    """文档版EG两步法（默认与库保持一致：ADF regression='n', autolag='AIC', maxlag=T**(1/3)）"""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) != len(y) or len(x) < 20:
        raise ValueError("Invalid series length for EG test")

    # 步骤1：OLS
    if direction == 'y_on_x':
        X = add_constant(x)
        model = OLS(y, X).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        resid = np.asarray(model.resid)
    else:  # x_on_y
        Y = add_constant(y)
        model = OLS(x, Y).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        resid = np.asarray(model.resid)

    # 步骤2：ADF on residuals
    if maxlag is None:
        maxlag = int(np.floor(len(resid) ** (1/3)))
    adf_stat, pvalue, *_ = adfuller(resid, regression=adf_regression, autolag=autolag, maxlag=maxlag)

    return {
        'pvalue': float(pvalue),
        'adf_stat': float(adf_stat),
        'beta': round(beta, 6),  # 与库精度对齐
        'alpha': float(alpha),
        'residuals': resid,
        'r_squared': float(model.rsquared),
    }


def halflife_doc(residuals: np.ndarray) -> Optional[float]:
    """文档/行业常用：AR(1)系数rho，hl = -ln(2) / ln(|rho|)，若|rho|>=1或不可估计返回None"""
    if residuals is None or len(residuals) < 10:
        return None

    lagged = residuals[:-1]
    curr = residuals[1:]
    if len(lagged) < 5 or np.var(lagged) < 1e-12:
        return None

    X = add_constant(lagged)
    try:
        model = OLS(curr, X).fit()
        rho = float(model.params[1])  # AR(1)系数
        if not (0 < abs(rho) < 1):
            return None
        hl = -math.log(2.0) / math.log(abs(rho))
        return float(hl) if np.isfinite(hl) and hl > 0 else None
    except Exception:
        return None


def multi_window_doc(x: np.ndarray, y: np.ndarray, direction: str) -> Dict[str, Optional[Dict]]:
    results: Dict[str, Optional[Dict]] = {}
    n = len(x)
    for wname, wsize in WINDOWS_TRADING_DAYS.items():
        if n >= wsize:
            subx = x[-wsize:]
            suby = y[-wsize:]
            try:
                eg = eg_test_doc(subx, suby, direction=direction)
                eg['halflife'] = halflife_doc(eg['residuals'])
                results[wname] = eg
            except Exception as e:
                logger.warning(f"EG failed on window {wname}: {e}")
                results[wname] = None
        else:
            results[wname] = None
    return results


def screen_all_pairs_doc(data: pd.DataFrame, volatility_start: Optional[str] = None) -> pd.DataFrame:
    symbols: List[str] = list(data.columns)
    out_rows: List[Dict] = []
    for s1, s2 in combinations(symbols, 2):
        x_series = data[s1].values
        y_series = data[s2].values
        dates = data.index

        direction, symbol_x, symbol_y = determine_direction_doc(
            data[s1].values, data[s2].values, dates, s1, s2, volatility_start
        )

        # 调整最终回归输入序列
        if symbol_x == s1:
            x_final, y_final = x_series, y_series
        else:
            x_final, y_final = y_series, x_series

        mw = multi_window_doc(x_final, y_final, direction)
        row: Dict[str, object] = {
            'pair': f"{symbol_x}-{symbol_y}",
            'symbol_x': symbol_x,
            'symbol_y': symbol_y,
            'direction': direction,
        }

        for w in WINDOWS_TRADING_DAYS:
            if mw[w] is not None:
                row[f'pvalue_{w}'] = mw[w]['pvalue']
                row[f'beta_{w}'] = mw[w]['beta']
                row[f'halflife_{w}'] = mw[w]['halflife'] if 'halflife' in mw[w] else None
            else:
                row[f'pvalue_{w}'] = np.nan
                row[f'beta_{w}'] = np.nan
                row[f'halflife_{w}'] = np.nan

        # 计算最近一年的波动率（用于对比）
        vol_x = compute_volatility_from_log_prices(data[symbol_x].values, dates, volatility_start)
        vol_y = compute_volatility_from_log_prices(data[symbol_y].values, dates, volatility_start)
        row['volatility_x'] = vol_x
        row['volatility_y'] = vol_y

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def load_symbols_from_yaml(config_path: Path) -> Tuple[List[str], Optional[str]]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        symbols = cfg.get('symbols', {}).get('all', [])
        vol_start = cfg.get('periods', {}).get('volatility_start', '2024-01-01')
        return symbols, vol_start
    except Exception as e:
        logger.warning(f"读取配置失败: {e}. 使用默认符号列表与'2024-01-01'")
        # 默认14个品种
        default_syms = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']
        return default_syms, '2024-01-01'


def main():
    config_path = ROOT / 'configs' / 'business.yaml'
    symbols, vol_start = load_symbols_from_yaml(config_path)

    logger.info(f"符号数量: {len(symbols)} | 波动率起始: {vol_start}")

    # 使用库的加载函数，返回列名为 {symbol}_{col}
    data = load_data(symbols, start_date='2020-01-01', columns=['close'], log_price=True)
    if data.index.name is None:
        data.index.name = 'date'

    # 与库的结果
    analyzer = CointegrationAnalyzer(data)
    lib_df = analyzer.screen_all_pairs(p_threshold=1.0)  # 获取全部，便于逐对比

    # 文档版结果
    doc_df = screen_all_pairs_doc(data, volatility_start=vol_start)

    # 对齐与比较（按pair join）
    on = 'pair'
    merged = pd.merge(lib_df, doc_df, on=on, suffixes=('_lib', '_doc'), how='inner')
    if len(merged) == 0:
        logger.warning("没有可比较的配对（pair键不匹配）")

    # 差异度量
    def safe_abs_diff(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float(abs(a - b))

    for w in ['1y', '5y']:
        merged[f'diff_pvalue_{w}'] = [safe_abs_diff(a, b) for a, b in zip(merged[f'pvalue_{w}_lib'], merged[f'pvalue_{w}_doc'])]
        merged[f'diff_beta_{w}'] = [safe_abs_diff(a, b) for a, b in zip(merged[f'beta_{w}_lib'], merged[f'beta_{w}_doc'])]
        # 半衰期允许NaN（不可估计）
        merged[f'diff_halflife_{w}'] = [safe_abs_diff(a, b) for a, b in zip(merged.get(f'halflife_{w}_lib', np.nan), merged.get(f'halflife_{w}_doc', np.nan))]

    # 方向一致性
    merged['direction_match'] = (merged['direction_lib'] == merged['direction_doc'])

    # 统计
    def ratio_ok(series: pd.Series, tol: float) -> float:
        s = series.dropna()
        if len(s) == 0:
            return 0.0
        return float((s <= tol).mean())

    p_tol = 1e-6
    b_tol = 1e-6
    hl_tol = 0.5  # days

    stats_lines = []
    for w in ['1y', '5y']:
        stats_lines.append(f"pvalue_{w} 绝对误差≤{p_tol}: {ratio_ok(merged[f'diff_pvalue_{w}'], p_tol):.2%}")
        stats_lines.append(f"beta_{w}   绝对误差≤{b_tol}: {ratio_ok(merged[f'diff_beta_{w}'], b_tol):.2%}")
        stats_lines.append(f"halflife_{w} 误差≤{hl_tol}天: {ratio_ok(merged[f'diff_halflife_{w}'], hl_tol):.2%}")
    stats_lines.append(f"方向一致: {float(merged['direction_match'].mean() if len(merged)>0 else 0):.2%}")

    # 同时满足筛选：5y与1y都 <0.05 的通过率对比
    def pass_mask(df: pd.DataFrame) -> pd.Series:
        return (df['pvalue_5y'] < 0.05) & (df['pvalue_1y'] < 0.05)

    lib_pass_rate = float(pass_mask(lib_df).mean() if len(lib_df) else 0)
    doc_pass_rate = float(pass_mask(doc_df).mean() if len(doc_df) else 0)

    # 生成报告
    out_dir = ROOT / 'output' / 'cointegration'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = out_dir / f'cointegration_verification_{ts}.md'

    lines: List[str] = []
    lines.append(f"# 协整算法验证报告\n")
    lines.append(f"- 生成时间: {ts}")
    lines.append(f"- ADF参数: regression='n', autolag='AIC', maxlag=T**(1/3)")
    lines.append(f"- 波动率起始: {vol_start}")
    lines.append(f"- 配对数量(可比较): {len(merged)}/{len(lib_df)}\n")

    lines.append("## 误差统计（文档算法 vs 现有实现）")
    for s in stats_lines:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## 同时满足筛选（5y<0.05 且 1y<0.05）通过率")
    lines.append(f"- 现有实现: {lib_pass_rate:.2%}")
    lines.append(f"- 文档算法: {doc_pass_rate:.2%}\n")

    # 列出差异最大的前N项
    def top_diff(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
        s = df[[on, col]].dropna().sort_values(col, ascending=False).head(n)
        return s

    for met in ['diff_pvalue_5y', 'diff_pvalue_1y', 'diff_beta_5y', 'diff_beta_1y', 'diff_halflife_5y', 'diff_halflife_1y']:
        top = top_diff(merged, met, 10)
        if len(top) > 0:
            lines.append(f"## Top差异: {met}")
            lines.append("| pair | diff |")
            lines.append("|---|---:|")
            for _, r in top.iterrows():
                lines.append(f"| {r[on]} | {r[met]:.6g} |")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding='utf-8')
    logger.info(f"验证报告已生成: {report_path}")


if __name__ == '__main__':
    main()






