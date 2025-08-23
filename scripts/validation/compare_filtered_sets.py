#!/usr/bin/env python3
"""
比较“5年且1年同时满足(p<0.05)”筛选出的协整配对集合：
- 集合A：lib.coint.CointegrationAnalyzer.screen_all_pairs(p_threshold=0.05)
- 集合B：文档算法(screen_all_pairs_doc)计算后再手工应用(5y & 1y)过滤

输出：两集合是否一致、各自数量，以及差集列表。
不修改库代码，仅作只读验证。
"""

from pathlib import Path
import sys
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.data import load_data
from lib.coint import CointegrationAnalyzer

# 复用文档算法的实现（与 validate_coint_algorithms.py 保持一致）
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

WINDOWS_TRADING_DAYS = {'5y': 1260, '4y': 1008, '3y': 756, '2y': 504, '1y': 252}


def compute_volatility_from_log_prices(log_prices, dates, start_date=None):
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
    return float(np.std(returns) * (252 ** 0.5))


def determine_direction_doc(series_1, series_2, dates, symbol_1, symbol_2, volatility_start=None):
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


def eg_test_doc(x, y, direction='y_on_x', adf_regression='n', autolag='AIC', maxlag=None):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) != len(y) or len(x) < 20:
        raise ValueError('Invalid series length for EG test')
    if direction == 'y_on_x':
        X = add_constant(x)
        model = OLS(y, X).fit()
        beta = float(model.params[1])
        resid = np.asarray(model.resid)
    else:
        Y = add_constant(y)
        model = OLS(x, Y).fit()
        beta = float(model.params[1])
        resid = np.asarray(model.resid)
    if maxlag is None:
        maxlag = int(np.floor(len(resid) ** (1/3)))
    adf_stat, pvalue, *_ = adfuller(resid, regression=adf_regression, autolag=autolag, maxlag=maxlag)
    return {'pvalue': float(pvalue), 'beta': round(beta, 6), 'residuals': resid}


def multi_window_doc(x, y, direction):
    results = {}
    n = len(x)
    for wname, wsize in WINDOWS_TRADING_DAYS.items():
        if n >= wsize:
            subx = x[-wsize:]
            suby = y[-wsize:]
            try:
                eg = eg_test_doc(subx, suby, direction=direction)
                results[wname] = eg
            except Exception:
                results[wname] = None
        else:
            results[wname] = None
    return results


def screen_all_pairs_doc(data: pd.DataFrame, volatility_start=None) -> pd.DataFrame:
    symbols = list(data.columns)
    out_rows = []
    for s1, s2 in combinations(symbols, 2):
        x_series = data[s1].values
        y_series = data[s2].values
        dates = data.index
        direction, symbol_x, symbol_y = determine_direction_doc(data[s1].values, data[s2].values, dates, s1, s2, volatility_start)
        if symbol_x == s1:
            x_final, y_final = x_series, y_series
        else:
            x_final, y_final = y_series, x_series
        mw = multi_window_doc(x_final, y_final, direction)
        row = {'pair': f'{symbol_x}-{symbol_y}', 'symbol_x': symbol_x, 'symbol_y': symbol_y, 'direction': direction}
        for w in WINDOWS_TRADING_DAYS:
            if mw[w] is not None:
                row[f'pvalue_{w}'] = mw[w]['pvalue']
                row[f'beta_{w}'] = mw[w]['beta']
            else:
                row[f'pvalue_{w}'] = np.nan
                row[f'beta_{w}'] = np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def load_symbols_from_yaml(config_path: Path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        symbols = cfg.get('symbols', {}).get('all', [])
        vol_start = cfg.get('periods', {}).get('volatility_start', '2024-01-01')
        return symbols, vol_start
    except Exception:
        default_syms = ['AG0', 'AL0', 'AU0', 'CU0', 'HC0', 'I0', 'NI0', 'PB0', 'RB0', 'SF0', 'SM0', 'SN0', 'SS0', 'ZN0']
        return default_syms, '2024-01-01'


def main():
    config_path = ROOT / 'configs' / 'business.yaml'
    symbols, vol_start = load_symbols_from_yaml(config_path)
    data = load_data(symbols, start_date='2020-01-01', columns=['close'], log_price=True)
    analyzer = CointegrationAnalyzer(data)

    # 集合A：库内筛选（已按5年且1年同时满足，并按1年排序）
    lib_df = analyzer.screen_all_pairs(p_threshold=0.05)
    lib_set = set(lib_df['pair'].tolist()) if not lib_df.empty else set()

    # 集合B：文档算法 + 手工双阈值过滤
    doc_all = screen_all_pairs_doc(data, volatility_start=vol_start)
    if not doc_all.empty and {'pvalue_1y', 'pvalue_5y'}.issubset(set(doc_all.columns)):
        doc_filtered = doc_all[(doc_all['pvalue_1y'] < 0.05) & (doc_all['pvalue_5y'] < 0.05)].copy()
    else:
        doc_filtered = pd.DataFrame(columns=['pair'])
    doc_set = set(doc_filtered['pair'].tolist()) if not doc_filtered.empty else set()

    # 对比
    same = (lib_set == doc_set)
    only_in_lib = sorted(lib_set - doc_set)
    only_in_doc = sorted(doc_set - lib_set)

    print(f"集合一致: {same}")
    print(f"库集合数量: {len(lib_set)} | 文档集合数量: {len(doc_set)}")
    if only_in_lib:
        print("仅在库集合中的配对:")
        for p in only_in_lib:
            print(f"  - {p}")
    if only_in_doc:
        print("仅在文档集合中的配对:")
        for p in only_in_doc:
            print(f"  - {p}")


if __name__ == '__main__':
    main()


