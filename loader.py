# loader.py — Loads data from P2SAMAPA/p2-etf-deepm-data HuggingFace dataset
# Uses proven _fix_index pattern to handle Date column vs index.

import os
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config as cfg


def _download(filename: str) -> str:
    return hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
        force_download=True,
    )


def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a proper DatetimeIndex regardless of how parquet saved it."""
    for col in ["Date", "date", "DATE"]:
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    # Drop any residual index columns
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ("date", "index", "level_0"):
            df = df.drop(columns=[col])
    return df.sort_index()


def load_master() -> pd.DataFrame:
    """Load the full master aligned file."""
    path = _download(cfg.FILE_MASTER)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] Master: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_etf_ohlcv() -> pd.DataFrame:
    path = _download(cfg.FILE_ETF_OHLCV)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] OHLCV: {df.shape}")
    return df


def load_etf_returns() -> pd.DataFrame:
    path = _download(cfg.FILE_ETF_RETURNS)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] Returns: {df.shape}")
    return df


def load_macro_fred() -> pd.DataFrame:
    path = _download(cfg.FILE_MACRO_FRED)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] FRED macro: {df.shape}")
    return df


def load_macro_derived() -> pd.DataFrame:
    path = _download(cfg.FILE_MACRO_DERIVED)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] Derived macro: {df.shape}")
    return df


def get_option_data(option: str, master: pd.DataFrame) -> dict:
    """
    Extract Option A (FI) or Option B (Equity) data from master DataFrame.

    Returns dict with:
        prices      — Close prices for universe ETFs
        returns     — Simple returns
        log_returns — Log returns
        vol         — Annualised volatility
        macro       — Raw FRED macro features
        macro_derived — Engineered macro features
        benchmark_ret — Benchmark simple returns
        tickers     — List of ETF tickers for this option
        benchmark   — Benchmark ticker string
        cash_rate   — Daily T-bill rate series
    """
    if option == "A":
        tickers   = cfg.FI_ETFS
        benchmark = cfg.FI_BENCHMARK
    elif option == "B":
        tickers   = cfg.EQ_ETFS
        benchmark = cfg.EQ_BENCHMARK
    else:
        raise ValueError(f"option must be 'A' or 'B', got {option!r}")

    # Close prices
    price_cols = [f"{t}_Close" for t in tickers if f"{t}_Close" in master.columns]
    prices = master[price_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    # Returns
    ret_cols = [f"{t}_ret" for t in tickers if f"{t}_ret" in master.columns]
    returns = master[ret_cols].copy()
    returns.columns = [c.replace("_ret", "") for c in returns.columns]

    logret_cols = [f"{t}_logret" for t in tickers if f"{t}_logret" in master.columns]
    log_returns = master[logret_cols].copy()
    log_returns.columns = [c.replace("_logret", "") for c in log_returns.columns]

    vol_cols = [f"{t}_vol" for t in tickers if f"{t}_vol" in master.columns]
    if vol_cols:
        vol = master[vol_cols].copy()
        vol.columns = [c.replace("_vol", "") for c in vol.columns]
    else:
        # Compute from log returns if not in master
        vol = log_returns.rolling(cfg.VOL_WINDOW).std() * np.sqrt(252)

    # Macro
    macro_cols = [c for c in cfg.FRED_SERIES.keys() if c in master.columns]
    macro = master[macro_cols].copy()

    derived_cols = [c for c in master.columns if any(
        c.startswith(p) for p in [
            "VIX_", "YC_", "DGS10_", "HY_", "IG_", "HY_IG",
            "credit_", "USD_", "OIL_", "TBILL_", "macro_stress"
        ]
    )]
    macro_derived = master[derived_cols].copy()

    # Benchmark
    bench_ret_col = f"{benchmark}_ret"
    if bench_ret_col in master.columns:
        benchmark_ret = master[bench_ret_col].rename(benchmark)
    else:
        bench_close = master[f"{benchmark}_Close"]
        benchmark_ret = bench_close.pct_change().rename(benchmark)

    # Cash rate
    cash_rate = master["TBILL_daily"] if "TBILL_daily" in master.columns else \
                (master["DTB3"] / 252 / 100 if "DTB3" in master.columns else
                 pd.Series(0.0, index=master.index, name="TBILL_daily"))

    # Drop rows with all NaN prices
    valid_idx = prices.dropna(how="all").index
    prices        = prices.loc[valid_idx]
    returns       = returns.reindex(valid_idx)
    log_returns   = log_returns.reindex(valid_idx)
    vol           = vol.reindex(valid_idx)
    macro         = macro.reindex(valid_idx).ffill()
    macro_derived = macro_derived.reindex(valid_idx).ffill()
    benchmark_ret = benchmark_ret.reindex(valid_idx)
    cash_rate     = cash_rate.reindex(valid_idx).ffill().fillna(0.0)

    print(f"[loader] Option {option} ({len(tickers)} ETFs): "
          f"{len(prices)} days, {prices.index[0].date()} -> {prices.index[-1].date()}")

    return {
        "option":        option,
        "tickers":       tickers,
        "benchmark":     benchmark,
        "prices":        prices,
        "returns":       returns,
        "log_returns":   log_returns,
        "vol":           vol,
        "macro":         macro,
        "macro_derived": macro_derived,
        "benchmark_ret": benchmark_ret,
        "cash_rate":     cash_rate,
    }


def split_train_live(data: dict) -> tuple:
    """
    Split all series in data dict into train (up to TRAIN_END)
    and live (from LIVE_START) subsets.
    Returns (train_data, live_data) — same structure as input dict.
    """
    def _split(df_or_series):
        if isinstance(df_or_series, pd.DataFrame):
            train = df_or_series[df_or_series.index <= cfg.TRAIN_END]
            live  = df_or_series[df_or_series.index >= cfg.LIVE_START]
        else:
            train = df_or_series[df_or_series.index <= cfg.TRAIN_END]
            live  = df_or_series[df_or_series.index >= cfg.LIVE_START]
        return train, live

    keys = ["prices", "returns", "log_returns", "vol",
            "macro", "macro_derived", "benchmark_ret", "cash_rate"]

    train_data = {k: v for k, v in data.items() if k not in keys}
    live_data  = {k: v for k, v in data.items() if k not in keys}

    for k in keys:
        if k in data:
            t, l = _split(data[k])
            train_data[k] = t
            live_data[k]  = l

    return train_data, live_data
