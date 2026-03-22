# data_utils.py — Shared data download, transform and HuggingFace I/O
# Used by both seed.py and update_daily.py

import io
import json
import logging
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download, upload_file

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ── Market calendar ────────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str = None) -> pd.DatetimeIndex:
    """Return NYSE trading days between start and end (inclusive)."""
    nyse = mcal.get_calendar("NYSE")
    end  = end or date.today().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)


def last_trading_day() -> str:
    """Return the most recent completed NYSE trading day."""
    today = date.today()
    nyse  = mcal.get_calendar("NYSE")
    end   = today.strftime("%Y-%m-%d")
    start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    if len(days) == 0:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    latest = days[-1]
    if latest.date() >= today:
        latest = days[-2] if len(days) > 1 else days[-1]
    return latest.strftime("%Y-%m-%d")


# ── ETF OHLCV download ─────────────────────────────────────────────────────────

def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for all tickers via yfinance.
    Returns DataFrame with MultiIndex columns (ticker, field).
    Fields: Open, High, Low, Close, Volume
    Index: DatetimeIndex (tz-naive)
    """
    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Downloading OHLCV: {tickers} from {start} to {end}")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {tickers}")

    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.copy()
    else:
        ticker = tickers[0] if len(tickers) == 1 else tickers[0]
        df = pd.concat({ticker: raw}, axis=1)
        df.columns = pd.MultiIndex.from_tuples(
            [(t, f) for t, f in df.columns], names=["Ticker", "Field"]
        )

    df.columns.names = ["Ticker", "Field"]
    df = df.sort_index()
    logger.info(f"OHLCV shape: {df.shape}, range: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex OHLCV DataFrame to single-level columns.
    Columns become: TLT_Close, TLT_Open, TLT_High, TLT_Low, TLT_Volume, etc.
    """
    df = df.copy()
    df.columns = [f"{ticker}_{field}" for ticker, field in df.columns]
    return df


# ── ETF returns ────────────────────────────────────────────────────────────────

def compute_returns(ohlcv_flat: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Compute simple and log daily returns from flat OHLCV DataFrame.
    Returns DataFrame with columns: TLT_ret, TLT_logret, LQD_ret, etc.
    """
    rets = pd.DataFrame(index=ohlcv_flat.index)
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col not in ohlcv_flat.columns:
            logger.warning(f"Missing Close for {ticker}, skipping returns")
            continue
        close = ohlcv_flat[close_col]
        rets[f"{ticker}_ret"]    = close.pct_change()
        rets[f"{ticker}_logret"] = np.log(close / close.shift(1))
    rets = rets.dropna(how="all")
    logger.info(f"Returns shape: {rets.shape}")
    return rets


# ── FRED macro download ────────────────────────────────────────────────────────

def download_fred(start: str, end: str = None) -> pd.DataFrame:
    """
    Download all FRED macro series defined in config.FRED_SERIES.
    Returns DataFrame indexed by date, columns = friendly names (VIX, T10Y2Y, etc.)
    Missing values forward-filled (FRED releases lag by 1 business day typically).
    """
    end = end or date.today().strftime("%Y-%m-%d")
    fred = Fred(api_key=cfg.FRED_API_KEY)

    frames = {}
    for name, series_id in cfg.FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
            logger.info(f"FRED {series_id} ({name}): {len(s)} observations")
        except Exception as e:
            logger.error(f"Failed to fetch FRED {series_id}: {e}")

    if not frames:
        raise ValueError("No FRED series downloaded successfully")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    trading_days = get_trading_days(start, end)
    df = df.reindex(trading_days)
    df = df.ffill()
    df = df.dropna(how="all")

    logger.info(f"Macro FRED shape: {df.shape}, range: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ── Derived macro features ─────────────────────────────────────────────────────

def compute_macro_derived(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered macro features from raw FRED series.
    All projects can use these pre-computed features directly.
    """
    d = pd.DataFrame(index=macro.index)

    w = cfg.ZSCORE_WINDOW

    def zscore(s: pd.Series) -> pd.Series:
        mu  = s.rolling(w, min_periods=w // 2).mean()
        sig = s.rolling(w, min_periods=w // 2).std()
        return (s - mu) / (sig + 1e-8)

    if "VIX" in macro.columns:
        d["VIX_zscore"]          = zscore(macro["VIX"])
        d["VIX_log"]             = np.log(macro["VIX"].clip(lower=0.01))
        d["VIX_chg1d"]           = macro["VIX"].pct_change()

    if "T10Y2Y" in macro.columns:
        d["YC_slope"]            = macro["T10Y2Y"]
        d["YC_slope_zscore"]     = zscore(macro["T10Y2Y"])
        d["YC_slope_chg"]        = macro["T10Y2Y"].diff()

    if "DGS10" in macro.columns:
        d["DGS10_zscore"]        = zscore(macro["DGS10"])
        d["DGS10_chg"]           = macro["DGS10"].diff()

    if "HY_SPREAD" in macro.columns:
        d["HY_spread_zscore"]    = zscore(macro["HY_SPREAD"])
        d["HY_spread_chg"]       = macro["HY_SPREAD"].diff()

    if "IG_SPREAD" in macro.columns:
        d["IG_spread_zscore"]    = zscore(macro["IG_SPREAD"])

    if "HY_SPREAD" in macro.columns and "IG_SPREAD" in macro.columns:
        d["HY_IG_ratio"]         = macro["HY_SPREAD"] / (macro["IG_SPREAD"] + 1e-8)
        d["HY_IG_ratio_zscore"]  = zscore(d["HY_IG_ratio"])
        d["credit_stress"]       = (
            zscore(macro["HY_SPREAD"]) + zscore(macro["IG_SPREAD"])
        ) / 2.0

    if "USD_INDEX" in macro.columns:
        d["USD_zscore"]          = zscore(macro["USD_INDEX"])
        d["USD_chg"]             = macro["USD_INDEX"].pct_change()

    if "WTI_OIL" in macro.columns:
        d["OIL_zscore"]          = zscore(macro["WTI_OIL"])
        d["OIL_chg"]             = macro["WTI_OIL"].pct_change()
        d["OIL_log"]             = np.log(macro["WTI_OIL"].clip(lower=0.01))

    if "DTB3" in macro.columns:
        d["TBILL_daily"]         = macro["DTB3"] / 252.0 / 100.0

    if all(c in macro.columns for c in ["VIX", "HY_SPREAD", "T10Y2Y"]):
        vix_z   = zscore(macro["VIX"])
        hy_z    = zscore(macro["HY_SPREAD"])
        yc_z    = -zscore(macro["T10Y2Y"])
        d["macro_stress_composite"] = (vix_z + hy_z + yc_z) / 3.0

    d = d.dropna(how="all")
    logger.info(f"Derived macro shape: {d.shape}, cols: {list(d.columns)}")
    return d


# ── Master aligned file ────────────────────────────────────────────────────────

def build_master(
    ohlcv_flat: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    macro_derived: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join all DataFrames on common trading days to produce master.parquet.
    No lookahead: macro is only forward-filled, never backward-filled.
    """
    common = (
        ohlcv_flat.index
        .intersection(returns.index)
        .intersection(macro.index)
        .intersection(macro_derived.index)
    )
    common = common.sort_values()

    master = pd.concat(
        [
            ohlcv_flat.reindex(common),
            returns.reindex(common),
            macro.reindex(common),
            macro_derived.reindex(common),
        ],
        axis=1,
    )
    master.index.name = "Date"
    logger.info(f"Master shape: {master.shape}, range: {master.index[0].date()} -> {master.index[-1].date()}")
    return master


# ── HuggingFace I/O ────────────────────────────────────────────────────────────

def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True, engine="pyarrow")
    return buf.getvalue()


def upload_parquet(df: pd.DataFrame, hf_path: str, commit_msg: str) -> None:
    """Upload a DataFrame as parquet to HuggingFace dataset repo."""
    api = HfApi(token=cfg.HF_TOKEN)
    data = _df_to_bytes(df)
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=hf_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
    )
    logger.info(f"Uploaded {hf_path} to {cfg.HF_DATASET_REPO}")


def upload_json(obj: dict, hf_path: str, commit_msg: str) -> None:
    """Upload a dict as JSON to HuggingFace dataset repo."""
    api = HfApi(token=cfg.HF_TOKEN)
    data = json.dumps(obj, indent=2, default=str).encode("utf-8")
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=hf_path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
    )
    logger.info(f"Uploaded {hf_path} to {cfg.HF_DATASET_REPO}")


def load_parquet(hf_path: str) -> pd.DataFrame:
    """Download and load a parquet file from HuggingFace dataset repo."""
    local = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=hf_path,
        repo_type="dataset",
        token=cfg.HF_TOKEN,
        force_download=True,
    )
    df = pd.read_parquet(local)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_metadata() -> dict:
    """Load metadata.json from HuggingFace dataset repo."""
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=cfg.FILE_METADATA,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            force_download=True,
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return {}
