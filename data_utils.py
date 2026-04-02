# data_utils.py — Shared data download, transform and HuggingFace I/O
# Now with ultra‑robust downloading: persistent session, long delays, multiple Stooq suffixes.

import io
import json
import logging
import os
import random
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download, upload_file
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------------------------------------
# Persistent session with aggressive retries
# ------------------------------------------------------------
def _get_yf_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

_YF_SESSION = _get_yf_session()

# ------------------------------------------------------------
# Helper: fetch one ticker with retries and Stooq fallback
# ------------------------------------------------------------
def _fetch_one_ticker_robust(ticker: str, start: str, end: str, data_type: str = "ohlcv",
                              max_retries: int = 5, base_delay: float = 10.0):
    """
    Fetch data for a single ticker.
    data_type: 'ohlcv' -> returns OHLCV, 'close' -> returns only close.
    Returns DataFrame or None if both sources fail.
    """
    for attempt in range(max_retries + 1):
        try:
            time.sleep(random.uniform(1, 3))
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df.empty:
                raise ValueError("Empty data from yfinance")

            # Flatten MultiIndex if present (yfinance sometimes returns MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            if data_type == "ohlcv":
                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep_cols].copy()
            else:
                if "Close" not in df.columns:
                    raise ValueError("No Close column")
                df = df[["Close"]].copy()
                df.columns = [ticker]

            return df

        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"yfinance failed for {ticker} after {max_retries} retries: {e}")
                break
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 5)
            logger.info(f"Retry {attempt+1}/{max_retries} for {ticker} after {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # Stooq fallback
    suffixes = ['', '.US', '-US', '_U', '.U', '-U', ' US']
    tried = set()
    for suffix in suffixes:
        stooq_ticker = f"{ticker}{suffix}"
        if stooq_ticker in tried:
            continue
        tried.add(stooq_ticker)
        try:
            logger.info(f"Falling back to Stooq for {stooq_ticker}...")
            from pandas_datareader import DataReader
            df = DataReader(stooq_ticker, 'stooq', start, end)
            if df.empty:
                short_start = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
                df = DataReader(stooq_ticker, 'stooq', short_start, end)
            if df.empty:
                raise ValueError("Empty data from Stooq")

            if data_type == "ohlcv":
                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep_cols].copy()
            else:
                if "Close" not in df.columns:
                    raise ValueError("No Close column")
                df = df[["Close"]].copy()
                df.columns = [ticker]

            return df
        except Exception as e:
            logger.warning(f"Stooq {stooq_ticker} failed: {e}")
            continue

    logger.error(f"All fallbacks failed for {ticker}")
    return None


# ------------------------------------------------------------
# ETF OHLCV download (individual with fallback)
# ------------------------------------------------------------
def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for all tickers via yfinance (individual downloads).
    Returns DataFrame with MultiIndex columns (ticker, field).
    """
    end = end or date.today().strftime("%Y-%m-%d")

    # Chunk-based batch download (used by update_daily.py directly)
    # Try batch first for efficiency, fall back to individual
    chunk_size = getattr(cfg, 'DOWNLOAD_CHUNK_SIZE', 3)
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    max_retries = getattr(cfg, 'DOWNLOAD_MAX_RETRIES', 5)

    logger.info(f"Downloading OHLCV: {len(tickers)} tickers {start}→{end}")
    logger.info(f"Chunk size: {chunk_size}, max retries: {max_retries}")

    frames = []
    successful = []

    for chunk in chunks:
        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(random.uniform(1, 3))
                raw = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by='ticker',
                )
                if raw is None or raw.empty:
                    raise ValueError("Empty response")

                # Parse MultiIndex result
                if isinstance(raw.columns, pd.MultiIndex):
                    # Standard yfinance MultiIndex: (field, ticker) or (ticker, field)
                    # Determine order by checking if first level values are tickers
                    lvl0 = raw.columns.get_level_values(0).unique().tolist()
                    lvl1 = raw.columns.get_level_values(1).unique().tolist()
                    if any(t in lvl0 for t in chunk):
                        # (ticker, field) order
                        for ticker in chunk:
                            if ticker in lvl0:
                                tk_df = raw[ticker].copy()
                                tk_df.columns = pd.MultiIndex.from_tuples(
                                    [(ticker, col) for col in tk_df.columns]
                                )
                                frames.append(tk_df)
                                successful.append(ticker)
                    else:
                        # (field, ticker) order
                        for ticker in chunk:
                            if ticker in lvl1:
                                tk_cols = [(f, t) for f, t in raw.columns if t == ticker]
                                tk_df = raw[[c for c in raw.columns if c[1] == ticker]].copy()
                                tk_df.columns = pd.MultiIndex.from_tuples(
                                    [(ticker, f) for f, t in tk_df.columns]
                                )
                                frames.append(tk_df)
                                successful.append(ticker)
                else:
                    # Single ticker returned flat columns
                    ticker = chunk[0]
                    tk_df = raw.copy()
                    tk_df.columns = pd.MultiIndex.from_tuples(
                        [(ticker, col) for col in tk_df.columns]
                    )
                    frames.append(tk_df)
                    successful.append(ticker)

                logger.info(f"  ✓ {chunk} downloaded (attempt {attempt})")
                break

            except Exception as e:
                if attempt == max_retries:
                    logger.warning(f"  ✗ {chunk} failed after {max_retries} attempts: {e}")
                    # Fall back to individual downloads for this chunk
                    for ticker in chunk:
                        df = _fetch_one_ticker_robust(ticker, start, end, data_type="ohlcv", max_retries=3)
                        if df is not None:
                            df.columns = pd.MultiIndex.from_tuples(
                                [(ticker, col) for col in df.columns]
                            )
                            frames.append(df)
                            successful.append(ticker)
                else:
                    wait = 2 ** attempt + random.uniform(0, 2)
                    time.sleep(wait)

    logger.info(f"yfinance success: {len(successful)}/{len(tickers)} tickers")

    if not frames:
        raise ValueError("No OHLCV data fetched for any ticker.")

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    logger.info(f"OHLCV download complete. Shape: {combined.shape}")
    return combined


# ------------------------------------------------------------
# FIX 1: flatten_ohlcv — robust to both flat and MultiIndex columns
# ------------------------------------------------------------
def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex OHLCV DataFrame to single-level columns like TLT_Close.

    Handles three cases:
      1. MultiIndex columns (ticker, field)  → "ticker_field"
      2. Already-flat columns like "TLT_Close" → returned as-is
      3. Single-level field names (Open/Close/…) → should not occur here
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # Use get_level_values to avoid unpacking issues with variable-depth tuples
        lvl0 = df.columns.get_level_values(0).tolist()
        lvl1 = df.columns.get_level_values(1).tolist()

        # Drop columns where either level is empty/NaN
        new_cols = []
        keep = []
        for i, (t, f) in enumerate(zip(lvl0, lvl1)):
            t = str(t).strip()
            f = str(f).strip()
            if t and f and t != 'nan' and f != 'nan':
                new_cols.append(f"{t}_{f}")
                keep.append(i)

        df = df.iloc[:, keep]
        df.columns = new_cols
    else:
        # Already flat — check if columns look like "TICKER_Field" already
        # If so, return as-is; otherwise something unexpected happened
        pass

    return df


# ------------------------------------------------------------
# Returns computation
# ------------------------------------------------------------
def compute_returns(ohlcv_flat: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Compute simple and log daily returns from flat OHLCV DataFrame."""
    rets = pd.DataFrame(index=ohlcv_flat.index)
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col not in ohlcv_flat.columns:
            logger.warning(f"Missing Close for {ticker}, skipping returns")
            continue
        close = ohlcv_flat[close_col]
        rets[f"{ticker}_ret"] = close.pct_change()
        rets[f"{ticker}_logret"] = np.log(close / close.shift(1))

    rets = rets.dropna(how="all")
    logger.info(f"Returns shape: {rets.shape}")
    return rets


# ------------------------------------------------------------
# FRED download
# ------------------------------------------------------------
def download_fred(start: str, end: str = None) -> pd.DataFrame:
    """Download all FRED macro series."""
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


# ------------------------------------------------------------
# Calendar helpers
# ------------------------------------------------------------
def get_trading_days(start: str, end: str = None) -> pd.DatetimeIndex:
    """Return NYSE trading days between start and end (inclusive)."""
    nyse = mcal.get_calendar("NYSE")
    end = end or date.today().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)


def last_trading_day() -> str:
    """Return the most recent completed NYSE trading day."""
    today = date.today()
    nyse = mcal.get_calendar("NYSE")
    end = today.strftime("%Y-%m-%d")
    start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    if len(days) == 0:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    latest = days[-1]
    if latest.date() >= today:
        latest = days[-2] if len(days) > 1 else days[-1]
    return latest.strftime("%Y-%m-%d")


# ------------------------------------------------------------
# Derived macro features
# ------------------------------------------------------------
def compute_macro_derived(macro: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered macro features from raw FRED series."""
    d = pd.DataFrame(index=macro.index)
    w = cfg.ZSCORE_WINDOW

    def zscore(s: pd.Series) -> pd.Series:
        mu = s.rolling(w, min_periods=w // 2).mean()
        sig = s.rolling(w, min_periods=w // 2).std()
        return (s - mu) / (sig + 1e-8)

    if "VIX" in macro.columns:
        d["VIX_zscore"] = zscore(macro["VIX"])
        d["VIX_log"] = np.log(macro["VIX"].clip(lower=0.01))
        d["VIX_chg1d"] = macro["VIX"].pct_change()

    if "T10Y2Y" in macro.columns:
        d["YC_slope"] = macro["T10Y2Y"]
        d["YC_slope_zscore"] = zscore(macro["T10Y2Y"])
        d["YC_slope_chg"] = macro["T10Y2Y"].diff()

    if "DGS10" in macro.columns:
        d["DGS10_zscore"] = zscore(macro["DGS10"])
        d["DGS10_chg"] = macro["DGS10"].diff()

    if "HY_SPREAD" in macro.columns:
        d["HY_spread_zscore"] = zscore(macro["HY_SPREAD"])
        d["HY_spread_chg"] = macro["HY_SPREAD"].diff()

    if "IG_SPREAD" in macro.columns:
        d["IG_spread_zscore"] = zscore(macro["IG_SPREAD"])

    if "HY_SPREAD" in macro.columns and "IG_SPREAD" in macro.columns:
        d["HY_IG_ratio"] = macro["HY_SPREAD"] / (macro["IG_SPREAD"] + 1e-8)
        d["HY_IG_ratio_zscore"] = zscore(d["HY_IG_ratio"])
        d["credit_stress"] = (
            zscore(macro["HY_SPREAD"]) + zscore(macro["IG_SPREAD"])
        ) / 2.0

    if "USD_INDEX" in macro.columns:
        d["USD_zscore"] = zscore(macro["USD_INDEX"])
        d["USD_chg"] = macro["USD_INDEX"].pct_change()

    if "WTI_OIL" in macro.columns:
        d["OIL_zscore"] = zscore(macro["WTI_OIL"])
        d["OIL_chg"] = macro["WTI_OIL"].pct_change()
        d["OIL_log"] = np.log(macro["WTI_OIL"].clip(lower=0.01))

    if "DTB3" in macro.columns:
        d["TBILL_daily"] = macro["DTB3"] / 252.0 / 100.0

    if all(c in macro.columns for c in ["VIX", "HY_SPREAD", "T10Y2Y"]):
        vix_z = zscore(macro["VIX"])
        hy_z = zscore(macro["HY_SPREAD"])
        yc_z = -zscore(macro["T10Y2Y"])
        d["macro_stress_composite"] = (vix_z + hy_z + yc_z) / 3.0

    d = d.dropna(how="all")
    logger.info(f"Derived macro shape: {d.shape}, cols: {list(d.columns)}")
    return d


# ------------------------------------------------------------
# FIX 2: build_master — use outer union then forward-fill, not inner join
# ------------------------------------------------------------
def build_master(
    ohlcv_flat: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    macro_derived: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align all DataFrames on the OHLCV trading-day index.

    Strategy: use ohlcv_flat's index as the master spine (it is always the
    most complete), then reindex all other frames to that spine and forward-
    fill short gaps (≤5 days) from FRED publication lags.  This avoids the
    previous inner-join behaviour that reduced the dataset to 1 row on daily
    update runs where macro/returns had slightly different date ranges.
    """
    logger.info(
        f"build_master inputs: ohlcv={ohlcv_flat.shape}, returns={returns.shape}, "
        f"macro={macro.shape}, macro_derived={macro_derived.shape}"
    )

    if ohlcv_flat.empty:
        raise ValueError("OHLCV DataFrame is empty — cannot build master")

    # Use OHLCV index as the authoritative spine
    spine = ohlcv_flat.index.sort_values()

    # Handle empty returns gracefully
    if returns.empty:
        logger.warning("Returns DataFrame is empty — filling with zeros")
        returns = pd.DataFrame(index=spine)
        for col in [c for c in ohlcv_flat.columns if '_Close' in c]:
            ticker = col.replace('_Close', '')
            returns[f"{ticker}_ret"] = 0.0
            returns[f"{ticker}_logret"] = 0.0

    # Reindex everything to the OHLCV spine; ffill up to 5 days for FRED lags
    returns_aligned     = returns.reindex(spine).ffill(limit=1)   # returns: only 1-day fill
    macro_aligned       = macro.reindex(spine).ffill(limit=5)     # FRED: up to 5-day fill
    macro_derived_align = macro_derived.reindex(spine).ffill(limit=5)

    master = pd.concat(
        [
            ohlcv_flat.reindex(spine),
            returns_aligned,
            macro_aligned,
            macro_derived_align,
        ],
        axis=1,
    )
    master.index.name = "Date"

    if master.empty:
        raise ValueError("Master dataset is empty after concatenation")

    logger.info(
        f"Master shape: {master.shape}, "
        f"range: {master.index[0].date()} -> {master.index[-1].date()}"
    )
    return master


# ------------------------------------------------------------
# HuggingFace I/O
# ------------------------------------------------------------
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
