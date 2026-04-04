# data_utils.py — Updated with HURST-repo style sequential download + Stooq fallback
import io
import json
import logging
import os
import random
import time
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# ------------------------------------------------------------
# OHLCV download — Sequential single-ticker with exponential backoff
# ------------------------------------------------------------

def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV using HURST-style sequential approach with exponential backoff.
    This avoids YF rate limits better than batch downloads from cloud IPs.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    yf_tickers = [t for t in tickers if t != "CASH"]
    
    logger.info(f"Downloading OHLCV (sequential): {len(yf_tickers)} tickers {start}→{end}")
    
    frames = []
    failed = []
    
    for i, ticker in enumerate(yf_tickers):
        logger.info(f"[{i+1}/{len(yf_tickers)}] Fetching {ticker}...")
        
        df = _fetch_yf_single(ticker, start, end)
        
        if df is None:
            logger.warning(f"🔄 YF failed for {ticker}, trying Stooq fallback...")
            df = _fetch_stooq_single(ticker, start, end)
        
        if df is not None:
            frames.append(df)
        else:
            failed.append(ticker)
            logger.error(f"❌ All sources failed for {ticker}")
        
        if i < len(yf_tickers) - 1:
            delay = random.uniform(1.0, 2.5)
            time.sleep(delay)
    
    if not frames:
        raise ValueError("No data fetched from any source for any ticker.")
    
    if failed:
        logger.warning(f"⚠️ Failed tickers: {failed} — continuing with {len(frames)} tickers.")
    
    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.ffill()
    
    logger.info(f"OHLCV download complete. Shape: {combined.shape}")
    return combined


def _fetch_yf_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch single ticker from Yahoo Finance with exponential backoff."""
    for attempt in range(6):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            
            if raw is None or raw.empty:
                raise ValueError(f"Empty response for {ticker}")
            
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]
            
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            if not available:
                raise ValueError(f"No OHLCV columns found for {ticker}")
            
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns],
                names=["Ticker", "Field"]
            )
            
            logger.info(f"✅ {ticker} (YF): {len(df)} rows")
            return df
            
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["rate limit", "too many", "429", "ratelimit"])
            
            if is_rate_limit and attempt < 5:
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                logger.warning(f"⚠️ YF rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"❌ YF failed for {ticker} after {attempt+1} attempts: {e}")
                return None
    
    return None


def _fetch_stooq_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch single ticker from Stooq as fallback (no API key required)."""
    stooq_symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    
    for attempt in range(3):
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            
            if raw.empty:
                raise ValueError(f"Empty Stooq response for {ticker}")
            
            raw = raw.sort_index()
            mask = (raw.index >= start) & (raw.index <= end)
            raw = raw.loc[mask]
            
            if raw.empty:
                raise ValueError(f"No data in range for {ticker} from Stooq")
            
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns],
                names=["Ticker", "Field"]
            )
            
            logger.info(f"✅ {ticker} (Stooq): {len(df)} rows")
            return df
            
        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                logger.warning(f"⚠️ Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    
    return None


# ------------------------------------------------------------
# flatten_ohlcv
# ------------------------------------------------------------

def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex OHLCV DataFrame to single-level columns like TLT_Close.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).tolist()
        lvl1 = df.columns.get_level_values(1).tolist()

        new_cols = []
        keep = []
        for i, (t, f) in enumerate(zip(lvl0, lvl1)):
            t = str(t).strip()
            f = str(f).strip()
            if t and f and t.lower() != "nan" and f.lower() != "nan":
                new_cols.append(f"{t}_{f}")
                keep.append(i)

        df = df.iloc[:, keep]
        df.columns = new_cols

    return df


# ------------------------------------------------------------
# Returns computation
# ------------------------------------------------------------

def compute_returns(ohlcv_flat: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Compute simple and log daily returns from flat OHLCV DataFrame."""
    rets = pd.DataFrame(index=ohlcv_flat.index)
    for ticker in tickers:
        if ticker == "CASH":
            continue
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
# Derived macro features — FIX: preserve DatetimeIndex
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

    # FIX: drop rows where ALL values are NaN only — never reset the index
    d = d.dropna(how="all")

    # FIX: ensure index is a proper named DatetimeIndex
    d.index = pd.to_datetime(d.index)
    d.index.name = "Date"

    logger.info(f"Derived macro shape: {d.shape}, cols: {list(d.columns)}")
    logger.info(f"Derived macro date range: {d.index[0].date()} → {d.index[-1].date()}")
    return d


# ------------------------------------------------------------
# build_master
# ------------------------------------------------------------

def build_master(
    ohlcv_flat: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    macro_derived: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align all DataFrames on the OHLCV trading-day index (authoritative spine).
    """
    logger.info(
        f"build_master inputs: ohlcv={ohlcv_flat.shape}, returns={returns.shape}, "
        f"macro={macro.shape}, macro_derived={macro_derived.shape}"
    )

    if ohlcv_flat.empty:
        raise ValueError("OHLCV DataFrame is empty — cannot build master")

    spine = ohlcv_flat.index.sort_values()

    if returns.empty:
        logger.warning("Returns DataFrame is empty — filling with zeros")
        returns = pd.DataFrame(index=spine)
        for col in [c for c in ohlcv_flat.columns if "_Close" in c]:
            ticker = col.replace("_Close", "")
            returns[f"{ticker}_ret"] = 0.0
            returns[f"{ticker}_logret"] = 0.0

    returns_aligned = returns.reindex(spine).ffill(limit=1)
    macro_aligned = macro.reindex(spine).ffill(limit=5)
    macro_derived_aligned = macro_derived.reindex(spine).ffill(limit=5)

    master = pd.concat(
        [ohlcv_flat.reindex(spine), returns_aligned, macro_aligned, macro_derived_aligned],
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
# HuggingFace I/O — FIX: ensure index name preserved on save
# ------------------------------------------------------------

def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    # FIX: ensure index always has a name so it survives parquet round-trip
    if df.index.name is None:
        df = df.copy()
        df.index.name = "Date"
    df.to_parquet(buf, index=True, engine="pyarrow")
    return buf.getvalue()


def upload_parquet(df: pd.DataFrame, hf_path: str, commit_msg: str) -> None:
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
    local = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=hf_path,
        repo_type="dataset",
        token=cfg.HF_TOKEN,
        force_download=True,
    )
    df = pd.read_parquet(local)
    # Handle date stored as column or index
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    return df
