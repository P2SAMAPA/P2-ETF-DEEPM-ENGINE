# data_utils.py — Updated with robust rate-limit handling
import io
import json
import logging
import os
import random
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------------------------------------
# Session factory — rotate user agents to avoid detection
# ------------------------------------------------------------
def _create_session() -> requests.Session:
    """Create a session with browser-like headers to avoid bot detection."""
    session = requests.Session()
    
    # Rotate user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ]
    
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    
    # Mount retry adapter for connection errors (not for yfinance logic errors)
    adapter = HTTPAdapter(max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    ))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

# ------------------------------------------------------------
# ETF OHLCV download — sequential with aggressive delays
# ------------------------------------------------------------
def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV with aggressive rate-limit handling.
    Strategy: Try batch first with long delays, fallback to sequential if needed.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    yf_tickers = [t for t in tickers if t != "CASH"]
    
    logger.info(f"Downloading OHLCV: {len(yf_tickers)} tickers {start}→{end}")
    
    # Strategy 1: Try batch download with aggressive backoff (but threads=False to avoid DB locks)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Clear yfinance cache to avoid DB locks
            yf.shared._DFS = {}
            yf.shared._ERRORS = {}
            
            # Use session with custom headers
            session = _create_session()
            
            # threads=False is crucial - prevents SQLite concurrency issues
            raw = yf.download(
                yf_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,  # CRITICAL: Prevents 'database is locked' errors
                prepost=False,  # Reduce data size/request time
                session=session,
            )
            
            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty DataFrame")
            
            combined = _normalize_columns(raw, yf_tickers)
            logger.info(f"Batch download success: {len(yf_tickers)} tickers")
            return combined
            
        except Exception as e:
            error_str = str(e)
            if "Rate limited" in error_str or "Too Many Requests" in error_str:
                # Aggressive backoff: 60s, 120s, 180s + jitter
                wait = 60 * attempt + random.uniform(10, 30)
                logger.warning(f"Rate limited on attempt {attempt}/{max_retries}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.warning(f"Batch attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    wait = 30 * attempt + random.uniform(5, 15)
                    time.sleep(wait)
    
    # Strategy 2: Fallback to sequential single-ticker downloads with delays
    logger.warning("Batch failed, falling back to sequential single-ticker downloads...")
    return _download_sequential(yf_tickers, start, end)

def _download_sequential(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download tickers one by one with generous delays between each.
    This is slower but respects rate limits better than batch from cloud IPs.
    """
    all_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Clear cache before each ticker to prevent DB locks
                yf.shared._DFS = {}
                yf.shared._ERRORS = {}
                
                session = _create_session()
                
                # Download single ticker
                ticker_obj = yf.Ticker(ticker, session=session)
                hist = ticker_obj.history(
                    start=start, 
                    end=end, 
                    auto_adjust=True, 
                    prepost=False
                )
                
                if hist is not None and not hist.empty:
                    all_data[ticker] = hist
                    logger.info(f"[{i+1}/{len(tickers)}] {ticker}: {len(hist)} rows")
                    break
                else:
                    raise ValueError("Empty data")
                    
            except Exception as e:
                if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                    wait = 60 + random.uniform(10, 30)  # 60-90s delay on rate limit
                    logger.warning(f"{ticker} rate limited, waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"{ticker} attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
        
        if ticker not in all_data:
            failed_tickers.append(ticker)
            logger.error(f"Failed to download {ticker} after all retries")
        
        # Delay between tickers: 2-5 seconds (randomized to look less robotic)
        if i < len(tickers) - 1:
            delay = random.uniform(2, 5)
            time.sleep(delay)
    
    if not all_data:
        raise ValueError("All tickers failed to download")
    
    # Combine all single-ticker DataFrames into MultiIndex format
    combined = pd.concat(all_data, axis=1)
    
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns.names = ["Ticker", "Field"]
    else:
        # Single column level - create MultiIndex
        cols = [(ticker, col) for ticker, df in all_data.items() for col in df.columns]
        combined.columns = pd.MultiIndex.from_tuples(cols, names=["Ticker", "Field"])
    
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    
    if failed_tickers:
        logger.warning(f"Missing tickers: {failed_tickers}")
    
    logger.info(f"Sequential download complete. Shape: {combined.shape}")
    return combined

def _normalize_columns(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Normalize yfinance output to consistent MultiIndex (Ticker, Field)."""
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        
        if any(t in lvl0 for t in tickers):
            # Already (ticker, field) order
            combined = raw.copy()
            combined.columns.names = ["Ticker", "Field"]
        else:
            # (field, ticker) order — swap
            combined = raw.swaplevel(axis=1).sort_index(axis=1)
            combined.columns.names = ["Ticker", "Field"]
    else:
        # Single-ticker fallback
        ticker = tickers[0]
        combined = raw.copy()
        combined.columns = pd.MultiIndex.from_tuples(
            [(ticker, col) for col in combined.columns],
            names=["Ticker", "Field"],
        )
    
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    return combined

# ------------------------------------------------------------
# flatten_ohlcv — unchanged
# ------------------------------------------------------------
def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex OHLCV DataFrame to single-level columns like TLT_Close."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).tolist()
        lvl1 = df.columns.get_level_values(1).tolist()

        new_cols = []
        keep = []
        for i, (t, f) in enumerate(zip(lvl0, lvl1)):
            t = str(t).strip()
            f = str(f).strip()
            if t and f and t != "nan" and f != "nan":
                new_cols.append(f"{t}_{f}")
                keep.append(i)

        df = df.iloc[:, keep]
        df.columns = new_cols

    return df

# ------------------------------------------------------------
# Returns computation — unchanged
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
# FRED download — unchanged
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
# Calendar helpers — unchanged
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
# Derived macro features — unchanged
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
# build_master — unchanged
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
# HuggingFace I/O — unchanged
# ------------------------------------------------------------
def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
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
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def load_metadata() -> dict:
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
