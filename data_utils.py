# data_utils.py — Shared data download, transform and HuggingFace I/O
# FIX: Removed Alpha Vantage entirely, improved yfinance rate limit handling
#      with: (1) yfinance batch download with exponential backoff + smaller chunks
#            (2) direct HTTP Stooq fallback (bypasses pandas_datareader)

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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download, upload_file

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Rate limit tuning
MAX_BATCH_RETRIES = 5
BATCH_CHUNK_SIZE = 3  # Download 3 tickers at a time to avoid rate limits
REQUEST_DELAY = 2  # Seconds between requests

# ─────────────────────────────────────────────────────────────
# Shared requests session
# ─────────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session

_SESSION = _make_session()


# ─────────────────────────────────────────────────────────────
# PRIMARY: yfinance chunked download (smaller chunks = less rate limiting)
# ─────────────────────────────────────────────────────────────
def _yfinance_chunked(tickers: list, start: str, end: str,
                      max_attempts: int = MAX_BATCH_RETRIES) -> tuple:
    """
    Download tickers in small chunks with exponential backoff.
    Returns (data_dict, successful_list, failed_list)
    """
    all_data = {}
    successful = []
    failed = []
    
    for i in range(0, len(tickers), BATCH_CHUNK_SIZE):
        chunk = tickers[i:i+BATCH_CHUNK_SIZE]
        chunk_success = False
        
        for attempt in range(1, max_attempts + 1):
            try:
                delay = 30 * (2 ** (attempt - 1)) + random.uniform(0, 10)
                if attempt > 1:
                    logger.info(f"  Retry {attempt}/{max_attempts} for {chunk} after {delay:.0f}s...")
                    time.sleep(delay)
                
                data = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,  # Single thread = fewer requests
                )
                
                if data is not None and not data.empty:
                    # Handle both single and multi-ticker responses
                    if len(chunk) == 1:
                        # Single ticker returns simple DataFrame
                        data.columns = pd.MultiIndex.from_tuples(
                            [(chunk[0], c) for c in data.columns],
                            names=["Ticker", "Field"]
                        )
                        all_data[chunk[0]] = data
                    else:
                        # Multi-ticker returns MultiIndex
                        if isinstance(data.columns, pd.MultiIndex):
                            # Ensure (ticker, field) format
                            if data.columns.names[0] != "Ticker":
                                data.columns = data.columns.swaplevel(0, 1)
                            for ticker in chunk:
                                if ticker in data.columns.get_level_values(0):
                                    ticker_data = data[ticker].dropna(how="all")
                                    if not ticker_data.empty:
                                        all_data[ticker] = ticker_data
                    successful.extend(chunk)
                    chunk_success = True
                    logger.info(f"  ✓ {chunk} downloaded (attempt {attempt})")
                    break
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "too many" in error_msg:
                    logger.warning(f"  Rate limited for {chunk}, backing off...")
                else:
                    logger.debug(f"  Attempt {attempt} failed for {chunk}: {e}")
        
        if not chunk_success:
            failed.extend(chunk)
            logger.warning(f"  ✗ {chunk} failed after {max_attempts} attempts")
        
        # Pause between chunks to avoid rate limits
        time.sleep(REQUEST_DELAY * 2)
    
    return all_data, successful, failed


# ─────────────────────────────────────────────────────────────
# FALLBACK: direct HTTP to Stooq (ONLY fallback - no Alpha Vantage)
# ─────────────────────────────────────────────────────────────
def _stooq_direct(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV from Stooq via direct HTTP GET.
    pandas_datareader's Stooq reader is broken; this hits the CSV endpoint directly.
    """
    symbol = ticker.lower() + ".us"
    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={symbol}&d1={start.replace('-','')}&d2={end.replace('-','')}&i=d"
    )
    for attempt in range(3):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.text.strip()
            if not content or "No data" in content or len(content) < 50:
                return None
            df = pd.read_csv(io.StringIO(content), parse_dates=["Date"], index_col="Date")
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            # Stooq columns: Open, High, Low, Close, Volume
            df.columns = [c.capitalize() for c in df.columns]
            logger.info(f"  Stooq OK for {ticker}: {len(df)} rows")
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            logger.debug(f"Stooq failed for {ticker}: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# Per-ticker rescue (used when batch misses some tickers)
# ─────────────────────────────────────────────────────────────
def _rescue_tickers(missing: list, start: str, end: str) -> dict:
    """
    Try Stooq only (no Alpha Vantage) for each missing ticker.
    Returns {ticker: DataFrame(OHLCV)} for successes.
    """
    recovered = {}
    for idx, ticker in enumerate(missing):
        logger.info(f"Rescuing {ticker} via Stooq direct...")
        
        # Progressive delay between rescue attempts
        if idx > 0:
            time.sleep(REQUEST_DELAY)
        
        df = _stooq_direct(ticker, start, end)
        if df is not None and not df.empty:
            recovered[ticker] = df
            logger.info(f"  ✓ Recovered {ticker}: {len(df)} rows")
        else:
            logger.warning(f"  ✗ All sources failed for {ticker} — will be NaN")
    
    return recovered


# ─────────────────────────────────────────────────────────────
# Public API: download_ohlcv
# ─────────────────────────────────────────────────────────────
def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for all tickers with layered fallbacks.
    Returns MultiIndex DataFrame (ticker, field) — same contract as before.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    # CASH is synthetic (from FRED DTB3) — skip for market data download
    market_tickers = [t for t in tickers if t != "CASH"]
    logger.info(f"Downloading OHLCV: {len(market_tickers)} tickers {start}→{end}")
    logger.info(f"Chunk size: {BATCH_CHUNK_SIZE}, max retries: {MAX_BATCH_RETRIES}")

    # Step 1: yfinance chunked download (reduced rate limit impact)
    yf_data, yf_success, yf_failed = _yfinance_chunked(market_tickers, start, end)
    
    logger.info(f"yfinance success: {len(yf_success)}/{len(market_tickers)} tickers")
    if yf_failed:
        logger.info(f"yfinance failed: {yf_failed}")

    # Step 2: per-ticker Stooq rescue for anything still missing
    frames_multi = []
    
    # Add successful yfinance data
    for ticker, df in yf_data.items():
        frames_multi.append(df)
    
    # Rescue missing tickers
    if yf_failed:
        logger.info(f"Rescuing {len(yf_failed)} missing tickers...")
        recovered = _rescue_tickers(yf_failed, start, end)
        for ticker, df in recovered.items():
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep]
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, c) for c in df.columns], names=["Ticker", "Field"]
            )
            frames_multi.append(df)

    if not frames_multi:
        raise ValueError(
            "No OHLCV data fetched for any ticker. "
            "yfinance is rate-limited and Stooq fallback failed. "
            "Try: 1) Re-run later (off-peak hours), or "
            "2) Increase BATCH_CHUNK_SIZE and MAX_BATCH_RETRIES in data_utils.py"
        )

    combined = pd.concat(frames_multi, axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    combined.sort_index(axis=1, inplace=True)
    
    # Forward fill any missing values
    combined = combined.ffill()

    logger.info(f"OHLCV download complete. Shape: {combined.shape}")
    return combined


# ─────────────────────────────────────────────────────────────
# The rest of the file — UNCHANGED from original
# ─────────────────────────────────────────────────────────────

def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex OHLCV DataFrame to single-level columns."""
    df = df.copy()
    df.columns = [f"{ticker}_{field}" for ticker, field in df.columns]
    return df


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


def build_master(
    ohlcv_flat: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    macro_derived: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join all DataFrames on common trading days."""
    logger.info(
        f"build_master inputs: ohlcv={ohlcv_flat.shape}, returns={returns.shape}, "
        f"macro={macro.shape}, macro_derived={macro_derived.shape}"
    )
    if returns.empty:
        logger.warning("Returns DataFrame is empty! Creating minimal returns with zeros.")
        returns = pd.DataFrame(index=ohlcv_flat.index)
        for col in [c for c in ohlcv_flat.columns if "_Close" in c]:
            ticker = col.replace("_Close", "")
            returns[f"{ticker}_ret"] = 0.0
            returns[f"{ticker}_logret"] = 0.0

    common = (
        ohlcv_flat.index
        .intersection(returns.index)
        .intersection(macro.index)
        .intersection(macro_derived.index)
    )
    common = common.sort_values()
    if len(common) == 0:
        logger.error("No common dates found between datasets!")
        raise ValueError("Cannot build master dataset: no overlapping dates")

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
    logger.info(
        f"Master shape: {master.shape}, "
        f"range: {master.index[0].date()} -> {master.index[-1].date()}"
    )
    return master


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
