# data_utils.py — Shared data download, transform and HuggingFace I/O
#
# FIXES applied (2026-04-18):
#   1. Stooq fallback now uses direct CSV download instead of pandas_datareader
#      (pandas_datareader's Stooq reader is broken — missing Date column error)
#   2. yfinance: use yf.Ticker().history() which handles crumb/cookie auth better
#      than yf.download() in CI/GitHub Actions environments
#   3. download_ohlcv: no longer raises if SOME tickers fail — warns and continues
#      (previously a single ticker failure killed the entire run)
#   4. Added longer initial backoff and jitter to reduce 429 rate-limit hits

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
import requests
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download, upload_file
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Persistent requests session (used for Stooq direct CSV downloads)
# ─────────────────────────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,          # delays: 2, 4, 8, 16, 32 s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Mimic a browser so Stooq/YF don't reject us outright
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

_SESSION = _get_session()


# ─────────────────────────────────────────────────────────────────────────────
# Stooq direct CSV fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_stooq_csv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV from Stooq as a direct CSV request.
    Tries bare ticker first, then common suffixes (.US, -US).
    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by date, or None if all attempts fail.
    """
    # Stooq date format: YYYYMMDD
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")

    suffixes = ["", ".US", "-US"]
    for suffix in suffixes:
        stooq_sym = f"{ticker}{suffix}".lower()
        url = (
            f"https://stooq.com/q/d/l/"
            f"?s={stooq_sym}&d1={d1}&d2={d2}&i=d"
        )
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()

            # Stooq returns "No data" as plain text when symbol not found
            if b"No data" in resp.content[:50] or len(resp.content) < 50:
                logger.warning(f"  Stooq: no data for {stooq_sym}")
                continue

            df = pd.read_csv(io.StringIO(resp.text))

            if df.empty:
                logger.warning(f"  Stooq CSV empty for {stooq_sym}")
                continue

            # Normalise the date column — Stooq uses 'Date' but let's be safe
            date_col = next(
                (c for c in df.columns if c.strip().lower() == "date"), None
            )
            if date_col is None:
                logger.warning(f"  Stooq: no Date column for {stooq_sym}. Cols: {list(df.columns)}")
                continue

            df.index = pd.to_datetime(df[date_col])
            df.index.name = "Date"
            df = df.drop(columns=[date_col])
            df = df.sort_index()

            # Rename to standard capitalisation
            rename = {c: c.strip().capitalize() for c in df.columns}
            df = df.rename(columns=rename)
            # Volume column might be 'Volume' already — ensure it exists
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    raise ValueError(f"Missing expected column '{col}'")

            logger.info(f"  ✅ Stooq success for {stooq_sym}: {len(df)} rows")
            return df

        except Exception as exc:
            logger.warning(f"  Stooq attempt failed for {stooq_sym}: {exc}")
            time.sleep(random.uniform(5, 10))
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-ticker robust fetch  (yfinance → Stooq CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_one_ticker_robust(
    ticker: str,
    start: str,
    end: str,
    data_type: str = "ohlcv",
    max_yf_retries: int = 4,
    base_delay: float = 8.0,
) -> pd.DataFrame | None:
    """
    Try yfinance first (up to max_yf_retries), then fall back to Stooq CSV.

    data_type:
        'ohlcv' → return DataFrame with columns [Open, High, Low, Close, Volume]
        'close' → return DataFrame with single column named <ticker>
    """

    # ── 1. yfinance via Ticker.history() ──────────────────────────────────────
    #   Using Ticker().history() handles crumb/cookie auth much better than
    #   yf.download() inside GitHub Actions / cloud runners.
    for attempt in range(max_yf_retries):
        try:
            wait = base_delay * (2 ** attempt) + random.uniform(0, 5)
            if attempt > 0:
                logger.info(f"  YF retry {attempt}/{max_yf_retries} for {ticker} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                time.sleep(random.uniform(2, 5))   # always a small initial delay

            t = yf.Ticker(ticker)
            df = t.history(
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                raise ValueError(f"Empty response from yfinance for {ticker}")

            # Drop timezone so downstream joins work cleanly
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            if data_type == "ohlcv":
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].copy()
            else:
                if "Close" not in df.columns:
                    raise ValueError("No Close column in yfinance response")
                df = df[["Close"]].copy()
                df.columns = [ticker]

            logger.info(f"  ✅ YF success for {ticker}: {len(df)} rows")
            return df

        except Exception as exc:
            logger.warning(f"  ❌ YF attempt {attempt+1}/{max_yf_retries} failed for {ticker}: {exc}")
            if attempt == max_yf_retries - 1:
                logger.warning(f"  YF exhausted for {ticker} — trying Stooq CSV fallback")

    # ── 2. Stooq CSV fallback ─────────────────────────────────────────────────
    logger.info(f"  🔄 Stooq CSV fallback for {ticker}...")
    df = _fetch_stooq_csv(ticker, start, end)

    if df is None:
        logger.error(f"  ❌ All sources failed for {ticker}")
        return None

    # Normalise output format to match yfinance output
    if data_type == "ohlcv":
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
    else:
        if "Close" not in df.columns:
            logger.error(f"  Stooq result for {ticker} has no Close column")
            return None
        df = df[["Close"]].copy()
        df.columns = [ticker]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Batch OHLCV download
# ─────────────────────────────────────────────────────────────────────────────

def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for all tickers one-by-one (batch yf.download is too
    aggressive on rate limits from CI runners).

    Returns a DataFrame with MultiIndex columns (Ticker, Field).

    CHANGED: no longer raises if some tickers fail — logs a warning and
    continues with the tickers that succeeded.  Only raises if ALL fail.
    """
    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Downloading OHLCV (sequential): {len(tickers)} tickers {start}→{end}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")

        # Progressive pacing: small base delay + extra pause every 5 tickers
        base_wait = random.uniform(2, 4)
        if i > 0 and i % 5 == 0:
            extra = random.uniform(8, 12)
            logger.info(f"  ⏳ Pacing delay ({extra:.1f}s) after every 5 tickers...")
            time.sleep(extra)
        else:
            time.sleep(base_wait)

        df = _fetch_one_ticker_robust(ticker, start, end, data_type="ohlcv")

        if df is not None and not df.empty:
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, col) for col in df.columns],
                names=["Ticker", "Field"],
            )
            frames.append(df)
        else:
            logger.warning(f"  ⚠️ Skipping {ticker} — no data from any source")
            failed.append(ticker)

    if not frames:
        raise ValueError("No data fetched from any source for any ticker.")

    if failed:
        logger.warning(
            f"⚠️  {len(failed)}/{len(tickers)} tickers could not be fetched "
            f"and will be MISSING from the dataset: {failed}"
        )

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    logger.info(f"OHLCV download complete. Shape: {combined.shape}. Failed: {failed or 'none'}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Flatten / returns / volatility
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# FRED macro
# ─────────────────────────────────────────────────────────────────────────────

def download_fred(start: str, end: str = None) -> pd.DataFrame:
    """Download all FRED macro series defined in config.FRED_SERIES."""
    end = end or date.today().strftime("%Y-%m-%d")
    fred = Fred(api_key=cfg.FRED_API_KEY)
    frames = {}
    for name, series_id in cfg.FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = name
            frames[name] = s
            logger.info(f"FRED {series_id} ({name}): {len(s)} observations")
        except Exception as exc:
            logger.error(f"Failed to fetch FRED {series_id}: {exc}")

    if not frames:
        raise ValueError("No FRED series downloaded successfully")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    trading_days = get_trading_days(start, end)
    df = df.reindex(trading_days).ffill().dropna(how="all")
    logger.info(f"Macro FRED shape: {df.shape}, range: {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Trading calendar helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str = None) -> pd.DatetimeIndex:
    """Return NYSE trading days between start and end (inclusive)."""
    nyse = mcal.get_calendar("NYSE")
    end = end or date.today().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)


def last_trading_day() -> str:
    """Return the most recent completed NYSE trading day as YYYY-MM-DD."""
    today = date.today()
    nyse = mcal.get_calendar("NYSE")
    start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    if len(days) == 0:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    latest = days[-1]
    if latest.date() >= today:
        latest = days[-2] if len(days) > 1 else days[-1]
    return latest.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Derived macro features
# ─────────────────────────────────────────────────────────────────────────────

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
        d["credit_stress"] = (zscore(macro["HY_SPREAD"]) + zscore(macro["IG_SPREAD"])) / 2.0

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


# ─────────────────────────────────────────────────────────────────────────────
# Master dataset builder
# ─────────────────────────────────────────────────────────────────────────────

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

    # Guard: empty returns
    if returns.empty:
        logger.warning("Returns DataFrame is empty — filling with zeros.")
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
    ).sort_values()

    if len(common) == 0:
        logger.error("No common dates found between datasets!")
        logger.error(f"  ohlcv:        {ohlcv_flat.index.min()} → {ohlcv_flat.index.max()}")
        logger.error(f"  returns:      {returns.index.min()} → {returns.index.max()}")
        logger.error(f"  macro:        {macro.index.min()} → {macro.index.max()}")
        logger.error(f"  macro_derived:{macro_derived.index.min()} → {macro_derived.index.max()}")
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

    if master.empty:
        raise ValueError("Master dataset is empty after concatenation")

    logger.info(
        f"Master shape: {master.shape}, "
        f"range: {master.index[0].date()} → {master.index[-1].date()}"
    )
    return master


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace I/O
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True, engine="pyarrow")
    return buf.getvalue()


def upload_parquet(df: pd.DataFrame, hf_path: str, commit_msg: str) -> None:
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=_df_to_bytes(df),
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
    return df.sort_index()


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
