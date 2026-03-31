# data_utils.py — Shared data download, transform and HuggingFace I/O
# Now includes robust downloading with retries and fallback to Stooq.

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
#  Robust yfinance session
# ------------------------------------------------------------
def _get_yf_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


# ------------------------------------------------------------
#  Helper: fetch one ticker with retries and Stooq fallback
# ------------------------------------------------------------
def _fetch_one_ticker_robust(ticker: str, start: str, end: str, data_type: str = "ohlcv",
                              max_retries: int = 5, base_delay: float = 5.0):
    """
    Fetch data for a single ticker.
    data_type: 'ohlcv' -> returns OHLCV, 'close' -> returns only close.
    Returns DataFrame or None if both sources fail.
    """
    # ----- yfinance attempt with retries -----
    for attempt in range(max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                session=_get_yf_session(),
                threads=False,
            )
            if df.empty:
                raise ValueError("Empty data from yfinance")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            if data_type == "ohlcv":
                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep_cols].copy()
            else:  # close only
                if "Close" not in df.columns:
                    raise ValueError("No Close column")
                df = df[["Close"]].copy()
                df.columns = [ticker]
            return df

        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"yfinance failed for {ticker} after {max_retries} retries: {e}")
                break
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"Retry {attempt+1}/{max_retries} for {ticker} after {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # ----- Stooq fallback -----
    stooq_tickers = [ticker, f"{ticker}.US"]
    for stooq_ticker in stooq_tickers:
        try:
            logger.info(f"Falling back to Stooq for {stooq_ticker}...")
            # Use pandas_datareader
            from pandas_datareader import DataReader
            df = DataReader(stooq_ticker, 'stooq', start, end)
            if df.empty:
                # try shorter range
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
#  ETF OHLCV download (now with fallback)
# ------------------------------------------------------------
def download_ohlcv(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """
    Download OHLCV for all tickers via yfinance.
    If batch download fails, falls back to one-by-one with retries and Stooq.
    Returns DataFrame with MultiIndex columns (ticker, field).
    """
    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Downloading OHLCV: {tickers} from {start} to {end}")

    # First attempt: batch download with retries
    for attempt in range(3):   # up to 3 batch attempts
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                session=_get_yf_session(),
            )
            if raw.empty:
                raise ValueError("Empty batch result")
            # Success
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.copy()
            else:
                # Single ticker case
                ticker = tickers[0] if len(tickers) == 1 else tickers[0]
                df = pd.concat({ticker: raw}, axis=1)
                df.columns = pd.MultiIndex.from_tuples(
                    [(t, f) for t, f in df.columns], names=["Ticker", "Field"]
                )
            df.columns.names = ["Ticker", "Field"]
            df = df.sort_index()
            logger.info(f"OHLCV batch download successful. Shape: {df.shape}")
            return df
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Batch download failed after 3 attempts: {e}. Falling back to one-by-one.")
                break
            sleep_time = 10 * (2 ** attempt) + random.uniform(0, 2)
            logger.info(f"Batch retry {attempt+1}/3 after {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # Fallback: download each ticker individually with robust helper
    frames = []
    for ticker in tickers:
        logger.info(f"Downloading {ticker} individually...")
        df = _fetch_one_ticker_robust(ticker, start, end, data_type="ohlcv")
        if df is not None:
            # Re‑create MultiIndex columns
            df.columns = pd.MultiIndex.from_tuples([(ticker, col) for col in df.columns],
                                                   names=["Ticker", "Field"])
            frames.append(df)
        else:
            logger.warning(f"Could not fetch {ticker} after all attempts")
        time.sleep(0.5)   # small delay between individual downloads

    if not frames:
        raise ValueError(f"No OHLCV data fetched for any ticker.")

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    combined.sort_index(inplace=True)
    logger.info(f"OHLCV fallback download successful. Shape: {combined.shape}")
    return combined


# ------------------------------------------------------------
#  The rest of the file remains unchanged
# ------------------------------------------------------------
# (flatten_ohlcv, compute_returns, download_fred, etc. stay exactly as before)
# I'll include them for completeness, but they are identical to your original.

def flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex OHLCV DataFrame to single-level columns.
    Columns become: TLT_Close, TLT_Open, TLT_High, TLT_Low, TLT_Volume, etc.
    """
    df = df.copy()
    df.columns = [f"{ticker}_{field}" for ticker, field in df.columns]
    return df


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
