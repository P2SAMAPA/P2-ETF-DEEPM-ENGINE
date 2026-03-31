# data_download.py — ETF OHLCV + FRED macro downloader
# Proven pattern: one ticker at a time to avoid yfinance rate limits.
# Saves all parquet files locally to data/ directory.
# Then data_upload_hf.py pushes them to HuggingFace.
#
# Usage:
#   python data_download.py --mode seed         # full history from 2008
#   python data_download.py --mode incremental  # append latest day only

import argparse
import os
import random
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

warnings.filterwarnings("ignore")
os.makedirs(config.DATA_DIR, exist_ok=True)


# ------------------------------------------------------------
#  Helper: robust yfinance session with retries
# ------------------------------------------------------------
def get_yf_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


# ------------------------------------------------------------
#  Ticker data fetcher with retries and Stooq fallback
# ------------------------------------------------------------
def fetch_ticker_with_fallback(
    ticker: str,
    start: str,
    end: str,
    data_type: str = "ohlcv",
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> pd.DataFrame | None:
    """
    Fetch data for a single ticker.
    data_type: 'ohlcv' -> returns OHLCV columns, 'close' -> returns only close.
    Retries with exponential backoff + jitter. Falls back to Stooq with .US suffix.
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
                session=get_yf_session(),
                threads=False,
            )
            if df.empty:
                raise ValueError("Empty data from yfinance")
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
                print(f"yfinance failed for {ticker} after {max_retries} retries: {e}")
                break
            # Exponential backoff with jitter
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retry {attempt+1}/{max_retries} for {ticker} after {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    # ----- Stooq fallback -----
    # Try original ticker first, then with .US suffix (common for US ETFs)
    stooq_tickers = [ticker, f"{ticker}.US"]
    for stooq_ticker in stooq_tickers:
        try:
            print(f"Falling back to Stooq for {stooq_ticker}...")
            df = pdr.DataReader(stooq_ticker, 'stooq', start, end)
            if df.empty:
                raise ValueError("Empty data from Stooq")
            # Stooq returns same columns: Open, High, Low, Close, Volume
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
            print(f"Stooq {stooq_ticker} failed: {e}")
            continue

    print(f"All fallbacks failed for {ticker}")
    return None


# ------------------------------------------------------------
#  Price fetching (close only) with fallback
# ------------------------------------------------------------
def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch Close prices one ticker at a time, with retries and Stooq fallback.
    Returns DataFrame: index=Date, columns=tickers (Close only).
    """
    print(f"Fetching prices {start} -> {end} ({len(tickers)} tickers)")
    frames = []

    for ticker in tqdm(tickers, desc="Prices"):
        df = fetch_ticker_with_fallback(ticker, start, end, data_type="close")
        if df is not None:
            frames.append(df)
        else:
            print(f"  WARNING: {ticker} returned empty after all attempts")
        # Increased delay between tickers to avoid rate limits
        time.sleep(0.8)

    if not frames:
        raise RuntimeError("No price data fetched — all tickers failed.")

    prices = pd.concat(frames, axis=1)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] if col[0] != "" else col[1] for col in prices.columns]
    prices.columns = [str(c).strip() for c in prices.columns]

    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"

    print(f"  Prices: {prices.shape}, range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    return prices.sort_index()


# ------------------------------------------------------------
#  OHLCV fetching with fallback
# ------------------------------------------------------------
def fetch_ohlcv(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch full OHLCV one ticker at a time, with retries and Stooq fallback.
    Returns flat DataFrame: TLT_Open, TLT_High, TLT_Low, TLT_Close, TLT_Volume, ...
    """
    print(f"Fetching OHLCV {start} -> {end} ({len(tickers)} tickers)")
    frames = []

    for ticker in tqdm(tickers, desc="OHLCV"):
        df = fetch_ticker_with_fallback(ticker, start, end, data_type="ohlcv")
        if df is not None:
            df.columns = [f"{ticker}_{c}" for c in df.columns]
            frames.append(df)
        else:
            print(f"  WARNING: {ticker} returned empty after all attempts")
        # Increased delay between tickers
        time.sleep(0.8)

    if not frames:
        raise RuntimeError("No OHLCV data fetched — all tickers failed.")

    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns = [str(c).strip() for c in ohlcv.columns]
    ohlcv.index = pd.to_datetime(ohlcv.index)
    if ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)
    ohlcv.index.name = "Date"

    print(f"  OHLCV: {ohlcv.shape}, range: {ohlcv.index[0].date()} -> {ohlcv.index[-1].date()}")
    return ohlcv.sort_index()


# ------------------------------------------------------------
#  The rest of the script (unchanged)
# ------------------------------------------------------------

def compute_returns_simple(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    rets = prices.pct_change().dropna(how="all")
    rets.index.name = "Date"
    return rets


def compute_returns_log(prices: pd.DataFrame) -> pd.DataFrame:
    """Log daily returns."""
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    rets.index.name = "Date"
    return rets


def compute_volatility(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised rolling volatility."""
    vol = log_returns.rolling(config.VOL_WINDOW).std() * np.sqrt(252)
    vol = vol.dropna(how="all")
    vol.index.name = "Date"
    return vol


def compute_all_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combined simple + log returns in one DataFrame.
    Columns: TLT_ret, TLT_logret, LQD_ret, LQD_logret, ...
    """
    simple = compute_returns_simple(prices)
    log = compute_returns_log(prices)

    simple.columns = [f"{c}_ret" for c in simple.columns]
    log.columns = [f"{c}_logret" for c in log.columns]

    combined = pd.concat([simple, log], axis=1).sort_index(axis=1)
    combined.index.name = "Date"
    return combined.dropna(how="all")


def fetch_macro(start: str, end: str) -> pd.DataFrame:
    """
    Download all FRED macro series from config.FRED_SERIES.
    Forward-fills gaps (FRED releases lag by 1 business day).
    Returns DataFrame indexed by Date, columns = friendly names.
    """
    print(f"Fetching FRED macro {start} -> {end}")
    fred = Fred(api_key=config.FRED_API_KEY)
    frames = {}

    for name, series_id in tqdm(config.FRED_SERIES.items(), desc="FRED"):
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            s.name = name
            frames[name] = s
            print(f"  {name} ({series_id}): {len(s)} obs")
        except Exception as e:
            print(f"  WARNING: FRED {name} ({series_id}) failed: {e}")

    if not frames:
        raise RuntimeError("No FRED series downloaded successfully.")

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)
    macro.index.name = "Date"
    macro = macro.sort_index().ffill()

    print(f"  Macro: {macro.shape}")
    return macro


def compute_macro_derived(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer macro features from raw FRED series.
    All rolling z-scores use ZSCORE_WINDOW (63 days ~ 1 quarter).
    """
    d = pd.DataFrame(index=macro.index)
    w = config.ZSCORE_WINDOW

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
        d["macro_stress_composite"] = (
            zscore(macro["VIX"]) +
            zscore(macro["HY_SPREAD"]) +
            (-zscore(macro["T10Y2Y"]))
        ) / 3.0

    d.index.name = "Date"
    d = d.dropna(how="all")
    print(f"  Derived macro: {d.shape}, cols: {list(d.columns)}")
    return d


def save_parquet(df: pd.DataFrame, name: str) -> None:
    """
    Save DataFrame to data/{name}.parquet.
    Date is saved as a column (not index) — consistent with proven pattern.
    """
    path = os.path.join(config.DATA_DIR, f"{name}.parquet")
    df_save = df.copy()

    # Flatten any MultiIndex columns
    if isinstance(df_save.columns, pd.MultiIndex):
        df_save.columns = [
            col[0] if col[0] != "" else col[1] for col in df_save.columns
        ]
    df_save.columns = [str(c).strip() for c in df_save.columns]

    # Reset index so Date becomes a column
    if df_save.index.name == "Date" or isinstance(df_save.index, pd.DatetimeIndex):
        df_save = df_save.reset_index()

    if "Date" in df_save.columns:
        df_save["Date"] = pd.to_datetime(df_save["Date"])
        if df_save["Date"].dt.tz is not None:
            df_save["Date"] = df_save["Date"].dt.tz_localize(None)

    df_save.to_parquet(path, index=False, engine="pyarrow")
    print(f"  Saved {name}.parquet ({len(df_save)} rows, {df_save.shape[1]} cols)")


def load_parquet(name: str) -> pd.DataFrame:
    """
    Load data/{name}.parquet and restore Date as DatetimeIndex.
    """
    path = os.path.join(config.DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run seed first.")

    df = pd.read_parquet(path)

    # Restore Date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    # Drop any residual index columns
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ("date", "index", "level_0"):
            df = df.drop(columns=[col])

    return df.sort_index()


def build_master(
    ohlcv: pd.DataFrame,
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    macro_derived: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join all DataFrames on common trading days → master.parquet.
    No lookahead: FRED is only forward-filled, never backward-filled.
    """
    common = (
        ohlcv.index
        .intersection(returns.index)
        .intersection(macro.index)
        .intersection(macro_derived.index)
    )
    common = common.sort_values()

    master = pd.concat(
        [
            ohlcv.reindex(common),
            returns.reindex(common),
            macro.reindex(common),
            macro_derived.reindex(common),
        ],
        axis=1,
    )
    master.index.name = "Date"
    print(f"  Master: {master.shape}, range: {master.index[0].date()} -> {master.index[-1].date()}")
    return master


def build_all(start: str, end: str) -> None:
    print(f"\n{'='*60}")
    print(f"P2-ETF-DEEPM DATASET BUILD: {start} -> {end}")
    print(f"Tickers : {config.ALL_TICKERS}")
    print(f"{'='*60}\n")

    # 1. OHLCV
    print("[1/6] ETF OHLCV...")
    ohlcv = fetch_ohlcv(config.ALL_TICKERS, start=start, end=end)
    save_parquet(ohlcv, "etf_ohlcv")

    # 2. Close prices only (for returns)
    close_cols = [c for c in ohlcv.columns if c.endswith("_Close")]
    prices = ohlcv[close_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    # 3. Returns
    print("[2/6] ETF returns...")
    returns = compute_all_returns(prices)
    save_parquet(returns, "etf_returns")

    # 4. Volatility (stored separately — some models use it directly)
    print("[3/6] ETF volatility...")
    log_rets = compute_returns_log(prices)
    vol = compute_volatility(log_rets)
    vol.columns = [f"{c}_vol" for c in vol.columns]
    save_parquet(vol, "etf_vol")

    # 5. FRED macro
    print("[4/6] FRED macro...")
    macro = fetch_macro(start=start, end=end)
    save_parquet(macro, "macro_fred")

    # 6. Derived macro features
    print("[5/6] Derived macro features...")
    macro_derived = compute_macro_derived(macro)
    save_parquet(macro_derived, "macro_derived")

    # 7. Master aligned file
    print("[6/6] Master aligned file...")
    master = build_master(ohlcv, returns, macro, macro_derived)
    save_parquet(master, "master")

    print(f"\n{'='*60}")
    print("BUILD COMPLETE")
    print(f"  Trading days : {len(master)}")
    print(f"  Master cols  : {master.shape[1]}")
    print(f"{'='*60}")


def incremental_update() -> None:
    print("\nIncremental update mode...")

    # Load existing OHLCV to find last date
    try:
        ohlcv_existing = load_parquet("etf_ohlcv")
        last_date = ohlcv_existing.index.max()
    except FileNotFoundError:
        print("No local data found — running full seed instead.")
        seed()
        return

    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    if start >= end:
        print(f"Already up to date (last: {last_date.date()}). Nothing to do.")
        return

    print(f"Fetching new data: {start} -> {end}")

    # New OHLCV
    new_ohlcv = fetch_ohlcv(config.ALL_TICKERS, start=start, end=end)
    if new_ohlcv.empty:
        print("No new data returned (market closed today?).")
        return

    ohlcv = pd.concat([ohlcv_existing, new_ohlcv])
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()
    save_parquet(ohlcv, "etf_ohlcv")

    # Recompute returns + vol on full history
    close_cols = [c for c in ohlcv.columns if c.endswith("_Close")]
    prices = ohlcv[close_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    returns = compute_all_returns(prices)
    save_parquet(returns, "etf_returns")

    log_rets = compute_returns_log(prices)
    vol = compute_volatility(log_rets)
    vol.columns = [f"{c}_vol" for c in vol.columns]
    save_parquet(vol, "etf_vol")

    # Always re-fetch full FRED history (catches publication lags)
    macro = fetch_macro(start=config.DATA_START, end=end)
    save_parquet(macro, "macro_fred")

    macro_derived = compute_macro_derived(macro)
    save_parquet(macro_derived, "macro_derived")

    master = build_master(ohlcv, returns, macro, macro_derived)
    save_parquet(master, "master")

    print(f"\nUpdate complete. Latest: {master.index[-1].date()}, Total days: {len(master)}")


def seed() -> None:
    end = datetime.today().strftime("%Y-%m-%d")
    build_all(start=config.DATA_START, end=end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-ETF-DEEPM dataset builder")
    parser.add_argument(
        "--mode",
        choices=["seed", "incremental"],
        default="incremental",
        help="seed = full rebuild from 2008, incremental = append latest day",
    )
    args = parser.parse_args()

    if args.mode == "seed":
        seed()
    else:
        incremental_update()

    print("\nDataset build complete.")
