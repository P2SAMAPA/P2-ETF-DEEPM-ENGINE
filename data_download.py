# data_download.py — ETF OHLCV + FRED macro downloader
# Uses robust download functions from data_utils.py.
# Saves all parquet files locally to data/ directory.
# Then data_upload_hf.py pushes them to HuggingFace.
#
# Usage:
#   python data_download.py --mode seed         # full history from 2008
#   python data_download.py --mode incremental  # append latest day only

import argparse
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import the robust utilities from the same directory
import data_utils as du
import config

warnings.filterwarnings("ignore")
os.makedirs(config.DATA_DIR, exist_ok=True)


# ------------------------------------------------------------
#  Helper: load existing data (incremental mode)
# ------------------------------------------------------------
def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from data/ and restore Date as index."""
    path = os.path.join(config.DATA_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def save_parquet(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame to data/{name}.parquet, with Date as a column."""
    path = os.path.join(config.DATA_DIR, f"{name}.parquet")
    df_save = df.copy()
    if df_save.index.name == "Date" or isinstance(df_save.index, pd.DatetimeIndex):
        df_save = df_save.reset_index()
    if "Date" in df_save.columns:
        df_save["Date"] = pd.to_datetime(df_save["Date"])
        if df_save["Date"].dt.tz is not None:
            df_save["Date"] = df_save["Date"].dt.tz_localize(None)
    df_save.to_parquet(path, index=False, engine="pyarrow")
    print(f"  Saved {name}.parquet ({len(df_save)} rows, {df_save.shape[1]} cols)")


# ------------------------------------------------------------
#  Derived calculations (unchanged from your original)
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


# ------------------------------------------------------------
#  Build dataset using robust data_utils
# ------------------------------------------------------------
def build_all(start: str, end: str) -> None:
    print(f"\n{'='*60}")
    print(f"P2-ETF-DEEPM DATASET BUILD: {start} -> {end}")
    print(f"Tickers : {config.ALL_TICKERS}")
    print(f"{'='*60}\n")

    # 1. OHLCV (using robust batch download)
    print("[1/6] ETF OHLCV...")
    ohlcv_multi = du.download_ohlcv(config.ALL_TICKERS, start=start, end=end)
    ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)
    save_parquet(ohlcv_flat, "etf_ohlcv")

    # 2. Close prices only (for returns)
    close_cols = [c for c in ohlcv_flat.columns if c.endswith("_Close")]
    prices = ohlcv_flat[close_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    # 3. Returns
    print("[2/6] ETF returns...")
    returns = compute_all_returns(prices)
    save_parquet(returns, "etf_returns")

    # 4. Volatility
    print("[3/6] ETF volatility...")
    log_rets = compute_returns_log(prices)
    vol = compute_volatility(log_rets)
    vol.columns = [f"{c}_vol" for c in vol.columns]
    save_parquet(vol, "etf_vol")

    # 5. FRED macro (using robust data_utils)
    print("[4/6] FRED macro...")
    macro = du.download_fred(start=start, end=end)
    save_parquet(macro, "macro_fred")

    # 6. Derived macro features
    print("[5/6] Derived macro features...")
    macro_derived = du.compute_macro_derived(macro)
    save_parquet(macro_derived, "macro_derived")

    # 7. Master aligned file
    print("[6/6] Master aligned file...")
    master = du.build_master(ohlcv_flat, returns, macro, macro_derived)
    save_parquet(master, "master")

    print(f"\n{'='*60}")
    print("BUILD COMPLETE")
    print(f"  Trading days : {len(master)}")
    print(f"  Master cols  : {master.shape[1]}")
    print(f"{'='*60}")


# ------------------------------------------------------------
#  Incremental update (using robust download for new data)
# ------------------------------------------------------------
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
    new_ohlcv_multi = du.download_ohlcv(config.ALL_TICKERS, start=start, end=end)
    if new_ohlcv_multi.empty:
        print("No new data returned (market closed today?).")
        return
    new_ohlcv_flat = du.flatten_ohlcv(new_ohlcv_multi)

    ohlcv_flat = pd.concat([ohlcv_existing, new_ohlcv_flat])
    ohlcv_flat = ohlcv_flat[~ohlcv_flat.index.duplicated(keep="last")].sort_index()
    save_parquet(ohlcv_flat, "etf_ohlcv")

    # Recompute returns + vol on full history
    close_cols = [c for c in ohlcv_flat.columns if c.endswith("_Close")]
    prices = ohlcv_flat[close_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    returns = compute_all_returns(prices)
    save_parquet(returns, "etf_returns")

    log_rets = compute_returns_log(prices)
    vol = compute_volatility(log_rets)
    vol.columns = [f"{c}_vol" for c in vol.columns]
    save_parquet(vol, "etf_vol")

    # Always re-fetch full FRED history (catches publication lags)
    macro = du.download_fred(start=config.DATA_START, end=end)
    save_parquet(macro, "macro_fred")

    macro_derived = du.compute_macro_derived(macro)
    save_parquet(macro_derived, "macro_derived")

    master = du.build_master(ohlcv_flat, returns, macro, macro_derived)
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
