#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE
# Daily incremental update: fetches OHLCV for all tickers and macro data
# for the last trading day and appends to the HuggingFace dataset.

import os
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, upload_file
import yfinance as yf
import requests
import pandas_market_calendars as mcal

import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next NYSE trading day after the given date."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=date, end_date=date + pd.Timedelta(days=10))
    trading_days = schedule.index
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    # fallback: skip weekends only
    d = date + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def fetch_ticker_batch(tickers: list, target_date: pd.Timestamp,
                       retries: int = 3, backoff_factor: float = 2.0) -> pd.DataFrame:
    """
    Fetch OHLCV for a batch of tickers for a single date.
    Implements exponential backoff with jitter for rate‑limit errors.
    Returns DataFrame with columns like TICKER_Open, TICKER_High, ...
    """
    start = target_date - timedelta(days=5)
    end   = target_date + timedelta(days=1)

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=False, threads=False)
            if raw.empty:
                raise ValueError("Empty download")
            # Convert multi‑index to flat
            if isinstance(raw.columns, pd.MultiIndex):
                flat = pd.DataFrame()
                for t in tickers:
                    for col in cfg.OHLCV_COLS:
                        if (col, t) in raw.columns:
                            flat[f"{t}_{col}"] = raw[(col, t)]
                        else:
                            flat[f"{t}_{col}"] = np.nan
                flat.index = raw.index
            else:
                flat = raw.copy()
                for col in cfg.OHLCV_COLS:
                    flat.rename(columns={col: f"{tickers[0]}_{col}"}, inplace=True)
            # Get the row for target_date
            if target_date not in flat.index:
                # Try the most recent date <= target_date
                dates = flat.index[flat.index <= target_date]
                if len(dates) == 0:
                    raise ValueError(f"No data on or before {target_date}")
                actual_date = dates[-1]
                row = flat.loc[actual_date]
            else:
                row = flat.loc[target_date]
            return row
        except Exception as e:
            if attempt == retries:
                log.error(f"Batch {tickers} failed after {retries} attempts: {e}")
                raise
            sleep_time = backoff_factor ** (attempt - 1) + random.uniform(0, 1)
            log.warning(f"Batch {tickers} attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)


def fetch_all_etfs(target_date: pd.Timestamp, batch_size: int = 5) -> pd.Series:
    """
    Fetch OHLCV for all ETFs and benchmarks, in batches.
    Returns a Series indexed by column name (e.g., TLT_Close).
    """
    all_tickers = cfg.ALL_TICKERS
    batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]

    rows = []
    for i, batch in enumerate(batches):
        log.info(f"Fetching batch {i+1}/{len(batches)}: {batch}")
        try:
            row = fetch_ticker_batch(batch, target_date)
            rows.append(row)
        except Exception as e:
            log.error(f"Batch {batch} failed: {e}")
            # Continue with other batches (maybe some tickers can be fetched later)
        # Pause between batches to avoid rate limits
        time.sleep(random.uniform(1.0, 2.0))

    if not rows:
        raise ValueError("No data fetched for any ticker")

    # Combine all rows (each is a Series)
    combined = pd.concat(rows, axis=0)
    return combined


def fetch_fred_data(target_date: pd.Timestamp) -> pd.Series:
    """Fetch all FRED series for the target date."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    data = {}
    for col_name, series_code in cfg.FRED_SERIES.items():
        params = {
            "series_id": series_code,          # use the actual FRED code
            "api_key": cfg.FRED_API_KEY,
            "file_type": "json",
            "observation_start": target_date.strftime("%Y-%m-%d"),
            "observation_end":   target_date.strftime("%Y-%m-%d"),
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                data[col_name] = float(obs[0]["value"])
            else:
                data[col_name] = np.nan
        except Exception as e:
            log.warning(f"FRED {col_name} failed: {e}")
            data[col_name] = np.nan
        time.sleep(0.3)  # respectful pacing
    return pd.Series(data, name=target_date)


def update_master() -> None:
    """Main incremental update."""
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # Load existing master
    try:
        master_path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=cfg.FILE_MASTER,
            repo_type="dataset",
            token=cfg.HF_TOKEN,
            local_dir="/tmp",
            local_dir_use_symlinks=False,
        )
        master = pd.read_parquet(master_path)
        master.index = pd.to_datetime(master.index)
        last_date = master.index[-1]
        log.info(f"Last stored: {last_date.date()} | rows: {len(master)}")
    except Exception as e:
        log.error(f"Failed to load master: {e}")
        sys.exit(1)

    # Determine next trading day to update
    next_td = next_trading_day(last_date)
    if next_td.date() > datetime.now().date():
        log.info("Next trading day is in the future – nothing to update.")
        return

    target_date = next_td
    log.info(f"Update window: {target_date.date()} to {target_date.date()}")

    # Fetch new OHLCV
    try:
        etf_row = fetch_all_etfs(target_date)
    except Exception as e:
        log.error(f"OHLCV fetch failed: {e}")
        sys.exit(1)

    # Fetch new FRED data
    try:
        fred_row = fetch_fred_data(target_date)
    except Exception as e:
        log.error(f"FRED fetch failed: {e}")
        sys.exit(1)

    # Combine into a single row (same columns as master)
    new_row = pd.Series(index=master.columns, dtype=float)
    # Copy OHLCV columns
    for col in etf_row.index:
        if col in master.columns:
            new_row[col] = etf_row[col]
    # Copy FRED columns
    for col in fred_row.index:
        if col in master.columns:
            new_row[col] = fred_row[col]

    # Append to master (replace deprecated append with concat)
    new_df = pd.DataFrame([new_row], index=[target_date])
    master = pd.concat([master, new_df]).sort_index()

    # Save locally and upload
    tmp_path = "/tmp/master.parquet"
    master.to_parquet(tmp_path)
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=cfg.FILE_MASTER,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Daily update: added {target_date.date()}",
    )
    log.info(f"Updated master with {target_date.date()}")


if __name__ == "__main__":
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    update_master()
