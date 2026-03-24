#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE
# Daily incremental update using the same logic as seed.py.
# Updates: master, etf_ohlcv, etf_returns, etf_vol, macro_fred, macro_derived, metadata.

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
import data_utils as du

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers (unchanged)
# ----------------------------------------------------------------------
def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=date, end_date=date + pd.Timedelta(days=10))
    trading_days = schedule.index
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    d = date + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def fetch_ticker_batch(tickers: list, target_date: pd.Timestamp,
                       retries: int = 3, backoff_factor: float = 2.0) -> pd.Series:
    start = target_date - timedelta(days=5)
    end   = target_date + timedelta(days=1)

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=False, threads=False)
            if raw.empty:
                raise ValueError("Empty download")
            if isinstance(raw.columns, pd.MultiIndex):
                row = {}
                for t in tickers:
                    for col in cfg.OHLCV_COLS:
                        if (col, t) in raw.columns:
                            row[f"{t}_{col}"] = raw[(col, t)].iloc[-1]
                        else:
                            row[f"{t}_{col}"] = np.nan
            else:
                row = {}
                for col in cfg.OHLCV_COLS:
                    row[f"{tickers[0]}_{col}"] = raw[col].iloc[-1]
            return pd.Series(row)
        except Exception as e:
            if attempt == retries:
                log.error(f"Batch {tickers} failed after {retries} attempts: {e}")
                raise
            sleep_time = backoff_factor ** (attempt - 1) + random.uniform(0, 1)
            log.warning(f"Batch {tickers} attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)


def fetch_all_etfs(target_date: pd.Timestamp, batch_size: int = 5) -> pd.Series:
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
        time.sleep(random.uniform(1.0, 2.0))

    if not rows:
        raise ValueError("No data fetched for any ticker")
    return pd.concat(rows)


def fetch_fred_row(target_date: pd.Timestamp) -> pd.Series:
    """Fetch FRED data for a single date using data_utils.download_fred."""
    start = target_date.strftime("%Y-%m-%d")
    end   = start
    df = du.download_fred(start=start, end=end)
    if df.empty:
        log.warning(f"FRED returned empty for {start}")
        return pd.Series(index=cfg.FRED_SERIES.keys(), dtype=float)
    return df.iloc[0]


def load_parquet(path: str) -> pd.DataFrame:
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO, filename=path,
            repo_type="dataset", token=cfg.HF_TOKEN,
            local_dir="/tmp", local_dir_use_symlinks=False,
        )
        df = pd.read_parquet(local)
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def save_and_upload(df: pd.DataFrame, path: str, commit_msg: str):
    tmp = f"/tmp/{path.replace('/', '_')}.parquet"
    df.to_parquet(tmp)
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo=path,
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
    )
    log.info(f"Uploaded {path}")


def update_metadata():
    """Update metadata.json with the latest trading day and timestamp."""
    master = load_parquet(cfg.FILE_MASTER)
    if master.empty:
        return
    metadata = {
        "last_updated":       datetime.utcnow().isoformat(),
        "last_trading_day":   str(master.index[-1].date()),
        "n_trading_days":     len(master),
    }
    du.upload_json(metadata, cfg.FILE_METADATA, f"[daily] update metadata {master.index[-1].date()}")


def update_master() -> None:
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # Load existing master
    master = load_parquet(cfg.FILE_MASTER)
    if master.empty:
        log.error("No master dataset found. Run seed.py first.")
        sys.exit(1)
    last_date = master.index[-1]
    log.info(f"Last stored: {last_date.date()} | rows: {len(master)}")

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
        fred_row = fetch_fred_row(target_date)
    except Exception as e:
        log.error(f"FRED fetch failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Append to master
    new_master_row = pd.Series(index=master.columns, dtype=float)
    for col in etf_row.index:
        if col in master.columns:
            new_master_row[col] = etf_row[col]
    for col in fred_row.index:
        if col in master.columns:
            new_master_row[col] = fred_row[col]
    new_master_df = pd.DataFrame([new_master_row], index=[target_date])
    master = pd.concat([master, new_master_df]).sort_index()
    save_and_upload(master, cfg.FILE_MASTER, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 2. Append to etf_ohlcv.parquet
    ohlcv = load_parquet(cfg.FILE_ETF_OHLCV)
    new_ohlcv_row = etf_row.reindex(ohlcv.columns, fill_value=np.nan) if not ohlcv.empty else etf_row
    new_ohlcv_df = pd.DataFrame([new_ohlcv_row], index=[target_date])
    ohlcv = pd.concat([ohlcv, new_ohlcv_df]).sort_index()
    save_and_upload(ohlcv, cfg.FILE_ETF_OHLCV, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 3. Append to etf_returns.parquet
    ohlcv_full = load_parquet(cfg.FILE_ETF_OHLCV)  # includes new day
    if len(ohlcv_full) >= 2:
        prev = ohlcv_full.iloc[-2]
        curr = ohlcv_full.iloc[-1]
        ret_row = {}
        for t in cfg.ALL_TICKERS:
            close_col = f"{t}_Close"
            if close_col in ohlcv_full.columns and close_col in prev and close_col in curr:
                prev_c = prev[close_col]
                curr_c = curr[close_col]
                if not np.isnan(prev_c) and not np.isnan(curr_c):
                    ret = curr_c / prev_c - 1
                    ret_row[f"{t}_ret"] = ret
                    ret_row[f"{t}_logret"] = np.log(curr_c / prev_c)
                else:
                    ret_row[f"{t}_ret"] = np.nan
                    ret_row[f"{t}_logret"] = np.nan
        new_ret_df = pd.DataFrame([ret_row], index=[target_date])
        existing_ret = load_parquet(cfg.FILE_ETF_RETURNS)
        if not existing_ret.empty:
            new_ret_df = new_ret_df.reindex(existing_ret.columns, fill_value=np.nan)
        ret_df = pd.concat([existing_ret, new_ret_df]).sort_index()
        save_and_upload(ret_df, cfg.FILE_ETF_RETURNS, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 4. Recompute etf_vol.parquet (full recompute from updated returns)
    returns = load_parquet(cfg.FILE_ETF_RETURNS)
    if not returns.empty:
        vol = pd.DataFrame(index=returns.index)
        for t in cfg.ALL_TICKERS:
            ret_col = f"{t}_ret"
            if ret_col in returns.columns:
                # Annualised vol using rolling 21-day window
                vol[f"{t}_vol"] = returns[ret_col].rolling(21).std() * np.sqrt(252)
        vol = vol.dropna(how="all")
        save_and_upload(vol, cfg.FILE_ETF_VOL, f"Daily update: recomputed vol to {returns.index[-1].date()}")

    # ------------------------------------------------------------------
    # 5. Append to macro_fred.parquet
    macro_fred = load_parquet(cfg.FILE_MACRO_FRED)
    new_fred_row = fred_row.reindex(macro_fred.columns, fill_value=np.nan) if not macro_fred.empty else fred_row
    new_fred_df = pd.DataFrame([new_fred_row], index=[target_date])
    macro_fred = pd.concat([macro_fred, new_fred_df]).sort_index()
    save_and_upload(macro_fred, cfg.FILE_MACRO_FRED, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 6. Recompute macro_derived.parquet (full recompute from updated macro_fred)
    macro_fred_full = load_parquet(cfg.FILE_MACRO_FRED)
    if not macro_fred_full.empty:
        derived = du.compute_macro_derived(macro_fred_full)
        save_and_upload(derived, cfg.FILE_MACRO_DERIVED, f"Daily update: recomputed derived to {macro_fred_full.index[-1].date()}")

    # ------------------------------------------------------------------
    # 7. Update metadata.json
    update_metadata()

    log.info("All datasets updated successfully.")


if __name__ == "__main__":
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    update_master()
