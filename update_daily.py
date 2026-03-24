#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE
# Daily incremental update: fetches OHLCV for all tickers and macro data
# for the last trading day, appends to all derived datasets.

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
from features import compute_macro_features   # for derived macro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
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
            # Convert to flat row
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


def fetch_fred_data(target_date: pd.Timestamp) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    data = {}
    for col_name, series_code in cfg.FRED_SERIES.items():
        params = {
            "series_id": series_code,
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
        time.sleep(0.3)
    return pd.Series(data, name=target_date)


def load_or_create(path: str, index_col: str = None) -> pd.DataFrame:
    try:
        local = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO, filename=path,
            repo_type="dataset", token=cfg.HF_TOKEN,
            local_dir="/tmp", local_dir_use_symlinks=False,
        )
        df = pd.read_parquet(local)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
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


def compute_returns(ohlcv: pd.Series, existing_returns: pd.DataFrame) -> pd.Series:
    """Compute simple returns for the new day using the last close and new close."""
    new_close = {}
    for ticker in cfg.ALL_TICKERS:
        close_col = f"{ticker}_Close"
        if close_col in ohlcv.index:
            new_close[ticker] = ohlcv[close_col]
        else:
            new_close[ticker] = np.nan

    if existing_returns.empty:
        # No previous returns, cannot compute
        return pd.Series(index=[f"{t}_Ret" for t in cfg.ALL_TICKERS], dtype=float)

    last_date = existing_returns.index[-1]
    last_close = {}
    for ticker in cfg.ALL_TICKERS:
        close_col = f"{ticker}_Close"
        if close_col in existing_returns.columns:
            # But existing_returns probably does not have close. We need the last close from OHLCV file.
            # Better to load OHLCV file separately to get previous close.
            pass
    # To avoid complexity, we'll compute returns using the OHLCV file that we have already appended.
    # We can load the full OHLCV file (which now includes the new day) and compute returns for the new day only.
    # That's safer.
    return pd.Series()


def update_master() -> None:
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # Load existing master
    master = load_or_create(cfg.FILE_MASTER)
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
        fred_row = fetch_fred_data(target_date)
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
    ohlcv = load_or_create(cfg.FILE_ETF_OHLCV)
    # Ensure OHLCV columns match what we have
    new_ohlcv_row = etf_row.reindex(ohlcv.columns, fill_value=np.nan) if not ohlcv.empty else etf_row
    new_ohlcv_df = pd.DataFrame([new_ohlcv_row], index=[target_date])
    ohlcv = pd.concat([ohlcv, new_ohlcv_df]).sort_index()
    save_and_upload(ohlcv, cfg.FILE_ETF_OHLCV, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 3. Append to etf_returns.parquet (compute simple and log returns from OHLCV)
    # We need previous close to compute returns. Use the OHLCV file we just updated.
    # Re‑load it (or use the one in memory) to get the previous day's close.
    # For simplicity, we'll load the just‑saved file, but we can also compute from the existing returns.
    ohlcv = load_or_create(cfg.FILE_ETF_OHLCV)  # includes the new row now
    if len(ohlcv) >= 2:
        prev = ohlcv.iloc[-2]
        curr = ohlcv.iloc[-1]
        returns = {}
        for t in cfg.ALL_TICKERS:
            close_col = f"{t}_Close"
            if close_col in ohlcv.columns and close_col in prev and close_col in curr:
                prev_c = prev[close_col]
                curr_c = curr[close_col]
                if not np.isnan(prev_c) and not np.isnan(curr_c):
                    ret = curr_c / prev_c - 1
                    returns[f"{t}_ret"] = ret
                    returns[f"{t}_logret"] = np.log(curr_c / prev_c)
                else:
                    returns[f"{t}_ret"] = np.nan
                    returns[f"{t}_logret"] = np.nan
        new_ret_row = pd.Series(returns, name=target_date)
        existing_ret = load_or_create(cfg.FILE_ETF_RETURNS)
        # Ensure columns match
        if not existing_ret.empty:
            new_ret_row = new_ret_row.reindex(existing_ret.columns, fill_value=np.nan)
        new_ret_df = pd.DataFrame([new_ret_row], index=[target_date])
        ret_df = pd.concat([existing_ret, new_ret_df]).sort_index()
        save_and_upload(ret_df, cfg.FILE_ETF_RETURNS, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 4. Append to etf_vol.parquet (rolling 21d volatility)
    # Use the returns file (which now includes the new day) to compute the 21d vol for the new day.
    # The volatility for the new day uses the last 21 returns.
    ret_df = load_or_create(cfg.FILE_ETF_RETURNS)
    if not ret_df.empty and len(ret_df) >= 21:
        last_21 = ret_df.iloc[-21:]
        vol = {}
        for t in cfg.ALL_TICKERS:
            col = f"{t}_ret"
            if col in last_21.columns:
                daily_std = last_21[col].std()
                vol[f"{t}_vol"] = daily_std * np.sqrt(252) if not np.isnan(daily_std) else np.nan
        new_vol_row = pd.Series(vol, name=target_date)
        existing_vol = load_or_create(cfg.FILE_ETF_VOL)
        if not existing_vol.empty:
            new_vol_row = new_vol_row.reindex(existing_vol.columns, fill_value=np.nan)
        new_vol_df = pd.DataFrame([new_vol_row], index=[target_date])
        vol_df = pd.concat([existing_vol, new_vol_df]).sort_index()
        save_and_upload(vol_df, cfg.FILE_ETF_VOL, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 5. Append to macro_fred.parquet
    macro_fred = load_or_create(cfg.FILE_MACRO_FRED)
    new_fred_row = fred_row.reindex(macro_fred.columns, fill_value=np.nan) if not macro_fred.empty else fred_row
    new_fred_df = pd.DataFrame([new_fred_row], index=[target_date])
    macro_fred = pd.concat([macro_fred, new_fred_df]).sort_index()
    save_and_upload(macro_fred, cfg.FILE_MACRO_FRED, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 6. Append to macro_derived.parquet
    # We need the full macro_fred (including new day) to compute derived features.
    macro_fred = load_or_create(cfg.FILE_MACRO_FRED)  # fresh load
    if not macro_fred.empty:
        # Compute derived features only for the new date
        derived_full = compute_macro_features(macro_fred)  # this recomputes all rows
        # But we only want the last row (new date) to append
        new_derived = derived_full.iloc[[-1]].copy()
        existing_derived = load_or_create(cfg.FILE_MACRO_DERIVED)
        if not existing_derived.empty:
            # Ensure columns align
            new_derived = new_derived.reindex(existing_derived.columns, fill_value=np.nan)
        derived_df = pd.concat([existing_derived, new_derived]).sort_index()
        save_and_upload(derived_df, cfg.FILE_MACRO_DERIVED, f"Daily update: added {target_date.date()}")

    log.info("All datasets updated successfully.")


if __name__ == "__main__":
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    update_master()
