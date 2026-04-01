#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE (with debug mode)
# Daily update: fetch new trading day data (if any) and regenerate all derived files.

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import requests
from io import BytesIO

import config as cfg
import data_utils as du

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("update_daily.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def safe_to_datetime_index(df):
    """Ensure index is timezone-naive datetime64[ns]."""
    if df is None or df.empty:
        return df
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df


def fetch_ohlcv_stooq(ticker, start, end):
    """Fetch OHLCV from Stooq as fallback."""
    stooq_symbol = ticker.lower() + '.us'
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
        if df.empty:
            return None
        df = df.sort_index()
        mask = (df.index >= start) & (df.index <= end)
        df = df.loc[mask]
        if df.empty:
            return None
        # Rename columns to match yfinance format
        df.columns = [c.lower() for c in df.columns]
        # Stooq returns columns: Open, High, Low, Close, Volume
        # We need to prefix with ticker
        df = df.rename(columns={col: f"{ticker}_{col}" for col in df.columns})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        log.debug(f"Stooq failed for {ticker}: {e}")
        return None


def fetch_ohlcv_robust(tickers, start, end, target_date, max_retries=3, debug=False):
    """
    Fetch OHLCV for all tickers, using yfinance first, then Stooq fallback per ticker.
    Returns a MultiIndex DataFrame (ticker, field) or None.
    """
    # First try yfinance for all tickers
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Fetching OHLCV via yfinance for {start} to {end} (attempt {attempt})")
            data = du.download_ohlcv(tickers, start=start, end=end)
            if data is not None and not data.empty:
                data = safe_to_datetime_index(data)
                if target_date in data.index:
                    log.info(f"yfinance success: found {target_date.date()}")
                    return data
                else:
                    log.warning(f"yfinance returned data but missing target date {target_date.date()}")
            else:
                log.warning(f"yfinance returned empty")
        except Exception as e:
            log.warning(f"yfinance exception (attempt {attempt}): {e}")
        
        if attempt < max_retries and not debug:
            time.sleep(2**attempt + random.uniform(0, 1))
    
    # yfinance failed; try Stooq per ticker
    log.info("yfinance failed, trying Stooq fallback per ticker...")
    frames = []
    for ticker in tickers:
        stooq_df = fetch_ohlcv_stooq(ticker, start, end)
        if stooq_df is not None:
            frames.append(stooq_df)
            log.info(f"Stooq success for {ticker}")
        else:
            log.warning(f"No Stooq data for {ticker}")
        if not debug:
            time.sleep(random.uniform(0.5, 1.5))
    
    if not frames:
        log.error("All fetch methods failed")
        return None
    
    # Combine and convert to MultiIndex
    combined = pd.concat(frames, axis=1)
    # Convert to MultiIndex (ticker, field)
    multi_cols = []
    for col in combined.columns:
        ticker, field = col.split('_', 1)
        multi_cols.append((ticker, field))
    combined.columns = pd.MultiIndex.from_tuples(multi_cols)
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    return combined


def fetch_fred_robust(target_date, max_retries=3, debug=False):
    """
    Fetch FRED data robustly.
    FRED requires start < end, so we fetch a window and extract target date.
    """
    target_str = target_date.strftime("%Y-%m-%d")
    
    # Fetch 5-day window to ensure we get the target date
    start = (target_date - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Exclusive end
    
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Fetching FRED data for window {start} to {end}")
            data = du.download_fred(start=start, end=end)
            
            if data is None or data.empty:
                log.warning(f"FRED returned empty (attempt {attempt})")
                if not debug:
                    time.sleep(2**attempt)
                continue
            
            data = safe_to_datetime_index(data)
            
            # Check for target date specifically
            if target_date in data.index:
                result = data.loc[[target_date]]
                log.info(f"Found FRED data for {target_str}")
                return result
            else:
                # If target is missing, use most recent available (forward fill logic)
                available = data.index[data.index <= target_date]
                if len(available) > 0:
                    last_avail = available[-1]
                    result = data.loc[[last_avail]].copy()
                    result.index = [target_date]  # Reindex to target date
                    log.warning(f"FRED data for {target_str} missing, using {last_avail.date()} (forward fill)")
                    return result
                else:
                    log.warning(f"No FRED data available before {target_str}")
                    return pd.DataFrame()
                    
        except Exception as e:
            log.error(f"FRED fetch exception (attempt {attempt}): {e}")
            if attempt < max_retries and not debug:
                time.sleep(2**attempt)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()


def atomic_update(target_date, ohlcv, macro, debug=False):
    """
    Perform atomic update: fetch new data and return updated DataFrames.
    Does NOT modify original DataFrames until all data is validated.
    """
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Fetch OHLCV with fallback
    ohlcv_multi = fetch_ohlcv_robust(cfg.ALL_TICKERS, start, end, target_date, debug=debug)
    if ohlcv_multi is None:
        raise ValueError(f"Failed to fetch OHLCV for {target_date.date()}")
    
    # Ensure target date exists
    ohlcv_multi = safe_to_datetime_index(ohlcv_multi)
    if target_date not in ohlcv_multi.index:
        raise ValueError(f"Date {target_date.date()} not in fetched OHLCV")
    
    # Flatten to flat columns
    new_ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)
    new_ohlcv_flat = safe_to_datetime_index(new_ohlcv_flat)
    
    # Reindex to match existing columns (missing columns become NaN)
    new_ohlcv_row = new_ohlcv_flat.loc[[target_date]]
    new_ohlcv_row = new_ohlcv_row.reindex(columns=ohlcv.columns)
    
    # Fetch Macro
    new_macro = fetch_fred_robust(target_date, debug=debug)
    if new_macro.empty:
        log.warning(f"No FRED data for {target_date.date()}, creating NaN row")
        new_macro = pd.DataFrame(index=[target_date], columns=macro.columns, dtype=float)
    else:
        new_macro = new_macro.reindex(columns=macro.columns)
    
    # Combine
    ohlcv_new = pd.concat([ohlcv, new_ohlcv_row]).sort_index()
    macro_new = pd.concat([macro, new_macro]).sort_index()
    
    # Final validation
    assert target_date in ohlcv_new.index, "OHLCV concat failed"
    assert target_date in macro_new.index, "Macro concat failed"
    assert not ohlcv_new.index.duplicated().any(), "Duplicate dates in OHLCV!"
    assert not macro_new.index.duplicated().any(), "Duplicate dates in Macro!"
    
    return ohlcv_new, macro_new


def validate_derived_data(returns, vol, macro_derived, master):
    """Validate computed derived data."""
    errors = []
    
    if returns.empty:
        errors.append("Returns DataFrame is empty")
    elif returns.isna().all().all():
        errors.append("Returns contains only NaN")
    
    if vol is not None and not vol.empty:
        recent_vol = vol.iloc[-5:].isna().all()
        if recent_vol.any():
            cols = recent_vol[recent_vol].index.tolist()
            log.warning(f"Recent volatility is NaN for: {cols}")
    
    if master.empty:
        errors.append("Master DataFrame is empty")
    
    # CRITICAL: Check for minimum history to prevent data loss
    if len(master) < 1000:
        errors.append(f"Master dataset suspiciously small: {len(master)} rows (expected >1000). Possible data loss!")
    
    return errors


def update_master(debug=False):
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if debug:
        log.info("*** DEBUG MODE – NO UPLOADS ***")
    log.info("=" * 60)

    # 1. Load current base files
    try:
        ohlcv = du.load_parquet(cfg.FILE_ETF_OHLCV)
        macro = du.load_parquet(cfg.FILE_MACRO_FRED)
        log.info(f"Loaded base files: OHLCV {ohlcv.shape}, Macro {macro.shape}")
    except Exception as e:
        log.error(f"Failed to load base files: {e}")
        sys.exit(1)

    # CRITICAL: Validate base files have sufficient history
    if len(ohlcv) < 1000:
        log.error(f"CRITICAL: Base OHLCV file has only {len(ohlcv)} rows. Possible corruption or missing data.")
        log.error("Aborting update to prevent data loss. Please re-seed the dataset.")
        sys.exit(1)
    
    if len(macro) < 1000:
        log.error(f"CRITICAL: Base Macro file has only {len(macro)} rows. Possible corruption or missing data.")
        log.error("Aborting update to prevent data loss. Please re-seed the dataset.")
        sys.exit(1)

    if ohlcv.empty or macro.empty:
        log.error("Base files empty. Run seed.py first.")
        sys.exit(1)

    # Normalize indices
    ohlcv = safe_to_datetime_index(ohlcv)
    macro = safe_to_datetime_index(macro)
    
    # Sort
    ohlcv = ohlcv.sort_index()
    macro = macro.sort_index()

    last_date = ohlcv.index[-1]
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    log.info(f"Last stored date: {last_date.date()}")
    log.info(f"Today: {today.date()}")

    # 2. Determine if update needed
    try:
        trading_days = du.get_trading_days(
            start=last_date.strftime("%Y-%m-%d"),
            end=(today + timedelta(days=10)).strftime("%Y-%m-%d")
        )
        trading_days = pd.to_datetime(trading_days)
        future_days = trading_days[trading_days > last_date]
    except Exception as e:
        log.error(f"Failed to get trading days: {e}")
        sys.exit(1)

    ohlcv_updated = ohlcv.copy()
    macro_updated = macro.copy()
    update_performed = False

    # Check for new trading day
    if len(future_days) > 0 and future_days[0].date() <= today.date():
        target_date = future_days[0]
        log.info(f"New trading day detected: {target_date.date()}")
        
        try:
            if debug:
                log.info("DEBUG: Simulating atomic update (will not upload)")
                ohlcv_updated, macro_updated = atomic_update(target_date, ohlcv, macro, debug=True)
                update_performed = True
                # FIXED: Use the returned DataFrames instead of undefined variables
                log.info(f"Would upload OHLCV row for {target_date.date()}")
                log.info(f"OHLCV shape after update: {ohlcv_updated.shape}")
                log.info(f"Macro shape after update: {macro_updated.shape}")
            else:
                ohlcv_updated, macro_updated = atomic_update(target_date, ohlcv, macro)
                update_performed = True
                
                # CRITICAL: Validate updated data before upload
                if len(ohlcv_updated) < len(ohlcv) + 1:
                    log.error(f"CRITICAL: OHLCV rows decreased from {len(ohlcv)} to {len(ohlcv_updated)}. Data loss detected!")
                    sys.exit(1)
                
                if len(macro_updated) < len(macro) + 1:
                    log.error(f"CRITICAL: Macro rows decreased from {len(macro)} to {len(macro_updated)}. Data loss detected!")
                    sys.exit(1)
                
                # Upload base files only after successful validation
                log.info("Uploading updated base files...")
                du.upload_parquet(ohlcv_updated, cfg.FILE_ETF_OHLCV, 
                                f"Daily update: added {target_date.date()}")
                du.upload_parquet(macro_updated, cfg.FILE_MACRO_FRED, 
                                f"Daily update: added {target_date.date()}")
                log.info("Base files uploaded successfully")
            
        except Exception as e:
            log.error(f"Atomic update failed: {e}")
            log.info("Reverting to original files (no upload performed)")
            update_performed = False
    else:
        log.info("No new trading day to add")

    # 3. Recompute derived files (always from current/updated base)
    log.info("Recomputing derived files...")
    
    try:
        returns = du.compute_returns(ohlcv_updated, cfg.ALL_TICKERS)
        if returns.empty:
            raise ValueError("Returns computation returned empty")
        
        # Volatility with min_periods to avoid losing data on new days
        vol = pd.DataFrame(index=returns.index)
        for t in cfg.ALL_TICKERS:
            ret_col = f"{t}_ret"
            if ret_col in returns.columns:
                # min_periods=1 ensures we calculate even with limited history
                vol[f"{t}_vol"] = returns[ret_col].rolling(21, min_periods=1).std() * np.sqrt(252)
        
        vol = vol.dropna(how="all")
        
        macro_derived = du.compute_macro_derived(macro_updated)
        if macro_derived.empty:
            raise ValueError("Macro derived computation returned empty")
        
        master = du.build_master(ohlcv_updated, returns, macro_updated, macro_derived)
        if master.empty:
            raise ValueError("Master build returned empty")
        
        # Validate
        errors = validate_derived_data(returns, vol, macro_derived, master)
        if errors:
            for err in errors:
                log.error(f"Validation error: {err}")
            raise ValueError("Derived data validation failed")
        
        if debug:
            log.info("DEBUG: Derived files computed, would upload:")
            log.info(f"  returns: {returns.shape}, last date {returns.index[-1].date()}")
            log.info(f"  vol: {vol.shape}, last date {vol.index[-1].date()}")
            log.info(f"  macro_derived: {macro_derived.shape}, last date {macro_derived.index[-1].date()}")
            log.info(f"  master: {master.shape}, last date {master.index[-1].date()}")
            log.info(f"  metadata: last_trading_day={master.index[-1].date()}")
        else:
            # CRITICAL: Final sanity check before uploading master
            if len(master) < 1000:
                log.error(f"CRITICAL: Final master dataset has only {len(master)} rows. Aborting upload to prevent data loss!")
                sys.exit(1)
            
            # Upload derived files
            du.upload_parquet(returns, cfg.FILE_ETF_RETURNS, 
                             f"Daily update: returns to {returns.index[-1].date()}")
            du.upload_parquet(vol, cfg.FILE_ETF_VOL, 
                             f"Daily update: vol to {vol.index[-1].date()}")
            du.upload_parquet(macro_derived, cfg.FILE_MACRO_DERIVED, 
                             f"Daily update: derived macro to {macro_derived.index[-1].date()}")
            du.upload_parquet(master, cfg.FILE_MASTER, 
                             f"Daily update: master to {master.index[-1].date()}")
            
            # Metadata
            metadata = {
                "last_updated": datetime.utcnow().isoformat(),
                "last_trading_day": str(master.index[-1].date()),
                "n_trading_days": len(master),
                "update_type": "incremental" if update_performed else "derived_only",
                "last_run_status": "success"
            }
            du.upload_json(metadata, cfg.FILE_METADATA, 
                          f"Daily update: metadata to {master.index[-1].date()}")
        
        log.info("=" * 60)
        log.info("SUCCESS: All datasets updated")
        log.info(f"Master: {master.shape}, Last date: {master.index[-1].date()}")
        log.info(f"Update type: {'New data added' if update_performed else 'Derived only'}")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"Derived computation failed: {e}")
        log.error("Base files may be updated but derived files are inconsistent!")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode – print what would be updated but do not upload")
    args = parser.parse_args()

    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    
    update_master(debug=args.debug)
