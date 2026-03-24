#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE (PRODUCTION-HARDENED)
# Daily update: fetch new trading day data (if any) and regenerate all derived files.

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random

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


def validate_ohlcv_structure(df, expected_tickers, target_date):
    """Validate that fetched OHLCV has expected structure."""
    if df is None or df.empty:
        return False, "Empty DataFrame"
    
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, f"Index is {type(df.index)}, expected DatetimeIndex"
    
    df = safe_to_datetime_index(df)
    
    if target_date not in df.index:
        return False, f"Target date {target_date.date()} not in index (have {df.index[0].date()} to {df.index[-1].date()})"
    
    expected_cols = {f"{t}_{field}" for t in expected_tickers for field in ['Open', 'High', 'Low', 'Close', 'Volume']}
    missing = expected_cols - set(df.columns)
    if missing:
        log.warning(f"Missing OHLCV columns: {missing}")
    
    # Check for all-NaN rows (common yfinance failure mode)
    nan_rows = df.isna().all(axis=1).sum()
    if nan_rows > 0:
        return False, f"DataFrame has {nan_rows} all-NaN rows"
    
    return True, "OK"


def fetch_ohlcv_robust(tickers, start, end, target_date, max_retries=3):
    """Fetch OHLCV with strict validation."""
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Fetching OHLCV for {start} to {end} (attempt {attempt})")
            data = du.download_ohlcv(tickers, start=start, end=end)
            
            # Validate structure
            valid, msg = validate_ohlcv_structure(data, tickers, target_date)
            if valid:
                log.info(f"Successfully fetched OHLCV with shape {data.shape}")
                return data
            else:
                log.warning(f"OHLCV validation failed: {msg}")
                if attempt < max_retries:
                    time.sleep(2**attempt + random.uniform(0, 1))
                else:
                    return None
                    
        except Exception as e:
            log.warning(f"OHLCV fetch exception (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(2**attempt + random.uniform(0, 1))
            else:
                return None
    return None


def fetch_fred_robust(target_date, max_retries=3):
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
            if attempt < max_retries:
                time.sleep(2**attempt)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()


def atomic_update(target_date, ohlcv, macro):
    """
    Perform atomic update: fetch new data and return updated DataFrames.
    Does NOT modify original DataFrames until all data is validated.
    """
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Fetch OHLCV
    ohlcv_multi = fetch_ohlcv_robust(cfg.ALL_TICKERS, start, end, target_date)
    if ohlcv_multi is None:
        raise ValueError(f"Failed to fetch OHLCV for {target_date.date()}")
    
    new_ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)
    new_ohlcv_flat = safe_to_datetime_index(new_ohlcv_flat)
    
    # Validate exact date present
    if target_date not in new_ohlcv_flat.index:
        raise ValueError(f"Date {target_date.date()} not in flattened OHLCV")
    
    new_ohlcv_row = new_ohlcv_flat.loc[[target_date]]
    
    # Fetch Macro
    new_macro = fetch_fred_robust(target_date)
    if new_macro.empty:
        log.warning(f"No FRED data for {target_date.date()}, creating NaN row")
        new_macro = pd.DataFrame(index=[target_date], columns=cfg.FRED_SERIES.keys(), dtype=float)
    
    # Align columns
    missing_ohlcv_cols = set(ohlcv.columns) - set(new_ohlcv_row.columns)
    if missing_ohlcv_cols:
        log.warning(f"New OHLCV missing columns: {missing_ohlcv_cols}")
    
    new_ohlcv_row = new_ohlcv_row.reindex(columns=ohlcv.columns)
    new_macro = new_macro.reindex(columns=macro.columns)
    
    # Check for existing date (shouldn't happen if logic is correct, but safety check)
    if target_date in ohlcv.index:
        log.warning(f"Overwriting existing OHLCV date {target_date.date()}")
    
    if target_date in macro.index:
        log.warning(f"Overwriting existing Macro date {target_date.date()}")
    
    # Create new DataFrames (don't modify in-place until sure)
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
    
    return errors


def update_master():
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # 1. Load current base files
    try:
        ohlcv = du.load_parquet(cfg.FILE_ETF_OHLCV)
        macro = du.load_parquet(cfg.FILE_MACRO_FRED)
        log.info(f"Loaded base files: OHLCV {ohlcv.shape}, Macro {macro.shape}")
    except Exception as e:
        log.error(f"Failed to load base files: {e}")
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
            ohlcv_updated, macro_updated = atomic_update(target_date, ohlcv, macro)
            update_performed = True
            
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
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    
    update_master()
