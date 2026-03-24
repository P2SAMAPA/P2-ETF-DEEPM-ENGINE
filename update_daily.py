#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE
# Daily update: fetch new trading day data (if any) and regenerate all derived files.

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

import config as cfg
import data_utils as du

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def update_master() -> None:
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # 1. Load current base files (OHLCV and macro)
    ohlcv = du.load_parquet(cfg.FILE_ETF_OHLCV)
    macro = du.load_parquet(cfg.FILE_MACRO_FRED)

    if ohlcv.empty or macro.empty:
        log.error("Base files missing. Run seed.py first.")
        sys.exit(1)

    last_date = ohlcv.index[-1]
    log.info(f"Last stored OHLCV date: {last_date.date()}")

    # 2. Determine next trading day
    trading_days = du.get_trading_days(
        start=last_date.strftime("%Y-%m-%d"),
        end=(datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    )
    candidates = [d for d in trading_days if d > last_date]
    if candidates and candidates[0].date() <= datetime.now().date():
        target_date = candidates[0]
        log.info(f"New trading day: {target_date.date()}. Fetching data...")

        # Fetch new OHLCV for the single day
        start = target_date.strftime("%Y-%m-%d")
        end   = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        ohlcv_multi = du.download_ohlcv(cfg.ALL_TICKERS, start=start, end=end)
        if ohlcv_multi.empty:
            log.error("Failed to fetch OHLCV data.")
            sys.exit(1)
        new_ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)

        # Fetch new FRED macro data for the same date
        new_macro = du.download_fred(start=start, end=end)
        if new_macro.empty:
            log.warning("FRED data not available; using NaNs.")
            new_macro = pd.DataFrame(index=[target_date], columns=cfg.FRED_SERIES.keys(), dtype=float)

        # Append to base files
        # Align columns with existing files
        new_ohlcv_flat = new_ohlcv_flat.reindex(ohlcv.columns, fill_value=np.nan)
        new_macro = new_macro.reindex(macro.columns, fill_value=np.nan)

        ohlcv = pd.concat([ohlcv, new_ohlcv_flat], axis=0).sort_index()
        macro = pd.concat([macro, new_macro], axis=0).sort_index()

        # Upload base files
        du.upload_parquet(ohlcv, cfg.FILE_ETF_OHLCV, f"Daily update: added {target_date.date()}")
        du.upload_parquet(macro, cfg.FILE_MACRO_FRED, f"Daily update: added {target_date.date()}")
        log.info("Base files updated and uploaded.")
    else:
        log.info("No new trading day – using existing base files.")

    # 3. Always recompute derived files from the (potentially updated) base files
    log.info("Recomputing derived files...")

    # Returns
    returns = du.compute_returns(ohlcv, cfg.ALL_TICKERS)
    du.upload_parquet(returns, cfg.FILE_ETF_RETURNS, f"Daily update: returns to {returns.index[-1].date()}")

    # Volatility
    if not returns.empty:
        vol = pd.DataFrame(index=returns.index)
        for t in cfg.ALL_TICKERS:
            ret_col = f"{t}_ret"
            if ret_col in returns.columns:
                vol[f"{t}_vol"] = returns[ret_col].rolling(21).std() * np.sqrt(252)
        vol = vol.dropna(how="all")
        du.upload_parquet(vol, cfg.FILE_ETF_VOL, f"Daily update: vol to {vol.index[-1].date()}")

    # Derived macro
    macro_derived = du.compute_macro_derived(macro)
    du.upload_parquet(macro_derived, cfg.FILE_MACRO_DERIVED, f"Daily update: derived macro to {macro_derived.index[-1].date()}")

    # Master
    master = du.build_master(ohlcv, returns, macro, macro_derived)
    du.upload_parquet(master, cfg.FILE_MASTER, f"Daily update: master to {master.index[-1].date()}")

    # Metadata
    metadata = {
        "last_updated":       datetime.utcnow().isoformat(),
        "last_trading_day":   str(master.index[-1].date()),
        "n_trading_days":     len(master),
    }
    du.upload_json(metadata, cfg.FILE_METADATA, f"Daily update: metadata to {master.index[-1].date()}")

    log.info("All datasets updated successfully.")


if __name__ == "__main__":
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    update_master()
