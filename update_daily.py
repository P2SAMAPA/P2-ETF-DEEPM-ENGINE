#!/usr/bin/env python
# update_daily.py — P2-ETF-DEEPM-ENGINE
# Daily incremental update: fetches one new trading day's data (if available)
# and rebuilds all derived datasets (etf_returns, etf_vol, macro_derived, master)
# from the updated base files (etf_ohlcv, macro_fred).
# This ensures all files stay consistent with the master.

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from huggingface_hub import HfApi

import config as cfg
import data_utils as du

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def update_master() -> None:
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    # Load existing master to get the last date
    master = du.load_parquet(cfg.FILE_MASTER)
    if master.empty:
        log.error("No master dataset found. Run seed.py first.")
        sys.exit(1)
    last_date = master.index[-1]
    log.info(f"Last stored: {last_date.date()} | rows: {len(master)}")

    # Determine the next trading day to update
    # Use get_trading_days to find the first trading day after last_date
    trading_days = du.get_trading_days(
        start=last_date.strftime("%Y-%m-%d"),
        end=(datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    )
    candidates = [d for d in trading_days if d > last_date]
    if not candidates:
        log.info("No new trading day – nothing to update.")
        return

    target_date = candidates[0]
    if target_date.date() > datetime.now().date():
        log.info("Next trading day is in the future – nothing to update.")
        return

    log.info(f"Updating data for {target_date.date()}")

    # ------------------------------------------------------------------
    # 1. Fetch new OHLCV for the single day (using download_ohlcv for a narrow range)
    start = target_date.strftime("%Y-%m-%d")
    end   = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
    ohlcv_multi = du.download_ohlcv(cfg.ALL_TICKERS, start=start, end=end)
    if ohlcv_multi.empty:
        log.error("Failed to fetch OHLCV data for the target date.")
        sys.exit(1)
    ohlcv_flat = du.flatten_ohlcv(ohlcv_multi)

    # 2. Fetch new FRED macro data for the same date
    macro_row = du.download_fred(start=start, end=end)
    if macro_row.empty:
        log.warning("FRED data for target date is empty (may not be available yet). Using NaNs.")
        macro_row = pd.DataFrame(index=[target_date], columns=cfg.FRED_SERIES.keys(), dtype=float)

    # ------------------------------------------------------------------
    # 3. Append to existing OHLCV file
    ohlcv_existing = du.load_parquet(cfg.FILE_ETF_OHLCV)
    # Ensure new row has all columns (align)
    new_ohlcv = ohlcv_flat.reindex(ohlcv_existing.columns, fill_value=np.nan)
    ohlcv_updated = pd.concat([ohlcv_existing, new_ohlcv], axis=0).sort_index()
    du.upload_parquet(ohlcv_updated, cfg.FILE_ETF_OHLCV, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 4. Append to macro_fred file
    macro_existing = du.load_parquet(cfg.FILE_MACRO_FRED)
    new_macro = macro_row.reindex(macro_existing.columns, fill_value=np.nan)
    macro_updated = pd.concat([macro_existing, new_macro], axis=0).sort_index()
    du.upload_parquet(macro_updated, cfg.FILE_MACRO_FRED, f"Daily update: added {target_date.date()}")

    # ------------------------------------------------------------------
    # 5. Recompute returns from the updated OHLCV (full history)
    ohlcv_full = du.load_parquet(cfg.FILE_ETF_OHLCV)
    returns = du.compute_returns(ohlcv_full, cfg.ALL_TICKERS)
    du.upload_parquet(returns, cfg.FILE_ETF_RETURNS, f"Daily update: recomputed returns to {ohlcv_full.index[-1].date()}")

    # ------------------------------------------------------------------
    # 6. Recompute volatility from the updated returns (full history)
    if not returns.empty:
        vol = pd.DataFrame(index=returns.index)
        for t in cfg.ALL_TICKERS:
            ret_col = f"{t}_ret"
            if ret_col in returns.columns:
                vol[f"{t}_vol"] = returns[ret_col].rolling(21).std() * np.sqrt(252)
        vol = vol.dropna(how="all")
        du.upload_parquet(vol, cfg.FILE_ETF_VOL, f"Daily update: recomputed vol to {returns.index[-1].date()}")

    # ------------------------------------------------------------------
    # 7. Recompute derived macro from the updated macro_fred (full history)
    macro_full = du.load_parquet(cfg.FILE_MACRO_FRED)
    macro_derived = du.compute_macro_derived(macro_full)
    du.upload_parquet(macro_derived, cfg.FILE_MACRO_DERIVED, f"Daily update: recomputed derived to {macro_full.index[-1].date()}")

    # ------------------------------------------------------------------
    # 8. Rebuild master from the updated base files
    ohlcv_full = du.load_parquet(cfg.FILE_ETF_OHLCV)
    returns = du.load_parquet(cfg.FILE_ETF_RETURNS)
    macro_full = du.load_parquet(cfg.FILE_MACRO_FRED)
    macro_derived = du.load_parquet(cfg.FILE_MACRO_DERIVED)
    master_updated = du.build_master(ohlcv_full, returns, macro_full, macro_derived)
    du.upload_parquet(master_updated, cfg.FILE_MASTER, f"Daily update: rebuilt master to {master_updated.index[-1].date()}")

    # ------------------------------------------------------------------
    # 9. Update metadata.json
    metadata = {
        "last_updated":       datetime.utcnow().isoformat(),
        "last_trading_day":   str(master_updated.index[-1].date()),
        "n_trading_days":     len(master_updated),
    }
    du.upload_json(metadata, cfg.FILE_METADATA, f"Daily update: metadata to {master_updated.index[-1].date()}")

    log.info("All datasets updated successfully.")


if __name__ == "__main__":
    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set")
        sys.exit(1)
    if not cfg.FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        sys.exit(1)
    update_master()
