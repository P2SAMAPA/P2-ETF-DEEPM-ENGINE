# update_daily.py — Incremental daily data update
# Appends only the latest trading day to all 5 parquet files.
# Runs via GitHub Actions at 22:00 UTC Mon-Fri.
#
# Logic:
#   1. Load existing parquet files from HF
#   2. Determine last date already stored
#   3. If last stored date == last trading day → already up to date, exit
#   4. Download only the missing days
#   5. Append and re-upload all 5 files + metadata

import logging
import sys
from datetime import datetime

import pandas as pd

import config as cfg
import data_utils as du

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def update() -> None:
    today_str        = datetime.utcnow().strftime("%Y-%m-%d")
    last_trading_day = du.last_trading_day()

    logger.info("=" * 60)
    logger.info(f"P2-ETF-DEEPM DAILY UPDATE — {today_str}")
    logger.info(f"Last NYSE trading day: {last_trading_day}")
    logger.info("=" * 60)

    # ── Load existing data ────────────────────────────────────────────────────
    logger.info("Loading existing parquet files from HuggingFace...")
    try:
        ohlcv_flat    = du.load_parquet(cfg.FILE_ETF_OHLCV)
        returns       = du.load_parquet(cfg.FILE_ETF_RETURNS)
        macro         = du.load_parquet(cfg.FILE_MACRO_FRED)
        macro_derived = du.load_parquet(cfg.FILE_MACRO_DERIVED)
    except Exception as e:
        logger.error(f"Failed to load existing data: {e}")
        logger.error("Run seed.py first to initialise the dataset.")
        sys.exit(1)

    last_stored = ohlcv_flat.index[-1].strftime("%Y-%m-%d")
    logger.info(f"Last stored date : {last_stored}")
    logger.info(f"Last trading day : {last_trading_day}")

    if last_stored >= last_trading_day:
        logger.info("Dataset already up to date. Nothing to do.")
        return

    # ── Determine fetch window ─────────────────────────────────────────────────
    # Fetch from day after last stored to last trading day
    fetch_start = (pd.Timestamp(last_stored) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = last_trading_day
    logger.info(f"Fetching new data: {fetch_start} to {fetch_end}")

    # ── 1. New OHLCV ──────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Downloading new OHLCV...")
    new_ohlcv_multi = du.download_ohlcv(cfg.ALL_TICKERS, start=fetch_start, end=fetch_end)
    new_ohlcv_flat  = du.flatten_ohlcv(new_ohlcv_multi)

    if new_ohlcv_flat.empty:
        logger.warning("No new OHLCV data returned. Market may be closed.")
        return

    # Remove any dates already in existing data (safety dedup)
    new_ohlcv_flat = new_ohlcv_flat[~new_ohlcv_flat.index.isin(ohlcv_flat.index)]
    if new_ohlcv_flat.empty:
        logger.info("No new trading days after deduplication. Already up to date.")
        return

    logger.info(f"New trading days: {len(new_ohlcv_flat)} | {list(new_ohlcv_flat.index.date)}")

    ohlcv_flat_updated = pd.concat([ohlcv_flat, new_ohlcv_flat]).sort_index()
    du.upload_parquet(
        ohlcv_flat_updated,
        cfg.FILE_ETF_OHLCV,
        f"[update] ETF OHLCV +{len(new_ohlcv_flat)}d to {fetch_end}"
    )

    # ── 2. New returns ────────────────────────────────────────────────────────
    logger.info("Step 2/5 — Computing new returns...")
    # Need one extra prior day for pct_change continuity
    prior_close = ohlcv_flat.iloc[[-1]]
    ohlcv_for_returns = pd.concat([prior_close, new_ohlcv_flat])
    new_returns = du.compute_returns(ohlcv_for_returns, cfg.ALL_TICKERS)
    new_returns = new_returns[new_returns.index.isin(new_ohlcv_flat.index)]

    returns_updated = pd.concat([returns, new_returns]).sort_index()
    du.upload_parquet(
        returns_updated,
        cfg.FILE_ETF_RETURNS,
        f"[update] ETF returns +{len(new_returns)}d to {fetch_end}"
    )

    # ── 3. New FRED macro ─────────────────────────────────────────────────────
    logger.info("Step 3/5 — Downloading new FRED macro...")
    # Fetch a few extra days back to catch FRED publication lags
    fred_start = (pd.Timestamp(last_stored) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    new_macro_raw = du.download_fred(start=fred_start, end=fetch_end)

    # Keep only new dates
    new_macro = new_macro_raw[~new_macro_raw.index.isin(macro.index)]
    macro_updated = pd.concat([macro, new_macro]).sort_index()
    # Re-ffill to catch any gaps
    macro_updated = macro_updated.ffill()

    du.upload_parquet(
        macro_updated,
        cfg.FILE_MACRO_FRED,
        f"[update] macro FRED +{len(new_macro)}d to {fetch_end}"
    )

    # ── 4. Re-derive macro features ───────────────────────────────────────────
    # Always recompute on full history (rolling z-scores need full window)
    logger.info("Step 4/5 — Recomputing derived macro features...")
    macro_derived_updated = du.compute_macro_derived(macro_updated)
    du.upload_parquet(
        macro_derived_updated,
        cfg.FILE_MACRO_DERIVED,
        f"[update] macro derived to {fetch_end}"
    )

    # ── 5. Rebuild master ─────────────────────────────────────────────────────
    logger.info("Step 5/5 — Rebuilding master aligned file...")
    master_updated = du.build_master(
        ohlcv_flat_updated,
        returns_updated,
        macro_updated,
        macro_derived_updated,
    )
    du.upload_parquet(
        master_updated,
        cfg.FILE_MASTER,
        f"[update] master to {fetch_end}"
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = du.load_metadata()
    metadata.update({
        "last_updated":       datetime.utcnow().isoformat(),
        "last_trading_day":   str(master_updated.index[-1].date()),
        "n_trading_days":     len(master_updated),
        "master_shape":       list(master_updated.shape),
        "last_update_added":  len(new_ohlcv_flat),
    })
    du.upload_json(metadata, cfg.FILE_METADATA, f"[update] metadata to {fetch_end}")

    logger.info("=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info(f"  New days added   : {len(new_ohlcv_flat)}")
    logger.info(f"  Latest date      : {master_updated.index[-1].date()}")
    logger.info(f"  Total days       : {len(master_updated)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        update()
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        raise
