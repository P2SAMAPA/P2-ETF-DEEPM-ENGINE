# seed.py — One-time full history seed to HuggingFace dataset
# Run once to initialise p2-etf-deepm-data with complete history from 2008.
#
# Usage:
#   python seed.py
#   python seed.py --start 2008-01-01 --end 2024-12-31
#
# GitHub Actions: trigger manually via workflow_dispatch

import argparse
import json
import logging
import sys
from datetime import datetime

import config as cfg
import data_utils as du

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def seed(start: str = None, end: str = None) -> None:
    start = start or cfg.DATA_START
    end   = end   or datetime.today().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"P2-ETF-DEEPM SEED — {start} to {end}")
    logger.info(f"Tickers : {cfg.ALL_TICKERS}")
    logger.info(f"FRED    : {list(cfg.FRED_SERIES.keys())}")
    logger.info("=" * 60)

    # ── 1. ETF OHLCV ──────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Downloading ETF OHLCV...")
    ohlcv_multi = du.download_ohlcv(cfg.ALL_TICKERS, start=start, end=end)
    ohlcv_flat  = du.flatten_ohlcv(ohlcv_multi)
    du.upload_parquet(ohlcv_flat, cfg.FILE_ETF_OHLCV, f"[seed] ETF OHLCV {start}:{end}")
    logger.info(f"etf_ohlcv.parquet uploaded — {ohlcv_flat.shape}")

    # ── 2. ETF returns ────────────────────────────────────────────────────────
    logger.info("Step 2/5 — Computing ETF returns...")
    returns = du.compute_returns(ohlcv_flat, cfg.ALL_TICKERS)
    du.upload_parquet(returns, cfg.FILE_ETF_RETURNS, f"[seed] ETF returns {start}:{end}")
    logger.info(f"etf_returns.parquet uploaded — {returns.shape}")

    # ── 3. FRED macro ─────────────────────────────────────────────────────────
    logger.info("Step 3/5 — Downloading FRED macro...")
    macro = du.download_fred(start=start, end=end)
    du.upload_parquet(macro, cfg.FILE_MACRO_FRED, f"[seed] macro FRED {start}:{end}")
    logger.info(f"macro_fred.parquet uploaded — {macro.shape}")

    # ── 4. Derived macro features ─────────────────────────────────────────────
    logger.info("Step 4/5 — Computing derived macro features...")
    macro_derived = du.compute_macro_derived(macro)
    du.upload_parquet(macro_derived, cfg.FILE_MACRO_DERIVED, f"[seed] macro derived {start}:{end}")
    logger.info(f"macro_derived.parquet uploaded — {macro_derived.shape}")

    # ── 5. Master aligned file ────────────────────────────────────────────────
    logger.info("Step 5/5 — Building master aligned file...")
    master = du.build_master(ohlcv_flat, returns, macro, macro_derived)
    du.upload_parquet(master, cfg.FILE_MASTER, f"[seed] master {start}:{end}")
    logger.info(f"master.parquet uploaded — {master.shape}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "last_updated":       datetime.utcnow().isoformat(),
        "last_trading_day":   str(master.index[-1].date()),
        "seed_start":         start,
        "seed_end":           end,
        "n_trading_days":     len(master),
        "etf_universe":       cfg.ALL_TICKERS,
        "fi_etfs":            cfg.FI_ETFS,
        "eq_etfs":            cfg.EQ_ETFS,
        "benchmarks":         cfg.BENCHMARKS,
        "fred_series":        cfg.FRED_SERIES,
        "ohlcv_shape":        list(ohlcv_flat.shape),
        "returns_shape":      list(returns.shape),
        "macro_shape":        list(macro.shape),
        "macro_derived_shape":list(macro_derived.shape),
        "master_shape":       list(master.shape),
        "master_columns":     list(master.columns),
    }
    du.upload_json(metadata, cfg.FILE_METADATA, "[seed] metadata")
    logger.info("metadata.json uploaded")

    logger.info("=" * 60)
    logger.info("SEED COMPLETE")
    logger.info(f"  Trading days : {len(master)}")
    logger.info(f"  Date range   : {master.index[0].date()} -> {master.index[-1].date()}")
    logger.info(f"  Master cols  : {master.shape[1]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed P2-ETF-DEEPM dataset")
    parser.add_argument("--start", default=cfg.DATA_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None,           help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    try:
        seed(start=args.start, end=args.end)
    except Exception as e:
        logger.error(f"Seed failed: {e}")
        sys.exit(1)
