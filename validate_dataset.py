# validate_dataset.py — Quick sanity check on the HF dataset
# Run after seeding to confirm everything looks correct.
#
# Usage: python validate_dataset.py

import logging
import sys

import pandas as pd

import config as cfg
import data_utils as du

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PASS = "✓"
FAIL = "✗"


def check(condition: bool, msg: str) -> bool:
    status = PASS if condition else FAIL
    logger.info(f"  {status} {msg}")
    return condition


def validate() -> bool:
    ok = True
    logger.info("=" * 60)
    logger.info("P2-ETF-DEEPM DATASET VALIDATION")
    logger.info("=" * 60)

    # ── Load all files ────────────────────────────────────────────────────────
    logger.info("\n[1] Loading files from HuggingFace...")
    try:
        ohlcv    = du.load_parquet(cfg.FILE_ETF_OHLCV)
        returns  = du.load_parquet(cfg.FILE_ETF_RETURNS)
        macro    = du.load_parquet(cfg.FILE_MACRO_FRED)
        derived  = du.load_parquet(cfg.FILE_MACRO_DERIVED)
        master   = du.load_parquet(cfg.FILE_MASTER)
        metadata = du.load_metadata()
        logger.info("  All files loaded successfully")
    except Exception as e:
        logger.error(f"  Failed to load files: {e}")
        return False

    # ── OHLCV checks ──────────────────────────────────────────────────────────
    logger.info("\n[2] ETF OHLCV checks...")
    ok &= check(len(ohlcv) > 3000, f"Row count {len(ohlcv)} > 3000 trading days")
    ok &= check(ohlcv.index[0].year == 2008, f"Starts in 2008 (got {ohlcv.index[0].date()})")
    ok &= check(not ohlcv.index.duplicated().any(), "No duplicate dates")
    ok &= check(ohlcv.index.is_monotonic_increasing, "Index is sorted")

    for ticker in cfg.ALL_TICKERS:
        col = f"{ticker}_Close"
        ok &= check(col in ohlcv.columns, f"{col} present")
        if col in ohlcv.columns:
            nulls = ohlcv[col].isna().sum()
            ok &= check(nulls < len(ohlcv) * 0.01, f"{ticker} Close < 1% nulls (got {nulls})")

    # ── Returns checks ────────────────────────────────────────────────────────
    logger.info("\n[3] ETF returns checks...")
    for ticker in cfg.ALL_TICKERS:
        col = f"{ticker}_ret"
        ok &= check(col in returns.columns, f"{col} present")
        if col in returns.columns:
            extreme = (returns[col].abs() > 0.5).sum()
            ok &= check(extreme < 10, f"{ticker} returns: <10 days with |ret|>50% (got {extreme})")

    # ── Macro checks ──────────────────────────────────────────────────────────
    logger.info("\n[4] FRED macro checks...")
    for name in cfg.FRED_SERIES:
        ok &= check(name in macro.columns, f"{name} present")
        if name in macro.columns:
            nulls = macro[name].isna().sum()
            ok &= check(nulls < len(macro) * 0.05, f"{name} < 5% nulls (got {nulls})")

    # ── Derived checks ────────────────────────────────────────────────────────
    logger.info("\n[5] Derived macro checks...")
    expected_derived = [
        "VIX_zscore", "YC_slope", "HY_spread_zscore",
        "credit_stress", "macro_stress_composite", "TBILL_daily"
    ]
    for col in expected_derived:
        ok &= check(col in derived.columns, f"{col} present")

    # ── Master checks ─────────────────────────────────────────────────────────
    logger.info("\n[6] Master file checks...")
    ok &= check(
        len(master) >= len(ohlcv) * 0.95,
        f"Master rows {len(master)} >= 95% of OHLCV rows {len(ohlcv)}"
    )
    ok &= check(
        master.index[-1] == ohlcv.index[-1],
        f"Master last date matches OHLCV ({master.index[-1].date()})"
    )

    # ── Project subset checks ─────────────────────────────────────────────────
    logger.info("\n[7] Project subset availability...")

    pcmci_cols = [f"{t}_ret" for t in cfg.FI_ETFS] + ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX"]
    ok &= check(
        all(c in master.columns for c in pcmci_cols),
        f"#1 PCMCI+ FI columns available"
    )

    deepm_cols = list(cfg.FRED_SERIES.keys()) + ["macro_stress_composite", "credit_stress"]
    ok &= check(
        all(c in master.columns for c in deepm_cols),
        f"#2 DeePM full macro available"
    )

    eq_ret_cols = [f"{t}_ret" for t in cfg.EQ_ETFS]
    ok &= check(
        all(c in master.columns for c in eq_ret_cols),
        f"Option B equity returns available"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    if ok:
        logger.info(f"VALIDATION PASSED")
        logger.info(f"  Trading days : {len(master)}")
        logger.info(f"  Date range   : {master.index[0].date()} -> {master.index[-1].date()}")
        logger.info(f"  Master cols  : {master.shape[1]}")
        logger.info(f"  Last updated : {metadata.get('last_updated', 'unknown')}")
    else:
        logger.error("VALIDATION FAILED — review errors above")
    logger.info("=" * 60)
    return ok


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
