"""
update_daily.py — P2-ETF-DEEPM-ENGINE
Daily incremental data update. Loads existing dataset from HF,
fetches only new trading days, rebuilds master, pushes back to HF.
Runs via GitHub Actions at 22:00 UTC Mon-Fri.
"""
import os
import io
import json
import random
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config (read from environment) ────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")

ALL_TICKERS = [
    "AGG", "GDX", "GLD", "HYG", "LQD", "MBB", "PFF",
    "QQQ", "SLV", "SPY", "TLT", "VNQ",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XME",
]

FRED_SERIES = {
    "VIX":       "VIXCLS",
    "T10Y2Y":    "T10Y2Y",
    "HY_SPREAD": "BAMLH0A0HYM2",
    "USD_INDEX": "DTWEXBGS",
    "DTB3":      "DTB3",
    "T10YIE":    "T10YIE",
    "UNRATE":    "UNRATE",
    "CPIAUCSL":  "CPIAUCSL",
    "FEDFUNDS":  "FEDFUNDS",
    "INDPRO":    "INDPRO",
    "HOUST":     "HOUST",
    "PAYEMS":    "PAYEMS",
    "DCOILWTICO":"DCOILWTICO",
    "BAMLC0A0CM":"BAMLC0A0CM",
    "WTI":       "DCOILWTICO",
}


# ── HuggingFace helpers ────────────────────────────────────────────────────────

def hf_load_parquet(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"data/{filename}",
        repo_type="dataset",
        token=HF_TOKEN or None,
        force_download=True,
    )
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def hf_upload_parquet(df: pd.DataFrame, filename: str, msg: str) -> None:
    api = HfApi(token=HF_TOKEN)
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"data/{filename}",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=msg,
    )
    log.info(f"Uploaded data/{filename} — {msg}")


# ── YFinance fetch (one ticker, with retry) ────────────────────────────────────

def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch one ticker using Ticker.history() with retry + backoff."""
    for attempt in range(4):
        try:
            t  = yf.Ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            raise ValueError("empty")
        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in
                          ["rate limit", "too many", "429", "ratelimit"])
            wait = (20 if is_rate else 5) * (2 ** attempt) + random.randint(3, 8)
            if attempt < 3:
                log.warning(f"  {ticker} attempt {attempt+1} failed: {e} — "
                            f"retrying in {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"  {ticker} failed after 4 attempts: {e}")
                return None
    return None


# ── OHLCV fetch ────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV for all tickers. Returns flat DataFrame."""
    frames = []
    for ticker in tickers:
        log.info(f"  Fetching {ticker}...")
        df = fetch_ticker(ticker, start, end)
        if df is None or df.empty:
            log.warning(f"  {ticker} — no data, skipping")
            continue
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in df.columns]
        df = df[keep].copy()
        df.columns = [f"{ticker}_{c}" for c in df.columns]
        frames.append(df)
        time.sleep(random.uniform(0.8, 2.0))   # polite pause between tickers

    if not frames:
        return pd.DataFrame()
    ohlcv = pd.concat(frames, axis=1).sort_index()
    ohlcv.index.name = "Date"
    return ohlcv


# ── Returns & derived features ─────────────────────────────────────────────────

def compute_returns(ohlcv: pd.DataFrame, tickers: list) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        col = f"{t}_Close"
        if col in ohlcv.columns:
            s = ohlcv[col]
            frames[f"{t}_ret"]    = s.pct_change()
            frames[f"{t}_logret"] = np.log(s / s.shift(1))
    return pd.DataFrame(frames, index=ohlcv.index).dropna(how="all")


def compute_vol(ohlcv: pd.DataFrame, tickers: list,
                windows=(5, 21, 63)) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        col = f"{t}_Close"
        if col in ohlcv.columns:
            logret = np.log(ohlcv[col] / ohlcv[col].shift(1))
            for w in windows:
                frames[f"{t}_vol{w}"] = logret.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=ohlcv.index).dropna(how="all")


# ── FRED macro fetch ───────────────────────────────────────────────────────────

def fetch_macro(start: str, end: str) -> pd.DataFrame:
    if not FRED_API_KEY:
        log.warning("No FRED_API_KEY — skipping macro update")
        return pd.DataFrame()
    fred   = Fred(api_key=FRED_API_KEY)
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id,
                                observation_start=start,
                                observation_end=end)
            s.index = pd.to_datetime(s.index)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            frames[name] = s
        except Exception as e:
            log.warning(f"  FRED {name} ({series_id}) failed: {e}")
    if not frames:
        return pd.DataFrame()
    macro = pd.DataFrame(frames)
    macro.index.name = "Date"
    return macro.ffill().sort_index()


# ── Master rebuild ─────────────────────────────────────────────────────────────

def build_master(ohlcv: pd.DataFrame, returns: pd.DataFrame,
                 vol: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    common = ohlcv.index
    for df in [returns, vol, macro]:
        if not df.empty:
            common = common.intersection(df.index)
    common = common.sort_values()
    parts  = [ohlcv.reindex(common), returns.reindex(common)]
    if not vol.empty:
        parts.append(vol.reindex(common))
    if not macro.empty:
        parts.append(macro.reindex(common))
    master = pd.concat(parts, axis=1).ffill()
    master.index.name = "Date"
    return master


# ── Main update logic ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.utcnow().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    if not HF_TOKEN:
        log.error("HF_TOKEN not set")
        raise SystemExit(1)

    # ── 1. Load existing data from HF ─────────────────────────────────────────
    log.info("Loading existing data from HuggingFace...")
    try:
        master_existing = hf_load_parquet("master.parquet")
        last_date = master_existing.index.max()
        log.info(f"Last stored date: {last_date.date()} | "
                 f"rows: {len(master_existing)}")
    except Exception as e:
        log.error(f"Cannot load existing data from HF: {e}")
        log.error("Run the seed workflow first to initialise the dataset.")
        raise SystemExit(1)

    # ── 2. Check if update needed ──────────────────────────────────────────────
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    start     = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    if start > today_str:
        log.info("Dataset already up to date. Nothing to do.")
        return

    log.info(f"Fetching new data: {start} → {today_str}")

    # ── 3. Fetch new OHLCV ────────────────────────────────────────────────────
    log.info("Fetching new OHLCV...")
    new_ohlcv = fetch_ohlcv(ALL_TICKERS, start=start, end=today_str)

    if new_ohlcv.empty:
        log.warning("No new OHLCV data returned — market may be closed today.")
        return

    log.info(f"New trading days: {len(new_ohlcv)} | "
             f"{list(new_ohlcv.index.date)}")

    # ── 4. Load existing component files and append ───────────────────────────
    log.info("Loading existing component files...")
    try:
        ohlcv_existing   = hf_load_parquet("etf_ohlcv.parquet")
        returns_existing = hf_load_parquet("etf_returns.parquet")
        macro_existing   = hf_load_parquet("macro_fred.parquet")
    except Exception as e:
        log.error(f"Failed to load component files: {e}")
        raise SystemExit(1)

    # Append new OHLCV
    ohlcv = pd.concat([ohlcv_existing, new_ohlcv])
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()

    # Compute new returns
    close_cols = [c for c in ohlcv.columns if c.endswith("_Close")]
    prices     = ohlcv[close_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]
    new_returns = compute_returns(new_ohlcv, ALL_TICKERS)
    returns     = pd.concat([returns_existing, new_returns])
    returns     = returns[~returns.index.duplicated(keep="last")].sort_index()

    # Vol (recompute on tail for rolling windows)
    vol = compute_vol(ohlcv, ALL_TICKERS)

    # ── 5. Update FRED macro ──────────────────────────────────────────────────
    log.info("Fetching new FRED macro...")
    # Fetch from 5 days before last stored to catch publication lags
    fred_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    new_macro  = fetch_macro(start=fred_start, end=today_str)

    if not new_macro.empty:
        macro = pd.concat([macro_existing, new_macro])
        macro = macro[~macro.index.duplicated(keep="last")].sort_index().ffill()
    else:
        macro = macro_existing

    # ── 6. Rebuild master ─────────────────────────────────────────────────────
    log.info("Rebuilding master file...")
    master = build_master(ohlcv, returns, vol, macro)

    # ── 7. Upload all files to HF ─────────────────────────────────────────────
    log.info("Uploading updated files to HuggingFace...")
    hf_upload_parquet(ohlcv,    "etf_ohlcv.parquet",
                      f"[update] OHLCV to {today_str}")
    hf_upload_parquet(returns,  "etf_returns.parquet",
                      f"[update] returns to {today_str}")
    hf_upload_parquet(macro,    "macro_fred.parquet",
                      f"[update] macro to {today_str}")
    hf_upload_parquet(master,   "master.parquet",
                      f"[update] master to {today_str}")

    log.info("=" * 60)
    log.info("UPDATE COMPLETE")
    log.info(f"  New days added : {len(new_ohlcv)}")
    log.info(f"  Latest date    : {master.index[-1].date()}")
    log.info(f"  Total rows     : {len(master)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
