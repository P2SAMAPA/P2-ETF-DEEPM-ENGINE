"""
update_daily.py — P2-ETF-DEEPM-ENGINE
Daily incremental data update.
Loads existing dataset from HF, fetches only new trading days,
rebuilds master, pushes back to HF.
"""
import io
import os
import time
import random
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")

ALL_TICKERS = [
    "AGG", "GDX", "GLD", "HYG", "LQD", "MBB", "PFF",
    "QQQ", "SLV", "SPY", "TLT", "VNQ",
    "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XME",
]

FRED_SERIES = {
    "VIX":        "VIXCLS",
    "T10Y2Y":     "T10Y2Y",
    "HY_SPREAD":  "BAMLH0A0HYM2",
    "USD_INDEX":  "DTWEXBGS",
    "DTB3":       "DTB3",
    "T10YIE":     "T10YIE",
    "UNRATE":     "UNRATE",
    "CPIAUCSL":   "CPIAUCSL",
    "FEDFUNDS":   "FEDFUNDS",
    "INDPRO":     "INDPRO",
    "HOUST":      "HOUST",
    "PAYEMS":     "PAYEMS",
    "WTI":        "DCOILWTICO",
    "BAMLC0A0CM": "BAMLC0A0CM",
}


def hf_load_parquet(filename):
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


def hf_upload_parquet(df, filename, msg):
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
    log.info(f"Uploaded {filename}")


def fetch_ohlcv_batch(tickers, start, end):
    """
    Download OHLCV data per ticker with heavy rate‑limit avoidance.
    Uses a custom session, random delays, and exponential backoff.
    """
    log.info(f"Downloading {len(tickers)} tickers individually from {start} to {end}")

    # Create a session with browser headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    })

    frames = []
    failed = []

    # Cooldown before first request
    time.sleep(random.uniform(2, 5))

    for ticker in tqdm(tickers, desc="Tickers"):
        # Random delay between tickers (4–8 seconds)
        time.sleep(random.uniform(4.0, 8.0))

        success = False
        for attempt in range(4):  # up to 4 attempts per ticker
            try:
                tkr = yf.Ticker(ticker, session=session)
                df = tkr.history(start=start, end=end, auto_adjust=True)

                if df.empty:
                    log.debug(f"{ticker}: no data for {start}–{end}")
                    break  # no data, not an error

                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].copy()
                if df.empty:
                    raise ValueError("No OHLCV columns found")

                df.columns = [f"{ticker}_{c}" for c in df.columns]
                frames.append(df)
                success = True
                break

            except Exception as e:
                log.warning(f"{ticker}: attempt {attempt+1} failed: {e}")
                if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                    # Long cooldown on rate‑limit errors
                    cooldown = random.uniform(30, 60)
                    log.warning(f"  Rate limited – sleeping {cooldown:.0f}s")
                    time.sleep(cooldown)

                if attempt < 3:
                    wait = (2 ** attempt) * 10 + random.uniform(0, 5)  # 10s, 20s, 40s
                    log.info(f"  Retrying {ticker} in {wait:.2f}s...")
                    time.sleep(wait)

        if not success and not df.empty:
            failed.append(ticker)

    if failed:
        log.error(f"Failed to download tickers: {failed}")

    if not frames:
        log.warning("No data downloaded for any ticker")
        return pd.DataFrame()

    ohlcv = pd.concat(frames, axis=1).sort_index()
    ohlcv.index = pd.to_datetime(ohlcv.index)
    if ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)
    ohlcv.index.name = "Date"
    log.info(f"OHLCV shape: {ohlcv.shape} | date range: {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")
    return ohlcv


def compute_returns(ohlcv):
    frames = {}
    for t in ALL_TICKERS:
        col = f"{t}_Close"
        if col in ohlcv.columns:
            s = ohlcv[col]
            frames[f"{t}_ret"]    = s.pct_change()
            frames[f"{t}_logret"] = np.log(s / s.shift(1))
    return pd.DataFrame(frames, index=ohlcv.index).dropna(how="all")


def compute_vol(ohlcv, windows=(5, 21, 63)):
    frames = {}
    for t in ALL_TICKERS:
        col = f"{t}_Close"
        if col in ohlcv.columns:
            lr = np.log(ohlcv[col] / ohlcv[col].shift(1))
            for w in windows:
                frames[f"{t}_vol{w}"] = lr.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=ohlcv.index).dropna(how="all")


def fetch_macro(start, end):
    if not FRED_API_KEY:
        log.warning("No FRED_API_KEY — skipping macro")
        return pd.DataFrame()
    fred = Fred(api_key=FRED_API_KEY)
    frames = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            frames[name] = s
        except Exception as e:
            log.warning(f"  FRED {name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index.name = "Date"
    return df.ffill().sort_index()


def build_master(ohlcv, returns, vol, macro):
    common = ohlcv.index
    for df in [returns, vol, macro]:
        if not df.empty:
            common = common.intersection(df.index)
    common = common.sort_values()
    parts = [ohlcv.reindex(common), returns.reindex(common)]
    if not vol.empty:
        parts.append(vol.reindex(common))
    if not macro.empty:
        parts.append(macro.reindex(common))
    master = pd.concat(parts, axis=1).ffill()
    master.index.name = "Date"
    return master


def main():
    log.info("=" * 60)
    log.info(f"P2-ETF-DEEPM DAILY UPDATE — {datetime.utcnow().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    if not HF_TOKEN:
        log.error("HF_TOKEN not set")
        raise SystemExit(1)

    log.info("Loading existing master from HuggingFace...")
    try:
        master_existing = hf_load_parquet("master.parquet")
        last_date = master_existing.index.max()
        log.info(f"Last stored: {last_date.date()} | rows: {len(master_existing)}")
    except Exception as e:
        log.error(f"Cannot load master from HF: {e}")
        raise SystemExit(1)

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    if start > today_str:
        log.info("Already up to date. Nothing to do.")
        return

    log.info(f"Update window: {start} to {today_str}")

    new_ohlcv = fetch_ohlcv_batch(ALL_TICKERS, start=start, end=today_str)
    if new_ohlcv.empty:
        log.warning("No new OHLCV data after all retries — market closed or YF unavailable.")
        log.warning("Skipping update. Will retry tomorrow.")
        return

    log.info(f"New days: {len(new_ohlcv)} | {list(new_ohlcv.index.date)}")

    log.info("Loading component files from HF...")
    try:
        ohlcv_existing = hf_load_parquet("etf_ohlcv.parquet")
        macro_existing = hf_load_parquet("macro_fred.parquet")
    except Exception as e:
        log.error(f"Failed to load component files: {e}")
        raise SystemExit(1)

    ohlcv = pd.concat([ohlcv_existing, new_ohlcv])
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")].sort_index()

    returns = compute_returns(ohlcv)
    vol     = compute_vol(ohlcv)

    fred_start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
    new_macro  = fetch_macro(start=fred_start, end=today_str)
    if not new_macro.empty:
        macro = pd.concat([macro_existing, new_macro])
        macro = macro[~macro.index.duplicated(keep="last")].sort_index().ffill()
    else:
        macro = macro_existing

    master = build_master(ohlcv, returns, vol, macro)

    log.info("Uploading to HuggingFace...")
    hf_upload_parquet(ohlcv,   "etf_ohlcv.parquet",   f"[update] OHLCV to {today_str}")
    hf_upload_parquet(returns, "etf_returns.parquet",  f"[update] returns to {today_str}")
    hf_upload_parquet(macro,   "macro_fred.parquet",   f"[update] macro to {today_str}")
    hf_upload_parquet(master,  "master.parquet",       f"[update] master to {today_str}")

    log.info("=" * 60)
    log.info("UPDATE COMPLETE")
    log.info(f"  New days  : {len(new_ohlcv)}")
    log.info(f"  Latest    : {master.index[-1].date()}")
    log.info(f"  Total rows: {len(master)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
