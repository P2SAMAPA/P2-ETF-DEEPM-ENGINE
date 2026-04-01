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
