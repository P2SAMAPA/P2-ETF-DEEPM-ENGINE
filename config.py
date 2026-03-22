# config.py — Master configuration for P2-ETF-DEEPM-ENGINE
# Shared dataset used by all 4 projects:
#   #1 PCMCI+ Causal Discovery
#   #2 DeePM (this project)
#   #3 SAMBA Graph-Mamba
#   #4 Multi-Agent DRL

import os
from datetime import date

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

# ── FRED ───────────────────────────────────────────────────────────────────────
FRED_API_KEY    = os.environ.get("FRED_API_KEY", "")

# ── Data window ────────────────────────────────────────────────────────────────
DATA_START      = "2008-01-01"
DATA_END        = None          # None = today

# ── Option A — Fixed Income / Alternatives ─────────────────────────────────────
FI_ETFS = [
    "TLT",   # 20+ Year Treasury Bond
    "LQD",   # Investment Grade Corporate Bond
    "HYG",   # High Yield Corporate Bond
    "VNQ",   # Real Estate (REITs)
    "GLD",   # Gold
    "SLV",   # Silver
    "PFF",   # Preferred Stock
    "MBB",   # Mortgage-Backed Securities
]
FI_BENCHMARK  = "AGG"
FI_CASH       = "CASH"

# ── Option B — Equity Sectors ──────────────────────────────────────────────────
EQ_ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "GDX",   # Gold Miners
    "XME",   # Metals & Mining
]
EQ_BENCHMARK  = "SPY"
EQ_CASH       = "CASH"

# ── All tickers to download (OHLCV) ───────────────────────────────────────────
ALL_ETFS = sorted(set(FI_ETFS + EQ_ETFS))
BENCHMARKS = ["AGG", "SPY"]
ALL_TICKERS = sorted(set(ALL_ETFS + BENCHMARKS))

# ── FRED macro series ──────────────────────────────────────────────────────────
# Full superset — DeePM uses all, other projects use subset
FRED_SERIES = {
    "VIX":          "VIXCLS",          # CBOE Volatility Index
    "T10Y2Y":       "T10Y2Y",          # 10Y-2Y Treasury spread (yield curve)
    "DGS10":        "DGS10",           # 10Y Treasury yield
    "DTB3":         "DTB3",            # 3M T-Bill rate (risk-free)
    "HY_SPREAD":    "BAMLH0A0HYM2",   # ICE BofA HY OAS spread
    "IG_SPREAD":    "BAMLC0A0CM",     # ICE BofA IG OAS spread
    "USD_INDEX":    "DTWEXBGS",        # Broad USD index
    "WTI_OIL":      "DCOILWTICO",      # WTI Crude Oil price
}

# Subset used by projects #1, #3, #4
MACRO_CORE = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── Parquet filenames in HF dataset ───────────────────────────────────────────
FILE_ETF_OHLCV      = "data/etf_ohlcv.parquet"
FILE_ETF_RETURNS    = "data/etf_returns.parquet"
FILE_MACRO_FRED     = "data/macro_fred.parquet"
FILE_MACRO_DERIVED  = "data/macro_derived.parquet"
FILE_MASTER         = "data/master.parquet"
FILE_METADATA       = "data/metadata.json"

# ── Derived macro feature windows ─────────────────────────────────────────────
ZSCORE_WINDOW       = 63    # ~1 quarter rolling z-score
VOL_WINDOW          = 21    # 21-day realised vol
RETURN_WINDOWS      = [1, 5, 21, 63, 126, 252]

# ── Train / live split ────────────────────────────────────────────────────────
TRAIN_END           = "2024-12-31"
LIVE_START          = "2025-01-01"

# ── Feature engineering ───────────────────────────────────────────────────────
LOOKBACK            = 60    # trading days (~3 months) sequence length

# ── Model architecture ────────────────────────────────────────────────────────
ASSET_HIDDEN_DIM    = 64
MACRO_HIDDEN_DIM    = 64
GRAPH_HIDDEN_DIM    = 64
N_ATTN_HEADS        = 2
DROPOUT             = 0.2

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_SPLIT         = 0.70   # 70% train
VAL_SPLIT           = 0.15   # 15% val (remaining 15% = test)
BATCH_SIZE          = 64
MAX_EPOCHS          = 150
PATIENCE            = 15     # early stopping patience
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
LOSS_FN             = "evar" # "evar" or "sharpe"
EVAR_BETA           = 0.95   # penalise worst 5% of days

# ── Local directories ─────────────────────────────────────────────────────────
DATA_DIR            = "data"
MODELS_DIR          = "models"

# ── GitHub Actions schedule ───────────────────────────────────────────────────
# Daily update runs at 22:00 UTC Mon-Fri (after US market close)
CRON_SCHEDULE       = "0 22 * * 1-5"
