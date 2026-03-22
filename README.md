# P2-ETF-DEEPM-ENGINE

**DeePM — Distributionally Robust Portfolio Engine with Macro Graph Prior**

Regime-aware ETF signal engine for Fixed Income / Alternatives and Equity Sectors,
implementing the DeePM framework with a shared master dataset that powers all 4 new projects.

---

## Shared Dataset — p2-etf-deepm-data

This repo seeds and maintains the HuggingFace dataset `P2SAMAPA/p2-etf-deepm-data`,
which is the **single data source for 4 projects**:

| Project | Uses |
|---------|------|
| #1 PCMCI+ Causal Discovery | etf_returns + core macro |
| #2 DeePM (this project) | all files — full macro superset |
| #3 SAMBA Graph-Mamba | etf_ohlcv + etf_returns + VIX/T10Y2Y |
| #4 Multi-Agent DRL | etf_ohlcv + etf_returns + core macro |

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)

| Ticker | Description |
|--------|-------------|
| TLT | 20+ Year Treasury Bond |
| LQD | Investment Grade Corporate Bond |
| HYG | High Yield Corporate Bond |
| VNQ | Real Estate (REITs) |
| GLD | Gold |
| SLV | Silver |
| PFF | Preferred Stock |
| MBB | Mortgage-Backed Securities |
| CASH | 3M T-Bill rate (DTB3/252) |

### Option B — Equity Sectors (benchmark: SPY)

| Ticker | Description |
|--------|-------------|
| SPY | S&P 500 |
| QQQ | NASDAQ 100 |
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Health Care |
| XLI | Industrials |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLU | Utilities |
| GDX | Gold Miners |
| XME | Metals & Mining |
| CASH | 3M T-Bill rate (DTB3/252) |

---

## Dataset Files (HuggingFace)

| File | Description | Shape (approx) |
|------|-------------|----------------|
| `data/etf_ohlcv.parquet` | Daily OHLCV for all tickers, flat columns (TLT_Close, etc.) | 4400 × 100 |
| `data/etf_returns.parquet` | Simple + log daily returns per ticker | 4400 × 40 |
| `data/macro_fred.parquet` | 8 raw FRED macro series, trading-day aligned | 4400 × 8 |
| `data/macro_derived.parquet` | Engineered features: z-scores, spreads, stress composite | 4400 × 20 |
| `data/master.parquet` | Single aligned file — all of the above | 4400 × 168 |
| `data/metadata.json` | Last update info, shapes, column lists | — |

---

## Macro Features (FRED)

| Name | Series | Description |
|------|--------|-------------|
| VIX | VIXCLS | CBOE Volatility Index |
| T10Y2Y | T10Y2Y | 10Y-2Y yield curve slope |
| DGS10 | DGS10 | 10Y Treasury yield |
| DTB3 | DTB3 | 3M T-Bill rate (risk-free) |
| HY_SPREAD | BAMLH0A0HYM2 | ICE BofA HY OAS spread |
| IG_SPREAD | BAMLC0A0CM | ICE BofA IG OAS spread |
| USD_INDEX | DTWEXBGS | Broad USD index |
| WTI_OIL | DCOILWTICO | WTI Crude Oil |

---

## Setup

### Secrets required

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | HuggingFace read/write token |
| `FRED_API_KEY` | FRED API key (free at fred.stlouisfed.org) |

### 1. Seed the dataset (run once)

```
GitHub Actions → Seed Dataset (run once) → Run workflow
```

Takes ~15-20 minutes. Seeds full history from 2008-01-01.

### 2. Verify the seed

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
export FRED_API_KEY=...
python validate_dataset.py
```

### 3. Daily updates run automatically

`daily_update.yml` runs at 22:00 UTC Mon-Fri via cron.
Can also be triggered manually via workflow_dispatch.

### 4. Local development

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
export FRED_API_KEY=...
python seed.py                  # one-time seed
python update_daily.py          # test incremental update
python validate_dataset.py      # sanity check
```

---

## How other projects load the data

```python
from huggingface_hub import hf_hub_download
import pandas as pd

def load(filename):
    path = hf_hub_download(
        repo_id="P2SAMAPA/p2-etf-deepm-data",
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN,
        force_download=True,
    )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

master = load("data/master.parquet")

# PCMCI+ — just needs returns + core macro
fi_returns = master[[f"{t}_ret" for t in FI_ETFS]]
core_macro = master[["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]]
```

---

## Disclaimer

Research and educational purposes only. Not financial advice.
