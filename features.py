# features.py — Feature engineering for DeePM
# Builds input tensors for both Option A (FI) and Option B (Equity).
#
# DeePM input feature set per asset per day:
#   - Log returns: 1d, 5d, 21d, 63d
#   - Realised vol (21d annualised)
#   - Momentum z-score (cross-sectional rank)
#   - Macro features (shared across all assets)
#   - Causal Sieve: async macro features aligned to trading days

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import config as cfg


# ── Per-asset time-series features ────────────────────────────────────────────

def build_asset_features(log_returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-asset features concatenated into a flat DataFrame.
    Columns: TLT_logret_1d, TLT_logret_5d, ..., TLT_vol, TLT_mom_zscore, ...
    """
    frames = []

    for ticker in log_returns.columns:
        lr = log_returns[ticker]
        v  = vol[ticker] if ticker in vol.columns else lr.rolling(cfg.VOL_WINDOW).std() * np.sqrt(252)

        f = pd.DataFrame(index=lr.index)

        # Multi-horizon log returns
        f[f"{ticker}_logret_1d"]  = lr
        f[f"{ticker}_logret_5d"]  = lr.rolling(5).sum()
        f[f"{ticker}_logret_21d"] = lr.rolling(21).sum()
        f[f"{ticker}_logret_63d"] = lr.rolling(63).sum()

        # Realised volatility
        f[f"{ticker}_vol"] = v

        # Vol-scaled return (Sharpe-like signal)
        f[f"{ticker}_vol_scaled"] = lr / (v.replace(0, np.nan) / np.sqrt(252) + 1e-8)

        frames.append(f)

    asset_features = pd.concat(frames, axis=1)

    # Cross-sectional momentum z-score (rank-based)
    ret_1d_cols = [c for c in asset_features.columns if c.endswith("_logret_1d")]
    ret_21d_cols = [c for c in asset_features.columns if c.endswith("_logret_21d")]

    for cols, suffix in [(ret_1d_cols, "mom1d_zrank"), (ret_21d_cols, "mom21d_zrank")]:
        rank_df = asset_features[cols].rank(axis=1, pct=True) * 2 - 1  # scale to [-1, 1]
        rank_df.columns = [c.replace(c.split("_")[-2] + "_" + c.split("_")[-1], suffix)
                           for c in cols]
        # Rename properly
        for orig, col in zip(cols, rank_df.columns):
            ticker = orig.split("_")[0]
            asset_features[f"{ticker}_{suffix}"] = rank_df[col]

    return asset_features.dropna(how="all")


def build_macro_features(macro: pd.DataFrame, macro_derived: pd.DataFrame) -> pd.DataFrame:
    """
    Combine raw FRED + derived macro features.
    These are shared across all assets (broadcast).
    """
    # Use derived features primarily — they're already z-scored and stationary
    derived_cols = [c for c in macro_derived.columns if c in [
        "VIX_zscore", "VIX_log", "VIX_chg1d",
        "YC_slope", "YC_slope_zscore", "YC_slope_chg",
        "DGS10_zscore", "DGS10_chg",
        "HY_spread_zscore", "HY_spread_chg",
        "IG_spread_zscore",
        "HY_IG_ratio_zscore",
        "credit_stress",
        "USD_zscore", "USD_chg",
        "OIL_zscore", "OIL_chg",
        "TBILL_daily",
        "macro_stress_composite",
    ]]

    macro_feat = macro_derived[derived_cols].copy()
    macro_feat.index.name = "Date"
    return macro_feat


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    asset_features: pd.DataFrame,
    macro_features: pd.DataFrame,
    tickers: list,
    lookback: int,
    target_returns: pd.DataFrame,
) -> tuple:
    """
    Build (X, y, dates) tensors for DeePM training.

    X shape: (N, lookback, n_features_per_asset + n_macro_features)
             One sample per day per asset — stacked as (N*n_assets, lookback, features)
             OR as (N, n_assets, lookback, features) for graph models.

    y shape: (N, n_assets) — next-day simple returns for all assets

    We return the multi-asset format: (N, n_assets, lookback, n_asset_features)
    plus macro separately: (N, lookback, n_macro_features)
    """
    # Align indices
    common_idx = (
        asset_features.index
        .intersection(macro_features.index)
        .intersection(target_returns.index)
    )
    common_idx = common_idx.sort_values()

    af = asset_features.reindex(common_idx).ffill().fillna(0.0)
    mf = macro_features.reindex(common_idx).ffill().fillna(0.0)
    tr = target_returns.reindex(common_idx)

    n_assets      = len(tickers)
    asset_feat_cols_per_ticker = [
        c for c in af.columns if c.startswith(tickers[0] + "_")
    ]
    n_asset_feats = len(asset_feat_cols_per_ticker)
    n_macro_feats = mf.shape[1]

    X_asset = np.zeros((len(common_idx) - lookback, n_assets, lookback, n_asset_feats), dtype=np.float32)
    X_macro = np.zeros((len(common_idx) - lookback, lookback, n_macro_feats), dtype=np.float32)
    y       = np.zeros((len(common_idx) - lookback, n_assets), dtype=np.float32)
    dates   = common_idx[lookback:]

    af_arr = af.values
    mf_arr = mf.values
    tr_arr = tr.values

    # Build per-asset column indices
    asset_col_indices = []
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)

    for i in range(len(common_idx) - lookback):
        window_slice = slice(i, i + lookback)
        for a, col_idxs in enumerate(asset_col_indices):
            X_asset[i, a] = af_arr[window_slice][:, col_idxs]
        X_macro[i] = mf_arr[window_slice]
        y[i] = tr_arr[i + lookback]

    # Replace NaN in y with 0 (CASH return)
    y = np.nan_to_num(y, nan=0.0)

    return X_asset, X_macro, y, dates


# ── Label builder ──────────────────────────────────────────────────────────────

def build_labels(
    returns: pd.DataFrame,
    tickers: list,
    cash_rate: pd.Series,
    vix: pd.Series = None,
    vix_cash_threshold: float = 25.0,
) -> tuple:
    """
    Build classification labels for the ETF classifier head.

    label[t] = argmax of next-day returns across tickers
    CASH label triggered when:
        - All ETFs have negative next-day return, OR
        - VIX > vix_cash_threshold

    Returns:
        labels       — pd.Series of int (0..n_tickers, last = CASH)
        label_names  — list of ticker names + ["CASH"]
        excess_ret   — returns minus daily cash rate (for EVaR loss)
    """
    ret_cols = [t for t in tickers if t in returns.columns]
    ret = returns[ret_cols].copy()

    label_names = ret_cols + [cfg.FI_CASH if cfg.FI_CASH else "CASH"]
    n_cash = len(label_names) - 1

    labels = pd.Series(index=ret.index, dtype=int, name="label")
    excess_ret = ret.subtract(cash_rate.reindex(ret.index).fillna(0.0), axis=0)

    for date in ret.index:
        row = ret.loc[date]
        if row.isna().all():
            labels[date] = n_cash
            continue

        # VIX-based cash override
        if vix is not None and date in vix.index:
            if vix.loc[date] > vix_cash_threshold:
                labels[date] = n_cash
                continue

        # All negative → CASH
        if (row.dropna() < 0).all():
            labels[date] = n_cash
            continue

        labels[date] = int(row.fillna(-np.inf).argmax())

    return labels, label_names, excess_ret


# ── Scaler ─────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """RobustScaler fitted on training data only — no data leakage."""

    def __init__(self):
        self.asset_scaler = RobustScaler()
        self.macro_scaler = RobustScaler()
        self._fitted = False

    def fit(self, X_asset: np.ndarray, X_macro: np.ndarray):
        """
        X_asset: (N, n_assets, lookback, n_asset_feats)
        X_macro: (N, lookback, n_macro_feats)
        """
        N, A, L, Fa = X_asset.shape
        self.asset_scaler.fit(X_asset.reshape(-1, Fa))
        N, L, Fm = X_macro.shape
        self.macro_scaler.fit(X_macro.reshape(-1, Fm))
        self._fitted = True
        return self

    def transform(self, X_asset: np.ndarray, X_macro: np.ndarray):
        N, A, L, Fa = X_asset.shape
        Xa = self.asset_scaler.transform(X_asset.reshape(-1, Fa)).reshape(N, A, L, Fa)
        N, L, Fm = X_macro.shape
        Xm = self.macro_scaler.transform(X_macro.reshape(-1, Fm)).reshape(N, L, Fm)
        return Xa.astype(np.float32), Xm.astype(np.float32)

    def fit_transform(self, X_asset: np.ndarray, X_macro: np.ndarray):
        return self.fit(X_asset, X_macro).transform(X_asset, X_macro)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def prepare_features(data: dict, lookback: int = None) -> dict:
    """
    Full feature preparation pipeline for one option (A or B).

    Args:
        data     : dict from loader.get_option_data()
        lookback : sequence length (default cfg.LOOKBACK)

    Returns dict with:
        X_asset, X_macro, y, dates, labels, label_names,
        excess_ret, scaler, tickers, macro_feature_names
    """
    lookback = lookback or cfg.LOOKBACK

    print(f"[features] Building features for Option {data['option']}...")

    # Asset features
    asset_feat = build_asset_features(data["log_returns"], data["vol"])

    # Macro features
    macro_feat = build_macro_features(data["macro"], data["macro_derived"])

    # Labels
    vix = data["macro"]["VIX"] if "VIX" in data["macro"].columns else None
    labels, label_names, excess_ret = build_labels(
        data["returns"], data["tickers"], data["cash_rate"], vix=vix
    )

    # Next-day returns (target for EVaR loss)
    target_ret = data["returns"].reindex(
        data["returns"].index.intersection(asset_feat.index)
    )

    # Sequences
    X_asset, X_macro, y, dates = build_sequences(
        asset_feat, macro_feat, data["tickers"], lookback, target_ret
    )

    print(f"[features] X_asset: {X_asset.shape}, X_macro: {X_macro.shape}, y: {y.shape}")

    return {
        "X_asset":            X_asset,
        "X_macro":            X_macro,
        "y":                  y,
        "dates":              dates,
        "labels":             labels.reindex(dates),
        "label_names":        label_names,
        "excess_ret":         excess_ret,
        "tickers":            data["tickers"],
        "macro_feature_names": list(macro_feat.columns),
        "n_assets":           len(data["tickers"]),
        "n_asset_feats":      X_asset.shape[-1],
        "n_macro_feats":      X_macro.shape[-1],
    }
