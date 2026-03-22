# predict.py — Daily signal generation for DeePM
# Generates next-trading-day ETF signal for Option A and Option B.
#
# Usage:
#   python predict.py
#   python predict.py --option A
#   python predict.py --option B

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch

import config as cfg
import loader
import features as feat
from model import DeePM

DEVICE = torch.device("cpu")


def next_trading_day(from_date: str = None) -> str:
    """Return the next NYSE trading day after from_date (default: today)."""
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    schedule = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())


def load_model(option: str) -> tuple:
    """Load trained DeePM model and metadata."""
    model_path = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_best.pt")
    meta_path  = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    model = DeePM(
        n_assets=meta["n_assets"],
        n_asset_feats=meta["n_asset_feats"],
        n_macro_feats=meta["n_macro_feats"],
        asset_hidden_dim=meta["config"]["asset_hidden_dim"],
        macro_hidden_dim=meta["config"]["macro_hidden_dim"],
        graph_hidden_dim=meta["config"]["graph_hidden_dim"],
        n_attn_heads=meta["config"]["n_attn_heads"],
        dropout=0.0,   # no dropout at inference
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, meta


def load_scaler(option: str):
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def generate_signal(option: str, master: pd.DataFrame) -> dict:
    """
    Generate next-day signal for one option.
    Returns dict with pick, conviction, weights, regime context etc.
    """
    print(f"\n[predict] Generating signal for Option {option}...")

    # Load model + scaler
    model, meta = load_model(option)
    scaler      = load_scaler(option)
    lookback    = meta["lookback"]
    tickers     = meta["tickers"]
    label_names = tickers + ["CASH"]

    # Get option data
    data = loader.get_option_data(option, master)

    # Build features on full data
    asset_feat = feat.build_asset_features(data["log_returns"], data["vol"])
    macro_feat = feat.build_macro_features(data["macro"], data["macro_derived"])

    # Align
    common_idx = asset_feat.index.intersection(macro_feat.index)
    af = asset_feat.reindex(common_idx).ffill().fillna(0.0)
    mf = macro_feat.reindex(common_idx).ffill().fillna(0.0)

    if len(af) < lookback:
        raise ValueError(
            f"Not enough data for lookback={lookback}. Have {len(af)} days."
        )

    # Take last `lookback` rows as the inference window
    af_window = af.iloc[-lookback:].values           # (L, Fa)
    mf_window = mf.iloc[-lookback:].values           # (L, Fm)

    # Build per-asset feature windows
    n_assets = len(tickers)
    asset_col_indices = []
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)

    n_asset_feats = len(asset_col_indices[0])
    X_asset = np.zeros((1, n_assets, lookback, n_asset_feats), dtype=np.float32)
    X_macro = mf_window[np.newaxis, :, :]            # (1, L, Fm)

    for a, col_idxs in enumerate(asset_col_indices):
        X_asset[0, a] = af_window[:, col_idxs]

    # Scale
    X_asset_s, X_macro_s = scaler.transform(X_asset, X_macro)

    # Inference
    with torch.no_grad():
        weights_t = model(
            torch.tensor(X_asset_s),
            torch.tensor(X_macro_s),
        )
    weights = weights_t.numpy()[0]                   # (A+1,)

    # Pick = argmax weight
    pick_idx   = int(np.argmax(weights))
    pick       = label_names[pick_idx]
    conviction = float(weights[pick_idx])

    # Weights dict
    weights_dict = {
        label_names[i]: round(float(weights[i]), 4)
        for i in range(len(label_names))
    }

    # Regime context from latest macro
    latest_macro = data["macro"].iloc[-1]
    regime_context = {
        "VIX":       round(float(latest_macro.get("VIX", 0)), 2),
        "T10Y2Y":    round(float(latest_macro.get("T10Y2Y", 0)), 3),
        "HY_SPREAD": round(float(latest_macro.get("HY_SPREAD", 0)), 2),
        "USD_INDEX": round(float(latest_macro.get("USD_INDEX", 0)), 2),
    }

    # Macro stress composite
    stress = None
    if "macro_stress_composite" in data["macro_derived"].columns:
        stress = round(float(data["macro_derived"]["macro_stress_composite"].iloc[-1]), 3)

    last_data_date = str(af.index[-1].date())
    signal_date    = next_trading_day(last_data_date)

    signal = {
        "option":         option,
        "option_name":    "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":    signal_date,
        "last_data_date": last_data_date,
        "generated_at":   datetime.utcnow().isoformat(),
        "pick":           pick,
        "conviction":     round(conviction, 4),
        "weights":        weights_dict,
        "regime_context": regime_context,
        "macro_stress":   stress,
        "trained_at":     meta["trained_at"],
        "test_sharpe":    meta["test_sharpe"],
        "test_ann_return":meta.get("test_ann_return", 0),
        "winning_loss":   meta.get("winning_loss", meta.get("loss_fn", "—")),
        "model_n_params": meta["n_params"],
    }

    print(f"  Option {option} signal: {pick} (conviction={conviction:.1%})")
    print(f"  Signal date: {signal_date}")
    print(f"  Macro: VIX={regime_context['VIX']}, "
          f"T10Y2Y={regime_context['T10Y2Y']}, "
          f"HY={regime_context['HY_SPREAD']}")

    return signal


def update_signal_history(signal: dict, option: str) -> None:
    """
    Append today's signal to the running history file.
    History is used by the app for the signal history table.
    """
    history_path = os.path.join(cfg.MODELS_DIR, f"signal_history_{option}.json")

    # Load existing history
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    # Build record
    record = {
        "signal_date":  signal["signal_date"],
        "pick":         signal["pick"],
        "conviction":   signal["conviction"],
        "generated_at": signal["generated_at"],
    }

    # Avoid duplicates for same signal_date
    existing_dates = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing_dates:
        history.append(record)

    # Save updated history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[predict] Signal history updated: {len(history)} records for Option {option}")


def save_signals(signal_A: dict = None, signal_B: dict = None) -> None:
    """Save signals to models/ directory for upload to HuggingFace."""
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    # Combined file
    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A":     signal_A,
        "option_B":     signal_B,
    }
    path = os.path.join(cfg.MODELS_DIR, "latest_signals.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[predict] Signals saved to {path}")

    # Individual files + history update
    for sig, name, option in [
        (signal_A, "signal_A", "A"),
        (signal_B, "signal_B", "B"),
    ]:
        if sig:
            p = os.path.join(cfg.MODELS_DIR, f"{name}.json")
            with open(p, "w") as f:
                json.dump(sig, f, indent=2)
            update_signal_history(sig, option)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DeePM daily signals")
    parser.add_argument(
        "--option",
        choices=["A", "B", "both"],
        default="both",
    )
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    sig_A = sig_B = None

    if args.option in ("A", "both"):
        sig_A = generate_signal("A", master)

    if args.option in ("B", "both"):
        sig_B = generate_signal("B", master)

    save_signals(sig_A, sig_B)

    print("\n[predict] Done.")
    if sig_A:
        print(f"  Option A: {sig_A['pick']} on {sig_A['signal_date']} "
              f"(conviction={sig_A['conviction']:.1%})")
    if sig_B:
        print(f"  Option B: {sig_B['pick']} on {sig_B['signal_date']} "
              f"(conviction={sig_B['conviction']:.1%})")
