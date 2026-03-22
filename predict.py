# predict.py — Daily signal generation for DeePM
# Generates signals from both fixed split and shrinking window models.
#
# Usage:
#   python predict.py --option both

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def next_trading_day(from_date: str = None) -> str:
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    schedule = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())


def _build_inference_tensors(option: str, master: pd.DataFrame, lookback: int, tickers: list):
    """Shared feature building for inference — used by both signal generators."""
    data       = loader.get_option_data(option, master)
    asset_feat = feat.build_asset_features(data["log_returns"], data["vol"])
    macro_feat = feat.build_macro_features(data["macro"], data["macro_derived"])

    common_idx = asset_feat.index.intersection(macro_feat.index)
    af = asset_feat.reindex(common_idx).ffill().fillna(0.0)
    mf = macro_feat.reindex(common_idx).ffill().fillna(0.0)

    asset_col_indices = []
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)

    n_assets      = len(tickers)
    n_asset_feats = len(asset_col_indices[0])
    af_window     = af.iloc[-lookback:].values
    mf_window     = mf.iloc[-lookback:].values

    X_asset = np.zeros((1, n_assets, lookback, n_asset_feats), dtype=np.float32)
    X_macro = mf_window[np.newaxis, :, :]

    for a, col_idxs in enumerate(asset_col_indices):
        X_asset[0, a] = af_window[:, col_idxs]

    last_data_date = str(af.index[-1].date())

    # Regime context
    latest_macro = data["macro"].iloc[-1]
    regime_context = {
        "VIX":       round(float(latest_macro.get("VIX", 0)), 2),
        "T10Y2Y":    round(float(latest_macro.get("T10Y2Y", 0)), 3),
        "HY_SPREAD": round(float(latest_macro.get("HY_SPREAD", 0)), 2),
        "USD_INDEX": round(float(latest_macro.get("USD_INDEX", 0)), 2),
    }
    stress = None
    if "macro_stress_composite" in data["macro_derived"].columns:
        stress = round(float(data["macro_derived"]["macro_stress_composite"].iloc[-1]), 3)

    return X_asset, X_macro, last_data_date, regime_context, stress


def _run_inference(model: DeePM, scaler, X_asset, X_macro, label_names: list) -> tuple:
    """Scale inputs and run model inference. Returns (pick, conviction, weights_dict)."""
    X_asset_s, X_macro_s = scaler.transform(X_asset, X_macro)
    with torch.no_grad():
        weights_t = model(torch.tensor(X_asset_s), torch.tensor(X_macro_s))
    weights    = weights_t.numpy()[0]
    pick_idx   = int(np.argmax(weights))
    pick       = label_names[pick_idx]
    conviction = float(weights[pick_idx])
    weights_dict = {label_names[i]: round(float(weights[i]), 4) for i in range(len(label_names))}
    return pick, conviction, weights_dict


# ── Fixed split signal ─────────────────────────────────────────────────────────

def load_model(option: str) -> tuple:
    model_path  = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_best.pt")
    meta_path   = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = DeePM(
        n_assets=meta["n_assets"],
        n_asset_feats=meta["n_asset_feats"],
        n_macro_feats=meta["n_macro_feats"],
        asset_hidden_dim=meta["config"]["asset_hidden_dim"],
        macro_hidden_dim=meta["config"]["macro_hidden_dim"],
        graph_hidden_dim=meta["config"]["graph_hidden_dim"],
        n_attn_heads=meta["config"]["n_attn_heads"],
        dropout=0.0,
        include_cash=meta.get("include_cash", False),  # use saved value
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, meta, scaler


def generate_signal(option: str, master: pd.DataFrame) -> dict:
    print(f"\n[predict] Generating fixed split signal for Option {option}...")
    model, meta, scaler = load_model(option)
    lookback    = meta["lookback"]
    tickers     = meta["tickers"]
    label_names = tickers  # no CASH for either option

    X_asset, X_macro, last_data_date, regime_context, stress = \
        _build_inference_tensors(option, master, lookback, tickers)

    pick, conviction, weights_dict = _run_inference(model, scaler, X_asset, X_macro, label_names)
    signal_date = next_trading_day(last_data_date)

    print(f"  Option {option}: {pick} (conviction={conviction:.1%}) for {signal_date}")
    return {
        "option":         option,
        "mode":           "fixed_split",
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
        "winning_loss":   meta.get("winning_loss", "—"),
        "test_sharpe":    meta.get("test_sharpe", 0),
        "test_ann_return":meta.get("test_ann_return", 0),
        "model_n_params": meta["n_params"],
    }


# ── Shrinking window signal ────────────────────────────────────────────────────

def load_window_model(option: str) -> tuple:
    model_path  = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_window_best.pt")
    meta_path   = os.path.join(cfg.MODELS_DIR, f"meta_option{option}_window.json")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_window.pkl")

    if not os.path.exists(model_path):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = DeePM(
        n_assets=meta["n_assets"],
        n_asset_feats=meta["n_asset_feats"],
        n_macro_feats=meta["n_macro_feats"],
        asset_hidden_dim=meta["config"]["asset_hidden_dim"],
        macro_hidden_dim=meta["config"]["macro_hidden_dim"],
        graph_hidden_dim=meta["config"]["graph_hidden_dim"],
        n_attn_heads=cfg.N_ATTN_HEADS,
        dropout=0.0,
        include_cash=meta.get("include_cash", False),  # use saved value
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, meta, scaler


def generate_window_signal(option: str, master: pd.DataFrame) -> dict:
    print(f"\n[predict] Generating window signal for Option {option}...")
    model, meta, scaler = load_window_model(option)

    if model is None:
        print(f"  No window model for Option {option} — run train_windows.py first.")
        return None

    lookback    = meta["config"]["lookback"]
    tickers     = meta["tickers"]
    label_names = tickers  # no CASH for either option

    X_asset, X_macro, last_data_date, regime_context, stress = \
        _build_inference_tensors(option, master, lookback, tickers)

    pick, conviction, weights_dict = _run_inference(model, scaler, X_asset, X_macro, label_names)
    signal_date = next_trading_day(last_data_date)

    print(f"  Option {option} window: {pick} (conviction={conviction:.1%}) | "
          f"Window {meta['winning_window']}: {meta['winning_train_start']}→{meta['winning_train_end']}")
    return {
        "option":              option,
        "mode":                "shrinking_window",
        "option_name":         "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":         signal_date,
        "last_data_date":      last_data_date,
        "generated_at":        datetime.utcnow().isoformat(),
        "pick":                pick,
        "conviction":          round(conviction, 4),
        "weights":             weights_dict,
        "regime_context":      regime_context,
        "macro_stress":        stress,
        "trained_at":          meta["trained_at"],
        "winning_window":      meta["winning_window"],
        "winning_train_start": meta["winning_train_start"],
        "winning_train_end":   meta["winning_train_end"],
        "winning_loss":        meta["winning_loss"],
        "oos_ann_return":      meta["oos_ann_return"],
        "oos_sharpe":          meta["oos_sharpe"],
    }


# ── History + save ─────────────────────────────────────────────────────────────

def update_signal_history(signal: dict, option: str) -> None:
    history_path = os.path.join(cfg.MODELS_DIR, f"signal_history_{option}.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    record = {
        "signal_date":  signal["signal_date"],
        "pick":         signal["pick"],
        "conviction":   signal["conviction"],
        "generated_at": signal["generated_at"],
    }
    existing_dates = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing_dates:
        history.append(record)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[predict] History updated: {len(history)} records for Option {option}")


def save_signals(
    signal_A: dict = None,
    signal_B: dict = None,
    signal_A_window: dict = None,
    signal_B_window: dict = None,
) -> None:
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    combined = {
        "generated_at":    datetime.utcnow().isoformat(),
        "option_A":        signal_A,
        "option_B":        signal_B,
        "option_A_window": signal_A_window,
        "option_B_window": signal_B_window,
    }
    with open(os.path.join(cfg.MODELS_DIR, "latest_signals.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[predict] All signals saved.")

    for sig, name, opt, update_hist in [
        (signal_A,        "signal_A",        "A", True),
        (signal_B,        "signal_B",        "B", True),
        (signal_A_window, "signal_A_window", "A", False),
        (signal_B_window, "signal_B_window", "B", False),
    ]:
        if sig:
            with open(os.path.join(cfg.MODELS_DIR, f"{name}.json"), "w") as f:
                json.dump(sig, f, indent=2)
            if update_hist:
                update_signal_history(sig, opt)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DeePM daily signals")
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    sig_A = sig_B = sig_Aw = sig_Bw = None

    if args.option in ("A", "both"):
        sig_A  = generate_signal("A", master)
        sig_Aw = generate_window_signal("A", master)

    if args.option in ("B", "both"):
        sig_B  = generate_signal("B", master)
        sig_Bw = generate_window_signal("B", master)

    save_signals(sig_A, sig_B, sig_Aw, sig_Bw)

    print("\n[predict] Done.")
    for sig, label in [(sig_A, "A fixed"), (sig_B, "B fixed"),
                       (sig_Aw, "A window"), (sig_Bw, "B window")]:
        if sig:
            print(f"  Option {label}: {sig['pick']} on {sig['signal_date']} "
                  f"(conviction={sig['conviction']:.1%})")
