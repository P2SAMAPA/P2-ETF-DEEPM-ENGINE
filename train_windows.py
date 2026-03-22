# train_windows.py — Shrinking window training for DeePM
# Trains 8 models per option (FI / Equity), each on a progressively
# shorter training window, all evaluated on the same OOS period (2025-present).
# Winner = highest OOS annualised return.
#
# Usage:
#   python train_windows.py --option A
#   python train_windows.py --option B
#   python train_windows.py --option both

import argparse
import json
import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
import loader
import features as feat
from model import DeePM, evar_loss, sharpe_loss
from train import train_epoch, eval_epoch, DEVICE

os.makedirs(cfg.MODELS_DIR, exist_ok=True)

# ── Shrinking window definitions ───────────────────────────────────────────────
WINDOWS = [
    {"id": 1, "train_start": "2008-01-01", "train_end": "2024-12-31"},
    {"id": 2, "train_start": "2010-01-01", "train_end": "2024-12-31"},
    {"id": 3, "train_start": "2012-01-01", "train_end": "2024-12-31"},
    {"id": 4, "train_start": "2014-01-01", "train_end": "2024-12-31"},
    {"id": 5, "train_start": "2016-01-01", "train_end": "2024-12-31"},
    {"id": 6, "train_start": "2018-01-01", "train_end": "2024-12-31"},
    {"id": 7, "train_start": "2020-01-01", "train_end": "2024-12-31"},
    {"id": 8, "train_start": "2022-01-01", "train_end": "2024-12-31"},
]

OOS_START = cfg.LIVE_START   # 2025-01-01
OOS_END   = None              # today


def make_window_dataloaders(
    feat_dict: dict,
    scaler,
    train_start: str,
    train_end: str,
    oos_start: str,
) -> tuple:
    """
    Split sequences into train (train_start→train_end) and
    OOS (oos_start→today) based on dates array.
    """
    dates    = feat_dict["dates"]
    X_asset  = feat_dict["X_asset"]
    X_macro  = feat_dict["X_macro"]
    y        = feat_dict["y"]

    train_mask = (dates >= np.datetime64(train_start)) & \
                 (dates <= np.datetime64(train_end))
    oos_mask   = dates >= np.datetime64(oos_start)

    # Require minimum training size
    n_train = train_mask.sum()
    n_oos   = oos_mask.sum()

    if n_train < cfg.LOOKBACK * 2:
        raise ValueError(
            f"Window {train_start}→{train_end} has only {n_train} training samples — too few."
        )

    Xa_tr = X_asset[train_mask]
    Xm_tr = X_macro[train_mask]
    y_tr  = y[train_mask]

    Xa_oos = X_asset[oos_mask]
    Xm_oos = X_macro[oos_mask]
    y_oos  = y[oos_mask]

    # Fit scaler on train only
    Xa_tr_s, Xm_tr_s   = scaler.fit_transform(Xa_tr, Xm_tr)
    Xa_oos_s, Xm_oos_s = scaler.transform(Xa_oos, Xm_oos)

    def to_ds(Xa, Xm, yy):
        return TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xm, dtype=torch.float32),
            torch.tensor(yy, dtype=torch.float32),
        )

    train_dl = DataLoader(to_ds(Xa_tr_s,  Xm_tr_s,  y_tr),  batch_size=cfg.BATCH_SIZE, shuffle=False)
    oos_dl   = DataLoader(to_ds(Xa_oos_s, Xm_oos_s, y_oos), batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_dl, oos_dl, int(n_train), int(n_oos), dates[oos_mask]


def train_single_window(
    window: dict,
    feat_dict: dict,
    data: dict,
    option: str,
    loss_fn: str,
) -> dict:
    """Train DeePM on a single window. Returns summary with OOS metrics."""
    wid = window["id"]
    print(f"\n  Window {wid}: {window['train_start']} → {window['train_end']} | loss={loss_fn}")

    scaler = feat.FeatureScaler()

    try:
        train_dl, oos_dl, n_train, n_oos, oos_dates = make_window_dataloaders(
            feat_dict, scaler,
            train_start=window["train_start"],
            train_end=window["train_end"],
            oos_start=OOS_START,
        )
    except ValueError as e:
        print(f"  Skipping window {wid}: {e}")
        return None

    print(f"  Train: {n_train} days | OOS: {n_oos} days")

    cash_rate_mean = float(data["cash_rate"].mean())

    # Build model
    include_cash = (option == "A")

    model = DeePM(
        n_assets=feat_dict["n_assets"],
        n_asset_feats=feat_dict["n_asset_feats"],
        n_macro_feats=feat_dict["n_macro_feats"],
        asset_hidden_dim=cfg.ASSET_HIDDEN_DIM,
        macro_hidden_dim=cfg.MACRO_HIDDEN_DIM,
        graph_hidden_dim=cfg.GRAPH_HIDDEN_DIM,
        n_attn_heads=cfg.N_ATTN_HEADS,
        dropout=cfg.DROPOUT,
        include_cash=include_cash,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model_path = os.path.join(
        cfg.MODELS_DIR, f"deepm_option{option}_w{wid}_{loss_fn}.pt"
    )

    # Training loop — use train set, evaluate on OOS directly
    best_oos_return = -float("inf")
    patience_count  = 0
    cfg.LOSS_FN     = loss_fn

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimizer, cash_rate_mean, loss_fn)
        oos_loss, oos_sharpe, oos_ann_return = eval_epoch(model, oos_dl, cash_rate_mean, loss_fn)
        scheduler.step(oos_loss)

        if oos_ann_return > best_oos_return:
            best_oos_return = oos_ann_return
            patience_count  = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        if patience_count >= cfg.PATIENCE:
            break

    # Final OOS eval with best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _, oos_sharpe, oos_ann_return = eval_epoch(model, oos_dl, cash_rate_mean, loss_fn)

    print(f"  Window {wid} result: OOS ann_return={oos_ann_return*100:.2f}% | Sharpe={oos_sharpe:.3f}")

    return {
        "window_id":      wid,
        "train_start":    window["train_start"],
        "train_end":      window["train_end"],
        "oos_start":      OOS_START,
        "loss_fn":        loss_fn,
        "n_train":        n_train,
        "n_oos":          n_oos,
        "oos_ann_return": round(oos_ann_return, 4),
        "oos_sharpe":     round(oos_sharpe, 4),
        "model_path":     model_path,
        "scaler":         scaler,
    }


def train_windows_option(option: str) -> dict:
    """
    Train all 8 shrinking windows for one option (A or B).
    Tries both loss functions per window, picks best OOS return overall.
    Saves winning model as canonical shrinking window model.
    Returns full summary.
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"DeePM Shrinking Windows — Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    # Load data + build features once
    print("\n[1/2] Loading data and building features...")
    master    = loader.load_master()
    data      = loader.get_option_data(option, master)
    feat_dict = feat.prepare_features(data, lookback=cfg.LOOKBACK)

    print(f"\n[2/2] Training {len(WINDOWS)} windows × 2 loss functions...")

    all_results  = []
    best_result  = None
    best_return  = -float("inf")

    for window in WINDOWS:
        for loss_fn in ["evar", "sharpe"]:
            result = train_single_window(window, feat_dict, data, option, loss_fn)
            if result is None:
                continue

            all_results.append({k: v for k, v in result.items() if k != "scaler"})

            if result["oos_ann_return"] > best_return:
                best_return = result["oos_ann_return"]
                best_result = result

    if best_result is None:
        raise RuntimeError("All windows failed — no valid results.")

    # Copy winning model to canonical shrinking window filename
    canonical_model  = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_window_best.pt")
    canonical_scaler = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_window.pkl")

    shutil.copy2(best_result["model_path"], canonical_model)
    with open(canonical_scaler, "wb") as f:
        pickle.dump(best_result["scaler"], f)

    elapsed = round(time.time() - t0, 1)

    summary = {
        "option":          option,
        "trained_at":      datetime.utcnow().isoformat(),
        "elapsed_sec":     elapsed,
        "n_windows":       len(WINDOWS),
        "winning_window":  best_result["window_id"],
        "winning_train_start": best_result["train_start"],
        "winning_train_end":   best_result["train_end"],
        "winning_loss":    best_result["loss_fn"],
        "oos_ann_return":  best_result["oos_ann_return"],
        "oos_sharpe":      best_result["oos_sharpe"],
        "all_windows":     all_results,
        "n_assets":        feat_dict["n_assets"],
        "tickers":         feat_dict["tickers"],
        "include_cash":    (option == "A"),
        "n_asset_feats":   feat_dict["n_asset_feats"],
        "n_macro_feats":   feat_dict["n_macro_feats"],
        "config": {
            "lookback":         cfg.LOOKBACK,
            "asset_hidden_dim": cfg.ASSET_HIDDEN_DIM,
            "macro_hidden_dim": cfg.MACRO_HIDDEN_DIM,
            "graph_hidden_dim": cfg.GRAPH_HIDDEN_DIM,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}_window.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Option {option} shrinking windows complete in {elapsed}s")
    print(f"  Winner: Window {best_result['window_id']} "
          f"({best_result['train_start']} → {best_result['train_end']}) "
          f"| loss={best_result['loss_fn']}")
    print(f"  OOS ann return : {best_result['oos_ann_return']*100:.2f}%")
    print(f"  OOS Sharpe     : {best_result['oos_sharpe']:.3f}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeePM shrinking windows")
    parser.add_argument(
        "--option",
        choices=["A", "B", "both"],
        default="both",
    )
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]

    summaries = {}
    for opt in options:
        summaries[opt] = train_windows_option(opt)

    print("\n" + "="*60)
    print("ALL SHRINKING WINDOW TRAINING COMPLETE")
    for opt, s in summaries.items():
        print(f"  Option {opt}: Window {s['winning_window']} "
              f"({s['winning_train_start']}→{s['winning_train_end']}) "
              f"| {s['winning_loss']} "
              f"| OOS return={s['oos_ann_return']*100:.2f}%")
    print("="*60)
