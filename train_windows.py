# train_windows.py — Shrinking window training for DeePM
# Refactored to support parallel execution in GitHub Actions.

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

WINDOWS = {
    1: {"train_start": "2008-01-01", "train_end": "2024-12-31"},
    2: {"train_start": "2010-01-01", "train_end": "2024-12-31"},
    3: {"train_start": "2012-01-01", "train_end": "2024-12-31"},
    4: {"train_start": "2014-01-01", "train_end": "2024-12-31"},
    5: {"train_start": "2016-01-01", "train_end": "2024-12-31"},
    6: {"train_start": "2018-01-01", "train_end": "2024-12-31"},
    7: {"train_start": "2020-01-01", "train_end": "2024-12-31"},
    8: {"train_start": "2022-01-01", "train_end": "2024-12-31"},
}

OOS_START = cfg.LIVE_START   # 2025-01-01


def make_window_dataloaders(feat_dict, scaler, train_start, train_end, oos_start):
    dates    = feat_dict["dates"]
    X_asset  = feat_dict["X_asset"]
    X_macro  = feat_dict["X_macro"]
    y        = feat_dict["y"]

    train_mask = (dates >= np.datetime64(train_start)) & (dates <= np.datetime64(train_end))
    oos_mask   = dates >= np.datetime64(oos_start)

    n_train = train_mask.sum()
    if n_train < cfg.LOOKBACK * 2:
        raise ValueError(f"Window {train_start} has only {n_train} samples — too few.")

    Xa_tr, Xm_tr, y_tr = X_asset[train_mask], X_macro[train_mask], y[train_mask]
    Xa_oos, Xm_oos, y_oos = X_asset[oos_mask], X_macro[oos_mask], y[oos_mask]

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

    return train_dl, oos_dl, int(n_train), int(oos_mask.sum()), dates[oos_mask]


def train_single_window(window_id, feat_dict, data, option, loss_fn):
    window = WINDOWS[window_id]
    print(f"\nTraining Option {option} | Window {window_id} | {window['train_start']}→{window['train_end']} | loss={loss_fn}")

    scaler = feat.FeatureScaler()
    try:
        train_dl, oos_dl, n_train, n_oos, oos_dates = make_window_dataloaders(
            feat_dict, scaler, window["train_start"], window["train_end"], OOS_START
        )
    except ValueError as e:
        print(f"Skipping: {e}")
        return None

    cash_rate_mean = float(data["cash_rate"].mean())

    model = DeePM(
        n_assets=feat_dict["n_assets"],
        n_asset_feats=feat_dict["n_asset_feats"],
        n_macro_feats=feat_dict["n_macro_feats"],
        asset_hidden_dim=cfg.ASSET_HIDDEN_DIM,
        macro_hidden_dim=cfg.MACRO_HIDDEN_DIM,
        graph_hidden_dim=cfg.GRAPH_HIDDEN_DIM,
        n_attn_heads=cfg.N_ATTN_HEADS,
        dropout=cfg.DROPOUT,
        include_cash=False,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model_path = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_w{window_id}_{loss_fn}.pt")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_w{window_id}_{loss_fn}.pkl")

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
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        else:
            patience_count += 1

        if patience_count >= cfg.PATIENCE:
            break

    # Save results metric for aggregation
    result = {
        "window_id": window_id,
        "train_start": window["train_start"],
        "train_end": window["train_end"],
        "loss_fn": loss_fn,
        "n_train": n_train,
        "n_oos": n_oos,
        "oos_ann_return": round(best_oos_return, 4),
        "oos_sharpe": round(oos_sharpe, 4),
    }
    
    res_path = os.path.join(cfg.MODELS_DIR, f"result_option{option}_w{window_id}_{loss_fn}.json")
    with open(res_path, "w") as f:
        json.dump(result, f)

    print(f"Result: OOS ann_return={best_oos_return*100:.2f}% | Sharpe={oos_sharpe:.3f}")
    return result


def aggregate_and_save(option):
    """Finds the best result from the downloaded JSON files, sets canonical model."""
    print(f"\nAggregating results for Option {option}...")
    
    best_result = None
    best_return = -float("inf")
    all_results = []

    for wid in WINDOWS.keys():
        for loss_fn in ["evar", "sharpe"]:
            res_path = os.path.join(cfg.MODELS_DIR, f"result_option{option}_w{wid}_{loss_fn}.json")
            if not os.path.exists(res_path):
                continue
                
            with open(res_path, "r") as f:
                result = json.load(f)
                all_results.append(result)
                
                if result["oos_ann_return"] > best_return:
                    best_return = result["oos_ann_return"]
                    best_result = result

    if best_result is None:
        raise RuntimeError(f"No valid window results found for Option {option}.")

    # Promote the winner to canonical filenames
    wid = best_result["window_id"]
    loss = best_result["loss_fn"]
    
    best_model_path = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_w{wid}_{loss}.pt")
    best_scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_w{wid}_{loss}.pkl")
    
    canonical_model = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_window_best.pt")
    canonical_scaler = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}_window.pkl")
    
    shutil.copy2(best_model_path, canonical_model)
    shutil.copy2(best_scaler_path, canonical_scaler)

    # Save final summary
    summary = {
        "option": option,
        "trained_at": datetime.utcnow().isoformat(),
        "winning_window": wid,
        "winning_loss": loss,
        "oos_ann_return": best_result["oos_ann_return"],
        "oos_sharpe": best_result["oos_sharpe"],
        "all_windows": all_results,
    }
    
    with open(os.path.join(cfg.MODELS_DIR, f"meta_option{option}_window.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Winner Option {option}: Window {wid} ({loss}) | OOS ann return: {best_return*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    parser.add_argument("--window", type=int, help="Specific window ID to train (1-8)")
    parser.add_argument("--loss", choices=["evar", "sharpe"], help="Specific loss function")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results instead of training")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]

    if args.aggregate:
        for opt in options:
            aggregate_and_save(opt)
    else:
        if not args.window or not args.loss:
            raise ValueError("Must provide --window and --loss to train, or --aggregate to merge.")
            
        # Load data only once per runner
        master = loader.load_master()
        
        for opt in options:
            data = loader.get_option_data(opt, master)
            feat_dict = feat.prepare_features(data, lookback=cfg.LOOKBACK)
            
            train_single_window(args.window, feat_dict, data, opt, args.loss)
