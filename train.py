# train.py — DeePM training pipeline
# Trains Option A (Fixed Income) and Option B (Equity) models.
#
# Usage:
#   python train.py --option A
#   python train.py --option B
#   python train.py --option both   # trains A then B

import argparse
import json
import os
import pickle
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

os.makedirs(cfg.MODELS_DIR, exist_ok=True)
os.makedirs(cfg.DATA_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  # GitHub Actions free tier — CPU only


# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler) -> tuple:
    """
    Split into train/val/test and return DataLoaders.
    Split: 70% train / 15% val / 15% test (chronological, no shuffle).
    """
    X_a = feat_dict["X_asset"]
    X_m = feat_dict["X_macro"]
    y   = feat_dict["y"]
    N   = len(X_a)

    n_train = int(N * cfg.TRAIN_SPLIT)
    n_val   = int(N * cfg.VAL_SPLIT)

    # Fit scaler on train only — no data leakage
    X_a_train, X_m_train = scaler.fit_transform(X_a[:n_train], X_m[:n_train])
    X_a_val,   X_m_val   = scaler.transform(X_a[n_train:n_train+n_val],
                                             X_m[n_train:n_train+n_val])
    X_a_test,  X_m_test  = scaler.transform(X_a[n_train+n_val:],
                                             X_m[n_train+n_val:])

    def to_tensors(Xa, Xm, y_):
        return TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xm, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32),
        )

    train_ds = to_tensors(X_a_train, X_m_train, y[:n_train])
    val_ds   = to_tensors(X_a_val,   X_m_val,   y[n_train:n_train+n_val])
    test_ds  = to_tensors(X_a_test,  X_m_test,  y[n_train+n_val:])

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    splits = {
        "n_train": n_train,
        "n_val":   n_val,
        "n_test":  N - n_train - n_val,
        "train_start": str(feat_dict["dates"][0].date()),
        "train_end":   str(feat_dict["dates"][n_train - 1].date()),
        "val_end":     str(feat_dict["dates"][n_train + n_val - 1].date()),
        "test_end":    str(feat_dict["dates"][-1].date()),
    }
    return train_dl, val_dl, test_dl, splits


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(
    model: DeePM,
    loader_: DataLoader,
    optimizer: torch.optim.Optimizer,
    cash_rate_mean: float,
    loss_fn: str = "evar",
) -> float:
    model.train()
    total_loss = 0.0
    cash_t = torch.tensor(cash_rate_mean, dtype=torch.float32)

    for X_a, X_m, y_batch in loader_:
        optimizer.zero_grad()
        weights = model(X_a, X_m)

        cash_batch = cash_t.expand(len(y_batch))

        if loss_fn == "evar":
            loss = evar_loss(weights, y_batch, cash_batch, beta=cfg.EVAR_BETA)
        else:
            loss = sharpe_loss(weights, y_batch, cash_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader_)


def eval_epoch(
    model: DeePM,
    loader_: DataLoader,
    cash_rate_mean: float,
    loss_fn: str = "evar",
) -> tuple:
    """Returns (loss, annualised_sharpe, annualised_return)."""
    model.eval()
    total_loss    = 0.0
    all_port_rets = []
    cash_t = torch.tensor(cash_rate_mean, dtype=torch.float32)

    with torch.no_grad():
        for X_a, X_m, y_batch in loader_:
            weights    = model(X_a, X_m)
            cash_batch = cash_t.expand(len(y_batch))

            if loss_fn == "evar":
                loss = evar_loss(weights, y_batch, cash_batch, beta=cfg.EVAR_BETA)
            else:
                loss = sharpe_loss(weights, y_batch, cash_batch)

            total_loss += loss.item()

            n_outputs = weights.shape[1]
            has_cash  = (n_outputs > y_batch.shape[1])
            n_assets_ = y_batch.shape[1]
            w_assets  = weights[:, :n_assets_]
            w_cash    = weights[:, n_assets_:n_assets_+1] if has_cash else \
                        torch.zeros(len(y_batch), 1)
            port_ret  = (w_assets * y_batch).sum(dim=1) + \
                        w_cash.squeeze(1) * cash_batch
            all_port_rets.extend(port_ret.numpy())

    avg_loss   = total_loss / len(loader_)
    rets_arr   = np.array(all_port_rets)
    ann_return = float(rets_arr.mean() * 252)
    sharpe     = float((rets_arr.mean() / (rets_arr.std() + 1e-8)) * np.sqrt(252))

    return avg_loss, sharpe, ann_return


# ── Main training function ─────────────────────────────────────────────────────

def train_option(option: str) -> dict:
    """Train DeePM for one option (A or B). Returns training summary."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"DeePM Training — Option {'A (Fixed Income)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    master = loader.load_master()
    data   = loader.get_option_data(option, master)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\n[2/5] Building features...")
    feat_dict = feat.prepare_features(data, lookback=cfg.LOOKBACK)
    scaler    = feat.FeatureScaler()

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("\n[3/5] Preparing dataloaders...")
    train_dl, val_dl, test_dl, splits = make_dataloaders(feat_dict, scaler)
    print(f"  Train: {splits['n_train']} | Val: {splits['n_val']} | Test: {splits['n_test']}")
    print(f"  Train: {splits['train_start']} -> {splits['train_end']}")
    print(f"  Val  : {splits['train_end']}   -> {splits['val_end']}")
    print(f"  Test : {splits['val_end']}     -> {splits['test_end']}")

    cash_rate_mean = float(data["cash_rate"].mean())

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Building model...")
    include_cash = False  # no CASH for either option

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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Training ({cfg.MAX_EPOCHS} epochs, patience={cfg.PATIENCE})...")
    best_val_loss      = float("inf")
    best_val_sharpe    = -float("inf")
    best_val_ann_return = -float("inf")
    patience_count     = 0
    history = {"train_loss": [], "val_loss": [], "val_sharpe": [], "val_ann_return": []}

    best_model_path = os.path.join(cfg.MODELS_DIR, f"deepm_option{option}_{cfg.LOSS_FN}_best.pt")

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_loss                       = train_epoch(model, train_dl, optimizer, cash_rate_mean, cfg.LOSS_FN)
        val_loss, val_sharpe, val_return  = eval_epoch(model, val_dl, cash_rate_mean, cfg.LOSS_FN)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_sharpe"].append(round(val_sharpe, 4))
        history["val_ann_return"].append(round(val_return, 4))

        scheduler.step(val_loss)

        # Save best model by highest annualised return on val set
        improved = val_return > best_val_ann_return
        if improved:
            best_val_loss       = val_loss
            best_val_sharpe     = val_sharpe
            best_val_ann_return = val_return
            patience_count      = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_ret={val_return:.4f} | "
                  f"val_sharpe={val_sharpe:.3f} | "
                  f"{'*BEST*' if improved else f'patience {patience_count}/{cfg.PATIENCE}'}")

        if patience_count >= cfg.PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_sharpe, test_ann_return = eval_epoch(model, test_dl, cash_rate_mean, cfg.LOSS_FN)
    print(f"  Test loss={test_loss:.4f} | Test Sharpe={test_sharpe:.3f} | "
          f"Test Ann Return={test_ann_return:.4f} ({test_ann_return*100:.2f}%)")

    # ── Save scaler ───────────────────────────────────────────────────────────
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # ── Save metadata ─────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    summary = {
        "option":          option,
        "trained_at":      datetime.utcnow().isoformat(),
        "elapsed_sec":     elapsed,
        "n_params":        n_params,
        "n_assets":        feat_dict["n_assets"],
        "tickers":         feat_dict["tickers"],
        "include_cash":    include_cash,
        "n_asset_feats":   feat_dict["n_asset_feats"],
        "n_macro_feats":   feat_dict["n_macro_feats"],
        "lookback":        cfg.LOOKBACK,
        "loss_fn":         cfg.LOSS_FN,
        "best_val_loss":      round(best_val_loss, 6),
        "best_val_sharpe":    round(best_val_sharpe, 4),
        "best_val_ann_return":round(best_val_ann_return, 4),
        "test_loss":          round(test_loss, 6),
        "test_sharpe":        round(test_sharpe, 4),
        "test_ann_return":    round(test_ann_return, 4),
        "splits":          splits,
        "history":         history,
        "config": {
            "asset_hidden_dim": cfg.ASSET_HIDDEN_DIM,
            "macro_hidden_dim": cfg.MACRO_HIDDEN_DIM,
            "graph_hidden_dim": cfg.GRAPH_HIDDEN_DIM,
            "n_attn_heads":     cfg.N_ATTN_HEADS,
            "dropout":          cfg.DROPOUT,
            "lr":               cfg.LEARNING_RATE,
            "batch_size":       cfg.BATCH_SIZE,
            "max_epochs":       cfg.MAX_EPOCHS,
            "patience":         cfg.PATIENCE,
            "evar_beta":        cfg.EVAR_BETA,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOption {option} training complete in {elapsed}s")
    print(f"  Best val Ann Return : {best_val_ann_return*100:.2f}%")
    print(f"  Best val Sharpe     : {best_val_sharpe:.3f}")
    print(f"  Test Ann Return     : {test_ann_return*100:.2f}%")
    print(f"  Test Sharpe         : {test_sharpe:.3f}")
    print(f"  Model saved         : {best_model_path}")

    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeePM model")
    parser.add_argument(
        "--option",
        choices=["A", "B", "both"],
        default="both",
        help="A = Fixed Income, B = Equity, both = train sequentially",
    )
    parser.add_argument(
        "--loss",
        choices=["evar", "sharpe", "both"],
        default="both",
        help="evar, sharpe, or both (trains both and keeps the better one)",
    )
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    loss_fns = ["evar", "sharpe"] if args.loss == "both" else [args.loss]

    final_summaries = {}

    for opt in options:
        best_summary    = None
        best_ann_return = -float("inf")

        for loss_fn in loss_fns:
            print(f"\n{'='*60}")
            print(f"Training Option {opt} with loss={loss_fn}")
            print(f"{'='*60}")
            cfg.LOSS_FN = loss_fn
            summary = train_option(opt)

            # Winner = highest annualised return on test set
            if summary["test_ann_return"] > best_ann_return:
                best_ann_return = summary["test_ann_return"]
                best_summary    = summary

                # Copy winning model to canonical filename used by predict.py
                import shutil
                src = os.path.join(cfg.MODELS_DIR, f"deepm_option{opt}_{loss_fn}_best.pt")
                dst = os.path.join(cfg.MODELS_DIR, f"deepm_option{opt}_best.pt")
                shutil.copy2(src, dst)

                # Save meta with winner flag
                best_summary["winning_loss"] = loss_fn
                meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{opt}.json")
                with open(meta_path, "w") as f:
                    json.dump(best_summary, f, indent=2)

        final_summaries[opt] = best_summary
        print(f"\nOption {opt} winner: loss={best_summary['winning_loss']} "
              f"| test ann_return={best_summary['test_ann_return']:.3f} "
              f"| test Sharpe={best_summary['test_sharpe']:.3f}")

    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE")
    for opt, summary in final_summaries.items():
        print(f"  Option {opt}: {summary['winning_loss']} | "
              f"ann_return={summary['test_ann_return']:.3f} | "
              f"Sharpe={summary['test_sharpe']:.3f}")
    print("="*60)
