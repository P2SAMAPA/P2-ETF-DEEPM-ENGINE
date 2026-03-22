# data_upload_hf.py — Push local data/ and models/ files to HuggingFace dataset
# Proven pattern using CommitOperationAdd for reliable batch uploads.
#
# Usage:
#   python data_upload_hf.py            # push data/ parquets only
#   python data_upload_hf.py --meta     # also push metadata.json
#   python data_upload_hf.py --models   # also push model weights + signals

import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd

import config

api          = HfApi(token=config.HF_TOKEN)
DATASET_REPO = config.HF_DATASET_REPO
REPO_TYPE    = "dataset"


def upload_files(local_paths: list, repo_paths: list, commit_msg: str) -> None:
    """Upload a batch of local files to HF dataset repo."""
    operations = []
    for local_path, repo_path in zip(local_paths, repo_paths):
        if not os.path.exists(local_path):
            print(f"  Skipping (not found): {local_path}")
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=local_path,
            )
        )

    if not operations:
        print("  Nothing to upload.")
        return

    api.create_commit(
        repo_id        = DATASET_REPO,
        repo_type      = REPO_TYPE,
        operations     = operations,
        commit_message = commit_msg,
    )
    print(f"  Pushed {len(operations)} file(s) to {DATASET_REPO}")


def push_data() -> None:
    """Push all parquet files from data/ to HF dataset repo."""
    data_files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.parquet")))

    if not data_files:
        print("No parquet files found in data/ — run data_download.py first.")
        return

    local_paths = data_files
    repo_paths  = [f"data/{os.path.basename(f)}" for f in data_files]

    print(f"\nPushing {len(data_files)} parquet file(s) to {DATASET_REPO}...")
    for lp, rp in zip(local_paths, repo_paths):
        size_mb = os.path.getsize(lp) / 1024 / 1024
        print(f"  {os.path.basename(lp)} ({size_mb:.1f} MB) -> {rp}")

    upload_files(
        local_paths,
        repo_paths,
        commit_msg="[auto] Update ETF OHLCV + returns + macro dataset",
    )


def push_metadata() -> None:
    """Build and push metadata.json summarising the dataset."""
    meta = {
        "last_updated":  datetime.utcnow().isoformat(),
        "repo":          DATASET_REPO,
        "fi_etfs":       config.FI_ETFS,
        "eq_etfs":       config.EQ_ETFS,
        "benchmarks":    config.BENCHMARKS,
        "data_start":    config.DATA_START,
        "fred_series":   config.FRED_SERIES,
        "files": {},
    }

    for name in ["etf_ohlcv", "etf_returns", "etf_vol", "macro_fred", "macro_derived", "master"]:
        path = os.path.join(config.DATA_DIR, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "Date" in df.columns:
                dates = pd.to_datetime(df["Date"])
                meta["files"][name] = {
                    "rows":      len(df),
                    "cols":      len(df.columns),
                    "date_from": str(dates.min().date()),
                    "date_to":   str(dates.max().date()),
                    "columns":   list(df.columns),
                }
            else:
                meta["files"][name] = {
                    "rows": len(df),
                    "cols": len(df.columns),
                }

    meta_path = os.path.join(config.DATA_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  metadata.json written")

    upload_files(
        [meta_path],
        ["data/metadata.json"],
        commit_msg="[auto] Update dataset metadata",
    )


def push_models() -> None:
    """Push trained model weights, signals and history from models/ to HF dataset repo."""
    model_files = (
        glob.glob(os.path.join(config.MODELS_DIR, "*.pt"))  +
        glob.glob(os.path.join(config.MODELS_DIR, "*.pkl")) +
        glob.glob(os.path.join(config.MODELS_DIR, "*.json"))
    )

    if not model_files:
        print("No model files found in models/ — run train.py first.")
        return

    local_paths = model_files
    repo_paths  = [f"models/{os.path.basename(f)}" for f in model_files]

    print(f"\nPushing {len(model_files)} model file(s) to {DATASET_REPO}...")
    for lp, rp in zip(local_paths, repo_paths):
        size_mb = os.path.getsize(lp) / 1024 / 1024
        print(f"  {os.path.basename(lp)} ({size_mb:.1f} MB) -> {rp}")

    upload_files(
        local_paths,
        repo_paths,
        commit_msg="[auto] Update DeePM model weights and signals",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push P2-ETF-DEEPM data to HuggingFace")
    parser.add_argument("--meta",   action="store_true", help="Also push metadata.json")
    parser.add_argument("--models", action="store_true", help="Also push model weights + signals")
    args = parser.parse_args()

    # Ensure repo exists (no-op if already created)
    api.create_repo(
        repo_id   = DATASET_REPO,
        repo_type = REPO_TYPE,
        exist_ok  = True,
        private   = False,
    )

    push_data()

    if args.meta:
        push_metadata()

    if args.models:
        push_models()

    print("\nHF upload complete.")
