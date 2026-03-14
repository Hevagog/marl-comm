import argparse
import os
import wandb
import pandas as pd

from utils import load_config

OUTPUT_DIR = "wandb_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_history(run, samples: int = 100_000) -> pd.DataFrame:
    """Pull the full metric history for a single run, one row per step."""
    df = run.history(samples=samples, pandas=True)

    # ── 1. Drop W&B internal / housekeeping columns ───────────────────────────
    internal = {"_runtime", "_timestamp", "_step", "_wandb"}
    drop_cols = [c for c in df.columns if c in internal or c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ── 2. Collapse sparse rows → one row per logging step ────────────────────
    # W&B sometimes emits one metric per row; grouping + taking the first
    # non-NaN value per column folds them back into a single dense row.
    if "step" in df.columns:
        group_col = "step"
    else:
        # fall back: use the DataFrame's natural index as a step proxy
        df["_idx"] = df.index.to_series().floordiv(df.shape[1]).clip(lower=0)
        group_col = "_idx"

    df = (
        df.groupby(group_col, sort=True)
        .first()  # first non-NaN value wins per column per step
        .reset_index()
    )

    df.insert(0, "run_name", run.name)
    df.insert(1, "run_id", run.id)
    return df


def download_metrics(
    project: str,
    entity: str | None = None,
    run_name_filter: str | None = None,
    samples: int = 100_000,
) -> None:
    api = wandb.Api()

    # Fetch all runs in the project (optionally scoped to an entity)
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from: {path}")
    runs = api.runs(path)

    matched = [r for r in runs if run_name_filter is None or r.name == run_name_filter]

    if not matched:
        print(
            "No runs found"
            + (f" matching name '{run_name_filter}'" if run_name_filter else "")
            + f" in project '{project}'."
        )
        return

    print(f"Found {len(matched)} run(s). Downloading …\n")

    all_frames = []

    for run in matched:
        print(f"  • [{run.id}] {run.name}  (state: {run.state})")

        df = fetch_history(run, samples=samples)

        if df.empty:
            print("    ↳ No metric history — skipping.")
            continue

        # Per-run CSV
        safe_name = run.name.replace("/", "_")
        per_run_path = os.path.join(OUTPUT_DIR, f"{safe_name}__{run.id}.csv")
        df.to_csv(per_run_path, index=False)
        print(f"    ↳ Saved {len(df):,} rows → {per_run_path}")

        all_frames.append(df)

    # Combined CSV (all matching runs together)
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, f"{project}_all_runs.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined CSV ({len(combined):,} rows total) → {combined_path}")
    elif all_frames:
        print("\nOnly one run matched — no combined CSV written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mappo_default.py",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    project = config["experiment"]["wandb_kwargs"]["project"]
    nm = config["experiment"]["name"]

    download_metrics(
        project=project,
        run_name_filter=nm,
    )
