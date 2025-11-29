#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Integration Demo (Session 4)

One script that ties together the core ideas:
- generate a small dataset
- save to CSV and SQLite (no ORM)
- mock an API pull
- run a few parallel CPU jobs
- emit plots and a JSON summary
"""

###############################################################################

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

###############################################################################

RNG_SEED = 42
np.random.seed(RNG_SEED)

###############################################################################


def create_dataset(rows: int = 60) -> pd.DataFrame:
    """Create a small engineering-style dataset"""
    minutes = np.arange(rows)
    temperature = 24 + 4 * np.sin(minutes / 8) + np.random.normal(0, 0.6, rows)
    pressure = 101.3 + 0.8 * np.cos(minutes / 6) + np.random.normal(0, 0.1, rows)
    flow = 5 + 0.7 * np.sin(minutes / 10) + np.random.normal(0, 0.05, rows)

    return pd.DataFrame(
        {
            "minute": minutes,
            "temperature_c": temperature,
            "pressure_kpa": pressure,
            "flow_l_min": flow,
        }
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[CSV] Saved {len(df)} rows to {path}")


def save_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("measurements", conn, if_exists="replace", index=False)
        count = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    print(f"[DB] Wrote {count} rows to measurements in {db_path}")


def mock_api_data(days: int = 5) -> pd.DataFrame:
    """Fake external data to avoid network calls"""
    records = []
    for day in range(days):
        records.append(
            {
                "day": day,
                "city": "SampleCity",
                "temperature_c": 18 + np.random.normal(0, 2),
                "humidity_pct": 60 + np.random.normal(0, 8),
                "condition": np.random.choice(["Clear", "Cloudy", "Rain"]),
            }
        )
    df = pd.DataFrame(records)
    print(f"[API] Generated {len(df)} mock weather records")
    return df


def run_parallel_jobs(num_jobs: int = 6, samples: int = 50_000):
    """Simple CPU-bound jobs to demonstrate multiprocessing"""

    def job(seed: int) -> float:
        rng = np.random.default_rng(seed)
        data = rng.normal(loc=0.0, scale=1.0, size=samples)
        return float(np.mean(data))

    start = time.time()
    seeds = [RNG_SEED + i for i in range(num_jobs)]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(job, seeds))
    elapsed = time.time() - start
    print(f"[PARALLEL] {num_jobs} jobs finished in {elapsed:.2f}s")
    return results, elapsed


def make_plots(measure_df: pd.DataFrame, api_df: pd.DataFrame, out_dir: Path):
    """Create a couple of simple plots"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    # Temperature over time
    plt.figure(figsize=(6, 4))
    plt.plot(measure_df["minute"], measure_df["temperature_c"], label="Temp (C)")
    plt.xlabel("Minute")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature Over Time")
    plt.grid(alpha=0.3)
    plt.legend()
    temp_path = out_dir / "temperature_line.png"
    plt.savefig(temp_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Flow vs Pressure
    plt.figure(figsize=(6, 4))
    plt.scatter(measure_df["pressure_kpa"], measure_df["flow_l_min"], alpha=0.7)
    plt.xlabel("Pressure (kPa)")
    plt.ylabel("Flow (L/min)")
    plt.title("Flow vs Pressure")
    plt.grid(alpha=0.3)
    scatter_path = out_dir / "flow_vs_pressure.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Humidity histogram from API-like data
    plt.figure(figsize=(6, 4))
    plt.hist(api_df["humidity_pct"], bins=8, color="#3b82f6", alpha=0.8)
    plt.xlabel("Humidity (%)")
    plt.ylabel("Frequency")
    plt.title("Humidity Distribution")
    plt.grid(alpha=0.3)
    hist_path = out_dir / "humidity_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[PLOTS] Saved plots to {out_dir}")


def build_summary(
    csv_path: Path,
    db_path: Path,
    api_records: int,
    parallel_time: float,
    output_dir: Path,
) -> Dict[str, Any]:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_file": str(csv_path),
        "sqlite_db": str(db_path),
        "api_records": api_records,
        "parallel_time_sec": parallel_time,
        "output_dir": str(output_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplified integration demo (CSV + SQLite + mock API + parallel + plots)",
        epilog="Example: python session4/05_integration_demo.py --output integrated_output",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="integrated_output",
        help="Folder for CSV, DB, plots, and summary JSON",
    )
    parser.add_argument(
        "--rows", "-r", type=int, default=60, help="Number of rows to generate"
    )
    parser.add_argument(
        "--api-days", type=int, default=5, help="Mock API days to generate"
    )
    parser.add_argument(
        "--jobs", type=int, default=6, help="Parallel jobs to run"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== SIMPLIFIED INTEGRATION DEMO ===")
    print(f"Output directory: {output_dir.resolve()}")

    # 1) Dataset -> CSV + SQLite
    df = create_dataset(rows=args.rows)
    csv_path = data_dir / "engineering_test_data.csv"
    db_path = data_dir / "research_data.db"
    save_csv(df, csv_path)
    save_sqlite(df, db_path)

    # 2) Mock API
    api_df = mock_api_data(days=args.api_days)

    # 3) Parallel jobs
    _, elapsed = run_parallel_jobs(num_jobs=args.jobs)

    # 4) Plots
    make_plots(df, api_df, plots_dir)

    # 5) Summary
    summary = build_summary(
        csv_path=csv_path,
        db_path=db_path,
        api_records=len(api_df),
        parallel_time=elapsed,
        output_dir=output_dir,
    )
    summary_path = output_dir / "integrated_system_report.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] Saved {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
