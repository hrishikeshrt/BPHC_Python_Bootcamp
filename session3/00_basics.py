#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session 3 Warm-up: Files and Folders

- pick a workspace folder
- create a tiny text file
- (optionally) list CSV files that exist there
"""

###############################################################################

import os
import argparse
from pathlib import Path
from datetime import datetime

###############################################################################


def write_note_file(workspace: Path, name: str) -> Path:
    """Create a small text file with a timestamped note"""
    workspace.mkdir(parents=True, exist_ok=True)
    file_path = workspace / f"{name}.txt"

    lines = [
        "Session 3: Python as a scripting language",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"Workspace: {workspace.resolve()}",
    ]
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path


def list_csv_files(workspace: Path) -> None:
    """Print existing CSV files in the workspace (if any)"""
    csv_files = sorted(workspace.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in this workspace yet.")
        return

    print("\nCSV files in this workspace:")
    for path in csv_files:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:30}  {size_kb:6.1f} KB")


def build_parser() -> argparse.ArgumentParser:
    """Configure command-line options for the warm-up script"""
    parser = argparse.ArgumentParser(
        description="Session 3 warm-up: simple file and folder operations",
        epilog="Example: python session3/00_basics.py --workspace . --list-csv",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="Folder where the note file will be created",
    )
    parser.add_argument(
        "--name",
        "-n",
        default="session3_note",
        help="Name for the note file (without .txt)",
    )
    parser.add_argument(
        "--list-csv",
        action="store_true",
        help="List CSV files in the workspace after creating the note",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser()
    note_path = write_note_file(workspace, args.name)

    print("=== SESSION 3 WARM-UP ===")
    print(f"Created note file: {note_path}")

    if args.list_csv:
        list_csv_files(workspace)


if __name__ == "__main__":
    main()
