from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List


def _extract_query_prefix(path: Path) -> str:
    # Expected filename pattern: Q1--online_stage3_output.json
    m = re.match(r"(?i)q(\d+)--online_stage3_output\.json$", path.name)
    if not m:
        raise ValueError(f"Unsupported filename pattern: {path.name}")
    return f"q{int(m.group(1))}"


def _assign_ids(rows: List[Any], prefix: str) -> None:
    for i, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            row["dpr_id"] = f"{prefix}_{i}"


def process_file(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list.")
    prefix = _extract_query_prefix(path)
    _assign_ids(data, prefix)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return len(data)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Assign dpr_id values in Q*--online_stage3_output.json files (e.g. q1_1, q1_2, ...)."
    )
    parser.add_argument(
        "--dir",
        default=str(here),
        help="Directory containing Q*--online_stage3_output.json files.",
    )
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    files = sorted(target_dir.glob("Q*--online_stage3_output.json"), key=lambda p: p.name.lower())
    if not files:
        print(f"No matching files found in {target_dir}")
        return

    print(f"Found {len(files)} file(s) in {target_dir}")
    for path in files:
        n = process_file(path)
        print(f"Updated {path.name}: assigned {n} dpr_id values")


if __name__ == "__main__":
    main()
