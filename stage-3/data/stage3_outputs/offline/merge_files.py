import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _batch_sort_key(path: Path) -> Tuple[int, str]:
    """
    Sort batch files by trailing batch number if present.
    Example: stage3output_batch7.json -> (7, "name")
    """
    m = re.search(r"batch(\d+)\.json$", path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def _load_batch_rows(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return [row for row in data if isinstance(row, dict)]


def merge_batches(
    source_dir: Path,
    output_file: Path,
    dedupe_key: str = "dpr_id",
) -> Tuple[int, int, List[Path]]:
    """
    Merge stage3 batch JSON files into a single JSON list.

    - Includes files matching stage3output_batch*.json
    - Excludes *_execution_summary.json
    - De-duplicates by `dedupe_key` (later batch wins)
    """
    files = sorted(
        (
            p
            for p in source_dir.glob("stage3output_batch*.json")
            if "_execution_summary" not in p.name
        ),
        key=_batch_sort_key,
    )
    if not files:
        output_file.write_text("[]\n", encoding="utf-8")
        return (0, 0, [])

    merged_rows: List[dict] = []
    by_key: Dict[str, dict] = {}
    for path in files:
        for row in _load_batch_rows(path):
            merged_rows.append(row)
            key_val = row.get(dedupe_key)
            if key_val is not None:
                by_key[str(key_val)] = row

    if by_key:
        # Preserve deterministic output order by first appearance, replacing with latest row values.
        seen: set = set()
        deduped_rows: List[dict] = []
        for row in merged_rows:
            key_val = row.get(dedupe_key)
            if key_val is None:
                deduped_rows.append(row)
                continue
            k = str(key_val)
            if k in seen:
                continue
            seen.add(k)
            deduped_rows.append(by_key[k])
    else:
        deduped_rows = merged_rows

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(deduped_rows, indent=2) + "\n", encoding="utf-8")
    return (len(merged_rows), len(deduped_rows), files)


def _fingerprint(paths: List[Path]) -> Tuple[Tuple[str, int, int], ...]:
    out = []
    for p in paths:
        st = p.stat()
        out.append((p.name, st.st_size, int(st.st_mtime)))
    return tuple(sorted(out))


def _run_once(source_dir: Path, output_file: Path, dedupe_key: str) -> None:
    total, unique, files = merge_batches(source_dir, output_file, dedupe_key=dedupe_key)
    print(f"Found {len(files)} batch file(s).")
    print(f"Merged rows: {total} | Unique rows ({dedupe_key}): {unique}")
    print(f"Wrote: {output_file}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge stage3 batch outputs into one JSON file."
    )
    parser.add_argument(
        "--source-dir",
        default=str(here),
        help="Directory containing stage3output_batch*.json",
    )
    parser.add_argument(
        "--output",
        default=str(here / "stage3_offline_output_groq.json"),
        help="Output merged JSON file path",
    )
    parser.add_argument(
        "--dedupe-key",
        default="dpr_id",
        help="Key used to de-duplicate records (default: dpr_id)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch batch files and re-merge on changes",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=5.0,
        help="Polling interval for --watch mode (seconds)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_file = Path(args.output).resolve()
    dedupe_key = args.dedupe_key

    if not args.watch:
        _run_once(source_dir, output_file, dedupe_key)
        return

    print(f"Watching {source_dir} for batch updates...")
    last_state: Optional[Tuple[Tuple[str, int, int], ...]] = None
    while True:
        files = sorted(
            (
                p
                for p in source_dir.glob("stage3output_batch*.json")
                if "_execution_summary" not in p.name
            ),
            key=_batch_sort_key,
        )
        state = _fingerprint(files)
        if state != last_state:
            _run_once(source_dir, output_file, dedupe_key)
            last_state = state
        time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    main()