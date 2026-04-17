from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from pipelinenew_query import run_stage3_pipeline


def _iter_input_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        files.extend(input_dir.glob(pattern))
    return sorted(files, key=lambda p: p.name.lower())


def _looks_like_summary_file(path: Path) -> bool:
    n = path.name.lower()
    return "summary" in n or "execution_summary" in n


def _iter_query_folder_inputs(
    stage2_root: Path,
    folder_suffix: str,
) -> List[Tuple[str, Path]]:
    """
    Discover per-query input files from folders such as:
      Q1--online, Q2--online, ...
    and pick the first non-summary .jsonl/.json file in each folder.
    Returns: [(folder_name, input_file_path), ...]
    """
    pairs: List[Tuple[str, Path]] = []
    query_dirs = sorted(
        (
            p
            for p in stage2_root.iterdir()
            if p.is_dir() and p.name.lower().endswith(folder_suffix.lower())
        ),
        key=lambda p: p.name.lower(),
    )
    for qdir in query_dirs:
        candidates: List[Path] = []
        for pattern in ("*.jsonl", "*.json"):
            candidates.extend(qdir.glob(pattern))
        candidates = [p for p in sorted(candidates, key=lambda p: p.name.lower()) if not _looks_like_summary_file(p)]
        if not candidates:
            continue
        pairs.append((qdir.name, candidates[0]))
    return pairs


def _build_output_path(output_dir: Path, input_file: Path) -> Path:
    stem = input_file.stem
    return output_dir / f"{stem}_stage3_output.json"


def _run_group(
    group_name: str,
    input_dir: Path,
    output_dir: Path,
    tables_meta: Path,
    require_non_empty: bool,
) -> None:
    input_files = _iter_input_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_files:
        print(f"[stage3-batch] {group_name}: no input files in {input_dir}")
        return

    print(f"[stage3-batch] {group_name}: found {len(input_files)} input file(s)")
    for idx, input_file in enumerate(input_files, start=1):
        output_file = _build_output_path(output_dir, input_file)
        print(
            f"[stage3-batch] {group_name} [{idx}/{len(input_files)}] "
            f"input={input_file.name} -> output={output_file.name}",
            flush=True,
        )
        out = run_stage3_pipeline(
            input_path=str(input_file),
            output_path=str(output_file),
            limit=None,
            offset=0,
            tables_meta_path=str(tables_meta),
            require_non_empty=require_non_empty,
        )
        print(
            f"[stage3-batch] {group_name}: wrote {len(out)} DPR rows -> {output_file}",
            flush=True,
        )

    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "group": group_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tables_meta": str(tables_meta),
        "inputs_processed": [p.name for p in input_files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[stage3-batch] {group_name}: manifest -> {manifest_path}", flush=True)


def _run_query_folder_group(
    group_name: str,
    stage2_root: Path,
    folder_suffix: str,
    output_dir: Path,
    tables_meta: Path,
    require_non_empty: bool,
) -> None:
    pairs = _iter_query_folder_inputs(stage2_root, folder_suffix=folder_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pairs:
        print(
            f"[stage3-batch] {group_name}: no matching query folders (*{folder_suffix}) with DPR json/jsonl in {stage2_root}",
            flush=True,
        )
        return

    print(f"[stage3-batch] {group_name}: found {len(pairs)} query folder input(s)", flush=True)
    processed: List[dict] = []
    for idx, (folder_name, input_file) in enumerate(pairs, start=1):
        output_file = output_dir / f"{folder_name}_stage3_output.json"
        print(
            f"[stage3-batch] {group_name} [{idx}/{len(pairs)}] "
            f"folder={folder_name} input={input_file.name} -> output={output_file.name}",
            flush=True,
        )
        out = run_stage3_pipeline(
            input_path=str(input_file),
            output_path=str(output_file),
            limit=None,
            offset=0,
            tables_meta_path=str(tables_meta),
            require_non_empty=require_non_empty,
        )
        processed.append(
            {
                "query_folder": folder_name,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "rows_written": len(out),
            }
        )
        print(
            f"[stage3-batch] {group_name}: wrote {len(out)} DPR rows -> {output_file}",
            flush=True,
        )

    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "group": group_name,
        "stage2_root": str(stage2_root),
        "folder_suffix": folder_suffix,
        "output_dir": str(output_dir),
        "tables_meta": str(tables_meta),
        "processed": processed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[stage3-batch] {group_name}: manifest -> {manifest_path}", flush=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage-3 for all query-set DPR files under offline_with_query and "
            "online_with_query input directories."
        )
    )
    parser.add_argument(
        "--offline-input-dir",
        default=str(repo_root / "data" / "stage2_outputs" / "dprs_offline_with_query"),
        help="Directory containing offline_with_query Stage-2 DPR files (query1/query2...).",
    )
    parser.add_argument(
        "--online-input-dir",
        default=str(repo_root / "data" / "stage2_outputs" / "dprs_online_with_query"),
        help="Directory containing online_with_query Stage-2 DPR files (query1/query2...).",
    )
    parser.add_argument(
        "--online-query-root",
        default=str(repo_root / "data" / "stage2_outputs"),
        help="Stage-2 root where per-query folders like Q1--online, Q2--online exist.",
    )
    parser.add_argument(
        "--online-query-folder-suffix",
        default="--online",
        help="Folder suffix used for online query folders (default: --online).",
    )
    parser.add_argument(
        "--offline-output-dir",
        default=str(repo_root / "data" / "stage3_outputs" / "offline_with_query"),
        help="Directory where offline_with_query Stage-3 outputs should be written.",
    )
    parser.add_argument(
        "--online-output-dir",
        default=str(repo_root / "data" / "stage3_outputs" / "online_with_query"),
        help="Directory where online_with_query Stage-3 outputs should be written.",
    )
    parser.add_argument(
        "--tables-meta",
        default=str(repo_root / "data" / "stage1_outputs" / "tables_clean"),
        help="Path to Stage-1 tables metadata (tables_clean dir or tables.json).",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "offline", "online"),
        default="all",
        help="Which set(s) to process.",
    )
    parser.add_argument(
        "--require-non-empty",
        action="store_true",
        help="If set, marks empty SQL result sets as not grounded.",
    )
    args = parser.parse_args()

    offline_input_dir = Path(args.offline_input_dir).resolve()
    online_input_dir = Path(args.online_input_dir).resolve()
    online_query_root = Path(args.online_query_root).resolve()
    offline_output_dir = Path(args.offline_output_dir).resolve()
    online_output_dir = Path(args.online_output_dir).resolve()
    tables_meta = Path(args.tables_meta).resolve()

    if args.mode in ("all", "offline"):
        _run_group(
            group_name="offline_with_query",
            input_dir=offline_input_dir,
            output_dir=offline_output_dir,
            tables_meta=tables_meta,
            require_non_empty=bool(args.require_non_empty),
        )
    if args.mode in ("all", "online"):
        # Priority 1: explicit online input dir with files directly under it
        # Priority 2: per-query folders under stage2 root (e.g. Q1--online/Q2--online)
        if online_input_dir.exists() and _iter_input_files(online_input_dir):
            _run_group(
                group_name="online_with_query",
                input_dir=online_input_dir,
                output_dir=online_output_dir,
                tables_meta=tables_meta,
                require_non_empty=bool(args.require_non_empty),
            )
        else:
            _run_query_folder_group(
                group_name="online_with_query",
                stage2_root=online_query_root,
                folder_suffix=args.online_query_folder_suffix,
                output_dir=online_output_dir,
                tables_meta=tables_meta,
                require_non_empty=bool(args.require_non_empty),
            )


if __name__ == "__main__":
    main()
