"""Stage 3 — SQL generation + SQL grounding execution.

This module is designed to plug directly into **Stage 2** output (Athulya).

### Input (Stage 2)
Two supported Stage‑2 artifacts:

1) DPR list file (recommended):
   `dprs-*.json` — list of objects:
   - `dpr_id`
   - `DPR`
   - `model` (optional)
   - `ground_truth.table_uids` (e.g. ["T2","T3"])

2) Filtered clusters file:
   `filtered_clusters.json` — list of objects:
   - `dpr_id`
   - `cluster_key`
   - `tables`: list of table objects (each with `table_id`, `columns`, ...).
   This format is joined with a neighboring `dprs-*.json` to obtain DPR text.

Required table metadata:
`tables.json` (Stage‑2 input artifact), mapping `T1..T10` -> {columns, numeric_columns, ...}.

### Output (Stage 3)
JSON list, one object per DPR:
- `dpr_id`
- `DPR`
- `tables` (table_uids)
- `ground_truth`
- `generated_sql`
- `execution_status` (SQL executed successfully against schema)
- `result`: {validation, row_count, preview} or {validation, error}
- `schema_mapping` (original columns -> SQL-safe columns)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple


# Load .env from project root so GROQ_API_KEY can be set there
def _load_dotenv() -> None:
    try:
        from pathlib import Path
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
    except Exception:
        pass


_load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama2"
RETRY_DELAY_SEC = 2.0


def get_llm_client():
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with `pip install -r requirements.txt`.")
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    use_ollama = os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes")
    if use_ollama:
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL), OLLAMA_MODEL
    if groq_key:
        return OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL), GROQ_MODEL
    raise RuntimeError(
        "GROQ_API_KEY not set. Get a free key at https://console.groq.com and set it in .env or env vars."
    )


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def generate_sql(client, model: str, dpr_text: str, schema_string: str) -> str:
    prompt = f"""You are a SQL expert. Generate exactly one SQLite query that answers the following question using ONLY the tables and columns listed in the schema. Use the exact table and column names from the schema.

Schema:
{schema_string}

Question: {dpr_text}

Rules:
- Output only the SQL statement. No markdown, no explanation, no extra text.
- Use JOINs when the question involves more than one table.
- Use LIMIT when appropriate to avoid huge result sets.

SQL:"""

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return _strip_code_fence(raw)
        except Exception as e:
            err = str(e).lower()
            if attempt == 0 and ("rate" in err or "429" in err or "overloaded" in err):
                time.sleep(RETRY_DELAY_SEC)
                continue
            raise


def execute_and_validate(cursor, sql: str, require_non_empty: bool) -> Tuple[bool, List[Dict[str, Any]], Optional[str], int]:
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        if require_non_empty and not rows:
            return False, [], "Query returned no rows (empty result)", 0
        preview = [dict(zip(cols, r)) for r in rows[:5]]
        return True, preview, None, len(rows)
    except Exception as e:
        return False, [], str(e), 0


def _sql_safe_identifier(name: str) -> str:
    s = re.sub(r"\W+", "_", (name or "").strip())
    s = re.sub(r"^_+", "", s)
    if not s:
        return "col"
    if s[0].isdigit():
        s = f"col_{s}"
    if s.lower() in {
        "group",
        "order",
        "select",
        "from",
        "where",
        "join",
        "limit",
        "table",
        "by",
        "having",
        "as",
        "on",
        "and",
        "or",
        "union",
        "distinct",
        "into",
        "values",
        "create",
        "drop",
        "insert",
        "update",
        "delete",
    }:
        s = f"col_{s}"
    return s


def _build_empty_db_from_table_metadata(table_metas: Dict[str, Any]) -> Tuple[sqlite3.Connection, sqlite3.Cursor, str, Dict[str, Dict[str, str]]]:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    schema_lines: List[str] = []
    mapping: Dict[str, Dict[str, str]] = {}

    for table_id, meta in table_metas.items():
        cols = meta.get("columns") or []
        numeric_cols = set(meta.get("numeric_columns") or [])

        used: Set[str] = set()
        col_map: Dict[str, str] = {}
        col_defs: List[str] = []
        schema_cols: List[str] = []

        for c in cols:
            base = _sql_safe_identifier(str(c))
            safe = base
            if safe in used:
                k = 2
                while f"{base}_{k}" in used:
                    k += 1
                safe = f"{base}_{k}"
            used.add(safe)

            col_map[str(c)] = safe
            col_type = "REAL" if c in numeric_cols else "TEXT"
            col_defs.append(f"{safe} {col_type}")
            schema_cols.append(f"{safe} ({col_type})")

        mapping[str(table_id)] = col_map
        if not col_defs:
            col_defs = ["col TEXT"]
            schema_cols = ["col (TEXT)"]

        cursor.execute(f"DROP TABLE IF EXISTS {table_id}")
        cursor.execute(f"CREATE TABLE {table_id} ({', '.join(col_defs)})")
        schema_lines.append(f"Table: {table_id}, Columns: {', '.join(schema_cols)}")

    return conn, cursor, "\n".join(schema_lines), mapping


def _infer_tables_json(stage2_output_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(stage2_output_path))
    parent = os.path.dirname(d)
    candidate = os.path.join(parent, "input", "tables.json")
    if os.path.exists(candidate):
        return candidate
    return None


def _find_neighbor_dprs_file(path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(path))
    for pat in (os.path.join(d, "dprs-*.json"), os.path.join(os.path.dirname(d), "output", "dprs-*.json")):
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def run_stage3_pipeline(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    tables_meta_path: Optional[str] = None,
    require_non_empty: bool = False,
) -> List[Dict[str, Any]]:
    """Runs stage-3 on stage-2 artifacts."""

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError(
            "Stage‑3 expects a Stage‑2 JSON list (dprs-*.json or filtered_clusters.json)."
        )

    client, model = get_llm_client()

    # Detect which stage-2 list type we got
    first = payload[0]

    # Type A: DPR list
    if "DPR" in first and "ground_truth" in first:
        dprs_list = payload[: int(limit)] if limit is not None else payload

        tables_json = tables_meta_path or _infer_tables_json(input_path)
        if not tables_json:
            raise RuntimeError(
                "Could not infer tables.json. Pass --tables-meta <path/to/tables.json>."
            )
        with open(tables_json, "r", encoding="utf-8") as tf:
            tables_meta_all = json.load(tf)

        print(f"Using LLM: {model} | Processing {len(dprs_list)} DPRs (stage-2 dprs list)")

        out: List[Dict[str, Any]] = []
        for i, d in enumerate(dprs_list):
            dpr_id = d.get("dpr_id")
            dpr_text = d.get("DPR")
            gt = d.get("ground_truth") or {}
            table_uids = (gt.get("table_uids") or []) if isinstance(gt, dict) else []

            per_tables_meta = {tid: tables_meta_all.get(tid) for tid in table_uids if tables_meta_all.get(tid)}
            conn, cursor, schema_string, name_mapping = _build_empty_db_from_table_metadata(per_tables_meta)
            try:
                sql = generate_sql(client, model, dpr_text, schema_string)
                ok, preview, err, row_count = execute_and_validate(cursor, sql, require_non_empty=require_non_empty)
            except Exception as e:
                ok, preview, err, row_count, sql = False, [], str(e), 0, None
            finally:
                conn.close()

            out.append(
                {
                    "dpr_id": dpr_id,
                    "DPR": dpr_text,
                    "tables": table_uids,
                    "ground_truth": gt,
                    "generated_sql": sql,
                    "execution_status": ok,
                    "result": (
                        {"validation": "Success", "row_count": row_count, "preview": preview}
                        if ok
                        else {"validation": "Failed", "error": err}
                    ),
                    "schema_mapping": name_mapping,
                    "llm_model": model,
                    "upstream_model": d.get("model"),
                }
            )

            if (i + 1) % 5 == 0 or (i + 1) == len(dprs_list):
                print(f"Processed {i + 1}/{len(dprs_list)} DPRs")

        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out, wf, indent=2)
        return out

    # Type B: filtered_clusters list
    if "cluster_key" in first and "tables" in first and "dpr_id" in first:
        clusters_list = payload[: int(limit)] if limit is not None else payload

        dprs_file = _find_neighbor_dprs_file(input_path)
        if not dprs_file:
            raise RuntimeError("filtered_clusters.json requires a neighboring dprs-*.json to get DPR text.")

        with open(dprs_file, "r", encoding="utf-8") as df:
            dprs_list = json.load(df)
        dpr_by_id = {str(x.get("dpr_id")): x for x in dprs_list if isinstance(x, dict)}

        print(f"Using LLM: {model} | Processing {len(clusters_list)} DPRs (filtered_clusters)")

        out: List[Dict[str, Any]] = []
        for i, c in enumerate(clusters_list):
            dpr_id = str(c.get("dpr_id"))
            dpr_text = (dpr_by_id.get(dpr_id) or {}).get("DPR")
            gt = (dpr_by_id.get(dpr_id) or {}).get("ground_truth") or {}

            table_objs = c.get("tables") or []
            per_tables_meta = {t.get("table_id"): t for t in table_objs if isinstance(t, dict) and t.get("table_id")}
            conn, cursor, schema_string, name_mapping = _build_empty_db_from_table_metadata(per_tables_meta)
            table_uids = list(per_tables_meta.keys())

            try:
                sql = generate_sql(client, model, dpr_text, schema_string)
                ok, preview, err, row_count = execute_and_validate(cursor, sql, require_non_empty=require_non_empty)
            except Exception as e:
                ok, preview, err, row_count, sql = False, [], str(e), 0, None
            finally:
                conn.close()

            out.append(
                {
                    "dpr_id": dpr_id,
                    "DPR": dpr_text,
                    "tables": table_uids,
                    "ground_truth": gt,
                    "cluster_key": c.get("cluster_key"),
                    "generated_sql": sql,
                    "execution_status": ok,
                    "result": (
                        {"validation": "Success", "row_count": row_count, "preview": preview}
                        if ok
                        else {"validation": "Failed", "error": err}
                    ),
                    "schema_mapping": name_mapping,
                    "llm_model": model,
                    "upstream_model": (dpr_by_id.get(dpr_id) or {}).get("model"),
                }
            )

            if (i + 1) % 5 == 0 or (i + 1) == len(clusters_list):
                print(f"Processed {i + 1}/{len(clusters_list)} DPRs")

        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out, wf, indent=2)
        return out

    raise RuntimeError("Unknown Stage‑2 list format. Expected dprs-*.json or filtered_clusters.json.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: SQL generation + grounding from Stage‑2 output")
    parser.add_argument("--input", "-i", required=True, help="Stage‑2 input file (dprs-*.json or filtered_clusters.json)")
    parser.add_argument("--output", "-o", required=True, help="Stage‑3 output JSON path")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Process only first N DPRs (default: 5)")
    parser.add_argument("--tables-meta", default=None, help="Path to Stage‑2 tables.json (if not inferable)")
    parser.add_argument(
        "--require-non-empty",
        action="store_true",
        help="If set, marks empty result sets as not grounded. Default is schema-only grounding (allow empty).",
    )
    args = parser.parse_args()

    run_stage3_pipeline(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        tables_meta_path=args.tables_meta,
        require_non_empty=bool(args.require_non_empty),
    )
    print(f"Done. Wrote {args.limit} entries to {args.output}")
