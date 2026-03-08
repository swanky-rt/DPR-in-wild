"""
Stage 3 — SQL generation + SQL grounding execution.

This module is designed to plug directly into **Stage 2** output.

### Input (Stage 2)
Two supported Stage-2 artifacts:

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
`tables.json` (Stage-2 input artifact), mapping `T1..T10` -> {columns, numeric_columns, ...}.

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
GROQ_MODEL = "llama-3.1-8b-instant"
RETRY_DELAY_SEC = 2.0

# Maximum number of logical rows to sample per table when populating the
# in-memory SQLite database from T1..T10 JSON metadata.
MAX_SAMPLE_ROWS_PER_TABLE = 10

# How many rows to show the LLM as "example values" per table.
LLM_SAMPLE_ROWS_PER_TABLE = 5


def get_llm_client():
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with `pip install -r requirements.txt`.")
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
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


def _fetch_table_samples(cursor: sqlite3.Cursor, table_ids: List[str], limit: int = LLM_SAMPLE_ROWS_PER_TABLE) -> str:
    """
    Fetch a few example rows from each table (from the in-memory SQLite DB)
    to help the LLM choose REAL filter values (e.g., Role='Himself' instead
    of hallucinating Role='Weird Al Yankovic').
    """
    blocks: List[str] = []
    for tid in table_ids:
        try:
            cursor.execute(f"SELECT * FROM {tid} LIMIT {int(limit)}")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            preview = [dict(zip(cols, r)) for r in rows]
            blocks.append(f"TABLE {tid} SAMPLE ROWS:\n{json.dumps(preview, ensure_ascii=False, indent=2)}")
        except Exception:
            continue
    return "\n\n".join(blocks).strip()


def generate_subquestions(client, model: str, dpr_text: str, max_questions: int = 5) -> List[str]:
    """
    Decompose a DPR into a small set of atomic, answerable sub-questions.
    """
    prompt = f"""You are helping to decompose a data product request (DPR) into a small set of atomic sub-questions.

Given the DPR below, produce between 3 and {max_questions} concise, non-overlapping sub-questions that:
- are directly implied by the DPR,
- can each be answered independently using tabular data and SQL,
- avoid restating the full DPR or each other.

Return ONLY a valid JSON list of strings. No explanations, no extra keys, no markdown.

DPR:
\"\"\"{dpr_text}\"\"\""""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = _strip_code_fence(raw)
        subqs = json.loads(raw)
        if isinstance(subqs, list):
            return [str(q).strip() for q in subqs if isinstance(q, (str, int, float)) and str(q).strip()]
    except Exception:
        pass

    return [str(dpr_text).strip()] if dpr_text else []


def generate_sql(
    client,
    model: str,
    question: str,
    schema_string: str,
    samples_string: str = "",
) -> str:
    """
    Base SQL generator used for both full DPRs and atomic sub-questions.

    Key improvement: include a few sample rows from each table so the LLM can
    ground WHERE filters to real values.
    """
    prompt = f"""You are a SQLite SQL expert. Generate exactly one SELECT query that answers the question using ONLY the tables and columns listed in the schema. Use the exact table and column names from the schema.

Schema:
{schema_string}

Example rows (use these to choose correct filter values; do NOT invent values not shown here):
{samples_string if samples_string else "(no samples available)"}

Question: {question}

Rules:
- Output only the SQL statement. No markdown, no explanation, no extra text.
- Never invent tables or columns that are not present in the schema.
- When filtering by categorical columns (e.g., Role, Party, Nationality), prefer values seen in the example rows.
- Only use JOIN when there is a clear, shared column name between the tables (e.g. Title + Year, Member, Athlete).
- Do NOT use ON 1 = 1, CROSS JOIN, or any other cartesian join pattern.
- Prefer querying a single most relevant table if there is no obvious join key.
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


def refine_sql_with_error(
    client,
    model: str,
    question: str,
    schema_string: str,
    previous_sql: str,
    error_message: Optional[str],
    empty_result: bool,
    samples_string: str = "",
) -> str:
    """
    Self-correcting SQL generation step driven by observed errors and empty results.

    Key improvement: include sample rows so the refinement step ALSO stays grounded.
    """
    feedback_parts: List[str] = []
    if error_message:
        feedback_parts.append(f"The previous SQL resulted in an error:\n{error_message}")
    if empty_result:
        feedback_parts.append(
            "The previous SQL executed successfully but returned 0 rows. "
            "Try a broader or alternative query (for example using LIKE, wider filters, or "
            "aggregations) that still meaningfully addresses the question."
        )
    feedback_block = "\n\n".join(feedback_parts) if feedback_parts else "The previous SQL can be improved."

    prompt = f"""You are a SQL expert working in an iterative, self-correcting loop.

Your task is to REVISE the previous SQLite query so that it better answers the question,
using ONLY the tables and columns listed in the schema. Use the exact table and column names.

Schema:
{schema_string}

Example rows (use these to choose correct filter values; do NOT invent values not shown here):
{samples_string if samples_string else "(no samples available)"}

Question:
\"\"\"{question}\"\"\"

Previous SQL:
```sql
{previous_sql}
```

Feedback:
{feedback_block}

Important constraints:
- Keep the query focused on the question.
- Do NOT hallucinate new tables or columns.
- When filtering by categorical columns, prefer values shown in the example rows.
- Prefer robust joins and reasonable limits.

Return ONLY the revised SQL statement. No markdown, no explanation, no extra text.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _strip_code_fence(raw)


def summarize_subquestion_result(
    client,
    model: str,
    dpr_text: str,
    sub_question: str,
    table_uids: List[str],
    preview_rows: List[Dict[str, Any]],
) -> str:
    """
    Produce a short, grounded mini-summary for a successful sub-question result.
    """
    try:
        preview_json = json.dumps(preview_rows, indent=2, ensure_ascii=False)
    except TypeError:
        preview_json = str(preview_rows)

    prompt = f"""You are helping to summarize grounded results from SQL queries over tabular data.

Original DPR:
\"\"\"{dpr_text}\"\"\"

Sub-question:
\"\"\"{sub_question}\"\"\"

Relevant tables (IDs only): {table_uids}

Result preview (first few rows, JSON list of objects):
```json
{preview_json}
```

Write 1–2 short sentences that state ONLY what is directly supported by the result preview,
phrased as a factual finding that clearly answers the sub-question. Do not speculate beyond
the shown rows. Do not mention SQL, queries, or tables; just describe the finding.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=160,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _strip_code_fence(raw)
    except Exception:
        return ""


def generate_final_summary(client, model: str, dpr_text: str, mini_summaries: List[str]) -> str:
    """
    Synthesize all grounded mini-summaries for a DPR into a single final summary.
    """
    if not mini_summaries:
        return ""

    bullet_points = "\n".join(f"- {s}" for s in mini_summaries if s.strip())

    prompt = f"""You are helping to synthesize grounded findings from a data product request (DPR).

Original DPR:
\"\"\"{dpr_text}\"\"\"

Grounded atomic findings (each is supported by SQL over the data):
{bullet_points}

Write a concise, well-structured paragraph that:
- Directly answers the DPR using ONLY the grounded findings.
- Clearly weaves together the atomic findings into a coherent narrative.
- Does not speculate beyond what is stated.
- Does not mention SQL, queries, or tables.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _strip_code_fence(raw)
    except Exception:
        return ""


def _is_schema_error(error_message: Optional[str]) -> bool:
    if not error_message:
        return False
    msg = error_message.lower()
    return any(
        key in msg
        for key in [
            "no such table",
            "no such column",
            "has no column named",
            "mismatched input",
            "syntax error",
            "no such function",
        ]
    )


def _is_empty_result_error(error_message: Optional[str]) -> bool:
    """
    When --require-non-empty is enabled, execute_and_validate() returns ok=False
    with a specific error string for empty result sets. Treat that as retryable.
    """
    if not error_message:
        return False
    return error_message.strip().lower().startswith("query returned no rows")


def _is_cartesian_sql(sql: Optional[str]) -> bool:
    """
    Detect obvious cartesian join patterns such as ON 1=1 or CROSS JOIN.
    """
    if not sql:
        return False
    s = sql.lower()
    if " on 1=1" in s or " on 1 = 1" in s:
        return True
    if "cross join" in s:
        return True
    return False


def execute_and_validate(
    cursor, sql: str, require_non_empty: bool
) -> Tuple[bool, List[Dict[str, Any]], Optional[str], int]:
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


def _build_empty_db_from_table_metadata(
    table_metas: Dict[str, Any]
) -> Tuple[sqlite3.Connection, sqlite3.Cursor, str, Dict[str, Dict[str, str]]]:
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
            col_name = str(c)
            base = _sql_safe_identifier(col_name)
            safe = base
            if safe in used:
                k = 2
                while f"{base}_{k}" in used:
                    k += 1
                safe = f"{base}_{k}"
            used.add(safe)

            col_map[col_name] = safe

            # Heuristic: avoid treating obvious time/duration-like columns as numeric.
            lowered = col_name.lower()
            looks_like_time = any(
                key in lowered for key in ["time", "duration", "length", "minutes", "minute", "seconds", "second"]
            )

            if c in numeric_cols and not looks_like_time:
                col_type = "REAL"
            else:
                col_type = "TEXT"
            col_defs.append(f"{safe} {col_type}")
            schema_cols.append(f"{safe} ({col_type})")

        mapping[str(table_id)] = col_map
        if not col_defs:
            col_defs = ["col TEXT"]
            schema_cols = ["col (TEXT)"]

        cursor.execute(f"DROP TABLE IF EXISTS {table_id}")
        cursor.execute(f"CREATE TABLE {table_id} ({', '.join(col_defs)})")

        # Small semantics hint for a common failure mode.
        note = ""
        if any(str(c).strip().lower() == "role" for c in cols):
            note = " NOTE: 'Role' is the character/credit (e.g., 'Himself'), not the person's name."
        schema_lines.append(f"Table: {table_id}, Columns: {', '.join(schema_cols)}{note}")

        # Populate the in-memory tables with a small sample of rows if row data is available.
        rows_meta = meta.get("rows") or []
        logical_rows: List[Dict[str, Any]] = []

        if rows_meta:
            sample = rows_meta[0]
            # Case 1: row-wise dicts
            if isinstance(sample, dict) and any(k in sample for k in cols):
                for r in rows_meta[:MAX_SAMPLE_ROWS_PER_TABLE]:
                    if not isinstance(r, dict):
                        continue
                    row_obj: Dict[str, Any] = {}
                    for c in cols:
                        row_obj[str(c)] = r.get(str(c))
                    logical_rows.append(row_obj)
            # Case 2: flattened cell list (HybridQA-style)
            elif isinstance(sample, dict) and "value" in sample and cols:
                step = len(cols)
                limit = min(len(rows_meta), MAX_SAMPLE_ROWS_PER_TABLE * step)
                for i in range(0, limit, step):
                    chunk = rows_meta[i : i + step]
                    if len(chunk) != step:
                        break
                    row_obj = {}
                    for col_name, cell in zip(cols, chunk):
                        if isinstance(cell, dict):
                            row_obj[str(col_name)] = cell.get("value")
                    logical_rows.append(row_obj)

        if logical_rows:
            safe_cols = [col_map[str(c)] for c in cols]
            placeholders = ", ".join(["?"] * len(safe_cols))
            col_list_sql = ", ".join(safe_cols)
            insert_sql = f"INSERT INTO {table_id} ({col_list_sql}) VALUES ({placeholders})"
            for r in logical_rows:
                values = [r.get(str(c)) for c in cols]
                cursor.execute(insert_sql, values)

    conn.commit()
    return conn, cursor, "\n".join(schema_lines), mapping


def _load_tables_meta(path: str) -> Dict[str, Any]:
    """
    Load table metadata for Stage 3.

    Supports:
    - A single JSON file: tables.json mapping table_id -> meta.
    - A directory containing one JSON per table.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        metas: Dict[str, Any] = {}
        for fname in sorted(os.listdir(path)):
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    table_id = meta.get("table_id") or os.path.splitext(fname)[0]
                    metas[str(table_id)] = meta
            except Exception:
                continue
        return metas

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    raise RuntimeError(f"Unsupported tables meta format at {path!r}. Expected dict or directory of JSON files.")


def _infer_tables_json(stage2_output_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(stage2_output_path))
    parent = os.path.dirname(d)
    candidate = os.path.join(parent, "input", "tables.json")
    if os.path.exists(candidate):
        return candidate
    return None


def _find_neighbor_dprs_file(path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(path))
    for pat in (
        os.path.join(d, "dprs-*.json"),
        os.path.join(os.path.dirname(d), "output", "dprs-*.json"),
    ):
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
        raise RuntimeError("Stage-3 expects a Stage-2 JSON list (dprs-*.json or filtered_clusters.json).")

    client, model = get_llm_client()
    first = payload[0]

    # ------------------------
    # Type A: DPR list
    # ------------------------
    if "DPR" in first and "ground_truth" in first:
        dprs_list = payload[: int(limit)] if limit is not None else payload

        tables_meta_root = tables_meta_path or _infer_tables_json(input_path)
        if not tables_meta_root:
            raise RuntimeError(
                "Could not infer table metadata. Pass --tables-meta <path/to/tables.json or tables_clean dir>."
            )

        tables_meta_all = _load_tables_meta(tables_meta_root)
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
                sub_questions = generate_subquestions(client, model, dpr_text)

                subquery_results: List[Dict[str, Any]] = []
                all_mini_summaries: List[str] = []

                any_success = False
                first_success_preview: List[Dict[str, Any]] = []
                first_success_row_count = 0
                representative_sql: Optional[str] = None
                last_error: Optional[str] = None

                for sub_q in sub_questions:
                    attempts: List[Dict[str, Any]] = []
                    best_sql: Optional[str] = None
                    best_preview: List[Dict[str, Any]] = []
                    best_row_count = 0
                    best_ok = False
                    mini_summary = ""

                    samples_string = _fetch_table_samples(cursor, table_uids, limit=LLM_SAMPLE_ROWS_PER_TABLE)

                    sql: Optional[str] = None
                    err: Optional[str] = None
                    row_count = 0

                    for attempt_idx in range(3):
                        if attempt_idx == 0 or not sql:
                            sql = generate_sql(client, model, sub_q, schema_string, samples_string=samples_string)
                        else:
                            empty_result = (row_count == 0) and (err is None or _is_empty_result_error(err))
                            sql = refine_sql_with_error(
                                client,
                                model,
                                sub_q,
                                schema_string,
                                sql,
                                err,
                                empty_result=empty_result,
                                samples_string=samples_string,
                            )

                        if _is_cartesian_sql(sql):
                            ok = False
                            preview = []
                            err = "Detected disallowed cartesian join pattern (e.g., ON 1=1 or CROSS JOIN)."
                            row_count = 0
                        else:
                            ok, preview, err, row_count = execute_and_validate(
                                cursor, sql, require_non_empty=require_non_empty
                            )

                        attempts.append({"sql": sql, "execution_status": ok, "error": err, "row_count": row_count})

                        if ok and row_count > 0:
                            best_sql = sql
                            best_preview = preview
                            best_row_count = row_count
                            best_ok = True
                            break

                        should_retry = (
                            _is_schema_error(err)
                            or _is_cartesian_sql(sql)
                            or _is_empty_result_error(err)
                            or (ok and row_count == 0)
                        )
                        if not should_retry:
                            break

                    if best_ok:
                        any_success = True
                        if representative_sql is None:
                            representative_sql = best_sql
                            first_success_preview = best_preview
                            first_success_row_count = best_row_count

                        mini_summary = summarize_subquestion_result(
                            client, model, dpr_text, sub_q, table_uids, best_preview
                        )
                        if mini_summary:
                            all_mini_summaries.append(mini_summary)

                    last_error = err

                    subquery_results.append(
                        {
                            "sub_question": sub_q,
                            "attempts": attempts,
                            "final_sql": best_sql,
                            "final_execution_status": best_ok,
                            "final_row_count": best_row_count,
                            "mini_summary": mini_summary,
                        }
                    )

                if any_success:
                    ok = True
                    preview = first_success_preview
                    row_count = first_success_row_count
                    err = None
                    sql = representative_sql
                else:
                    ok = False
                    preview = []
                    row_count = 0
                    sql = representative_sql
                    err = last_error or "No successful sub-queries were executed."

                final_summary = generate_final_summary(client, model, dpr_text, all_mini_summaries)

            except Exception as e:
                ok, preview, err, row_count, sql, sub_questions, subquery_results, all_mini_summaries, final_summary = (
                    False,
                    [],
                    str(e),
                    0,
                    None,
                    [],
                    [],
                    [],
                    "",
                )
            finally:
                conn.close()

            out.append(
                {
                    "dpr_id": dpr_id,
                    "DPR": dpr_text,
                    "sub_questions": sub_questions,
                    "subquery_results": subquery_results,
                    "mini_summaries": all_mini_summaries,
                    "final_summary": final_summary,
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

    # ------------------------
    # Type B: filtered_clusters list
    # ------------------------
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
                sub_questions = generate_subquestions(client, model, dpr_text)

                subquery_results: List[Dict[str, Any]] = []
                all_mini_summaries: List[str] = []

                any_success = False
                first_success_preview: List[Dict[str, Any]] = []
                first_success_row_count = 0
                representative_sql: Optional[str] = None
                last_error: Optional[str] = None

                for sub_q in sub_questions:
                    attempts: List[Dict[str, Any]] = []
                    best_sql: Optional[str] = None
                    best_preview: List[Dict[str, Any]] = []
                    best_row_count = 0
                    best_ok = False
                    mini_summary = ""

                    samples_string = _fetch_table_samples(cursor, table_uids, limit=LLM_SAMPLE_ROWS_PER_TABLE)

                    sql: Optional[str] = None
                    err: Optional[str] = None
                    row_count = 0

                    for attempt_idx in range(3):
                        if attempt_idx == 0 or not sql:
                            sql = generate_sql(client, model, sub_q, schema_string, samples_string=samples_string)
                        else:
                            empty_result = (row_count == 0) and (err is None or _is_empty_result_error(err))
                            sql = refine_sql_with_error(
                                client,
                                model,
                                sub_q,
                                schema_string,
                                sql,
                                err,
                                empty_result=empty_result,
                                samples_string=samples_string,
                            )

                        if _is_cartesian_sql(sql):
                            ok = False
                            preview = []
                            err = "Detected disallowed cartesian join pattern (e.g., ON 1=1 or CROSS JOIN)."
                            row_count = 0
                        else:
                            ok, preview, err, row_count = execute_and_validate(
                                cursor, sql, require_non_empty=require_non_empty
                            )

                        attempts.append({"sql": sql, "execution_status": ok, "error": err, "row_count": row_count})

                        if ok and row_count > 0:
                            best_sql = sql
                            best_preview = preview
                            best_row_count = row_count
                            best_ok = True
                            break

                        should_retry = (
                            _is_schema_error(err)
                            or _is_cartesian_sql(sql)
                            or _is_empty_result_error(err)
                            or (ok and row_count == 0)
                        )
                        if not should_retry:
                            break

                    if best_ok:
                        any_success = True
                        if representative_sql is None:
                            representative_sql = best_sql
                            first_success_preview = best_preview
                            first_success_row_count = best_row_count

                        mini_summary = summarize_subquestion_result(
                            client, model, dpr_text, sub_q, table_uids, best_preview
                        )
                        if mini_summary:
                            all_mini_summaries.append(mini_summary)

                    last_error = err

                    subquery_results.append(
                        {
                            "sub_question": sub_q,
                            "attempts": attempts,
                            "final_sql": best_sql,
                            "final_execution_status": best_ok,
                            "final_row_count": best_row_count,
                            "mini_summary": mini_summary,
                        }
                    )

                if any_success:
                    ok = True
                    preview = first_success_preview
                    row_count = first_success_row_count
                    err = None
                    sql = representative_sql
                else:
                    ok = False
                    preview = []
                    row_count = 0
                    sql = representative_sql
                    err = last_error or "No successful sub-queries were executed."

                final_summary = generate_final_summary(client, model, dpr_text, all_mini_summaries)

            except Exception as e:
                ok, preview, err, row_count, sql, sub_questions, subquery_results, all_mini_summaries, final_summary = (
                    False,
                    [],
                    str(e),
                    0,
                    None,
                    [],
                    [],
                    [],
                    "",
                )
            finally:
                conn.close()

            out.append(
                {
                    "dpr_id": dpr_id,
                    "DPR": dpr_text,
                    "sub_questions": sub_questions,
                    "subquery_results": subquery_results,
                    "mini_summaries": all_mini_summaries,
                    "final_summary": final_summary,
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

    raise RuntimeError("Unknown Stage-2 list format. Expected dprs-*.json or filtered_clusters.json.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: SQL generation + grounding from Stage-2 output")
    parser.add_argument("--input", "-i", required=True, help="Stage-2 input file (dprs-*.json or filtered_clusters.json)")
    parser.add_argument("--output", "-o", required=True, help="Stage-3 output JSON path")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Process only first N DPRs (default: 5)")
    parser.add_argument("--tables-meta", default=None, help="Path to Stage-2 tables.json (if not inferable)")
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
