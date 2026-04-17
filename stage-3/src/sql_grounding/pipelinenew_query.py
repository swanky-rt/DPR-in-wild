"""
Stage 3 — SQL generation + SQL grounding execution.

This module is designed to plug directly into **Stage 2** output.

### Input (Stage 2)
**DPR list only** — `dprs-*.json` (JSON array) or `dprs-*.jsonl` (one object per line).
Each row must include:
   - `dpr_id`
   - `DPR`
   - `model` (optional)
   - `ground_truth.table_uids` (cluster table ids, e.g. ["T2","T3"])

Required table metadata (separate from Stage 2 rows):
`tables.json` (Stage-2 input artifact), mapping `T1..T100` -> {columns, numeric_columns, ...},
or a directory like `tables_clean/` with one JSON file per table.

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
- `sub_questions`, `subquery_results`, `mini_summaries`, `final_summary` (synthesis from executed rows)

**Execution vs evaluation:** Stage 3 stops at successful SQL execution and evidence-bound synthesis.
`execution_status` plus `result.row_count` / `preview` are the in-pipeline execution signals; LLM-based
quality/relevance/clarity metrics belong in **Stage 4**.

**Grounding check:** For each DPR, sub-questions drive `SELECT`s against an in-memory DB loaded
with cluster tables. Successful execution with non-empty rows supports mini-summaries.
`SELECT * FROM T LIMIT n` probe/fallback queries are **not** treated as answering a sub-question;
the loop refines until a non-trivial query is produced or attempts are exhausted.

**API failures (transient 429 / TPM):** Retries use short capped sleeps. **Tokens-per-day (TPD)** hits
**fail fast**: the run stops immediately, logs the error, and writes **completed** DPRs only (no
multi-hour sleeps). Within a DPR, earlier sub-question progress is still preserved when a later call
fails with a normal error (not TPD).

**Schema grounding:** Sub-questions and SQL prompts include an **allowed column whitelist** parsed from
the schema string. Sub-questions that reference risky tokens (e.g. year/date/rank) with no matching
column may be **skipped** (`skipped: true`). SQL uses up to **MAX_SQL_ATTEMPTS** retries per sub-question.
Fallback SQL uses **concrete column names** (not `SELECT *`) so it is not treated as a trivial probe.

**Truth / anti-hallucination:** Decomposition prompts discourage lazy generic sub-questions and ask for
preserved analytical intent when columns exist. SQL prompts bias toward **LIKE** / fuzzy text filters.
Refinement after **0 rows** steers toward discovery (MIN/MAX), LIKE, or another table—not meaningless
`SELECT *`. Mini- and final summaries are **evidence-constrained** (only facts supported by result rows).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict


# Load .env from project root.
# Supported LLM envs (OpenAI-compatible gateway):
# - LLM_API_KEY
# - LLM_API_BASE
# - LLM_MODEL
# override=True: values in .env win over stale shell / OS / IDE vars (dotenv default is False).
def _load_dotenv() -> None:
    try:
        from pathlib import Path
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env", override=True)
    except Exception:
        pass


_load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__end__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


RETRY_DELAY_SEC = 2.0
MAX_API_RETRIES = 6
# How many rows to show the LLM as "example values" per table (prompt only).
# The in-memory SQLite DB is populated with the full row list from each
# cluster table JSON under data/stage1_outputs/tables_clean (not all ~100 tables).
LLM_SAMPLE_ROWS_PER_TABLE = 3
# Truncate long sample cell text to keep prompts under TPM budget.
LLM_SAMPLE_VALUE_MAX_CHARS = 100
# Cap DPR length in LLM prompts only (full DPR still stored in output) to reduce TPD/TPM usage.
LLM_DPR_PROMPT_MAX_CHARS = 4000
# Cap sleep duration for TPM-style rate limits ("try again in XmYs" / seconds).
MAX_RATE_LIMIT_SLEEP_SEC = 60.0


class TokensPerDayExhaustedError(RuntimeError):
    """Daily token / hard quota exhausted; do not sleep or retry."""

# SQL generation / refinement attempts per sub-question (typed retry policy).
# With MAX_SQL_ATTEMPTS=3: 0 generate → 1 simplify/drop join → 2 adaptive (discovery OR alternate-table pivot).
MAX_SQL_ATTEMPTS = 3

# Minimum successful sub-questions required for DPR execution success.
# Recommended: 2 (for 2-3 generated sub-questions).
try:
    STAGE3_MIN_SUBQ_SUCCESSES = max(1, int(os.environ.get("STAGE3_MIN_SUBQ_SUCCESSES", "2")))
except ValueError:
    STAGE3_MIN_SUBQ_SUCCESSES = 2

# Defensive execution limits for SQLite queries.
# - Progress handler wall-time timeout per SQL statement (helps avoid implicit cartesian joins)
# - Fetch-capping prevents pulling huge result sets into Python memory.
SQL_EXECUTION_WALLTIME_SEC = 5.0
SQL_PROGRESS_HANDLER_OPCODE_INTERVAL = 20000
SQL_MAX_FETCH_ROWS = 6

# Optional pacing between DPRs to reduce bursty API traffic.
# Can be overridden with env STAGE3_DPR_DELAY_SEC (e.g. 20).
try:
    DPR_PROCESS_DELAY_SEC = max(0.0, float(os.environ.get("STAGE3_DPR_DELAY_SEC", "20")))
except ValueError:
    DPR_PROCESS_DELAY_SEC = 10.0

# Hybrid SQL proposal: lightweight LLM JSON router chooses generate_sql vs refine_sql (see propose_sql node).
# Set STAGE3_SQL_ACTION_ROUTER=0 to skip the extra call and always refine after a failed execution.
try:
    STAGE3_SQL_ACTION_ROUTER = os.environ.get("STAGE3_SQL_ACTION_ROUTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
except Exception:
    STAGE3_SQL_ACTION_ROUTER = True

# Sub-question feasibility: tokens that often refer to columns; if absent from schema, skip or warn.
_COLUMN_RISK_STEMS = frozenset(
    {
        "year",
        "years",
        "date",
        "dates",
        "rank",
        "month",
        "months",
        "week",
        "weeks",
        "day",
        "days",
        "hour",
        "minute",
        "second",
        "quarter",
        "season",
        "decade",
        "timestamp",
    }
)
_QUESTION_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "for",
        "and",
        "or",
        "not",
        "in",
        "on",
        "at",
        "to",
        "of",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "where",
        "why",
        "how",
        "when",
        "if",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "you",
        "he",
        "she",
        "data",
        "dataset",
        "table",
        "tables",
        "row",
        "rows",
        "film",
        "films",
        "movie",
        "movies",
        "show",
        "music",
        "chart",
        "charts",
        "list",
        "using",
        "use",
        "based",
        "each",
        "all",
        "any",
        "some",
        "such",
        "than",
        "into",
        "about",
        "over",
    }
)


def get_llm_client():
    if OpenAI is None:
        raise RuntimeError("Missing dependency: openai. Install with `pip install -r requirements.txt`.")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_API_BASE", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not (api_key and base_url and model):
        raise RuntimeError(
            "Set LLM_API_KEY, LLM_API_BASE, and LLM_MODEL in .env or env vars "
            "(OpenAI-compatible endpoint, e.g. campus LiteLLM / Unity local)."
        )
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/")), model


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()

def _strip_reasoning_tags(text: str) -> str:
    """
    Strip reasoning/thinking tags emitted by reasoning models.
    Safe no-op on Llama output. Does NOT touch code fences.
    """
    s = text or ""
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", s, flags=re.IGNORECASE | re.DOTALL)
    return s.strip()


def _clean_llm_prose_response(text: str) -> str:
    """
    Final user-facing text from models that emit reasoning (Qwen, etc.).
    """
    s = _strip_reasoning_tags(text or "")
    s = _strip_code_fence(s)
    s = s.strip()
    # Remove common preamble lines left after tag stripping
    s = re.sub(
        r"^(Okay,|Sure,|Let me|I'll |I will )[^\n]+\n+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = s.strip()
    # Drop first paragraph if it is clearly meta-commentary, keep factual rest
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if len(paras) > 1 and re.match(
        r"^(Okay,|Sure,|Let me|Looking|The user|I need|Wait,)",
        paras[0],
        re.IGNORECASE,
    ):
        s = "\n\n".join(paras[1:])
    return s.strip()


def _sql_is_trivial_probe(sql: Optional[str]) -> bool:
    """
    True if SQL is only SELECT * FROM one table (+ LIMIT), with no WHERE/JOIN/GROUP BY.
    Such queries are used as schema probes / fallbacks and must not count as answering
    a sub-question (otherwise the agent stops without a real query).
    """
    if not sql or not str(sql).strip():
        return True
    s = re.sub(r"\s+", " ", sql.strip())
    if re.search(r"\b(WHERE|JOIN|GROUP\s+BY|HAVING|UNION|INTERSECT)\b", s, re.IGNORECASE):
        return False
    # Optional DISTINCT on * still trivial
    return bool(
        re.match(
            r"^SELECT\s+(DISTINCT\s+)?\*\s+FROM\s+[A-Za-z0-9_]+(\s+LIMIT\s+\d+)?\s*;?\s*$",
            s,
            re.IGNORECASE,
        )
    )


def _parse_schema_tables_columns(schema_string: str) -> Dict[str, List[str]]:
    """
    Parse lines like: Table: T8, Columns: International_market (TEXT), Film (TEXT), ...
    Returns table_id -> [SQL-safe column names].
    """
    table_cols: Dict[str, List[str]] = {}
    for raw_line in (schema_string or "").split("\n"):
        line = raw_line.strip()
        if not line.startswith("Table:"):
            continue
        if " NOTE:" in line:
            line = line.split(" NOTE:")[0].strip()
        m = re.match(r"Table:\s*([A-Za-z0-9_]+)\s*,\s*Columns:\s*(.*)", line)
        if not m:
            continue
        table_id = m.group(1)
        rest = m.group(2)
        cols = re.findall(r"([A-Za-z_][\w]*)\s*\(", rest)
        table_cols[table_id] = cols
    return table_cols


def _allowed_columns_text(schema_string: str) -> str:
    """Human-readable whitelist for LLM prompts."""
    tc = _parse_schema_tables_columns(schema_string)
    if not tc:
        return "(no columns parsed from schema string)"
    lines = [f"{t}: {', '.join(cols)}" for t, cols in sorted(tc.items())]
    return "\n".join(lines)


def _all_known_sql_columns(schema_string: str) -> Set[str]:
    known: Set[str] = set()
    for cols in _parse_schema_tables_columns(schema_string).values():
        for c in cols:
            known.add(c.lower())
    return known


def _token_matches_known_column(token: str, known: Set[str]) -> bool:
    t = token.lower()
    if t in known:
        return True
    if len(t) > 2 and t.endswith("s") and t[:-1] in known:
        return True
    return False


def _question_references_unknown_risk_columns(question: str, schema_string: str) -> bool:
    """
    Heuristic: question mentions common column-concept words (year, date, rank, …)
    that do not appear as columns in the cluster schema — likely impossible to answer as stated.
    """
    known = _all_known_sql_columns(schema_string)
    table_ids = set(_parse_schema_tables_columns(schema_string).keys())
    for raw in re.findall(r"\b[A-Za-z_][\w]*\b", question or ""):
        w = raw.lower()
        if w in _QUESTION_STOPWORDS or w in table_ids:
            continue
        if w not in _COLUMN_RISK_STEMS:
            continue
        if _token_matches_known_column(w, known):
            continue
        return True
    return False


def _first_sql_select_or_cte_start(s: str) -> Optional[int]:
    """
    First real SQL start. English 'with ...' is not SQLite WITH; require
    WITH name AS ( for CTEs.
    """
    sel = re.search(r"(?is)\bSELECT\b", s)
    cte = re.search(
        r"(?is)\bWITH\s+(?:RECURSIVE\s+)?[A-Za-z_][\w]*\s+AS\s*\(",
        s,
    )
    positions = [m.start() for m in (sel, cte) if m]
    return min(positions) if positions else None


def _trim_trailing_prose_after_sql(sql: str) -> str:
    t = sql.strip()
    for split_re in (
        r"\n\n+(?:First|Here|This |The |Note |However|But |So |Looking|Wait )",
        r"\n\n+[A-Za-z][^\n]{20,}",
    ):
        parts = re.split(split_re, t, maxsplit=1)
        if len(parts) > 1 and len(parts[0]) > 20:
            t = parts[0].strip()
            break
    return t


def _is_valid_sql_start(sql: str) -> bool:
    t = (sql or "").lstrip()
    if re.match(r"^select\b", t, flags=re.IGNORECASE):
        return True
    if re.match(r"^with\b", t, flags=re.IGNORECASE):
        return bool(
            re.match(
                r"^with\s+(?:recursive\s+)?[A-Za-z_][\w]*\s+as\s*\(",
                t,
                flags=re.IGNORECASE,
            )
        )
    return False


def _extract_sql_candidate(text: str) -> str:
    """
    Normalize model output and extract the SQL statement.
    Handles cases where reasoning models prepend <think>...</think>.
    """
    s = _strip_reasoning_tags(text or "")
    s = _strip_code_fence(s)
    s = s.strip()

    start = _first_sql_select_or_cte_start(s)
    if start is not None:
        s = s[start:].strip()
    s = _trim_trailing_prose_after_sql(s)
    # Keep exactly one SQL statement. If additional non-whitespace appears after the first
    # semicolon, truncate to the first statement to avoid multi-statement output/prose bleed.
    if ";" in s:
        first, rest = s.split(";", 1)
        if rest.strip():
            s = first.strip()

    return s.strip()


def _fallback_sql_from_schema(schema_string: str) -> str:
    """
    Safe fallback when the model fails to produce executable SQL.
    Uses concrete column names (not SELECT *) so it is not rejected as a trivial probe.
    """
    tc = _parse_schema_tables_columns(schema_string)
    if not tc:
        return "SELECT 1"
    m = re.search(r"Table:\s*([A-Za-z0-9_]+)\s*,\s*Columns:", schema_string)
    table = m.group(1) if m else next(iter(tc.keys()))
    cols = tc.get(table, [])
    if not cols:
        return "SELECT 1 AS fallback_value LIMIT 1"
    take = cols[:2]
    col_sql = ", ".join(take)
    return f"SELECT {col_sql} FROM {table} LIMIT 5"


def _truncate_for_llm(text: Optional[str], max_chars: int = LLM_DPR_PROMPT_MAX_CHARS) -> str:
    """Shorten long DPR text for prompts only (keeps tail context)."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 30] + "\n[...truncated for API size...]\n" + t[-400:]


def _extract_retry_wait_seconds(err_text: str) -> float:
    """
    Parse provider error text for suggested retry delay, e.g. "try again in 28.59s"
    or "try again in 41m35.232s". Capped to avoid multi-hour sleeps inside retries.
    """
    m = re.search(r"try again in\s+(\d+)m(\d+(?:\.\d+)?)s", err_text, flags=re.IGNORECASE)
    if m:
        try:
            sec = int(m.group(1)) * 60 + float(m.group(2)) + 0.5
            return min(MAX_RATE_LIMIT_SLEEP_SEC, max(RETRY_DELAY_SEC, sec))
        except Exception:
            pass
    m = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", err_text, flags=re.IGNORECASE)
    if not m:
        return RETRY_DELAY_SEC
    try:
        return min(MAX_RATE_LIMIT_SLEEP_SEC, max(RETRY_DELAY_SEC, float(m.group(1)) + 0.5))
    except Exception:
        return RETRY_DELAY_SEC


def _chat_with_retry(client, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
    """
    Wrapper around chat completion with robust retry/backoff for rate limits and transient errors.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_exc = e
            err = str(e).lower()
            # Hard quota: fail fast (tokens-per-day, billing, etc.).
            if (
                "tokens per day" in err
                or ("tpd" in err and "limit" in err)
                or "insufficient_quota" in err
                or "quota exceeded" in err
            ):
                raise TokensPerDayExhaustedError(str(e)) from e
            retryable = any(k in err for k in ["rate", "429", "overloaded", "timeout", "temporarily"])
            if not retryable or attempt == MAX_API_RETRIES - 1:
                raise
            wait_s = _extract_retry_wait_seconds(str(e))
            time.sleep(wait_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown chat completion failure.")


def _fetch_table_samples(cursor: sqlite3.Cursor, table_ids: List[str], limit: int = LLM_SAMPLE_ROWS_PER_TABLE) -> str:
    """
    Fetch a few example rows from each table (from the in-memory SQLite DB)
    to help the LLM choose REAL filter values (e.g., Role='Himself' instead
    of hallucinating Role='Weird Al Yankovic').

    Call **once per DPR** (after the cluster DB is built): the hint string is the same
    for every sub-question, so it should not be recomputed inside the sub-question loop.
    """
    blocks: List[str] = []
    for tid in table_ids:
        try:
            cursor.execute(f"SELECT * FROM {tid} LIMIT {int(limit)}")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            preview: List[Dict[str, Any]] = []
            for r in rows:
                obj = dict(zip(cols, r))
                compact: Dict[str, Any] = {}
                for k, v in obj.items():
                    if isinstance(v, str):
                        s = v.strip().replace("\n", " ")
                        if len(s) > LLM_SAMPLE_VALUE_MAX_CHARS:
                            s = s[:LLM_SAMPLE_VALUE_MAX_CHARS] + "..."
                        compact[k] = s
                    else:
                        compact[k] = v
                preview.append(compact)
            blocks.append(f"TABLE {tid} SAMPLE ROWS:\n{json.dumps(preview, ensure_ascii=False, indent=2)}")
        except Exception:
            continue
    return "\n\n".join(blocks).strip()


def _parse_subquestions_json(raw: str) -> Optional[List[str]]:
    """Parse strict {{\"questions\": [...]}} or legacy JSON list."""
    s = _strip_code_fence((raw or "").strip())
    s = _strip_reasoning_tags(s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "questions" in obj:
            qs = obj["questions"]
            if isinstance(qs, list):
                out = [str(x).strip() for x in qs if str(x).strip()]
                return out or None
        if isinstance(obj, list):
            out = [str(x).strip() for x in obj if str(x).strip()]
            return out or None
    except Exception:
        pass
    m = re.search(r"\{\s*\"questions\"\s*:\s*\[[\s\S]*?\]\s*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            qs = obj.get("questions")
            if isinstance(qs, list):
                return [str(x).strip() for x in qs if str(x).strip()]
        except Exception:
            pass
    return None


def _sanitize_subquestions(
    subqs: List[str],
    dpr_text: str,
    table_uids: List[str],
    min_questions: int,
    max_questions: int,
) -> List[str]:
    """Drop heading fragments and over-long blobs; refill from heuristics if needed."""
    out: List[str] = []
    for q in subqs:
        t = str(q).strip()
        if len(t) < 8:
            continue
        if t.endswith(":") and len(t) < 100:
            continue
        if len(t) > 350:
            t = t[:350].rsplit(" ", 1)[0] + "..."
        out.append(t)
        if len(out) >= max_questions:
            break
    if len(out) >= min_questions:
        return out[:max_questions]

    fallback = _fallback_atomic_questions(dpr_text, table_uids, max_questions)[:max_questions]
    if not out:
        return fallback

    # Keep any already-produced subqs, and top up with fallback (while avoiding exact duplicates).
    merged: List[str] = []
    seen: Set[str] = set()
    for x in out + fallback:
        xx = str(x).strip()
        if not xx or xx in seen:
            continue
        merged.append(xx)
        seen.add(xx)
        if len(merged) >= max_questions:
            break
    return merged[:max_questions]


def _fallback_atomic_questions(dpr_text: str, table_uids: List[str], max_questions: int) -> List[str]:
    """Heuristic decomposition when the model returns one blob or invalid JSON."""
    t = (dpr_text or "").strip()
    if not t:
        return []
    # Numbered list items (1. ... 2. ...)
    parts = re.split(r"\n\s*\d+\.\s+", t)
    if len(parts) > 1:
        out = [p.strip() for p in parts if p.strip() and len(p.strip()) > 15]
        if len(out) >= 2:
            return out[:max_questions]
    sentences = re.split(r"(?<=[.!?])\s+", t)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
    if len(sentences) >= 2:
        return sentences[:max_questions]
    if table_uids:
        return [
            f"Summarize rows and key values in table {uid} relevant to the DPR."
            for uid in table_uids[:max_questions]
        ]
    return [t]


def _normalize_question_tokens_for_overlap(text: str) -> Set[str]:
    """
    Token set for lightweight semantic dedupe + schema overlap scoring.
    """
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9_]+", " ", t)
    toks = [x.strip() for x in t.split() if x.strip()]
    return {x for x in toks if x not in _QUESTION_STOPWORDS and len(x) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _score_subquestion(sub_question: str, known_columns: Set[str]) -> float:
    """
    Cheap score to select higher-quality sub-questions.

    Important: zero schema overlap should not be a hard removal.
    """
    q = (sub_question or "").strip()
    if not q:
        return float("-inf")
    if len(q) > 300:
        # Hard filter to avoid long/verbose blobs that waste downstream tokens.
        return float("-inf")

    q_lower = q.lower()
    tokens = _normalize_question_tokens_for_overlap(q)

    overlap_matches = 0
    for tok in tokens:
        if _token_matches_known_column(tok, known_columns):
            overlap_matches += 1

    # Prefer constraint-heavy questions.
    constraint_keywords = [
        "between",
        "range",
        "highest",
        "lowest",
        "max",
        "min",
        "top",
        "bottom",
        "average",
        "mean",
        "median",
        "count",
        "number of",
        "sum",
        "total",
        "compare",
        "difference",
        "within",
        "earliest",
        "latest",
    ]
    constraint_score = sum(1 for kw in constraint_keywords if kw in q_lower)

    # Heuristic penalties for list-all style decompositions.
    generic_penalty = 0.0
    if re.search(r"\b(list|show|what)\b.*\b(all|every|any|rows|titles)\b", q_lower):
        generic_penalty -= 5.0
    if re.search(r"\bwhat\s+titles?\s+are\s+in\s+t\d+\b", q_lower):
        generic_penalty -= 4.0
    if "summarize rows" in q_lower:
        generic_penalty -= 2.0
    if "relevant to the dpr" in q_lower:
        generic_penalty -= 1.0

    # Mild preference for mid-length questions.
    length_boost = 0.0
    if 45 <= len(q) <= 220:
        length_boost = 1.0
    elif len(q) < 45:
        length_boost = -0.5

    # Core score: schema overlap + constraints + heuristics.
    return (overlap_matches * 1.8) + (constraint_score * 0.9) + generic_penalty + length_boost


def _quality_select_subquestions(
    subqs: List[str],
    schema_string: str,
    max_questions: int,
) -> List[str]:
    """
    Cheap rule-based scoring + semantic dedupe.

    Safety:
    - If <2 survive, fall back to the original first 2.
    - "no schema overlap" yields low score but does not get hard-filtered.
    """
    original = [str(x).strip() for x in (subqs or []) if str(x).strip()]
    if not original:
        return []

    top_k = min(3, max_questions, len(original))
    if top_k <= 0:
        return original[:max_questions]

    known_columns = _all_known_sql_columns(schema_string)

    scored: List[Tuple[float, str, Set[str]]] = []
    for q in original:
        score = _score_subquestion(q, known_columns)
        if score == float("-inf"):
            continue
        toks = _normalize_question_tokens_for_overlap(q)
        scored.append((score, q, toks))

    # If everything got filtered as -inf (e.g., all >300 chars), keep original.
    if len(scored) < 2:
        return original[:max_questions] if len(original) >= 2 else original[:2]

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[str] = []
    selected_token_sets: List[Set[str]] = []
    dedupe_threshold = 0.78

    for _score, q, toks in scored:
        is_dup = any(_jaccard(toks, st) >= dedupe_threshold for st in selected_token_sets)
        if is_dup:
            continue
        selected.append(q)
        selected_token_sets.append(toks)
        if len(selected) >= top_k:
            break

    if len(selected) < 2:
        return original[:2]
    return selected[:max_questions]


def generate_subquestions(
    client,
    model: str,
    dpr_text: str,
    table_uids: Optional[List[str]] = None,
    min_questions: int = 2,
    max_questions: int = 3,
    schema_string: str = "",
) -> List[str]:
    """
    Decompose a DPR into atomic sub-questions (one simple sentence each, one table when possible).
    """
    if min_questions < 1:
        min_questions = 1
    if max_questions < min_questions:
        max_questions = min_questions

    table_uids = list(table_uids or [])
    tables_line = ", ".join(table_uids) if table_uids else "(not specified)"
    allowed = _allowed_columns_text(schema_string) if schema_string else "(not provided)"
    schema_compact = _truncate_for_llm(schema_string, 6000) if schema_string else ""

    prompt = f"""You decompose a data product request (DPR) into atomic analytical steps.

Return ONLY valid JSON with exactly this shape (no markdown, no code fences, no other keys):
{{"questions": ["...", "...", ...]}}

Requirements:
- Put between {min_questions} and {max_questions} strings in "questions".
- Each string must be ONE short sentence only (under 35 words if needed for constraints).
- **Analytical intent:** Each question must preserve the DPR's substantive goals (time windows, metrics, comparisons, entities) **when** those can be expressed using columns that exist in the schema. Do NOT replace a complex DPR with lazy inventory questions like "What titles are in T1?" unless the DPR is only about listing titles.
- **Schema truth:** Only reference filters or dimensions that appear in the allowed columns for at least one cluster table. Before mentioning Year/Date/Rank/Revenue etc., verify that column exists in the list below for the table you have in mind.
- **Data gaps:** If the DPR demands a concept no column supports (e.g. year filter but a table has no year column), ask a **specific** alternative: e.g. "What range of Year values appears in T1?" or "Which revenue-related columns exist in T8 and sample values?"—not a generic "list everything".
- Cluster table ids (for scoping): {tables_line}
- Do NOT paste the entire DPR into one bullet. Do NOT use numbered lists inside strings.

Allowed SQL column names per table (you MUST NOT require any column not listed here):
{allowed}

Schema (cluster tables, truncated if long):
{schema_compact}

DPR:
\"\"\"{_truncate_for_llm(dpr_text)}\"\"\""""

    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        raw = (resp.choices[0].message.content or "").strip()
        subqs = _parse_subquestions_json(raw)

        if not subqs:
            subqs = _fallback_atomic_questions(dpr_text, table_uids, max_questions)

        if len(subqs) == 1:
            one = subqs[0].replace("\n", " ").strip()
            dpr_norm = (dpr_text or "").replace("\n", " ").strip()
            if len(dpr_norm) > 80 and len(one) > 0.85 * len(dpr_norm):
                subqs = _fallback_atomic_questions(dpr_text, table_uids, max_questions)

        subqs = subqs[:max_questions]
        if len(subqs) < min_questions and dpr_text:
            fallback = _fallback_atomic_questions(dpr_text, table_uids, max_questions)[:max_questions]
            # Keep any already-produced subqs, and top up with fallback.
            merged: List[str] = []
            seen: Set[str] = set()
            for x in subqs + fallback:
                xx = str(x).strip()
                if not xx or xx in seen:
                    continue
                merged.append(xx)
                seen.add(xx)
                if len(merged) >= max_questions:
                    break
            subqs = merged[:max_questions]

        return _sanitize_subquestions(subqs, dpr_text, table_uids, min_questions, max_questions)
    except TokensPerDayExhaustedError:
        raise
    except Exception:
        return _sanitize_subquestions(
            _fallback_atomic_questions(dpr_text, table_uids, max_questions)[:max_questions],
            dpr_text,
            table_uids,
            min_questions,
            max_questions,
        )


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
    allowed_cols = _allowed_columns_text(schema_string)

    prompt = f"""You are a SQLite SQL expert. Generate exactly one SELECT query that answers the question using ONLY the tables and columns listed in the schema. Use the exact table and column names from the schema.

Schema:
{schema_string}

Allowed columns per table (you MUST use ONLY these names; if the question implies a missing column, answer with the closest valid columns or a broad SELECT without that filter):
{allowed_cols}

Example rows (GROUND TRUTH for literals — copy substrings from these cells for filters when possible):
{samples_string if samples_string else "(no samples available)"}

Question: {_truncate_for_llm(question, 2000)}

Rules:
- **Sample grounding:** Before choosing WHERE literals, scan the JSON sample rows above. Prefer filter values that **appear** in those cells (or obvious substrings). Do not assume column names or years that are not in the schema **and** not visible in samples.
- **Single-table default:** Prefer **one** table unless two tables share a clear same-named join key (e.g. Title + Year). Do not join unrelated entities (e.g. Composer ↔ Artist) without a documented key.
- Output ONLY one SQL statement. Start with SELECT or WITH name AS ( ... ). No English sentences, no "First", no reasoning.
- No markdown, no code fences, no text before or after the SQL.
- Never invent tables or columns that are not present in the allowed list above.
- **Categorical / text matching:** For names, countries, titles, studios, roles, etc., avoid brittle exact equality when labels may vary. Prefer `LIKE '%substring%'` or `LOWER(col) LIKE LOWER('%substring%')` using substrings suggested by the example rows (e.g. "USA" vs "United States").
- For numeric years or amounts in the question, use columns that exist; use `BETWEEN` for ranges when appropriate.
- For aggregates/sorting on numeric-like text metrics (e.g., Revenue, Admissions, Domestic_gross), coerce to numeric first, e.g. `CAST(REPLACE(REPLACE(REPLACE(col, '$',''), ',', ''), ' ', '') AS REAL)`.
- When filtering by categorical columns, copy spellings from the example rows when possible.
- Only use JOIN when there is a clear, shared column name between the tables (e.g. Title + Year, Member, Athlete).
- Do NOT use fuzzy JOIN predicates (e.g. `ON ... LIKE ...`) between semantically different columns.
- Do NOT use ON 1 = 1, CROSS JOIN, or any other cartesian join pattern.
- Prefer querying a single most relevant table if there is no obvious join key.
- If a year column is missing, do NOT filter title/name columns using year literals like `Title LIKE '%1974%'` unless the question explicitly asks for titles containing years.
- Use LIMIT when appropriate to avoid huge result sets.
- SQLite identifiers: use the exact spellings from the schema (snake_case). If a name had spaces it is already aliased—do not invent new names.

SQL:"""

    resp = _chat_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=384,
    )
    raw = (resp.choices[0].message.content or "").strip()
    sql = _extract_sql_candidate(raw)
    if not _is_valid_sql_start(sql):
        return _fallback_sql_from_schema(schema_string)
    return sql


def refine_sql_with_error(
    client,
    model: str,
    question: str,
    schema_string: str,
    previous_sql: str,
    error_message: Optional[str],
    empty_result: bool,
    samples_string: str = "",
    refine_strategy: str = "default",
    excluded_tables: Optional[Set[str]] = None,
    alternate_table_hint: Optional[str] = None,
    attempts_log: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Self-correcting SQL generation step driven by observed errors and empty results.

    Key improvement: include sample rows so the refinement step ALSO stays grounded.
    Optional attempts_log is the full per-sub-question execution history (including the
    latest failed/empty attempt) for trace-aware refinement.
    """
    feedback_parts: List[str] = []
    if error_message:
        feedback_parts.append(f"The previous SQL resulted in an error:\n{error_message}")
        if _is_execution_timeout_error(error_message):
            feedback_parts.append(
                "The previous SQL timed out (likely an implicit cartesian join or an unbounded query). "
                "Revise to be more selective: add `LIMIT`, reduce joins, and prefer a single relevant table with focused predicates."
            )
    if empty_result:
        feedback_parts.append(
            "The previous SQL executed successfully but returned **0 rows** (database-first recovery):\n"
            "- If you used `=` on a TEXT column (country, title, studio, role, etc.), retry with "
            "`LIKE '%' || literal || '%'` or match a value **copied from the example rows** (avoid "
            "'United States' vs 'USA' style mismatches).\n"
            "- If the question filters on a year/date range that may not exist in this table, run a "
            "**discovery** query first: e.g. `SELECT MIN(Year), MAX(Year) FROM Tx` if Year exists, then "
            "adjust filters to the actual range—or query a **different cluster table** that holds the metric.\n"
            "- If a year column is missing, do NOT use `Title LIKE '%1954%'` / `Film LIKE '%1974%'` style filters "
            "unless the question explicitly asks for titles containing years.\n"
            "- Remove impossible conjuncts; return a smaller scoped SELECT that still reflects the sub-question intent.\n"
            "- Do NOT answer with a meaningless `SELECT *` probe; fix the predicate or change the table."
        )
    feedback_block = "\n\n".join(feedback_parts) if feedback_parts else "The previous SQL can be improved."

    allowed_cols = _allowed_columns_text(schema_string)
    exec_trace = _format_refinement_execution_trace(attempts_log)
    trace_section = f"{exec_trace}\n" if exec_trace else ""

    prompt = f"""You are a SQL expert working in an iterative, self-correcting loop.

Your task is to REVISE the previous SQLite query so that it better answers the question,
using ONLY the tables and columns listed in the schema. Use the exact table and column names.

Schema:
{schema_string}

Allowed columns per table (MUST NOT introduce any other names):
{allowed_cols}

Example rows (GROUND TRUTH — copy filter literals from here when possible):
{samples_string if samples_string else "(no samples available)"}

Question:
\"\"\"{_truncate_for_llm(question, 2000)}\"\"\"

{trace_section}SQL to revise (must match the latest attempt in the trace above when a trace is present):
```sql
{previous_sql}
```

Feedback:
{feedback_block}

{_refine_strategy_block(refine_strategy, excluded_tables=excluded_tables or set(), alternate_table_hint=alternate_table_hint)}

Important constraints:
- If an execution trace is listed above, learn from it: do not repeat the same failing join, table choice, or literals unless you have a concrete fix.
- Keep the query focused on the question.
- Do NOT hallucinate new tables or columns.
- When filtering by categorical columns, prefer values shown in the example rows; use LIKE for fuzzy match when 0 rows.
- For numeric-like text metrics (Revenue/Admissions/Domestic_gross), cast cleaned values to REAL before MIN/MAX/SUM/AVG/ORDER.
- Avoid speculative fuzzy joins (`ON ... LIKE ...`); prefer a single table unless there is a clear relational key.
- Prefer robust joins and reasonable limits.
- After empty results, prefer discovery (MIN/MAX of time columns) or LIKE-based filters before giving up.

Return ONLY the revised SQL statement starting with SELECT or WITH ... AS (. No English prose.
"""

    resp = _chat_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=384,
    )
    raw = (resp.choices[0].message.content or "").strip()
    sql = _extract_sql_candidate(raw)
    if not _is_valid_sql_start(sql):
        return _fallback_sql_from_schema(schema_string)
    return sql


def tool_generate_sql(
    *,
    client,
    model: str,
    question: str,
    schema_string: str,
    samples_string: str = "",
) -> str:
    """Explicit tool wrapper for hybrid routing; delegates to generate_sql."""
    return generate_sql(client, model, question, schema_string, samples_string=samples_string)


def tool_refine_sql(
    *,
    client,
    model: str,
    question: str,
    schema_string: str,
    previous_sql: str,
    error_message: Optional[str],
    empty_result: bool,
    samples_string: str = "",
    refine_strategy: str = "default",
    excluded_tables: Optional[Set[str]] = None,
    alternate_table_hint: Optional[str] = None,
    attempts_log: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Explicit tool wrapper for hybrid routing; delegates to refine_sql_with_error."""
    return refine_sql_with_error(
        client,
        model,
        question,
        schema_string,
        previous_sql,
        error_message,
        empty_result=empty_result,
        samples_string=samples_string,
        refine_strategy=refine_strategy,
        excluded_tables=excluded_tables,
        alternate_table_hint=alternate_table_hint,
        attempts_log=attempts_log,
    )


def _parse_json_object_from_llm(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            o = json.loads(m.group(0))
            return o if isinstance(o, dict) else None
        except Exception:
            pass
    return None


def _llm_route_sql_proposal_action(
    client,
    model: str,
    *,
    question: str,
    last_error: Optional[str],
    empty_result: bool,
    refine_strategy: str,
    attempt_idx: int,
) -> str:
    """
    Lightweight JSON router: generate_sql (fresh SELECT) vs refine_sql (fix last).
    Execution stays in LangGraph; this only chooses which SQL tool to call in propose_sql.
    """
    prompt = f"""You choose the next SQL authoring step for a bounded retry loop.
Return ONLY valid JSON (no markdown, no code fences):
{{"action":"generate_sql"|"refine_sql","rationale":"one short phrase"}}

Sub-question:
\"\"\"{_truncate_for_llm(question, 1500)}\"\"\"

Context:
- After-execution attempt index (0 = first execution done; higher = more retries): {attempt_idx}
- System policy hint for this round: {refine_strategy}
- Last execution error (if any): {_truncate_for_llm(last_error or "(none)", 800)}
- Last run returned 0 rows (empty result): {empty_result}

Rules:
- choose "refine_sql" when the last query is close and needs fixes (syntax/schema, predicates, JOIN simplification, discovery tweaks).
- choose "generate_sql" only when abandoning the prior approach for a clean new SELECT (e.g. wrong table family, irreparable join).
- prefer "refine_sql" if the error looks like schema/syntax/no such column/table.

JSON:"""

    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=160,
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = _parse_json_object_from_llm(raw) or {}
        action = str(obj.get("action", "")).strip().lower()
        if action in ("generate_sql", "refine_sql"):
            return action
    except TokensPerDayExhaustedError:
        raise
    except Exception:
        pass
    return "refine_sql"


def summarize_subquestion_result(
    client,
    model: str,
    dpr_text: str,
    sub_question: str,
    table_uids: List[str],
    preview_rows: List[Dict[str, Any]],
    total_row_count: int = 0,
) -> str:
    """
    Evidence-constrained mini-summary: only facts entailed by the SQL result rows (preview).
    """
    try:
        preview_json = json.dumps(preview_rows, indent=2, ensure_ascii=False)
    except TypeError:
        preview_json = str(preview_rows)

    n = total_row_count if total_row_count > 0 else len(preview_rows)
    preview_note = (
        f"The query returned {n} row(s) total; the JSON below shows at most the first few rows."
        if n > len(preview_rows)
        else f"The query returned {n} row(s); all are shown below."
    )

    prompt = f"""You are an evidence checker for tabular query results (natural-language inference style).

Original DPR (context only — do NOT import facts from it that are absent from the JSON):
\"\"\"{_truncate_for_llm(dpr_text)}\"\"\"

Sub-question:
\"\"\"{_truncate_for_llm(sub_question, 2000)}\"\"\"

Relevant tables (IDs only): {table_uids}

{preview_note}

Evidence — result rows (ONLY source of truth for your answer):
```json
{preview_json}
```

Rules (strict):
- You may ONLY state facts that are **directly entailed** by the cell values in the JSON above.
- If the sub-question asks for something not present in these rows (e.g. a year or metric that does not appear), say clearly that **the result set does not show** that information, and only describe what **does** appear.
- Do NOT infer industry trends, causality, or background knowledge. Do NOT mention SQL, queries, or table names.
- Write 1–2 short sentences. If nothing in the rows answers the sub-question, one sentence stating that is enough.

Output ONLY those sentences. No planning, no reasoning, no tags, no "Okay" preambles.
"""

    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _clean_llm_prose_response(raw)
    except TokensPerDayExhaustedError:
        raise
    except Exception:
        return ""


def generate_final_summary(client, model: str, dpr_text: str, mini_summaries: List[str]) -> str:
    """
    Synthesize all grounded mini-summaries for a DPR into a single final summary.
    """
    if not mini_summaries:
        return ""

    bullet_points = "\n".join(f"- {s}" for s in mini_summaries if s.strip())

    prompt = f"""You synthesize a DPR response from **evidence bullets only**. Treat the bullets as the only facts you may use.

Original DPR (for framing only — you must NOT add facts that are not in the bullets):
\"\"\"{_truncate_for_llm(dpr_text)}\"\"\"

Evidence bullets (each sentence was constrained to SQL result rows):
{bullet_points}

Write one concise paragraph that:
- Combines and paraphrases **only** what is stated in the bullets.
- Does **not** introduce dates, names, numbers, or claims that do not appear in the bullets.
- If the bullets leave the DPR partly unanswered, you may briefly note that limitations are visible in the evidence (without inventing data).
- Does not mention SQL, queries, or tables.

Output ONLY that paragraph. No reasoning blocks, tags, or meta-commentary.
"""

    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=320,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _clean_llm_prose_response(raw)
    except TokensPerDayExhaustedError:
        raise
    except Exception:
        return ""


def compute_stage3_execution_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate execution outcomes (no LLM verdict — Stage 4 owns evaluation)."""
    total = len(rows)
    exec_ok = sum(1 for r in rows if r.get("execution_status") is True)
    with_rows = 0
    for r in rows:
        if r.get("execution_status") is not True:
            continue
        res = r.get("result")
        if isinstance(res, dict) and int(res.get("row_count") or 0) > 0:
            with_rows += 1
    return {
        "total_dprs": total,
        "execution_success_dprs": exec_ok,
        "execution_failed_dprs": total - exec_ok,
        "success_with_positive_rowcount": with_rows,
    }


def _is_schema_error(error_message: Optional[str]) -> bool:
    if not error_message:
        return False
    msg = error_message.lower()
    if "schema validation:" in msg:
        return True
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


def _is_execution_timeout_error(error_message: Optional[str]) -> bool:
    if not error_message:
        return False
    msg = error_message.strip().lower()
    return "query execution timeout" in msg or "sqlite query execution timeout" in msg or "progress handler" in msg


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


def _on_clause_scan_end(tail: str) -> int:
    """First top-level clause after ON: next JOIN / WHERE / GROUP / ORDER / HAVING / LIMIT."""
    m = re.search(
        r"\b(?:(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b",
        tail,
        flags=re.IGNORECASE,
    )
    return m.start() if m else len(tail)


def _is_speculative_join(sql: Optional[str]) -> bool:
    """
    True only if LIKE appears inside an ON clause (not in WHERE after a clean equality ON).
    Previous bug: r'\\bon\\b[^;]*\\blike\\b' matched LIKE in WHERE and blocked valid joins.
    """
    if not sql:
        return False
    s = re.sub(r"\s+", " ", str(sql).strip())
    if not re.search(r"\bjoin\b", s, flags=re.IGNORECASE):
        return False
    idx = 0
    while True:
        m = re.search(r"\bON\b", s[idx:], flags=re.IGNORECASE)
        if not m:
            break
        pos = idx + m.end()
        tail = s[pos:]
        end = _on_clause_scan_end(tail)
        on_fragment = tail[:end]
        if re.search(r"\bLIKE\b", on_fragment, flags=re.IGNORECASE):
            return True
        idx = pos
    return False


def _uses_unsafe_text_metric_aggregate(sql: Optional[str]) -> bool:
    """
    Detect likely lexical (TEXT) aggregation/sorting on numeric-like metrics.
    Example risk: MIN(Domestic_gross) where values are '$116,000,000' strings.
    """
    if not sql:
        return False
    s = re.sub(r"\s+", " ", str(sql).strip())
    metric_cols = r"(revenue|admissions|domestic_gross|gross)"
    has_metric = re.search(rf"\b{metric_cols}\b", s, flags=re.IGNORECASE)
    if not has_metric:
        return False
    # Risky patterns: MIN/MAX/SUM/AVG/ORDER BY directly on metric text columns.
    risky_agg = re.search(
        rf"\b(min|max|sum|avg)\s*\(\s*([A-Za-z_][\w]*\.)?{metric_cols}\s*\)",
        s,
        flags=re.IGNORECASE,
    )
    risky_sort = re.search(
        rf"\border\s+by\s+([A-Za-z_][\w]*\.)?{metric_cols}\b",
        s,
        flags=re.IGNORECASE,
    )
    # If CAST is present, assume the model attempted numeric coercion.
    has_cast = re.search(r"\bcast\s*\(", s, flags=re.IGNORECASE)
    return bool((risky_agg or risky_sort) and not has_cast)


def _extract_tables_from_sql(sql: str) -> Set[str]:
    """Table identifiers after FROM / JOIN (best-effort; does not resolve aliases)."""
    found: Set[str] = set()
    for m in re.finditer(r"(?is)\b(?:from|join)\s+([A-Za-z_][\w]*)", sql or ""):
        found.add(m.group(1))
    return found


def _sql_has_explicit_join(sql: str) -> bool:
    return bool(re.search(r"(?is)\bjoin\b", sql or ""))


def _validate_sql_against_schema(sql: str, schema_string: str) -> Optional[str]:
    """
    Pre-flight validation: unknown FROM/JOIN tables and unknown qualified Table.column.
    Catches many hallucinated columns before SQLite executes (saves retries/latency).
    Unqualified columns are not checked (alias-heavy queries).
    """
    tc = _parse_schema_tables_columns(schema_string)
    if not tc or not (sql or "").strip():
        return None
    s = sql.strip()
    known_tables = set(tc.keys())
    for t in _extract_tables_from_sql(s):
        if t not in known_tables:
            return f"Schema validation: unknown table {t!r} (not in cluster schema)."
    for m in re.finditer(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b", s):
        tname, cname = m.group(1), m.group(2)
        if tname not in known_tables:
            continue
        cols = tc.get(tname, [])
        colset_lower = {c.lower() for c in cols}
        if cname.lower() not in colset_lower:
            return f"Schema validation: unknown column {tname}.{cname} for this table."
    return None


def _pick_alternate_table(cluster_ids: List[str], used: Set[str]) -> Optional[str]:
    for tid in cluster_ids:
        if tid not in used:
            return tid
    return None


def _refine_strategy_block(
    strategy: str,
    *,
    excluded_tables: Optional[Set[str]] = None,
    alternate_table_hint: Optional[str] = None,
) -> str:
    """Typed recovery instructions for refine_sql_with_error (policy graph)."""
    excluded_tables = excluded_tables or set()
    if strategy == "simplify_drop_join":
        return (
            "**Recovery mode — SIMPLIFY (attempt policy):**\n"
            "- If the query uses JOIN, **remove JOINs** and answer using **one** table only unless a true shared key exists.\n"
            "- Broaden or drop conjunctive filters; prefer fewer predicates copied from sample rows.\n"
            "- If the question needs two domains with no join path, return the best **single-table** evidence query.\n"
        )
    if strategy == "discovery":
        return (
            "**Recovery mode — DISCOVERY (attempt policy):**\n"
            "- Emit a small exploratory query: `SELECT DISTINCT col FROM T LIMIT 50`, or "
            "`SELECT MIN(col), MAX(col) FROM T` for numeric/time-like columns that exist.\n"
            "- Use only columns present in the schema; ground literals using sample rows.\n"
        )
    if strategy == "alternate_table":
        excl = ", ".join(sorted(excluded_tables)) if excluded_tables else "(none)"
        alt = alternate_table_hint or "(pick another cluster table)"
        return (
            "**Recovery mode — ALTERNATE TABLE (attempt policy):**\n"
            f"- Previous query used table(s): {excl}. **Use a different cluster table** as the primary source; "
            f"prefer starting from **{alt}** if listed.\n"
            "- Still no speculative fuzzy JOINs; single-table preferred.\n"
        )
    return ""


def _format_refinement_execution_trace(
    attempts_log: Optional[List[Dict[str, Any]]],
    *,
    max_entries: int = 8,
    max_sql_chars: int = 1200,
) -> str:
    """
    Full per-sub-question execution history for refine_sql_with_error, including the
    latest attempt (marked as the one to revise). Fixes the first-retry gap where only
    one execution existed but no structured trace was shown.
    """
    if not attempts_log:
        return ""
    entries = attempts_log[-max_entries:]
    lines: List[str] = []
    base = len(attempts_log) - len(entries)
    for i, att in enumerate(entries):
        n = base + i + 1
        is_last = i == len(entries) - 1
        marker = " **← REVISE THIS CANDIDATE**" if is_last else ""
        sql = (att.get("sql") or "").strip()
        if len(sql) > max_sql_chars:
            sql = sql[: max_sql_chars - 3] + "..."
        ok = att.get("execution_status")
        err = att.get("error")
        rc = int(att.get("row_count", 0) or 0)
        phase = att.get("retry_phase") or "?"
        status = "executed OK" if ok else "failed"
        detail_parts: List[str] = [f"rows={rc}"]
        if err:
            detail_parts.append(f"error={_truncate_for_llm(str(err), 500)}")
        hint_parts: List[str] = []
        if att.get("explicit_join"):
            hint_parts.append("uses JOIN")
        tabs = att.get("tables")
        if isinstance(tabs, list) and tabs:
            hint_parts.append("tables=" + ",".join(str(t) for t in tabs[:12]))
        if hint_parts:
            detail_parts.append("; ".join(hint_parts))
        detail = "; ".join(detail_parts)
        lines.append(
            f"Attempt {n} (phase={phase}): {status} ({detail}){marker}\n"
            f"SQL:\n{sql}"
        )
    block = "\n\n".join(lines)
    return (
        "## Execution trace for this sub-question (do not repeat failed patterns)\n"
        f"{block}\n"
        "---\n"
    )


def execute_and_validate(
    cursor, sql: str, require_non_empty: bool
) -> Tuple[bool, List[Dict[str, Any]], Optional[str], int]:
    conn = cursor.connection
    start_t = time.monotonic()

    def progress_handler() -> None:
        if time.monotonic() - start_t > SQL_EXECUTION_WALLTIME_SEC:
            raise sqlite3.OperationalError("Query execution timeout (possible cartesian join).")

    try:
        conn.set_progress_handler(progress_handler, SQL_PROGRESS_HANDLER_OPCODE_INTERVAL)
        cursor.execute(sql)

        # Defensive: avoid `fetchall()` on huge result sets.
        rows = cursor.fetchmany(SQL_MAX_FETCH_ROWS)
        cols = [d[0] for d in cursor.description] if cursor.description else []

        # Try to compute the true result cardinality in SQLite (not Python preview length).
        # Fall back to preview length if counting fails/timeouts.
        row_count = len(rows)
        try:
            sql_for_count = (sql or "").strip().rstrip(";")
            count_sql = f"SELECT COUNT(*) FROM ({sql_for_count}) AS __stage3_count_subquery"
            cursor.execute(count_sql)
            one = cursor.fetchone()
            if one and len(one) > 0 and one[0] is not None:
                row_count = int(one[0])
        except Exception:
            # Keep pipeline robust: counting failures should not discard valid preview rows.
            row_count = len(rows)

        if require_non_empty and not rows:
            return False, [], "Query returned no rows (empty result)", 0

        preview = [dict(zip(cols, r)) for r in rows[:5]]
        return True, preview, None, row_count
    except Exception as e:
        return False, [], str(e), 0
    finally:
        try:
            conn.set_progress_handler(None, 0)
        except Exception:
            pass


class SQLLoopState(TypedDict, total=False):
    sub_question: str
    schema_string: str
    samples_string: str
    attempt_count: int
    max_attempts: int
    last_sql: Optional[str]
    last_error: Optional[str]
    last_execution_ok: bool
    last_row_count: int
    last_preview: List[Dict[str, Any]]
    attempts_log: List[Dict[str, Any]]
    # Explicit "reasoning/strategy" stage for the next SQL proposal.
    empty_result: bool
    refine_strategy: str
    excluded_tables: Optional[List[str]]
    alternate_table_hint: Optional[str]
    retry_phase: str
    done: bool
    success: bool
    best_sql: Optional[str]
    best_preview: List[Dict[str, Any]]
    best_row_count: int


def _should_retry_sql_attempt(sql: Optional[str], ok: bool, row_count: int, err: Optional[str]) -> bool:
    return bool(
        _is_schema_error(err)
        or _is_execution_timeout_error(err)
        or _is_cartesian_sql(sql)
        or _is_speculative_join(sql)
        or _uses_unsafe_text_metric_aggregate(sql)
        or _is_empty_result_error(err)
        or (ok and row_count == 0)
        or (ok and row_count > 0 and _sql_is_trivial_probe(sql))
    )


def _execute_sql_candidate(
    *,
    cursor,
    schema_string: str,
    sql: Optional[str],
    require_non_empty: bool,
) -> Tuple[bool, List[Dict[str, Any]], Optional[str], int]:
    if not sql:
        return False, [], "Empty SQL candidate.", 0
    if _is_cartesian_sql(sql):
        return False, [], "Detected disallowed cartesian join pattern (e.g., ON 1=1 or CROSS JOIN).", 0
    if _is_speculative_join(sql):
        return (
            False,
            [],
            "Detected speculative fuzzy JOIN predicate (ON ... LIKE ...). Use a clear relational key or prefer a single-table query.",
            0,
        )
    if _uses_unsafe_text_metric_aggregate(sql):
        return (
            False,
            [],
            "Detected text-based aggregation/sort on numeric-like metric columns (Revenue/Admissions/Gross) without numeric CAST.",
            0,
        )
    schema_val_err = _validate_sql_against_schema(sql, schema_string)
    if schema_val_err:
        return False, [], schema_val_err, 0
    return execute_and_validate(cursor, sql, require_non_empty=require_non_empty)


def tool_execute_sql(
    *,
    cursor,
    schema_string: str,
    sql: Optional[str],
    require_non_empty: bool,
) -> Tuple[bool, List[Dict[str, Any]], Optional[str], int]:
    """Agent tool wrapper: validate + execute SQL against the in-memory SQLite cluster DB."""
    return _execute_sql_candidate(
        cursor=cursor,
        schema_string=schema_string,
        sql=sql,
        require_non_empty=require_non_empty,
    )


def _run_subquestion_sql_loop_langgraph(
    *,
    client,
    model: str,
    cursor,
    sub_q: str,
    schema_string: str,
    samples_string: str,
    table_uids: List[str],
    require_non_empty: bool,
) -> Dict[str, Any]:
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        raise RuntimeError("LangGraph not installed. Install with `pip install langgraph` for pipelinenew.py SQL loop.")

    def reason_node(state: SQLLoopState) -> SQLLoopState:
        """
        Explicit "reason about last failure" stage.

        This does not require another LLM call: it converts the latest tool outputs
        (error text, row_count, and SQL patterns) into a structured refinement plan
        for the *next* SQL proposal.
        """
        attempt_idx = int(state.get("attempt_count", 0))
        sql_prev = state.get("last_sql")
        err_prev = state.get("last_error")
        row_prev = int(state.get("last_row_count", 0))

        # Defaults for attempt 0 (no previous execution).
        if attempt_idx == 0 or not sql_prev:
            return {
                "retry_phase": "generate",
                "refine_strategy": "generate",
                "empty_result": False,
                "excluded_tables": None,
                "alternate_table_hint": None,
            }

        empty_result = (row_prev == 0) and (err_prev is None or _is_empty_result_error(err_prev))

        if attempt_idx == 1:
            refine_strategy = "simplify_drop_join"
            excluded_tables: Optional[List[str]] = None
            alt_hint: Optional[str] = None
        else:
            prev_attempts = state.get("attempts_log", []) or []
            prev_attempt_had_empty = (
                len(prev_attempts) > 0
                and prev_attempts[-1].get("execution_status") is True
                and int(prev_attempts[-1].get("row_count", 0) or 0) == 0
            )
            hard_pivot_signals = (
                _is_speculative_join(sql_prev)
                or _is_cartesian_sql(sql_prev)
                or _is_schema_error(err_prev)
                or _is_execution_timeout_error(err_prev)
                or prev_attempt_had_empty
            )
            if hard_pivot_signals:
                refine_strategy = "alternate_table"
                excluded_set = _extract_tables_from_sql(sql_prev or "")
                excluded_tables = sorted(excluded_set) if excluded_set else None
                alt_hint = _pick_alternate_table(table_uids, excluded_set) if excluded_set else None
            else:
                refine_strategy = "discovery"
                excluded_tables = None
                alt_hint = None

        retry_phase = (
            "simplify_drop_join"
            if refine_strategy == "simplify_drop_join"
            else "alternate_table"
            if refine_strategy == "alternate_table"
            else "discovery"
        )

        return {
            "retry_phase": retry_phase,
            "refine_strategy": refine_strategy,
            "empty_result": empty_result,
            "excluded_tables": excluded_tables,
            "alternate_table_hint": alt_hint,
        }

    def propose_sql_node(state: SQLLoopState) -> SQLLoopState:
        attempt_idx = int(state.get("attempt_count", 0))
        sql_prev = state.get("last_sql")
        err_prev = state.get("last_error")

        if attempt_idx == 0 or not sql_prev:
            sql_new = generate_sql(client, model, sub_q, schema_string, samples_string=samples_string)
            return {"last_sql": sql_new}

        empty_result = bool(state.get("empty_result", False))
        refine_strategy = str(state.get("refine_strategy", "discovery") or "discovery")
        excluded_tables_list = state.get("excluded_tables")
        excluded_tables_set: Optional[Set[str]] = set(excluded_tables_list) if excluded_tables_list else None
        alternate_table_hint = state.get("alternate_table_hint")
        log = state.get("attempts_log") or []

        action = "refine_sql"
        if STAGE3_SQL_ACTION_ROUTER:
            action = _llm_route_sql_proposal_action(
                client,
                model,
                question=sub_q,
                last_error=err_prev,
                empty_result=empty_result,
                refine_strategy=refine_strategy,
                attempt_idx=attempt_idx,
            )
        if action not in ("generate_sql", "refine_sql"):
            action = "refine_sql"
        if _is_schema_error(err_prev):
            action = "refine_sql"

        if action == "generate_sql":
            sql_new = tool_generate_sql(
                client=client,
                model=model,
                question=sub_q,
                schema_string=schema_string,
                samples_string=samples_string,
            )
        else:
            sql_new = tool_refine_sql(
                client=client,
                model=model,
                question=sub_q,
                schema_string=schema_string,
                previous_sql=sql_prev or "",
                error_message=err_prev,
                empty_result=empty_result,
                samples_string=samples_string,
                refine_strategy=refine_strategy,
                excluded_tables=excluded_tables_set,
                alternate_table_hint=alternate_table_hint,
                attempts_log=log,
            )
        return {"last_sql": sql_new}

    def execute_sql_node(state: SQLLoopState) -> SQLLoopState:
        sql_now = state.get("last_sql")
        ok, preview, err, row_count = tool_execute_sql(
            cursor=cursor,
            schema_string=schema_string,
            sql=sql_now,
            require_non_empty=require_non_empty,
        )
        attempts_log = list(state.get("attempts_log", []))
        sn = sql_now or ""
        attempts_log.append(
            {
                "sql": sql_now,
                "execution_status": ok,
                "error": err,
                "row_count": row_count,
                "retry_phase": state.get("retry_phase", "generate"),
                "explicit_join": _sql_has_explicit_join(sn),
                "tables": sorted(_extract_tables_from_sql(sn)),
            }
        )
        out: SQLLoopState = {
            "attempt_count": int(state.get("attempt_count", 0)) + 1,
            "attempts_log": attempts_log,
            "last_error": err,
            "last_execution_ok": ok,
            "last_row_count": row_count,
            "last_preview": preview,
            "success": False,
            "done": False,
        }
        if ok and row_count > 0 and not _sql_is_trivial_probe(sql_now):
            out.update(
                {
                    "success": True,
                    "done": True,
                    "best_sql": sql_now,
                    "best_preview": preview,
                    "best_row_count": row_count,
                }
            )
        return out

    def decide_next(state: SQLLoopState) -> str:
        if bool(state.get("done")) or bool(state.get("success")):
            return "end"
        attempts_used = int(state.get("attempt_count", 0))
        if attempts_used >= int(state.get("max_attempts", MAX_SQL_ATTEMPTS)):
            return "end"
        if not _should_retry_sql_attempt(
            state.get("last_sql"),
            bool(state.get("last_execution_ok")),
            int(state.get("last_row_count", 0)),
            state.get("last_error"),
        ):
            return "end"

        return "reason"

    graph = StateGraph(SQLLoopState)
    graph.add_node("reason", reason_node)
    graph.add_node("propose_sql", propose_sql_node)
    graph.add_node("execute_sql", execute_sql_node)

    graph.set_entry_point("reason")
    graph.add_edge("reason", "propose_sql")
    graph.add_edge("propose_sql", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        decide_next,
        {"reason": "reason", "end": END},
    )
    app = graph.compile()

    try:
        final_state = app.invoke(
            {
                "sub_question": sub_q,
                "schema_string": schema_string,
                "samples_string": samples_string,
                "attempt_count": 0,
                "max_attempts": MAX_SQL_ATTEMPTS,
                "last_sql": None,
                "last_error": None,
                "last_execution_ok": False,
                "last_row_count": 0,
                "last_preview": [],
                "attempts_log": [],
                "success": False,
                "done": False,
                "best_sql": None,
                "best_preview": [],
                "best_row_count": 0,
                "retry_phase": "generate",
                "refine_strategy": "generate",
                "empty_result": False,
                "excluded_tables": None,
                "alternate_table_hint": None,
            }
        )
    except TokensPerDayExhaustedError:
        raise

    return {
        "attempts": list(final_state.get("attempts_log", [])),
        "best_ok": bool(final_state.get("success", False)),
        "best_sql": final_state.get("best_sql"),
        "best_preview": list(final_state.get("best_preview", [])),
        "best_row_count": int(final_state.get("best_row_count", 0)),
        "last_error": final_state.get("last_error"),
    }


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


def _build_cluster_sqlite_from_table_metadata(
    table_metas: Dict[str, Any]
) -> Tuple[sqlite3.Connection, sqlite3.Cursor, str, Dict[str, Dict[str, str]]]:
    """
    Build an in-memory SQLite DB for the DPR cluster tables only (``table_metas``).

    For each table: CREATE from declared columns/types, then INSERT **all** rows from
    the JSON ``rows`` field (row-wise dicts or HybridQA-style flattened cells). Empty
    ``rows`` ⇒ table exists but has no inserted data (metadata issue, not schema-only by design).
    """
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

        # Populate with the full row list from JSON (cluster tables only — not all 100 tables).
        rows_meta = meta.get("rows") or []
        logical_rows: List[Dict[str, Any]] = []

        if rows_meta:
            sample = rows_meta[0]
            # Case 1: row-wise dicts
            if isinstance(sample, dict) and any(k in sample for k in cols):
                for r in rows_meta:
                    if not isinstance(r, dict):
                        continue
                    row_obj: Dict[str, Any] = {}
                    for c in cols:
                        row_obj[str(c)] = r.get(str(c))
                    logical_rows.append(row_obj)
            # Case 2: flattened cell list (HybridQA-style)
            elif isinstance(sample, dict) and "value" in sample and cols:
                step = len(cols)
                for i in range(0, len(rows_meta), step):
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
            # Performance: avoid per-row execute() overhead for large tables.
            # Chunked executemany keeps memory bounded while still speeding up inserts.
            batch_values: List[List[Any]] = []
            batch_size = 500
            for r in logical_rows:
                values = [r.get(str(c)) for c in cols]
                batch_values.append(values)
                if len(batch_values) >= batch_size:
                    cursor.executemany(insert_sql, batch_values)
                    batch_values.clear()
            if batch_values:
                cursor.executemany(insert_sql, batch_values)

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
    """
    Resolve table metadata without --tables-meta.

    Prefers ``data/stage1_outputs/tables_clean`` (one JSON per table, full rows)
    at the repo root, then other common layouts.
    """
    stage2 = Path(stage2_output_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    d = stage2.parent
    parent = d.parent

    candidates: List[Path] = [
        repo_root / "data" / "stage1_outputs" / "tables_clean",
        repo_root / "data" / "stage1" / "tables_clean",
        parent / "stage1_outputs" / "tables_clean",
        d / "tables_clean",
        parent / "input" / "tables.json",
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except OSError:
            continue
    return None


def _load_stage2_payload(input_path: str) -> List[Dict[str, Any]]:
    """
    Load Stage-2 payload from either:
    - JSON list file (dprs-*.json), or
    - JSONL file (e.g., dprs-qwen3-32b.jsonl), one DPR object per line.
    """
    p = os.path.abspath(input_path)
    if p.lower().endswith(".jsonl"):
        out: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    out.append(item)
        return out

    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Stage-3 expects a JSON list or JSONL objects as input.")


def _write_stage3_outputs(out: List[Dict[str, Any]], output_path: str, *, partial: bool = False) -> None:
    """Write Stage-3 JSON plus execution summary sidecar (used for normal completion and TPD fail-fast)."""
    with open(output_path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, indent=2)
    exec_stats = compute_stage3_execution_stats(out)
    output_p = Path(output_path)
    summary_path = str(output_p.with_name(f"{output_p.stem}_execution_summary.json"))
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(exec_stats, sf, indent=2)
    prefix = "[stage3] Partial save (checkpoint / quota). " if partial else ""
    print(
        f"{prefix}Execution stats: "
        f"ok={exec_stats['execution_success_dprs']}/{exec_stats['total_dprs']} "
        f"| with_rows>0={exec_stats['success_with_positive_rowcount']}",
        flush=True,
    )
    print(f"Execution summary JSON: {summary_path}", flush=True)


def run_stage3_pipeline(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    offset: int = 0,
    tables_meta_path: Optional[str] = None,
    require_non_empty: bool = False,
) -> List[Dict[str, Any]]:
    """Runs stage-3 on stage-2 artifacts."""

    payload = _load_stage2_payload(input_path)

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError("Stage-3 expects a Stage-2 DPR list (dprs-*.json or dprs-*.jsonl).")

    client, model = get_llm_client()
    first = payload[0]

    # Stage 2 = DPR list (each row: DPR + ground_truth.table_uids); table payloads come from tables_clean.
    if "DPR" in first and "ground_truth" in first:
        start_idx = max(0, int(offset or 0))
        if start_idx >= len(payload):
            dprs_list = []
        else:
            dprs_slice = payload[start_idx:]
            dprs_list = dprs_slice[: int(limit)] if limit is not None else dprs_slice

        tables_meta_root = tables_meta_path or _infer_tables_json(input_path)
        if not tables_meta_root:
            raise RuntimeError(
                "Could not infer table metadata. Pass --tables-meta <path/to/tables.json or tables_clean dir>."
            )

        tables_meta_all = _load_tables_meta(tables_meta_root)
        print(
            f"Using LLM: {model} | tables_meta: {tables_meta_root} "
            f"| loaded {len(tables_meta_all)} table JSON(s) | DPRs to run: {len(dprs_list)}"
        )
        if not LANGGRAPH_AVAILABLE or StateGraph is None:
            raise RuntimeError(
                "pipelinenew.py requires LangGraph for the SQL agentic loop. "
                "Install dependency: `pip install langgraph`."
            )
        print(
            f"[stage3] SQL loop engine=langgraph | max_attempts={MAX_SQL_ATTEMPTS} "
            f"| min_subq_successes={STAGE3_MIN_SUBQ_SUCCESSES}",
            flush=True,
        )

        out: List[Dict[str, Any]] = []
        n_dprs = len(dprs_list)
        for i, d in enumerate(dprs_list):
            dpr_id = d.get("dpr_id")
            dpr_text = d.get("DPR")
            gt = d.get("ground_truth") or {}
            table_uids = (gt.get("table_uids") or []) if isinstance(gt, dict) else []

            dpr_head = (dpr_text or "").replace("\n", " ").strip()
            if len(dpr_head) > 72:
                dpr_head = dpr_head[:69] + "..."
            print(
                f"[stage3] --- DPR {i + 1}/{n_dprs} starting --- dpr_id={dpr_id!r} | "
                f"tables={len(table_uids)} | preview: {dpr_head!r}",
                flush=True,
            )

            per_tables_meta = {tid: tables_meta_all.get(tid) for tid in table_uids if tables_meta_all.get(tid)}
            conn, cursor, schema_string, name_mapping = _build_cluster_sqlite_from_table_metadata(per_tables_meta)

            # Defaults so a late LLM failure (e.g. 429) does not wipe decomposition / partial results.
            sub_questions: List[str] = []
            subquery_results: List[Dict[str, Any]] = []
            all_mini_summaries: List[str] = []
            final_summary = ""
            ok = False
            preview: List[Dict[str, Any]] = []
            row_count = 0
            sql = None
            err: Optional[str] = None

            try:
                # Dynamic decomposition depth: fewer sub-questions for short DPRs to save tokens.
                dpr_len = len(dpr_text or "")
                if dpr_len < 200:
                    min_q, max_q = 2, 2
                else:
                    min_q, max_q = 2, 3

                sub_questions = generate_subquestions(
                    client,
                    model,
                    dpr_text,
                    table_uids=table_uids,
                    min_questions=min_q,
                    max_questions=max_q,
                    schema_string=schema_string,
                )
                # Cheap rule-based scoring to reduce low-quality / generic sub-questions.
                sub_questions = _quality_select_subquestions(sub_questions, schema_string=schema_string, max_questions=max_q)
                if len(sub_questions) < min_q:
                    # Safety fallback if the quality selector prunes too aggressively.
                    sub_questions = _fallback_atomic_questions(dpr_text, table_uids, max_q)[:max_q]

                subquery_results = []
                all_mini_summaries = []

                any_success = False
                first_success_preview: List[Dict[str, Any]] = []
                first_success_row_count = 0
                representative_sql: Optional[str] = None
                last_error: Optional[str] = None

                # Sample rows are identical for every sub-question (same cluster DB); fetch once per DPR.
                samples_string = _fetch_table_samples(cursor, table_uids, limit=LLM_SAMPLE_ROWS_PER_TABLE)

                success_subq_count = 0

                for sub_idx, sub_q in enumerate(sub_questions, start=1):
                    feasibility_warning = ""
                    if _question_references_unknown_risk_columns(sub_q, schema_string):
                        # Soft warning only: do NOT skip. Let SQL generation/refinement try to pivot.
                        feasibility_warning = (
                            "Feasibility warning: question references risky concepts (e.g. year/date/rank) "
                            "that may not exist in this cluster schema."
                        )

                    print(
                        f"[stage3]   subq {sub_idx}/{len(sub_questions)} start | engine=langgraph",
                        flush=True,
                    )
                    mini_summary = ""
                    sub_loop = _run_subquestion_sql_loop_langgraph(
                        client=client,
                        model=model,
                        cursor=cursor,
                        sub_q=sub_q,
                        schema_string=schema_string,
                        samples_string=samples_string,
                        table_uids=table_uids,
                        require_non_empty=require_non_empty,
                    )

                    attempts = list(sub_loop.get("attempts", []))
                    best_ok = bool(sub_loop.get("best_ok"))
                    best_sql = sub_loop.get("best_sql")
                    best_preview = list(sub_loop.get("best_preview", []))
                    best_row_count = int(sub_loop.get("best_row_count", 0))
                    err = sub_loop.get("last_error")

                    if best_ok:
                        any_success = True
                        success_subq_count += 1
                        if representative_sql is None:
                            representative_sql = best_sql
                            first_success_preview = best_preview
                            first_success_row_count = best_row_count

                        mini_summary = summarize_subquestion_result(
                            client,
                            model,
                            dpr_text,
                            sub_q,
                            table_uids,
                            best_preview,
                            total_row_count=best_row_count,
                        )
                        if mini_summary:
                            all_mini_summaries.append(mini_summary)

                    last_error = err

                    subquery_results.append(
                        {
                            "sub_question": sub_q,
                            "feasibility_warning": feasibility_warning,
                            "attempts": attempts,
                            "final_sql": best_sql,
                            "final_execution_status": best_ok,
                            "final_row_count": best_row_count,
                            "mini_summary": mini_summary,
                        }
                    )
                    print(
                        f"[stage3]   subq {sub_idx}/{len(sub_questions)} done | attempts={len(attempts)} "
                        f"| success={'yes' if best_ok else 'no'} | rows={best_row_count}",
                        flush=True,
                    )

                required_successes = 1
                if len(sub_questions) >= 2:
                    required_successes = min(STAGE3_MIN_SUBQ_SUCCESSES, len(sub_questions))

                if success_subq_count >= required_successes:
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
                    err = (
                        f"Insufficient successful sub-questions: {success_subq_count}/{len(sub_questions)} "
                        f"(required={required_successes}). Last error: {last_error or 'n/a'}"
                    )
                print(
                    f"[stage3] DPR sub-question success count: {success_subq_count}/{len(sub_questions)} "
                    f"(required={required_successes})",
                    flush=True,
                )

                try:
                    final_summary = generate_final_summary(client, model, dpr_text, all_mini_summaries)
                except TokensPerDayExhaustedError:
                    raise
                except Exception:
                    final_summary = ""

            except TokensPerDayExhaustedError as tpd_exc:
                print(
                    f"[stage3] Tokens-per-day limit (fail-fast): {tpd_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[stage3] Saving {len(out)} completed DPR(s) to {output_path}.",
                    file=sys.stderr,
                    flush=True,
                )
                _write_stage3_outputs(out, output_path, partial=True)
                raise
            except Exception as e:
                ok = False
                preview = []
                row_count = 0
                sql = None
                err = str(e)
                # Preserve sub_questions / subquery_results / all_mini_summaries / final_summary if already filled
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
                    "llm_model_summaries": model,
                    "upstream_model": d.get("model"),
                }
            )
            _write_stage3_outputs(out, output_path, partial=True)

            parts = [
                f"[stage3] DPR {i + 1}/{n_dprs} completed",
                f"exec={'OK' if ok else 'FAIL'}",
                f"sql_rowcount={row_count}" if ok else None,
                f"sub_questions={len(sub_questions)}",
            ]
            if not ok and err:
                e1 = (err or "").replace("\n", " ").strip()
                if len(e1) > 140:
                    e1 = e1[:137] + "..."
                parts.append(f"error={e1!r}")
            print(" | ".join(p for p in parts if p), flush=True)

            # Pace requests between DPRs (skip after final DPR).
            if DPR_PROCESS_DELAY_SEC > 0 and (i + 1) < len(dprs_list):
                time.sleep(DPR_PROCESS_DELAY_SEC)

        _write_stage3_outputs(out, output_path, partial=False)
        return out

    raise RuntimeError(
        "Stage-3 requires Stage-2 rows with 'DPR' and 'ground_truth' (including table_uids). "
        f"First row keys: {sorted(first.keys())!r}. Use dprs-*.json or dprs-*.jsonl."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: SQL generation + grounding from Stage-2 output")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Stage-2 DPR list: dprs-*.json or dprs-*.jsonl (each row: DPR + ground_truth)",
    )
    parser.add_argument("--output", "-o", required=True, help="Stage-3 output JSON path")
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process first N DPRs only. Default: all rows in the input file. Ignored if --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all DPRs in the input file (overrides --limit).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start processing from this 0-based DPR index (use with --limit for batching).",
    )
    parser.add_argument("--tables-meta", default=None, help="Path to Stage-2 tables.json (if not inferable)")
    parser.add_argument(
        "--require-non-empty",
        action="store_true",
        help="If set, marks empty result sets as not grounded. Default is schema-only grounding (allow empty).",
    )
    args = parser.parse_args()

    try:
        out = run_stage3_pipeline(
            input_path=args.input,
            output_path=args.output,
            limit=None if args.all else args.limit,
            offset=0 if args.all else args.offset,
            tables_meta_path=args.tables_meta,
            require_non_empty=bool(args.require_non_empty),
        )
    except TokensPerDayExhaustedError:
        sys.exit(2)
    print(f"Done. Wrote {len(out)} record(s) to {args.output}")