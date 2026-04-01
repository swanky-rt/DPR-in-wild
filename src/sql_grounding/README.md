# Stage 3: SQL grounding and evidence synthesis

Stage 3 sits between Stage 2 (DPR generation) and Stage 4 (evaluation). For each DPR it loads the cluster tables into SQLite, turns the prose question into executable sub-questions, runs SQL until it gets useful answers or hits limits, then writes summaries that are only allowed to cite what actually came back from the database.

---

## What you run

There are two Python entry points in this folder:

| Script | Role |
|--------|------|
| **`pipelinenew.py`** | Current implementation. Per sub-question it uses a **LangGraph** state machine for the SQL retry loop (reason → propose → execute → decide). **Requires `langgraph`** (`pip install langgraph` or `pip install -r requirements.txt`). |
| **`pipeline.py`** | Earlier loop without LangGraph. Same overall Stage 3 contract (inputs/outputs), different internal control flow. Use if you cannot install LangGraph or need to compare behavior. |

Unless you have a reason to use `pipeline.py`, treat **`pipelinenew.py` as the default** for new runs.

---

## Inputs

**Stage 2 file** — JSON array (`dprs-*.json`) or JSONL (`dprs-*.jsonl`). Each record needs at least:

- `dpr_id`
- `DPR` (the natural-language question)
- `ground_truth.table_uids` — list of table IDs in the cluster (e.g. `["T1","T8"]`)
- `model` (optional; passed through for traceability)

**Table metadata** — Point `--tables-meta` at either:

- a directory of per-table JSON files (e.g. `data/stage1_outputs/tables_clean`), or  
- a single `tables.json` mapping table id → metadata including `rows`.

If you omit `--tables-meta`, Stage 3 tries to infer a `tables_clean`-style path from the repo layout.

**Environment** — Groq (or compatible OpenAI-style API) credentials and models, typically via project root `.env` (`GROQ_API_KEY`, optional `GROQ_MODEL` / `GROQ_MODEL_LIGHT`).

---

## End-to-end flow (one DPR)

1. **Build a cluster database** — For that DPR’s `table_uids`, create an in-memory SQLite database: normalize column names for SQL safety, create tables, insert all rows from metadata. Prompts may see a few sample rows; execution sees the full loaded data.

2. **Decompose** — An LLM proposes 2–3 short sub-questions that can be answered with SELECTs on the cluster schema. A small rule-based pass drops weak or redundant questions; fallbacks exist if decomposition is thin.

3. **Per sub-question: SQL loop** — For each sub-question, run up to **`MAX_SQL_ATTEMPTS` (3)** tries to get a non-trivial, successful query with rows (subject to `--require-non-empty` if you enable it).

4. **Summarize** — For each sub-question that succeeds, a light model writes a mini-summary using **only** the preview rows returned by SQL (not free-form world knowledge).

5. **Final paragraph** — Another call stitches mini-summaries into one DPR-level summary, again grounded in what was summarized.

6. **Persist** — Append one JSON object for this DPR to the output list. If enough sub-questions succeeded (`STAGE3_MIN_SUBQ_SUCCESSES`, default 2 when there are 2+ sub-questions), the DPR is marked executed successfully at the top level.

7. **Sidecar** — After the run, an execution summary JSON is written next to the main output (`{stem}_execution_summary.json`).

---

## LangGraph SQL loop (`pipelinenew.py`)

The sub-question loop is a compiled graph with four ideas:

- **`reason`** — No extra LLM call here. Python looks at the last execution (error text, row count, SQL shape) and sets the next **refinement strategy**: first retry tends toward simplify/drop join; later retries may switch to discovery-style queries or an alternate table in the cluster.

- **`propose_sql`** — First round: plain SQL generation from the sub-question and schema. Later rounds: either **refine** the last SQL (with error feedback, sample rows, strategy text, and a **full execution trace** of prior attempts in the same sub-question) or, optionally, **regenerate** from scratch. A small JSON router LLM can choose `generate_sql` vs `refine_sql`; schema-style errors **force** refine. Disable the extra router call with `STAGE3_SQL_ACTION_ROUTER=0` if you want always-refine after a failure.

- **`execute_sql`** — Validates SQL (schema tables/columns, cartesian joins, fuzzy joins, unsafe text aggregates where applicable), runs SQLite with a wall-clock guard and fetch cap, records `row_count` (via `COUNT(*)` when possible) and a short **preview** of rows. Each attempt is appended to `attempts_log` (SQL, status, error, row count, phase, join/table hints).

- **`decide_next`** — Deterministic: stop on success, stop when attempts are exhausted, stop when the outcome is not worth retrying; otherwise go back to `reason`.

Execution itself is not delegated to the router LLM: the graph always executes in `execute_sql`. That keeps costs and behavior bounded.

---

## DPR-level success vs sub-question success

- Each **sub-question** can succeed or fail independently; failures and every SQL attempt are listed under `subquery_results[].attempts`.

- **DPR `execution_status`** is true only if enough sub-questions hit success (`STAGE3_MIN_SUBQ_SUCCESSES`, environment overridable). The **representative** `generated_sql` and top-level `result.preview` / `result.row_count` come from the **first** successful sub-question, not a merge of all sub-queries.

---

## Output shape (main JSON)

Each element in the output array roughly contains:

- **Identity:** `dpr_id`, `DPR`, `tables`, `ground_truth`
- **Decomposition:** `sub_questions`, `subquery_results` (per sub-question: `attempts`, `final_sql`, `final_execution_status`, `final_row_count`, `mini_summary`)
- **Narrative:** `mini_summaries`, `final_summary`
- **Top-level SQL signal:** `generated_sql`, `execution_status`, `result`
  - On success: `result.validation`, **`result.row_count`** (total rows for that representative query), **`result.preview`** (first few rows as objects — enough for manual sanity checks, not the full result set)
  - On failure: `result.validation`, `result.error`
- **Technical:** `schema_mapping` (original → SQL-safe column names), `llm_model`, `llm_model_summaries`, optional `upstream_model`

Intermediate attempts in `subquery_results[].attempts` store counts and errors but **not** full row previews per attempt; use `result.preview` at DPR level or re-run `final_sql` locally if you need exact row-level regression tests per sub-question.

---

## Safety and grounding

- Prompts include allowed column names derived from the schema string.
- Preflight checks block obvious bad patterns (e.g. cartesian joins, speculative `ON ... LIKE ...`, some unsafe aggregations on text metric columns).
- SQLite execution uses a progress-handler timeout and a small fetch limit so runaway queries do not dominate memory.
- Summaries are instructed to stick to cells present in the preview JSON.

These measures reduce hallucinated columns and unsupported claims; they do not guarantee correct analytics—that is what Stage 4 and human review are for.

---

## Models and rate limits

Defaults (overridable via environment):

- **Heavy model** (`GROQ_MODEL`): decomposition, sub-question SQL generation/refinement, SQL action router.
- **Light model** (`GROQ_MODEL_LIGHT`): mini-summaries and final summary.

Transient rate limits and similar errors are retried with bounded backoff. **Tokens-per-day** exhaustion is fail-fast: the process exits without long sleeps, and any DPRs already finished are flushed to the output file.

---

## Useful environment variables

| Variable | Effect |
|----------|--------|
| `GROQ_API_KEY` | Required for API access. |
| `GROQ_MODEL` / `GROQ_MODEL_LIGHT` | Override default models. |
| `STAGE3_MIN_SUBQ_SUCCESSES` | Minimum successful sub-questions for DPR-level success (default 2 when 2+ sub-questions exist). |
| `STAGE3_DPR_DELAY_SEC` | Optional pause between DPRs to smooth API traffic. |
| `STAGE3_SQL_ACTION_ROUTER` | `0` / `false` / `no` disables the extra LLM call that chooses generate vs refine after a failure. |

---

## CLI

Run from the **repository root**.

**Full file (recommended script):**

```bash
python src/sql_grounding/pipelinenew.py -i "data/stage2_outputs/dprs-qwen3-32b.jsonl" -o "data/stage3/stage3_output.json" --tables-meta "data/stage1_outputs/tables_clean"
```

**Batch slice** (e.g. five DPRs starting at index 0):

```bash
python src/sql_grounding/pipelinenew.py -i "data/stage2_outputs/dprs-qwen3-32b.jsonl" -o "data/stage3/stage3_output_batch1.json" --offset 0 -n 5 --tables-meta "data/stage1_outputs/tables_clean"
```

Next slice: `--offset 5 -n 5`, and so on. Use `--all` to process the entire input (ignores `-n`).

**Strict empty results** — treat zero-row success as failure for retry logic:

```bash
python src/sql_grounding/pipelinenew.py -i "..." -o "..." --tables-meta "..." --require-non-empty
```

**Legacy entry point** (same flags, no LangGraph):

```bash
python src/sql_grounding/pipeline.py -i "..." -o "..." --tables-meta "..."
```

---

## Dependencies

Install project requirements from the repo root:

```bash
pip install -r requirements.txt
```

`pipelinenew.py` needs **`langgraph`** (listed in `requirements.txt`). If imports fail at runtime, install it explicitly: `pip install langgraph`.
