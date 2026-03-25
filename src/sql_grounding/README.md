# Stage 3: SQL Grounding and Evidence Synthesis

Stage 3 is the execution layer between DPR generation (Stage 2) and evaluation (Stage 4).

- **Input:** DPRs with associated table clusters (`ground_truth.table_uids`).
- **Core job:** Convert each DPR into SQL-backed sub-questions, execute them against real table rows, and synthesize evidence-constrained text.
- **Output:** Per-DPR execution artifacts and summaries, plus an execution summary sidecar.

---

## What Stage 3 Implements

For each DPR:

1. Build an in-memory SQLite database for the DPR's cluster tables.
2. Generate a small set of SQL-answerable sub-questions.
3. Generate/refine SQL for each sub-question (up to a fixed number of attempts).
4. Execute SQL safely, validate outputs, and collect previews/counts.
5. Produce mini-summaries from successful SQL result rows.
6. Produce one final synthesis paragraph from mini-summaries.
7. Save structured results and aggregate execution stats.

This design keeps final text grounded in executed table evidence.

---

## Inputs

### Stage 2 DPR file

Pass one of:

- `dprs-*.json` (JSON array), or
- `dprs-*.jsonl` (one object per line)

Each row should include:

- `dpr_id`
- `DPR`
- `ground_truth.table_uids` (cluster table IDs, for example `["T2", "T3"]`)
- `model` (optional upstream metadata)

### Table metadata

Provide `--tables-meta` pointing to either:

- a directory like `data/stage1_outputs/tables_clean` (one JSON per table), or
- a single `tables.json` mapping `table_id -> table_meta`.

If omitted, the pipeline attempts to infer a `tables_clean` path from repo conventions.

---

## How Each Step Is Implemented

### Database Initialization

- For each DPR, Stage 3 reads its `ground_truth.table_uids` cluster.
- It calls `_build_cluster_sqlite_from_table_metadata(...)` to create an in-memory SQLite database.
- For each table in the cluster:
  - columns are normalized into SQL-safe identifiers,
  - the table is created in SQLite with inferred types,
  - rows from `meta["rows"]` are inserted (row-wise or HybridQA flattened layout).
- Result: SQL runs on real cluster data, not a schema-only mock.

### Decomposition

- Stage 3 calls `generate_subquestions(...)` to split the DPR into 2 to 3 SQL-answerable sub-questions.
- It then applies quality filtering (`_quality_select_subquestions`) to remove vague or redundant prompts.
- If decomposition is weak, fallback atomic questions are generated to guarantee coverage.
- Decomposition is constrained by schema context and table IDs to keep questions executable.

### Agentic SQL Loop

- For each sub-question, Stage 3 runs up to `MAX_SQL_ATTEMPTS = 3`.
- Attempt pattern:
  1. initial SQL generation,
  2. simplify/drop-join correction,
  3. adaptive recovery (`discovery` or `alternate_table`) based on prior error signals.
- The loop tracks retry phases and errors per attempt for transparency in `subquery_results`.
- It rejects risky SQL patterns (cartesian joins, speculative fuzzy joins, trivial probes) and keeps refining.

### Execution & Validation

- SQL is executed via `execute_and_validate(...)` with safety limits:
  - SQLite progress-handler timeout,
  - fetch cap for previews,
  - row-count estimation via wrapped `COUNT(*)` query when possible.
- Validation marks success/failure and records:
  - execution status,
  - row count,
  - preview rows,
  - error message (if any).
- Empty-result behavior is handled in the retry loop to attempt better SQL before giving up.

### Evidence Synthesis

- On successful sub-queries with rows, Stage 3 creates mini-summaries using `summarize_subquestion_result(...)`.
- Mini-summaries are prompt-constrained to only use values present in SQL result previews.
- Stage 3 then generates one final paragraph with `generate_final_summary(...)` from the mini-summaries.
- This keeps the final text tied to observed evidence from the executed queries.

### Artifact Generation

- For each DPR, Stage 3 writes a structured object with:
  - input DPR metadata,
  - sub-questions and attempt history,
  - SQL/execution outputs,
  - mini and final summaries.
- After the run, it writes:
  - main output JSON (`-o`),
  - execution sidecar `{output_stem}_execution_summary.json`.
- If TPD quota is hit, Stage 3 fail-fast exits but still flushes completed DPR artifacts.

---

## Database Grounding Details

Stage 3 creates a fresh in-memory SQLite DB per DPR cluster via
`_build_cluster_sqlite_from_table_metadata(...)`.

- Uses declared table columns and normalized SQL-safe column names.
- Inserts **all rows from `meta["rows"]`** for each cluster table.
- Supports both row-wise JSON and flattened HybridQA-style cell layouts.
- Uses small prompt samples for LLM grounding, but SQL executes on the full loaded rows.

So SQL validation runs against real table content, not schema-only tables.

---

## SQL Generation and Retry Policy

Per sub-question, Stage 3 runs up to `MAX_SQL_ATTEMPTS = 3`:

1. Initial SQL generation.
2. Simplify/drop-join style correction.
3. Adaptive recovery (`discovery` or `alternate_table`) based on prior failures/signals.

The retry loop catches and adapts to:

- schema errors (`no such column`, etc.),
- execution timeout signals,
- disallowed join patterns,
- trivial probe SQL,
- empty-result outcomes.

The objective is to return non-trivial, schema-valid SQL with useful rows when possible.

---

## Safety and Anti-Hallucination Controls

Implemented safeguards include:

- allowed-column whitelist in prompts,
- detection of speculative fuzzy joins (`ON ... LIKE ...`),
- cartesian-pattern guardrails,
- typed checks for unsafe text-based metric aggregation,
- SQLite execution wall-time cap and fetch cap,
- evidence-constrained summarization prompts.

These controls reduce fabricated columns, poor joins, and unsupported claims.

---

## Models

By default:

- `GROQ_MODEL=llama-3.3-70b-versatile` for decomposition + SQL reasoning.
- `GROQ_MODEL_LIGHT=llama-3.1-8b-instant` for mini/final summary generation.

Override with environment variables when needed.

---

## Rate Limits and Failure Behavior

### TPM / transient 429-style failures

Stage 3 retries with bounded backoff for transient rate-limit/overload patterns.

### TPD (tokens per day)

TPD exhaustion is **fail-fast** by design:

- no long sleep,
- current run stops quickly,
- completed DPR outputs are written before exit.

This avoids wasting compute during long quota windows.

---

## Outputs

### Main output (`-o`)

JSON list, one object per DPR, including:

- identifiers and original DPR text,
- sub-questions and per-sub-question attempts/results,
- mini summaries and final summary,
- representative SQL and execution status,
- execution result payload (`validation`, `row_count`, `preview` / `error`),
- schema mapping and model metadata.

---

## CLI Usage

Run from repo root.

### Full run

```bash
python src/sql_grounding/pipeline.py -i "data/stage2_outputs/dprs-qwen3-32b.jsonl" -o "data/stage3/stage3_output.json"
```

### Explicit table metadata path

```bash
python src/sql_grounding/pipeline.py -i "data/stage2_outputs/your_dprs.jsonl" -o "data/stage3/stage3_output.json" --tables-meta "data/stage1_outputs/tables_clean"
```

### Batch/slice runs

- `--offset`: start index in DPR list
- `-n` / `--limit`: number of DPRs to run
- `--all`: process full input (overrides limit)

Example:

```bash
python src/sql_grounding/pipeline.py -i "data/stage2_outputs/dprs-qwen3-32b.jsonl" -o "data/stage3/stage3_output_batch1.json" --offset 0 -n 5 --tables-meta "data/stage1_outputs/tables_clean"
```

### Optional strictness flag

- `--require-non-empty`: treats empty SQL results as failed execution.

---

