# Stage 3 — SQL generation + grounding (from Stage 2 output)

**Stage 2** (DPRs + `ground_truth.table_uids`) → **Stage 3** (this module) → **Stage 4** (metrics).

## Inputs

- **DPR list:** `dprs-*.json` (JSON array) or `dprs-*.jsonl` (one JSON object per line). Each object should include `dpr_id`, `DPR`, `ground_truth.table_uids` (cluster — not all ~100 tables).

## Table data (Stage 1)

- Pass **`--tables-meta`** to `data/stage1_outputs/tables_clean` (directory of per-table JSON with `rows`), **or** omit it: the pipeline tries to resolve `data/stage1_outputs/tables_clean` from the repo root.
- **SQLite:** For each DPR, only **cluster** tables are loaded, with **all rows** from each JSON. The LLM still sees only a **small sample** of rows in the prompt (`LLM_SAMPLE_ROWS_PER_TABLE`).

## Model

- Default Groq model: **`llama-3.3-70b-versatile`** (override with env **`GROQ_MODEL`**).

## Output sidecar

Next to `-o` JSON, the pipeline also writes **`{stem}_execution_summary.json`** (counts: execution success and positive row counts). LLM-based relevance/quality metrics belong in Stage 4, not here.

## Running (project root)

Full Stage 2 JSONL (e.g. 30 DPRs), inferred `tables_clean`:

```bash
python src/sql_grounding/pipeline.py -i "data/stage2_outputs/dprs-qwen3-32b.jsonl" -o "data/stage3/stage3_output.json"
```

Explicit paths:

```bash
python src/sql_grounding/pipeline.py -i "data/stage2_outputs/your_dprs.jsonl" -o "data/stage3/stage3_output.json" --tables-meta "data/stage1_outputs/tables_clean"
```

- **`-n N`:** process only the first `N` DPRs (smoke test).
- **`--all`:** same as default now (process entire file); kept for clarity.

Optional: `--require-non-empty` to treat empty `SELECT` results as failed grounding.

## Behaviour notes

- **SQL attempts:** Up to **4** generation/refinement attempts per sub-question (`MAX_SQL_ATTEMPTS`).
- **Column whitelist:** Prompts include allowed columns per table to reduce hallucinated fields.
- **Feasibility:** Sub-questions that look like they require missing columns (e.g. `year` when no Year exists) may be **skipped** with `skipped: true` in `subquery_results`.
- **Decomposition:** Prompts discourage lazy generic questions; sub-questions should preserve analytical intent when schema supports it.
- **Empty results:** Refinement prompt steers toward LIKE/fuzzy text match, MIN/MAX discovery, or another cluster table—not blind `SELECT *`.
- **Summaries:** Mini- and final summaries are **evidence-constrained** (only what appears in result rows / prior bullets).

## Troubleshooting “empty” output

If `result.error` mentions **tokens per day (TPD)** or **429**, Groq’s free tier quota is exhausted — the implementation is not “empty SQL”; the API refused the request. Options: wait for reset, upgrade tier, switch `GROQ_MODEL` to another model, or run fewer DPRs (`-n 3`). Partial progress (`sub_questions`, `subquery_results`) is preserved when a later call fails.
