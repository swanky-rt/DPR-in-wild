# Stage 3 — SQL generation + grounding (from Stage 2 output)

This folder contains **Stage 3** of your project pipeline:

**Stage 2 output (Athulya)** → **Stage 3 (this code: SQL generation + grounding)** → **Stage 4 metrics**.

## Inputs (Stage 2)

Stage 3 accepts either of these Stage‑2 output files:

### 1) DPR list (recommended)
`dprs-*.json` (example: `stage-2 sampledata/output/dprs-llama-3.3-70b-versatile.json`)

A JSON list of objects:
- **dpr_id**
- **DPR**
- **model** (optional; which LLM produced the DPR)
- **ground_truth.table_uids** (e.g. `["T2","T3"]`)

Stage 3 will automatically infer `tables.json` from the stage‑2 folder structure:
`stage-2 sampledata/input/tables.json`

### 2) Filtered clusters
`filtered_clusters.json`

This file contains per‑DPR table metadata but does **not** contain the DPR text, so Stage 3 automatically joins it with a neighboring `dprs-*.json`.

## Output (for Stage 4 metrics)

Stage 3 writes a JSON list, one object per DPR:
- **dpr_id**
- **DPR**
- **tables**: list of `T*` table ids used
- **ground_truth**
- **generated_sql**
- **execution_status**: whether SQL executed successfully against the schema
- **result**: `{ validation, row_count, preview }` or `{ validation, error }`
- **schema_mapping**: original column → SQL-safe column (important because stage‑2 schemas often include spaces like `"GDP Nominal (USD)"`)
- **llm_model**: model used for SQL generation
- **upstream_model**: model that produced the DPR upstream (if present)

## Grounding note (important)

Stage‑2 sample data includes **schemas only**, not table rows, so “grounding” here defaults to:
- **SQL executes successfully against the schema** (even if 0 rows are returned).

If later you have real rows and want to fail empty results, run with `--require-non-empty`.

## Running

From project root:

```bash
python -m src.sql_grounding.pipeline -i "stage-2 sampledata/output/dprs-llama-3.3-70b-versatile.json" -o stage3_output.json -n 5
```

If `tables.json` cannot be inferred, pass it explicitly:

```bash
python -m src.sql_grounding.pipeline -i <dprs.json> -o stage3_output.json -n 5 --tables-meta "stage-2 sampledata/input/tables.json"
```

To run all DPRs, set `-n 100` (or whatever count you have).
