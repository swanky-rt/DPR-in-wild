# Stage 3 — SQL generation + grounding (from Stage 2 output)

This folder contains **Stage 3** of your project pipeline:

**Stage 2 output** → **Stage 3 (this code: SQL generation + grounding)** → **Stage 4 metrics**.

## Inputs (Stage 2)

Stage 3 accepts either a DPR list (`dprs-*.json`) or `filtered_clusters.json`. See module docstring in `pipeline.py` for fields.

## Output

Stage 3 writes a JSON list to the path you pass with `-o`. Use **`data/stage3/stage3_output_final.json`** so output lives under `data/stage3/`.

## Running

From the **project root**:

```bash
python src/sql_grounding/pipeline.py -i "data/stage2/output/run_test_groq/dprs.json" -o "data/stage3/stage3_output_final.json" -n 5 --tables-meta "data/stage1/tables_clean" --require-non-empty
```

Paths are relative to the project root. Tables metadata can be a directory (e.g. `data/stage1/tables_clean`) or a single `tables.json` file.
