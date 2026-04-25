# Stage 3: SQL Grounding and Evidence Synthesis

Stage 3 converts Stage 2 DPR files into grounded SQL evidence.  
For each DPR, it loads the relevant tables into in-memory SQLite, decomposes the DPR into sub-questions, runs a bounded LangGraph SQL loop, and writes structured output with execution traces and summaries.

This README reflects the **current completed setup** with three Stage 2 variants:

- baseline `offline`
- `offline_with_query`
- `online_with_query`

The important point: **Stage 3 implementation is the same for all variants**.  
What changes is only the Stage 2 input organization and output folder paths.

---

## What is shared across all variants

All variants use the same Stage 3 engine (`run_stage3_pipeline` in `pipelinenew_query.py`) with the same behavior:

1. Build cluster SQLite from `ground_truth.table_uids`
2. Generate 2–3 sub-questions
3. Run LangGraph SQL loop with bounded retries
4. Produce mini summaries from result previews
5. Produce final summary from mini summaries
6. Write DPR output row + execution summary sidecar

So you can compare variants fairly: only Stage 2 DPR generation strategy differs.

---

## Repository structure (Stage 3 relevant)

```text
stage-3/
├── .env
├── requirements.txt
├── README.md
├── data/
│   ├── stage1_outputs/tables_clean/
│   ├── stage2_outputs/
│   │   ├── dprs-qwen3-32b-merged.jsonl               # baseline offline input
│   │   ├── Q1--offline.jsonl ... Q5--offline.jsonl   # offline_with_query flat files
│   │   └── dprs_online_with_query/
│   │       ├── Q1--online.jsonl ... Q5--online.jsonl # online_with_query files
│   └── stage3_outputs/
│       ├── offline/
│       ├── offline_with_query/
│       ├── online/
│       └── online_with_query/
└── src/sql_grounding/
    ├── pipelinenew.py                 # baseline stage-3 entry
    ├── pipelinenew_query.py           # query-variant stage-3 engine
    └── run_stage3_query_sets.py       # batch orchestrator for query variants
```

Commands below assume you run from repo root: `dpr-discovery/`.

---

## Dependencies

```bash
pip install -r stage-3/requirements.txt
```

Required packages:

- `openai`
- `python-dotenv`
- `langgraph`

---

## Required input schema for Stage 3

Each DPR row must include:

- `dpr_id` (can be null, but recommended to provide)
- `DPR`
- `ground_truth.table_uids`

Supported formats:

- JSON list (`.json`)
- JSONL (`.jsonl`)

---

## LLM configuration (`stage-3/.env`)

Stage 3 is fully env-driven for model connection. Set:

```env
LLM_API_KEY=...
LLM_API_BASE=https://<your-openai-compatible-endpoint>/v1
LLM_MODEL=<model-name>
```

Examples:

- Campus gateway (`thekeymaker`)
- Groq via OpenAI-compatible base URL
- Unity/local OpenAI-compatible LLM gateway

No code changes are required when switching providers/models if the endpoint is OpenAI-compatible.

---

## Outputs written by Stage 3

For each run output file `X.json`, Stage 3 also writes:

- `X_execution_summary.json`

Main output row includes:

- `dpr_id`, `DPR`, `tables`, `ground_truth`
- `sub_questions`, `subquery_results` (with attempt traces)
- `generated_sql`, `execution_status`, `result`
- `mini_summaries`, `final_summary`

Checkpoint behavior:

- Output is rewritten after each DPR.
- On hard quota errors (`insufficient_quota`, etc.) it fails fast but preserves completed rows.

---

## Variant 1: Baseline `offline`

Use when input is a single merged Stage 2 DPR file.

```bash
python stage-3/src/sql_grounding/pipelinenew.py \
  -i stage-3/data/stage2_outputs/dprs-qwen3-32b-merged.jsonl \
  -o stage-3/data/stage3_outputs/offline/stage3_offline_output.json \
  --tables-meta stage-3/data/stage1_outputs/tables_clean
```

Optional batched baseline run:

```bash
python stage-3/src/sql_grounding/pipelinenew.py \
  -i stage-3/data/stage2_outputs/dprs-qwen3-32b-merged.jsonl \
  -o stage-3/data/stage3_outputs/offline/stage3output_batch1.json \
  --offset 0 -n 5 \
  --tables-meta stage-3/data/stage1_outputs/tables_clean
```

---

## Variant 2: `offline_with_query` (Q1..Q5 flat files)

Input layout:

- `stage-3/data/stage2_outputs/Q1--offline.jsonl`
- ...
- `stage-3/data/stage2_outputs/Q5--offline.jsonl`

Run all files in one command:

```bash
python stage-3/src/sql_grounding/run_stage3_query_sets.py --mode offline
```

Outputs go to:

- `stage-3/data/stage3_outputs/offline_with_query/`

Expected output names:

- `Q1--offline_stage3_output.json`
- ...
- `Q5--offline_stage3_output.json`
- plus per-file execution summary JSONs and `run_manifest.json`

---

## Variant 3: `online_with_query` (Q1..Q5 online files)

Input layout:

- `stage-3/data/stage2_outputs/dprs_online_with_query/Q1--online.jsonl`
- ...
- `stage-3/data/stage2_outputs/dprs_online_with_query/Q5--online.jsonl`

Run all files in one command:

```bash
python stage-3/src/sql_grounding/run_stage3_query_sets.py --mode online
```

Outputs go to:

- `stage-3/data/stage3_outputs/online_with_query/`

Expected output names:

- `Q1--online_stage3_output.json`
- ...
- `Q5--online_stage3_output.json`
- plus per-file execution summary JSONs and `run_manifest.json`

---

## Post-processing helper for missing `dpr_id` in online_with_query outputs

If Stage 2 online files were generated without `dpr_id`, Stage 3 outputs can contain null ids.  
Use this helper to assign deterministic IDs:

- `q1_1 ... q1_20` for Q1 file
- `q2_1 ... q2_20` for Q2 file
- etc.

Command:

```bash
python stage-3/data/stage3_outputs/online_with_query/assign_query_dpr_ids.py
```

---

## Merge helper for baseline offline batch files

If you ran baseline offline as multiple `stage3output_batch*.json` files:

```bash
python stage-3/data/stage3_outputs/offline/merge_files.py
```

Default merged output:

- `stage-3/data/stage3_outputs/offline/stage3_offline_output_groq.json`

---

## Quick run checklist

1. Install deps: `pip install -r stage-3/requirements.txt`
2. Set `.env`: `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL`
3. Confirm Stage 2 inputs include `DPR` + `ground_truth.table_uids`
4. Pick variant command:
   - baseline offline -> `pipelinenew.py`
   - query variants -> `run_stage3_query_sets.py --mode offline|online`
5. Verify output + `*_execution_summary.json`

---

## Final note on comparability

For all three variants, Stage 3 algorithm is the same.  
Differences in downstream performance come from Stage 2 DPR generation differences, not a different Stage 3 implementation.
