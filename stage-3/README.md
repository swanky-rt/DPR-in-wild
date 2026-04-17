# Stage 3: SQL grounding and evidence synthesis

Stage 3 takes **Stage 2 DPRs** (questions plus which tables belong to each cluster) and turns them into **grounded, executable evidence**. For every DPR it loads those tables into an in-memory SQLite database, asks an LLM to decompose the question into sub-questions, runs a **bounded SQL retry loop** (LangGraph), and writes structured JSON that downstream stages can evaluate.

This document is the single place to understand **inputs, configuration, how a run behaves, and what gets written to disk**.

---

## What Stage 3 does (one sentence)

It answers: *“Given this DPR and these cluster tables, what SQL can we run that actually returns rows, and what can we say about the answer using only those rows?”*

---

## Repository layout (what matters here)

```
stage-3/
├── .env                          # LLM credentials (not committed; you create it)
├── requirements.txt              # Python deps for this stage
├── README.md                     # This file
├── data/
│   ├── stage1_outputs/tables_clean/   # Per-table JSON (T1.json …)
│   ├── stage2_outputs/                # DPR lists (.json / .jsonl)
│   └── stage3_outputs/                # Split: offline/ vs online/ runs
│       ├── offline/                   # Main (pipelinenew.py) outputs + merge_files.py
│       └── online/                    # pipelinenew_online.py outputs (same code path)
└── src/sql_grounding/
    ├── pipelinenew.py            # Main entry (write outputs under stage3_outputs/offline/)
    └── pipelinenew_online.py     # Same implementation (write outputs under stage3_outputs/online/)
```

Run commands below assume your **current working directory is the repo root** (`dpr-discovery/`), so paths start with `stage-3/`.

---

## Dependencies

From the repo root:

```bash
pip install -r stage-3/requirements.txt
```

You need: **`openai`**, **`python-dotenv`**, **`langgraph`**.  
You do **not** need the `litellm` Python package for `pipelinenew.py`: the script uses the **OpenAI-compatible** HTTP API (your campus gateway or Groq both speak that shape).

---

## Inputs

### 1. Stage 2 DPR file

Formats:

- **JSON array** (`.json`), or  
- **JSONL** (`.jsonl`, one JSON object per line)

Each row must include at least:

| Field | Role |
|--------|------|
| `dpr_id` | Stable id (string or number; treated as string when merging) |
| `DPR` | Full natural-language question |
| `ground_truth.table_uids` | List of table ids in this cluster, e.g. `["T2","T7"]` |

Optional fields (e.g. upstream `model`) are preserved for traceability when present.

### 2. Stage 1 table metadata

Pass **`--tables-meta`** pointing to either:

- A directory of per-table JSON files (**recommended**), e.g. `stage-3/data/stage1_outputs/tables_clean`, or  
- A single aggregated `tables.json` if your project uses that layout.

The pipeline loads **only** the tables referenced by each DPR’s `table_uids`.

---

## LLM configuration (`.env`)

Configuration lives in **`stage-3/.env`**. The pipeline loads it automatically (via `Path(__file__).parents[2] / ".env"`).

### LLM API configuration (required)

Stage 3 uses an **OpenAI-compatible** chat endpoint. Provide these in `stage-3/.env`:

```env
LLM_API_KEY=...
LLM_API_BASE=https://thekeymaker.umass.edu/v1
LLM_MODEL=gpt4o
```

`LLM_API_BASE` must point to the server’s OpenAI-compatible base URL (commonly it ends with `/v1`, but follow your Unity/LiteLLM docs).

### Single model for all LLM steps

The pipeline uses **one** chat model for decomposition, SQL, refinement, and summaries. There is no separate “light” model variable anymore; output JSON may still include `llm_model_summaries` for compatibility, set to the same model id as `llm_model`.

---

## How a run behaves

### Per DPR (high level)

1. Build **in-memory SQLite** from the DPR’s `table_uids` and Stage 1 metadata.  
2. **Decompose** the DPR into 2–3 sub-questions (shorter DPRs may use 2).  
3. For each sub-question, run the **LangGraph SQL loop** with at most **`MAX_SQL_ATTEMPTS`** (default 3).  
4. On success, write **mini-summaries** from preview rows only (evidence-bound).  
5. Build a **final summary** from those mini-summaries.  
6. Append one object to the output list.

### Success at DPR level

A DPR can succeed on some sub-questions and fail on others. **`execution_status`** is `true` only if enough sub-questions succeed: controlled by **`STAGE3_MIN_SUBQ_SUCCESSES`** (default **2** when there are at least two sub-questions).

### Checkpoints and failures

- After **each** DPR finishes, the pipeline **rewrites the main output JSON** and the **`_execution_summary.json`** sidecar. If the process stops midway, you keep all completed DPRs up to that point.  
- **Hard quota** errors (e.g. tokens-per-day, `insufficient_quota`, `quota exceeded`) **fail fast**: already-completed DPRs are saved, then the process exits with a non-zero code. Resume with **`--offset` / `-n`** on the same input if needed.

### Rate limits (non-fatal)

Transient errors (429, overload, timeouts) are retried with capped backoff. See **`MAX_API_RETRIES`** and **`MAX_RATE_LIMIT_SLEEP_SEC`** in `pipelinenew.py`.

---

## Outputs

### Main file (`-o`)

A **JSON array**: one object per processed DPR, including:

- `sub_questions`, `subquery_results` (with `attempts` traces)  
- `generated_sql`, `execution_status`, `result`  
- `mini_summaries`, `final_summary`  
- `schema_mapping`, `llm_model`, `llm_model_summaries`, etc.

### Sidecar: execution summary

Next to the main file:

`{output_basename_without_extension}_execution_summary.json`

Aggregated counts (e.g. how many DPRs executed successfully, how many had positive row counts).

---

## Running Stage 3

Always from **repo root**, unless you adjust paths.

### Full file (all rows in the input)

Omitting `--limit` processes **every** row in the JSONL (not only 30—count lines in your Stage 2 file to know how many).

```bash
python stage-3/src/sql_grounding/pipelinenew.py \
  -i stage-3/data/stage2_outputs/dprs-qwen3-32b-merged.jsonl \
  -o stage-3/data/stage3_outputs/offline/stage3_offline_output.json \
  --tables-meta stage-3/data/stage1_outputs/tables_clean
```

### Batched runs (same input, different slices)

Use **`--offset`** (0-based index into the Stage 2 list) and **`-n`** (count):

```bash
python stage-3/src/sql_grounding/pipelinenew.py \
  -i stage-3/data/stage2_outputs/dprs-qwen3-32b-merged.jsonl \
  -o stage-3/data/stage3_outputs/offline/stage3output_batch1.json \
  --offset 0 -n 5 \
  --tables-meta stage-3/data/stage1_outputs/tables_clean
```

Repeat with `--offset 5 -n 5`, then `10`, `15`, … as needed.

### Same pipeline, `online/` output folder

Use the same script or `pipelinenew_online.py`; only the **`-o`** path changes:

```bash
python stage-3/src/sql_grounding/pipelinenew_online.py \
  -i stage-3/data/stage2_outputs/online-dprs-qwen3-32b.jsonl \
  -o stage-3/data/stage3_outputs/online/stage3_online_output.json \
  --tables-meta stage-3/data/stage1_outputs/tables_clean
```

### Stricter grounding (non-empty results)

```bash
python stage-3/src/sql_grounding/pipelinenew.py \
  -i stage-3/data/stage2_outputs/dprs-qwen3-32b-merged.jsonl \
  -o stage-3/data/stage3_outputs/offline/stage3_strict.json \
  --tables-meta stage-3/data/stage1_outputs/tables_clean \
  --require-non-empty
```

---

## Merging batch JSON files into one bundle

If you produced **`stage3output_batch1.json`**, **`stage3output_batch2.json`**, … under **`offline/`**, use **`stage-3/data/stage3_outputs/offline/merge_files.py`**:

- Finds all **`stage3output_batch*.json`** in the chosen directory (skips `*_execution_summary.json`).
- Merges lists into one JSON array and **de-duplicates** by **`dpr_id`** (later batch number wins).

**One-shot** (defaults: source = `offline/`, output = `offline/stage3_offline_output_groq.json`):

```bash
python stage-3/data/stage3_outputs/offline/merge_files.py
```

**Custom paths:**

```bash
python stage-3/data/stage3_outputs/offline/merge_files.py \
  --source-dir stage-3/data/stage3_outputs/offline \
  --output stage-3/data/stage3_outputs/offline/stage3_offline_output_groq.json
```

**Auto-refresh** when new batch files appear:

```bash
python stage-3/data/stage3_outputs/offline/merge_files.py --watch --interval-sec 5
```

For **online** runs, keep batch files under **`online/`** and either copy `merge_files.py` there or pass `--source-dir` / `--output` pointing at `online/`.

---

## Useful environment knobs (optional)

| Variable | Role |
|----------|------|
| `STAGE3_MIN_SUBQ_SUCCESSES` | Minimum successful sub-questions for DPR success (default `2`) |
| `STAGE3_DPR_DELAY_SEC` | Seconds to sleep between DPRs (default `20`) |
| `STAGE3_SQL_ACTION_ROUTER` | `1` / `0` — extra lightweight call to route SQL actions |

See comments at the top of `pipelinenew.py` for prompt caps and SQL safety limits.



---

## Quick checklist before you run

1. `pip install -r stage-3/requirements.txt`  
2. `stage-3/.env` with **`LLM_API_KEY`**, **`LLM_API_BASE`**, and **`LLM_MODEL`** as above  
3. Stage 2 file path and **`--tables-meta`** path correct  
4. For batches: plan **`--offset`** / **`-n`** slices; merge with **`offline/merge_files.py`** when needed  

That is the full Stage 3 path from data to merged outputs.
