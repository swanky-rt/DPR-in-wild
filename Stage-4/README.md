# Stage-4: DPR Evaluation & Ranking Pipeline

## Overview

Stage-4 evaluates and ranks Data Product Requests (DPRs) generated from Stage-3. It supports two modes:

- **Offline** (`run_eval_v3.py`): evaluates a single merged Stage-3 output file
- **Online** (`run_eval_all_queries.py`): evaluates multiple Stage-3 output files (one per query) and aggregates results

Both modes compute multiple metrics to assess the quality, diversity, and usefulness of each DPR and produce ranked outputs.

---

## Input

### Offline Input

**`stage3_output_final.json`**

Created by merging all Stage-3 batch outputs. Each entry contains:

- DPR text
- Tables used
- Sub-query SQLs
- Summaries

### Online Input

**Multiple stage-3 output files** (one per query), e.g.:
- `stage3_newoutput_batch1.json`
- `stage3_newoutput_batch2.json`
- ...

**`user_queries_from_50matched_dprS.json`** (from Stage-2)

Required for query-based relevance metrics. Each entry contains a `dpr_id` and `user_query` used to match DPRs to their originating queries.

---

## Execution Steps

### Offline Mode

#### 1. Run Stage-3

Follow instructions in the Stage-3 README.

This generates batch output files: `stage3_output_batch*.json`

#### 2. Merge Stage-3 Outputs

```bash
python merge_files.py
```

Generates: `stage3_output_final.json`

#### 3. Run Stage-4 Offline Evaluation

```bash
python run_eval_v3.py \
  --input ../Stage-4/stage3_output_final.json \
  --output_dir ../Stage-4/output \
  --top_k 100 \
  --llm_api_key $LLM_API_KEY \
  --llm_api_base $LLM_API_BASE \
  --llm_model gpt-4
```

---

### Online Mode

#### 1. Run Stage-3 (per query)

Stage-3 produces one output file per query batch.

#### 2. Run Stage-4 Online Evaluation

```bash
python run_eval_all_queries.py \
  --input_dir ../stage-3/data/stage3 \
  --dpr_filename_pattern "stage3_newoutput_batch*.json" \
  --queries_file ../stage-2/data/user_queries_from_50matched_dprS.json \
  --output_dir output \
  --top_k 100 \
  --llm_api_key $LLM_API_KEY \
  --llm_api_base $LLM_API_BASE \
  --llm_model gpt-4
```

> `--queries_file` is optional but required for query-based relevance metrics.
> If not provided, `Query-Summary Rel.` and `Query-DPR Rel.` default to `0.5`.

---

## Metrics

### SQL / Schema-based

| Metric | Description |
|---|---|
| Coverage | `\|tables_used ∩ ground_truth\| / \|ground_truth\|` |
| Complexity | Mean of 5 binary SQL dimensions (join, agg, subquery, etc.) |

### Embedding-based

| Metric | Description |
|---|---|
| Diversity | `1 - avg_cosine_sim(DPR_i, all other DPRs)` |
| Surprisal | `-log P(tables_used) / log(\|all_tables\|)` |
| Uniqueness | `1` if no near-duplicate exists (cosine > 0.85), else `0` |

### LLM-based

| Metric | Description | Mode |
|---|---|---|
| LLM Quality | Is the DPR well-formed and analytical? | Both |
| DPR-Summary Rel. | How well does the summary address the DPR? | Both |
| Query-Summary Rel. | How well does the summary address the user query? | Online only |
| Query-DPR Rel. | How well does the DPR align with the user query? | Online only |

All LLM scores: `0 / 0.5 / 0.75 / 1.0`

---

## Output

### Per-query output (online) / single output (offline)

| File | Description |
|---|---|
| `dpr_ranked_results.json` | Full ranked DPR list with all metrics |
| `dpr_ranking_summary.txt` | Human-readable ranking with SQL and summaries |
| `metrics_stats.txt` | Min / Max / Mean per metric |
| `metrics_avg_this_query.txt` | Per-query average stats *(online only)* |

### Aggregate output (online only, written to top-level output dir)

| File | Description |
|---|---|
| `metrics_avg_all_queries.txt` | Aggregate stats across all queries and DPRs |
| `summary_all_queries.json` | Structured per-query breakdown with averages |

---

## Optional: LLM Evaluation

Set environment variables:

```bash
export LLM_API_KEY=your_key
export LLM_API_BASE=https://api.openai.com/v1
export LLM_MODEL=gpt-4
```

Or pass via CLI:

```bash
--llm_api_key <key> --llm_api_base <url> --llm_model <model>
```

> If not provided, LLM scores default to `0.5`.

---

## Folder Structure

```
698DS/
├── stage-2/
│   └── data/
│       └── user_queries_from_50matched_dprS.json
│
├── stage-3/
│   └── data/stage3/
│       ├── stage3_newoutput_batch1.json
│       ├── stage3_newoutput_batch2.json
│       └── ...
│
└── Stage-4/
    ├── stage3_output_final.json        ← offline input (merged)
    ├── run_eval_v3.py                  ← offline evaluation script
    ├── run_eval_all_queries.py         ← online evaluation script
    └── output/
        ├── dpr_ranked_results.json
        ├── dpr_ranking_summary.txt
        ├── metrics_stats.txt
        ├── metrics_avg_all_queries.txt ← online only
        ├── summary_all_queries.json    ← online only
        └── stage3_newoutput_batch1/    ← online only (per-query folders)
            ├── dpr_ranked_results.json
            ├── dpr_ranking_summary.txt
            ├── metrics_stats.txt
            └── metrics_avg_this_query.txt
```

---

## Notes

- Execution summary files (`*_execution_summary.json`) are automatically excluded
- Input merging is deterministic
- Designed for scalable DPR evaluation
- LLM usage is optional for all metrics except Quality, DPR-Summary, Query-Summary, and Query-DPR relevance
- `run_eval_v3.py` (offline) is unchanged — all online additions are in `run_eval_all_queries.py`