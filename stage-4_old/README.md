# Stage-4: DPR Evaluation & Ranking Pipeline

## Overview

Stage-4 evaluates and ranks Data Product Requests (DPRs) generated from Stage-3. It consumes a merged DPR output file and computes multiple metrics to assess the quality, diversity, and usefulness of each DPR.

The pipeline produces:

- Ranked DPRs based on a composite score
- Metric summaries
- Human-readable evaluation outputs

There are two evaluation scripts depending on the pipeline variant:

| Script | Use Case |
|---|---|
| `run_eval_v3.py` | Offline (no query) and Offline+Query |
| `run_eval_all_queries.py` | Online (multi-query, UCB-based) |

---

## Input

### Primary Input

**`stage3_output_final.json`**

Created by merging all Stage-3 batch outputs. Each entry contains:

- DPR text
- Tables used
- Sub-query SQLs
- Sub-query execution results
- `final_summary` — the stage-3 generated answer (used by Surprisal metric)
- Schema mapping

---

## Execution Steps

### 1. Run Stage-3

Follow instructions in the Stage-3 README (Rishitha can provide details if needed).

This generates batch output files: `stage3_output_batch*.json`

### 2. Merge Stage-3 Outputs

```bash
python merge_files.py
```

Generates: `stage3_output_final.json`

### 3. Run Stage-4 Evaluation

**Offline (no query) or Offline+Query:**

```bash
python run_eval_v3.py \
  --input ../stage-4/stage3_output_final.json \
  --output_dir ../stage-4/output \
  --top_k 100
```

**Online (multi-query):**

```bash
python run_eval_all_queries.py \
  --input_dir  ../stage-3_old/data/stage3_outputs/online_with_query \
  --dpr_filename_pattern "Q*--online_stage3_output.json" \
  --output_dir output/online_eval_final \
  --queries_file data/online_user_queries.json \
  --llm_api_key  $LLM_API_KEY \
  --llm_api_base $LLM_API_BASE \
  --llm_model    $LLM_MODEL \
  --top_k 100
```

---

## Pipeline Steps

### Offline (`run_eval_v3.py`)

#### 1. Load DPR Data
Reads merged JSON file and processes each DPR independently.

#### 2. Compute Per-DPR Metrics

**SQL / Schema-based:**
- **Coverage** — `|tables_used ∩ ground_truth| / |ground_truth|`
- **Complexity** — Halstead volume (Halstead 1977) normalized to [0,1], averaged with 5 binary SQL dimensions (multi_table, join, agg, subquery, multi_entity). Citation: Vashistha et al., "Measuring Query Complexity in SQLShare Workload."
- **LLM Quality** — GPT-4 scores DPR quality: 0 / 0.5 / 0.75 / 1.0
- **DPR-Summary Relevance** — GPT-4 scores how well `final_summary` addresses the DPR

**Pool-wide (computed across all DPRs together):**
- **Diversity** — `1 - avg_cosine_sim(DPR_i, all other DPRs)` via TF-IDF embeddings
- **Surprisal** — AutoDiscovery beta-distribution method: LLM is queried k=5 times on `final_summary` as hypothesis; surprisal = `|1 - μ_prior|` where `μ_prior = α/k`. Falls back to frequency-based method if no LLM API key is provided.
- **Uniqueness** — 1 if no near-duplicate exists (cosine > 0.85), else 0

#### 3. Ranking

DPRs are ranked using a normalized weighted combined score:

| Metric | Weight |
|---|---|
| Coverage | 0.2143 |
| Complexity | 0.1429 |
| Diversity | 0.1429 |
| Surprisal | 0.1429 |
| Uniqueness | 0.1429 |
| LLM Quality | 0.1071 |
| DPR-Summary Rel. | 0.1071 |

Weights are normalized to sum to 1.0.

#### 4. Output Generation

- `dpr_ranked_results.json`
- `dpr_ranking_summary.txt`
- `metrics_stats.txt`

---

### Online (`run_eval_all_queries.py`)

Runs three passes across all query files:

**Pass 1 — Per-query evaluation**
Calls `run_eval_v3.run_pipeline()` for each query file. Computes all 7 metrics including per-DPR beta surprisal. Outputs are not written yet.

**Pass 2 — Cross-query pool recomputation**
Diversity and Uniqueness are recomputed across ALL DPRs from ALL queries so each DPR's score reflects how different it is from the full pool. Surprisal is NOT recomputed here — it is computed per-DPR in Pass 1 using the beta-distribution method and is independent of pool composition. Combined scores are recalculated after this pass.

**Pass 3 — Query-relevance enrichment (display-only)**
Adds two supplementary metrics per DPR — not included in combined score:

| Metric | Description |
|---|---|
| Query-DPR Relevance | How well the DPR aligns with the user query |
| Query-Summary Relevance | How well the summary addresses the user query |

These are display-only because offline has no user query and including them would make the three pipelines incomparable.

Per-query and aggregate stats files are written after all passes complete.

---

## LLM Configuration

Set environment variables:

```bash
export LLM_API_KEY=your_key
export LLM_API_BASE=https://thekeymaker.umass.edu/v1
export LLM_MODEL=qwen.qwen3-235b-a22b-2507-v1:0
```

Or pass via CLI:

```bash
--llm_api_key <key> --llm_api_base <base_url> --llm_model <model>
```

> If LLM API key is not provided:
> - LLM Quality and Summary Relevance default to `0.5`
> - Surprisal falls back to frequency-based method (table combination rarity)

---

## Output Files

### Per-query (offline) or per-query subfolder (online):

| File | Description |
|---|---|
| `dpr_ranked_results.json` | Full ranked DPR data with all metrics |
| `dpr_ranking_summary.txt` | Human-readable ranking with SQL and summaries |
| `metrics_stats.txt` | Min / Max / Mean per metric |

### Online only (aggregate):

| File | Description |
|---|---|
| `metrics_avg_this_query.txt` | Per-query metric averages |
| `metrics_avg_all_queries.txt` | Aggregate across all queries |
| `summary_all_queries.json` | JSON summary across all queries |

---

## Folder Structure

```
698DS/
├── stage-3/
│   └── data/stage3/
│       └── stage3_output_batch*.json
│
└── Stage-4/
    ├── stage3_output_final.json
    ├── run_eval_v3.py           ← offline eval
    ├── run_eval_all_queries.py  ← online eval
    └── output/
        ├── dpr_ranked_results.json
        ├── dpr_ranking_summary.txt
        └── metrics_stats.txt
```

---

## Notes

- Execution summary files are excluded from merging
- Input merging is deterministic
- LLM usage is optional but recommended — significantly improves Surprisal quality
- Surprisal uses `final_summary` from Stage-3 as the hypothesis — ensure this field is populated in input records
- The three pipeline variants (offline, offline+query, online) use the same 7-metric combined score for comparability; query relevance metrics are supplementary and display-only