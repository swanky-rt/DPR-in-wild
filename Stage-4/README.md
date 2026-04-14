# Stage-4: DPR Evaluation & Ranking Pipeline

## Overview

Stage-4 evaluates and ranks Data Product Requests (DPRs) generated from Stage-3. It consumes a merged DPR output file and computes multiple metrics to assess the quality, diversity, and usefulness of each DPR.

The pipeline produces:

- Ranked DPRs based on a composite score
- Metric summaries
- Human-readable evaluation outputs

---

## Input

### Primary Input

**`stage3_output_final.json`**

Created by merging all Stage-3 batch outputs. Each entry contains:

- DPR text
- Tables used
- Sub-query SQLs
- Summaries

---

## Execution Steps

To run the complete pipeline:

### 1. Run Stage-3

Follow instructions in the Stage-3 README (Rishitha can provide details if needed).

This generates batch output files: `stage3_output_batch*.json`

### 2. Merge Stage-3 Outputs

```bash
python merge_files.py
```

Generates: `stage3_output_final.json`

### 3. Run Stage-4 Evaluation

```bash
python run_eval_v3.py \
  --input ../Stage-4/stage3_output_final.json \
  --output_dir ../Stage-4/output \
  --top_k 100
```

---

## Pipeline Steps

### 1. Load DPR Data

- Reads merged JSON file
- Processes each DPR independently

### 2. Compute Metrics

**SQL / Schema-based**
- Coverage
- Complexity

**Embedding-based**
- Diversity
- Surprisal
- Uniqueness

**LLM-based**
- LLM Quality
- Summary Relevance

### 3. Ranking

DPRs are ranked using a weighted combined score based on all computed metrics.

### 4. Output Generation

The pipeline generates:

- `dpr_ranked_results.json`
- `dpr_ranking_summary.txt`
- `metrics_stats.txt`

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
--llm_api_key <key>
```

> If not provided, LLM scores default to `0.5`.

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
    ├── run_eval_v3.py
    └── output/
        ├── dpr_ranked_results.json
        ├── dpr_ranking_summary.txt
        └── metrics_stats.txt
```

---

## Notes

- Execution summary files are excluded from merging
- Input merging is deterministic
- Designed for scalable DPR evaluation
- LLM usage is optional