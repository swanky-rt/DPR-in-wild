# Stage-4: DPR Evaluation & Ranking Pipeline

## Overview

Stage-4 is responsible for evaluating and ranking Data Product Requests (DPRs) generated from Stage-3. It consumes a merged DPR output file and computes multiple metrics to assess the quality, diversity, and usefulness of each DPR.

The pipeline produces:
- Ranked DPRs based on a composite score  
- Metric summaries  
- Human-readable evaluation outputs  

---

## Input

### Primary Input
- `stage3_output_final.json`  
  - Created by merging all Stage-3 batch outputs  
  - Each entry contains:
    - DPR text  
    - tables used  
    - sub-query SQLs  
    - summaries  

---

## Pipeline Steps

### 1. Load DPR Data
- Reads the merged JSON file  
- Processes each DPR independently  

---

### 2. Compute Per-DPR Metrics

#### SQL / Schema-based Metrics
- **Coverage**  
  Measures overlap between used tables and ground truth  

- **Complexity**  
  Based on:
  - multi-table usage  
  - joins  
  - aggregations  
  - subqueries  
  - multi-entity references  

---

#### Embedding-based Metrics
- **Diversity**  
  Ensures DPRs are not redundant  

- **Surprisal**  
  Measures novelty of table combinations  

- **Uniqueness**  
  Penalizes near-duplicate DPRs  

---

#### LLM-based Metrics
- **LLM Quality**  
  Evaluates how well-formed and analytical the DPR is  

- **Summary Relevance**  
  Measures how well the generated summary answers the DPR  

---

### 3. Ranking

Each DPR is assigned a combined score using weighted metrics:

- Coverage  
- Complexity  
- Diversity  
- Surprisal  
- Uniqueness  
- LLM Quality  
- Summary Relevance  

Weights are normalized to sum to 1.0.

---

### 4. Output Generation

The pipeline generates the following files:

#### 1. `dpr_ranked_results.json`
- Ranked DPRs  
- Includes:
  - DPR text  
  - metrics  
  - summaries  
  - sub-query details  

#### 2. `dpr_ranking_summary.txt`
- Human-readable output  
- Includes:
  - DPR text  
  - metrics  
  - LLM reasoning  
  - SQL queries  

#### 3. `metrics_stats.txt`
- Aggregate statistics:
  - min / max / mean for each metric  
  - overall score distribution  

---

## How to Run

```bash
python run_eval_v3.py \              
  --input stage3_output_final.json \
  --output_dir output