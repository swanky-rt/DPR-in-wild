# DPR Discovery Pipeline

Autonomous discovery of Data Product Requests (DPRs) over a data lake, without predefined QA pairs.
IBM Research collaboration — branch `1-no-query`.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPR DISCOVERY PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║  PRE-COMPUTED                                                                ║
║                                                                              ║
║  10,993 HybridQA tables  ──►  Qwen3-Embedding-8B  ──►  hybridqa_table_Qwen_embeddings.json  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1  ·  stage-1/layer1_descriptions.py                                 │
│                                                                              │
│  tables_raw/T1..T100.json                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  Qwen3-235B (thekeymaker)                                                    │
│  → description, numeric_cols, categorical_cols, entities                    │
│       │                                                                      │
│       ▼                                                                      │
│  tables_clean/T1..T100.json   +   schema_descriptions.json                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2a  ·  stage-2/src/run_pipeline.py → cluster.py + filter.py          │
│                                                                              │
│  hybridqa_table_Qwen_embeddings.json                                         │
│       │                                                                      │
│       ▼                                                                      │
│  UMAP (dim reduction)  ──►  HDBSCAN  ──►  BERTopic                          │
│       │                                                                      │
│       ▼                                                                      │
│  clusters.json          (all clusters, incl. noise topic -1)                │
│  clusters_summary.json  (per-cluster stats)                                 │
│  filtered_clusters.json (noise removed, min/max table size applied)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2b  ·  stage-2/src/experiments/cross_cluster/generate.py             │
│                                                                              │
│  filtered_clusters.json                                                      │
│       │                                                                      │
│       ▼                                                                      │
│  Compute cluster centroids (mean of table embeddings)                        │
│  Rank all pairs by cosine distance (most dissimilar first)                   │
│       │                                                                      │
│       ▼  for each pair:                                                      │
│  Qwen3-235B: "Can these two clusters be related?" → score 1–5               │
│       ├── score < 3  → skip                                                 │
│       └── score ≥ 3  → generate DPR for the pair                            │
│                │                                                             │
│                ▼                                                             │
│  dprs_$TS.jsonl  ◄──── CREATED here (cross-cluster DPRs)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLUSTER HEALTH REPORT  ·  stage-2/src/cluster_health.py                    │
│                                                                              │
│  clusters.json + filtered_clusters.json + cross_cluster results             │
│       │                                                                      │
│       ▼                                                                      │
│  cluster_health_$TS.json                                                     │
│  (n_clusters, noise %, silhouette score, size distribution, pair scores)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2c  ·  stage-2/src/generate_random_cluster_dprs.py                   │
│                                                                              │
│  ward/user_queries_top100.txt  (100 queries)                                 │
│  filtered_clusters.json                                                      │
│       │                                                                      │
│       ▼  for each query (100 total):                                         │
│         repeat 50 times:                                                     │
│           pick 1 random cluster  (with replacement, duplicates allowed)      │
│           Qwen3-235B: generate DPR for (query, cluster)                      │
│                                                                              │
│  = 100 × 50 = 5,000 DPRs                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  dprs_$TS.jsonl  ◄──── APPENDED here (random DPRs added to same file)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3  ·  stage-3/src/sql_grounding/run_stage3_query_sets.py             │
│                                                                              │
│  dprs_$TS.jsonl                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  For each DPR:                                                               │
│    → identify relevant tables (ground truth)                                 │
│    → generate SQL query grounded to actual table schemas                     │
│    → validate SQL executes correctly                                         │
│       │                                                                      │
│       ▼                                                                      │
│  dprs_$TS_stage3_output.json                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4  ·  Stage-4/run_eval_v3.py                                          │
│                                                                              │
│  dprs_$TS_stage3_output.json                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  Evaluate DPR quality:                                                       │
│    → relevance scoring (TF-IDF + cosine similarity)                          │
│    → SQL validity checks                                                     │
│    → Qwen3-235B LLM judge                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  eval_results_$TS.json                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

LLM (all stages): Qwen3-235B  ·  https://thekeymaker.umass.edu/v1
```

---

## Running on Unity (HPC)

### 1. Connect and set up

```bash
ssh <your-netid>@unity.rc.umass.edu

# Clone / pull the repo
cd /work/<your-netid>
git clone <repo-url> dpr-discovery   # first time
# or
cd dpr-discovery && git pull          # subsequent runs

git checkout 1-no-query
```

### 2. Create the virtual environment (first time only)

```bash
cd /work/<your-netid>/dpr-discovery

module load python/3.10

python -m venv venv
source venv/bin/activate

pip install -r stage-2/requirements.txt
pip install -r stage-3/requirements.txt
pip install -r Stage-4/requirements.txt
```

### 3. Set environment variables

```bash
export LLM_API_KEY="your-thekeymaker-api-key"
export LLM_API_BASE="https://thekeymaker.umass.edu/v1"
export LLM_MODEL="qwen.qwen3-235b-a22b-2507-v1:0"

# Qwen3-Embedding-8B embeddings (4096-dim, 10,993 tables) — scp from stage-1 branch
export EMBEDDINGS="/project/pi_dagarwal_umass_edu/project_18/athulyaanil/dpr-discovery/hybridqa_table_Qwen_embeddings.json"
```

> No GPU needed — embeddings are pre-computed and the LLM runs as a remote API.
> A CPU node is sufficient.

### 4. Request a CPU node

```bash
srun --partition=cpu \
     --ntasks=1 \
     --cpus-per-task=4 \
     --mem=32G \
     --time=08:00:00 \
     --pty bash
```

Then activate the environment in the interactive session:

```bash
source /work/<your-netid>/dpr-discovery/venv/bin/activate
cd /work/<your-netid>/dpr-discovery
```

### 5. Run the pipeline

```bash
# Full pipeline (all stages)
bash run_pipeline.sh --full

# Clustering only (stage 2a)
bash run_pipeline.sh

# Clustering + DPR generation (2a → 2b → health → 2c)
bash run_pipeline.sh --with-generate

# Re-run a single step from an existing run
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2a
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2b
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only health
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2c
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 3
RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 4
```

> `RUN_TAG` is the timestamp printed at the start of a run (e.g. `2026-04-28_02-05-45`).
> All output file paths are derived from it, so any step can be re-run in isolation.

### 6. Output locations

All outputs land in `output/<RUN_TAG>/`:

| File | Stage |
|------|-------|
| `stage2/clusters_$TS.json` | 2a — all BERTopic clusters |
| `stage2/filtered_clusters_$TS.json` | 2a — noise-filtered clusters |
| `stage2/cluster_health_$TS.json` | health check |
| `stage2/dprs/$TS.jsonl` | 2b + 2c combined DPRs |
| `stage3/dprs_$TS_stage3_output.json` | stage 3 SQL-grounded DPRs |
| `stage4/eval_results_$TS.json` | stage 4 evaluation results |

### 7. Submitting as a batch job (optional)

Create `run_pipeline.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=dpr-pipeline
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_%j.log

source /work/<your-netid>/dpr-discovery/venv/bin/activate
cd /work/<your-netid>/dpr-discovery

export LLM_API_KEY="your-thekeymaker-api-key"
export LLM_API_BASE="https://thekeymaker.umass.edu/v1"
export LLM_MODEL="qwen.qwen3-235b-a22b-2507-v1:0"
export EMBEDDINGS="/project/pi_dagarwal_umass_edu/project_18/athulyaanil/dpr-discovery/hybridqa_table_Qwen_embeddings.json"

bash run_pipeline.sh --full
```

Submit with:

```bash
mkdir -p logs
sbatch run_pipeline.slurm
```
