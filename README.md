# DPR Discovery Pipeline

Autonomous discovery of Data Product Requests (DPRs) over a data lake, without predefined QA pairs.
IBM Research collaboration — branch `1-no-query`.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPR DISCOVERY PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════╗
║  PRE-COMPUTED                                                                       ║
║                                                                                     ║
║  10,993 HybridQA tables  ──►  Qwen3-Embedding-8B (4096-dim)  ──►                  ║
║                               hybridqa_table_Qwen_embeddings.json                  ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
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
│  hybridqa_table_Qwen_embeddings.json  (4096-dim, 10,993 tables)             │
│       │                                                                      │
│       ▼                                                                      │
│  UMAP (dim reduction)  ──►  HDBSCAN  ──►  BERTopic                          │
│       │                                                                      │
│       ▼                                                                      │
│  clusters.json          (all clusters, incl. noise topic -1)                │
│  clusters_summary.json  (per-cluster stats)                                 │
│  filtered_clusters.json (noise removed, min/max table size applied)         │
│                                                                              │
│  Heartbeat every 30s · ~10 min on CPU for 10k tables                        │
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
│                                                                              │
│  Heartbeat every 30s                                                         │
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
│  filtered_clusters.json        (535 clusters from 10k tables)               │
│       │                                                                      │
│       ▼  for each query (100 total):                                         │
│         repeat 50 times:                                                     │
│           pick 1 random cluster  (with replacement, duplicates allowed)      │
│           Qwen3-235B: generate DPR for (query, cluster)                      │
│                                                                              │
│  = 100 × 50 = 5,000 DPRs                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  dprs_$TS.jsonl  ◄──── APPENDED here (written to disk after each DPR)      │
│                                                                              │
│  Progress bar (tqdm) · Heartbeat every 30s · 10 parallel workers            │
│  Crash-safe: each DPR flushed to disk immediately                           │
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

LLM (all stages):        Qwen3-235B  ·  https://thekeymaker.umass.edu/v1
Table embeddings:        Qwen3-Embedding-8B  ·  4096-dim  ·  10,993 tables
Clusters (2026-05-04):   535 filtered clusters from 10,993 tables  ·  ~10 min on CPU
```

---

## Running on Unity (HPC)

### 1. Connect and set up

```bash
ssh <netid>@unity.rc.umass.edu

cd /project/pi_dagarwal_umass_edu/project_18/<netid>
git clone https://github.com/dhdhagar/dpr-discovery.git   # first time
# or
cd dpr-discovery && git pull                               # subsequent runs

git checkout 1-no-query
```

### 2. Get the embeddings file (first time only)

The embeddings file is too large for git — scp it from your local machine:

```bash
# On your LOCAL machine:
scp /path/to/hybridqa_table_Qwen_embeddings.json \
    <netid>@unity.rc.umass.edu:/project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery/
```

### 3. Create the virtual environment (first time only)

```bash
cd /project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery

module load python/3.11.7
python -m venv venv
source venv/bin/activate

pip install -r stage-2/requirements.txt
pip install -r stage-3/requirements.txt
pip install -r Stage-4/requirements.txt
```

### 4. Run as a batch job (recommended)

Batch jobs keep running even when you close your laptop. No GPU needed — the LLM is a remote API.

**Step 1 — Run clustering (stage 2a):**

```bash
source venv/bin/activate
export EMBEDDINGS="/project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery/hybridqa_table_Qwen_embeddings.json"
export LLM_API_KEY="your-thekeymaker-api-key"
export LLM_API_BASE="https://thekeymaker.umass.edu/v1"
export LLM_MODEL="qwen.qwen3-235b-a22b-2507-v1:0"

bash run_pipeline.sh
```

Note the `Run tag` printed at the top (e.g. `2026-05-04_13-54-29`). Takes ~10 min.

**Step 2 — Submit DPR generation as a batch job (stage 2c):**

```bash
cat > run_2c.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=dpr-2c
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery/data/runs/<RUN_TAG>/dpr_gen_%j.log

source /project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery/venv/bin/activate
cd /project/pi_dagarwal_umass_edu/project_18/<netid>/dpr-discovery

export LLM_API_KEY="your-thekeymaker-api-key"
export LLM_API_BASE="https://thekeymaker.umass.edu/v1"
export LLM_MODEL="qwen.qwen3-235b-a22b-2507-v1:0"

RUN_TAG=<RUN_TAG> bash run_pipeline.sh --only 2c
EOF

sbatch run_2c.slurm
```

Replace `<RUN_TAG>` with the timestamp from step 1. Close your laptop — it runs independently.

Check job status:
```bash
squeue -u <netid>
```

Check progress in the log:
```bash
tail -f data/runs/<RUN_TAG>/dpr_gen_<jobid>.log
```

### 5. Pipeline commands reference

```bash
# Clustering only (stage 2a) — always run this first
bash run_pipeline.sh

# Clustering + all DPR generation (2a → 2b → health → 2c)
bash run_pipeline.sh --with-generate

# Full pipeline (all stages)
bash run_pipeline.sh --full

# Re-run a single step using an existing run's tag
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only 2a
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only 2b
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only health
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only 2c
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only 3
RUN_TAG=2026-05-04_13-54-29 bash run_pipeline.sh --only 4
```

> `RUN_TAG` is the timestamp printed at the start of each run. All output file paths are derived from it, so any step can be re-run in isolation without rerunning earlier steps.

### 6. Tunable parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS` | `stage-1/table_embeddings.json` | Path to embeddings file |
| `LLM_MODEL` | `qwen.qwen3-235b-a22b-2507-v1:0` | LLM model for all stages |
| `LLM_API_BASE` | `https://thekeymaker.umass.edu/v1` | API endpoint |
| `N_QUERIES` | `100` | Number of user queries for stage 2c |
| `N_SAMPLES_PER_QUERY` | `50` | Random clusters per query (50×100 = 5,000 DPRs) |
| `MAX_WORKERS` | `10` | Parallel LLM calls in stage 2c |
| `CROSS_CLUSTER_TOP_K` | `20` | Top-K dissimilar pairs for stage 2b |

### 7. Output locations

All outputs land in `data/runs/<RUN_TAG>/`:

| File | Stage |
|------|-------|
| `stage2/clusters_$TS.json` | 2a — all BERTopic clusters |
| `stage2/clusters_summary_$TS.json` | 2a — per-cluster stats |
| `stage2/filtered_clusters_$TS.json` | 2a — noise-filtered clusters |
| `stage2/cluster_health_$TS.json` | health check after 2b |
| `stage2/dprs/dprs_$TS.jsonl` | 2b + 2c combined DPRs (written incrementally) |
| `stage3/dprs_$TS_stage3_output.json` | stage 3 SQL-grounded DPRs |
| `stage4/eval_results_$TS.json` | stage 4 evaluation results |

### 8. Progress monitoring

Stage 2c shows a live progress bar and heartbeat:

```
Generating DPRs:  4%|▍  | 216/5000 [11:35<3:32:34, 2.67s/dpr]
[14:09:31] still running: Stage2c Random DPR Gen (90s elapsed)...
```

Each DPR is written to disk immediately — safe to interrupt and resume.
