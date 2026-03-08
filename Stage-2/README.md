# DPR Discovery Pipeline

Open-ended Data Product Request (DPR) generation in the wild. Stage 2 takes the pre-computed embeddings from Stage 1, clusters semantically similar tables, and generates DPRs using an LLM.

## Pipeline

```
table_embeddings.json  (from Stage 1)
        |
  [cluster.py]   BERTopic (UMAP + HDBSCAN) on pre-computed embeddings
        |              --> clusters.json, clusters_summary.json
  [filter.py]    Drop noise / singletons, split oversized clusters
        |              --> filtered_clusters.json
  [generate.py]  DSPy ChainOfThought, 3 variants per cluster, temp=0.3
                       --> dprs-<model>.jsonl
```

## `src/cluster.py`

Clusters tables using BERTopic with pre-computed 384-dim embeddings (all-MiniLM-L6-v2). UMAP reduces dimensionality, HDBSCAN finds density-based clusters, CountVectorizer extracts topic keywords from descriptions.

DPBench clusters on questions concatenated with `[SEP]` tokens. We cluster on table descriptions instead since we have no QA pairs. Embeddings are passed directly to `topic_model.fit_transform(docs, embeddings=embeddings)` — no encoding happens here.

Parameters are tuned for 10 test tables (e.g., `min_cluster_size=2`, `n_neighbors=3`). For the full dataset (500+ tables), these should be increased to DPBench defaults (`min_cluster_size=5`, `n_neighbors=5`).

Computes silhouette score on the UMAP-reduced space for cluster quality evaluation.

## `src/filter.py`

Post-processes BERTopic output:
- Drops noise cluster (topic -1) — tables HDBSCAN couldn't confidently assign
- Drops clusters with fewer than 2 tables — can't form a cross-table DPR from a single table
- Splits clusters larger than 30 tables using KMeans on schema embeddings (title + columns + description)
- Assigns sequential `dpr_id`s to remaining clusters

Same logic as DPBench's `filter.py`.

## `src/generate.py`

Generates multiple DPR variants per cluster using DSPy `ChainOfThought`. The LLM reasons about how the tables relate before writing the DPR. Prompt follows DPBench's `QuestionAbstraction` signature — same structure, same examples — but passes table descriptions instead of questions.

- **3 variants per cluster** by default (configurable via `--n_variants`). Multiple variants increase diversity across the generated DPR set.
- **Temperature 0.3** — enough variation between variants without losing coherence.
- Parallel generation via `ThreadPoolExecutor`.
- Logs all LLM inputs, outputs, and reasoning to `data/output/dpr_llm_<timestamp>.log`.

Output is JSONL (one JSON object per line), matching DPBench's format. Each entry includes `dpr_id`, `cluster_id`, `variant`, `temperature`, `DPR` text, `reasoning`, and `ground_truth.table_uids`.

JSONL allows incremental writes — if the pipeline crashes mid-run, completed DPRs are already saved rather than losing the entire batch.   

## `src/run_pipeline.py`

Runs cluster -> filter -> generate in sequence. Use `--skip_generate` to run clustering only.

## Output for Downstream Stages

### SQL Layer (Stage 3)
- `data/output/dprs-<model>.jsonl` — DPR text + `ground_truth.table_uids` (which tables should answer this DPR)
- `data/output/filtered_clusters.json` — cluster metadata (table titles, columns, descriptions per cluster)
- `tables_clean/` — full table data with rows for SQL execution

### Metrics + Evaluation (Stage 4)
- `data/output/dprs-<model>.jsonl` — DPRs to score
- `data/output/clusters_summary.json` — clustering quality metrics

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys
```

LLM config in `.env`: set `LLM_MODEL` and the corresponding API key. For testing use Groq (free), for final run use the university proxy (`gpt4o`).
