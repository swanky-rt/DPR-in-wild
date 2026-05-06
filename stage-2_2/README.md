# DPR Discovery Pipeline

Open-ended Data Product Request (DPR) generation in the wild. Takes pre-computed table embeddings (Stage 1), clusters semantically similar tables, and generates DPRs using an LLM.

## Pipeline

```
table_embeddings.json  (from Stage 1)
        |
  [cluster.py]        BERTopic (UMAP + HDBSCAN) on pre-computed embeddings
        |                   --> clusters.json, clusters_summary.json
  [filter.py]         Drop noise / singletons, split oversized clusters
        |                   --> filtered_clusters.json
  [generate.py]       DSPy ChainOfThought, N variants per cluster with rotating perspectives
        |                   --> dprs-<model>.jsonl
  [experiments/cross_cluster/generate.py]
        |             Pairwise LLM reasoning on most-dissimilar cluster pairs
        |                   --> cross_cluster/cross_cluster_dprs.jsonl
  [run_pipeline.py]   Merges single-cluster + cross-cluster DPRs
        |                   --> dprs-<model>-merged.jsonl
  [evaluate_clusters.py]  Cluster quality metrics on embeddings + cluster assignments
                            --> cluster_quality_report.json
```

## `src/cluster.py`

Clusters tables using BERTopic with pre-computed 384-dim embeddings. UMAP reduces dimensionality, HDBSCAN finds density-based clusters, CountVectorizer extracts topic keywords from descriptions.

Pre-computed embeddings are passed directly to `topic_model.fit_transform(docs, embeddings=embeddings)` — no SentenceTransformer encoding happens here.

**Default parameters (tuned for ~100 tables):**

| Component | Param | Default | Notes |
|---|---|---|---|
| UMAP | `n_neighbors` | 10 | ~sqrt(n_tables) |
| UMAP | `n_components` | 10 | better separation at this scale |
| UMAP | `min_dist` | 0.1 | |
| UMAP | `metric` | cosine | |
| HDBSCAN | `min_cluster_size` | 3 | more granular clusters |
| HDBSCAN | `cluster_selection_method` | leaf | |
| BERTopic | `min_topic_size` | 3 | |
| CountVectorizer | `min_df` | 2 | 1–2 gram keywords |

Tables HDBSCAN cannot confidently assign are labeled topic `-1` (noise) and dropped in the filter step. Silhouette score is computed on the UMAP-reduced space.

**Outputs:** `clusters.json`, `clusters_summary.json`

## `src/filter.py`

Post-processes BERTopic output:
- Drops noise cluster (topic -1)
- Drops clusters with fewer than `--min_tables` tables (default: 2)
- Splits clusters larger than `--max_tables` (default: 30) using KMeans on schema embeddings (title + columns + description via `all-MiniLM-L6-v2`)
- Assigns sequential `dpr_id`s to remaining clusters

**Output:** `filtered_clusters.json`

## `src/generate.py`

Generates multiple DPR variants per cluster using DSPy `ChainOfThought`.

**How it works:**
- Each call receives the cluster's `{title, columns, description}` for all tables plus a rotating analytical perspective
- 5 perspectives cycle across variants: relationships/comparisons, trends over time, decision-making, measurable metrics, narrative/story
- The LLM reasons step-by-step (chain-of-thought) before writing the DPR
- DPRs are constrained to a **single sentence** — no bullet points, numbered lists, or line breaks
- Rate limit retries with backoff — parses wait time directly from Groq error messages
- 20s sleep between successful calls to stay within Groq TPM limits
- Parallel generation via `ThreadPoolExecutor` (`--max_workers`, default 2)
- All LLM inputs, outputs, and reasoning logged to `data/output/dpr_llm_<timestamp>.log`

**Default params:** 3 variants per cluster, temperature 1.0

**Output schema (JSONL):**
```json
{
  "dpr_id": "3_v2",
  "cluster_id": "3",
  "variant": 2,
  "temperature": 1.0,
  "DPR": "...",
  "reasoning": "...",
  "ground_truth": { "table_uids": ["table_a", "table_b"] }
}
```

**Output:** `dprs-<model>.jsonl`

## `src/experiments/cross_cluster/generate.py`

Finds non-obvious cross-domain connections between clusters that embedding similarity cannot capture (e.g. weather + traffic sharing time/location dimensions). Inspired by ClusterLLM (Zhang et al., 2023).

**Pipeline:**
1. Compute cluster centroids from pre-computed embeddings
2. Rank all cluster pairs by cosine distance — most dissimilar first
3. Evaluate top-K pairs (default: 20) with a `CrossClusterCheck` DSPy signature
4. LLM scores each pair 1–5 on relationship strength and identifies a cross-domain question
5. For pairs scoring ≥ 3, generate a `CrossClusterDPR` requiring data from both clusters
6. Checkpoints after every pair — safe to resume after failure

**Output:** `cross_cluster/pair_decisions.json`, `cross_cluster/cross_cluster_dprs.jsonl`

## `src/run_pipeline.py`

Runs cluster → filter → generate → cross-cluster → merge in sequence.

- `--skip_generate`: run clustering + filtering only
- `--skip_cross_cluster`: skip cross-cluster step, output single-cluster DPRs only
- After both generation steps, merges outputs into `dprs-<model>-merged.jsonl`, tagging each record with `"source": "single_cluster"` or `"cross_cluster"`

## `src/evaluate_clusters.py`

Standalone script that computes cluster quality metrics from pre-computed embeddings and BERTopic output. Does not re-run clustering.

**Inputs:**
- `--embeddings_path` — table embeddings JSON
- `--clusters_summary_path` — `clusters_summary.json` from clustering step
- `--output_dir` — where to write `cluster_quality_report.json`

**Metrics computed:**

| Group | Metric | Direction |
|---|---|---|
| Geometric | Silhouette Score (cosine, raw embeddings) | ↑ higher better |
| Geometric | Davies-Bouldin Index | ↓ lower better |
| Geometric | Calinski-Harabasz Index | ↑ higher better |
| Semantic | Mean intra-cluster cosine similarity | ↑ higher better |
| Semantic | Mean inter-cluster centroid cosine distance | ↑ higher better |
| Schema | Mean intra-cluster Jaccard similarity (column names) | ↑ higher = shared columns |
| Health | Noise rate, cluster size stats, singleton/doubleton rate | — |

**Example:**
```bash
python src/evaluate_clusters.py \
  --embeddings_path data/input_2026-04-14/qwen_table_embeddings.json \
  --clusters_summary_path data/output_2026-04-14/clusters_summary.json \
  --output_dir data/output_2026-04-14
```

**Results (100-table dataset):**

| Metric | Qwen | Granite |
|---|---|---|
| Silhouette Score ↑ | **0.429** | 0.314 |
| Intra-cluster Cosine Sim ↑ | 0.713 | **0.787** |
| Inter-cluster Centroid Distance ↑ | **0.615** | 0.350 |
| Schema Jaccard Similarity ↑ | **0.452** | 0.444 |
| Noise Rate ↓ | **5%** | 7% |

Qwen embeddings produce better-separated, more geometrically distinct clusters with less noise. Granite embeddings yield slightly tighter within-cluster groupings but clusters overlap more with each other.

## `src/experiments/no_umap/cluster.py`

Ablation that runs HDBSCAN directly on L2-normalized raw embeddings, skipping UMAP. Uses an `IdentityReducer` as a drop-in UMAP replacement so BERTopic's interface is unchanged. Silhouette score is computed on raw embeddings with cosine metric and is not directly comparable to the UMAP-reduced baseline score.

## `src/run_pipeline.py` — full usage

```bash
python src/run_pipeline.py \
  --input_path data/input_2026-04-14/qwen_table_embeddings.json \
  --output_dir data/output_2026-04-14 \
  --model groq/qwen/qwen3-32b \
  --n_variants 3 \
  --skip_cross_cluster   # omit to also run cross-cluster generation
```

## Output for Downstream Stages

### SQL Layer (Stage 3)
- `dprs-<model>-merged.jsonl` — all DPRs (single-cluster + cross-cluster), tagged with `source`, each with `dpr_id`, `DPR` text, and `ground_truth.table_uids`
- `filtered_clusters.json` — primary cluster file: `dpr_id`, `cluster_key` (BERTopic topic ID), and full table metadata per cluster (`table_id`, `columns`, `description`)
- `clusters_summary.json` — BERTopic topic names and keyword representations, joinable on `cluster_key` → `topic_id`
- `tables_clean/` — full table data with rows for SQL execution

### Metrics + Evaluation (Stage 4)
- `dprs-<model>-merged.jsonl` — all DPRs to score
- `clusters_summary.json` — clustering quality metrics

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys
```

LLM config in `.env`: set `LLM_MODEL` and the corresponding API key. For testing use Groq (free), for final run use the university proxy (`gpt4o`).
