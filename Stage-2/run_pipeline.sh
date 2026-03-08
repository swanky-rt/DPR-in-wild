#!/bin/bash
# Run the full DPR discovery pipeline
#
# Usage:
#   ./run_pipeline.sh                     # Clustering + filtering only
#   ./run_pipeline.sh --with-generate     # Full pipeline with DPR generation
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. Place embeddings JSON in data/input/table_embeddings.json
#   3. (Optional) Place table JSONs in tables_clean/ for extra metadata
#   4. Set LLM env vars in .env (copy from .env.example) if generating DPRs

set -e

INPUT_PATH="data/input/table_embeddings.json"
TABLES_DIR="tables_clean"
OUTPUT_DIR="data/output"

# BERTopic clustering parameters
# For small datasets (<20 tables): n_neighbors=3, min_cluster_size=2, min_topic_size=2
# For large datasets (500+ tables): n_neighbors=5, min_cluster_size=5, min_topic_size=5
UMAP_N_NEIGHBORS=3
UMAP_MIN_DIST=0.1
UMAP_N_COMPONENTS=5
HDBSCAN_MIN_CLUSTER_SIZE=2
HDBSCAN_EPSILON=0.0
MIN_TOPIC_SIZE=2

# Filter parameters
MIN_TABLES=2
MAX_TABLES=30

# Generation parameters
N_VARIANTS=1

if [ "$1" == "--with-generate" ]; then
    python src/run_pipeline.py \
        --input_path "$INPUT_PATH" \
        --tables_dir "$TABLES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_neighbors $UMAP_N_NEIGHBORS \
        --umap_min_dist $UMAP_MIN_DIST \
        --umap_n_components $UMAP_N_COMPONENTS \
        --hdbscan_min_cluster_size $HDBSCAN_MIN_CLUSTER_SIZE \
        --hdbscan_epsilon $HDBSCAN_EPSILON \
        --min_topic_size $MIN_TOPIC_SIZE \
        --min_tables $MIN_TABLES \
        --max_tables $MAX_TABLES \
        --n_variants $N_VARIANTS
else
    python src/run_pipeline.py \
        --input_path "$INPUT_PATH" \
        --tables_dir "$TABLES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_neighbors $UMAP_N_NEIGHBORS \
        --umap_min_dist $UMAP_MIN_DIST \
        --umap_n_components $UMAP_N_COMPONENTS \
        --hdbscan_min_cluster_size $HDBSCAN_MIN_CLUSTER_SIZE \
        --hdbscan_epsilon $HDBSCAN_EPSILON \
        --min_topic_size $MIN_TOPIC_SIZE \
        --min_tables $MIN_TABLES \
        --max_tables $MAX_TABLES \
        --skip_generate
fi
