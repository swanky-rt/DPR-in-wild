#!/usr/bin/env bash
# run_pipeline.sh — Full DPR Discovery Pipeline
#
# Flow (matches flowchart):
#   Stage 2a  BERTopic clustering on table embeddings
#   Stage 2b  Match user queries to clusters via embedding similarity
#   Stage 2c  Generate DPRs from matched clusters + user queries
#   Stage 3   SQL grounding: sub-questions → SQL → execution → summary
#   Stage 4   LLM-as-Judge evaluation + metrics
#
# Stage 1 (table embedding) is run separately — provide its output via EMBEDDINGS.
#
# Usage:
#   bash run_pipeline.sh                   # clustering only (stage 2a)
#   bash run_pipeline.sh --with-generate   # stages 2a + 2b + 2c
#   bash run_pipeline.sh --full            # all stages (2a → 2b → 2c → 3 → 4)
#
# Override any setting via environment variables:
#   EMBEDDINGS=/path/to/embeddings.json bash run_pipeline.sh --full
#   QUERIES_FILE=/path/to/queries.json RUN_TAG=my-run bash run_pipeline.sh --full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
WITH_GENERATE=false
FULL_PIPELINE=false

for arg in "$@"; do
    case "$arg" in
        --with-generate) WITH_GENERATE=true ;;
        --full)          WITH_GENERATE=true; FULL_PIPELINE=true ;;
    esac
done

# ── Inputs ────────────────────────────────────────────────────────────────────
EMBEDDINGS="${EMBEDDINGS:-$SCRIPT_DIR/stage-1/table_embeddings.json}"
TABLES_CLEAN="${TABLES_CLEAN:-$SCRIPT_DIR/stage-1/tables_clean}"
QUERIES_FILE="${QUERIES_FILE:-$SCRIPT_DIR/stage-2/data/user_queries_from_50matched_dprS.json}"

# ── Timestamp — used for run directory AND all inter-stage file names ─────────
TS="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_BASE="${SCRIPT_DIR}/data/runs/${TS}"

# ── Standardized file names (all derived from TS) ────────────────────────────
#
#  Stage 2a outputs (in STAGE2_OUT/):
#    clusters_$TS.json
#    clusters_summary_$TS.json
#    filtered_clusters_$TS.json
#
#  Stage 2b output (in STAGE2_OUT/):
#    query_cluster_matches_$TS.json
#
#  Stage 2c output (in STAGE2_OUT/dprs/):
#    dprs_$TS.jsonl                  ← JSONL fed into stage 3
#    dprs_$TS-structured.json
#
#  Stage 3 output (in STAGE3_OUT/):
#    dprs_$TS_stage3_output.json     ← named automatically from input stem
#    run_manifest.json
#
#  Stage 4 output (in STAGE4_OUT/):
#    dpr_ranked_results.json
#    dpr_ranking_summary.txt
#    metrics_stats.txt

STAGE2_OUT="${OUTPUT_BASE}/stage2"
STAGE2_DPR_DIR="${STAGE2_OUT}/dprs"       # isolated dir so stage-3 sees only the DPR file
STAGE3_OUT="${OUTPUT_BASE}/stage3"
STAGE4_OUT="${OUTPUT_BASE}/stage4"

# Derived file paths
CLUSTERS_JSON="${STAGE2_OUT}/clusters_${TS}.json"
CLUSTERS_SUMMARY="${STAGE2_OUT}/clusters_summary_${TS}.json"
FILTERED_CLUSTERS="${STAGE2_OUT}/filtered_clusters_${TS}.json"
QUERY_MATCHES="${STAGE2_OUT}/query_cluster_matches_${TS}.json"
DPR_STEM="dprs_${TS}"                      # stem used by generate + stage-3 naming
DPR_JSONL="${STAGE2_DPR_DIR}/${DPR_STEM}.jsonl"
STAGE3_OUTPUT="${STAGE3_OUT}/${DPR_STEM}_stage3_output.json"  # deterministic from stem

# ── LLM config ────────────────────────────────────────────────────────────────
LLM_MODEL="${LLM_MODEL:-qwen.qwen3-235b-a22b-2507-v1:0}"
LLM_API_BASE="${LLM_API_BASE:-https://thekeymaker.umass.edu/v1}"
LLM_API_KEY="${LLM_API_KEY:-${THEKEYMAKER_API_KEY:-}}"

# ── BERTopic params ───────────────────────────────────────────────────────────
# Defaults tuned for ~10k tables.
# For small datasets (<200 tables): UMAP_N_NEIGHBORS=5 HDBSCAN_MIN_CLUSTER_SIZE=2 MIN_TOPIC_SIZE=2
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-10}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-5}"
HDBSCAN_EPSILON="${HDBSCAN_EPSILON:-0.0}"
MIN_TOPIC_SIZE="${MIN_TOPIC_SIZE:-5}"
MIN_TABLES="${MIN_TABLES:-2}"
MAX_TABLES="${MAX_TABLES:-30}"

# ── Progress helpers ──────────────────────────────────────────────────────────
HEARTBEAT_PID=""

start_heartbeat() {
    local label="$1"
    (
        local elapsed=0
        while true; do
            sleep 30
            elapsed=$((elapsed + 30))
            echo "  [$(date +%H:%M:%S)] still running: $label (${elapsed}s elapsed)..."
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    if [ -n "$HEARTBEAT_PID" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    fi
}

run_stage() {
    local label="$1"; shift
    local log_file="$1"; shift

    local start
    start=$(date +%s)
    echo ""
    echo "  Started : $(date +%H:%M:%S)"

    start_heartbeat "$label"
    "$@" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    stop_heartbeat

    local elapsed=$(( $(date +%s) - start ))
    if [ $exit_code -eq 0 ]; then
        echo "  Finished: $(date +%H:%M:%S) (${elapsed}s)"
    else
        echo "  FAILED  : $(date +%H:%M:%S) (${elapsed}s) — exit code $exit_code"
        exit $exit_code
    fi
}

trap 'stop_heartbeat; echo ""; echo "Interrupted."; exit 1' INT TERM

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# ── Validate inputs ───────────────────────────────────────────────────────────
if [ ! -f "$EMBEDDINGS" ]; then
    echo "ERROR: Embeddings file not found: $EMBEDDINGS"
    echo "  Set EMBEDDINGS=/path/to/table_embeddings.json"
    exit 1
fi
if [ ! -d "$TABLES_CLEAN" ]; then
    echo "WARNING: tables_clean directory not found: $TABLES_CLEAN"
fi

mkdir -p "$STAGE2_OUT" "$STAGE2_DPR_DIR" "$STAGE3_OUT" "$STAGE4_OUT"

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "DPR Discovery Pipeline"
echo "Started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Run tag     : $TS"
echo "Output      : $OUTPUT_BASE"
echo "Embeddings  : $EMBEDDINGS"
echo "Tables      : $TABLES_CLEAN"
echo "Queries     : $QUERIES_FILE"
echo "LLM model   : $LLM_MODEL"
echo "Stages      : 2a-clustering$([ "$WITH_GENERATE" = true ] && echo " | 2b-query-match | 2c-dpr-gen")$([ "$FULL_PIPELINE" = true ] && echo " | 3-sql | 4-eval")"
echo "============================================================"

TOTAL_START=$(date +%s)

# ── Stage 2a: BERTopic Clustering ────────────────────────────────────────────
echo ""
echo "=== Stage 2a: BERTopic Clustering ==="
run_stage "Stage2a BERTopic" "$STAGE2_OUT/stage2a_clustering_${TS}.log" \
    python "$SCRIPT_DIR/stage-2/src/run_pipeline.py" \
        --input_path "$EMBEDDINGS" \
        --tables_dir "$TABLES_CLEAN" \
        --output_dir "$STAGE2_OUT" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --hdbscan_min_cluster_size "$HDBSCAN_MIN_CLUSTER_SIZE" \
        --hdbscan_epsilon "$HDBSCAN_EPSILON" \
        --min_topic_size "$MIN_TOPIC_SIZE" \
        --min_tables "$MIN_TABLES" \
        --max_tables "$MAX_TABLES" \
        --skip_generate

# Rename cluster outputs to timestamped names
mv "$STAGE2_OUT/clusters.json"          "$CLUSTERS_JSON"
mv "$STAGE2_OUT/clusters_summary.json"  "$CLUSTERS_SUMMARY"
mv "$STAGE2_OUT/filtered_clusters.json" "$FILTERED_CLUSTERS"

echo "  → $CLUSTERS_JSON"
echo "  → $CLUSTERS_SUMMARY"
echo "  → $FILTERED_CLUSTERS"

if [ "$WITH_GENERATE" = false ]; then
    echo ""
    echo "Clustering done. Run with --with-generate or --full to continue."
    echo "Total time: $(( $(date +%s) - TOTAL_START ))s"
    exit 0
fi

# ── Stage 2b: Query-Cluster Matching ─────────────────────────────────────────
echo ""
echo "=== Stage 2b: Query-Cluster Matching ==="
run_stage "Stage2b Query Matching" "$STAGE2_OUT/stage2b_query_match_${TS}.log" \
    python "$SCRIPT_DIR/stage-1/online_query_guided_cluster_retrieval.py" \
        --embeddings_path "$EMBEDDINGS" \
        --tables_clean_dir "$TABLES_CLEAN" \
        --clusters_summary_path "$CLUSTERS_SUMMARY" \
        --queries_file "$QUERIES_FILE" \
        --output_path "$QUERY_MATCHES"

echo "  → $QUERY_MATCHES"

# ── Stage 2c: DPR Generation ─────────────────────────────────────────────────
echo ""
echo "=== Stage 2c: DPR Generation ==="
run_stage "Stage2c DPR Generation" "$STAGE2_OUT/stage2c_dpr_gen_${TS}.log" \
    python "$SCRIPT_DIR/stage-2/src/generate_dprs_for_queries.py" \
        --query_results_path "$QUERY_MATCHES" \
        --tables_clean_dir "$TABLES_CLEAN" \
        --output_dir "$STAGE2_DPR_DIR" \
        --output_name "$DPR_STEM" \
        --model "$LLM_MODEL" \
        --api_base "$LLM_API_BASE" \
        --api_key "$LLM_API_KEY"

echo "  → $DPR_JSONL"
echo "  → ${STAGE2_DPR_DIR}/${DPR_STEM}-structured.json"

if [ "$FULL_PIPELINE" = false ]; then
    echo ""
    echo "DPR generation done. Run with --full to continue to SQL grounding + evaluation."
    echo "Total time: $(( $(date +%s) - TOTAL_START ))s"
    exit 0
fi

# ── Stage 3: SQL Grounding ────────────────────────────────────────────────────
# Input:  $DPR_JSONL  (dprs_$TS.jsonl)
# Output: $STAGE3_OUTPUT  (dprs_${TS}_stage3_output.json)
echo ""
echo "=== Stage 3: SQL Grounding ==="
run_stage "Stage3 SQL Grounding" "$STAGE3_OUT/stage3_sql_${TS}.log" \
    python "$SCRIPT_DIR/stage-3/src/sql_grounding/run_stage3_query_sets.py" \
        --offline-input-dir "$STAGE2_DPR_DIR" \
        --offline-output-dir "$STAGE3_OUT" \
        --online-input-dir "$STAGE2_DPR_DIR" \
        --online-output-dir "$STAGE3_OUT" \
        --tables-meta "$TABLES_CLEAN" \
        --mode offline

echo "  → $STAGE3_OUTPUT"

# ── Stage 4: LLM Evaluation ───────────────────────────────────────────────────
# Input:  $STAGE3_OUTPUT  (deterministic — stem from stage 2c + _stage3_output.json)
# Output: $STAGE4_OUT/dpr_ranked_results.json, metrics_stats.txt, dpr_ranking_summary.txt
echo ""
echo "=== Stage 4: LLM-as-Judge Evaluation ==="
if [ ! -f "$STAGE3_OUTPUT" ]; then
    echo "ERROR: Expected stage-3 output not found: $STAGE3_OUTPUT"
    exit 1
fi

run_stage "Stage4 Evaluation" "$STAGE4_OUT/stage4_eval_${TS}.log" \
    python "$SCRIPT_DIR/Stage-4/run_eval_v3.py" \
        --input "$STAGE3_OUTPUT" \
        --output_dir "$STAGE4_OUT" \
        --llm_model "$LLM_MODEL" \
        --llm_api_base "$LLM_API_BASE" \
        --llm_api_key "$LLM_API_KEY"

echo "  → $STAGE4_OUT/dpr_ranked_results.json"
echo "  → $STAGE4_OUT/metrics_stats.txt"
echo "  → $STAGE4_OUT/dpr_ranking_summary.txt"

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
MINS=$((TOTAL_ELAPSED / 60))
SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Finished    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time  : ${MINS}m ${SECS}s"
echo "Run tag     : $TS"
echo "Results in  : $OUTPUT_BASE"
echo ""
echo "File flow:"
echo "  [2a] $CLUSTERS_JSON"
echo "  [2a] $CLUSTERS_SUMMARY"
echo "  [2a] $FILTERED_CLUSTERS"
echo "  [2b] $QUERY_MATCHES"
echo "  [2c] $DPR_JSONL"
echo "  [3]  $STAGE3_OUTPUT"
echo "  [4]  $STAGE4_OUT/dpr_ranked_results.json"
echo "============================================================"
