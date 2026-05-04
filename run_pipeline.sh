#!/usr/bin/env bash
# run_pipeline.sh — DPR Discovery Pipeline
#
# Stages:
#   2a   BERTopic clustering on table embeddings
#   2b   Cross-cluster DPR generation (most dissimilar pairs → LLM)
#   health  Cluster health report (runs after 2b)
#   2c   Random cluster DPR generation (100 queries × 50 random clusters)
#   merge   Combine 2b + 2c DPRs into one JSONL
#   3    SQL grounding: DPR → sub-questions → SQL → summary
#   4    LLM-as-Judge evaluation + metrics
#
# Usage:
#   bash run_pipeline.sh                    # stage 2a only
#   bash run_pipeline.sh --with-generate    # 2a → 2b → health → 2c → merge
#   bash run_pipeline.sh --full             # all stages
#
#   # Run a single step in isolation (requires RUN_TAG pointing to existing run):
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2a
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2b
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only health
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 2c
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only merge
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 3
#   RUN_TAG=2026-04-28_02-05-45 bash run_pipeline.sh --only 4
#
# Override inputs via env vars:
#   EMBEDDINGS=/path/to/embeddings.json RUN_TAG=my-run bash run_pipeline.sh --full
#   QUERIES_FILE=/path/to/queries.txt   RUN_TAG=my-run bash run_pipeline.sh --only 2c

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
WITH_GENERATE=false
FULL_PIPELINE=false
ONLY_STEP=""

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --with-generate) WITH_GENERATE=true ;;
        --full)          WITH_GENERATE=true; FULL_PIPELINE=true ;;
        --only)
            i=$((i+1))
            ONLY_STEP="${!i}"
            ;;
    esac
    i=$((i+1))
done

# --only implies we run exactly one step (all guard conditions are bypassed)
if [ -n "$ONLY_STEP" ]; then
    WITH_GENERATE=true
    FULL_PIPELINE=true
fi

# ── Inputs ────────────────────────────────────────────────────────────────────
EMBEDDINGS="${EMBEDDINGS:-$SCRIPT_DIR/stage-1/table_embeddings.json}"
TABLES_CLEAN="${TABLES_CLEAN:-$SCRIPT_DIR/stage-1/tables_clean}"
QUERIES_FILE="${QUERIES_FILE:-$SCRIPT_DIR/ward/user_queries_top100.txt}"

# ── Run tag — shared key for all file names in this run ──────────────────────
TS="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_BASE="${SCRIPT_DIR}/data/runs/${TS}"

# ── All file paths derived from TS (pass RUN_TAG to reuse an existing run) ───
STAGE2_OUT="${OUTPUT_BASE}/stage2"
STAGE2_DPR_DIR="${STAGE2_OUT}/dprs"
STAGE3_OUT="${OUTPUT_BASE}/stage3"
STAGE4_OUT="${OUTPUT_BASE}/stage4"

CLUSTERS_JSON="${STAGE2_OUT}/clusters_${TS}.json"
CLUSTERS_SUMMARY="${STAGE2_OUT}/clusters_summary_${TS}.json"
FILTERED_CLUSTERS="${STAGE2_OUT}/filtered_clusters_${TS}.json"

CROSS_CLUSTER_DIR="${STAGE2_OUT}/cross_cluster"
CLUSTER_HEALTH="${STAGE2_OUT}/cluster_health_${TS}.json"

# 2b and 2c both append into the same file — no separate merge step
DPR_STEM="dprs_${TS}"
DPR_JSONL="${STAGE2_DPR_DIR}/${DPR_STEM}.jsonl"
STAGE3_OUTPUT="${STAGE3_OUT}/${DPR_STEM}_stage3_output.json"

# ── LLM config ────────────────────────────────────────────────────────────────
LLM_MODEL="${LLM_MODEL:-openai/qwen.qwen3-235b-a22b-2507-v1:0}"
LLM_API_BASE="${LLM_API_BASE:-https://thekeymaker.umass.edu/v1}"
LLM_API_KEY="${LLM_API_KEY:-${THEKEYMAKER_API_KEY:-}}"

# ── BERTopic params ───────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.1}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-10}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-5}"
HDBSCAN_EPSILON="${HDBSCAN_EPSILON:-0.0}"
MIN_TOPIC_SIZE="${MIN_TOPIC_SIZE:-5}"
MIN_TABLES="${MIN_TABLES:-2}"
MAX_TABLES="${MAX_TABLES:-30}"

# ── Cross-cluster params ──────────────────────────────────────────────────────
CROSS_CLUSTER_TOP_K="${CROSS_CLUSTER_TOP_K:-20}"
CROSS_CLUSTER_SLEEP="${CROSS_CLUSTER_SLEEP:-20}"

# ── Random DPR params ────────────────────────────────────────────────────────
N_QUERIES="${N_QUERIES:-100}"
N_SAMPLES_PER_QUERY="${N_SAMPLES_PER_QUERY:-50}"
MAX_WORKERS="${MAX_WORKERS:-10}"

# ── Helpers ───────────────────────────────────────────────────────────────────
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

should_run() {
    # Returns 0 (true) if this step should run given flags
    local step="$1"
    [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "$step" ]
}

trap 'stop_heartbeat; echo ""; echo "Interrupted."; exit 1' INT TERM

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# ── Validate inputs (skip for --only steps that don't need embeddings) ────────
if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2a" ]; then
    if [ ! -f "$EMBEDDINGS" ]; then
        echo "ERROR: Embeddings file not found: $EMBEDDINGS"
        echo "  Set EMBEDDINGS=/path/to/table_embeddings.json"
        exit 1
    fi
    if [ ! -d "$TABLES_CLEAN" ]; then
        echo "WARNING: tables_clean directory not found: $TABLES_CLEAN"
    fi
fi

mkdir -p "$STAGE2_OUT" "$STAGE2_DPR_DIR" "$STAGE3_OUT" "$STAGE4_OUT" "$CROSS_CLUSTER_DIR"

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
if [ -n "$ONLY_STEP" ]; then
    echo "Mode        : --only $ONLY_STEP"
elif [ "$FULL_PIPELINE" = true ]; then
    echo "Mode        : --full (2a → 2b → health → 2c → 3 → 4)"
elif [ "$WITH_GENERATE" = true ]; then
    echo "Mode        : --with-generate (2a → 2b → health → 2c)"
else
    echo "Mode        : clustering only (2a)"
fi
echo "============================================================"

TOTAL_START=$(date +%s)

# ── Stage 2a: BERTopic Clustering ────────────────────────────────────────────
if should_run "2a"; then
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

    mv "$STAGE2_OUT/clusters.json"          "$CLUSTERS_JSON"
    mv "$STAGE2_OUT/clusters_summary.json"  "$CLUSTERS_SUMMARY"
    mv "$STAGE2_OUT/filtered_clusters.json" "$FILTERED_CLUSTERS"

    echo "  → $CLUSTERS_JSON"
    echo "  → $CLUSTERS_SUMMARY"
    echo "  → $FILTERED_CLUSTERS"
fi

if [ -z "$ONLY_STEP" ] && [ "$WITH_GENERATE" = false ]; then
    echo ""
    echo "Clustering done. Use --with-generate or --full to continue, or --only <step> to run one step."
    echo "Total time: $(( $(date +%s) - TOTAL_START ))s"
    exit 0
fi

# ── Stage 2b: Cross-Cluster DPR Generation ───────────────────────────────────
if should_run "2b"; then
    echo ""
    echo "=== Stage 2b: Cross-Cluster DPR Generation ==="
    run_stage "Stage2b Cross-Cluster" "$STAGE2_OUT/stage2b_cross_cluster_${TS}.log" \
        python "$SCRIPT_DIR/stage-2/src/experiments/cross_cluster/generate.py" \
            --clusters_path "$CLUSTERS_JSON" \
            --embeddings_path "$EMBEDDINGS" \
            --output_dir "$CROSS_CLUSTER_DIR" \
            --top_k "$CROSS_CLUSTER_TOP_K" \
            --sleep_between "$CROSS_CLUSTER_SLEEP" \
            --model "$LLM_MODEL" \
            --api_base "$LLM_API_BASE" \
            --api_key "$LLM_API_KEY"

    # Append cross-cluster DPRs directly into the shared DPR file
    python - <<PYEOF
import json, os
src = "$CROSS_CLUSTER_DIR/cross_cluster_dprs.jsonl"
dst = "$DPR_JSONL"
os.makedirs(os.path.dirname(dst), exist_ok=True)
if os.path.exists(src):
    with open(src) as f_in, open(dst, "a") as f_out:
        n = 0
        for line in f_in:
            if line.strip():
                r = json.loads(line)
                r["source"] = "cross_cluster"
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"  Appended {n} cross-cluster DPRs → {dst}")
PYEOF
    echo "  → $DPR_JSONL"
fi

# ── Cluster Health Report ─────────────────────────────────────────────────────
if should_run "health"; then
    echo ""
    echo "=== Cluster Health Report ==="
    python "$SCRIPT_DIR/stage-2/src/cluster_health.py" \
        --clusters_path "$CLUSTERS_JSON" \
        --clusters_summary_path "$CLUSTERS_SUMMARY" \
        --filtered_clusters_path "$FILTERED_CLUSTERS" \
        --cross_cluster_dir "$CROSS_CLUSTER_DIR" \
        --output_dir "$STAGE2_OUT" \
        --output_name "cluster_health_${TS}.json"
    echo "  → $CLUSTER_HEALTH"
fi

# ── Stage 2c: Random Cluster DPR Generation ──────────────────────────────────
if should_run "2c"; then
    echo ""
    echo "=== Stage 2c: Random Cluster DPR Generation ==="
    run_stage "Stage2c Random DPR Gen" "$STAGE2_OUT/stage2c_random_dpr_${TS}.log" \
        python "$SCRIPT_DIR/stage-2/src/generate_random_cluster_dprs.py" \
            --filtered_clusters_path "$FILTERED_CLUSTERS" \
            --tables_clean_dir "$TABLES_CLEAN" \
            --queries_file "$QUERIES_FILE" \
            --output_dir "$STAGE2_DPR_DIR" \
            --output_name "$DPR_STEM" \
            --append \
            --n_queries "$N_QUERIES" \
            --n_samples_per_query "$N_SAMPLES_PER_QUERY" \
            --max_workers "$MAX_WORKERS" \
            --model "$LLM_MODEL" \
            --api_base "$LLM_API_BASE" \
            --api_key "$LLM_API_KEY"
    echo "  → $DPR_JSONL"
fi

if [ -z "$ONLY_STEP" ] && [ "$FULL_PIPELINE" = false ]; then
    echo ""
    echo "DPR generation done. Use --full to continue, or --only 3 / --only 4 to run individual stages."
    echo "Total time: $(( $(date +%s) - TOTAL_START ))s"
    exit 0
fi

# ── Stage 3: SQL Grounding ────────────────────────────────────────────────────
if should_run "3"; then
    echo ""
    echo "=== Stage 3: SQL Grounding ==="
    if [ ! -f "$DPR_JSONL" ]; then
        echo "ERROR: Merged DPR file not found: $DPR_JSONL"
        echo "  Run merge step first: RUN_TAG=$TS bash run_pipeline.sh --only merge"
        exit 1
    fi
    run_stage "Stage3 SQL Grounding" "$STAGE3_OUT/stage3_sql_${TS}.log" \
        python "$SCRIPT_DIR/stage-3/src/sql_grounding/run_stage3_query_sets.py" \
            --offline-input-dir "$STAGE2_DPR_DIR" \
            --offline-output-dir "$STAGE3_OUT" \
            --online-input-dir "$STAGE2_DPR_DIR" \
            --online-output-dir "$STAGE3_OUT" \
            --tables-meta "$TABLES_CLEAN" \
            --mode offline
    echo "  → $STAGE3_OUTPUT"
fi

# ── Stage 4: LLM Evaluation ──────────────────────────────────────────────────
if should_run "4"; then
    echo ""
    echo "=== Stage 4: LLM-as-Judge Evaluation ==="
    if [ ! -f "$STAGE3_OUTPUT" ]; then
        echo "ERROR: Stage-3 output not found: $STAGE3_OUTPUT"
        echo "  Run stage 3 first: RUN_TAG=$TS bash run_pipeline.sh --only 3"
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
fi

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
MINS=$((TOTAL_ELAPSED / 60))
SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "DONE"
echo "Finished    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time  : ${MINS}m ${SECS}s"
echo "Run tag     : $TS"
echo "Results in  : $OUTPUT_BASE"
echo ""
echo "File flow:"
echo "  [2a]     $CLUSTERS_JSON"
echo "  [2a]     $CLUSTERS_SUMMARY"
echo "  [2a]     $FILTERED_CLUSTERS"
echo "  [health] $CLUSTER_HEALTH"
echo "  [2b+2c]  $DPR_JSONL"
echo "  [3]      $STAGE3_OUTPUT"
echo "  [4]      $STAGE4_OUT/dpr_ranked_results.json"
echo "============================================================"
