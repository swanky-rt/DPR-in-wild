#!/bin/bash
set -e

echo "===== ENV SETUP ====="

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

echo "===== INSTALL DEPENDENCIES ====="

pip install \
  dspy-ai \
  litellm \
  python-dotenv \
  tqdm \
  scikit-learn \
  requests \
  numpy \
  pandas \
  langgraph

echo "===== LOAD ENV VARIABLES ====="

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env not found. Create .env in repo root."
  exit 1
fi

export LLM_MODEL="gpt4o"

echo "===== STAGE 2: ONLINE DPR GENERATION LOCAL TEST: 1 query × 5 DPRs ====="

python stage-2/src/online_iterative_pipeline.py \
  --queries_file ward/user_queries_top100.txt \
  --clusters_path runs/2026-05-04_13-54-29/stage2/filtered_clusters_2026-05-04_13-54-29.json \
  --user_report_path ward/user_queries_report.json \
  --output_dir stage-2/data/output-online-local-1q-5dprs \
  --num_queries 1 \
  --target_dprs 5 \
  --max_attempts 15 \
  --sleep_between 0 \
  --temperature 0.0 \
  --llm_api_key "$LLM_API_KEY" \
  --llm_api_base "$LLM_API_BASE" \
  --llm_model "$LLM_MODEL"

echo "===== STAGE 3: SQL + GROUNDING ====="

python stage-3/src/sql_grounding/run_stage3_query_sets.py \
  --mode online \
  --input-file stage-2/data/output-online-local-1q-5dprs/online_dprs_all.jsonl \
  --online-output-dir stage-3/data/output-online-local-1q-5dprs \
  --tables-meta stage-1/tables_clean

echo "===== STAGE 4: EVALUATION + PER-QUERY AGGREGATION ====="

python stage-4/run_eval_all_queries.py \
  --input_dir stage-3/data/output-online-local-1q-5dprs \
  --dpr_filename_pattern "*_stage3_output.json" \
  --queries_file ward/user_queries_top100.txt \
  --output_dir stage-4/output-online-local-1q-5dprs \
  --top_k 5 \
  --llm_api_key "$LLM_API_KEY" \
  --llm_api_base "$LLM_API_BASE" \
  --llm_model "$LLM_MODEL"

echo "===== DONE: LOCAL 1 QUERY × 5 DPRs TEST COMPLETE ====="