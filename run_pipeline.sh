#!/bin/bash
set -e

echo "===== ENV SETUP ====="

# Create venv if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
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

# Preferred: use .env file
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "WARNING: .env not found, using inline values"
  export OPENAI_API_KEY="sk-Gl8J7OJ_GYHbWSTW1ZVVcA"
  export OPENAI_API_BASE="https://thekeymaker.umass.edu/v1"
fi

# Model (LiteLLM requires provider prefix)
export LLM_MODEL="openai/gpt4o"

echo "===== STAGE 2: ONLINE DPR GENERATION DRY RUN ====="
python stage-2/src/online_iterative_pipeline.py \
  --queries_file ward/user_queries_top100.txt \
  --clusters_path runs/2026-05-04_13-54-29/stage2/filtered_clusters_2026-05-04_13-54-29.json \
  --user_report_path ward/user_queries_report.json \
  --output_dir stage-2/data/output-online-full \
  --num_queries 100 \
  --target_dprs 50 \
  --max_attempts 200 \
  --sleep_between 2 \
  --temperature 0.0 \
  --api_key "$LLM_API_KEY" \
  --api_base "$LLM_API_BASE" \
  --model "$LLM_MODEL"

echo "===== STAGE 3: SQL + GROUNDING ====="

 python stage-3/src/sql_grounding/run_stage3_query_sets.py \
  --mode online \
  --input-file stage-2/data/output-online-full/online_dprs_all.jsonl \
  --online-output-dir stage-3/data/output-online-full \
  --tables-meta stage-1/tables_clean \
  --llm_api_key "$LLM_API_KEY" \
  --llm_api_base "$LLM_API_BASE" \
  --llm_model "$LLM_MODEL"

echo "===== STAGE 4: EVALUATION + PER-QUERY AGGREGATION ====="

python stage-4/run_eval_all_queries.py \
  --input_dir stage-3/data/output-online-full \
  --dpr_filename_pattern "*_stage3_output.json" \
  --queries_file ward/user_queries_top100.txt \
  --output_dir stage-4/output-online-full \
  --top_k 50 \
  --llm_api_key "$LLM_API_KEY" \
  --llm_api_base "$LLM_API_BASE" \
  --llm_model "gpt4o"

echo "===== DONE ====="