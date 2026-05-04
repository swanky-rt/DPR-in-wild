#!/bin/bash
#SBATCH --job-name=dpr-pipeline
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/pi_dagarwal_umass_edu/project_18/athulyaanil/dpr-discovery/logs/pipeline_%j.log

REPO="/project/pi_dagarwal_umass_edu/project_18/athulyaanil/dpr-discovery"

source "$REPO/venv/bin/activate"
cd "$REPO"

export LLM_API_KEY="sk-Gl8J7OJ_GYHbWSTW1ZVVcA"
export LLM_API_BASE="https://thekeymaker.umass.edu/v1"
export LLM_MODEL="qwen.qwen3-235b-a22b-2507-v1:0"
export EMBEDDINGS="$REPO/hybridqa_table_Qwen_embeddings.json"

mkdir -p logs

# Run 2a (clustering) + 2c (5k DPRs) in one go
bash run_pipeline.sh --with-generate
