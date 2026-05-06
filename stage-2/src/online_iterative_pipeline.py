"""
Online Iterative DPR Generation Pipeline (Option A).

Reads a user_queries JSON file and runs the UCB-based iterative DPR
generation loop for each query, writing results to a separate output
folder per query.

Stops when exactly TARGET_DPRS (default: 20) DPRs are successfully
collected per query, or MAX_ATTEMPTS (default: 50) is reached.

Two independent LLMs:

    LLM-A  GenerateDPRWithQuery  (defined here)
    - Sees: selected cluster's table info + user query + history + perspective
    - Generates: one DPR per attempt, oriented toward user query

  LLM-B  ScoreRelevance  (defined here)
    - Sees: user query + current DPR only
    - Does NOT see: cluster info or history
    - Returns: relevance score 1-5

Cluster selection: UCB algorithm from ucb.py (article-exact).

Reused from generate_dprs_for_queries.py (unchanged):
  - setup_llm
  - load_table_metadata
  - build_cluster_info_from_tables
  - VARIANT_PERSPECTIVES

Repo layout assumed:
  dpr-discovery/
    stage-1_1/
      tables_clean/
    stage-2_2/
      src/
        generate_dprs_for_queries.py   <- imported
        ucb.py                         <- imported
        online_iterative_pipeline.py   <- this file
      data/
        output_qwen_emb_v2/
          clusters.json
        user_queries_from_50matched_dprs.json

Run:
  cd stage-2_2
  python src/online_iterative_pipeline.py \
      --queries_file data/user_queries_from_50matched_dprs.json \
      --num_queries 5

All args have sensible defaults — only --queries_file is required.
"""

import os
import sys
import json
import random
import logging
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# stage-2_2/src/
SCRIPT_DIR = Path(__file__).resolve().parent
# stage-2_2/
STAGE2_DIR = SCRIPT_DIR.parent
# dpr-discovery/  (repo root — stage-1_1 and stage-2_2 are siblings here)
REPO_ROOT = STAGE2_DIR.parent

# Add src/ to path so we can import sibling modules
sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Imports from existing files — nothing duplicated
# ---------------------------------------------------------------------------

from generate_dprs_for_queries import (   # noqa: E402
    setup_llm,                      # configures DSPy + LiteLLM
    load_table_metadata,            # loads tables_clean/*.json -> dict
    build_cluster_info_from_tables, # builds [{title, columns, description}]
    VARIANT_PERSPECTIVES,           # 5 rotating analytical angles
)
from ucb import select_cluster, compute_ucb   # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Default paths  (all relative to repo root / stage-2_2)
# ---------------------------------------------------------------------------

DEFAULT_TABLES_CLEAN_DIR = REPO_ROOT / "stage-1_1" / "tables_clean"
DEFAULT_CLUSTERS_PATH    = STAGE2_DIR / "data" / "output_qwen_emb_v2" / "clusters.json"
DEFAULT_OUTPUT_DIR       = STAGE2_DIR / "data" / "output-online-iterative"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = DEFAULT_OUTPUT_DIR / f"online_iterative_{_ts}.log"

logging.basicConfig(
    filename=str(_log_path),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DPRS  = 20   # stop once this many DPRs are successfully collected
MAX_ATTEMPTS = 50   # safety cap — prevents infinite loop on persistent failures

# ---------------------------------------------------------------------------
# DSPy Signatures  (new — not in any existing file)
# ---------------------------------------------------------------------------

class GenerateDPRWithQuery(dspy.Signature):
    """
    You are a Data Product Request Generator.

    A Data Product is a self-contained, reusable, and consumable data asset
    designed to deliver specific value to its users for data-driven use cases.

    You are given:
    - cluster_info: tables in the selected cluster (title, columns, description).
    - user_query: the analytical question the user wants answered. Use this
      to orient the DPR toward the user's intent.
    - history: past successful rounds with DPRs and their relevance scores.
    - perspective: the analytical angle to frame this DPR.
    - previous_dprs_this_cluster: DPRs already generated from this cluster.

    Rules:
    - Write exactly ONE sentence. No bullet points, lists, or line breaks.
    - Ground your DPR in the cluster's table information.
    - Align the DPR with the user_query intent.
    - Use a clear, professional tone suitable for real-world data requests.
    - Make this DPR meaningfully different from all previous DPRs shown.
    """

    cluster_info: list[dict] = dspy.InputField(
        desc="Tables in the selected cluster: list of {title, columns, description}."
    )
    user_query: str = dspy.InputField(
        desc="The user's analytical question. Orient the DPR toward this intent."
    )
    perspective: str = dspy.InputField(
        desc="Analytical angle — use it to frame a DPR different from previous ones."
    )
    history: list[dict] = dspy.InputField(
        desc="Past successful rounds: [{round, cluster_id, dpr, relevance_score}]."
    )
    previous_dprs_this_cluster: list[str] = dspy.InputField(
        desc="DPRs already generated from this cluster. Do not repeat these."
    )

    data_product_request: str = dspy.OutputField(
        desc="A single-sentence DPR grounded in cluster tables and aligned with the user query."
    )


class ScoreRelevance(dspy.Signature):
    """
    You are a relevance judge for data product requests.

    Given a user query and a generated Data Product Request (DPR), score
    how relevant the DPR is to the user's analytical intent.

    Scoring rubric:
      1 - Completely unrelated to the query
      2 - Tangentially related but misses the core intent
      3 - Partially relevant, covers some aspects of the query
      4 - Mostly relevant, addresses the main intent with minor gaps
      5 - Highly relevant, directly addresses the query intent

    Return only the integer score. No explanation needed.
    """

    user_query: str           = dspy.InputField(desc="The user's original query.")
    data_product_request: str = dspy.InputField(desc="The generated DPR to evaluate.")

    relevance_score: int = dspy.OutputField(desc="Integer relevance score 1-5.")


# ---------------------------------------------------------------------------
# Cluster pool builder
# ---------------------------------------------------------------------------

def build_cluster_pool(
    matched_clusters: List[Dict],
    offline_clusters: Dict[str, List[Dict]],
    table_metadata: Dict[str, Dict],
) -> Dict[str, List[str]]:
    """
    Build the pool of clusters available for this query.

    matched_clusters  — synthetic list built from matched_local_table_ids,
                        each entry has cluster_id, topic_id, all_tables_in_cluster
    offline_clusters  — from clusters.json (topic_id -> list of table dicts)
                        each table dict has table_id, title, columns, description
    table_metadata    — from tables_clean/ keyed by table_id (fallback)

    Returns dict: cluster_id (str) -> list of table_id strings.
    Only includes clusters where at least one table exists in table_metadata.
    """
    pool: Dict[str, List[str]] = {}

    for mc in matched_clusters:
        cid      = str(mc["cluster_id"])
        topic_id = str(mc["topic_id"])

        # clusters.json is keyed by topic_id (BERTopic topic number)
        if topic_id in offline_clusters:
            table_ids = [
                t["table_id"] for t in offline_clusters[topic_id]
                if "table_id" in t
            ]
        else:
            # Fall back to all_tables_in_cluster from the retrieval output
            table_ids = mc.get("all_tables_in_cluster", [])

        # Only keep clusters where at least one table has metadata
        resolvable = [tid for tid in table_ids if tid in table_metadata]
        if resolvable:
            pool[cid] = table_ids

    return pool


# ---------------------------------------------------------------------------
# LLM call helpers  (same retry pattern as generate_dprs_for_queries.py)
# ---------------------------------------------------------------------------

def call_generator(
    cot_a: dspy.ChainOfThought,
    cluster_info: List[Dict],
    user_query: str,          # NEW
    perspective: str,
    history: List[Dict],
    previous_dprs_this_cluster: List[str],
    max_retries: int = 5,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            out = cot_a(
                cluster_info=cluster_info,
                user_query=user_query,            # NEW
                perspective=perspective,
                history=history,
                previous_dprs_this_cluster=previous_dprs_this_cluster,
            )
            return out.data_product_request.strip()
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "RateLimitError" in err:
                match = re.search(r"try again in ([\d.]+)s", err)
                wait  = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning("LLM-A rate limit (attempt %d/%d). Waiting %.1fs.",
                                attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError("LLM-A failed after max retries.")


def call_scorer(
    cot_b: dspy.ChainOfThought,
    user_query: str,
    dpr_text: str,
    max_retries: int = 5,
) -> int:
    """Call LLM-B with rate-limit retry. Returns int score 1-5."""
    for attempt in range(1, max_retries + 1):
        try:
            out   = cot_b(user_query=user_query, data_product_request=dpr_text)
            raw   = str(out.relevance_score).strip()
            match = re.search(r"[1-5]", raw)
            return int(match.group()) if match else 3
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "RateLimitError" in err:
                match = re.search(r"try again in ([\d.]+)s", err)
                wait  = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning("LLM-B rate limit (attempt %d/%d). Waiting %.1fs.",
                                attempt, max_retries, wait)
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError("LLM-B failed after max retries.")


# ---------------------------------------------------------------------------
# Core iterative UCB loop
# ---------------------------------------------------------------------------

def run_iterative_generation(
    query_id: str,
    query_text: str,
    cluster_pool: Dict[str, List[str]],
    table_metadata: Dict[str, Dict],
    cot_a: dspy.ChainOfThought,
    cot_b: dspy.ChainOfThought,
    target_dprs: int = TARGET_DPRS,
    max_attempts: int = MAX_ATTEMPTS,
    sleep_between: float = 20.0,
    seed: int = 42,
) -> List[Dict]:
    """
    UCB-based iterative DPR generation loop for one query.

    Runs until target_dprs DPRs are collected or max_attempts is reached.

    Each attempt:
      1. UCB selects cluster
         - Bootstrap: visit each cluster once in order first
         - UCB phase: pick highest UCB score after bootstrap
      2. LLM-A generates DPR
         - Sees: cluster info + history + perspective
         - Does NOT see: user query
      3. LLM-B scores relevance
         - Sees: user query + DPR only
         - Does NOT see: cluster info or history
      4. Both succeeded → append to results, update UCB state + history
         Either failed → skip attempt, UCB state unchanged

    Returns list of result records in generation order.
    """
    rng         = random.Random(seed)
    cluster_ids = list(cluster_pool.keys())

    # UCB state — mirrors article's class attributes exactly
    visit_counts: Dict[str, int]   = {cid: 0   for cid in cluster_ids}
    score_sums:   Dict[str, float] = {cid: 0.0 for cid in cluster_ids}
    total_trials: int = 0

    # Per-cluster DPR list for diversity enforcement (passed to LLM-A)
    dprs_per_cluster: Dict[str, List[str]] = {cid: [] for cid in cluster_ids}

    # History passed to LLM-A — only successful rounds appended
    history:  List[Dict] = []
    results:  List[Dict] = []
    attempts: int        = 0

    logging.info("Query %s | clusters: %s", query_id, cluster_ids)

    print(f"\n{'='*65}")
    print(f"Query {query_id}: {query_text}")
    print(f"Clusters in pool ({len(cluster_ids)}): {cluster_ids}")
    print(f"Target: {target_dprs} DPRs | Safety cap: {max_attempts} attempts")
    print(f"{'='*65}\n")

    while len(results) < target_dprs and attempts < max_attempts:
        attempts += 1
        dpr_num   = len(results) + 1

        # ── Step 1: UCB cluster selection ───────────────────────────────────
        selected_cid = select_cluster(
            cluster_ids, total_trials, visit_counts, score_sums, rng
        )

        phase      = "bootstrap" if total_trials < len(cluster_ids) else "ucb"
        visit_idx  = visit_counts[selected_cid]          # before increment
        perspective = VARIANT_PERSPECTIVES[visit_idx % len(VARIANT_PERSPECTIVES)]

        # UCB snapshot before update — for logging and output record
        ucb_snapshot = {
            cid: round(compute_ucb(cid, total_trials, visit_counts, score_sums), 4)
            for cid in cluster_ids
        }

        # Build cluster_info using existing helper from generate_dprs_for_queries.py
        table_ids_for_cluster = cluster_pool[selected_cid]
        cluster_info = build_cluster_info_from_tables(table_metadata, table_ids_for_cluster)

        prev_dprs = dprs_per_cluster[selected_cid].copy()

        logging.info(
            "Attempt %02d (DPR %d/%d) | phase=%s | trials=%d | "
            "cluster=%s | visit#=%d | perspective=%d",
            attempts, dpr_num, target_dprs, phase, total_trials,
            selected_cid, visit_idx + 1,
            (visit_idx % len(VARIANT_PERSPECTIVES)) + 1,
        )
        logging.info("UCB scores: %s", ucb_snapshot)

        # ── Step 2: LLM-A generates DPR (no query) ──────────────────────────
        try:
            dpr_text = call_generator(
                cot_a,
                cluster_info=cluster_info,
                user_query=query_text,
                perspective=perspective,
                history=history,
                previous_dprs_this_cluster=prev_dprs,
            )
            time.sleep(sleep_between)
            logging.info("LLM-A DPR: %s", dpr_text)
        except Exception as e:
            logging.error("LLM-A failed on attempt %d: %s", attempts, e)
            print(f"  [attempt {attempts}] LLM-A failed, skipping: {e}")
            continue   # UCB state unchanged

        # ── Step 3: LLM-B scores relevance (query + DPR only) ───────────────
        try:
            score = call_scorer(cot_b, user_query=query_text, dpr_text=dpr_text)
            time.sleep(sleep_between)
            logging.info("LLM-B score: %d", score)
        except Exception as e:
            logging.error("LLM-B failed on attempt %d: %s", attempts, e)
            print(f"  [attempt {attempts}] LLM-B failed, skipping: {e}")
            continue   # UCB state unchanged

        # ── Step 4: Update UCB state — only on full success ──────────────────
        visit_counts[selected_cid] += 1
        score_sums[selected_cid]   += score
        total_trials               += 1
        dprs_per_cluster[selected_cid].append(dpr_text)

        avg_this_cluster = score_sums[selected_cid] / visit_counts[selected_cid]

        # ── Step 5: Append to history for LLM-A next round ──────────────────
        history.append({
            "round":           dpr_num,
            "cluster_id":      selected_cid,
            "dpr":             dpr_text,
            "relevance_score": score,
        })

        # ── Build output record ──────────────────────────────────────────────
        record = {
            "query_id":                 query_id,
            "query_text":               query_text,
            "dpr_number":               dpr_num,
            "attempt":                  attempts,
            "phase":                    phase,
            "cluster_id":               selected_cid,
            "visit_number":             visit_counts[selected_cid],
            "total_trials":             total_trials,
            "perspective":              perspective,
            "DPR":                      dpr_text,
            "relevance_score":          score,
            "avg_cluster_score_so_far": round(avg_this_cluster, 4),
            "ucb_scores_before":        ucb_snapshot,
            "all_visit_counts":         dict(visit_counts),
            "ground_truth": {
                "table_uids": table_ids_for_cluster,
            },
        }
        results.append(record)

        print(
            f"[DPR {dpr_num:02d}/{target_dprs} | attempt {attempts} | {phase}] "
            f"cluster={selected_cid} visit#{visit_counts[selected_cid]} "
            f"score={score} | {dpr_text[:75]}..."
        )

    # Safety cap warning
    if len(results) < target_dprs:
        logging.warning(
            "Safety cap hit: collected %d/%d DPRs after %d attempts.",
            len(results), target_dprs, attempts,
        )
        print(
            f"\n[WARN] Safety cap ({max_attempts} attempts) reached. "
            f"Collected {len(results)}/{target_dprs} DPRs."
        )
    else:
        print(f"\n[OK] Collected {len(results)} DPRs in {attempts} attempts.")

    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(
    query_id: str,
    query_text: str,
    results: List[Dict],
    cluster_pool: Dict[str, List[str]],
) -> Dict:
    """Build structured summary for one query's run."""
    cluster_ids = list(cluster_pool.keys())

    score_lists: Dict[str, List[int]] = {cid: [] for cid in cluster_ids}
    for r in results:
        score_lists[r["cluster_id"]].append(r["relevance_score"])

    cluster_stats = {
        cid: {
            "visits":    len(score_lists[cid]),
            "avg_score": (
                round(sum(score_lists[cid]) / len(score_lists[cid]), 4)
                if score_lists[cid] else None
            ),
            "scores": score_lists[cid],
        }
        for cid in cluster_ids
    }

    overall_avg = (
        round(sum(r["relevance_score"] for r in results) / len(results), 4)
        if results else 0.0
    )

    return {
        "query_id":              query_id,
        "query_text":            query_text,
        "total_dprs_collected":  len(results),
        "overall_avg_relevance": overall_avg,
        "cluster_stats":         cluster_stats,
        "dprs":                  results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online iterative DPR generation with UCB exploration"
    )
    parser.add_argument(
        "--queries_file", type=str, required=True,
        help="Path to user_queries JSON file"
    )
    parser.add_argument(
        "--num_queries", type=int, default=5,
        help="Number of queries to process from the file (default: 5)"
    )
    parser.add_argument(
        "--tables_clean_dir", type=str,
        default=str(DEFAULT_TABLES_CLEAN_DIR),
        help="Path to tables_clean/ directory (stage-1_1 output)"
    )
    parser.add_argument(
        "--clusters_path", type=str,
        default=str(DEFAULT_CLUSTERS_PATH),
        help="Path to clusters.json from stage-2_2 offline clustering"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base directory to write per-query output folders"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LiteLLM model string (or set LLM_MODEL env var)"
    )
    parser.add_argument(
        "--api_base", type=str, default=None,
        help="API base URL (or set LLM_API_BASE env var)"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="API key (or set OPENAI_API_KEY / GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--target_dprs", type=int, default=TARGET_DPRS,
        help="Number of DPRs to collect before stopping (default: 20)"
    )
    parser.add_argument(
        "--max_attempts", type=int, default=MAX_ATTEMPTS,
        help="Safety cap on total attempts (default: 50)"
    )
    parser.add_argument(
        "--sleep_between", type=float, default=20.0,
        help="Seconds to sleep between LLM calls (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for UCB tie-breaking (default: 42)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load user queries file ──────────────────────────────────────────────
    print(f"[INFO] Loading queries: {args.queries_file}")
    with open(args.queries_file, encoding="utf-8") as f:
        all_queries = json.load(f)

    selected = all_queries[:args.num_queries]
    print(f"[INFO] Will process {len(selected)} queries")

    # ── Load table metadata and clusters once (shared across all queries) ───
    print(f"[INFO] Loading table metadata: {args.tables_clean_dir}")
    table_metadata = load_table_metadata(args.tables_clean_dir)
    print(f"[INFO] Loaded {len(table_metadata)} tables")

    print(f"[INFO] Loading offline clusters: {args.clusters_path}")
    with open(args.clusters_path, encoding="utf-8") as f:
        raw_clusters = json.load(f)
    offline_clusters: Dict[str, List[Dict]] = {
        str(k): v for k, v in raw_clusters.items() if str(k) != "-1"
    }
    print(f"[INFO] Loaded {len(offline_clusters)} offline clusters (noise excluded)")

    # ── Build reverse map: table_id -> topic_id ─────────────────────────────
    tid_to_topic: Dict[str, str] = {}
    for topic_id, tables in offline_clusters.items():
        for t in tables:
            tid_to_topic[t["table_id"]] = topic_id

    # ── Configure LLM — reuses setup_llm from generate_dprs_for_queries.py ─
    model = args.model or os.getenv("LLM_MODEL", "openai/gpt-4o")
    setup_llm(model, api_base=args.api_base, api_key=args.api_key)
    print(f"[INFO] Model: {model}")

    # Two independent CoT instances — same model, no shared state
    cot_a = dspy.ChainOfThought(GenerateDPRWithQuery, temperature=1.0)  # LLM-A: generator
    cot_b = dspy.ChainOfThought(ScoreRelevance,     temperature=0.0)  # LLM-B: scorer

    # ── Loop over selected queries ──────────────────────────────────────────
    for idx, entry in enumerate(selected, start=1):
        query_id   = entry["dpr_id"]
        query_text = entry["user_query"]
        table_ids  = entry["matched_local_table_ids"]

        print(f"\n[INFO] === Query {idx}/{len(selected)}: {query_id} ===")
        logging.info("Starting query %d/%d: %s", idx, len(selected), query_id)

        # Build matched_clusters from table_ids via reverse map
        seen_topics: Dict[str, List[Dict]] = {}
        for tid in table_ids:
            topic = tid_to_topic.get(tid)
            if topic and topic not in seen_topics:
                seen_topics[topic] = offline_clusters[topic]

        matched_clusters = [
            {
                "cluster_id":             str(i),
                "topic_id":               topic_id,
                "all_tables_in_cluster":  [t["table_id"] for t in tables],
            }
            for i, (topic_id, tables) in enumerate(seen_topics.items())
        ]

        cluster_pool = build_cluster_pool(matched_clusters, offline_clusters, table_metadata)
        print(f"[INFO] Cluster pool for {query_id}: {list(cluster_pool.keys())}")
        logging.info("Cluster pool for %s: %s", query_id, list(cluster_pool.keys()))

        if not cluster_pool:
            print(f"[WARN] No resolvable clusters for {query_id}, skipping.")
            logging.warning("No resolvable clusters for %s, skipping.", query_id)
            continue

        # ── Run the UCB loop ────────────────────────────────────────────────
        results = run_iterative_generation(
            query_id=query_id,
            query_text=query_text,
            cluster_pool=cluster_pool,
            table_metadata=table_metadata,
            cot_a=cot_a,
            cot_b=cot_b,
            target_dprs=args.target_dprs,
            max_attempts=args.max_attempts,
            sleep_between=args.sleep_between,
            seed=args.seed,
        )

        # ── Save outputs ────────────────────────────────────────────────────
        summary     = build_summary(query_id, query_text, results, cluster_pool)
        model_short = model.split("/")[-1] if "/" in model else model
        safe_id     = query_id.replace("/", "_").replace(" ", "_")
        out_dir     = os.path.join(args.output_dir, f"Q{idx}_{safe_id}")
        os.makedirs(out_dir, exist_ok=True)

        jsonl_path = os.path.join(out_dir, f"online_dprs-{model_short}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary_path = os.path.join(out_dir, f"online_dprs-{model_short}-summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── Print final summary ─────────────────────────────────────────────
        print(f"\n{'='*65}")
        print(f"FINAL SUMMARY — {query_id}")
        print(f"{'='*65}")
        print(f"Query:              {query_id} — {query_text}")
        print(f"DPRs collected:     {len(results)}/{args.target_dprs}")
        print(f"Overall avg score:  {summary['overall_avg_relevance']}")
        print(f"\nCluster breakdown:")
        for cid, stats in summary["cluster_stats"].items():
            print(
                f"  Cluster {cid:>3}: "
                f"visits={stats['visits']:>2}  "
                f"avg_score={stats['avg_score']}  "
                f"scores={stats['scores']}"
            )
        print(f"\nAll {len(results)} DPRs (generation order):")
        for r in results:
            print(
                f"  [DPR {r['dpr_number']:02d} | {r['phase']}] "
                f"cluster={r['cluster_id']} "
                f"score={r['relevance_score']} | "
                f"{r['DPR'][:80]}..."
            )
        print(f"\nOutputs:")
        print(f"  JSONL:   {jsonl_path}")
        print(f"  Summary: {summary_path}")
        print(f"  Log:     {_log_path}")

        logging.info(
            "Query %s done: %d DPRs, avg_score=%.4f, saved to %s",
            query_id, len(results), summary["overall_avg_relevance"], out_dir,
        )


if __name__ == "__main__":
    main()