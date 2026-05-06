"""
Online Iterative DPR Generation Pipeline.

Purpose
-------
This file should NOT define a separate DPR-generation prompt/signature.
It reuses generate_dprs_for_queries.py as the single source of truth for
query-based DPR generation, and only adds:

1. UCB cluster selection
2. LLM-B relevance feedback
3. Per-query iterative online loop
4. Consistent output schema for Stage 3 and Stage 4

Architecture
------------
- UCB cluster selection is query-blind.
- DPR generation sees the selected cluster plus the user query.
- DPR generation is delegated to generate_dprs_for_queries.generate_dpr_for_query_cluster().
- LLM-B relevance scoring sees only user_query + generated DPR.
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
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent          # stage-2/src
STAGE2_DIR = SCRIPT_DIR.parent                        # stage-2
REPO_ROOT = STAGE2_DIR.parent                         # repo root

sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Single source of truth for query-based DPR generation
# ---------------------------------------------------------------------------

from generate_dprs_for_queries import (  # noqa: E402
    setup_llm,
    load_table_metadata,
    build_cluster_info_from_tables,
    generate_dpr_for_query_cluster,
    VARIANT_PERSPECTIVES,
)

# ---------------------------------------------------------------------------
# UCB import with local fallback
# ---------------------------------------------------------------------------

try:
    from ucb import select_cluster, compute_ucb  # noqa: E402
except Exception as _ucb_import_error:
    import math

    def compute_ucb(cluster_id, total_trials, visit_counts, score_sums):
        n = visit_counts.get(cluster_id, 0)
        if n == 0:
            return float("inf")
        avg = score_sums.get(cluster_id, 0.0) / n
        bonus = math.sqrt(2 * math.log(total_trials + 1) / n)
        return avg + bonus

    def select_cluster(cluster_ids, total_trials, visit_counts, score_sums, rng):
        if not cluster_ids:
            raise ValueError("cluster_ids is empty; cannot run UCB selection")

        # Bootstrap: visit each cluster once, query-blind.
        unvisited = [cid for cid in cluster_ids if visit_counts.get(cid, 0) == 0]
        if unvisited:
            return unvisited[0]

        scores = {
            cid: compute_ucb(cid, total_trials, visit_counts, score_sums)
            for cid in cluster_ids
        }
        max_score = max(scores.values())
        best = [cid for cid, score in scores.items() if score == max_score]
        return rng.choice(best)

    logging.warning(
        "Could not import ucb.py (%s); using local UCB fallback.",
        _ucb_import_error,
    )

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TABLES_CLEAN_DIR = REPO_ROOT / "stage-1" / "tables_clean"
DEFAULT_CLUSTERS_PATH = REPO_ROOT / "ward" / "filtered_clusters.json"
DEFAULT_QUERIES_FILE = REPO_ROOT / "ward" / "user_queries_top100.txt"
DEFAULT_USER_REPORT_PATH = REPO_ROOT / "ward" / "user_queries_report.json"
DEFAULT_OUTPUT_DIR = STAGE2_DIR / "data" / "output-online-iterative"

DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = DEFAULT_OUTPUT_DIR / f"online_iterative_{_ts}.log"

logging.basicConfig(
    filename=str(_log_path),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

TARGET_DPRS = 20
MAX_ATTEMPTS = 50

# ---------------------------------------------------------------------------
# LLM-B relevance scorer
# ---------------------------------------------------------------------------

class ScoreRelevance(dspy.Signature):
    """
    You are a relevance judge for data product requests.

    Given a user query and a generated Data Product Request, score how relevant
    the DPR is to the user's analytical intent.

    Scoring rubric:
      1 - Completely unrelated to the query
      2 - Tangentially related but misses the core intent
      3 - Partially relevant, covers some aspects of the query
      4 - Mostly relevant, addresses the main intent with minor gaps
      5 - Highly relevant, directly addresses the query intent

    Return only the integer score.
    """

    user_query: str = dspy.InputField(desc="The user's original query.")
    data_product_request: str = dspy.InputField(desc="The generated DPR to evaluate.")
    relevance_score: int = dspy.OutputField(desc="Integer relevance score from 1 to 5.")


def call_scorer(
    cot_b: dspy.ChainOfThought,
    user_query: str,
    dpr_text: str,
    max_retries: int = 5,
) -> int:
    for attempt in range(1, max_retries + 1):
        try:
            out = cot_b(user_query=user_query, data_product_request=dpr_text)
            raw = str(out.relevance_score).strip()
            match = re.search(r"[1-5]", raw)
            return int(match.group()) if match else 3
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "RateLimitError" in err:
                match = re.search(r"try again in ([\d.]+)s", err)
                wait = float(match.group(1)) + 1.0 if match else 15.0
                logging.warning(
                    "LLM-B rate limit on attempt %d/%d. Waiting %.1fs.",
                    attempt,
                    max_retries,
                    wait,
                )
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise

    raise RuntimeError("LLM-B failed after max retries.")


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def load_user_queries(path: str, num_queries: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Supports:
    1. JSON list:
       [{"query_id": "Q1", "query_text": "..."}]
    2. JSON dict:
       {"query_results": [{"query_id": "Q1", "query_text": "..."}]}
    3. TXT:
       one query per line, query_id generated as Q1, Q2, ...
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"queries_file not found: {path}")

    text = p.read_text(encoding="utf-8").strip()
    queries: List[Dict[str, str]] = []

    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "query_results" in data:
            data = data["query_results"]
        if isinstance(data, dict):
            # Accept {"Q1": "...", "Q2": "..."}.
            data = [
                {"query_id": str(k), "query_text": str(v)}
                for k, v in data.items()
            ]
        if not isinstance(data, list):
            raise ValueError("JSON queries_file must be a list, dict, or contain query_results list.")

        for idx, item in enumerate(data, start=1):
            if isinstance(item, str):
                qid = f"Q{idx}"
                qtext = item
            else:
                qid = str(item.get("query_id") or item.get("dpr_id") or f"Q{idx}")
                qtext = str(item.get("query_text") or item.get("user_query") or item.get("query") or "")
            if qtext.strip():
                queries.append({"query_id": qid, "query_text": qtext.strip()})
    else:
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Accept "Q1<TAB>query", "Q1|query", or plain line.
            qid = f"Q{len(queries) + 1}"
            qtext = line
            if "\t" in line:
                left, right = line.split("\t", 1)
                if right.strip():
                    qid, qtext = left.strip(), right.strip()
            elif "|" in line and line.split("|", 1)[0].strip().upper().startswith("Q"):
                left, right = line.split("|", 1)
                if right.strip():
                    qid, qtext = left.strip(), right.strip()

            queries.append({"query_id": qid, "query_text": qtext})

    if num_queries is not None and num_queries > 0:
        queries = queries[:num_queries]

    if not queries:
        raise ValueError(f"No valid queries found in {path}")

    return queries


def load_user_report(path: Optional[str]) -> Dict[str, Any]:
    """
    Loads ward/user_queries_report.json when available.

    Returns a lookup dictionary keyed by query_id where possible.
    Also preserves raw content under _raw if the structure is not query-id keyed.
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        logging.warning("user_report_path does not exist: %s", path)
        return {}

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    lookup: Dict[str, Any] = {}

    if isinstance(data, dict):
        # Already keyed by Q1/Q2/etc.
        for k, v in data.items():
            lookup[str(k)] = v

        # Common list inside dict.
        for list_key in ("query_results", "queries", "results", "reports"):
            if isinstance(data.get(list_key), list):
                for idx, item in enumerate(data[list_key], start=1):
                    if isinstance(item, dict):
                        qid = str(item.get("query_id") or item.get("dpr_id") or f"Q{idx}")
                        lookup[qid] = item
                break

        lookup.setdefault("_raw", data)

    elif isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                qid = str(item.get("query_id") or item.get("dpr_id") or f"Q{idx}")
                lookup[qid] = item
        lookup["_raw"] = data

    return lookup


def load_clusters(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"clusters_path not found: {path}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_cluster_pool(
    clusters_data: Any,
    table_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Accepts common filtered_clusters shapes and converts them to:
      cluster_id -> [table_id, ...]

    Supported shapes:
    1. [{"dpr_id": "1", "tables": [{"table_id": "..."}]}]
    2. [{"cluster_id": "1", "all_tables_in_cluster": ["T1", "T2"]}]
    3. {"1": [{"table_id": "T1"}, {"table_id": "T2"}]}
    4. {"1": ["T1", "T2"]}
    """
    pool: Dict[str, List[str]] = {}

    if isinstance(clusters_data, list):
        for idx, cluster in enumerate(clusters_data, start=1):
            if not isinstance(cluster, dict):
                continue

            cid = str(
                cluster.get("cluster_id")
                or cluster.get("dpr_id")
                or cluster.get("topic_id")
                or idx
            )

            table_ids: List[str] = []

            if isinstance(cluster.get("all_tables_in_cluster"), list):
                table_ids = [str(t) for t in cluster["all_tables_in_cluster"]]

            elif isinstance(cluster.get("table_ids"), list):
                table_ids = [str(t) for t in cluster["table_ids"]]

            elif isinstance(cluster.get("tables"), list):
                for t in cluster["tables"]:
                    if isinstance(t, dict):
                        tid = t.get("table_id") or t.get("id") or t.get("uid")
                        if tid:
                            table_ids.append(str(tid))
                    else:
                        table_ids.append(str(t))

            table_ids = [tid for tid in table_ids if tid in table_metadata]
            if table_ids:
                pool[cid] = table_ids

    elif isinstance(clusters_data, dict):
        for cid_raw, value in clusters_data.items():
            cid = str(cid_raw)
            table_ids: List[str] = []

            if isinstance(value, list):
                for t in value:
                    if isinstance(t, dict):
                        tid = t.get("table_id") or t.get("id") or t.get("uid")
                        if tid:
                            table_ids.append(str(tid))
                    else:
                        table_ids.append(str(t))

            elif isinstance(value, dict):
                if isinstance(value.get("all_tables_in_cluster"), list):
                    table_ids = [str(t) for t in value["all_tables_in_cluster"]]
                elif isinstance(value.get("table_ids"), list):
                    table_ids = [str(t) for t in value["table_ids"]]
                elif isinstance(value.get("tables"), list):
                    for t in value["tables"]:
                        if isinstance(t, dict):
                            tid = t.get("table_id") or t.get("id") or t.get("uid")
                            if tid:
                                table_ids.append(str(tid))
                        else:
                            table_ids.append(str(t))

            table_ids = [tid for tid in table_ids if tid in table_metadata]
            if table_ids:
                pool[cid] = table_ids

    if not pool:
        raise ValueError(
            "No resolvable clusters found. Check filtered_clusters.json table IDs "
            "and stage-1/tables_clean metadata."
        )

    return pool


# ---------------------------------------------------------------------------
# Online loop
# ---------------------------------------------------------------------------

def run_iterative_generation(
    query_id: str,
    query_text: str,
    cluster_pool: Dict[str, List[str]],
    table_metadata: Dict[str, Dict[str, Any]],
    cot_b: dspy.ChainOfThought,
    user_report_for_query: Optional[Any] = None,
    target_dprs: int = TARGET_DPRS,
    max_attempts: int = MAX_ATTEMPTS,
    sleep_between: float = 20.0,
    seed: int = 42,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    UCB-based online loop for one query.

    Important:
    - UCB selection is query-blind.
    - DPR generation is query-aware and delegated to generate_dprs_for_queries.py.
    - UCB state updates only after DPR generation and relevance scoring both succeed.
    """
    rng = random.Random(seed)
    cluster_ids = list(cluster_pool.keys())

    visit_counts: Dict[str, int] = {cid: 0 for cid in cluster_ids}
    score_sums: Dict[str, float] = {cid: 0.0 for cid in cluster_ids}
    total_trials = 0

    results: List[Dict[str, Any]] = []
    attempts = 0

    logging.info("Query %s | clusters=%s", query_id, cluster_ids)

    print(f"\n{'=' * 72}")
    print(f"Query {query_id}: {query_text}")
    print(f"Clusters in pool: {len(cluster_ids)}")
    print(f"Target DPRs: {target_dprs} | Max attempts: {max_attempts}")
    print(f"{'=' * 72}")

    while len(results) < target_dprs and attempts < max_attempts:
        attempts += 1
        dpr_num = len(results) + 1

        selected_cid = select_cluster(
            cluster_ids,
            total_trials,
            visit_counts,
            score_sums,
            rng,
        )

        phase = "bootstrap" if total_trials < len(cluster_ids) else "ucb"
        variant = visit_counts[selected_cid] + 1
        table_ids_for_cluster = cluster_pool[selected_cid]

        ucb_snapshot = {
            cid: round(compute_ucb(cid, total_trials, visit_counts, score_sums), 4)
            for cid in cluster_ids
        }

        logging.info(
            "Attempt %d | query=%s | phase=%s | selected_cluster=%s | variant=%d",
            attempts,
            query_id,
            phase,
            selected_cid,
            variant,
        )
        logging.info("UCB snapshot before generation: %s", ucb_snapshot)

        # -------------------------------------------------------------------
        # DPR generation: same function and prompt as generate_dprs_for_queries.py
        # -------------------------------------------------------------------
        try:
            generated = generate_dpr_for_query_cluster(
                query_id=query_id,
                query_text=query_text,
                cluster_id=selected_cid,
                cluster_tables=table_ids_for_cluster,
                tables_metadata=table_metadata,
                variant=variant,
                temperature=temperature,
            )
            if not generated:
                logging.warning("DPR generation returned None for query=%s cluster=%s", query_id, selected_cid)
                continue

            dpr_text = generated["DPR"]
            dpr_id = generated["dpr_id"]

            if sleep_between > 0:
                time.sleep(sleep_between)

        except Exception as e:
            logging.error("DPR generation failed on attempt %d: %s", attempts, e)
            print(f"[attempt {attempts}] DPR generation failed, skipping: {e}")
            continue

        # -------------------------------------------------------------------
        # Relevance scoring: LLM-B sees only user query + DPR
        # -------------------------------------------------------------------
        try:
            score = call_scorer(cot_b, user_query=query_text, dpr_text=dpr_text)
            if sleep_between > 0:
                time.sleep(sleep_between)
        except Exception as e:
            logging.error("LLM-B scoring failed on attempt %d: %s", attempts, e)
            print(f"[attempt {attempts}] LLM-B failed, skipping: {e}")
            continue

        # -------------------------------------------------------------------
        # UCB update only after successful DPR + score
        # -------------------------------------------------------------------
        visit_counts[selected_cid] += 1
        score_sums[selected_cid] += score
        total_trials += 1

        avg_this_cluster = score_sums[selected_cid] / visit_counts[selected_cid]

        # Keep Stage 3 and Stage 4 handoff fields stable.
        record = {
            **generated,
            "dpr_number": dpr_num,
            "attempt": attempts,
            "phase": phase,
            "visit_number": visit_counts[selected_cid],
            "total_trials": total_trials,
            "relevance_score": score,
            "avg_cluster_score_so_far": round(avg_this_cluster, 4),
            "ucb_scores_before": ucb_snapshot,
            "all_visit_counts": dict(visit_counts),
            "tables": table_ids_for_cluster,
            "ground_truth": {
                "table_uids": table_ids_for_cluster,
            },
            "user_report": user_report_for_query,
        }

        # Defensive normalization in case imported generator changes.
        record["query_id"] = query_id
        record["query_text"] = query_text
        record["cluster_id"] = selected_cid
        record["dpr_id"] = dpr_id
        record["DPR"] = dpr_text

        results.append(record)

        print(
            f"[DPR {dpr_num:02d}/{target_dprs} | attempt {attempts} | {phase}] "
            f"query={query_id} cluster={selected_cid} visit={visit_counts[selected_cid]} "
            f"score={score} dpr_id={dpr_id}"
        )

    if len(results) < target_dprs:
        print(
            f"\n[WARN] Safety cap reached. Collected {len(results)}/{target_dprs} DPRs "
            f"after {attempts} attempts."
        )
    else:
        print(f"\n[OK] Collected {len(results)} DPRs in {attempts} attempts.")

    return results


def build_summary(
    query_id: str,
    query_text: str,
    results: List[Dict[str, Any]],
    cluster_pool: Dict[str, List[str]],
) -> Dict[str, Any]:
    cluster_ids = list(cluster_pool.keys())
    score_lists: Dict[str, List[int]] = {cid: [] for cid in cluster_ids}

    for r in results:
        score_lists[str(r["cluster_id"])].append(int(r["relevance_score"]))

    cluster_stats = {
        cid: {
            "visits": len(score_lists[cid]),
            "avg_score": (
                round(sum(score_lists[cid]) / len(score_lists[cid]), 4)
                if score_lists[cid]
                else None
            ),
            "scores": score_lists[cid],
            "table_ids": cluster_pool[cid],
        }
        for cid in cluster_ids
    }

    overall_avg = (
        round(sum(int(r["relevance_score"]) for r in results) / len(results), 4)
        if results
        else 0.0
    )

    return {
        "query_id": query_id,
        "query_text": query_text,
        "total_dprs_collected": len(results),
        "overall_avg_relevance": overall_avg,
        "cluster_stats": cluster_stats,
        "dprs": results,
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online iterative DPR generation with query-aware DPR generation and query-blind UCB."
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=str(DEFAULT_QUERIES_FILE),
        help="Path to ward/user_queries_top100.txt or JSON query file.",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=5,
        help="Number of queries to process. Use 0 or negative to process all.",
    )
    parser.add_argument(
        "--tables_clean_dir",
        type=str,
        default=str(DEFAULT_TABLES_CLEAN_DIR),
        help="Path to stage-1/tables_clean.",
    )
    parser.add_argument(
        "--clusters_path",
        type=str,
        default=str(DEFAULT_CLUSTERS_PATH),
        help="Path to ward/filtered_clusters.json.",
    )
    parser.add_argument(
        "--user_report_path",
        type=str,
        default=str(DEFAULT_USER_REPORT_PATH),
        help="Path to ward/user_queries_report.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_api_base", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    # Backward-compatible aliases
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)

    parser.add_argument("--target_dprs", type=int, default=TARGET_DPRS)
    parser.add_argument("--max_attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--sleep_between", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_queries = args.num_queries if args.num_queries and args.num_queries > 0 else None

    print(f"[INFO] Loading queries from: {args.queries_file}")
    queries = load_user_queries(args.queries_file, num_queries=num_queries)
    print(f"[INFO] Loaded {len(queries)} query/query-set record(s).")

    print(f"[INFO] Loading user report from: {args.user_report_path}")
    user_report_lookup = load_user_report(args.user_report_path)

    print(f"[INFO] Loading table metadata from: {args.tables_clean_dir}")
    table_metadata = load_table_metadata(args.tables_clean_dir)
    print(f"[INFO] Loaded metadata for {len(table_metadata)} table(s).")

    print(f"[INFO] Loading clusters from: {args.clusters_path}")
    clusters_data = load_clusters(args.clusters_path)
    cluster_pool = normalize_cluster_pool(clusters_data, table_metadata)
    print(f"[INFO] Resolved {len(cluster_pool)} cluster(s).")

    # ------------------------------------------------------------------
    # LLM config
    # ------------------------------------------------------------------
    # Canonical style now matches Stage-3:
    #   LLM_API_KEY, LLM_API_BASE, LLM_MODEL
    #
    # Stage-3 direct OpenAI-compatible client usually wants:
    #   LLM_MODEL=gpt4o
    #
    # Stage-2 uses DSPy/LiteLLM, which needs provider prefix:
    #   openai/gpt4o
    #
    # So if LLM_MODEL=gpt4o, Stage-2 automatically converts it to:
    #   openai/gpt4o
    # ------------------------------------------------------------------

    llm_api_key = (
            args.llm_api_key
            or args.api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or ""
    )

    llm_api_base = (
            args.llm_api_base
            or args.api_base
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or ""
    )

    llm_model_raw = (
            args.llm_model
            or args.model
            or os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt4o"
    )

    # Keep env vars populated for imported helpers, DSPy, and LiteLLM.
    if llm_api_key:
        os.environ["LLM_API_KEY"] = llm_api_key
        os.environ["OPENAI_API_KEY"] = llm_api_key

    if llm_api_base:
        os.environ["LLM_API_BASE"] = llm_api_base
        os.environ["OPENAI_API_BASE"] = llm_api_base
        os.environ["OPENAI_BASE_URL"] = llm_api_base

    os.environ["LLM_MODEL"] = llm_model_raw

    # LiteLLM requires provider prefix.
    # If user gives "gpt4o", convert to "openai/gpt4o".
    # If user already gives "openai/gpt4o", keep it.
    if "/" in llm_model_raw:
        stage2_model = llm_model_raw
    else:
        stage2_model = f"openai/{llm_model_raw}"

    setup_llm(stage2_model, api_base=llm_api_base, api_key=llm_api_key)

    print(f"[INFO] Using Stage-2 LiteLLM model: {stage2_model}")
    print(f"[INFO] LLM_API_BASE: {llm_api_base}")

    cot_b = dspy.ChainOfThought(ScoreRelevance)

    all_records: List[Dict[str, Any]] = []
    all_summaries: Dict[str, Any] = {}

    for q in queries:
        query_id = str(q["query_id"])
        query_text = str(q["query_text"])
        user_report_for_query = user_report_lookup.get(query_id)

        results = run_iterative_generation(
            query_id=query_id,
            query_text=query_text,
            cluster_pool=cluster_pool,
            table_metadata=table_metadata,
            cot_b=cot_b,
            user_report_for_query=user_report_for_query,
            target_dprs=args.target_dprs,
            max_attempts=args.max_attempts,
            sleep_between=args.sleep_between,
            seed=args.seed,
            temperature=args.temperature,
        )

        all_records.extend(results)
        all_summaries[query_id] = build_summary(
            query_id=query_id,
            query_text=query_text,
            results=results,
            cluster_pool=cluster_pool,
        )

        safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)
        per_query_jsonl = output_dir / f"{safe_qid}--online_dprs.jsonl"
        per_query_summary = output_dir / f"{safe_qid}--online_summary.json"

        write_jsonl(per_query_jsonl, results)
        write_json(per_query_summary, all_summaries[query_id])

        print(f"[INFO] Saved per-query DPRs: {per_query_jsonl}")
        print(f"[INFO] Saved per-query summary: {per_query_summary}")

    combined_jsonl = output_dir / "online_dprs_all.jsonl"
    combined_json = output_dir / "online_dprs_all.json"
    summary_json = output_dir / "online_generation_summary.json"

    write_jsonl(combined_jsonl, all_records)
    write_json(combined_json, all_records)
    write_json(
        summary_json,
        {
            "num_queries": len(queries),
            "total_dprs": len(all_records),
            "queries": all_summaries,
            "inputs": {
                "queries_file": args.queries_file,
                "clusters_path": args.clusters_path,
                "tables_clean_dir": args.tables_clean_dir,
                "user_report_path": args.user_report_path,
            },
            "outputs": {
                "combined_jsonl": str(combined_jsonl),
                "combined_json": str(combined_json),
                "summary_json": str(summary_json),
            },
        },
    )

    print(f"\n[OK] Saved combined DPR JSONL: {combined_jsonl}")
    print(f"[OK] Saved combined DPR JSON:  {combined_json}")
    print(f"[OK] Saved summary:            {summary_json}")


if __name__ == "__main__":
    main()
