"""
run_eval_all_queries.py  —  Online evaluation runner (Stage-4)

Reads multiple stage-3 output JSON files (one per query), runs
run_eval_v3.run_pipeline() on each, writes per-query eval output to
Stage-4/output/<filename_stem>/, then writes two aggregate files at
Stage-4/output/:
  - metrics_avg_all_queries.txt   (stats across all DPRs)
  - summary_all_queries.json      (structured per-query breakdown)

Additionally computes three query-based relevance metrics (requires
--queries_file from stage-2):
  - Query  → DPR     : how well the DPR aligns with the user query
  - Query  → Summary : how well the final summary addresses the user query
  - DPR    → Summary : how well the summary addresses the DPR (existing)

Usage (from Stage-4/):
    python run_eval_all_queries.py \
        --input_dir ../stage-3/data/stage3 \
        --dpr_filename_pattern "stage3_newoutput_batch*.json" \
        --queries_file ../stage-2/data/user_queries_from_50matched_dprS.json \
        --llm_api_key $LLM_API_KEY \
        --llm_api_base $LLM_API_BASE \
        --llm_model gpt-4

Only --input_dir and --dpr_filename_pattern are required.
--queries_file is optional but needed for query-based metrics.
"""

import os
import sys
import re
import json
import glob
import argparse
from pathlib import Path

# Stage-4/ is the script's directory
SCRIPT_DIR         = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# Import existing offline pipeline — no duplication
sys.path.insert(0, str(SCRIPT_DIR))
from run_eval_v3 import run_pipeline

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

VALID_LLM_SCORES = {0.0, 0.5, 0.75, 1.0}

METRIC_KEYS = [
    ("Coverage",            "coverage"),
    ("Complexity",          "complexity"),
    ("Diversity",           "diversity"),
    ("Surprisal",           "surprisal"),
    ("Uniqueness",          "uniqueness"),
    ("LLM Quality",         "llm_quality"),
    ("DPR-Summary Rel.",    "summary_relevance"),
    ("Query-Summary Rel.",  "query_summary_relevance"),
    ("Query-DPR Rel.",      "query_dpr_relevance"),
]


# ---------------------------------------------------------------------------
# Query-based LLM relevance functions
# ---------------------------------------------------------------------------

QUERY_SUMMARY_PROMPT = """You are an expert evaluator assessing how well a generated summary addresses a user's original query.

User Query:
\"{user_query}\"

Generated Summary:
\"{summary_text}\"

Score how well the summary addresses the user's query:
  0    = Summary is completely unrelated to the query
  0.5  = Summary partially addresses the query but misses key aspects
  0.75 = Summary mostly addresses the query with minor gaps
  1.0  = Summary directly and completely addresses the query

Respond ONLY with valid JSON, no extra text:
{{"query_summary_relevance": <score>, "reasoning": "<one sentence>"}}
"""

QUERY_DPR_PROMPT = """You are an expert evaluator assessing how well a Data Product Request (DPR) aligns with a user's original query.

User Query:
\"{user_query}\"

Data Product Request (DPR):
\"{dpr_text}\"

Score how well the DPR aligns with the user's analytical intent:
  0    = DPR is completely unrelated to the query
  0.5  = DPR partially aligns but misses key aspects of the query
  0.75 = DPR mostly aligns with minor gaps
  1.0  = DPR directly and completely addresses the query intent

Respond ONLY with valid JSON, no extra text:
{{"query_dpr_relevance": <score>, "reasoning": "<one sentence>"}}
"""


def _snap(v):
    v = float(v)
    return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))


def llm_query_summary_relevance(
    user_query: str,
    summary_text: str,
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4",
) -> dict:
    """Score how well the final summary addresses the user query."""
    fallback = {
        "query_summary_relevance": 0.5,
        "query_summary_reasoning": "LLM judge skipped (no API key or query).",
    }
    if not api_key or not _HAS_REQUESTS or not summary_text or not user_query:
        return fallback
    prompt = QUERY_SUMMARY_PROMPT.format(
        user_query=user_query, summary_text=summary_text
    )
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 150, "temperature": 0.0},
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(
            r"```json|```",
            "",
            resp.json()["choices"][0]["message"]["content"].strip()
        ).strip()
        scores = json.loads(content)
        return {
            "query_summary_relevance": _snap(scores.get("query_summary_relevance", 0.5)),
            "query_summary_reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["query_summary_reasoning"] = f"LLM judge error: {e}"
        return fallback


def llm_query_dpr_relevance(
    user_query: str,
    dpr_text: str,
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4",
) -> dict:
    """Score how well the DPR aligns with the user query."""
    fallback = {
        "query_dpr_relevance": 0.5,
        "query_dpr_reasoning": "LLM judge skipped (no API key or query).",
    }
    if not api_key or not _HAS_REQUESTS or not dpr_text or not user_query:
        return fallback
    prompt = QUERY_DPR_PROMPT.format(
        user_query=user_query, dpr_text=dpr_text
    )
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 150, "temperature": 0.0},
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(
            r"```json|```",
            "",
            resp.json()["choices"][0]["message"]["content"].strip()
        ).strip()
        scores = json.loads(content)
        return {
            "query_dpr_relevance": _snap(scores.get("query_dpr_relevance", 0.5)),
            "query_dpr_reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["query_dpr_reasoning"] = f"LLM judge error: {e}"
        return fallback


# ---------------------------------------------------------------------------
# Stats / output helpers
# ---------------------------------------------------------------------------

def compute_stats(values: list) -> dict:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def write_aggregate_stats(all_metrics: dict, all_scores: list, path: str, title: str):
    """Write a metrics_stats.txt-style table."""
    lines = []
    lines.append(f"  {title}\n\n")
    lines.append(f"  {'Metric':<22}  {'Min':>8}  {'Max':>8}  {'Mean':>8}\n")
    lines.append(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    for label, key in METRIC_KEYS:
        s = compute_stats(all_metrics.get(key, []))
        lines.append(f"  {label:<22}  {s['min']:>8.4f}  {s['max']:>8.4f}  {s['mean']:>8.4f}\n")
    lines.append(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    s = compute_stats(all_scores)
    lines.append(f"  {'Combined Score':<22}  {s['min']:>8.4f}  {s['max']:>8.4f}  {s['mean']:>8.4f}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"      Saved → {path}")


def enrich_with_query_metrics(
    output: dict,
    query_lookup: dict,
    llm_api_key: str,
    llm_api_base: str,
    llm_model: str,
) -> dict:
    """
    Add query_summary_relevance and query_dpr_relevance to each ranked DPR
    in the run_pipeline output dict, in-place.
    """
    for r in output["ranked_dprs"]:
        dpr_id     = str(r["dpr_id"])
        user_query = query_lookup.get(dpr_id, "")

        if not user_query:
            r["metrics"]["query_summary_relevance"] = 0.5
            r["metrics"]["query_summary_reasoning"] = "No matching user query found."
            r["metrics"]["query_dpr_relevance"]     = 0.5
            r["metrics"]["query_dpr_reasoning"]     = "No matching user query found."
            continue

        qs = llm_query_summary_relevance(
            user_query, r.get("final_summary", ""),
            api_key=llm_api_key, api_base=llm_api_base, model=llm_model,
        )
        qd = llm_query_dpr_relevance(
            user_query, r.get("dpr_text", ""),
            api_key=llm_api_key, api_base=llm_api_base, model=llm_model,
        )
        r["metrics"].update(qs)
        r["metrics"].update(qd)
        r["user_query"] = user_query

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Online eval runner — processes multiple stage-3 output files"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Folder containing stage-3 output JSON files"
    )
    parser.add_argument(
        "--dpr_filename_pattern", type=str, default="*.json",
        help="Glob pattern to match stage-3 DPR files, excluding execution summaries "
             "(default: *.json)"
    )
    parser.add_argument(
        "--queries_file", type=str, default=None,
        help="Path to user_queries JSON from stage-2 "
             "(e.g. ../stage-2/data/user_queries_from_50matched_dprS.json). "
             "Required for query-based relevance metrics."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory inside Stage-4 (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--top_k", type=int, default=100,
        help="Max DPRs to rank per file (default: 100)"
    )
    parser.add_argument(
        "--llm_api_key",  type=str, default="",
        help="LLM API key (or set LLM_API_KEY env var)"
    )
    parser.add_argument(
        "--llm_api_base", type=str, default="https://api.openai.com/v1",
        help="LLM API base URL (or set LLM_API_BASE env var)"
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4",
        help="LLM model name (or set LLM_MODEL env var)"
    )
    args = parser.parse_args()

    # ── Resolve LLM credentials from env if not passed via CLI ──────────────
    llm_api_key  = os.getenv("LLM_API_KEY",  args.llm_api_key)
    llm_api_base = os.getenv("LLM_API_BASE", args.llm_api_base)
    llm_model    = os.getenv("LLM_MODEL",    args.llm_model)

    # ── Load user query lookup: dpr_id -> user_query ─────────────────────────
    query_lookup = {}
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, encoding="utf-8") as f:
            queries = json.load(f)
        for q in queries:
            query_lookup[str(q["dpr_id"])] = q["user_query"]
        print(f"[INFO] Loaded {len(query_lookup)} user queries from {args.queries_file}")
    else:
        print("[INFO] No --queries_file provided — query-based metrics will default to 0.5")

    # ── Find matching files, sorted, excluding execution summaries ───────────
    pattern   = os.path.join(args.input_dir, args.dpr_filename_pattern)
    all_files = sorted([
        f for f in glob.glob(pattern)
        if "execution_summary" not in os.path.basename(f)
    ])

    if not all_files:
        print(f"[ERROR] No files matched pattern: {pattern}")
        return

    print(f"[INFO] Found {len(all_files)} stage-3 output file(s):")
    for f in all_files:
        print(f"       {f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Accumulators for cross-query aggregate ───────────────────────────────
    all_metrics:     dict = {key: [] for _, key in METRIC_KEYS}
    all_scores:      list = []
    query_summaries: list = []

    # ── Per-file eval loop ───────────────────────────────────────────────────
    for idx, filepath in enumerate(all_files, start=1):
        stem    = Path(filepath).stem
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"[INFO] [{idx}/{len(all_files)}] Evaluating: {os.path.basename(filepath)}")
        print(f"       Output → {out_dir}")

        # Load — support both JSON array and JSONL
        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            raw_data = json.loads(content)
        else:
            raw_data = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]

        if not raw_data:
            print(f"[WARN] No DPRs found in {filepath}, skipping.")
            continue

        print(f"[INFO] Loaded {len(raw_data)} DPRs")

        # ── Run existing eval pipeline ───────────────────────────────────────
        # Writes dpr_ranked_results.json, dpr_ranking_summary.txt,
        # metrics_stats.txt into out_dir
        output = run_pipeline(
            raw_data=raw_data,
            output_dir=out_dir,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            top_k=args.top_k,
        )

        # ── Enrich with query-based metrics ─────────────────────────────────
        if query_lookup:
            print(f"\n[INFO] Computing query-based relevance metrics...")
            output = enrich_with_query_metrics(
                output, query_lookup, llm_api_key, llm_api_base, llm_model
            )
            # Overwrite dpr_ranked_results.json with enriched version
            enriched_path = os.path.join(out_dir, "dpr_ranked_results.json")
            with open(enriched_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            print(f"      Updated → {enriched_path}")

        # ── Per-file average stats ───────────────────────────────────────────
        q_metrics: dict = {key: [] for _, key in METRIC_KEYS}
        q_scores:  list = []

        for r in output["ranked_dprs"]:
            q_scores.append(r["combined_score"])
            for _, key in METRIC_KEYS:
                v = r["metrics"].get(key)
                if isinstance(v, (int, float)):
                    q_metrics[key].append(float(v))
            # Accumulate into cross-query totals
            all_scores.append(r["combined_score"])
            for _, key in METRIC_KEYS:
                v = r["metrics"].get(key)
                if isinstance(v, (int, float)):
                    all_metrics[key].append(float(v))

        avg_path = os.path.join(out_dir, "metrics_avg_this_query.txt")
        write_aggregate_stats(
            q_metrics, q_scores, avg_path,
            f"Average metrics for {stem} ({len(q_scores)} DPRs)"
        )

        query_summaries.append({
            "file":            os.path.basename(filepath),
            "output_folder":   stem,
            "num_dprs":        len(q_scores),
            "avg_score":       compute_stats(q_scores)["mean"],
            "metric_averages": {
                key: compute_stats(q_metrics[key])["mean"]
                for _, key in METRIC_KEYS
            },
        })

    if not all_scores:
        print("[WARN] No DPRs were evaluated across any file.")
        return

    # ── Cross-query aggregate files ──────────────────────────────────────────
    agg_stats_path = os.path.join(args.output_dir, "metrics_avg_all_queries.txt")
    write_aggregate_stats(
        all_metrics, all_scores, agg_stats_path,
        f"Aggregate across {len(all_files)} file(s) ({len(all_scores)} total DPRs)"
    )

    agg_json_path = os.path.join(args.output_dir, "summary_all_queries.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files":                len(query_summaries),
            "total_dprs":                 len(all_scores),
            "overall_avg_combined_score": compute_stats(all_scores)["mean"],
            "overall_metric_stats": {
                key: compute_stats(all_metrics[key])
                for _, key in METRIC_KEYS
            },
            "per_file": query_summaries,
        }, f, indent=2)
    print(f"\n[INFO] Aggregate JSON → {agg_json_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AGGREGATE SUMMARY — {len(query_summaries)} file(s), {len(all_scores)} total DPRs")
    print(f"{'='*65}")
    print(f"  {'File':<45} {'DPRs':>6}  {'Avg Score':>10}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*10}")
    for qs in query_summaries:
        print(f"  {qs['file']:<45} {qs['num_dprs']:>6}  {qs['avg_score']:>10.4f}")
    print(f"\n  Overall avg combined score: {compute_stats(all_scores)['mean']:.4f}")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()