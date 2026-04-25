"""
run_eval_all_queries_1.py  —  Online evaluation runner (Stage-4)

Reads multiple stage-3 output JSON files (one per query), runs
run_eval_v3.run_pipeline() on each, then does a CROSS-QUERY POST-PASS
to recompute Surprisal, Diversity, and Uniqueness across the full DPR pool
(not per-file), which is the correct behaviour for the online pipeline
where every file may have only one dominant table combination.

Output layout
─────────────
Stage-4/output/<output_dir>/
  <Q1--stem>/
    dpr_ranked_results.json        full ranked output per query
    dpr_ranking_summary.txt        human-readable ranking
    metrics_stats.txt              per-DPR metric table
    metrics_avg_this_query.txt     avg/min/max for this query's DPRs
  <Q2--stem>/  ...
  metrics_avg_all_queries.txt      aggregate across ALL queries
  summary_all_queries.json         structured per-query breakdown

Averaging explained
───────────────────
  metrics_avg_this_query.txt
    → mean of the N DPRs that belong to ONE query file
      e.g. Q1 has 20 DPRs → mean coverage of those 20

  metrics_avg_all_queries.txt
    → mean of ALL DPRs across ALL query files (flat pool)
      e.g. 5 queries × 20 DPRs = 100 DPRs → mean coverage of 100

  summary_all_queries.json  per_file[i].avg_score
    → mean combined_score of the DPRs in that query file

  summary_all_queries.json  overall_avg_combined_score
    → mean combined_score across all files

Surprisal fix
─────────────
  Old: surprisal computed per file (20 DPRs, 1 unique table combo) → always 0
  New: surprisal recomputed AFTER all files are processed, using the
       full DPR pool so different queries' table combos provide signal.
       Combined scores are then recalculated with the updated metrics.

Usage (from Stage-4/):
    python run_eval_all_queries_1.py \
        --input_dir  ../stage-3/data/stage3_outputs/online_with_query \
        --dpr_filename_pattern "Q*--online_stage3_output.json" \
        --output_dir output/online_eval_final \
        --llm_api_key  $LLM_API_KEY \
        --llm_api_base $LLM_API_BASE \
        --llm_model    gpt4o

Optional:
    --queries_file  path to user_queries JSON
                    Format: [{"dpr_id": "q1_1", "user_query": "..."}, ...]
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import re
import json
import glob
import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

# Stage-4/ is the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# Import existing offline pipeline
sys.path.insert(0, str(SCRIPT_DIR))
from run_eval_v3_1 import run_pipeline, compute_surprisal, compute_diversity, compute_uniqueness

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

VALID_LLM_SCORES = {0.0, 0.5, 0.75, 1.0}

# Metrics shown in all output tables — order matters for display
# METRIC_KEYS = [
#     ("Coverage",            "coverage"),
#     ("Complexity",          "complexity"),
#     ("Diversity",           "diversity"),
#     ("Surprisal",           "surprisal"),
#     ("Uniqueness",          "uniqueness"),
#     ("LLM Quality",         "llm_quality"),
#     ("DPR-Summary Rel.",    "summary_relevance"),
#     ("Query-DPR Rel.",      "query_dpr_relevance"),
#     ("Query-Summary Rel.",  "query_summary_relevance"),
# ]

METRIC_KEYS = [
    ("Coverage",            "coverage"),
    ("Complexity",          "complexity"),
    ("Uniqueness",          "uniqueness"),
    ("LLM Quality",         "llm_quality"),
    ("DPR-Summary Rel.",    "summary_relevance"),
    ("Query-DPR Rel.",      "query_dpr_relevance"),
    ("Query-Summary Rel.",  "query_summary_relevance"),
]

# Weights must match run_eval_v3_1.py exactly if run_eval_v3 also recomputes combined_score.
# If run_eval_v3 uses old weights, this file will override combined_score here.
# _W = 0.25 + 0.15 + 0.15 + 0.15 + 0.10 + 0.10 + 0.10 + 0.10 + 0.10
#
# WEIGHTS = {
#     "coverage":                round(0.25 / _W, 4),
#     "complexity":              round(0.15 / _W, 4),
#     "diversity":               round(0.15 / _W, 4),
#     "surprisal":               round(0.15 / _W, 4),
#     "uniqueness":              round(0.10 / _W, 4),
#     "llm_quality":             round(0.10 / _W, 4),
#     "summary_relevance":       round(0.10 / _W, 4),
#     "query_dpr_relevance":     round(0.10 / _W, 4),
#     "query_summary_relevance": round(0.10 / _W, 4),
# }
_W = 0.25 + 0.15 + 0.10 + 0.10 + 0.10 + 0.10 + 0.10

WEIGHTS = {
    "coverage":                round(0.25 / _W, 4),  # 0.2778
    "complexity":              round(0.15 / _W, 4),  # 0.1667
    "uniqueness":              round(0.10 / _W, 4),  # 0.1111
    "llm_quality":             round(0.10 / _W, 4),  # 0.1111
    "summary_relevance":       round(0.10 / _W, 4),  # 0.1111
    "query_dpr_relevance":     round(0.10 / _W, 4),  # 0.1111
    "query_summary_relevance": round(0.10 / _W, 4),  # 0.1111
}

QUERY_SUMMARY_PROMPT = """You are an expert evaluator assessing how well a generated summary addresses a user's original query.

User Query:
"{user_query}"

Generated Summary:
"{summary_text}"

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
"{user_query}"

Data Product Request (DPR):
"{dpr_text}"

Score how well the DPR aligns with the user's analytical intent:
  0    = DPR is completely unrelated to the query
  0.5  = DPR partially aligns but misses key aspects of the query
  0.75 = DPR mostly aligns with minor gaps
  1.0  = DPR directly and completely addresses the query intent

Respond ONLY with valid JSON, no extra text:
{{"query_dpr_relevance": <score>, "reasoning": "<one sentence>"}}
"""


def _snap(v):
    """Snap a float to the nearest valid LLM score."""
    v = float(v)
    return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))


def llm_query_summary_relevance(
    user_query,
    summary_text,
    api_key="",
    api_base="https://api.openai.com/v1",
    model="gpt-4",
):
    fallback = {
        "query_summary_relevance": 0.5,
        "query_summary_reasoning": "LLM judge skipped (no API key or query).",
    }
    if not api_key or not _HAS_REQUESTS or not summary_text or not user_query:
        return fallback

    prompt = QUERY_SUMMARY_PROMPT.format(
        user_query=user_query,
        summary_text=summary_text,
    )

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(
            r"```json|```",
            "",
            resp.json()["choices"][0]["message"]["content"].strip(),
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
    user_query,
    dpr_text,
    api_key="",
    api_base="https://api.openai.com/v1",
    model="gpt-4",
):
    fallback = {
        "query_dpr_relevance": 0.5,
        "query_dpr_reasoning": "LLM judge skipped (no API key or query).",
    }
    if not api_key or not _HAS_REQUESTS or not dpr_text or not user_query:
        return fallback

    prompt = QUERY_DPR_PROMPT.format(
        user_query=user_query,
        dpr_text=dpr_text,
    )

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(
            r"```json|```",
            "",
            resp.json()["choices"][0]["message"]["content"].strip(),
        ).strip()
        scores = json.loads(content)
        return {
            "query_dpr_relevance": _snap(scores.get("query_dpr_relevance", 0.5)),
            "query_dpr_reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["query_dpr_reasoning"] = f"LLM judge error: {e}"
        return fallback


def compute_stats(values):
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def write_stats_table(all_metrics, all_scores, path, title):
    """Write a formatted metrics table to a text file."""
    lines = [f"  {title}\n\n"]
    lines.append(f"  {'Metric':<22}  {'Min':>8}  {'Max':>8}  {'Mean':>8}\n")
    lines.append(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    for label, key in METRIC_KEYS:
        s = compute_stats(all_metrics.get(key, []))
        lines.append(f"  {label:<22}  {s['min']:>8.4f}  {s['max']:>8.4f}  {s['mean']:>8.4f}\n")
    lines.append(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    s = compute_stats(all_scores)
    lines.append(f"  {'Combined Score':<22}  {s['min']:>8.4f}  {s['max']:>8.4f}  {s['mean']:>8.4f}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"      Saved → {path}")


def recompute_combined_scores_and_ranks(output):
    """
    Recompute combined_score for every DPR using the current metric values,
    then sort and re-rank the DPRs.
    """
    for r in output["ranked_dprs"]:
        r.setdefault("metrics", {})
        r["combined_score"] = round(
            sum(WEIGHTS[k] * r["metrics"].get(k, 0.0) for k in WEIGHTS),
            4,
        )

    ranked = sorted(
        output["ranked_dprs"],
        key=lambda x: x["combined_score"],
        reverse=True,
    )
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
    output["ranked_dprs"] = ranked
    return output


def recompute_pool_metrics(all_output_data):
    """
    Recompute Surprisal, Diversity and Uniqueness across the FULL pool
    of all DPRs from all queries combined.
    """
    print("\n[POST] Recomputing pool-wide metrics across ALL queries combined...")

    flat_records = []
    for output, out_dir, stem in all_output_data:
        for r in output["ranked_dprs"]:
            flat_records.append({
                "tables": r.get("tables_used") or r.get("tables") or [],
                "dpr_text": r.get("dpr_text", "") or r.get("DPR", ""),
                "_ref": r,
            })

    n = len(flat_records)
    print(f"      Pool size: {n} DPRs across {len(all_output_data)} queries")

    if not flat_records:
        return all_output_data

    new_surprisals = compute_surprisal(flat_records)

    texts = [r["dpr_text"] if r["dpr_text"] else "" for r in flat_records]
    if any(t.strip() for t in texts):
        vect = TfidfVectorizer().fit(texts)
        emb = vect.transform(texts).toarray()
        new_diversity = compute_diversity(emb)
        new_uniqueness = compute_uniqueness(emb, threshold=0.85)
    else:
        new_diversity = [0.0] * len(flat_records)
        new_uniqueness = [1.0] * len(flat_records)

    for i, rec in enumerate(flat_records):
        r = rec["_ref"]
        r.setdefault("metrics", {})
        r["metrics"]["surprisal"] = round(float(new_surprisals[i]), 4)
        r["metrics"]["diversity"] = round(float(new_diversity[i]), 4)
        r["metrics"]["uniqueness"] = float(new_uniqueness[i])

    for output, out_dir, stem in all_output_data:
        recompute_combined_scores_and_ranks(output)

    surp_vals = [r["_ref"]["metrics"]["surprisal"] for r in flat_records]
    unique_combos = len(set(frozenset(r["tables"]) for r in flat_records))
    print(f"      Unique table combos in pool: {unique_combos}")
    print(
        f"      Surprisal  min={min(surp_vals):.4f}  "
        f"max={max(surp_vals):.4f}  "
        f"mean={sum(surp_vals)/len(surp_vals):.4f}"
    )

    return all_output_data


def enrich_with_query_metrics(output, query_lookup, llm_api_key, llm_api_base, llm_model):
    """
    Add query_summary_relevance and query_dpr_relevance to each DPR.
    Requires --queries_file mapping dpr_id → user_query.
    Without it, both default to 0.5 (neutral).
    """
    for r in output["ranked_dprs"]:
        r.setdefault("metrics", {})

        dpr_id = str(r.get("dpr_id", ""))
        user_query = query_lookup.get(dpr_id, "")

        if not user_query:
            r["metrics"]["query_summary_relevance"] = 0.5
            r["metrics"]["query_summary_reasoning"] = "No matching user query found."
            r["metrics"]["query_dpr_relevance"] = 0.5
            r["metrics"]["query_dpr_reasoning"] = "No matching user query found."
            continue

        qs = llm_query_summary_relevance(
            user_query,
            r.get("final_summary", ""),
            api_key=llm_api_key,
            api_base=llm_api_base,
            model=llm_model,
        )
        qd = llm_query_dpr_relevance(
            user_query,
            r.get("dpr_text", "") or r.get("DPR", ""),
            api_key=llm_api_key,
            api_base=llm_api_base,
            model=llm_model,
        )

        r["metrics"].update(qs)
        r["metrics"].update(qd)
        r["user_query"] = user_query

    return output


def build_query_stats(output, stem):
    """
    Compute per-query metric averages from the ranked DPR list.
    """
    q_metrics = {key: [] for _, key in METRIC_KEYS}
    q_scores = []

    for r in output["ranked_dprs"]:
        q_scores.append(r.get("combined_score", 0.0))
        metrics = r.get("metrics", {})
        for _, key in METRIC_KEYS:
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                q_metrics[key].append(float(v))

    summary = {
        "output_folder": stem,
        "num_dprs": len(q_scores),
        "avg_score": compute_stats(q_scores)["mean"],
        "metric_averages": {
            key: compute_stats(q_metrics[key])["mean"]
            for _, key in METRIC_KEYS
        },
    }
    return q_metrics, q_scores, summary


def main():
    parser = argparse.ArgumentParser(
        description="Online eval runner — processes multiple stage-3 output files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing stage-3 output JSON files",
    )
    parser.add_argument(
        "--dpr_filename_pattern",
        type=str,
        default="*.json",
        help='Glob pattern to match stage-3 DPR files (default: "*.json")',
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=None,
        help='Path to user_queries JSON: [{"dpr_id": "q1_1", "user_query": "..."}]',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Max DPRs to rank per query file (default: 100)",
    )
    parser.add_argument("--llm_api_key", type=str, default="")
    parser.add_argument("--llm_api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--llm_model", type=str, default="gpt-4")
    args = parser.parse_args()

    llm_api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    llm_api_base = args.llm_api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    llm_model = args.llm_model or os.getenv("LLM_MODEL", "gpt-4")

    query_lookup = {}
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, encoding="utf-8") as f:
            queries = json.load(f)
        for q in queries:
            query_lookup[str(q["dpr_id"])] = q["user_query"]
        print(f"[INFO] Loaded {len(query_lookup)} user queries from {args.queries_file}")
    else:
        print("[INFO] No --queries_file → query-based metrics will be 0.5 (neutral)")

    pattern = os.path.join(args.input_dir, args.dpr_filename_pattern)
    all_files = sorted(
        f for f in glob.glob(pattern)
        if "execution_summary" not in os.path.basename(f)
    )

    if not all_files:
        print(f"[ERROR] No files matched: {pattern}")
        return

    print(f"[INFO] Found {len(all_files)} stage-3 file(s):")
    for f in all_files:
        print(f"       {f}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_output_data = []

    for idx, filepath in enumerate(all_files, start=1):
        stem = Path(filepath).stem
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"[PASS1] [{idx}/{len(all_files)}] {os.path.basename(filepath)}")
        print(f"        Output → {out_dir}")

        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()

        raw_data = (
            json.loads(content)
            if content.startswith("[")
            else [json.loads(l) for l in content.splitlines() if l.strip()]
        )

        if not raw_data:
            print(f"[WARN] Empty file, skipping: {filepath}")
            continue

        print(f"[INFO] Loaded {len(raw_data)} DPRs")

        output = run_pipeline(
            raw_data=raw_data,
            output_dir=out_dir,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            top_k=args.top_k,
        )

        all_output_data.append((output, out_dir, stem))

    if not all_output_data:
        print("[WARN] No files were processed.")
        return

    all_output_data = recompute_pool_metrics(all_output_data)

    all_metrics = {key: [] for _, key in METRIC_KEYS}
    all_scores = []
    query_summaries = []

    for output, out_dir, stem in all_output_data:
        if query_lookup:
            print(f"\n[INFO] Query metrics for {stem}...")
            output = enrich_with_query_metrics(
                output,
                query_lookup,
                llm_api_key,
                llm_api_base,
                llm_model,
            )

            # Query-based metrics were added after the earlier score calculation,
            # so recompute combined_score and rank again here.
            output = recompute_combined_scores_and_ranks(output)

        ranked_path = os.path.join(out_dir, "dpr_ranked_results.json")
        with open(ranked_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[INFO] Saved → {ranked_path}")

        q_metrics, q_scores, summary = build_query_stats(output, stem)
        summary["file"] = stem + ".json"

        avg_path = os.path.join(out_dir, "metrics_avg_this_query.txt")
        write_stats_table(
            q_metrics,
            q_scores,
            avg_path,
            f"Average metrics for {stem} ({len(q_scores)} DPRs)",
        )

        all_scores.extend(q_scores)
        for _, key in METRIC_KEYS:
            all_metrics[key].extend(q_metrics.get(key, []))

        query_summaries.append(summary)

    agg_stats_path = os.path.join(args.output_dir, "metrics_avg_all_queries.txt")
    write_stats_table(
        all_metrics,
        all_scores,
        agg_stats_path,
        f"Aggregate across {len(all_files)} file(s) ({len(all_scores)} total DPRs)",
    )

    agg_json_path = os.path.join(args.output_dir, "summary_all_queries.json")
    with open(agg_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_files": len(query_summaries),
                "total_dprs": len(all_scores),
                "overall_avg_combined_score": compute_stats(all_scores)["mean"],
                "overall_metric_stats": {
                    key: compute_stats(all_metrics[key])
                    for _, key in METRIC_KEYS
                },
                "per_file": query_summaries,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Aggregate JSON → {agg_json_path}")

    print(f"\n{'='*65}")
    print(f"AGGREGATE SUMMARY — {len(query_summaries)} queries, {len(all_scores)} total DPRs")
    print(f"{'='*65}")
    print(f"\n  How to read avg_score:")
    print(f"  - Per query  : mean combined_score of the N DPRs in that query file")
    print(f"  - Overall    : mean combined_score across all {len(all_scores)} DPRs\n")
    print(f"  {'Query File':<45} {'DPRs':>6}  {'Avg Score':>10}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*10}")
    for qs in query_summaries:
        print(f"  {qs['file']:<45} {qs['num_dprs']:>6}  {qs['avg_score']:>10.4f}")

    print(f"\n  Overall avg combined score : {compute_stats(all_scores)['mean']:.4f}")
    print(f"\n  Metric averages across all DPRs:")
    print(f"  {'Metric':<22}  {'Mean':>8}")
    print(f"  {'-'*22}  {'-'*8}")
    for label, key in METRIC_KEYS:
        mean = compute_stats(all_metrics[key])["mean"]
        print(f"  {label:<22}  {mean:>8.4f}")

    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()