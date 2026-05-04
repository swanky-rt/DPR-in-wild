"""
run_eval_all_queries.py  —  Online evaluation runner (Stage-4)

Architecture
────────────
PASS 1  Per-query eval via run_eval_v3.run_pipeline()
        Computes: Coverage, Complexity, Diversity, Surprisal (per-file),
                  Uniqueness, LLM Quality, DPR-Summary Relevance
        Combined score uses these 7 metrics (weights normalised to 1.0).

# PASS 2  Cross-query pool recomputation
#         Diversity and Uniqueness are recomputed across ALL DPRs so that
#         each DPR's score reflects how different it is from the full pool.
#         Surprisal is NOT recomputed here — it is computed per-DPR in Pass 1
#         using the beta-distribution method (final_summary as hypothesis).
#         Combined scores are recalculated after this pass.

PASS 3  Query-relevance enrichment  (display-only, NOT in combined score)
        Adds per-DPR:
          query_dpr_relevance      — how well the DPR aligns with the query
          query_summary_relevance  — how well the summary addresses the query
        Source: --queries_file OR query_text embedded in stage-3 records.

Why query metrics are display-only
───────────────────────────────────
  Offline has no user query → cannot be computed.
  Offline+Query uses a static query → neutral 0.5 by default.
  Online has real user-query alignment → scores are meaningful.
  Including them in combined_score would make the three pipelines
  incomparable. They are supplementary signal only.

Metric weights (combined score — identical to run_eval_v3.py)
──────────────────────────────────────────────────────────────
  _W = 1.40  (0.30+0.20+0.20+0.20+0.20+0.15+0.15)
  Coverage          0.2143
  Complexity        0.1429
  Diversity         0.1429
  Surprisal         0.1429
  Uniqueness        0.1429
  LLM Quality       0.1071
  DPR-Summary Rel.  0.1071

Usage (from Stage-4/):
    python run_eval_all_queries.py \\
        --input_dir  ../stage-3/data/stage3_outputs/online_with_query \\
        --dpr_filename_pattern "Q*--online_stage3_output.json" \\
        --output_dir output/online_eval_final \\
        --queries_file data/online_user_queries.json \\
        --llm_api_key  $LLM_API_KEY \\
        --llm_api_base $LLM_API_BASE \\
        --llm_model    gpt4o \\
        --top_k 100
"""

from dotenv import load_dotenv
load_dotenv()

import os, sys, re, json, glob, argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR         = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

sys.path.insert(0, str(SCRIPT_DIR))
from run_eval_v3 import run_pipeline, compute_surprisal_frequency, compute_diversity, compute_uniqueness, _save_and_print

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

VALID_LLM_SCORES = {0.0, 0.5, 0.75, 1.0}

# ── Metric key lists ──────────────────────────────────────────────────────────
# 7 metrics that go into combined_score (must match run_eval_v3.py weights)
COMBINED_METRIC_KEYS = [
    ("Coverage",          "coverage"),
    ("Complexity",        "complexity"),
    ("Diversity",         "diversity"),
    ("Surprisal",         "surprisal"),
    ("Uniqueness",        "uniqueness"),
    ("LLM Quality",       "llm_quality"),
    ("DPR-Summary Rel.",  "summary_relevance"),
]

# 2 display-only query relevance metrics (NOT in combined_score)
QUERY_METRIC_KEYS = [
    ("Query-DPR Rel.",     "query_dpr_relevance"),
    ("Query-Summary Rel.", "query_summary_relevance"),
]

ALL_METRIC_KEYS = COMBINED_METRIC_KEYS + QUERY_METRIC_KEYS

# Weights — must stay in sync with run_eval_v3.py
_W = 0.30 + 0.20 + 0.20 + 0.20 + 0.20 + 0.15 + 0.15   # = 1.40
WEIGHTS = {
    "coverage":          round(0.30 / _W, 4),
    "complexity":        round(0.20 / _W, 4),
    "diversity":         round(0.20 / _W, 4),
    "surprisal":         round(0.20 / _W, 4),
    "uniqueness":        round(0.20 / _W, 4),
    "llm_quality":       round(0.15 / _W, 4),
    "summary_relevance": round(0.15 / _W, 4),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Query-relevance LLM helpers  (display-only)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_DPR_PROMPT = """You are an expert evaluator assessing how well a Data Product Request (DPR) aligns with a user's original query.

User Query:
\"{user_query}\"

Data Product Request (DPR):
\"{dpr_text}\"

Score how well the DPR aligns with the user's analytical intent:
  0    = Completely unrelated to the query
  0.5  = Partially aligns but misses key aspects
  0.75 = Mostly aligns with minor gaps
  1.0  = Directly and completely addresses the query intent

Respond ONLY with valid JSON, no extra text:
{{\"query_dpr_relevance\": <score>, \"reasoning\": \"<one sentence>\"}}
"""

QUERY_SUMMARY_PROMPT = """You are an expert evaluator assessing how well a generated summary addresses a user's original query.

User Query:
\"{user_query}\"

Generated Summary:
\"{summary_text}\"

Score how well the summary addresses the user's query:
  0    = Completely unrelated to the query
  0.5  = Partially addresses the query but misses key aspects
  0.75 = Mostly addresses the query with minor gaps
  1.0  = Directly and completely addresses the query

Respond ONLY with valid JSON, no extra text:
{{\"query_summary_relevance\": <score>, \"reasoning\": \"<one sentence>\"}}
"""


def _snap(v):
    v = float(v)
    return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))


def llm_query_dpr_relevance(user_query, dpr_text,
                             api_key="", api_base="https://api.openai.com/v1",
                             model="gpt-4"):
    fallback = {"query_dpr_relevance": 0.5,
                "query_dpr_reasoning": "LLM judge skipped."}
    if not api_key or not _HAS_REQUESTS or not dpr_text or not user_query:
        return fallback
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user",
                                "content": QUERY_DPR_PROMPT.format(
                                    user_query=user_query, dpr_text=dpr_text)}],
                  "max_tokens": 150, "temperature": 0.0},
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(r"```json|```", "",
                         resp.json()["choices"][0]["message"]["content"].strip()).strip()
        scores = json.loads(content)
        return {"query_dpr_relevance": _snap(scores.get("query_dpr_relevance", 0.5)),
                "query_dpr_reasoning": scores.get("reasoning", "")}
    except Exception as e:
        fallback["query_dpr_reasoning"] = f"LLM error: {e}"
        return fallback


def llm_query_summary_relevance(user_query, summary_text,
                                 api_key="", api_base="https://api.openai.com/v1",
                                 model="gpt-4"):
    fallback = {"query_summary_relevance": 0.5,
                "query_summary_reasoning": "LLM judge skipped."}
    if not api_key or not _HAS_REQUESTS or not summary_text or not user_query:
        return fallback
    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user",
                                "content": QUERY_SUMMARY_PROMPT.format(
                                    user_query=user_query, summary_text=summary_text)}],
                  "max_tokens": 150, "temperature": 0.0},
            timeout=30,
        )
        resp.raise_for_status()
        content = re.sub(r"```json|```", "",
                         resp.json()["choices"][0]["message"]["content"].strip()).strip()
        scores = json.loads(content)
        return {"query_summary_relevance": _snap(scores.get("query_summary_relevance", 0.5)),
                "query_summary_reasoning": scores.get("reasoning", "")}
    except Exception as e:
        fallback["query_summary_reasoning"] = f"LLM error: {e}"
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
#  Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(values):
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min":  round(min(values), 4),
            "max":  round(max(values), 4),
            "mean": round(sum(values) / len(values), 4)}


def write_stats_table(all_metrics, all_scores, path, title):
    lines = [f"  {title}\n\n"]
    lines.append(f"  {'Metric':<24}  {'Min':>8}  {'Max':>8}  {'Mean':>8}  Note\n")
    lines.append(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}\n")
    for label, key in ALL_METRIC_KEYS:
        s    = compute_stats(all_metrics.get(key, []))
        note = "[display only]" if key in ("query_dpr_relevance",
                                            "query_summary_relevance") else ""
        lines.append(f"  {label:<24}  {s['min']:>8.4f}  {s['max']:>8.4f}"
                     f"  {s['mean']:>8.4f}  {note}\n")
    lines.append(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    s = compute_stats(all_scores)
    lines.append(f"  {'Combined Score':<24}  {s['min']:>8.4f}  {s['max']:>8.4f}"
                 f"  {s['mean']:>8.4f}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"      Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 2: Cross-query pool recomputation
# ─────────────────────────────────────────────────────────────────────────────

def recompute_pool_metrics(all_output_data):
    """
    Recompute Diversity and Uniqueness across ALL DPRs from ALL queries,
    then recalculate the 7-metric combined_score and re-rank.

    Surprisal is NO LONGER recomputed here — it is now computed per-DPR
    in run_pipeline() using the beta-distribution method (AutoDiscovery),
    which uses each DPR's final_summary as hypothesis. It is independent
    of pool composition, so cross-query pooling adds no value.

    Diversity and Uniqueness still benefit from cross-query pooling because
    they measure how different each DPR is from ALL other DPRs in the pool.
    """
    print("\n[POST] Recomputing pool-wide metrics (Diversity, Uniqueness) across ALL queries...")

    flat = []
    for output, out_dir, stem in all_output_data:
        for r in output["ranked_dprs"]:
            flat.append({"dpr_text": r.get("dpr_text", ""),
                         "_ref":     r})

    n = len(flat)
    print(f"      Pool size: {n} DPRs across {len(all_output_data)} queries")

    texts = [r["dpr_text"] or "" for r in flat]
    if any(t.strip() for t in texts):
        vect           = TfidfVectorizer().fit(texts)
        emb            = vect.transform(texts).toarray()
        new_diversity  = compute_diversity(emb)
        new_uniqueness = compute_uniqueness(emb, threshold=0.85)
    else:
        new_diversity  = [0.0] * n
        new_uniqueness = [1.0] * n

    for i, rec in enumerate(flat):
        r = rec["_ref"]
        r["metrics"]["diversity"]  = round(float(new_diversity[i]),  4)
        r["metrics"]["uniqueness"] = float(new_uniqueness[i])
        # surprisal already set per-DPR in Pass 1 — do not overwrite
        r["combined_score"] = round(
            sum(WEIGHTS[k] * r["metrics"].get(k, 0.0) for k in WEIGHTS), 4)

    # Re-rank within each query
    for output, out_dir, stem in all_output_data:
        ranked = sorted(output["ranked_dprs"],
                        key=lambda x: x["combined_score"], reverse=True)
        for rank, r in enumerate(ranked, start=1):
            r["rank"] = rank
        output["ranked_dprs"] = ranked

    div_vals    = [rec["_ref"]["metrics"]["diversity"]  for rec in flat]
    uniq_count  = sum(1 for rec in flat if rec["_ref"]["metrics"]["uniqueness"] == 1.0)
    print(f"      Diversity  min={min(div_vals):.4f}  max={max(div_vals):.4f}  mean={sum(div_vals)/len(div_vals):.4f}")
    print(f"      Unique DPRs: {uniq_count}/{n}")
    return all_output_data


# ─────────────────────────────────────────────────────────────────────────────
#  PASS 3: Query-relevance enrichment  (display-only)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_query_metrics(output, user_query,
                               llm_api_key, llm_api_base, llm_model):
    """
    Add query_dpr_relevance and query_summary_relevance to each DPR.
    These are NOT included in combined_score — display-only supplementary signal.
    """
    if not user_query:
        print("      [WARN] No user_query — query metrics will be 0.5 neutral.")
        return output
    for r in output["ranked_dprs"]:
        qd = llm_query_dpr_relevance(
            user_query, r.get("dpr_text", ""),
            api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        qs = llm_query_summary_relevance(
            user_query, r.get("final_summary", ""),
            api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        r["metrics"].update(qd)
        r["metrics"].update(qs)
        r["user_query"] = user_query
    return output


# ─────────────────────────────────────────────────────────────────────────────
#  Per-query summary stats
# ─────────────────────────────────────────────────────────────────────────────

def build_query_stats(output, stem):
    q_metrics = {key: [] for _, key in ALL_METRIC_KEYS}
    q_scores  = []
    for r in output["ranked_dprs"]:
        q_scores.append(r["combined_score"])
        for _, key in ALL_METRIC_KEYS:
            v = r["metrics"].get(key)
            if isinstance(v, (int, float)):
                q_metrics[key].append(float(v))
    summary = {
        "output_folder":   stem,
        "num_dprs":        len(q_scores),
        "avg_score":       compute_stats(q_scores)["mean"],
        "metric_averages": {key: compute_stats(q_metrics[key])["mean"]
                            for _, key in ALL_METRIC_KEYS},
    }
    return q_metrics, q_scores, summary


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Online eval — 7-metric combined score + display-only query relevance"
    )
    parser.add_argument("--input_dir",           required=True)
    parser.add_argument("--dpr_filename_pattern", default="*.json")
    parser.add_argument("--queries_file",         default=None,
                        help='JSON: [{"dpr_id":"q1_1","user_query":"..."}]')
    parser.add_argument("--output_dir",           default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top_k",                type=int, default=100)
    parser.add_argument("--llm_api_key",          default="")
    parser.add_argument("--llm_api_base",         default="https://api.openai.com/v1")
    parser.add_argument("--llm_model",            default="gpt-4")
    args = parser.parse_args()

    llm_api_key  = args.llm_api_key  or os.getenv("LLM_API_KEY",  "")
    llm_api_base = args.llm_api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    llm_model    = args.llm_model    or os.getenv("LLM_MODEL",    "gpt-4")

    # Build query lookup from --queries_file if provided
    preloaded_lookup = {}
    if args.queries_file and os.path.exists(args.queries_file):
        with open(args.queries_file, encoding="utf-8") as f:
            qs = json.load(f)
        for q in qs:
            preloaded_lookup[str(q["dpr_id"])] = q["user_query"]
        print(f"[INFO] Loaded {len(preloaded_lookup)} queries from {args.queries_file}")
    else:
        print("[INFO] No --queries_file — will extract query_text from stage-3 records")

    # Find input files
    pattern   = os.path.join(args.input_dir, args.dpr_filename_pattern)
    all_files = sorted([f for f in glob.glob(pattern)
                        if "execution_summary" not in os.path.basename(f)])
    if not all_files:
        print(f"[ERROR] No files matched: {pattern}")
        return
    print(f"[INFO] Found {len(all_files)} stage-3 file(s):")
    for f in all_files:
        print(f"       {f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── PASS 1 ────────────────────────────────────────────────────────────────
    all_output_data = []
    file_query_map  = {}

    for idx, filepath in enumerate(all_files, start=1):
        stem    = Path(filepath).stem
        out_dir = os.path.join(args.output_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"[PASS1] [{idx}/{len(all_files)}] {os.path.basename(filepath)}")

        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()
        raw_data = (json.loads(content) if content.startswith("[")
                    else [json.loads(l) for l in content.splitlines() if l.strip()])

        if not raw_data:
            print(f"[WARN] Empty, skipping.")
            continue

        print(f"[INFO] Loaded {len(raw_data)} DPRs")

        # Resolve user_query: queries_file → stage-3 embedded query_text → none
        first_id   = str(raw_data[0].get("dpr_id", ""))
        user_query = preloaded_lookup.get(first_id, "")
        if not user_query:
            for rec in raw_data:
                qt = rec.get("query_text", "")
                if qt:
                    user_query = qt
                    break
        if user_query:
            file_query_map[stem] = user_query
            print(f"[INFO] Query: {user_query[:90]}{'...' if len(user_query)>90 else ''}")
        else:
            print("[WARN] No user_query found — query metrics will be 0.5 neutral")

        output = run_pipeline(
            raw_data=raw_data,
            output_dir=out_dir,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            top_k=args.top_k,
            write_outputs=False,   # suppress files/console — Pass 2 corrects surprisal first
        )
        all_output_data.append((output, out_dir, stem))

    if not all_output_data:
        print("[WARN] No files processed.")
        return

    # ── PASS 2 ────────────────────────────────────────────────────────────────
    all_output_data = recompute_pool_metrics(all_output_data)

    # ── PASS 3 ────────────────────────────────────────────────────────────────
    all_metrics     = {key: [] for _, key in ALL_METRIC_KEYS}
    all_scores      = []
    query_summaries = []

    for output, out_dir, stem in all_output_data:
        user_query = file_query_map.get(stem, "")
        if user_query:
            print(f"\n[INFO] Query-relevance metrics (display-only) for {stem}...")
            output = enrich_with_query_metrics(
                output, user_query, llm_api_key, llm_api_base, llm_model)

        ranked_path = os.path.join(out_dir, "dpr_ranked_results.json")
        with open(ranked_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[INFO] Saved → {ranked_path}")

        # Write ranking_summary.txt and metrics_stats.txt with corrected values
        _save_and_print(output, out_dir)

        q_metrics, q_scores, summary = build_query_stats(output, stem)
        summary["file"] = stem + ".json"

        write_stats_table(q_metrics, q_scores,
                          os.path.join(out_dir, "metrics_avg_this_query.txt"),
                          f"Average metrics for {stem} ({len(q_scores)} DPRs)")

        all_scores.extend(q_scores)
        for _, key in ALL_METRIC_KEYS:
            all_metrics[key].extend(q_metrics.get(key, []))
        query_summaries.append(summary)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    write_stats_table(all_metrics, all_scores,
                      os.path.join(args.output_dir, "metrics_avg_all_queries.txt"),
                      f"Aggregate across {len(all_files)} file(s) ({len(all_scores)} total DPRs)")

    with open(os.path.join(args.output_dir, "summary_all_queries.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "total_files": len(query_summaries),
            "total_dprs":  len(all_scores),
            "note": (
                "combined_score uses 7 metrics: coverage, complexity, diversity, "
                "surprisal, uniqueness, llm_quality, summary_relevance. "
                "query_dpr_relevance and query_summary_relevance are DISPLAY-ONLY."
            ),
            "overall_avg_combined_score": compute_stats(all_scores)["mean"],
            "overall_metric_stats": {
                key: compute_stats(all_metrics[key]) for _, key in ALL_METRIC_KEYS
            },
            "per_file": query_summaries,
        }, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AGGREGATE SUMMARY — {len(query_summaries)} queries, {len(all_scores)} DPRs")
    print(f"{'='*65}")
    print(f"  {'Query File':<45} {'DPRs':>6}  {'Avg Score':>10}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*10}")
    for qs in query_summaries:
        print(f"  {qs['file']:<45} {qs['num_dprs']:>6}  {qs['avg_score']:>10.4f}")
    print(f"\n  Overall avg combined score: {compute_stats(all_scores)['mean']:.4f}")
    print(f"\n  Metric averages:")
    print(f"  {'Metric':<24}  {'Mean':>8}  Note")
    print(f"  {'-'*24}  {'-'*8}  {'-'*20}")
    for label, key in ALL_METRIC_KEYS:
        mean = compute_stats(all_metrics[key])["mean"]
        note = "[display only]" if key in ("query_dpr_relevance",
                                            "query_summary_relevance") else "[in combined]"
        print(f"  {label:<24}  {mean:>8.4f}  {note}")
    print(f"\n  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
