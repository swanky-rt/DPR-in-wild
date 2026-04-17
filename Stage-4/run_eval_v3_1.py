"""
DPR Evaluation & Ranking — v3

Metric computations:
  Coverage   = |tables_used ∩ ground_truth| / |ground_truth|
  Complexity = mean(multi_table, join, agg, subquery, multi_entity)
  Diversity  = 1 - avg_cosine_sim(DPR_i, all other DPRs_i)
  Surprisal  = -log P(tables_used) / log(|all_tables|)
  Uniqueness = 1 if no near-duplicate exists (cosine > 0.85), else 0
  LLM Quality = GPT-style score: 0 / 0.5 / 0.75 / 1.0
  Summary Relevance = relevance of final_summary to DPR text

Note:
  This file computes an initial 7-metric ranking.
  Final ranking may later be recomputed in run_eval_all_queries.py
  after query-based relevance metrics are added.
"""

import json
import re
import math
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def extract_tables_from_sql(sql: str) -> set:
    if not sql:
        return set()
    matches = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', sql, re.IGNORECASE)
    keywords = {
        "select", "where", "on", "and", "or", "not", "null", "limit", "order",
        "group", "by", "having", "union", "as", "inner", "outer", "left",
        "right", "cross", "natural", "set"
    }
    return {m.upper() for m in matches if m.lower() not in keywords}


def compute_coverage(subquery_results: list, ground_truth_tables: list) -> float:
    gt = set(t.upper() for t in ground_truth_tables)
    if not gt:
        return 0.0

    tables_used = set()
    for sq in subquery_results:
        sql = sq.get("final_sql") or ""
        tables_used |= extract_tables_from_sql(sql)

    if not tables_used:
        return 0.0

    return round(len(tables_used & gt) / len(gt), 4)


def compute_complexity_single(sql: str, dpr_text: str = "") -> tuple:
    sql_up = (sql or "").upper()
    tables = extract_tables_from_sql(sql)
    breakdown = {
        "multi_table": float(len(tables) > 1),
        "join": float("JOIN" in sql_up),
        "agg": float(any(k in sql_up for k in ["GROUP BY", "COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])),
        "subquery": float(sql_up.count("SELECT") > 1),
        "multi_entity": float(len(set(re.findall(r'\b[A-Z][a-z]+\b', dpr_text))) > 3),
    }
    score = round(float(np.mean(list(breakdown.values()))), 4)
    return score, breakdown


def compute_complexity_dpr(subquery_results: list, dpr_text: str) -> tuple:
    if not subquery_results:
        return 0.0, {}

    scores = []
    all_breakdowns = []
    for sq in subquery_results:
        sql = sq.get("final_sql") or ""
        score, bd = compute_complexity_single(sql, dpr_text)
        scores.append(score)
        all_breakdowns.append(bd)

    avg_score = round(float(np.mean(scores)), 4)
    avg_breakdown = {
        key: round(float(np.mean([bd[key] for bd in all_breakdowns])), 4)
        for key in all_breakdowns[0]
    }
    return avg_score, avg_breakdown


def compute_diversity(embeddings: np.ndarray) -> np.ndarray:
    if embeddings is None or len(embeddings) < 2:
        return np.zeros(len(embeddings) if embeddings is not None else 0)

    sims = cosine_similarity(embeddings)
    n = sims.shape[0]
    np.fill_diagonal(sims, 0.0)
    avg_sim = sims.sum(axis=1) / (n - 1)
    return 1.0 - avg_sim


def compute_surprisal(all_records: list) -> list:
    n = len(all_records)
    if n == 0:
        return []

    all_tables_seen = set()
    table_combo_counts = {}

    for r in all_records:
        combo = frozenset(t.upper() for t in r["tables"])
        all_tables_seen |= combo
        table_combo_counts[combo] = table_combo_counts.get(combo, 0) + 1

    total_tables = len(all_tables_seen)
    if total_tables <= 1:
        return [0.0] * n

    surprisals = []
    for r in all_records:
        combo = frozenset(t.upper() for t in r["tables"])
        freq = table_combo_counts[combo]
        p = freq / n
        raw = -math.log(p + 1e-9)
        norm = raw / math.log(total_tables + 1e-9)
        surprisals.append(round(norm, 4))

    return surprisals


def compute_uniqueness(embeddings: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    if embeddings is None or len(embeddings) < 2:
        return np.ones(len(embeddings) if embeddings is not None else 0)

    sims = cosine_similarity(embeddings)
    n = sims.shape[0]
    flags = np.ones(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] > threshold:
                flags[j] = 0

    return flags


JUDGE_PROMPT = """You are an expert evaluator of Data Product Requests (DPRs).

A DPR is a high-level, actionable specification of what data and analysis a user needs.

Evaluate the DPR on ONE dimension and respond with ONLY one of these scores:
  0, 0.5, 0.75, or 1.0

DPR TEXT:
"{dpr_text}"

SCHEMA CONTEXT (tables and columns this DPR was derived from):
{schema_context}

QUALITY (Is the DPR well-formed, analytical, and appropriate in scope for the tables?):
  0    = Factoid/trivial
  0.5  = Borderline
  0.75 = Analytical
  1.0  = Excellent

Respond ONLY with valid JSON, no extra text:
{{"quality": <score>, "reasoning": "<one sentence>"}}
"""

VALID_LLM_SCORES = {0.0, 0.5, 0.75, 1.0}


def _build_schema_context(schema_mapping: dict) -> str:
    lines = []
    for tid, cols in schema_mapping.items():
        col_list = ", ".join(cols.keys()) if isinstance(cols, dict) else str(cols)
        lines.append(f"  Table {tid}: columns = {col_list}")
    return "\n".join(lines) if lines else "No schema available."


def llm_judge(dpr_text, schema_mapping, api_key="", api_base="https://api.openai.com/v1", model="gpt-4") -> dict:
    fallback = {
        "quality": 0.5,
        "reasoning": "LLM judge skipped (no API key provided).",
    }
    if not api_key or not _HAS_REQUESTS:
        return fallback

    prompt = JUDGE_PROMPT.format(
        dpr_text=dpr_text,
        schema_context=_build_schema_context(schema_mapping),
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
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"```json|```", "", content).strip()
        scores = json.loads(content)

        def snap(v):
            v = float(v)
            return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))

        return {
            "quality": snap(scores.get("quality", 0.5)),
            "reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["reasoning"] = f"LLM judge error: {e}"
        return fallback


SUMMARY_RELEVANCE_PROMPT = """You are an expert evaluator assessing how well a generated summary addresses a Data Product Request (DPR).

DPR (the original request):
"{dpr_text}"

GENERATED SUMMARY (the output produced for this DPR):
"{summary_text}"

Score how relevant the summary is to the DPR on this scale:
  0    = Summary is unrelated or addresses completely different questions
  0.5  = Summary partially addresses the DPR but misses key analytical goals
  0.75 = Summary mostly addresses the DPR with minor gaps
  1.0  = Summary directly and completely addresses the DPR's analytical goals

Respond ONLY with valid JSON, no extra text:
{{"summary_relevance": <score>, "reasoning": "<one sentence explaining the score>"}}
"""


def llm_summary_relevance(
    dpr_text: str,
    summary_text: str,
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4",
) -> dict:
    fallback = {
        "summary_relevance": 0.5,
        "summary_relevance_reasoning": "LLM judge skipped (no API key provided).",
    }

    if not api_key or not _HAS_REQUESTS or not summary_text:
        return fallback

    prompt = SUMMARY_RELEVANCE_PROMPT.format(
        dpr_text=dpr_text,
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
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"```json|```", "", content).strip()
        scores = json.loads(content)

        def snap(v):
            v = float(v)
            return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))

        return {
            "summary_relevance": snap(scores.get("summary_relevance", 0.5)),
            "summary_relevance_reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["summary_relevance_reasoning"] = f"LLM judge error: {e}"
        return fallback


def run_pipeline(raw_data, output_dir, llm_api_key, llm_api_base, llm_model, top_k):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Computing per-DPR metrics for {len(raw_data)} DPRs...")
    if llm_api_key:
        print(f"      LLM judge: {llm_model} @ {llm_api_base}")
    else:
        print("      LLM judge: SKIPPED (no --llm_api_key provided) — defaulting to 0.5")

    records = []
    for i, d in enumerate(raw_data):
        dpr_id = str(d.get("dpr_id", ""))
        dpr_text = d.get("DPR", "")
        tables = d.get("tables", [])
        ground_truth = d.get("ground_truth", {}).get("table_uids", tables)
        subquery_results = d.get("subquery_results", [])
        schema_mapping = d.get("schema_mapping", {})
        final_summary = d.get("final_summary", "")
        mini_summaries = d.get("mini_summaries", [])

        coverage = compute_coverage(subquery_results, ground_truth)
        complexity, cpx_breakdown = compute_complexity_dpr(subquery_results, dpr_text)

        judge = llm_judge(
            dpr_text,
            schema_mapping,
            api_key=llm_api_key,
            api_base=llm_api_base,
            model=llm_model,
        )

        summary_judge = llm_summary_relevance(
            dpr_text,
            final_summary,
            api_key=llm_api_key,
            api_base=llm_api_base,
            model=llm_model,
        )

        sub_evals = []
        for sq in subquery_results:
            sql = sq.get("final_sql") or ""
            exec_ok = bool(sq.get("final_execution_status", False))
            row_count = int(sq.get("final_row_count") or 0)
            attempts = sq.get("attempts", [])
            mini_sum = sq.get("mini_summary") or ""
            sub_q = sq.get("sub_question") or ""
            sq_complexity, sq_cpx_bd = compute_complexity_single(sql, dpr_text)

            attempt_success = (
                sum(1 for a in attempts if a.get("execution_status", False)) / len(attempts)
                if attempts else 0.0
            )

            sub_evals.append({
                "sub_question": sub_q,
                "sql": sql,
                "mini_summary": mini_sum,
                "metrics": {
                    "execution_status": float(exec_ok),
                    "complexity": sq_complexity,
                    "complexity_breakdown": sq_cpx_bd,
                    "attempt_success_rate": round(attempt_success, 4),
                    "n_attempts": len(attempts),
                    "rows_returned": row_count,
                }
            })

        records.append({
            "dpr_id": dpr_id,
            "dpr_text": dpr_text,
            "tables": tables,
            "ground_truth": ground_truth,
            "schema_mapping": schema_mapping,
            "final_summary": final_summary,
            "mini_summaries": mini_summaries,
            "sub_evals": sub_evals,
            "metrics": {
                "coverage": coverage,
                "complexity": complexity,
                "complexity_breakdown": cpx_breakdown,
                "llm_quality": judge["quality"],
                "llm_reasoning": judge["reasoning"],
                "summary_relevance": summary_judge["summary_relevance"],
                "summary_relevance_reasoning": summary_judge["summary_relevance_reasoning"],
            }
        })

        print(
            f"      [{i+1}/{len(raw_data)}] DPR {dpr_id}  "
            f"coverage={coverage:.3f}  complexity={complexity:.3f}  "
            f"llm_q={judge['quality']}"
        )

    print("\n[2/4] Computing pool-wide metrics (Diversity, Surprisal, Uniqueness)...")
    n = len(records)
    texts = [r["dpr_text"] if r["dpr_text"] else "" for r in records]

    if any(t.strip() for t in texts):
        vect = TfidfVectorizer().fit(texts)
        emb = vect.transform(texts).toarray()
        diversity_arr = compute_diversity(emb)
        uniqueness_arr = compute_uniqueness(emb, threshold=0.85)
    else:
        diversity_arr = np.zeros(n)
        uniqueness_arr = np.ones(n)

    surprisal_list = compute_surprisal(records)

    for i, r in enumerate(records):
        r["metrics"]["diversity"] = round(float(diversity_arr[i]), 4)
        r["metrics"]["surprisal"] = round(float(surprisal_list[i]), 4)
        r["metrics"]["uniqueness"] = float(uniqueness_arr[i])

    print(f"\n[3/4] Ranking (top-{top_k})...")

    _W = 0.30 + 0.20 + 0.20 + 0.20 + 0.20 + 0.15 + 0.15
    WEIGHTS = {
        "coverage": round(0.30 / _W, 4),
        "complexity": round(0.20 / _W, 4),
        "diversity": round(0.20 / _W, 4),
        "surprisal": round(0.20 / _W, 4),
        "uniqueness": round(0.20 / _W, 4),
        "llm_quality": round(0.15 / _W, 4),
        "summary_relevance": round(0.15 / _W, 4),
    }
    assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, f"Weights must sum to 1.0, got {sum(WEIGHTS.values())}"
    print(f"      Weights sum = {sum(WEIGHTS.values()):.4f} ✓")

    for r in records:
        m = r["metrics"]
        r["combined_score"] = round(sum(WEIGHTS[k] * m.get(k, 0.0) for k in WEIGHTS), 4)

    ranked = sorted(records, key=lambda x: x["combined_score"], reverse=True)[:top_k]
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank

    output = {
        "summary": {
            "total_dprs": n,
            "total_ranked": len(ranked),
            "unique_count": int(uniqueness_arr.sum()),
            "multi_table_count": sum(1 for r in records if len(r["tables"]) > 1),
            "llm_model": llm_model,
            "llm_api_base": llm_api_base,
            "weights_used": WEIGHTS,
            "formulas": {
                "coverage": "|tables_used ∩ ground_truth| / |ground_truth|",
                "complexity": "mean(multi_table, join, agg, subquery, multi_entity)",
                "diversity": "1 - avg_cosine_sim(DPR_i, all other DPRs)",
                "surprisal": "-log P(tables_used) / log(|all_tables|)",
                "uniqueness": "1 if no near-duplicate (cosine > 0.85), else 0",
                "llm_scores": "GPT-4 scores: 0 / 0.5 / 0.75 / 1.0",
            }
        },
        "ranked_dprs": []
    }

    for r in ranked:
        m = r["metrics"]
        output["ranked_dprs"].append({
            "rank": r["rank"],
            "dpr_id": r["dpr_id"],
            "combined_score": r["combined_score"],
            "dpr_text": r["dpr_text"],
            "tables_used": r["tables"],
            "ground_truth": r["ground_truth"],
            "summary_source": "stage3_output",
            "final_summary": r["final_summary"],
            "metrics": {
                "coverage": m["coverage"],
                "complexity": m["complexity"],
                "complexity_breakdown": m["complexity_breakdown"],
                "diversity": m["diversity"],
                "surprisal": m["surprisal"],
                "uniqueness": m["uniqueness"],
                "llm_quality": m["llm_quality"],
                "llm_reasoning": m["llm_reasoning"],
                "summary_relevance": m["summary_relevance"],
                "summary_relevance_reasoning": m["summary_relevance_reasoning"],
            },
            "sub_queries": [
                {
                    "sub_query_index": idx + 1,
                    "sub_question": se["sub_question"],
                    "sql": se["sql"],
                    "summary_source": "stage3_output",
                    "mini_summary": se["mini_summary"],
                    "metrics": se["metrics"],
                }
                for idx, se in enumerate(r["sub_evals"])
            ]
        })

    print("\n[4/4] Saving outputs...")

    json_path = os.path.join(output_dir, "dpr_ranked_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"      Saved JSON      → {json_path}")

    ranking_path = os.path.join(output_dir, "dpr_ranking_summary.txt")
    _write_ranking_summary(output, ranking_path)
    print(f"      Saved ranking   → {ranking_path}")

    stats_path = os.path.join(output_dir, "metrics_stats.txt")
    _write_metrics_stats(output, stats_path)
    print(f"      Saved stats     → {stats_path}")

    print("\n" + "=" * 72)
    print("  DPR EVALUATION v3 — RANKED RESULTS")
    print("=" * 72)
    print(f"\n{'Rank':<6} {'DPR ID':<12} {'Score':<8} {'Coverage':<10} {'Complexity':<12} "
          f"{'Diversity':<10} {'Surprisal':<10} {'Unique':<8} {'LLM-Q':<7} {'Sum-Rel'}")
    print("-" * 90)
    for r in ranked:
        m = r["metrics"]
        dup = " ⚠️" if m["uniqueness"] == 0 else ""
        print(
            f"#{r['rank']:<5} {r['dpr_id']:<12} {r['combined_score']:<8.4f} "
            f"{m['coverage']:<10.4f} {m['complexity']:<12.4f} "
            f"{m['diversity']:<10.4f} {m['surprisal']:<10.4f} "
            f"{int(m['uniqueness']):<8} {m['llm_quality']:<7} {m['summary_relevance']}{dup}"
        )

    print("\nSub-query detail:")
    for r in output["ranked_dprs"]:
        print(f"\n  DPR {r['dpr_id']} — {len(r['sub_queries'])} sub-queries:")
        for sq in r["sub_queries"]:
            sm = sq["metrics"]
            status = "✅" if sm["execution_status"] else "❌"
            print(
                f"    [{sq['sub_query_index']}] {status} complexity={sm['complexity']:.2f} "
                f"rows={sm['rows_returned']} | {sq['sub_question'][:65]}..."
            )
            print(f"        SQL: {(sq['sql'] or 'N/A')[:65]}...")
            print(f"        Summary: {(sq['mini_summary'] or 'N/A')[:75]}...")

    print(f"\n✅ Evaluation complete. Outputs in: {output_dir}")
    return output


def _wrap(text, width=74, indent=2):
    words = (text or "N/A").split()
    lines, buf, col = [], [], 0
    for w in words:
        if col + len(w) + 1 > width:
            lines.append(" " * indent + " ".join(buf))
            buf, col = [w], len(w)
        else:
            buf.append(w)
            col += len(w) + 1
    if buf:
        lines.append(" " * indent + " ".join(buf))
    return "\n".join(lines)


def _write_ranking_summary(output: dict, path: str):
    lines = []
    SEP = "=" * 60
    DASH = "-" * 60

    for r in output["ranked_dprs"]:
        m = r["metrics"]

        lines.append(f"Rank #{r['rank']}  |  DPR: {r['dpr_id']}  |  Combined Score: {r['combined_score']:.4f}\n")
        lines.append(SEP + "\n")

        lines.append("  Original DPR:\n")
        lines.append(_wrap(r["dpr_text"], width=72, indent=4) + "\n\n")

        for label, key in [
            ("Coverage", "coverage"),
            ("Complexity", "complexity"),
            ("Diversity", "diversity"),
            ("Surprisal", "surprisal"),
            ("Uniqueness", "uniqueness"),
            ("LLM Quality", "llm_quality"),
            ("Summary Relevance", "summary_relevance"),
        ]:
            lines.append(f"  {label:<20} {float(m.get(key, 0.0)):.4f}\n")

        if m.get("llm_reasoning"):
            lines.append("  LLM Reasoning:     " + _wrap(m["llm_reasoning"], width=72, indent=22).lstrip() + "\n")
        sum_rel_reason = m.get("summary_relevance_reasoning", "")
        if sum_rel_reason:
            lines.append("  Summary Reasoning: " + _wrap(sum_rel_reason, width=72, indent=22).lstrip() + "\n")
        lines.append("\n")
        lines.append("  Summary:\n")
        lines.append(_wrap(r["final_summary"], width=72, indent=4) + "\n\n")

        sub_queries = [sq for sq in r.get("sub_queries", []) if (sq.get("sql") or "").strip() not in ("", "N/A")]
        if sub_queries:
            lines.append(f"  Sub-Queries ({len(sub_queries)} total):\n")
            lines.append(DASH + "\n")
            for idx, sq in enumerate(sub_queries, start=1):
                lines.append(f"  [{idx}] SQL:\n")
                sql_text = sq["sql"].strip()
                for sql_line in sql_text.splitlines():
                    lines.append(f"        {sql_line}\n")
                lines.append("\n")
            lines.append(DASH + "\n")

        lines.append(SEP + "\n\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_metrics_stats(output: dict, path: str):
    metric_keys = [
        ("Coverage", "coverage"),
        ("Complexity", "complexity"),
        ("Diversity", "diversity"),
        ("Surprisal", "surprisal"),
        ("Uniqueness", "uniqueness"),
        ("LLM Quality", "llm_quality"),
        ("Summary Relevance", "summary_relevance"),
    ]
    vals = {key: [] for _, key in metric_keys}
    scores = []

    for r in output["ranked_dprs"]:
        scores.append(r["combined_score"])
        for _, key in metric_keys:
            v = r["metrics"].get(key)
            if isinstance(v, (int, float)):
                vals[key].append(float(v))

    lines = []
    lines.append(f"  {'Metric':<20}  {'Min':>8}  {'Max':>8}  {'Mean':>8}\n")
    lines.append(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    for label, key in metric_keys:
        v = vals[key]
        mn = min(v) if v else 0.0
        mx = max(v) if v else 0.0
        avg = sum(v) / len(v) if v else 0.0
        lines.append(f"  {label:<20}  {mn:>8.4f}  {mx:>8.4f}  {avg:>8.4f}\n")
    lines.append(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    lines.append(f"  {'Combined Score':<20}  {min(scores):>8.4f}  {max(scores):>8.4f}  {sum(scores)/len(scores):>8.4f}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPR Evaluation Pipeline v3 — multi sub-query, exact slide formulas"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--llm_api_key", default="")
    parser.add_argument("--llm_api_base", default="https://api.openai.com/v1")
    parser.add_argument("--llm_model", default="gpt-4")
    args = parser.parse_args()

    print(f"[Config]")
    print(f"  input       : {args.input}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  top_k       : {args.top_k}")

    env_api_key = os.getenv("LLM_API_KEY")
    env_api_base = os.getenv("LLM_API_BASE")
    env_model = os.getenv("LLM_MODEL")

    llm_api_key = env_api_key if env_api_key else args.llm_api_key
    llm_api_base = env_api_base if env_api_base else args.llm_api_base
    llm_model = env_model if env_model else args.llm_model

    print(f"  llm_model   : {llm_model}")
    print(f"  llm_api_base: {llm_api_base}")
    print(f"  llm_api_key : {'SET (env)' if env_api_key else 'SET (cli)' if args.llm_api_key else 'NOT SET'}")

    with open(args.input, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"\nLoaded {len(raw_data)} DPRs from {args.input}")

    run_pipeline(
        raw_data=raw_data,
        output_dir=args.output_dir,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        top_k=args.top_k,
    )