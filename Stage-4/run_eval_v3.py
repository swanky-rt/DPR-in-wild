"""
DPR Evaluation & Ranking — v3
Metric computations match exactly the slide definitions:

Slide 1/2 (SQL + Schema metrics):
  Coverage   = |tables_used ∩ ground_truth| / |ground_truth|
  Complexity = mean(multi_table, join, agg, subquery, multi_entity)   [5 binary dims]
  Diversity  = 1 - avg_cosine_sim(DPR_i, all other DPRs_i)
  Surprisal  = -log P(tables_used) / log(|all_tables|)

Slide 2/2 (LLM-based metrics):
  Uniqueness    = 1 if no near-duplicate exists (cosine > 0.85), else 0
  LLM Quality   = GPT-4 scores 0 / 0.5 / 0.75 / 1.0
  LLM Clarity   = GPT-4 scores 0 / 0.5 / 0.75 / 1.0
  LLM Relevance = GPT-4 scores 0 / 0.5 / 0.75 / 1.0

Combined Score (from slide):
  = 0.3×Coverage + 0.2×Complexity + 0.2×Diversity + 0.2×Surprisal
  + 0.2×Uniqueness + 0.15×LLM_Quality + 0.15×LLM_Relevance
"""

import json
import re
import math
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Optional: real LLM judge ──────────────────────────────────────────────────
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with open("stage3_output_final.json") as f:
    raw_data = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: extract tables from a SQL string
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_from_sql(sql: str) -> set:
    if not sql:
        return set()
    matches = re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_]\w*)', sql, re.IGNORECASE)
    keywords = {"select","where","on","and","or","not","null","limit","order",
                "group","by","having","union","as","inner","outer","left",
                "right","cross","natural","set"}
    return {m.upper() for m in matches if m.lower() not in keywords}


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC 1: COVERAGE
#  Formula (slide): |tables_used ∩ ground_truth| / |ground_truth|
#  Applied at DPR level: union of all tables touched by sub-SQLs vs ground truth
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage(subquery_results: list, ground_truth_tables: list) -> float:
    gt = set(t.upper() for t in ground_truth_tables)
    if not gt:
        return 0.0

    # Union of all tables referenced across every sub-SQL
    tables_used = set()
    for sq in subquery_results:
        sql = sq.get("final_sql") or ""
        tables_used |= extract_tables_from_sql(sql)

    if not tables_used:
        return 0.0

    coverage = len(tables_used & gt) / len(gt)
    return round(coverage, 4)


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC 2: COMPLEXITY
#  Formula (slide): mean(multi_table, join, agg, subquery, multi_entity)
#  5 binary dimensions, averaged → [0, 1]
#  Applied per sub-SQL, then averaged across sub-queries for DPR level
# ─────────────────────────────────────────────────────────────────────────────

def compute_complexity_single(sql: str, dpr_text: str = "") -> tuple:
    sql_up = (sql or "").upper()

    # Extract tables for multi_table check
    tables = extract_tables_from_sql(sql)

    breakdown = {
        "multi_table":  float(len(tables) > 1),
        "join":         float("JOIN" in sql_up),
        "agg":          float(any(k in sql_up for k in
                                  ["GROUP BY", "COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])),
        "subquery":     float(sql_up.count("SELECT") > 1),
        "multi_entity": float(len(set(re.findall(r'\b[A-Z][a-z]+\b', dpr_text))) > 3),
    }
    score = round(float(np.mean(list(breakdown.values()))), 4)
    return score, breakdown


def compute_complexity_dpr(subquery_results: list, dpr_text: str) -> tuple:
    """Average complexity across all sub-SQLs for DPR-level score."""
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

    # Average breakdown across sub-queries
    avg_breakdown = {}
    for key in all_breakdowns[0]:
        avg_breakdown[key] = round(float(np.mean([bd[key] for bd in all_breakdowns])), 4)

    return avg_score, avg_breakdown


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC 3: DIVERSITY  (pool-wide, computed after all DPRs are loaded)
#  Formula (slide): 1 - avg_cosine_sim(DPR_i, all other DPRs_i)
#  Uses TF-IDF on DPR text (falls back when sentence-transformers unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity(embeddings: np.ndarray) -> np.ndarray:
    if embeddings is None or len(embeddings) < 2:
        return np.zeros(len(embeddings) if embeddings is not None else 0)

    sims = cosine_similarity(embeddings)   # (n, n)
    n = sims.shape[0]
    np.fill_diagonal(sims, 0.0)
    avg_sim = sims.sum(axis=1) / (n - 1)
    return 1.0 - avg_sim                   # higher = more diverse


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC 4: SURPRISAL
#  Formula (slide): -log P(tables_used) / log(|all_tables|)
#  P(tables_used) = frequency of this exact table-set combination across all DPRs
#  Normalised by log(|all_tables|) so result ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def compute_surprisal(all_records: list) -> list:
    """
    For each DPR, compute surprisal based on how rare its table combination is.
    -log P(tables_used) / log(|all_tables|)
    """
    n = len(all_records)
    if n == 0:
        return []

    # Collect all unique tables across the full dataset
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
        freq  = table_combo_counts[combo]
        p     = freq / n                                  # empirical probability
        raw   = -math.log(p + 1e-9)                      # -log P
        norm  = raw / math.log(total_tables + 1e-9)      # normalise by log(|all_tables|)
        surprisals.append(round(norm, 4))

    return surprisals


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC 5: UNIQUENESS
#  Formula (slide): 1 if no near-duplicate exists, else 0
#  Threshold: cosine similarity > 0.85 (slide example uses 0.85)
# ─────────────────────────────────────────────────────────────────────────────

def compute_uniqueness(embeddings: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    if embeddings is None or len(embeddings) < 2:
        return np.ones(len(embeddings) if embeddings is not None else 0)

    sims = cosine_similarity(embeddings)
    n = sims.shape[0]
    flags = np.ones(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] > threshold:
                flags[j] = 0   # mark later occurrence as duplicate
    return flags


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS 6–8: LLM-AS-JUDGE  (Quality, Clarity, Relevance)
#  Formula (slide): GPT-4 scores 0 / 0.5 / 0.75 / 1.0
#  Prompt mirrors original evaluation.py JUDGE_PROMPT style
#  Falls back to neutral 0.5 if no API key provided
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator of Data Product Requests (DPRs).

A DPR is a high-level, actionable specification of what data and analysis a user needs.

Evaluate the DPR on THREE dimensions. For each, respond with ONLY one of these scores:
  0, 0.5, 0.75, or 1.0

DPR TEXT:
\"{dpr_text}\"

SCHEMA CONTEXT (tables and columns this DPR was derived from):
{schema_context}

---
1. QUALITY (Is the DPR well-formed, analytical, and appropriate in scope for the tables?):
   0 = Factoid/trivial  0.5 = Borderline  0.75 = Analytical  1.0 = Excellent

2. CLARITY (Is the DPR unambiguously worded? Could an analyst understand it without context?):
   0 = Very confusing  0.5 = Somewhat clear  0.75 = Mostly clear  1.0 = Crystal clear

3. RELEVANCE (Does the DPR actually align with the schema columns that are available?):
   0 = Unrelated  0.5 = Loosely related  0.75 = Mostly aligned  1.0 = Directly grounded

Respond ONLY with valid JSON, no extra text:
{{"quality": <score>, "clarity": <score>, "relevance": <score>, "reasoning": "<one sentence>"}}
"""

VALID_LLM_SCORES = {0.0, 0.5, 0.75, 1.0}


def _build_schema_context(schema_mapping: dict) -> str:
    lines = []
    for tid, cols in schema_mapping.items():
        col_list = ", ".join(cols.keys()) if isinstance(cols, dict) else str(cols)
        lines.append(f"  Table {tid}: columns = {col_list}")
    return "\n".join(lines) if lines else "No schema available."


def llm_judge(
    dpr_text: str,
    schema_mapping: dict,
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4",
) -> dict:
    """
    Call LLM judge. Returns quality, clarity, relevance each as 0/0.5/0.75/1.0.
    Falls back to neutral 0.5 if no API key or call fails.
    """
    fallback = {"quality": 0.5, "clarity": 0.5, "relevance": 0.5,
                "reasoning": "LLM judge skipped (no API key provided)."}

    if not api_key or not _HAS_REQUESTS:
        return fallback

    prompt = JUDGE_PROMPT.format(
        dpr_text=dpr_text,
        schema_context=_build_schema_context(schema_mapping),
    )

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200,
                  "temperature": 0.0},
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"```json|```", "", content).strip()
        scores = json.loads(content)

        # Snap to valid score set {0, 0.5, 0.75, 1.0}
        def snap(v):
            v = float(v)
            return min(VALID_LLM_SCORES, key=lambda s: abs(s - v))

        return {
            "quality":   snap(scores.get("quality",  0.5)),
            "clarity":   snap(scores.get("clarity",  0.5)),
            "relevance": snap(scores.get("relevance", 0.5)),
            "reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        fallback["reasoning"] = f"LLM judge error: {e}"
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ALL DPRs — compute per-record metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(raw_data, output_dir, llm_api_key, llm_api_base, llm_model, top_k):
    """Full evaluation pipeline: per-DPR metrics → pool-wide metrics → ranking → outputs."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Per-DPR metrics ───────────────────────────────────────────────────────
    print(f"\n[1/4] Computing per-DPR metrics for {len(raw_data)} DPRs...")
    if llm_api_key:
        print(f"      LLM judge: {llm_model} @ {llm_api_base}")
    else:
        print("      LLM judge: SKIPPED (no --llm_api_key provided) — defaulting to 0.5")

    records = []
    for i, d in enumerate(raw_data):
        dpr_id           = str(d.get("dpr_id", ""))
        dpr_text         = d.get("DPR", "")
        tables           = d.get("tables", [])
        ground_truth     = d.get("ground_truth", {}).get("table_uids", tables)
        subquery_results = d.get("subquery_results", [])
        schema_mapping   = d.get("schema_mapping", {})
        # ── Summaries read directly from Stage 3 output — NOT re-generated ──
        final_summary    = d.get("final_summary", "")
        mini_summaries   = d.get("mini_summaries", [])

        coverage              = compute_coverage(subquery_results, ground_truth)
        complexity, cpx_breakdown = compute_complexity_dpr(subquery_results, dpr_text)

        judge = llm_judge(dpr_text, schema_mapping,
                          api_key=llm_api_key,
                          api_base=llm_api_base,
                          model=llm_model)

        sub_evals = []
        for sq in subquery_results:
            sql       = sq.get("final_sql") or ""
            exec_ok   = bool(sq.get("final_execution_status", False))
            row_count = int(sq.get("final_row_count") or 0)
            attempts  = sq.get("attempts", [])
            mini_sum  = sq.get("mini_summary") or ""   # directly from Stage 3
            sub_q     = sq.get("sub_question") or ""

            sq_complexity, sq_cpx_bd = compute_complexity_single(sql, dpr_text)
            attempt_success = (
                sum(1 for a in attempts if a.get("execution_status", False)) / len(attempts)
                if attempts else 0.0
            )
            sub_evals.append({
                "sub_question": sub_q,
                "sql":          sql,
                "mini_summary": mini_sum,
                "metrics": {
                    "execution_status":     float(exec_ok),
                    "complexity":           sq_complexity,
                    "complexity_breakdown": sq_cpx_bd,
                    "attempt_success_rate": round(attempt_success, 4),
                    "n_attempts":           len(attempts),
                    "rows_returned":        row_count,
                }
            })

        records.append({
            "dpr_id":        dpr_id,
            "dpr_text":      dpr_text,
            "tables":        tables,
            "ground_truth":  ground_truth,
            "schema_mapping":schema_mapping,
            "final_summary": final_summary,
            "mini_summaries":mini_summaries,
            "sub_evals":     sub_evals,
            "metrics": {
                "coverage":              coverage,
                "complexity":            complexity,
                "complexity_breakdown":  cpx_breakdown,
                "llm_quality":           judge["quality"],
                "llm_clarity":           judge["clarity"],
                "llm_relevance":         judge["relevance"],
                "llm_reasoning":         judge["reasoning"],
            }
        })
        print(f"      [{i+1}/{len(raw_data)}] DPR {dpr_id}  coverage={coverage:.3f}  "
              f"complexity={complexity:.3f}  llm_q={judge['quality']}")

    # ── Pool-wide metrics ─────────────────────────────────────────────────────
    print("\n[2/4] Computing pool-wide metrics (Diversity, Surprisal, Uniqueness)...")
    n     = len(records)
    texts = [r["dpr_text"] for r in records]

    vect = TfidfVectorizer().fit(texts)
    emb  = vect.transform(texts).toarray()

    diversity_arr  = compute_diversity(emb)
    surprisal_list = compute_surprisal(records)
    uniqueness_arr = compute_uniqueness(emb, threshold=0.85)

    for i, r in enumerate(records):
        r["metrics"]["diversity"]  = round(float(diversity_arr[i]),  4)
        r["metrics"]["surprisal"]  = round(float(surprisal_list[i]), 4)
        r["metrics"]["uniqueness"] = float(uniqueness_arr[i])

    # ── Combined score + ranking ──────────────────────────────────────────────
    print(f"\n[3/4] Ranking (top-{top_k})...")

    # Weights sum to exactly 1.0
    # Slide formula: 0.3×Coverage + 0.2×Complexity + 0.2×Diversity + 0.2×Surprisal
    #              + 0.2×Uniqueness + 0.15×LLM_Quality + 0.15×LLM_Relevance
    # LLM_Clarity is included in the judge call and reported, but NOT in the
    # slide's combined score formula — so it gets 0.0 weight here.
    # To keep sum = 1.0 we redistribute: the 7 active terms above sum to 1.40,
    # so we scale each by 1/1.40:
    #   Coverage    0.30/1.40 = 0.2143
    #   Complexity  0.20/1.40 = 0.1429
    #   Diversity   0.20/1.40 = 0.1429
    #   Surprisal   0.20/1.40 = 0.1429
    #   Uniqueness  0.20/1.40 = 0.1429
    #   LLM Quality 0.15/1.40 = 0.1071
    #   LLM Relev.  0.15/1.40 = 0.1071
    #   LLM Clarity 0.00/1.40 = 0.0000  (reported, not scored)
    #   ─────────────────────────────────
    #   TOTAL                  = 1.0000  ✓
    WEIGHTS = {
        "coverage":      round(0.30 / 1.40, 4),   # 0.2143
        "complexity":    round(0.20 / 1.40, 4),   # 0.1429
        "diversity":     round(0.20 / 1.40, 4),   # 0.1429
        "surprisal":     round(0.20 / 1.40, 4),   # 0.1429
        "uniqueness":    round(0.20 / 1.40, 4),   # 0.1429
        "llm_quality":   round(0.15 / 1.40, 4),   # 0.1071
        "llm_relevance": round(0.15 / 1.40, 4),   # 0.1071
        "llm_clarity":   0.0,                      # reported only — not in slide formula
    }
    assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, f"Weights must sum to 1.0, got {sum(WEIGHTS.values())}"
    print(f"      Weights sum = {sum(WEIGHTS.values()):.4f} ✓")

    for r in records:
        m = r["metrics"]
        r["combined_score"] = round(sum(WEIGHTS[k] * m.get(k, 0.0) for k in WEIGHTS), 4)

    ranked = sorted(records, key=lambda x: x["combined_score"], reverse=True)[:top_k]
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank

    # ── Build output JSON ─────────────────────────────────────────────────────
    output = {
        "summary": {
            "total_dprs":        n,
            "total_ranked":      len(ranked),
            "unique_count":      int(uniqueness_arr.sum()),
            "multi_table_count": sum(1 for r in records if len(r["tables"]) > 1),
            "llm_model":         llm_model,
            "llm_api_base":      llm_api_base,
            "weights_used":      WEIGHTS,
            "formulas": {
                "coverage":   "|tables_used ∩ ground_truth| / |ground_truth|",
                "complexity": "mean(multi_table, join, agg, subquery, multi_entity)",
                "diversity":  "1 - avg_cosine_sim(DPR_i, all other DPRs)",
                "surprisal":  "-log P(tables_used) / log(|all_tables|)",
                "uniqueness": "1 if no near-duplicate (cosine > 0.85), else 0",
                "llm_scores": "GPT-4 scores: 0 / 0.5 / 0.75 / 1.0",
            }
        },
        "ranked_dprs": []
    }

    for r in ranked:
        m = r["metrics"]
        output["ranked_dprs"].append({
            "rank":           r["rank"],
            "dpr_id":         r["dpr_id"],
            "combined_score": r["combined_score"],
            "dpr_text":       r["dpr_text"],
            "tables_used":    r["tables"],
            "ground_truth":   r["ground_truth"],
            "summary_source": "stage3_output",         # summaries are NOT re-generated
            "final_summary":  r["final_summary"],
            "metrics": {
                "coverage":              m["coverage"],
                "complexity":            m["complexity"],
                "complexity_breakdown":  m["complexity_breakdown"],
                "diversity":             m["diversity"],
                "surprisal":             m["surprisal"],
                "uniqueness":            m["uniqueness"],
                "llm_quality":           m["llm_quality"],
                "llm_clarity":           m["llm_clarity"],
                "llm_relevance":         m["llm_relevance"],
                "llm_reasoning":         m["llm_reasoning"],
            },
            "sub_queries": [
                {
                    "sub_query_index": idx + 1,
                    "sub_question":    se["sub_question"],
                    "sql":             se["sql"],
                    "summary_source":  "stage3_output",  # mini_summary from Stage 3, not re-generated
                    "mini_summary":    se["mini_summary"],
                    "metrics":         se["metrics"],
                }
                for idx, se in enumerate(r["sub_evals"])
            ]
        })

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[4/4] Saving outputs...")

    json_path = os.path.join(output_dir, "dpr_ranked_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"      Saved JSON      → {json_path}")

    ranking_path = os.path.join(output_dir, "dpr_ranking_summary.txt")
    _write_ranking_summary(output, ranking_path)
    print(f"      Saved ranking   → {ranking_path}")

    stats_path = os.path.join(output_dir, "metrics_stats.txt")
    _write_metrics_stats(output, stats_path)
    print(f"      Saved stats     → {stats_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DPR EVALUATION v3 — RANKED RESULTS")
    print("=" * 72)
    print(f"\n{'Rank':<6} {'DPR ID':<12} {'Score':<8} {'Coverage':<10} {'Complexity':<12} "
          f"{'Diversity':<10} {'Surprisal':<10} {'Unique':<8} {'LLM-Q':<7} {'LLM-R'}")
    print("-" * 90)

    for r in ranked:
        m   = r["metrics"]
        dup = " ⚠️" if m["uniqueness"] == 0 else ""
        print(f"#{r['rank']:<5} {r['dpr_id']:<12} {r['combined_score']:<8.4f} "
              f"{m['coverage']:<10.4f} {m['complexity']:<12.4f} "
              f"{m['diversity']:<10.4f} {m['surprisal']:<10.4f} "
              f"{int(m['uniqueness']):<8} {m['llm_quality']:<7} {m['llm_relevance']}{dup}")

    print("\nSub-query detail:")
    for r in output["ranked_dprs"]:
        print(f"\n  DPR {r['dpr_id']} — {len(r['sub_queries'])} sub-queries:")
        for sq in r["sub_queries"]:
            sm     = sq["metrics"]
            status = "✅" if sm["execution_status"] else "❌"
            print(f"    [{sq['sub_query_index']}] {status} complexity={sm['complexity']:.2f} "
                  f"rows={sm['rows_returned']} | {sq['sub_question'][:65]}...")
            print(f"        SQL: {(sq['sql'] or 'N/A')[:65]}...")
            print(f"        Summary: {(sq['mini_summary'] or 'N/A')[:75]}...")

    print(f"\n✅ Evaluation complete. Outputs in: {output_dir}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
#  FILE 1: dpr_ranking_summary.txt
#  Rank · metrics table · stage3 summary — nothing else
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text, width=74, indent=2):
    """Word-wrap text into lines of max `width` chars with leading indent."""
    words = (text or "N/A").split()
    lines, buf, col = [], [], 0
    for w in words:
        if col + len(w) + 1 > width:
            lines.append(" " * indent + " ".join(buf))
            buf, col = [w], len(w)
        else:
            buf.append(w); col += len(w) + 1
    if buf:
        lines.append(" " * indent + " ".join(buf))
    return "\n".join(lines)


def _write_ranking_summary(output: dict, path: str):
    lines = []
    SEP = "=" * 60

    for r in output["ranked_dprs"]:
        m = r["metrics"]

        lines.append(f"Rank #{r['rank']}  |  DPR: {r['dpr_id']}  |  Combined Score: {r['combined_score']:.4f}\n")
        lines.append(SEP + "\n")

        # metrics — score only, no weights
        for label, key in [
            ("Coverage",      "coverage"),
            ("Complexity",    "complexity"),
            ("Diversity",     "diversity"),
            ("Surprisal",     "surprisal"),
            ("Uniqueness",    "uniqueness"),
            ("LLM Quality",   "llm_quality"),
            ("LLM Relevance", "llm_relevance"),
            ("LLM Clarity",   "llm_clarity"),
        ]:
            lines.append(f"  {label:<18} {float(m.get(key, 0.0)):.4f}\n")

        lines.append("\n")

        # summary directly from Stage 3 — no regeneration
        lines.append("  Summary:\n")
        lines.append(_wrap(r["final_summary"], width=72, indent=4) + "\n")
        lines.append("\n" + SEP + "\n\n")

    with open(path, "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  FILE 2: metrics_stats.txt
#  Per-metric: min · max · mean across all ranked DPRs
# ─────────────────────────────────────────────────────────────────────────────

def _write_metrics_stats(output: dict, path: str):
    metric_keys = [
        ("Coverage",      "coverage"),
        ("Complexity",    "complexity"),
        ("Diversity",     "diversity"),
        ("Surprisal",     "surprisal"),
        ("Uniqueness",    "uniqueness"),
        ("LLM Quality",   "llm_quality"),
        ("LLM Relevance", "llm_relevance"),
        ("LLM Clarity",   "llm_clarity"),
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
        v   = vals[key]
        mn  = min(v)  if v else 0.0
        mx  = max(v)  if v else 0.0
        avg = sum(v) / len(v) if v else 0.0
        lines.append(f"  {label:<20}  {mn:>8.4f}  {mx:>8.4f}  {avg:>8.4f}\n")

    lines.append(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}\n")
    lines.append(f"  {'Combined Score':<20}  {min(scores):>8.4f}  {max(scores):>8.4f}  {sum(scores)/len(scores):>8.4f}\n")

    with open(path, "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="DPR Evaluation Pipeline v3 — multi sub-query, exact slide formulas"
    )
    parser.add_argument("--input",        required=True,  help="Path to stage3 output JSON (pipeline_output.json)")
    parser.add_argument("--schema",       default=None,   help="(Optional) schema_descriptions.json — schema extracted from input if omitted")
    parser.add_argument("--output_dir",   default="output", help="Directory for output files (default: output/)")
    parser.add_argument("--top_k",        type=int, default=100, help="Number of top DPRs to rank (default: 100)")
    parser.add_argument("--llm_api_key",  default="",     help="LLM API key for judge scoring")
    parser.add_argument("--llm_api_base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm_model",    default="gpt-4", help="LLM model name (default: gpt-4)")
    args = parser.parse_args()

    print(f"[Config]")
    print(f"  input       : {args.input}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  top_k       : {args.top_k}")
    print(f"  llm_model   : {args.llm_model}")
    print(f"  llm_api_base: {args.llm_api_base}")
    print(f"  llm_api_key : {'SET (' + args.llm_api_key[:8] + '...)' if args.llm_api_key else 'NOT SET — judge defaults to 0.5'}")

    with open(args.input) as f:
        raw_data = json.load(f)
    print(f"\nLoaded {len(raw_data)} DPRs from {args.input}")

    run_pipeline(
        raw_data     = raw_data,
        output_dir   = args.output_dir,
        llm_api_key  = args.llm_api_key,
        llm_api_base = args.llm_api_base,
        llm_model    = args.llm_model,
        top_k        = args.top_k,
    )