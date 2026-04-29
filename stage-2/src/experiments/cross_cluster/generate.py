"""
Experiment: Cross-cluster DPR discovery via pairwise LLM reasoning.

Inspired by ClusterLLM (Zhang et al., 2023) — uses LLM pairwise judgements
to find non-obvious relationships between clusters that embedding similarity
cannot capture (e.g. weather + traffic sharing time and location dimensions).

Pipeline:
  1. Load BERTopic clusters + pre-computed embeddings
  2. Compute cluster centroids and rank all pairs by cosine distance
  3. Evaluate only the top-K most dissimilar pairs (where surprises live)
  4. Ask LLM: "Could these two clusters be analytically combined to answer
     an interesting cross-domain question neither could answer alone?"
  5. For pairs where LLM says YES, generate a cross-cluster DPR
  6. Save results to output_dir

Usage:
    source venv/bin/activate
    python src/experiments/cross_cluster/generate.py \
        --clusters_path data/experiments/umap/clusters.json \
        --embeddings_path data/input_100tables/embeddings.json \
        --output_dir data/experiments/cross_cluster \
        --top_k 20
"""

import json
import argparse
import os
import time
import re
import itertools
import threading
import numpy as np
from dotenv import load_dotenv
import dspy

load_dotenv()


# ── DSPy signatures ──────────────────────────────────────────────────────────

class CrossClusterCheck(dspy.Signature):
    """
    You are an expert data analyst with a talent for finding surprising connections
    between seemingly unrelated datasets.

    You are given two clusters of tables that come from completely different domains —
    they will never appear similar by topic or keyword. Your job is to think creatively
    about how they might be analytically related.

    Ask yourself: "Could these clusters be related?"

    Think beyond surface similarity. Consider:
    - Shared hidden dimensions: time, geography, population, weather, economics
    - Causal or correlational links: does one phenomenon drive or explain the other?
    - A third factor that influences both independently
    - Policy or decision contexts where both datasets would be needed together
    - Surprising patterns that only emerge when the two are combined

    Rate the relationship strength on a 1-5 scale:
    1 = No plausible connection
    2 = Weak or speculative connection
    3 = Possible connection with a plausible analytical question
    4 = Strong latent connection, combining them would reveal something non-obvious
    5 = Highly surprising and valuable cross-domain insight

    Output:
    - relation_score: integer 1-5
    - connection_type: the type of latent link (e.g. "shared geography", "causal",
                       "temporal co-occurrence", "policy context", "confounding factor")
                       or "none" if score is 1
    - reasoning: 2-3 sentences explaining the potential connection or why none exists
    - cross_domain_question: one concrete analytical question that would require data
                             from both clusters (empty string if score <= 2)
    """

    cluster_a_summary: str = dspy.InputField(
        desc="Summary of cluster A: topic and table descriptions"
    )
    cluster_b_summary: str = dspy.InputField(
        desc="Summary of cluster B: topic and table descriptions"
    )

    relation_score: int = dspy.OutputField(desc="Integer 1-5 rating of cross-domain relationship strength")
    connection_type: str = dspy.OutputField(desc="Type of latent connection, or 'none'")
    reasoning: str = dspy.OutputField(desc="2-3 sentence explanation of the connection or lack thereof")
    cross_domain_question: str = dspy.OutputField(
        desc="Concrete analytical question requiring both clusters (empty string if score <= 2)"
    )


class CrossClusterDPR(dspy.Signature):
    """
    You are a Data Product Request Generator.

    A Data Product is a self-contained, reusable data asset that delivers
    specific value for data-driven use cases.

    You are given two clusters of tables and a cross-domain analytical question
    that requires data from both clusters.

    Task:
    Write one data product request that captures the combined data needs of both
    clusters, framed around the cross-domain question.

    Instructions:
    - The request must require data from BOTH clusters to answer.
    - It should not be answerable from either cluster alone.
    - Use a clear, professional tone suitable for real-world user requests.
    - Be specific about what data and analysis is needed.
    - Do not use the word "analysis" as a standalone filler.

    Output format:
    Return only the final data product request as plain text.
    """

    cluster_a_summary: str = dspy.InputField(desc="Summary of cluster A")
    cluster_b_summary: str = dspy.InputField(desc="Summary of cluster B")
    cross_domain_question: str = dspy.InputField(
        desc="The cross-domain analytical question that requires both clusters"
    )

    data_product_request: str = dspy.OutputField(desc="The cross-cluster DPR")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_cluster_summary(cluster_id, tables, topic_name=None, max_tables=3):
    """Build a concise text summary of a cluster for LLM input."""
    lines = [f"Cluster {cluster_id} ({len(tables)} tables)"]
    if topic_name:
        # Strip the numeric prefix from topic name e.g. "0_country_international_..."
        clean = "_".join(topic_name.split("_")[1:]).replace("_", " ")
        lines.append(f"Topic: {clean}")
    for t in tables[:max_tables]:
        cols = ", ".join(t.get("columns", [])[:5])
        desc = t.get("description", "")[:80]
        lines.append(f"  - {desc} | cols: {cols}")
    return "\n".join(lines)


def load_embeddings(path):
    """Load pre-computed embeddings JSON and return table_id -> embedding dict."""
    with open(path) as f:
        data = json.load(f)
    return {row["table_id"]: np.array(row["embedding"]) for row in data}


def compute_centroids(clusters, embeddings):
    """Compute mean embedding centroid for each cluster."""
    centroids = {}
    for cid, tables in clusters.items():
        vecs = [embeddings[t["table_id"]] for t in tables if t["table_id"] in embeddings]
        if vecs:
            centroids[cid] = np.mean(vecs, axis=0)
    return centroids


def rank_pairs_by_distance(cluster_ids, centroids):
    """
    Return all cluster pairs sorted by cosine distance descending
    (most dissimilar first — these are the surprise candidates).
    """
    pairs = list(itertools.combinations(cluster_ids, 2))
    scored = []
    for id_a, id_b in pairs:
        if id_a not in centroids or id_b not in centroids:
            continue
        a, b = centroids[id_a], centroids[id_b]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        cos_dist = 1.0 - cos_sim
        scored.append((id_a, id_b, float(cos_dist)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def setup_llm(model=None, api_base=None, api_key=None):
    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM model not specified. Set LLM_MODEL env var or pass --model.")
    api_base = api_base or os.getenv("LLM_API_BASE")
    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    kwargs = {"model": model, "max_tokens": 3000, "cache": True}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    dspy.configure(lm=dspy.LM(**kwargs))
    return model


def start_heartbeat(label, counter, total, interval=30):
    """Print progress every `interval` seconds. Returns a stop event."""
    stop = threading.Event()
    start_time = time.time()

    def _beat():
        while not stop.wait(interval):
            elapsed = int(time.time() - start_time)
            done = counter[0]
            pct = 100 * done / total if total else 0
            print(
                f"  [heartbeat] {label} — {done}/{total} pairs ({pct:.1f}%) "
                f"elapsed {elapsed//60}m{elapsed%60:02d}s",
                flush=True,
            )

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    return stop


def call_with_retry(fn, *args, max_retries=5, sleep_between=60, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(sleep_between)
            return result
        except Exception as e:
            err = str(e)
            is_rate_limit = "rate_limit_exceeded" in err or "RateLimitError" in err
            is_parse_fail = "json_validate_failed" in err or "AdapterParseError" in err
            is_network = "nodename nor servname" in err or "ConnectError" in err or "InternalServerError" in err
            if is_rate_limit or is_parse_fail or is_network:
                match = re.search(r"try again in ([\d.]+)s", err)
                wait = float(match.group(1)) + 2.0 if match else 30.0
                reason = "Rate limit" if is_rate_limit else ("Network error" if is_network else "Parse error (likely truncation)")
                print(f"  {reason} (attempt {attempt}/{max_retries}). Waiting {wait:.0f}s...")
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            raise


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load clusters
    with open(args.clusters_path) as f:
        clusters_raw = json.load(f)

    # Exclude noise cluster (-1)
    clusters = {k: v for k, v in clusters_raw.items() if k != "-1"}
    print(f"Loaded {len(clusters)} clusters (excluding noise)")

    # Load topic names if summary available
    topic_names = {}
    summary_path = args.clusters_path.replace("clusters.json", "clusters_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for t in summary.get("topic_info", []):
            topic_names[str(t["Topic"])] = t.get("Name", "")

    # Setup LLM
    model = setup_llm(args.model, args.api_base, args.api_key)
    print(f"Using model: {model}")

    # Rank pairs by centroid cosine distance (most dissimilar first)
    cluster_ids = list(clusters.keys())
    if args.embeddings_path:
        print(f"Loading embeddings from {args.embeddings_path}...")
        embeddings = load_embeddings(args.embeddings_path)
        centroids = compute_centroids(clusters, embeddings)
        ranked = rank_pairs_by_distance(cluster_ids, centroids)
        pairs = [(a, b) for a, b, _ in ranked]
        print(f"Ranked {len(pairs)} pairs by cosine distance (most dissimilar first)")
        if args.top_k:
            pairs = pairs[:args.top_k]
            print(f"Evaluating top {args.top_k} most dissimilar pairs")
        for id_a, id_b, dist in ranked[:len(pairs)]:
            print(f"  {id_a} x {id_b}  dist={dist:.4f}")
    else:
        pairs = list(itertools.combinations(cluster_ids, 2))
        print(f"No embeddings path given — evaluating all {len(pairs)} pairs (unranked)")

    print()

    check_fn = dspy.ChainOfThought(CrossClusterCheck)
    dpr_fn = dspy.ChainOfThought(CrossClusterDPR)

    counter = [len(done_pairs)]  # start from already-done pairs
    stop_hb = start_heartbeat("Stage 2b cross-cluster", counter, len(pairs))

    # Load checkpoint if it exists (resume after network failures)
    pairs_path = os.path.join(args.output_dir, "pair_decisions.json")
    dprs_path = os.path.join(args.output_dir, "cross_cluster_dprs.jsonl")

    pair_results = []
    cross_cluster_dprs = []
    done_pairs = set()

    if os.path.exists(pairs_path):
        with open(pairs_path) as f:
            pair_results = json.load(f)
        done_pairs = {(r["cluster_a"], r["cluster_b"]) for r in pair_results}
        print(f"Resuming from checkpoint: {len(done_pairs)} pairs already done")

    if os.path.exists(dprs_path):
        with open(dprs_path) as f:
            cross_cluster_dprs = [json.loads(l) for l in f if l.strip()]

    for i, (id_a, id_b) in enumerate(pairs):
        if (id_a, id_b) in done_pairs:
            print(f"[{i+1}/{len(pairs)}] Cluster {id_a} x Cluster {id_b} ... skipped (done)")
            counter[0] += 1
            continue
        tables_a = clusters[id_a]
        tables_b = clusters[id_b]

        summary_a = build_cluster_summary(id_a, tables_a, topic_names.get(id_a))
        summary_b = build_cluster_summary(id_b, tables_b, topic_names.get(id_b))

        print(f"[{i+1}/{len(pairs)}] Cluster {id_a} x Cluster {id_b} ...", end=" ", flush=True)

        # Step 1: pairwise check
        check_result = call_with_retry(
            check_fn,
            cluster_a_summary=summary_a,
            cluster_b_summary=summary_b,
            sleep_between=args.sleep_between,
        )

        score = int(check_result.relation_score)
        connection_type = check_result.connection_type.strip()
        reasoning = check_result.reasoning.strip()
        question = check_result.cross_domain_question.strip()

        print(f"score={score} [{connection_type}]")
        print(f"  {reasoning}")
        if score >= 3 and question:
            print(f"  Q: {question}")

        pair_results.append({
            "cluster_a": id_a,
            "cluster_b": id_b,
            "relation_score": score,
            "connection_type": connection_type,
            "reasoning": reasoning,
            "cross_domain_question": question,
        })

        counter[0] += 1

        # Save checkpoint after every pair
        with open(pairs_path, "w") as f:
            json.dump(pair_results, f, indent=2)

        # Step 2: generate DPR for strong connections (score >= 3)
        if score >= 3 and question:
            print(f"  Generating cross-cluster DPR...", end=" ", flush=True)

            dpr_result = call_with_retry(
                dpr_fn,
                cluster_a_summary=summary_a,
                cluster_b_summary=summary_b,
                cross_domain_question=question,
                sleep_between=args.sleep_between,
            )

            dpr_text = dpr_result.data_product_request.strip()
            dpr_reasoning = getattr(dpr_result, "reasoning", "")
            print("done")

            entry = {
                "dpr_id": f"cross_{id_a}_{id_b}",
                "cluster_a": id_a,
                "cluster_b": id_b,
                "relation_score": score,
                "connection_type": connection_type,
                "cross_domain_question": question,
                "connection_reasoning": reasoning,
                "DPR": dpr_text,
                "reasoning": dpr_reasoning,
                "ground_truth": {
                    "table_uids": (
                        [t["table_id"] for t in tables_a] +
                        [t["table_id"] for t in tables_b]
                    )
                },
            }
            cross_cluster_dprs.append(entry)

            # Save DPR incrementally
            with open(dprs_path, "w") as f:
                for dpr in cross_cluster_dprs:
                    f.write(json.dumps(dpr) + "\n")

    stop_hb.set()

    # Summary
    strong_pairs = [r for r in pair_results if r.get("relation_score", 0) >= 3]
    print(f"\n{'='*60}")
    print(f"CROSS-CLUSTER RESULTS")
    print(f"{'='*60}")
    print(f"Total pairs evaluated: {len(pairs)}")
    print(f"Pairs with relation score >= 3: {len(strong_pairs)}")
    print(f"Cross-cluster DPRs generated: {len(cross_cluster_dprs)}")

    score_dist = {}
    for r in pair_results:
        s = r.get("relation_score", 0)
        score_dist[s] = score_dist.get(s, 0) + 1
    print(f"\nScore distribution: {dict(sorted(score_dist.items()))}")

    print(f"\nStrong pairs (score >= 3):")
    for r in sorted(strong_pairs, key=lambda x: x["relation_score"], reverse=True):
        print(f"  [{r['relation_score']}] Cluster {r['cluster_a']} x Cluster {r['cluster_b']} ({r['connection_type']})")
        print(f"    {r['reasoning']}")
        if r["cross_domain_question"]:
            print(f"    Q: {r['cross_domain_question']}")

    print(f"\nSaved pair decisions to {pairs_path}")
    print(f"Saved {len(cross_cluster_dprs)} cross-cluster DPRs to {dprs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-cluster DPR discovery via pairwise LLM reasoning"
    )
    parser.add_argument("--clusters_path", type=str, required=True,
                        help="Path to clusters.json from BERTopic")
    parser.add_argument("--output_dir", type=str, default="data/experiments/cross_cluster")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--sleep_between", type=float, default=60.0,
                        help="Seconds to sleep between LLM calls (for rate limiting)")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to pre-computed embeddings JSON (enables distance-ranked pair selection)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Only evaluate the top-K most dissimilar cluster pairs (requires --embeddings_path)")

    args = parser.parse_args()
    main(args)
