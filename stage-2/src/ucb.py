"""
UCB (Upper Confidence Bound) cluster selection.

Implements the UCB algorithm exactly as described in:
"The Multi-Armed Bandit Problem — A Beginner-Friendly Guide"
Saankhya Mondal, Towards Data Science, Dec 2024.
https://towardsdatascience.com/the-multi-armed-bandit-problem-a-beginner-friendly-guide-2293ce7d8da8/

Formula (derived from Hoeffding's inequality):
    UCB(c) = avg_score(c) + sqrt(2 * ln(total_trials + 1) / (n_visits(c) + 1e-5))

Bootstrap phase (article's choose_restaurant logic):
    if total_trials < n_clusters:
        visit cluster at index total_trials (each cluster once, in order)

This module has NO LLM code — pure selection logic only.
Imported by online_iterative_pipeline.py.
"""

import math
import random
from typing import Dict, List


def compute_ucb(
    cluster_id: str,
    total_trials: int,
    visit_counts: Dict[str, int],
    score_sums: Dict[str, float],
) -> float:
    """
    Compute UCB score for one cluster.

    Mirrors the article's UCB.choose_restaurant() formula exactly:
        avg_score + sqrt(2 * ln(total_trials + 1) / (n_visits + 1e-5))

    The +1 inside ln avoids ln(0) on the first UCB call.
    The +1e-5 denominator guard avoids division by zero (article uses this
    instead of an inf sentinel for unvisited clusters).

    Args:
        cluster_id:   cluster to score
        total_trials: total successful picks so far across all clusters
        visit_counts: dict of cluster_id -> number of times visited
        score_sums:   dict of cluster_id -> sum of relevance scores received

    Returns:
        UCB score as float. Higher = more worth selecting next.
    """
    n        = visit_counts.get(cluster_id, 0)
    avg      = score_sums.get(cluster_id, 0.0) / (n + 1e-5)
    bonus    = math.sqrt(2 * math.log(total_trials + 1) / (n + 1e-5))
    return avg + bonus


def select_cluster(
    cluster_ids: List[str],
    total_trials: int,
    visit_counts: Dict[str, int],
    score_sums: Dict[str, float],
    rng: random.Random,
) -> str:
    """
    Select the next cluster to visit using the article's UCB logic.

    Bootstrap phase (mirrors article's "First, visit each restaurant at least once"):
        If total_trials < n_clusters, return cluster_ids[total_trials].
        This guarantees every cluster is visited exactly once before UCB
        takes over, giving the formula enough data to work with.

    UCB phase:
        Compute UCB score for every cluster and return the highest.
        Ties broken randomly (rng ensures reproducibility).

    Args:
        cluster_ids:  ordered list of all cluster IDs in the pool
        total_trials: total successful picks so far
        visit_counts: dict cluster_id -> visit count
        score_sums:   dict cluster_id -> cumulative score
        rng:          seeded Random instance for tie-breaking

    Returns:
        Selected cluster_id as string.
    """
    n_clusters = len(cluster_ids)

    # Bootstrap: visit each cluster once in order before UCB kicks in
    if total_trials < n_clusters:
        return cluster_ids[total_trials]

    # UCB phase: pick cluster with highest UCB score
    scores      = {cid: compute_ucb(cid, total_trials, visit_counts, score_sums)
                   for cid in cluster_ids}
    max_score   = max(scores.values())
    top_clusters = [cid for cid, s in scores.items() if s == max_score]

    # Break ties randomly for reproducibility
    return rng.choice(top_clusters)