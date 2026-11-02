"""Shared evaluation protocol for every model.

A model gives a score to every (evaluated user, movie) pair. This module turns
the scores into top-10 lists over the full catalogue, never recommending a movie
the user has already rated, and computes Precision@10, Recall@10, NDCG@10 and
catalogue coverage.

Typical use:
    users, items = catalogue_index(train, validation, test)
    eval_set = evaluation_set(history, heldout, users, items)
    scores = ...  # array of shape (len(eval_set.user_ids), len(items))
    summary, per_user, top = evaluate(scores, eval_set)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import ndcg_score, precision_score, recall_score

RELEVANCE_THRESHOLD = 4
TOP_N = 10
SEED = 42
N_BOOTSTRAP = 2000


def catalogue_index(*frames):
    """Sorted user and movie ids over all given rating frames (the full filtered catalogue)."""
    everything = pd.concat(frames)
    return (pd.Index(np.sort(everything["UserID"].unique())),
            pd.Index(np.sort(everything["MovieID"].unique())))


def interaction_matrix(df, users, items, feedback="explicit"):
    """Users x items matrix. Explicit: the rating value. Implicit: 1 for ratings >=
    RELEVANCE_THRESHOLD, lower ratings dropped."""
    if feedback == "implicit":
        df = df[df["Rating"] >= RELEVANCE_THRESHOLD]
        values = np.ones(len(df), dtype=np.float32)
    else:
        values = df["Rating"].to_numpy(np.float32)
    rows = users.get_indexer(df["UserID"])
    cols = items.get_indexer(df["MovieID"])
    return sp.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))


@dataclass
class EvaluationSet:
    user_ids: np.ndarray
    rows: np.ndarray
    seen: np.ndarray
    relevant: np.ndarray
    item_popularity: np.ndarray
    n_cold: int


def evaluation_set(history, heldout, users, items):
    """Evaluated users are those with at least one relevant held-out movie and at least one
    rating in the history; a model cannot personalise for users without a history."""
    positives = heldout[heldout["Rating"] >= RELEVANCE_THRESHOLD]
    with_positives = positives["UserID"].unique()
    user_ids = np.intersect1d(with_positives, history["UserID"].unique())
    rows = users.get_indexer(user_ids)
    return EvaluationSet(
        user_ids=user_ids,
        rows=rows,
        seen=interaction_matrix(history, users, items)[rows].toarray() != 0,
        relevant=interaction_matrix(positives, users, items, "implicit")[rows].toarray() > 0,
        item_popularity=history["MovieID"].value_counts().reindex(items, fill_value=0).to_numpy(),
        n_cold=len(with_positives) - len(user_ids),
    )


def top_n(scores, seen, n=TOP_N):
    """Indices of the n highest-scoring movies per user, excluding already-rated movies.
    Ties are broken deterministically by movie position."""
    scores = np.where(seen, -np.inf, scores)
    top = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
    return np.take_along_axis(top, order, axis=1)


def per_user_metrics(top, relevant):
    """Precision@n, Recall@n (hits / all relevant movies of the user) and NDCG@n with binary gains."""
    n = top.shape[1]
    hits = np.take_along_axis(relevant, top, axis=1)
    n_relevant = relevant.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    ideal = np.cumsum(discounts)[np.minimum(n_relevant, n) - 1]
    return pd.DataFrame({
        "precision": hits.sum(axis=1) / n,
        "recall": hits.sum(axis=1) / n_relevant,
        "ndcg": (hits * discounts).sum(axis=1) / ideal,
    })


def evaluate(scores, eval_set):
    """Rank, then return the summary row, the per-user metrics and the top-n lists."""
    top = top_n(scores, eval_set.seen)
    per_user = per_user_metrics(top, eval_set.relevant).set_index(eval_set.user_ids)
    summary = {
        f"precision@{TOP_N}": per_user["precision"].mean(),
        f"recall@{TOP_N}": per_user["recall"].mean(),
        f"ndcg@{TOP_N}": per_user["ndcg"].mean(),
        "coverage": len(np.unique(top)) / scores.shape[1],
        "mean_rec_popularity": eval_set.item_popularity[top].mean(),
        "n_users": len(per_user),
        "n_cold_users_excluded": eval_set.n_cold,
    }
    return summary, per_user, top


def paired_bootstrap(diff):
    """95% bootstrap confidence interval of the mean of per-user differences."""
    rng = np.random.default_rng(SEED)
    boot = diff[rng.integers(0, len(diff), (N_BOOTSTRAP, len(diff)))].mean(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def check_against_sklearn(top, relevant, n_users=200):
    """Compare our metrics with scikit-learn on a random sample of users.
    Returns the largest absolute difference per metric (should be ~0)."""
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(top), size=min(n_users, len(top)), replace=False)
    ours = per_user_metrics(top[sample], relevant[sample])
    diffs = {"precision": [], "recall": [], "ndcg": []}
    for i, user in enumerate(sample):
        recommended = np.zeros(relevant.shape[1], dtype=bool)
        recommended[top[user]] = True
        ranking_scores = np.zeros(relevant.shape[1])
        ranking_scores[top[user]] = np.arange(top.shape[1], 0, -1)
        reference = {
            "precision": precision_score(relevant[user], recommended),
            "recall": recall_score(relevant[user], recommended),
            "ndcg": ndcg_score(relevant[user][None, :].astype(float), ranking_scores[None, :], k=TOP_N),
        }
        for metric, value in reference.items():
            diffs[metric].append(abs(ours[metric].iloc[i] - value))
    return pd.DataFrame({"metric": list(diffs), "max_abs_diff": [max(v) for v in diffs.values()],
                         "n_users": len(sample)})
