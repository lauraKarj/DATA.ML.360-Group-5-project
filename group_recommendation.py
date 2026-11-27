"""Group recommendations (Part 3, Track C): similar groups of four and the Least Misery strategy.

Every member gets item-kNN scores (the Part 1 model: implicit feedback, cosine, k = 800) fitted on all
ratings before the test cut, normalised by the member's highest score. A strategy turns the members'
normalised scores into one group score per movie, and the group list is the top 10 of the movies that
no member has rated. A new strategy is one entry in STRATEGIES; it is then evaluated on the same groups
and compared pairwise with the others.

From run_all.py:
    import group_recommendation
    group_recommendation.run(training_set, validation_set, test_set)

Standalone:
    python group_recommendation.py --data-dir ml-1m --out-dir results
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

import item_knn
from evaluation import SEED, TOP_N, catalogue_index, interaction_matrix, paired_bootstrap

GROUP_SIZE = 4
N_GROUPS = 100
MIN_POSITIVES = 10
KNN = ("cosine", 800)
METRICS = ["group_satisfaction", "disagreement", "min_satisfaction"]


def least_misery(member_scores):
    """Group score of a movie: the lowest normalised score any member gives it."""
    return member_scores.min(axis=0)

def average(member_scores):
    """Group score of a movie: mean of the members' normalised scores."""
    return member_scores.mean(axis=0)


STRATEGIES = {"Least Misery": least_misery,
              "Average": average,
}


def similar_groups(X, rng):
    """Greedy similar groups: a random seed user, then repeatedly the remaining user with the highest mean
    cosine similarity (over liked movies) to the current members. Users with fewer than MIN_POSITIVES
    liked movies are skipped and no user is in two groups. Returns member rows and the row-normalised X."""
    counts = X.getnnz(axis=1)
    X_norm = sp.diags(1 / np.sqrt(np.maximum(counts, 1))) @ X
    available = counts >= MIN_POSITIVES
    groups = []
    for _ in range(N_GROUPS):
        members = [rng.choice(np.flatnonzero(available))]
        available[members[0]] = False
        similarity_sum = np.zeros(X.shape[0])
        while len(members) < GROUP_SIZE:
            similarity_sum += (X_norm @ X_norm[members[-1]].T).toarray().ravel()
            candidates = np.flatnonzero(available)
            members.append(candidates[np.argmax(similarity_sum[candidates])])
            available[members[-1]] = False
        groups.append(members)
    return groups, X_norm


def mean_pairwise_similarity(X_norm, members):
    S = (X_norm[members] @ X_norm[members].T).toarray()
    return S[np.triu_indices(len(members), 1)].mean()


def group_top(member_scores, seen, strategy, n=TOP_N):
    """The n movies with the highest group score, ties broken by the members' mean score;
    movies rated by any member are excluded."""
    group = np.where(seen, -np.inf, strategy(member_scores))
    return np.lexsort((-member_scores.mean(axis=0), -group))[:n]


def satisfaction(member_scores, seen, top):
    """sat(u) = sum of u's scores over the group list / sum of u's scores over u's own top-n
    of the same candidate movies (1 = the group list is as good for u as u's personal list)."""
    candidates = np.where(seen, -np.inf, member_scores)
    own = -np.sort(-candidates, axis=1)[:, :len(top)].sum(axis=1)
    return member_scores[:, top].sum(axis=1) / own


def group_metrics(sat):
    return {"group_satisfaction": sat.mean(), "disagreement": sat.max() - sat.min(),
            "min_satisfaction": sat.min()}


def summarise(per_group):
    """Mean of each metric over groups with a 95% bootstrap CI (groups resampled)."""
    rows = []
    for strategy, frame in per_group.groupby("strategy", sort=False):
        row = {"strategy": strategy, "n_groups": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy()
            row[metric] = values.mean()
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = paired_bootstrap(values)
        rows.append(row)
    return pd.DataFrame(rows)


def compare(per_group, a, b):
    """Paired per-group differences a - b for each metric: mean, bootstrap CI and Wilcoxon p."""
    left = per_group[per_group["strategy"] == a].set_index("group")
    right = per_group[per_group["strategy"] == b].set_index("group")
    rows = []
    for metric in METRICS:
        d = (left[metric] - right[metric]).to_numpy()
        ci_low, ci_high = paired_bootstrap(d)
        rows.append({"strategy": a, "reference": b, "metric": metric, "mean_diff": d.mean(),
                     "ci_low": ci_low, "ci_high": ci_high,
                     "wilcoxon_p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0})
    return rows


def run(train, validation, test, out_dir="results"):
    """1) similar groups from the history (train + validation), 2) member item-kNN scores,
    3) each strategy's group top-10 and its satisfaction, disagreement and minimum satisfaction."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    users, items = catalogue_index(train, validation, test)
    history = pd.concat([train, validation])
    X = interaction_matrix(history, users, items, "implicit")
    rated = interaction_matrix(history, users, items)
    S_k = item_knn.prune(item_knn.item_similarity(X, KNN[0]), KNN[1])

    rng = np.random.default_rng(SEED)
    groups, X_norm = similar_groups(X, rng)
    eligible = np.flatnonzero(X.getnnz(axis=1) >= MIN_POSITIVES)
    random_pairs = rng.choice(eligible, (5000, 2))
    random_similarity = np.asarray(X_norm[random_pairs[:, 0]].multiply(X_norm[random_pairs[:, 1]])
                                   .sum(axis=1)).mean()

    group_rows, rows = [], []
    for g, members in enumerate(groups):
        seen = rated[members].getnnz(axis=0) > 0
        scores = np.asarray(item_knn.knn_scores(X[members], S_k))
        scores = scores / np.where(seen, 0, scores).max(axis=1, keepdims=True)
        group_rows.append({"group": g, "members": " ".join(map(str, users[members])),
                           "mean_pairwise_similarity": mean_pairwise_similarity(X_norm, members)})
        for name, strategy in STRATEGIES.items():
            top = group_top(scores, seen, strategy)
            sat = satisfaction(scores, seen, top)
            rows.append({"group": g, "strategy": name, **group_metrics(sat),
                         "member_satisfaction": " ".join(f"{s:.4f}" for s in sat),
                         "movies": " ".join(map(str, items[top]))})

    group_table = pd.DataFrame(group_rows)
    per_group = pd.DataFrame(rows)
    summary = summarise(per_group)
    group_table.to_csv(out_dir / "groups.csv", index=False)
    per_group.to_csv(out_dir / "group_per_group.csv", index=False)
    summary.to_csv(out_dir / "group_strategies.csv", index=False)
    print(f"{N_GROUPS} similar groups of {GROUP_SIZE}: mean pairwise cosine similarity "
          f"{group_table['mean_pairwise_similarity'].mean():.3f} (random user pairs: {random_similarity:.3f})")
    print(summary.round(4).to_string(index=False))

    pairs = [row for a, b in combinations(STRATEGIES, 2) for row in compare(per_group, a, b)]
    if pairs:
        pairs = pd.DataFrame(pairs)
        pairs.to_csv(out_dir / "group_strategy_pairs.csv", index=False)
        print(pairs.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where the CSV results are written")
    args = parser.parse_args()
    run(*item_knn._load_split(args.data_dir), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
