"""Item-based kNN baseline (Part 1).

Tunes the neighbourhood size k and the similarity measure on the validation
period only, refits the selected configuration on train + validation and
evaluates it once on the test period. Then compares kNN with Most Popular user
by user. Ranking and metrics come from evaluation.py.

From run_all.py, after the split:
    import item_knn
    item_knn.run(training_set, validation_set, test_set)

Standalone:
    python item_knn.py --data-dir ml-1m --out-dir results --fig-dir figures
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import most_popular
from evaluation import (TOP_N, catalogue_index, check_against_sklearn, evaluate, evaluation_set,
                        interaction_matrix, paired_bootstrap)

K_GRID = [5, 10, 20, 50, 100, 200, 400, 800, None]
SIMILARITIES = {
    "implicit": ["cosine", "jaccard"],
    "explicit": ["cosine", "adjusted_cosine", "pearson", "jaccard"],
}
POPULARITY_MODELS = {"MostPop-all_time": most_popular.all_time_scores,
                     "MostPop-temporal": most_popular.temporal_scores}
METRIC = f"ndcg@{TOP_N}"


def _means(X, axis):
    """Mean of the stored (rated) entries along an axis, 0 where nothing is stored."""
    sums = np.asarray(X.sum(axis=axis)).ravel()
    counts = X.getnnz(axis=axis)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def item_similarity(X, measure):
    """Dense items x items similarity matrix with a zero diagonal.

    Vectors are compared over all users; a missing rating counts as 0 after any
    centring, so movies with few co-raters get a low similarity.
    """
    if measure == "jaccard":
        B = (X != 0).astype(np.float32)
        intersection = (B.T @ B).toarray()
        size = np.diag(intersection).copy()
        union = size[:, None] + size[None, :] - intersection
        S = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    else:
        C = X.tocoo(copy=True)
        if measure == "adjusted_cosine":
            C.data -= _means(X, axis=1)[C.row]
        elif measure == "pearson":
            C.data -= _means(X, axis=0)[C.col]
        C = C.tocsr()
        dot = (C.T @ C).toarray()
        norm = np.sqrt(np.diag(dot))
        denominator = np.outer(norm, norm)
        S = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    np.fill_diagonal(S, 0)
    return S


def prune(S, k):
    """Keep the k most similar movies of each target movie (row); drop non-positive similarities.
    k = None keeps every positively similar movie (no pruning)."""
    S = np.maximum(S, 0)
    if k is not None and k < S.shape[1]:
        drop = np.argpartition(-S, k, axis=1)[:, k:]
        np.put_along_axis(S, drop, 0, axis=1)
    return S


def history_weights(X, feedback):
    """Weight of each history movie in the score: 1 for an implicit positive,
    the user-mean-centred rating for explicit feedback."""
    if feedback == "implicit":
        return X
    W = X.tocoo(copy=True)
    W.data -= _means(X, axis=1)[W.row]
    return W.tocsr()


def knn_scores(W, S_k):
    """score(u, i) = sum over the neighbours j of i of S_k[i, j] * W[u, j]."""
    return W @ S_k.T


def run_period(history, heldout, users, items, configs):
    """Evaluate the Most Popular variants (most_popular.py) and the given kNN configurations on
    one held-out period. Returns the summary table, the per-user metrics and the top-n lists."""
    eval_set = evaluation_set(history, heldout, users, items)
    summary, per_user, tops = [], {}, {}

    for name, popularity in POPULARITY_MODELS.items():
        scores = popularity(history, items, len(eval_set.rows))
        row, per_user[name], tops[name] = evaluate(scores, eval_set)
        summary.append({"model": name, **row})

    for feedback in dict.fromkeys(f for f, _, _ in configs):
        X = interaction_matrix(history, users, items, feedback)
        W = history_weights(X, feedback)[eval_set.rows]
        for measure in dict.fromkeys(m for f, m, _ in configs if f == feedback):
            S = item_similarity(X, measure)
            for k in [k for f, m, k in configs if f == feedback and m == measure]:
                scores = knn_scores(W, prune(S, k))
                name = f"ItemKNN-{feedback}-{measure}-k{k or 'all'}"
                row, per_user[name], tops[name] = evaluate(scores, eval_set)
                summary.append({
                    "model": name, "feedback": feedback, "similarity": measure, "k": k or "all", **row,
                    "scored_items": ((scores != 0) & ~eval_set.seen).sum(axis=1).mean(),
                })
    return pd.DataFrame(summary), per_user, tops, eval_set


def compare_with_popularity(knn, pop, history):
    """Per-user NDCG comparison overall and by user activity / mainstreamness."""
    diff = knn["ndcg"] - pop["ndcg"]
    users = diff.index
    activity = history.groupby("UserID").size().reindex(users)
    item_count = history["MovieID"].map(history["MovieID"].value_counts())
    mainstream = item_count.groupby(history["UserID"]).mean().reindex(users)

    def row(dimension, group, mask):
        d = diff[mask]
        return {"dimension": dimension, "group": group, "n_users": int(mask.sum()),
                "knn_ndcg": knn["ndcg"][mask].mean(), "pop_ndcg": pop["ndcg"][mask].mean(),
                "mean_diff": d.mean(), "knn_wins": (d > 0).mean(), "ties": (d == 0).mean(),
                "both_zero": ((knn["ndcg"][mask] == 0) & (pop["ndcg"][mask] == 0)).mean(),
                "pop_wins": (d < 0).mean()}

    rows = [row("all", "all users", np.ones(len(users), dtype=bool))]
    for dimension, values, n_bins, unit in [("activity", activity, 4, "ratings"),
                                            ("mainstreamness", mainstream, 3, "mean movie count")]:
        bins, edges = pd.qcut(values, n_bins, labels=False, retbins=True)
        for b in range(n_bins):
            label = f"Q{b + 1} ({edges[b]:.0f}-{edges[b + 1]:.0f} {unit})"
            rows.append(row(dimension, label, (bins == b).to_numpy()))

    d = diff.to_numpy()
    ci_low, ci_high = paired_bootstrap(d)
    test = {"mean_diff": d.mean(), "ci_low": ci_low, "ci_high": ci_high,
            "wilcoxon_p": stats.wilcoxon(d).pvalue}
    return pd.DataFrame(rows), test


def plot_k_sweep(grid, popularity, n_items, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for (feedback, measure), g in grid.groupby(["feedback", "similarity"], sort=False):
        x = [n_items if k == "all" else k for k in g["k"]]
        ax.plot(x, g[METRIC], marker="o", label=f"{measure} ({feedback})",
                linestyle="-" if feedback == "implicit" else "--")
    for (name, value), style in zip(popularity.items(), [":", "-."]):
        ax.axhline(value, color="grey", linestyle=style, linewidth=1.2, label=name)
    ax.set_xscale("log")
    ax.set_xticks([k for k in K_GRID if k is not None] + [n_items])
    ax.set_xticklabels([str(k) for k in K_GRID[:-1]] + ["all"])
    ax.set_xlabel("Neighbourhood size k")
    ax.set_ylabel(f"NDCG@{TOP_N} (validation)")
    ax.set_title("Item-kNN: validation NDCG@10 by k and similarity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(train, validation, test, out_dir="results", fig_dir="figures"):
    """1) grid search on validation (fitted on train), 2) selection by validation NDCG@10 and
    bootstrap check of k, 3) refit on train + validation and evaluate once on test,
    4) metric check against scikit-learn, 5) per-user comparison with Most Popular."""
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    users, items = catalogue_index(train, validation, test)

    start = time.time()
    configs = [(f, m, k) for f, measures in SIMILARITIES.items() for m in measures for k in K_GRID]
    validation_summary, validation_users, _, _ = run_period(train, validation, users, items, configs)
    validation_summary.to_csv(out_dir / "item_knn_validation.csv", index=False)
    is_knn = validation_summary["model"].str.startswith("ItemKNN")
    grid, popularity = validation_summary[is_knn], validation_summary[~is_knn]
    new_users = ~validation["UserID"].isin(train["UserID"])
    print(f"Validation grid: {len(grid)} configurations in {time.time() - start:.0f} s, "
          f"{int(grid['n_users'].iloc[0])} users evaluated, "
          f"{int(grid['n_cold_users_excluded'].iloc[0])} cold users without training history excluded")
    print(f"{validation.loc[new_users, 'UserID'].nunique()} users rated for the first time during the "
          f"validation period and made {new_users.mean():.1%} of the validation ratings")
    print(grid.pivot_table(index="k", columns=["feedback", "similarity"], values=METRIC, sort=False)
          .round(4).to_string())
    print(popularity[["model", f"precision@{TOP_N}", f"recall@{TOP_N}", METRIC, "coverage"]]
          .round(4).to_string(index=False))
    plot_k_sweep(grid, dict(zip(popularity["model"], popularity[METRIC])), len(items),
                 fig_dir / "item_knn_k_sweep.png")

    best = grid.loc[grid.groupby("feedback", sort=False)[METRIC].idxmax()]
    selected = best.loc[best[METRIC].idxmax(), "model"]
    best_pop = popularity.loc[popularity[METRIC].idxmax(), "model"]
    print(f"Selected on validation: {selected}; strongest popularity variant: {best_pop}")
    chosen = grid.set_index("model").loc[selected]
    robustness = []
    for r in grid[(grid["feedback"] == chosen["feedback"])
                  & (grid["similarity"] == chosen["similarity"])].itertuples():
        d = (validation_users[r.model]["ndcg"] - validation_users[selected]["ndcg"]).to_numpy()
        ci_low, ci_high = paired_bootstrap(d)
        robustness.append({"model": r.model, "k": r.k, "ndcg_diff_vs_selected": d.mean(),
                           "ci_low": ci_low, "ci_high": ci_high})
    robustness = pd.DataFrame(robustness)
    robustness.to_csv(out_dir / "item_knn_k_robustness.csv", index=False)
    print(robustness.round(4).to_string(index=False))

    final_configs = [(r.feedback, r.similarity, None if r.k == "all" else int(r.k))
                     for r in best.itertuples()]
    history = pd.concat([train, validation])
    test_summary, test_users, test_tops, test_set = run_period(history, test, users, items, final_configs)
    test_summary["selected_on_validation"] = test_summary["model"] == selected
    test_summary.to_csv(out_dir / "item_knn_test.csv", index=False)
    print(test_summary[["model", f"precision@{TOP_N}", f"recall@{TOP_N}", METRIC, "coverage",
                        "mean_rec_popularity", "n_users"]].round(4).to_string(index=False))

    check = check_against_sklearn(test_tops[selected], test_set.relevant)
    check.to_csv(out_dir / "metric_check.csv", index=False)
    print("\nMetric check against scikit-learn (largest absolute difference):")
    print(check.to_string(index=False))

    tables = []
    for period, per_user, hist in [("validation", validation_users, train), ("test", test_users, history)]:
        table, test_stats = compare_with_popularity(per_user[selected], per_user[best_pop], hist)
        table.insert(0, "period", period)
        tables.append(table)
        print(f"\n{period}: {selected} vs {best_pop}, per-user NDCG@{TOP_N} difference "
              f"{test_stats['mean_diff']:.4f} (95% bootstrap CI {test_stats['ci_low']:.4f} to "
              f"{test_stats['ci_high']:.4f}, Wilcoxon p = {test_stats['wilcoxon_p']:.2g})")
        print(table.round(3).to_string(index=False))
    pd.concat(tables).to_csv(out_dir / "item_knn_vs_popularity.csv", index=False)


def _load_split(data_dir):
    """Standalone runs only: same filter (movies with >= 5 ratings) and cut dates as run_all.py."""
    ratings = pd.read_csv(Path(data_dir) / "ratings.dat", sep="::", engine="python", header=None,
                          names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin-1")
    counts = ratings["MovieID"].value_counts()
    ratings = ratings[ratings["MovieID"].isin(counts[counts >= 5].index)]
    t_validation = pd.Timestamp("2000-11-23").timestamp()
    t_test = pd.Timestamp("2001-01-01").timestamp()
    return (ratings[ratings["Timestamp"] < t_validation],
            ratings[(ratings["Timestamp"] >= t_validation) & (ratings["Timestamp"] < t_test)],
            ratings[ratings["Timestamp"] >= t_test])