"""Disaggregated analysis and sparsity ablation for Mult-DAE against item-kNN and Most Popular (Part 2).

From run_all.py, after mult_dae.run:
    import mult_dae_analysis
    mult_dae_analysis.run(training_set, validation_set, test_set)

Standalone:
    python mult_dae_analysis.py --data-dir ml-1m --out-dir results

mult_dae is imported first so that torch loads before pandas (see the comment in run_all.py).
"""
import mult_dae

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import item_knn
import most_popular
from evaluation import (SEED, catalogue_index, evaluate, evaluation_set, interaction_matrix, paired_bootstrap,
                        per_user_metrics)

HEAD_SHARE = 0.2
ACTIVITY_BINS = 4
FRACTIONS = [1.0, 0.5, 0.2]
DRAWS = 3
DAE = "Mult-DAE"
KNN = "ItemKNN"
POP = "MostPop-all_time"


def selected_config(out_dir):
    """The Mult-DAE configuration and number of epochs chosen on validation by mult_dae.run."""
    grid = pd.read_csv(Path(out_dir) / "mult_dae_validation.csv")
    best = grid.loc[grid[mult_dae.METRIC].idxmax()]
    return {"hidden": str(best["hidden"]), "dropout": float(best["dropout"]), "l2": float(best["l2"])}, \
        int(best["best_epoch"])


def model_scores(history, eval_set, users, items, config, epochs, seed):
    """Scores of the three models fitted on `history` for the users of `eval_set`."""
    X = interaction_matrix(history, users, items, "implicit")
    dae, _ = mult_dae.fit(X, config, seed, epochs=epochs)
    return {DAE: mult_dae.score(dae, X[eval_set.rows]),
            KNN: mult_dae.knn_reference(history, eval_set, users, items),
            POP: most_popular.all_time_scores(history, items, len(eval_set.rows))}


def per_user(top, eval_set, head):
    """Per-user NDCG@10 and Recall@10, plus recall over the relevant head and tail movies
    (NaN for users without relevant movies in that group) and the tail share of the list."""
    base = per_user_metrics(top, eval_set.relevant)
    rows = {"ndcg": base["ndcg"].to_numpy(), "recall": base["recall"].to_numpy()}
    hits = np.take_along_axis(eval_set.relevant, top, axis=1)
    in_head = head[top]
    for group, mask in [("head", head), ("tail", ~head)]:
        relevant = (eval_set.relevant & mask).sum(axis=1)
        group_hits = (hits & (in_head if group == "head" else ~in_head)).sum(axis=1)
        rows[f"recall_{group}"] = np.where(relevant > 0, group_hits / np.maximum(relevant, 1), np.nan)
    rows["tail_share"] = (~in_head).mean(axis=1)
    return pd.DataFrame(rows, index=eval_set.user_ids)


def paired(a, b):
    """Mean of a - b over users present in both, with the 95% bootstrap CI."""
    d = (a - b).dropna().to_numpy()
    low, high = paired_bootstrap(d)
    return d.mean(), low, high, len(d)


def disaggregate(results, history, eval_set):
    """Table of mean metrics per model and group, with Mult-DAE minus item-kNN and its CI."""
    activity = history.groupby("UserID").size().reindex(eval_set.user_ids)
    bins, edges = pd.qcut(activity, ACTIVITY_BINS, labels=False, retbins=True)
    groups = [("all", "all users", "ndcg", np.ones(len(activity), dtype=bool))]
    groups += [("activity", f"Q{b + 1} ({edges[b]:.0f}-{edges[b + 1]:.0f} ratings)", "ndcg",
                (bins == b).to_numpy()) for b in range(ACTIVITY_BINS)]
    groups += [("popularity", f"{group} movies", f"recall_{group}", np.ones(len(activity), dtype=bool))
               for group in ["head", "tail"]]
    rows = []
    for dimension, label, metric, mask in groups:
        row = {"dimension": dimension, "group": label, "metric": metric}
        for model, frame in results.items():
            row[model] = frame.loc[mask, metric].mean()
        diff, low, high, n = paired(results[DAE].loc[mask, metric], results[KNN].loc[mask, metric])
        rows.append({**row, "n_users": n, "dae_minus_knn": diff, "ci_low": low, "ci_high": high})
    shares = {"dimension": "popularity", "group": "tail share of top-10 lists", "metric": "tail_share",
              **{model: frame["tail_share"].mean() for model, frame in results.items()},
              "n_users": len(activity)}
    return pd.DataFrame(rows + [shares])


def run(train, validation, test, out_dir="results"):
    """1) fit the three models on train + validation and disaggregate the test results,
    2) refit on 50% and 20% subsamples of the history and evaluate on the full test period."""
    out_dir = Path(out_dir)
    config, epochs = selected_config(out_dir)
    users, items = catalogue_index(train, validation, test)
    history = pd.concat([train, validation])
    eval_set = evaluation_set(history, test, users, items)
    counts = history["MovieID"].value_counts().reindex(items, fill_value=0)
    head = np.zeros(len(items), dtype=bool)
    head[np.argsort(-counts.to_numpy(), kind="stable")[:int(round(HEAD_SHARE * len(items)))]] = True
    print(f"Head: {head.sum()} movies with >= {counts[head].min()} history ratings "
          f"({counts[head].sum() / counts.sum():.1%} of history ratings); tail: {(~head).sum()} movies")

    levels, pairs = [], []
    for fraction in FRACTIONS:
        draws = {}
        for draw in range(DRAWS):
            seed = SEED + draw
            sample = history if fraction == 1.0 else history.sample(frac=fraction, random_state=seed)
            for model, scores in model_scores(sample, eval_set, users, items, config, epochs, seed).items():
                top = evaluate(scores, eval_set)[2]
                draws.setdefault(model, []).append(per_user(top, eval_set, head))
            empty = (interaction_matrix(sample, users, items, "implicit")[eval_set.rows].getnnz(axis=1) == 0).sum()
            print(f"{fraction:.0%} of the history, draw {draw + 1}: {len(sample)} ratings, "
                  f"{empty} evaluated users without a positive rating left")
        results = {model: pd.concat(frames).groupby(level=0).mean() for model, frames in draws.items()}
        if fraction == 1.0:
            table = disaggregate(results, history, eval_set)
            table.to_csv(out_dir / "mult_dae_disaggregated.csv", index=False)
            print(table.round(4).to_string(index=False))
        for model, frame in results.items():
            per_draw = [f["ndcg"].mean() for f in draws[model]]
            levels.append({"fraction": fraction, "model": model, "ndcg@10": frame["ndcg"].mean(),
                           "ndcg_min_draw": min(per_draw), "ndcg_max_draw": max(per_draw),
                           "recall@10": frame["recall"].mean(), "tail_share": frame["tail_share"].mean()})
        for a, b in [(DAE, KNN), (DAE, POP), (KNN, POP)]:
            diff, low, high, _ = paired(results[a]["ndcg"], results[b]["ndcg"])
            pairs.append({"fraction": fraction, "comparison": f"{a} - {b}", "ndcg_diff": diff,
                          "ci_low": low, "ci_high": high})
    for name, rows in [("mult_dae_sparsity.csv", levels), ("mult_dae_sparsity_pairs.csv", pairs)]:
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / name, index=False)
        print(table.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where mult_dae.run wrote its CSVs and where these go")
    args = parser.parse_args()
    run(*item_knn._load_split(args.data_dir), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
