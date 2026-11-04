"""
Most Popular baselines (Part 1).

Evaluates:
    - MostPop-all_time
    - MostPop-temporal

using the shared evaluation protocol in evaluation.py.

From run_all.py:

    import most_popular
    most_popular.run(train, validation, test)

Standalone:

    python most_popular.py --data-dir ml-1m --out-dir results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (
    TOP_N,
    catalogue_index,
    evaluate,
    evaluation_set,
)

TEMPORAL_QUANTILE = 0.8


def all_time_scores(history, items, n_users):
    """
    Movie popularity based on all interactions
    observed in the training history.
    """

    popularity = (
        history["MovieID"]
        .value_counts()
        .reindex(items, fill_value=0)
        .to_numpy(np.float64)
    )

    return np.tile(popularity, (n_users, 1))


def temporal_scores(
    history,
    items,
    n_users,
    quantile=TEMPORAL_QUANTILE,
):
    """
    Movie popularity based only on
    the most recent interactions.

    quantile=0.8 means:
        newest 20% of timestamps.
    """

    cutoff = history["Timestamp"].quantile(quantile)

    recent = history[
        history["Timestamp"] >= cutoff
    ]

    popularity = (
        recent["MovieID"]
        .value_counts()
        .reindex(items, fill_value=0)
        .to_numpy(np.float64)
    )

    return np.tile(popularity, (n_users, 1))


def run_period(
    history,
    heldout,
    users,
    items,
):
    """
    Evaluate all popularity variants
    on one held-out period.
    """

    eval_set = evaluation_set(
        history,
        heldout,
        users,
        items,
    )

    summary = []
    per_user = {}
    tops = {}

    models = {
        "MostPop-all_time": all_time_scores(
            history,
            items,
            len(eval_set.rows),
        ),

        "MostPop-temporal": temporal_scores(
            history,
            items,
            len(eval_set.rows),
        ),
    }

    for name, scores in models.items():

        row, per_user[name], tops[name] = evaluate(
            scores,
            eval_set,
        )

        summary.append({
            "model": name,
            **row,
        })

    return (
        pd.DataFrame(summary),
        per_user,
        tops,
        eval_set,
    )


def run(
    train,
    validation,
    test,
    out_dir="results",
):
    """
    Evaluate popularity baselines on:
        validation
        test
    """

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    users, items = catalogue_index(
        train,
        validation,
        test,
    )

    print(
        f"Catalogue contains "
        f"{len(users)} users and "
        f"{len(items)} movies"
    )

    #
    # Validation
    #

    validation_summary, _, _, validation_set = run_period(
        train,
        validation,
        users,
        items,
    )

    validation_summary.to_csv(
        out_dir / "most_popular_validation.csv",
        index=False,
    )

    print("\nValidation results")
    print(
        validation_summary[
            [
                "model",
                f"precision@{TOP_N}",
                f"recall@{TOP_N}",
                f"ndcg@{TOP_N}",
                "coverage",
                "mean_rec_popularity",
                "n_users",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    #
    # Test
    #

    history = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    test_summary, _, _, test_set = run_period(
        history,
        test,
        users,
        items,
    )

    test_summary.to_csv(
        out_dir / "most_popular_test.csv",
        index=False,
    )

    print("\nTest results")
    print(
        test_summary[
            [
                "model",
                f"precision@{TOP_N}",
                f"recall@{TOP_N}",
                f"ndcg@{TOP_N}",
                "coverage",
                "mean_rec_popularity",
                "n_users",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nValidation users evaluated: "
        f"{validation_set.n_cold} cold users excluded"
    )

    print(
        f"Test users evaluated: "
        f"{test_set.n_cold} cold users excluded"
    )


def _load_split(data_dir):
    """
    Same split as item_knn.py.
    """

    ratings = pd.read_csv(
        Path(data_dir) / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=[
            "UserID",
            "MovieID",
            "Rating",
            "Timestamp",
        ],
        encoding="latin-1",
    )

    counts = ratings["MovieID"].value_counts()

    ratings = ratings[
        ratings["MovieID"].isin(
            counts[counts >= 5].index
        )
    ]

    t_validation = pd.Timestamp(
        "2000-11-23"
    ).timestamp()

    t_test = pd.Timestamp(
        "2001-01-01"
    ).timestamp()

    train = ratings[
        ratings["Timestamp"] < t_validation
    ]

    validation = ratings[
        (ratings["Timestamp"] >= t_validation)
        &
        (ratings["Timestamp"] < t_test)
    ]

    test = ratings[
        ratings["Timestamp"] >= t_test
    ]

    return train, validation, test


def main():

    parser = argparse.ArgumentParser(
        description="Most Popular baselines"
    )

    parser.add_argument(
        "--data-dir",
        default="ml-1m",
    )

    parser.add_argument(
        "--out-dir",
        default="results",
    )

    args = parser.parse_args()

    run(
        *_load_split(args.data_dir),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()