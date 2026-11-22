"""
Matrix Factorization (ALS)

Part 2 recommender.

Uses Alternating Least Squares (ALS)
from the implicit library.

Workflow:
1. Tune factors, regularization and iterations on validation.
2. Select best validation NDCG@10.
3. Refit on train + validation.
4. Evaluate on test.

Ranking and metrics come from evaluation.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import stats

from implicit.als import AlternatingLeastSquares

from evaluation import (
    TOP_N,
    catalogue_index,
    evaluate,
    evaluation_set,
)

METRIC = f"ndcg@{TOP_N}"

TEST_SEEDS = [42, 43, 44, 45, 46]
SEED = 42
#
# Search space
#
FACTORS = [32, 64, 128, 256]

REGULARIZATION = [
    0.0001,
    0.001,
    0.01,
    0.1,
]

ITERATIONS = [
    20,
    50,
    100,
]


def build_interaction_matrix(history, users, items):

    user_map = {
        user: idx
        for idx, user in enumerate(users)
    }

    item_map = {
        item: idx
        for idx, item in enumerate(items)
    }

    rows = history["UserID"].map(user_map)
    cols = history["MovieID"].map(item_map)

    values = history["Rating"].astype(float)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(items)),
    )

    return matrix


def train_model(history, users, items,
                factors, reg, iterations,seed):

    matrix = build_interaction_matrix(
        history,
        users,
        items
    )

    #
    # implicit ALS expects item-user
    #
    item_user = matrix.T.tocsr()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=seed,
    )

    model.fit(item_user)

    return model


def score_matrix(model, eval_set, items):

    scores = np.zeros(
        (
            len(eval_set.rows),
            len(items)
        ),
        dtype=np.float32
    )

    #
    # Because we trained on item-user:
    #
    # model.user_factors -> items
    # model.item_factors -> users
    #

    for row, user_idx in enumerate(eval_set.rows):

        user_vector = model.item_factors[user_idx]

        scores[row] = (
            model.user_factors
            @
            user_vector
        )

    return scores


def run_period(history, heldout, users, items,
               factors, reg, iterations,seed=SEED):

    model = train_model(
        history,
        users,
        items,
        factors=factors,
        reg=reg,
        iterations=iterations,
        seed=seed,
    )

    eval_set = evaluation_set(
        history,
        heldout,
        users,
        items,
    )

    scores = score_matrix(
        model,
        eval_set,
        items,
    )

    summary, per_user, top = evaluate(
        scores,
        eval_set,
    )

    return summary, per_user, top, eval_set

def split_test_period(test):

    split_time = test["Timestamp"].median()

    early_test = test[
        test["Timestamp"] <= split_time
    ]

    late_test = test[
        test["Timestamp"] > split_time
    ]

    return early_test, late_test

def sparsity_ablation(
        train,
        validation,
        test,
        users,
        items,
        factors,
        reg,
        iterations):

    rows = []

    for frac in [1.0, 0.5, 0.2]:

        sparse_train = train.sample(
            frac=frac,
            random_state=42
        )

        history = pd.concat(
            [sparse_train, validation],
            ignore_index=True
        )

        summary, _, _, _ = run_period(
            history,
            test,
            users,
            items,
            factors=factors,
            reg=reg,
            iterations=iterations,
            seed=42,
        )

        rows.append({
            "train_fraction": frac,
            **summary,
        })

        print(
            f"Train fraction={frac:.1f} "
            f"NDCG={summary[f'ndcg@{TOP_N}']:.4f}"
        )

    return pd.DataFrame(rows)

def user_activity_breakdown(train, per_user):

    activity = (
        train
        .groupby("UserID")
        .size()
        .rename("activity")
    )


    df = per_user.join(activity)

    threshold = df["activity"].median()

    groups = {
        "low_activity":
            df[df["activity"] <= threshold],

        "high_activity":
            df[df["activity"] > threshold],
    }

    rows = []

    for name, group in groups.items():

        rows.append({
            "group": name,
            "users": len(group),
            "precision": group["precision"].mean(),
            "recall": group["recall"].mean(),
            "ndcg": group["ndcg"].mean(),
        })

    return pd.DataFrame(rows)

def validation_search(train, validation,
                      users, items):

    rows = []

    total = (
        len(FACTORS)
        * len(REGULARIZATION)
        * len(ITERATIONS)
    )

    current = 0

    for factors in FACTORS:

        for reg in REGULARIZATION:

            for iterations in ITERATIONS:

                current += 1

                summary, _, _, _ = run_period(
                    history=train,
                    heldout=validation,
                    users=users,
                    items=items,
                    factors=factors,
                    reg=reg,
                    iterations=iterations,
                )

                rows.append({
                    "factors": factors,
                    "reg": reg,
                    "iterations": iterations,
                    **summary,
                })

                print(
                    f"[{current:02d}/{total}] "
                    f"factors={factors:<4} "
                    f"reg={reg:<8} "
                    f"iters={iterations:<3} "
                    f"NDCG={summary[f'ndcg@{TOP_N}']:.4f}"
                )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        METRIC,
        ascending=False,
    )

    return results


def run(train, validation, test,
        out_dir="results"):

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

    #
    # Validation tuning
    #

    validation_results = validation_search(
        train,
        validation,
        users,
        items,
    )

    validation_results.to_csv(
        out_dir / "matrix_factorization_validation.csv",
        index=False,
    )

    print("\nTop validation configurations\n")

    print(
        validation_results
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    best = validation_results.iloc[0]

    best_factors = int(best["factors"])
    best_reg = float(best["reg"])
    best_iterations = int(best["iterations"])

    print(
        f"\nSelected configuration:"
        f" factors={best_factors},"
        f" reg={best_reg},"
        f" iterations={best_iterations}"
    )

    #
    # Refit on train + validation
    #

    history = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    early_test, late_test = split_test_period(
    test
    )

    summary_early, _, _, _ = run_period(
    history,
    early_test,
    users,
    items,
    factors=best_factors,
    reg=best_reg,
    iterations=best_iterations,
    seed=42,
    )

    summary_late, _, _, _ = run_period(
        history,
        late_test,
        users,
        items,
        factors=best_factors,
        reg=best_reg,
        iterations=best_iterations,
        seed=42,
    )

    period_results = pd.DataFrame([
    {
        "period": "early",
        **summary_early
    },
    {
        "period": "late",
        **summary_late
    }
    ])

    period_results.to_csv(
    out_dir / "matrix_factorization_early_late.csv",
    index=False,
    )

    print("\nEarly vs Late Test Period\n")

    print(
        period_results
        .round(4)
        .to_string(index=False)
    )

    seed_rows = []

    for seed in TEST_SEEDS:

        summary, per_user, top, _ = run_period(
            history,
            test,
            users,
            items,
            factors=best_factors,
            reg=best_reg,
            iterations=best_iterations,
            seed=seed,
        )

        seed_rows.append({
            "seed": seed,
            **summary,
        })

    seed_results = pd.DataFrame(seed_rows)

    seed_results.to_csv(
    out_dir / "matrix_factorization_test_seeds.csv",
    index=False,
    )

    print("\nTest Results\n")

    metrics = [
        f"precision@{TOP_N}",
        f"recall@{TOP_N}",
        f"ndcg@{TOP_N}",
        "coverage",
    ]

    summary_row = {}

    for metric in metrics:

        mean, ci_half_width = confidence_interval(
            seed_results[metric]
        )

        summary_row[
            f"{metric}_mean"
        ] = mean

        summary_row[
            f"{metric}_ci_half_width"
        ] = ci_half_width

    test_results = pd.DataFrame([{
        "model": "ALS",
        "factors": best_factors,
        "reg": best_reg,
        "iterations": best_iterations,
        **summary_row,
    }])

    test_results.to_csv(
        out_dir / "matrix_factorization_test.csv",
        index=False,
    )
    

    print("\nSeed Results\n")

    print(
        seed_results[
            [
                "seed",
                f"precision@{TOP_N}",
                f"recall@{TOP_N}",
                f"ndcg@{TOP_N}",
                "coverage",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nMean ± 95% CI half-width\n")

    print(
        test_results.round(4).to_string(index=False)
    )
    ablation_results = sparsity_ablation(
    train,
    validation,
    test,
    users,
    items,
    best_factors,
    best_reg,
    best_iterations,
)

    ablation_results.to_csv(
        out_dir / "matrix_factorization_sparsity.csv",
        index=False,
    )

    print("\nSparsity Ablation\n")
    print(
        ablation_results
        .round(4)
        .to_string(index=False)
    )

    activity_results = user_activity_breakdown(
    history,
    per_user,
    )

    activity_results.to_csv(
        out_dir / "matrix_factorization_user_activity.csv",
        index=False,
    )

    return test_results


def _load_split(data_dir):

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

def confidence_interval(values):

    values = np.asarray(values)

    mean = values.mean()

    t = stats.t.ppf(
        0.975,
        len(values) - 1
    )

    half_width = (
        values.std(ddof=1)
        * t
        / np.sqrt(len(values))
    )

    return mean, half_width


def main():

    parser = argparse.ArgumentParser()

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
              