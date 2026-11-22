"""
Content-based: profile matching over genre and release-year metadata
Part 2 model

    1. Creates implicit user matrices (threshold 4),
       parse genre and year matrices
    3. Form evaluation set
    4. Tune hyperparameters and rank best combos
    5. Fit model with tuned hyperparameters to test data
        -> final model

Ranking and metrics come from evaluation.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluation import (evaluation_set, catalogue_index, interaction_matrix,
                        evaluate, TOP_N, N_BOOTSTRAP)
from matrix_factorization import user_activity_breakdown

METRIC = f"ndcg@{TOP_N}"
SEED = 42

#
# Search space
#

year_weights = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
l2_options = [False, True] # l2 true or false
# Sparsity analysis
interactions = [0.2, 0.5, 1.0]  # 20%, 50% and 100% of all train interactions
seeds = [42, 43, 44]

def matrix_parser(df_movies, items):
    df_movies = df_movies.copy()

    # Form release years from title
    df_movies['Year'] = df_movies['Title'].str.extract(r'\((\d{4})\)').astype(int)
    movies_id_index = df_movies.set_index('MovieID').reindex(items)

    # Genre multi-hot encoding matrix
    genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]
    movie_genres = (
        movies_id_index['Genres'].str.get_dummies(sep='|')
        .reindex(columns=genres, fill_value=0)
        .to_numpy(dtype=np.float32)
    )

    # TfidfTransformer to weight more rare genres
    tfidf_genres = (
        TfidfTransformer().fit_transform(movie_genres)
        .toarray().astype(np.float32)
    )

    genre_variants = {
        'multi-hot': movie_genres,
        'tfidf': tfidf_genres
    }

    # release year: with decade scale and minmax scale
    years_col = movies_id_index[['Year']]

    decades_col = (years_col // 10) * 10  # movie release years as decades 80,90,00 etc.

    year_variants = {
        'minmax': MinMaxScaler().fit_transform(years_col).astype(np.float32),
        'decade_onehot': OneHotEncoder(sparse_output=False).fit_transform(decades_col).astype(np.float32)
    }

    return genre_variants, year_variants


def build_feature_matrix(genre_mat, years, year_weight=1.0, l2_norm_items=False):
    """
    Builds feature matrix from genres and release-year metadata.
    Genre is always 1.0 weight and year weight is tuned.

    If l2 normalization is done, movies with more genres in multi-hot doesn't
    get greater weight than movies with only a 1 genre.
    """
    if l2_norm_items:
        genre_mat = normalize(genre_mat, norm='l2') # Normalizes genre vector to lenght 1

    feature_matrix = np.hstack([ genre_mat, year_weight * years ])

    return feature_matrix


def validate_model(feature_matrix, user_matrix, eval_set):
    """
    Uses cosine similarity to evaluate model performance.
    """
    # Create user profiles
    user_profiles = user_matrix @ feature_matrix

    scores = cosine_similarity(
        user_profiles[eval_set.rows],
        feature_matrix
    )

    scores[eval_set.seen] = -np.inf # already seen removed
    summary, per_user, top = evaluate(scores, eval_set)

    return summary, per_user, top


def tune_hyperparameters( genre_variants, year_variants,
                          user_matrix, eval_val, out_dir="results" ):
    """
    Tunes hyperparameters to find the best model:
    genres: multi-hot or tfidf.
    l2 normalization for multi-hot: true or false.
    years: minmax or decade_onehot scaling.
    Best weight for year
    """
    print('Tuning hyperparameters...')
    results = []
    for genre_enc, genre_mat in genre_variants.items():
        for l2_norm in l2_options:
            if genre_enc == "tfidf" and l2_norm:
                continue
            for y_scale, y_mat in year_variants.items():
                for yw in year_weights:

                    feature_matrix = build_feature_matrix(
                        genre_mat, y_mat, year_weight=yw, l2_norm_items=l2_norm
                    )

                    summary, _, _ = validate_model(
                        feature_matrix, user_matrix, eval_val
                    )

                    results.append({
                        "genre_encoding": genre_enc,
                        "l2_genres": l2_norm,
                        "year_scale": y_scale,
                        "year_weight": yw,
                        **summary
                    })

    results_df = pd.DataFrame(results).sort_values("ndcg@10", ascending=False)
    results_df.to_csv(out_dir / "content_based_validation.csv", index=False)
    print('Results saved to "content_based_validation.csv".')
    return results_df

def CI_95(per_user):
    """95% CI for NDCG by generating N_BOOTSTRAP samples"""
    ndcg_scores = per_user["ndcg"]
    rng = np.random.default_rng(SEED)

    boot = np.array([
        rng.choice(ndcg_scores, size=len(ndcg_scores), replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])

    ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
    return ci_lower, ci_upper


def content_sparsity(train, validation, test, best_model_features, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    users, items = catalogue_index(train, validation, test)

    print("\n\nCalculating next evaluated model at 100%, 50%, and 20% of the training interactions...")
    sparsityresults = []
    for i in interactions:
        for seed in seeds:
            trainsample = train.sample(frac=i, random_state=seed) # new train sets
            history = pd.concat([trainsample, validation])
            user_matrix_val = interaction_matrix(history, users, items, feedback='implicit')

            # Form new evaluation set and validate model with updated train data
            eval_test = evaluation_set(history=history, heldout=test, users=users, items=items)
            test_summary, _, _ = validate_model(best_model_features, user_matrix_val, eval_test)

            sparsityresults.append({
                "training interactions": f"{i * 100:.0f}%",
                "seed": seed,
                **test_summary
            })

    sparsityresults_df = pd.DataFrame(sparsityresults)

    # calculate means (3 seeds used)
    sparsity_means = ( sparsityresults_df.groupby("training interactions", sort=False)
        .mean(numeric_only=True).drop(columns=["seed"]).reset_index()
    )

    sparsity_means.to_csv(out_dir / "content_based_sparsity.csv", index=False)
    print('Results saved to "content_based_sparsity.csv".')
    print("Results:")
    print(pd.DataFrame(sparsity_means).to_string(index=False))

def test_period_breakdown(history, test, users, items, feature_matrix):
    """
    Evaluate the model separately on early and late test periods.
    threshold median timestamp
    """
    threshold = test["Timestamp"].median()
    periods = {
        "early_period": test[test["Timestamp"] <= threshold],
        "late_period": test[test["Timestamp"] > threshold]
    }

    # Training user interaction matrix
    user_matrix = interaction_matrix(
        history, users, items, feedback="implicit"
    )

    print('\nRunning early vs late test period analysis...')
    results = []
    for period_name, period_data in periods.items():
        eval_period = evaluation_set(
            history=history,
            heldout=period_data,
            users=users,
            items=items
        )

        summary, per_user, top = validate_model(
            feature_matrix,
            user_matrix,
            eval_period
        )

        results.append({
            "Group": period_name,
            **summary
        })

    return pd.DataFrame(results)

def run_contentbased(train, validation, test, df_movies, out_dir="results"):
    """
    1. creates implicit user matrices
    2. parses genre and year matrices
    3. Forms evaluation set
    4. tunes hyperparameters and finds the 5 best combos
    5. fits model with tuned hyperparameters to test data
    6. runs content-based sparsity and disaggregated analysis
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    users, items = catalogue_index(train, validation, test)
    history = pd.concat([train, validation])

    user_matrix_train = interaction_matrix(train, users, items, feedback='implicit') # Implicit user matrix for training
    user_matrix_val = interaction_matrix(history, users, items, feedback='implicit') # Implicit user matrix for validation
    genre_variants, year_variants = matrix_parser(df_movies, items) # genre and year vectors

    # evalutiation with validation and tune hyperparameters
    eval_val = evaluation_set(history=train, heldout=validation, users=users, items=items)
    results_df = tune_hyperparameters(genre_variants, year_variants, user_matrix_train, eval_val, out_dir)
    print("Five best combos with validation:")
    print(results_df.head(5).to_string(index=False))

    print('\nFitting model with tuned hyperparameters...')

    # Model with the best hyperparameter combo
    bestModel = results_df.iloc[0] # Take first row
    tuned_y_scale = bestModel["year_scale"]
    best_features = build_feature_matrix(
        genre_variants[bestModel["genre_encoding"]],
        year_variants[tuned_y_scale],
        year_weight=bestModel["year_weight"],
        l2_norm_items=bestModel["l2_genres"]
    )

    # model with test data after tuning
    eval_test = evaluation_set(history=history, heldout=test, users=users, items=items)
    test_summary, per_user, _ = validate_model(best_features, user_matrix_val, eval_test)
    ci_lower, ci_upper = CI_95(per_user)

    test_result = []
    test_result.append({
        "genre_encoding": bestModel["genre_encoding"],
        "l2_genres": bestModel["l2_genres"],
        "year_scale": tuned_y_scale,
        "year_weight": bestModel["year_weight"],
        **test_summary,
        "CI_lower NDCG": ci_lower,
        "CI_upper NDCG": ci_upper
    })

    pd.DataFrame(test_result).to_csv(out_dir / "content_based_test.csv", index=False)
    print('Results saved to "content_based_test.csv".')
    print("Final model:")
    print(pd.DataFrame(test_result).to_string(index=False))

    # run sparsity analysis
    content_sparsity(train, validation, test, best_features, out_dir)

    # run low vs high activity user analysis
    print('\nRunning low- vs high-activity user analysis...')
    user_activity_results = user_activity_breakdown(history, per_user)
    user_activity_results.to_csv(out_dir / "content_based_user_activity.csv",index=False)
    print('Results saved to "content_based_user_activity.csv".')

    # early vs late test period
    test_period_results = test_period_breakdown(history, test, users, items, best_features)
    test_period_results.to_csv(out_dir / "content_based_early_late.csv", index=False)
    print('Results saved to "content_based_early_late.csv".')