"""
Content-based: profile matching over genre and release-year metadata
Part 2 model

    1. Creates implicit user profile matrices (threshold 4),
       parse genre and year matrices
    3. Form evaluation set
    4. Tune hyperparameters and rank best combos
    5. Fit model with tuned hyperparameters to test data
        -> final model

Ranking and metrics come from evaluation.py.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_preparation import load_data
from evaluation import evaluation_set, catalogue_index, interaction_matrix, evaluate, TOP_N
from item_knn import _load_split

METRIC = f"ndcg@{TOP_N}"
SEED = 42

#
# Search space
#

year_weights = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
norm_options = [False, True] # l2 true or false

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

    If l2 normalization is done, movies with more genres doesn't
    get greater weight than movies with only a 1 genre.
    """
    if l2_norm_items:
        genre_mat = normalize(genre_mat, norm='l2') # Normalizes genre vector to lenght 1

    feature_matrix = np.hstack([ genre_mat, year_weight * years ])

    return feature_matrix


def validate_model(feature_matrix, user_matrix, eval_set):
    """
    Uses cosine similarity to evaluate model performance.
    l2 normalization done also for user profiles
    """
    # Create and normalize user profiles
    user_profiles = user_matrix @ feature_matrix
    user_profiles = normalize(user_profiles, norm='l2')

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
    l2 normalization for genre: true or false.
    years: minmax or decade_onehot scaling.
    Best weight for year
    """
    print('Tuning hyperparameters...')
    results = []
    for genre_enc, genre_mat in genre_variants.items():
        for l2_norm in norm_options:
            for y_scale, y_mat in year_variants.items():
                for yw in year_weights:

                    if yw == 0.0 and y_scale != 'minmax':
                        continue # year doesn't affect

                    feature_matrix = build_feature_matrix(
                        genre_mat, y_mat, year_weight=yw, l2_norm_items=l2_norm
                    )

                    summary, _, _ = validate_model(
                        feature_matrix, user_matrix, eval_val
                    )

                    results.append({
                        "genre_encoding": genre_enc,
                        "l2_genres": l2_norm,
                        "year_scale": y_scale if yw > 0 else "none",
                        "year_weight": yw,
                        **summary
                    })

    results_df = pd.DataFrame(results).sort_values("ndcg@10", ascending=False)
    results_df.to_csv(out_dir / "content_based_validation.csv", index=False)
    print('Results saved to "content_based_validation.csv".')
    return results_df


def run_contentbased(train, validation, test, df_movies, out_dir="results"):
    """
    1. creates implicit user profile matrices
    2. parses genre and year matrices
    3. Forms evaluation set
    4. tunes hyperparameters and finds the 5 best combos
    5. fits model with tuned hyperparameters to test data
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users, items = catalogue_index(train, validation, test)

    # Implicit user profile matrix for training
    user_matrix_train = interaction_matrix(train, users, items, feedback='implicit')

    # Implicit user profile matrix for validation
    history = pd.concat([train, validation])
    user_matrix_val = interaction_matrix(history, users, items, feedback='implicit')

    genre_variants, year_variants = matrix_parser(df_movies, items)

    # evalutiation with validation
    eval_val = evaluation_set(history=train, heldout=validation, users=users, items=items)
    results_df = tune_hyperparameters(genre_variants, year_variants, user_matrix_train, eval_val, out_dir)

    print("Five best combos with validation:")
    print(results_df.head(5).to_string(index=False))

    print('\nFitting model with tuned hyperparameters...')
    bestModel = results_df.iloc[0] # Take first row
    tuned_genre_matrix = genre_variants[bestModel["genre_encoding"]]
    tuned_y_scale = "minmax" if bestModel["year_scale"] == "none" else bestModel["year_scale"]
    tuned_year_matrix = year_variants[tuned_y_scale]

    best_features = build_feature_matrix(
        tuned_genre_matrix, tuned_year_matrix,
        year_weight=bestModel["year_weight"],
        l2_norm_items=bestModel["l2_genres"]
    )

    # model with test data after tuning
    eval_test = evaluation_set(history=history, heldout=test, users=users, items=items)
    test_summary, _, _ = validate_model(best_features, user_matrix_val, eval_test)

    test_result = []
    test_result.append({
        "genre_encoding": bestModel["genre_encoding"],
        "l2_genres": bestModel["l2_genres"],
        "year_scale": tuned_y_scale if bestModel["year_weight"] > 0 else "none",
        "year_weight": bestModel["year_weight"],
        **test_summary
    })

    pd.DataFrame(test_result).to_csv(out_dir / "content_based_test.csv", index=False)
    print('Results saved to "content_based_test.csv".')
    print("Final model:")
    print(pd.DataFrame(test_result).to_string(index=False))



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where the CSV results are written")
    parser.add_argument("--fig-dir", default="figures", help="where the figures are written")
    args = parser.parse_args()

    # Load df
    df_movies, df_ratings, df_users = load_data(args.data_dir)
    run_contentbased(*_load_split(args.data_dir), df_movies, out_dir=args.out_dir,)


if __name__ == "__main__":
    main()


