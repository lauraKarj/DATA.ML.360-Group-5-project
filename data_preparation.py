"""Data preparation, plotting and filtering (Part 1).

Load the data and does the data characterization by it.
Plots following distributions: ratings per user, ratings per movie, ratings distribution,
rating activity over time
Filters the data for the analysis and checks duplicates, thresholds and scales.
How the split is done

From run_all.py firstly run:
    import data_preparation
    data_preparation.load_data(args.data_dir)
    data_preparation.characterize_and_plot(df_movies,df_ratings,df_users,fig_dir=args.fig_dir)
    data_preparation.filter_and_split(df_ratings)

Standalone:
    python data_preparation.py --data-dir ml-1m --out-dir results --fig-dir figures
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(data_dir="ml-1m"):
    """Loads the data from target directory"""
    df_movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::", engine="python", header=None,
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )

    df_ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::", engine="python", header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1"
    )

    df_users = pd.read_csv(
        f"{data_dir}/users.dat",
        sep="::", engine="python", header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="latin-1"
    )

    return df_movies, df_ratings, df_users

def run_preparation(df_movies, df_ratings, df_users, fig_dir="figures"):
    """
    Characterizes the data and plots distributions from it
    Filters data for the analysis and check duplicates, thresholds and scales.
    """
    print("Data characterization running...")
    # The data loaded and characterized: users, items, interactions, density;
    users = len(set(df_users['UserID'].dropna())) # filter out just in case duplicated and nulls
    movies = len(set(df_movies['MovieID'].dropna())) # filter out just in case duplicated and nulls
    ratings = len(df_ratings)
    users_ratingDf = df_ratings["UserID"].nunique()
    movies_ratingDf = df_ratings["MovieID"].nunique()

    print('All users:', users, " Users in ratings dataset:", users_ratingDf)
    print('All movies (items):', movies, " Movies in ratings dataset:", movies_ratingDf)
    print('Proportion of rated movies', movies_ratingDf / movies)

    # Density = what proportion of all possible interactions has actually happened
    print("Number of ratings (interactions):", ratings)
    density = ratings / (users * movies)
    print('Density', density)

    # Plotting part, plots goes to /figures folder
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Long-tail distribution of ratings per user
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    interactions_users = df_ratings["UserID"].value_counts().sort_values(ascending=False)
    print("Maximum ratings per user:", interactions_users.max())
    ax1.hist(interactions_users, bins=100)
    ax1.set_xlabel("Number of ratings per user")
    ax1.set_ylabel("Number of users")
    ax1.set_title("Long-tail distribution of ratings per user")

    # Long-tail distribution of ratings per movie
    interactions_movies = df_ratings["MovieID"].value_counts().sort_values(ascending=False)
    print("Maximum ratings per movie:", interactions_movies.max())
    ax2.hist(interactions_movies, bins=100)
    ax2.set_xlabel("Number of ratings per movie")
    ax2.set_ylabel("Number of movies")
    ax2.set_title("Long-tail distribution of ratings per movie")
    plt.tight_layout()
    plt.savefig(fig_dir / "ratings_per_user_movie.png", dpi=150)
    plt.close()

    # the rating distribution;
    plt.figure(figsize=(5, 4))
    rating_values = df_ratings['Rating']
    plt.hist(rating_values.tolist(), bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
             edgecolor='black', color='lightgreen')
    plt.xlabel("Value of rating")
    plt.ylabel("Number of ratings")
    plt.title("Distribution of ratings")
    plt.savefig(fig_dir / "rating_distribution.png", dpi=150)
    plt.close()

    # the time span and how activity is distributed across it.
    timestamps = pd.to_datetime(df_ratings['Timestamp'], unit='s')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.hist(timestamps.tolist(), bins=500, color='red')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Number of ratings")
    ax1.set_title("Rating activity over time")
    plt.xticks(rotation=45)

    timestamps_2000 = timestamps[(timestamps >= '2000-01-01') & (timestamps < '2001-01-01')]
    ax2.hist(timestamps_2000, bins=100, color='red')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Number of ratings")
    ax2.set_title("Rating activity during 2000")
    plt.tight_layout()
    plt.savefig(fig_dir / "time_activity_distribution.png", dpi=300)
    plt.close()

    # Check that all ratings are on scale 1 to 5
    rating_scale = df_ratings["Rating"]
    print("Ratings varies between:", rating_scale.min(), "-", rating_scale.max())

    # Check that all users have at least 20 ratings
    usercount = df_ratings["UserID"].value_counts().sort_values(ascending=True)
    if (usercount >= 20).all():
        print("True: All users have at least 20 rating.")
    else:
        print("False: Some users have under 20 ratings.")

    print("\nPlots added to figures/")
    print("Filtering starting...")

    # Filter out movies with less than 5 ratings
    movie_counts = df_ratings["MovieID"].value_counts()
    filtered_ratings = df_ratings[df_ratings["MovieID"].isin(movie_counts[movie_counts >= 5].index)]

    # Calculate new ratings per user after deleting movies
    new_usercount = filtered_ratings["UserID"].value_counts().sort_values(ascending=True)

    print("\nAfter filtering")
    print("Ratings:", len(filtered_ratings))
    print("Rows removed:", len(df_ratings)-len(filtered_ratings))
    print("Users:", filtered_ratings["UserID"].nunique())
    print("Movies:", filtered_ratings["MovieID"].nunique(),
          " Filtered out movies:", df_ratings["MovieID"].nunique()-filtered_ratings["MovieID"].nunique())
    print("\nUpdated threshold for user ratings:", new_usercount.head(1))

    # Now on the (filtered) data used is
    ratings_df = filtered_ratings

    # Exact timestamps computed by changing manually cutoff_date and looking how
    # rating counts and percentages act -> 70%, 20%, 10% found
    train_date = "2000-11-22"
    split70 = "2000-11-23"
    validation_date = "2000-12-31"
    split90 = "2001-01-01"
    cutoff_date = split70
    cutoff_timestamp = pd.Timestamp(cutoff_date).timestamp()

    # Ratings given before cutoff
    ratings_before = ratings_df[ratings_df["Timestamp"] < cutoff_timestamp]

    # Rating counts and percentage
    ratings_before_count = len(ratings_before)
    total_ratings = len(ratings_df)
    rating_percentage = ratings_before_count / total_ratings * 100

    # user counts and percentage
    total_users = ratings_df["UserID"].nunique()
    users_before_count = ratings_before["UserID"].nunique()
    user_percentage = users_before_count / total_users * 100

    #print("Ratings before cutoff date", cutoff_date, ":", ratings_before_count)
    #print(f"Percentage of ratings before cutoff: {rating_percentage:.2f}%")

    #print("Unique users before cutoff date", cutoff_date, ":", users_before_count)
    #print(f"Percentage of users before cutoff: {user_percentage:.2f}%")

    # split into train 70% / validation 20% / test 10%
    cutoff_datetrain = "2000-11-23"
    cutoff_datetest = "2001-01-01"
    cutoff_train = pd.Timestamp(cutoff_datetrain).timestamp()
    cutoff_test = pd.Timestamp(cutoff_datetest).timestamp()

    training_set = ratings_df[ratings_df["Timestamp"] < cutoff_train]
    print("Rows in training set", len(training_set))

    validation_set = ratings_df[(ratings_df["Timestamp"] >= cutoff_train) & (ratings_df["Timestamp"] < cutoff_test)]
    print("Rows in validation set", len(validation_set))

    test_set = ratings_df[ratings_df["Timestamp"] >= cutoff_test]
    print("Rows in test set", len(test_set))