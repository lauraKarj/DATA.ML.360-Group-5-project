"""
Run_all.py document which runs the whole project
"""

import argparse
from item_knn import (run, _load_split)
from data_preparation import (load_data, characterize_and_plot, filter_and_split)

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where the CSV results are written")
    parser.add_argument("--fig-dir", default="figures", help="where the figures are written")
    args = parser.parse_args()

    df_movies, df_ratings, df_users = load_data(args.data_dir)
    characterize_and_plot(df_movies,df_ratings,df_users,fig_dir=args.fig_dir)
    filter_and_split(df_ratings)

    run(*_load_split(args.data_dir), out_dir=args.out_dir, fig_dir=args.fig_dir)


if __name__ == "__main__":
    main()
