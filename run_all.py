"""
Run_all.py document which runs the whole project
"""

import argparse
from item_knn import (run, _load_split)
from data_preparation import (load_data, run_preparation)
from most_popular import (run_mostpop)

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where the CSV results are written")
    parser.add_argument("--fig-dir", default="figures", help="where the figures are written")
    args = parser.parse_args()

    # Load df
    df_movies, df_ratings, df_users = load_data(args.data_dir)
    # Run data preparation and distribution plots
    run_preparation(df_movies, df_ratings, df_users, args.fig_dir)
    # Run item_knn and evaluation
    run(*_load_split(args.data_dir), out_dir=args.out_dir, fig_dir=args.fig_dir)
    # Run most_popular and evaluation
    run_mostpop(*_load_split(args.data_dir),out_dir=args.out_dir,)

if __name__ == "__main__":
    main()
