"""Mult-DAE: a denoising autoencoder with a multinomial likelihood (Part 2, self-implemented).

From run_all.py, after the split:
    import mult_dae
    mult_dae.run(training_set, validation_set, test_set)

Standalone:
    python mult_dae.py --data-dir ml-1m --out-dir results --fig-dir figures
"""
import torch

import argparse
import copy
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import item_knn
import most_popular
from evaluation import (SEED, TOP_N, catalogue_index, evaluate, evaluation_set, interaction_matrix,
                        paired_bootstrap)

HIDDEN = {"200": [200], "600-200": [600, 200]}
DROPOUT = [0.2, 0.5]
L2 = [0.0, 0.001, 0.01, 0.1]
LEARNING_RATE = 0.001
BATCH_SIZE = 500
MAX_EPOCHS = 200
PATIENCE = 20
TEST_SEEDS = [42, 43, 44, 45, 46]
KNN_REFERENCE = ("implicit", "cosine", 800)
METRIC = f"ndcg@{TOP_N}"


class MultDAE(torch.nn.Module):
    """Encoder n_items -> hidden[0] -> ... -> hidden[-1] and a mirrored decoder back to n_items.
    tanh between layers, no activation on the output logits. Weights use Glorot-uniform initialisation."""

    def __init__(self, n_items, hidden, dropout, generator):
        super().__init__()
        sizes = [n_items, *hidden, *hidden[-2::-1], n_items]
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6 / (fan_in + fan_out))
            W = (torch.rand(fan_in, fan_out, generator=generator) * 2 - 1) * bound
            self.weights.append(torch.nn.Parameter(W))
            self.biases.append(torch.nn.Parameter(torch.zeros(fan_out)))
        self.dropout = dropout

    def forward(self, x, generator=None):
        """Logits for every movie. In training mode each input entry is dropped with probability
        `dropout` and the survivors are scaled by 1 / (1 - dropout) (the denoising corruption)."""
        h = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.training and self.dropout > 0:
            keep = torch.rand(h.shape, generator=generator) >= self.dropout
            h = h * keep / (1 - self.dropout)
        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if layer < len(self.weights) - 1:
                h = torch.tanh(h)
        return h


def multinomial_loss(logits, x, model, l2):
    """Mean over users of -sum_i x_ui * log softmax(logits_u)_i, plus l2 * the sum of squared weights."""
    negative_log_likelihood = -(torch.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
    penalty = sum((W ** 2).sum() for W in model.weights)
    return negative_log_likelihood + l2 * penalty


def score(model, X_rows):
    """Recommendation scores: the softmax over all movies for each given user row."""
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X_rows.toarray())), dim=1).numpy()


def fit(X, config, seed, epochs=None, validate=None):
    """Train on every user row of X (users x movies, binary) that has at least one positive movie.

    With `validate` (a function model -> validation metric), the metric is computed after every epoch,
    training stops after PATIENCE epochs without improvement and the best epoch's weights are restored.
    Without it, exactly `epochs` epochs are run. Returns the model and the per-epoch metric."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultDAE(X.shape[1], HIDDEN[config["hidden"]], config["dropout"], generator)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rows = np.flatnonzero(X.getnnz(axis=1) > 0)

    curve, best, best_epoch, best_state = [], -np.inf, 0, None
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        order = rng.permutation(rows)
        for start in range(0, len(order), BATCH_SIZE):
            x = torch.from_numpy(X[order[start:start + BATCH_SIZE]].toarray())
            loss = multinomial_loss(model(x, generator), x, model, config["l2"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if validate is None:
            continue
        curve.append(validate(model))
        if curve[-1] > best:
            best, best_epoch, best_state = curve[-1], epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve


def config_name(config):
    return f"MultDAE-h{config['hidden']}-d{config['dropout']}-l2{config['l2']}"


def compare(model, reference):
    """Paired per-user NDCG comparison: mean difference, bootstrap CI, Wilcoxon p and win shares."""
    d = (model["ndcg"] - reference["ndcg"]).to_numpy()
    ci_low, ci_high = paired_bootstrap(d)
    return {"model_ndcg": model["ndcg"].mean(), "reference_ndcg": reference["ndcg"].mean(),
            "mean_diff": d.mean(), "ci_low": ci_low, "ci_high": ci_high,
            "wilcoxon_p": stats.wilcoxon(d).pvalue, "model_wins": (d > 0).mean(),
            "ties": (d == 0).mean(), "reference_wins": (d < 0).mean()}


def plot_curves(curves, selected, references, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, curve in curves.items():
        is_selected = name == selected
        ax.plot(range(1, len(curve) + 1), curve, linewidth=2 if is_selected else 0.8,
                alpha=1 if is_selected else 0.4, color="C0" if is_selected else "grey",
                label=f"{name} (selected)" if is_selected else None)
    for (name, value), style in zip(references.items(), ["--", ":"]):
        ax.axhline(value, color="black", linestyle=style, linewidth=1.2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"NDCG@{TOP_N} (validation)")
    ax.set_title("Mult-DAE: validation NDCG@10 per epoch (grey: other configurations)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def knn_reference(history, eval_set, users, items):
    """Scores of the item-kNN configuration selected in Part 1 (implicit cosine, k = 800)."""
    feedback, measure, k = KNN_REFERENCE
    X = interaction_matrix(history, users, items, feedback)
    S_k = item_knn.prune(item_knn.item_similarity(X, measure), k)
    return item_knn.knn_scores(item_knn.history_weights(X, feedback)[eval_set.rows], S_k)


def run(train, validation, test, out_dir="results", fig_dir="figures"):
    """1) grid search on validation (fitted on train, epochs by early stopping), 2) selection by
    validation NDCG@10, 3) refit on train + validation with TEST_SEEDS and evaluate once on test,
    4) per-user comparison with item-kNN and Most Popular on test."""
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    users, items = catalogue_index(train, validation, test)

    X_train = interaction_matrix(train, users, items, "implicit")
    val_set = evaluation_set(train, validation, users, items)
    X_val_rows = X_train[val_set.rows]

    def validate(model):
        return evaluate(score(model, X_val_rows), val_set)[0][METRIC]

    rows, curves = [], {}
    for hidden, dropout, l2 in product(HIDDEN, DROPOUT, L2):
        start = time.time()
        config = {"hidden": hidden, "dropout": dropout, "l2": l2}
        model, curve = fit(X_train, config, SEED, validate=validate)
        summary = evaluate(score(model, X_val_rows), val_set)[0]
        name = config_name(config)
        curves[name] = curve
        rows.append({"model": name, **config, "best_epoch": int(np.argmax(curve)) + 1,
                     "epochs_run": len(curve), **summary, "seconds": round(time.time() - start, 1)})
        print(f"{name}: validation {METRIC} {summary[METRIC]:.4f} at epoch {rows[-1]['best_epoch']} "
              f"({rows[-1]['seconds']} s)")
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "mult_dae_validation.csv", index=False)

    selected = grid.loc[grid[METRIC].idxmax()]
    config = {"hidden": selected["hidden"], "dropout": float(selected["dropout"]), "l2": float(selected["l2"])}
    epochs = int(selected["best_epoch"])
    print(f"\nSelected on validation: {selected['model']}, {epochs} epochs, {METRIC} {selected[METRIC]:.4f}")

    val_references = {
        "ItemKNN-implicit-cosine-k800": evaluate(knn_reference(train, val_set, users, items), val_set)[0][METRIC],
        "MostPop-all_time": evaluate(most_popular.all_time_scores(train, items, len(val_set.rows)),
                                     val_set)[0][METRIC],
    }
    plot_curves(curves, selected["model"], val_references, fig_dir / "mult_dae_validation_curves.png")

    history = pd.concat([train, validation])
    X_history = interaction_matrix(history, users, items, "implicit")
    test_set = evaluation_set(history, test, users, items)
    seed_rows, seed_users = [], []
    for seed in TEST_SEEDS:
        model, _ = fit(X_history, config, seed, epochs=epochs)
        summary, per_user, _ = evaluate(score(model, X_history[test_set.rows]), test_set)
        seed_rows.append({"model": selected["model"], "seed": seed, "epochs": epochs, **summary})
        seed_users.append(per_user)
    seeds = pd.DataFrame(seed_rows)
    metrics = [f"precision@{TOP_N}", f"recall@{TOP_N}", METRIC, "coverage"]
    t = stats.t.ppf(0.975, len(seeds) - 1)
    spread = seeds[metrics].std(ddof=1) * t / np.sqrt(len(seeds))
    seeds.to_csv(out_dir / "mult_dae_test_seeds.csv", index=False)
    print(seeds[["seed", *metrics]].round(4).to_string(index=False))
    print("Mean over seeds (95% t-interval half-width):")
    print(pd.DataFrame({"mean": seeds[metrics].mean(), "ci_half_width": spread}).round(4).to_string())

    dae_users = pd.concat(seed_users).groupby(level=0).mean()
    references = {
        "ItemKNN-implicit-cosine-k800": knn_reference(history, test_set, users, items),
        "MostPop-all_time": most_popular.all_time_scores(history, items, len(test_set.rows)),
    }
    comparison = []
    for name, scores in references.items():
        summary, per_user, _ = evaluate(scores, test_set)
        comparison.append({"model": selected["model"] + " (mean of seeds)", "reference": name,
                           **compare(dae_users, per_user),
                           "seeds_above_reference": int((seeds[METRIC] > summary[METRIC]).sum())})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out_dir / "mult_dae_vs_baselines.csv", index=False)
    print("\nTest, per-user NDCG@10 of Mult-DAE (averaged over seeds) minus the reference:")
    print(comparison.drop(columns="model").round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="ml-1m", help="folder containing ratings.dat")
    parser.add_argument("--out-dir", default="results", help="where the CSV results are written")
    parser.add_argument("--fig-dir", default="figures", help="where the figures are written")
    args = parser.parse_args()
    run(*item_knn._load_split(args.data_dir), out_dir=args.out_dir, fig_dir=args.fig_dir)


if __name__ == "__main__":
    main()
