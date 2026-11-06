import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, load_count_data
from misc import kfold_indices, compute_relative_mean_sem_scores, get_best_dim_max_gain
from plotting_functions import plot_mean_scores


def compute_relative_mean_sem_scores_per_event(scores_list, dim_list, events_per_fold):
    """
    Per-event version of compute_relative_mean_sem_scores: the fold score difference to the first dimension is
    divided by the mean number of events per validation sample (ratio of means, in mnats/event).

    :param scores_list: Dictionary where keys are dimensions and values are (nb_folds, nb_val_samples) arrays of scores.
    :param dim_list: List of dimensions to evaluate, with the first dimension being the baseline.
    :param events_per_fold: (nb_folds, nb_val_samples) array of the total events (summed over neurons) of each validation sample.
    :return: Lists of the mean and SEM over folds of the relative scores per event.
    """
    mean_events = np.mean(events_per_fold, axis=1)
    mean_scores_first_dim = np.mean(scores_list[dim_list[0]], axis=1)

    mean_scores_list, sem_scores_list = [], []
    for dim in dim_list:

        # Ratio of means within each fold, in mnats/event
        mean_scores_difference = 1e3 * (np.mean(scores_list[dim], axis=1) - mean_scores_first_dim) / mean_events
        nb_cv_splits = len(mean_scores_difference)

        # Mean/sem over folds
        mean_scores_list.append(np.mean(mean_scores_difference))
        sem_scores_list.append(np.std(mean_scores_difference, ddof=1) / np.sqrt(nb_cv_splits))

    return mean_scores_list, sem_scores_list


def replot_cv_per_event(results_path, data_train, kfold_splits, fraction_threshold=0.85, out_path=None):
    """
    Replot the saved K-fold cross-validation scores in mnats/event and redo the dimension selection, without refitting.

    :param results_path: Folder with the saved *_cv_scores.pkl and best_*_dim.pkl files.
    :param data_train: Training data used for cross-validation (same samples and order as in the original run).
    :param kfold_splits: The same K-fold splits as in the original run.
    :param fraction_threshold: Fraction of the max CV gain used by get_best_dim_max_gain (same as the original run).
    :param out_path: Folder to save the plots (defaults to results_path).
    :return: Dictionary {model: (saved best dim, per-sample best dim, per-event best dim)}.
    """
    out_path = results_path if out_path is None else out_path
    os.makedirs(out_path, exist_ok=True)

    events = data_train["events"].sum(dim=1).numpy()
    events_per_fold = np.stack([events[val_idx] for _, val_idx in kfold_splits])

    models = [("event", "event_model_cv_scores.pkl", "best_event_dim.pkl", "tab:blue", "Somatic latent \n dimensions"),
              ("burst_base", "burst_model_cv_scores.pkl", "best_burst_base_dim.pkl", "tab:red", "Dendritic latent \n dimensions"),
              ("burst_hybrid", "burst_hybrid_model_cv_scores.pkl", "best_burst_hybrid_dim.pkl", "tab:red", "Dendritic latent \n dimensions")]

    selected = {}
    for name, scores_file, best_file, color, xlabel in models:
        scores_list = load_data(f"{results_path}/{scores_file}")
        dim_list = np.array(sorted(scores_list.keys()))

        # The saved per-sample scores must line up with the rebuilt validation folds
        assert scores_list[dim_list[0]].shape == events_per_fold.shape, f"{name}: scores {scores_list[dim_list[0]].shape} vs folds {events_per_fold.shape}"

        # Original per-sample selection (sanity check against the saved best dim)
        mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores(scores_list, dim_list)
        best_dim_sample = get_best_dim_max_gain(dim_list=dim_list, mean_scores_list=mean_scores_list,
                                                sem_scores_list=sem_scores_list, fraction_threshold=fraction_threshold)

        # Per-event selection
        mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores_per_event(scores_list, dim_list, events_per_fold)
        best_dim_event = get_best_dim_max_gain(dim_list=dim_list, mean_scores_list=mean_scores_list,
                                               sem_scores_list=sem_scores_list, fraction_threshold=fraction_threshold)

        plot_mean_scores(best_dim=best_dim_event, dim_list=dim_list, mean_scores_list=mean_scores_list,
                         sem_scores_list=sem_scores_list, color=color, xlabel=xlabel)
        plt.ylabel('Score \n (mnats/event)')
        plt.savefig(f"{out_path}/mean_cv_scores_{name}_per_event_plot.svg", dpi=300)
        plt.close('all')

        selected[name] = (load_data(f"{results_path}/{best_file}"), best_dim_sample, best_dim_event)

    return selected


if __name__ == "__main__":

    # Plots go next to the CV results unless an output folder is given; the best_*_dim.pkl files are never overwritten.
    out_root = sys.argv[1] if len(sys.argv) > 1 else None

    folder_name = "evidence_of_burst_coding"
    frac_intact_list = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cv_fraction = 0.1 # same fraction of the training data as cross_val_scenario
    nb_splits = 5     # same K-fold splits as cross_val_scenario

    changed = []
    for f in frac_intact_list:
        scenario = f"mixed_{f:.2f}_multiple_trials"
        results_path = f"../results/{folder_name}/{scenario}"
        data_train = load_count_data(f"{folder_name}/{scenario}/training_data", fraction=cv_fraction)
        kfold_splits = kfold_indices(nb_samples=data_train["events"].shape[0], nb_splits=nb_splits, shuffle=True)

        out_path = None if out_root is None else f"{out_root}/{scenario}"
        selected = replot_cv_per_event(results_path, data_train, kfold_splits, out_path=out_path)

        for name, (saved, sample, event) in selected.items():
            print(f"f={f:.2f}  {name:12s} saved={saved}  per-sample={sample}  per-event={event}")
            if not saved == sample == event:
                changed.append((f, name, saved, sample, event))

    print("\nSelected dims unchanged for every fraction and model." if not changed else f"\nChanged: {changed}")