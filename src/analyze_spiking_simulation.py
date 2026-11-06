import torch
import matplotlib.pyplot as plt
import numpy as np
np.seterr(invalid='ignore', divide='ignore') 
import seaborn as sns
import os
from plotting_functions import plot_score_difference_vs_fraction, plot_dims_vs_fraction

from utils import load_data, save_data, set_seed, load_count_data

device = torch.device('cpu')

from tqdm import tqdm
from models import fit_event_model, fit_burst_multiplexed_model, fit_burst_hybrid_model, fit_burst_null_model, log_evidence_scores, sample_posterior_predictive
from misc import kfold_indices, compute_relative_mean_sem_scores, get_best_dim_max_gain, compute_data_statistics, compute_data_statistics_multiple_datasets, convert_data_for_plot
from misc import bootstrap_confidence_interval
from plotting_functions import plot_mean_scores, plot_training_dynamics, plot_score_difference, plot_summary_statistics
import gc
from metrics import estimate_projection_distance



def cross_val_score_VI(model_fit_type, data, nb_latents, kfold_splits, training_params,
                       nb_posterior_samples=1000, print_fold_scores=False, print_loss=False, fit_burst_model=None):
    """
    Perform k-fold cross-validation for models trained using variational inference.
    
    :param model_fit_type: Type of model to fit ("event_model", "burst_multiplexed_model", "burst_null_model", "burst_hybrid_model").
    :param data: Dictionary containing 'events' and 'bursts' data as PyTorch tensors.
    :param nb_latents: Number of latent variables in the model.
    :param kfold_splits: K-fold splits as a list of (train_idx, val_idx) tuples.
    :param training_params: Dictionary containing training parameters for the model.
    :param nb_posterior_samples: Number of posterior samples for log evidence computation.
    :param print_fold_scores: Whether to print log evidence scores for each fold.
    :param print_loss: Whether to print the training loss during model fitting.
    :param fit_burst_model: Function to fit the burst model (required if model_fit_type is burst model).
    :return: List of log evidence scores for each fold.
    """

    nb_splits = len(kfold_splits)

    scores = []
    for fold_idx, (train_idx, val_idx) in tqdm(enumerate(kfold_splits), desc="Cross-validation folds", total=nb_splits, leave=False):

        # Setup data for this fold
        events_train, events_val = data["events"][train_idx], data["events"][val_idx]
        bursts_train, bursts_val = data["bursts"][train_idx], data["bursts"][val_idx]

        data_train = {"events": events_train, "bursts": bursts_train}
        data_val = {"events": events_val, "bursts": bursts_val}

        if model_fit_type == "event_model": # If only event model fitting

            event_model_train, _, event_model_val, _ = fit_event_model(data_train=data_train, data_test=data_val, nb_latents=nb_latents, 
                                                                    training_params=training_params, device=device, print_loss=print_loss)

            log_evidence_score = log_evidence_scores(model=event_model_val, data=data_val, device=device, nb_posterior_samples=nb_posterior_samples, 
                                         event_model=True)
            
            del event_model_train, event_model_val
    
        else: # If the burst multiplexed model or burst null model fitting

            event_model_train, _ = fit_event_model(data_train=data_train, data_test=None, nb_latents=nb_latents, 
                                                                    training_params=training_params, device=device, print_loss=print_loss)

            burst_model_train, _, burst_model_val, _ = fit_burst_model(data_train=data_train, data_test=data_val, event_model=event_model_train, 
                                                                    nb_latents=nb_latents, training_params=training_params, device=device, print_loss=print_loss)

            log_evidence_score = log_evidence_scores(model=burst_model_val, data=data_val, device=device, nb_posterior_samples=nb_posterior_samples, 
                                         event_model=False)
            
            del event_model_train, burst_model_train, burst_model_val

        if print_fold_scores:
            log_evidence_score_mean = log_evidence_score.mean().item()
            print(f"Fold {fold_idx+1}/{nb_splits} - Mean log evidence score: {log_evidence_score_mean:.4f}")

        scores.append(log_evidence_score.cpu().numpy())

        # Free up memory after each fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    scores = np.asarray(scores)

    return scores

def cross_val_dataset_burst_hybrid_model(data_train, best_event_dim, burst_dim_list, kfold_splits, training_params,
                                          results_path, fraction_threshold=0.85):
    """
    Cross-validate the hybrid model on the given training data.
    
    :param data_train: Training data for cross-validation.
    :param best_event_dim: Best event dimension to use in the burst multiplexed model.
    :param burst_dim_list: List of burst dimensions to cross-validate.
    :param kfold_splits: Indices for k-fold cross-validation.
    :param training_params: Training parameters for the model.
    :param results_path: Path to save the cross-validation results.
    """


    # Fit the hybrid model
    scores_burst_list = {}
    for burst_dim in tqdm(burst_dim_list, desc="Burst dimensions"):
        nb_latents = {"events": best_event_dim, "bursts": burst_dim}
        scores = cross_val_score_VI(model_fit_type='burst_hybrid_model', data=data_train,
                                nb_latents=nb_latents, kfold_splits=kfold_splits, training_params=training_params,
                                nb_posterior_samples=1000, print_fold_scores=False, print_loss=False, fit_burst_model=fit_burst_hybrid_model)
        scores_burst_list[burst_dim] = scores

    # Save the data for later plotting
    save_data(scores_burst_list, filename=f"{results_path}/burst_hybrid_model_cv_scores.pkl")

    # Compute the best dimensions from the cross-validation results for both datasets
    mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores(scores_burst_list, burst_dim_list)

    # Determine the best event dimension based on the mean relative scores
    best_burst_dim = get_best_dim_max_gain(dim_list=burst_dim_list, mean_scores_list=mean_scores_list, 
                                       sem_scores_list=sem_scores_list, fraction_threshold=fraction_threshold)

    print(f"Best burst dimension: {best_burst_dim}")
    save_data(best_burst_dim, filename=f"{results_path}/best_burst_hybrid_dim.pkl")

    # Plot the mean relative scores with error bars
    plot_mean_scores(best_dim=best_burst_dim, dim_list=burst_dim_list, 
                 mean_scores_list=mean_scores_list, sem_scores_list=sem_scores_list,
                   color='tab:red', file_name=f"{results_path}/mean_cv_scores_burst_hybrid_plot.svg", xlabel="Dendritic latent \n dimensions", xticks=None)

    return best_burst_dim

def cross_val_dataset_burst_multiplexed_model(data_train, best_event_dim, burst_dim_list, kfold_splits, training_params, 
                                              results_path, fraction_threshold=0.85):
    """
    Cross-validate the base model on the given training data.
    
    :param data_train: Training data for cross-validation.
    :param best_event_dim: Best event dimension to use in the burst multiplexed model.
    :param burst_dim_list: List of burst dimensions to cross-validate.
    :param kfold_splits: Indices for k-fold cross-validation.
    :param training_params: Training parameters for the model.
    :param results_path: Path to save the cross-validation results.
    """


    # Fit the burst multiplexed model
    scores_burst_list = {}
    for burst_dim in tqdm(burst_dim_list, desc="Burst dimensions"):
        nb_latents = {"events": best_event_dim, "bursts": burst_dim}
        scores = cross_val_score_VI(model_fit_type='burst_multiplexed_model', data=data_train,
                                nb_latents=nb_latents, kfold_splits=kfold_splits, training_params=training_params,
                                nb_posterior_samples=1000, print_fold_scores=False, print_loss=False, fit_burst_model=fit_burst_multiplexed_model)
        scores_burst_list[burst_dim] = scores

    # Save the data for later plotting
    save_data(scores_burst_list, filename=f"{results_path}/burst_model_cv_scores.pkl")

    # Compute the best dimensions from the cross-validation results for both datasets
    mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores(scores_burst_list, burst_dim_list)

    # Determine the best event dimension based on the mean relative scores
    best_burst_dim = get_best_dim_max_gain(dim_list=burst_dim_list, mean_scores_list=mean_scores_list, 
                                       sem_scores_list=sem_scores_list, fraction_threshold=fraction_threshold)

    print(f"Best burst dimension: {best_burst_dim}")
    save_data(best_burst_dim, filename=f"{results_path}/best_burst_base_dim.pkl")

    # Plot the mean relative scores with error bars
    plot_mean_scores(best_dim=best_burst_dim, dim_list=burst_dim_list, 
                 mean_scores_list=mean_scores_list, sem_scores_list=sem_scores_list,
                   color='tab:red', file_name=f"{results_path}/mean_cv_scores_burst_base_plot.svg", xlabel="Dendritic latent \n dimensions", xticks=None)

    return best_burst_dim

def cross_val_dataset_event_model(data_train, event_dim_list, kfold_splits, training_params,results_path, fraction_threshold=0.85):

    """
    Cross-validate the event model on the given training data.
    
    :param data_train: Training data for cross-validation.
    :param event_dim_list: List of event dimensions to cross-validate.
    :param kfold_splits: Indices for k-fold cross-validation.
    :param training_params: Training parameters for the model.
    :param results_path: Path to save the cross-validation results.
    """

    # Fit the event model 
    scores_event_list = {}
    for event_dim in tqdm(event_dim_list, desc="Event dimensions"):
        nb_latents = {"events": event_dim, "bursts": None}
        scores = cross_val_score_VI(model_fit_type='event_model', data=data_train,
                                nb_latents=nb_latents, kfold_splits=kfold_splits, training_params=training_params,
                                nb_posterior_samples=1000, print_fold_scores=False, print_loss=False, fit_burst_model=None)
        scores_event_list[event_dim] = scores

    # Save the scores 
    save_data(scores_event_list, filename=f"{results_path}/event_model_cv_scores.pkl")

    # Compute the best dimensions from the cross-validation results for both datasets
    mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores(scores_event_list, event_dim_list)

    # Determine the best event dimension based on the mean relative scores
    best_event_dim = get_best_dim_max_gain(dim_list=event_dim_list, mean_scores_list=mean_scores_list, 
                                       sem_scores_list=sem_scores_list, fraction_threshold=fraction_threshold)

    print(f"Best event dimension: {best_event_dim}")
    save_data(best_event_dim, filename=f"{results_path}/best_event_dim.pkl")

    # Plot the mean relative scores with error bars
    plot_mean_scores(best_dim=best_event_dim, dim_list=event_dim_list, 
                 mean_scores_list=mean_scores_list, sem_scores_list=sem_scores_list,
                   color='tab:blue', file_name=f"{results_path}/mean_cv_scores_event_plot.svg", xlabel="Somatic latent \n dimensions", xticks=None)

    return best_event_dim

def fit_dataset(data_train, data_test, nb_latents_base, nb_latents_hybrid, results_path):
    """
    Fit the dataset using the event, base, hybrid, and null models.
    
    :param data_train: Training data.
    :param data_test: Test data.
    :param nb_latents_base: Number of latent dimensions for the base model.
    :param nb_latents_hybrid: Number of latent dimensions for the hybrid model.
    :param results_path: Path to save the results.
    :return: Dictionary containing the fitted models and their statistics.
    """

    training_params = {"nb_iter": 5_000, "batch_size": None, "lr": 1e-2, "final_ratio": 1e-4}

    print('Train event model only on training data...')
    event_model_train, events_stats_train = fit_event_model(data_train=data_train, data_test=None, nb_latents=nb_latents_base, 
                                                        training_params=training_params, device=device, print_loss=False)

    print('Train multiplexed model on training and test data...')
    multiplexed_model_train, multiplexed_stats_train, multiplexed_model_test, multiplexed_stats_test = fit_burst_multiplexed_model(data_train=data_train,
                                                                                                                                data_test=data_test, 
                                                                                                                                event_model=event_model_train, 
                                                                                                                                nb_latents=nb_latents_base, 
                                                                                                                                training_params=training_params, 
                                                                                                                                device=device, print_loss=False)


    print('Train null model on training and test data...')
    null_model_train, null_stats_train, null_model_test, null_stats_test = fit_burst_null_model(data_train=data_train,
                                                                                                        data_test=data_test, 
                                                                                                        event_model=event_model_train, 
                                                                                                        nb_latents=nb_latents_base, 
                                                                                                        training_params=training_params, 
                                                                                                        device=device, print_loss=False)

    print('Train hybrid model on training and test data...')
    hybrid_model_train, stats_hist_train, hybrid_model_test, hybrid_stats_test = fit_burst_hybrid_model(data_train=data_train,
                                                                                                        data_test=data_test, 
                                                                                                        event_model=event_model_train, 
                                                                                                        nb_latents=nb_latents_hybrid, 
                                                                                                        training_params=training_params, 
                                                                                                        device=device, print_loss=False)
    
    results = {
        "event_model_train": event_model_train,
        "events_stats_train": events_stats_train,
        "multiplexed_model_train": multiplexed_model_train,
        "multiplexed_stats_train": multiplexed_stats_train,
        "multiplexed_model_test": multiplexed_model_test,
        "multiplexed_stats_test": multiplexed_stats_test,
        "null_model_train": null_model_train,
        "null_stats_train": null_stats_train,
        "null_model_test": null_model_test,
        "null_stats_test": null_stats_test,
        "hybrid_model_train": hybrid_model_train,
        "hybrid_stats_train": stats_hist_train,
        "hybrid_model_test": hybrid_model_test,
        "hybrid_stats_test": hybrid_stats_test
    }


    save_data(results, f"{results_path}/model_fitting_results.pkl")

    return


## Model fitting evaluation functions
def evaluate_subspace_recovery(fit_results, true_weights, results_path):
    """
    Evaluate the subspace recovery of the learned models by comparing the learned weights to the true weights.
    
    :param fit_results: Dictionary containing the fitted models and their statistics.
    :param true_weights: Dictionary containing the true weight matrices.
    :param results_path: Path to save the projection distances.
    """

    # Learned weights - base model
    W_event_learned_multiplexed = fit_results['multiplexed_model_train'].event_decoder.W.detach().cpu().numpy()
    W_burst_learned_multiplexed = fit_results['multiplexed_model_train'].burst_decoder.W.detach().cpu().numpy()

    # Estimate the projection distance between the true and estimated event latent spaces
    event_distance = estimate_projection_distance(true_weights['W_s'].cpu().numpy(), W_event_learned_multiplexed)
    print(f"Event projection distance: {event_distance:.4f}")

    burst_distance = estimate_projection_distance(true_weights['W_d'].cpu().numpy(), W_burst_learned_multiplexed)
    print(f"Burst projection distance: {burst_distance:.4f}")


    # Learned weights - hybrid model
    W_event_learned_hybrid = fit_results['hybrid_model_train'].event_decoder.W.detach().cpu().numpy()
    W_burst_learned_hybrid = fit_results['hybrid_model_train'].burst_decoder.W.detach().cpu().numpy()

    # Estimate the projection distance between the true and estimated event latent spaces
    event_distance_hybrid = estimate_projection_distance(true_weights['W_s'].cpu().numpy(), W_event_learned_hybrid)
    print(f"Hybrid model event projection distance: {event_distance_hybrid:.4f}")

    burst_distance_hybrid = estimate_projection_distance(true_weights['W_d'].cpu().numpy(), W_burst_learned_hybrid)
    print(f"Hybrid model burst projection distance: {burst_distance_hybrid:.4f}")

    # Save the projection distances for later analysis
    projection_distances = {
        "event_distance": event_distance,
        "burst_distance": burst_distance,
        "event_distance_hybrid": event_distance_hybrid,
        "burst_distance_hybrid": burst_distance_hybrid
    }
    save_data(projection_distances, f"{results_path}/projection_distances.pkl")

    return

def evaluate_model_performance(fit_results, data_test, file_name, ylim=None, yticks=None):
    """
    Evaluate the performance of different models on test data by computing log-evidence scores and plotting score differences.
    
    :param fit_results: Dictionary containing the fitted models for evaluation.
    :param data_test: Test dataset to evaluate the models on.
    :param file_name: File name to save the score difference plot.
    :param ylim: Y-axis limits for the plot.
    :param yticks: Y-axis tick values for the plot.
    """

    scores_multiplexed_model = log_evidence_scores(model=fit_results['multiplexed_model_test'], data=data_test, device=device, 
                                                   nb_posterior_samples=1000, event_model=False)

    scores_null_model = log_evidence_scores(model=fit_results['null_model_test'], data=data_test, device=device, 
                                            nb_posterior_samples=1000, event_model=False)

    scores_hybrid_model = log_evidence_scores(model=fit_results['hybrid_model_test'], data=data_test, device=device, 
                                             nb_posterior_samples=1000, event_model=False)


    # Mean number of events per test sample (summed over neurons)
    events_per_sample = data_test["events"].sum(dim=1)
    print(f"Mean events per sample: {events_per_sample.mean().item():.2f}")

    # Per sample (nats/sample)
    results_sample = plot_score_difference(scores_multiplexed=scores_multiplexed_model, scores_null=scores_null_model, scores_hybrid=scores_hybrid_model,
                                           file_name=file_name, ylim=ylim, yticks=yticks)

    # Per event (mnats/event); y-limits are set automatically since the scale differs
    results_event = plot_score_difference(scores_multiplexed=scores_multiplexed_model, scores_null=scores_null_model, scores_hybrid=scores_hybrid_model,
                                          events_per_sample=events_per_sample, file_name=file_name.replace(".svg", "_per_event.svg"), ylim=ylim, yticks=yticks)

    for name, (e, lo, hi, p), (ee, elo, ehi, ep) in zip(["Base", "Hybrid"], results_sample, results_event):
        print(f"{name} - null: {e:+.4f} [{lo:+.4f}, {hi:+.4f}] nats/sample (p={p:.4f}), "
              f"{ee:+.2f} [{elo:+.2f}, {ehi:+.2f}] mnats/event (p={ep:.4f})")

    return

def posterior_predictive_checks(fit_results, data_test, results_path):
    """
    Perform posterior predictive checks by comparing the statistics of the true test data with the statistics of data generated from the posterior predictive distribution of the fitted models.
        
    :param fit_results: Dictionary containing the fitted model results.
    :param data_test: Test data to compare against the posterior predictive samples.
    :param results_path: Path to save the resulting plots.
    """
    
    # True data statistics
    statistics_data_test = compute_data_statistics(convert_data_for_plot(data_test))

    # Posterior predictive statistics for the multiplexed and null models
    data_test_posterior_multiplexed_model = sample_posterior_predictive(fit_results['multiplexed_model_test'],
                                                                                    nb_posterior_samples=1000, event_model=False)

    data_test_posterior_null_model = sample_posterior_predictive(fit_results['null_model_test'],
                                                                                    nb_posterior_samples=1000, event_model=False)      

    data_test_posterior_hybrid_model = sample_posterior_predictive(fit_results['hybrid_model_test'],
                                                                                    nb_posterior_samples=1000, event_model=False)


    statistics_multiplexed_model_test = compute_data_statistics_multiple_datasets(data_test_posterior_multiplexed_model)
    statistics_null_model_test = compute_data_statistics_multiple_datasets(data_test_posterior_null_model)
    statistics_hybrid_model_test = compute_data_statistics_multiple_datasets(data_test_posterior_hybrid_model)

    # Base model vs. null model
    plot_summary_statistics(statistics_data=statistics_data_test["events"], 
                        statistics_multiplexed=statistics_multiplexed_model_test["events"], 
                        statistics_null=statistics_null_model_test["events"],
                        color_data='tab:blue', 
                        file_name=f'{results_path}/summary_statistics_events_base.svg')

    plot_summary_statistics(statistics_data=statistics_data_test["bursts"], 
                            statistics_multiplexed=statistics_multiplexed_model_test["bursts"], 
                            statistics_null=statistics_null_model_test["bursts"],
                            color_data='tab:orange',
                            file_name=f'{results_path}/summary_statistics_bursts_base.svg')

    # Hybrid model vs. null model
    plot_summary_statistics(statistics_data=statistics_data_test["events"], 
                        statistics_multiplexed=statistics_hybrid_model_test["events"], 
                        statistics_null=statistics_null_model_test["events"],
                        color_data='tab:blue', 
                        file_name=f'{results_path}/summary_statistics_events_hybrid.svg')

    plot_summary_statistics(statistics_data=statistics_data_test["bursts"], 
                            statistics_multiplexed=statistics_hybrid_model_test["bursts"], 
                            statistics_null=statistics_null_model_test["bursts"],
                            color_data='tab:orange',
                            file_name=f'{results_path}/summary_statistics_bursts_hybrid.svg')


    return

def replot_mean_scores(event_dim_list, burst_dim_list, results_path, fraction_threshold=0.85):

    # Load the scores from the cross-validation results
    event_scores = load_data(f"{results_path}/event_model_cv_scores.pkl")
    burst_scores = load_data(f"{results_path}/burst_model_cv_scores.pkl")

    mean_scores_event_list = event_scores['mean_scores_event_list']
    sem_scores_event_list = event_scores['sem_scores_event_list']
    mean_scores_burst_list = burst_scores['mean_scores_burst_list']
    sem_scores_burst_list = burst_scores['sem_scores_burst_list']

    # Plot the scores again
    best_event_dim = plot_mean_scores(event_dim_list, mean_scores_event_list, sem_scores_event_list, fraction_threshold=fraction_threshold,
                    color='tab:blue', xlabel='Somatic latent \n dimensions',
                    file_name=f"{results_path}/event_model_cv_scores.svg",
                    xticks=event_dim_list)

    best_burst_dim = plot_mean_scores(burst_dim_list, mean_scores_burst_list, sem_scores_burst_list, fraction_threshold=fraction_threshold,
                    color='tab:red', xlabel='Dendritic latent \n dimensions',
                    file_name=f"{results_path}/burst_model_cv_scores.svg",
                    xticks=burst_dim_list)
    
    print(f"Best event dim: {best_event_dim}, Best burst dim: {best_burst_dim}")


    return



## Wrapper functions
def cross_vall_all_models(folder_name, fraction=0.1):
    """
    Cross validation wrapper for all models in the spiking simulation study.
    """
  
    # Load a fraction of the count training data (0.1 -> 1000 samples) for faster testing of the cross-validation procedure.
    # Use the full dataset for the final model fitting and evaluation.
    two_compartment_data_train_fraction = load_count_data(f"{folder_name}/two_compartments_multiple_trials/training_data", fraction=fraction)
    single_compartment_data_train_fraction = load_count_data(f"{folder_name}/single_compartment_multiple_trials/training_data", fraction=fraction)

    two_compartment_data_test_fraction = load_count_data(f"{folder_name}/two_compartments_multiple_trials/testing_data", fraction=fraction)
    single_compartment_data_test_fraction = load_count_data(f"{folder_name}/single_compartment_multiple_trials/testing_data", fraction=fraction)
    
    ################## Cross-validation ########################
    # Dimensions to test
    event_dim_list = np.array([1, 2, 3, 4, 5, 6])
    burst_dim_list = np.array([1, 2, 3, 4, 5, 6])
    burst_dim_hybrid_list = np.array([0, 1, 2, 3, 4, 5, 6])

    # Generate K-fold splits for cross-validation
    kfold_params = {"nb_splits": 5, "shuffle": True}
    nb_samples_train, _ = two_compartment_data_train_fraction["events"].shape
    kfold_splits = kfold_indices(nb_samples=nb_samples_train, nb_splits=kfold_params["nb_splits"], shuffle=kfold_params["shuffle"])

    # Parameters for training
    training_params = {"nb_iter": 5_000, "batch_size": None, "lr": 1e-2, "final_ratio": 1e-4}

    # Run cross-validation for the two-compartment data
    best_event_dim_two_compartment = cross_val_dataset_event_model(two_compartment_data_train_fraction, event_dim_list, kfold_splits, training_params, 
                                  results_path=f"../results/{folder_name}/two_compartments_multiple_trials")

    # Run cross-validation for the single-compartment data
    best_event_dim_single_compartment = cross_val_dataset_event_model(single_compartment_data_train_fraction, event_dim_list, kfold_splits, training_params,
                                  results_path=f"../results/{folder_name}/single_compartment_multiple_trials")


    # Run cross-validation for the burst multiplexed model on the two-compartment data
    best_burst_dim_two_compartment = cross_val_dataset_burst_multiplexed_model(two_compartment_data_train_fraction, best_event_dim_two_compartment, burst_dim_list, kfold_splits, training_params,
                                  results_path=f"../results/{folder_name}/two_compartments_multiple_trials")

    # Run cross-validation for the burst multiplexed model on the single-compartment data
    best_burst_dim_single_compartment = cross_val_dataset_burst_multiplexed_model(single_compartment_data_train_fraction, best_event_dim_single_compartment, burst_dim_list, kfold_splits, training_params,
                                  results_path=f"../results/{folder_name}/single_compartment_multiple_trials")


    # Run cross-validation for the hybrid model on the two-compartment data
    best_hybrid_dim_two_compartment = cross_val_dataset_burst_hybrid_model(two_compartment_data_train_fraction, best_event_dim_two_compartment, burst_dim_hybrid_list, kfold_splits, training_params,
                                  results_path=f"../results/{folder_name}/two_compartments_multiple_trials")

    # Run cross-validation for the hybrid model on the single-compartment data
    best_hybrid_dim_single_compartment = cross_val_dataset_burst_hybrid_model(single_compartment_data_train_fraction, best_event_dim_single_compartment, burst_dim_hybrid_list, kfold_splits, training_params,
                                  results_path=f"../results/{folder_name}/single_compartment_multiple_trials")


    #replot_mean_scores(event_dim_list=event_dim_list, burst_dim_list=burst_dim_list, 
    #                   results_path=f"../results/{folder_name}/two_compartments_multiple_trials", 
    #                   fraction_threshold=0.85)
    
    #replot_mean_scores(event_dim_list=event_dim_list, burst_dim_list=burst_dim_list, 
    #                   results_path=f"../results/{folder_name}/single_compartment_multiple_trials", 
    #                   fraction_threshold=0.85)


    return

def fit_all_models(folder_name, fraction=1):
    """
    Wrapper to fit all models on the training data using the best dimensions obtained from cross-validation.
    
    :param folder_name: The name of the folder containing the cross-validation results and where the fitted model results will be saved.
    :param fraction: The fraction of the dataset to use for training and testing.
    """
    two_compartment_data_train = load_count_data(f"{folder_name}/two_compartments_multiple_trials/training_data", fraction=fraction)
    single_compartment_data_train = load_count_data(f"{folder_name}/single_compartment_multiple_trials/training_data", fraction=fraction)

    two_compartment_data_test = load_count_data(f"{folder_name}/two_compartments_multiple_trials/testing_data", fraction=fraction)
    single_compartment_data_test = load_count_data(f"{folder_name}/single_compartment_multiple_trials/testing_data", fraction=fraction)


    ## Two-comapartment simulations
    # Load the best dimensions from the cross-validation results
    best_event_dim_two_compartment = load_data(f"../results/{folder_name}/two_compartments_multiple_trials/best_event_dim.pkl")
    best_burst_base_dim_two_compartment = load_data(f"../results/{folder_name}/two_compartments_multiple_trials/best_burst_base_dim.pkl")
    best_burst_hybrid_dim_two_compartment = load_data(f"../results/{folder_name}/two_compartments_multiple_trials/best_burst_hybrid_dim.pkl")

    nb_latents_base = {"events": best_event_dim_two_compartment, "bursts": best_burst_base_dim_two_compartment}
    nb_latents_hybrid = {"events": best_event_dim_two_compartment, "bursts": best_burst_hybrid_dim_two_compartment}


    fit_dataset(data_train=two_compartment_data_train, data_test=two_compartment_data_test, 
                nb_latents_base=nb_latents_base, nb_latents_hybrid=nb_latents_hybrid, 
                results_path=f"../results/{folder_name}/two_compartments_multiple_trials")


    ## Single-compartment simulations
    best_event_dim_single_compartment = load_data(f"../results/{folder_name}/single_compartment_multiple_trials/best_event_dim.pkl")
    best_burst_base_dim_single_compartment = load_data(f"../results/{folder_name}/single_compartment_multiple_trials/best_burst_base_dim.pkl")
    best_burst_hybrid_dim_single_compartment = load_data(f"../results/{folder_name}/single_compartment_multiple_trials/best_burst_hybrid_dim.pkl")

    nb_latents_base_single_compartment = {"events": best_event_dim_single_compartment, "bursts": best_burst_base_dim_single_compartment}
    nb_latents_hybrid_single_compartment = {"events": best_event_dim_single_compartment, "bursts": best_burst_hybrid_dim_single_compartment}

    fit_dataset(data_train=single_compartment_data_train, data_test=single_compartment_data_test, 
                nb_latents_base=nb_latents_base_single_compartment, nb_latents_hybrid=nb_latents_hybrid_single_compartment, 
                results_path=f"../results/{folder_name}/single_compartment_multiple_trials")
    
    
    return

def evaluate_all_models(folder_name, fraction=1):
    """
    Evaluate all models for both two-compartment and single-compartment simulations.
    
    :param folder_name: The name of the folder containing the simulation results and data.
    :param fraction: The fraction of the dataset to load for evaluation. Should be the same value used in fit_all_models.
    """

    two_compartment_data_train = load_count_data(f"{folder_name}/two_compartments_multiple_trials/training_data", fraction=fraction)
    single_compartment_data_train = load_count_data(f"{folder_name}/single_compartment_multiple_trials/training_data", fraction=fraction)

    two_compartment_data_test = load_count_data(f"{folder_name}/two_compartments_multiple_trials/testing_data", fraction=fraction)
    single_compartment_data_test = load_count_data(f"{folder_name}/single_compartment_multiple_trials/testing_data", fraction=fraction)


    results_two_compartment = load_data(f"../results/{folder_name}/two_compartments_multiple_trials/model_fitting_results.pkl")
    results_single_compartment = load_data(f"../results/{folder_name}/single_compartment_multiple_trials/model_fitting_results.pkl")

    ## Plot the training dynamics for both two-compartment and single-compartment models
    plot_training_dynamics(fit_results=results_two_compartment, 
                            data_train=two_compartment_data_train, 
                            data_test=two_compartment_data_test, 
                            results_path=f"../results/{folder_name}/two_compartments_multiple_trials")
    
    plot_training_dynamics(fit_results=results_single_compartment, 
                             data_train=single_compartment_data_train, 
                             data_test=single_compartment_data_test, 
                              results_path=f"../results/{folder_name}/single_compartment_multiple_trials")


    ## Evaluate the recovered subspaces against the true weights
    # Load the weights of spiking data
    weights = load_data(f"../results/{folder_name}/weights.pkl")
    
    two_compartment_weights = {"W_s": weights['W_s'], "W_d": weights['W_d']}
    single_compartment_weights = {"W_s": weights['W_s'], "W_d": torch.zeros_like(weights['W_d'])}

    evaluate_subspace_recovery(fit_results=results_two_compartment, true_weights=two_compartment_weights,
                                results_path=f"../results/{folder_name}/two_compartments_multiple_trials")
    
    evaluate_subspace_recovery(fit_results=results_single_compartment, true_weights=single_compartment_weights,
                               results_path=f"../results/{folder_name}/single_compartment_multiple_trials")


    ## Evaluate model performance vs. the null model
    evaluate_model_performance(fit_results=results_two_compartment, data_test=two_compartment_data_test, 
                                file_name=f"../results/{folder_name}/two_compartments_multiple_trials/score_difference.svg", 
                                ylim=[-2, 2], yticks=[-2, -1, 0, 1, 2])
    
    evaluate_model_performance(fit_results=results_single_compartment, data_test=single_compartment_data_test, 
                               file_name=f"../results/{folder_name}/single_compartment_multiple_trials/score_difference.svg",
                               ylim=[-2, 2], yticks=[-2, -1, 0, 1, 2])
    

    ## Posterior predictive checks
    posterior_predictive_checks(results_two_compartment, two_compartment_data_test, 
                                f"../results/{folder_name}/two_compartments_multiple_trials")
    
    posterior_predictive_checks(results_single_compartment, single_compartment_data_test, 
                                f"../results/{folder_name}/single_compartment_multiple_trials")

    return


## Wrapper functions - new
def scenario_names(frac_intact_list=()):
    """
    Folder names of the simulated scenarios: two-compartment (intact), single-compartment (silenced) and mixed populations.

    :param frac_intact_list: Intact fractions of the mixed simulations (see spiking_simulation.simulate_mixed_trials).
    :return: List of scenario folder names.
    """
    return [f"mixed_{f:.2f}_multiple_trials" for f in frac_intact_list]

def load_true_weights(folder_name, scenario):
    """
    True somatic and dendritic weights of a scenario. Silenced neurons (all neurons in the single-compartment scenario,
    the non-intact neurons in a mixed scenario) receive no dendritic input, so their rows of W_d are zero.

    :param folder_name: The name of the folder containing the simulation results.
    :param scenario: Scenario folder name.
    :return: Dictionary with the true weights 'W_s' and 'W_d'.
    """
    weights = load_data(f"../results/{folder_name}/weights.pkl")
    W_s, W_d = weights['W_s'], weights['W_d']

    if scenario == "single_compartment_multiple_trials":
        W_d = torch.zeros_like(W_d)
    elif scenario.startswith("mixed_"):
        intact = load_data(f"../results/{folder_name}/{scenario}/training_data/intact_mask.pkl")['intact_mask']
        W_d = W_d * torch.from_numpy(intact.astype(np.float32))[:, None]

    return {"W_s": W_s, "W_d": W_d}

def cross_val_scenario(folder_name, scenario, fraction=0.1):
    """
    Cross validation of all models for one simulated scenario.

    :param folder_name: The name of the folder containing the simulation results.
    :param scenario: Scenario folder name.
    :param fraction: Fraction of the training data used for cross-validation (0.1 -> 1000 samples).
    """
    results_path = f"../results/{folder_name}/{scenario}"
    data_train = load_count_data(f"{folder_name}/{scenario}/training_data", fraction=fraction)

    # Dimensions to test
    event_dim_list = np.arange(1, 7)
    burst_dim_list = np.arange(1, 10)
    burst_dim_hybrid_list = np.arange(0, 10)

    # Generate K-fold splits for cross-validation
    kfold_params = {"nb_splits": 5, "shuffle": True}
    nb_samples_train, _ = data_train["events"].shape
    kfold_splits = kfold_indices(nb_samples=nb_samples_train, nb_splits=kfold_params["nb_splits"], shuffle=kfold_params["shuffle"])

    # Parameters for training
    training_params = {"nb_iter": 5_000, "batch_size": None, "lr": 1e-2, "final_ratio": 1e-4}

    best_event_dim = cross_val_dataset_event_model(data_train, event_dim_list, kfold_splits, training_params, results_path=results_path)
    cross_val_dataset_burst_multiplexed_model(data_train, best_event_dim, burst_dim_list, kfold_splits, training_params, results_path=results_path)
    cross_val_dataset_burst_hybrid_model(data_train, best_event_dim, burst_dim_hybrid_list, kfold_splits, training_params, results_path=results_path)

    return

def fit_scenario(folder_name, scenario, fraction=1):
    """
    Fit all models for one simulated scenario using the best dimensions obtained from cross-validation.

    :param folder_name: The name of the folder containing the simulation results.
    :param scenario: Scenario folder name.
    :param fraction: The fraction of the dataset to use for training and testing.
    """
    results_path = f"../results/{folder_name}/{scenario}"
    data_train = load_count_data(f"{folder_name}/{scenario}/training_data", fraction=fraction)
    data_test = load_count_data(f"{folder_name}/{scenario}/testing_data", fraction=fraction)

    best_event_dim = load_data(f"{results_path}/best_event_dim.pkl")
    nb_latents_base = {"events": best_event_dim, "bursts": load_data(f"{results_path}/best_burst_base_dim.pkl")}
    nb_latents_hybrid = {"events": best_event_dim, "bursts": load_data(f"{results_path}/best_burst_hybrid_dim.pkl")}

    fit_dataset(data_train=data_train, data_test=data_test, nb_latents_base=nb_latents_base,
                nb_latents_hybrid=nb_latents_hybrid, results_path=results_path)

    return

def evaluate_scenario(folder_name, scenario, fraction=1):
    """
    Evaluate all models for one simulated scenario.

    :param folder_name: The name of the folder containing the simulation results.
    :param scenario: Scenario folder name.
    :param fraction: The fraction of the dataset to load for evaluation. Should be the same value used in fit_scenario.
    """
    results_path = f"../results/{folder_name}/{scenario}"
    data_train = load_count_data(f"{folder_name}/{scenario}/training_data", fraction=fraction)
    data_test = load_count_data(f"{folder_name}/{scenario}/testing_data", fraction=fraction)
    fit_results = load_data(f"{results_path}/model_fitting_results.pkl")

    plot_training_dynamics(fit_results=fit_results, data_train=data_train, data_test=data_test, results_path=results_path)
    data_train = load_count_data(f"{folder_name}/{scenario}/training_data", fraction=fraction)
    data_test = load_count_data(f"{folder_name}/{scenario}/testing_data", fraction=fraction)
    fit_results = load_data(f"{results_path}/model_fitting_results.pkl")

    plot_training_dynamics(fit_results=fit_results, data_train=data_train, data_test=data_test, results_path=results_path)

    evaluate_subspace_recovery(fit_results=fit_results, true_weights=load_true_weights(folder_name, scenario),
                               results_path=results_path)

    evaluate_model_performance(fit_results=fit_results, data_test=data_test, file_name=f"{results_path}/score_difference.svg",
                               ylim=[-12, 12], yticks=[-10, 0, 10])

    posterior_predictive_checks(fit_results, data_test, results_path)

    return

# Plotting the score differences per event for mixed-population scenarios
def mixed_scenarios(folder_name, frac_intact_list):
    """
    Mixed-population scenarios for the given intact fractions, sorted by fraction intact.

    :param folder_name: The name of the folder containing the simulation results.
    :param frac_intact_list: Intact fractions to include (folders mixed_{f:.2f}_multiple_trials).
    :return: List of (fraction intact, scenario folder name) tuples.
    """
    scenarios = []
    for f in sorted(frac_intact_list):
        scenario = f"mixed_{f:.2f}_multiple_trials"
        if not os.path.exists(f"../results/{folder_name}/{scenario}/model_fitting_results.pkl"):
            raise FileNotFoundError(f"No fitted models for fraction intact {f} ({scenario})")
        scenarios.append((f, scenario))
    return scenarios


def score_differences_per_event(folder_name, scenario, nb_posterior_samples=1000, fraction=1):
    """
    Per-event score differences (base - null, hybrid - null) on the test data, in mnats/event with 95% bootstrap CIs.
    Computed once and saved to score_difference_per_event.pkl in the scenario folder.

    :param folder_name: The name of the folder containing the simulation results.
    :param scenario: Scenario folder name.
    :param nb_posterior_samples: Number of posterior samples for the log-evidence scores.
    :param fraction: The fraction of the dataset to load. Should be the same value used in fit_scenario.
    :return: Dictionary with (estimate, lower, upper, p_value) tuples for 'base' and 'hybrid'.
    """
    results_path = f"../results/{folder_name}/{scenario}"
    file_name = f"{results_path}/score_difference_per_event.pkl"
    if os.path.exists(file_name):
        return load_data(file_name)

    data_test = load_count_data(f"{folder_name}/{scenario}/testing_data", fraction=fraction)
    fit_results = load_data(f"{results_path}/model_fitting_results.pkl")

    scores = {name: log_evidence_scores(model=fit_results[key], data=data_test, device=device,
                                        nb_posterior_samples=nb_posterior_samples, event_model=False)
            for name, key in [("base", "multiplexed_model_test"), ("null", "null_model_test"), ("hybrid", "hybrid_model_test")]}
    events_per_sample = data_test["events"].sum(dim=1)

    results = {}
    for name in ["base", "hybrid"]:
        estimate, lower, upper, p_value = bootstrap_confidence_interval(scores[name] - scores["null"], events_per_sample=events_per_sample)
        results[name] = (1e3 * estimate, 1e3 * lower, 1e3 * upper, p_value)

    save_data(results, file_name)

    return results


def plot_fraction_intact_sweep(folder_name, frac_intact_list, color_base='tab:orange', color_hybrid='tab:purple', color_soma='tab:blue',
                               ylim=None, yticks=None):
    """
    Plot the per-event score differences and the cross-validated latent dimensions against the fraction of intact neurons.

    :param folder_name: The name of the folder containing the simulation results.
    :param frac_intact_list: Intact fractions to plot.
    :param color_base: Color of the base model.
    :param color_hybrid: Color of the hybrid model.
    :param color_soma: Color of the somatic dimensions.
    :param ylim: Y-axis limits of the score difference plot.
    :param yticks: Y-axis tick values of the score difference plot.
    """
    scenarios = mixed_scenarios(folder_name, frac_intact_list)
    fractions = [f for f, _ in scenarios]
    print(f"Fractions intact: {fractions}")

    results_base, results_hybrid, event_dims, base_dims, hybrid_dims = [], [], [], [], []
    for f, scenario in scenarios:
        set_seed(18)  # bootstrap resamples use the torch RNG
        results = score_differences_per_event(folder_name, scenario)
        results_base.append(results["base"])
        results_hybrid.append(results["hybrid"])

        results_path = f"../results/{folder_name}/{scenario}"
        event_dims.append(int(load_data(f"{results_path}/best_event_dim.pkl")))
        base_dims.append(int(load_data(f"{results_path}/best_burst_base_dim.pkl")))
        hybrid_dims.append(int(load_data(f"{results_path}/best_burst_hybrid_dim.pkl")))

        base_est, base_lo, base_hi, base_p = results["base"]
        hyb_est, hyb_lo, hyb_hi, hyb_p = results["hybrid"]
        print(f"f={f:.2f}  K_E={event_dims[-1]}  K_B base/hybrid={base_dims[-1]}/{hybrid_dims[-1]}  "
              f"base-null {base_est:+.2f} [{base_lo:+.2f}, {base_hi:+.2f}] (p={base_p:.4f})  "
              f"hybrid-null {hyb_est:+.2f} [{hyb_lo:+.2f}, {hyb_hi:+.2f}] (p={hyb_p:.4f}) mnats/event")

    plot_score_difference_vs_fraction(fractions, results_base, results_hybrid, color_base=color_base, color_hybrid=color_hybrid,
                                      ylim=ylim, yticks=yticks,
                                      file_name=f"../results/{folder_name}/score_difference_per_event_vs_fraction.svg")

    plot_dims_vs_fraction(fractions, {"Somatic": event_dims}, {"Somatic": color_soma}, ylabel="Somatic latent \n dimensions",
                          ylim=[0, max(event_dims) + 1],
                          file_name=f"../results/{folder_name}/somatic_dims_vs_fraction.svg")

    plot_dims_vs_fraction(fractions, {"Base": base_dims, "Hybrid": hybrid_dims}, {"Base": color_base, "Hybrid": color_hybrid},
                          ylabel="Dendritic latent \n dimensions", ylim=[0, max(base_dims + hybrid_dims) + 1],
                          file_name=f"../results/{folder_name}/dendritic_dims_vs_fraction.svg")

    return

def replot_cv_scores(folder_name, scenario, figsize=(1.5, 1.0), results_root="../results", xtick_step=None):
    """
    Replot the saved cross-validation curves of one scenario (or data session) without refitting.

    :param folder_name: The name of the folder containing the results (e.g. "evidence_of_burst_coding" or "data_analysis/rodgers").
    :param scenario: Scenario (or session) folder name.
    :param figsize: Figure size of the plots.
    :param results_root: Path of the results folder relative to the working directory.
    :param xtick_step: If given, x ticks every xtick_step dimensions (multiples of xtick_step) instead of every dimension.
    """
    results_path = f"{results_root}/{folder_name}/{scenario}"

    for scores_file, best_file, plot_file, color, xlabel in [
            ("event_model_cv_scores.pkl", "best_event_dim.pkl", "mean_cv_scores_event_plot.svg", 'tab:blue', "Somatic latent \n dimensions"),
            ("burst_model_cv_scores.pkl", "best_burst_base_dim.pkl", "mean_cv_scores_burst_base_plot.svg", 'tab:red', "Dendritic latent \n dimensions"),
            ("burst_hybrid_model_cv_scores.pkl", "best_burst_hybrid_dim.pkl", "mean_cv_scores_burst_hybrid_plot.svg", 'tab:red', "Dendritic latent \n dimensions")]:

        scores_list = load_data(f"{results_path}/{scores_file}")
        dim_list = np.array(sorted(scores_list.keys()))
        mean_scores_list, sem_scores_list = compute_relative_mean_sem_scores(scores_list, dim_list)

        xticks = None
        if xtick_step is not None:
            xticks = np.arange(0, dim_list.max() + 1, xtick_step)
            xticks = xticks[xticks >= dim_list.min()]

        plot_mean_scores(best_dim=load_data(f"{results_path}/{best_file}"), dim_list=dim_list, mean_scores_list=mean_scores_list,
                         sem_scores_list=sem_scores_list, color=color, file_name=f"{results_path}/{plot_file}", xlabel=xlabel,
                         xticks=xticks, figsize=figsize)
        plt.close('all')

    return

    
if __name__ == "__main__":

    folder_name = "evidence_of_burst_coding"
    frac_intact_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  # same list as in spiking_simulation.py

    for scenario in scenario_names(frac_intact_list):
        print(f"===== {scenario} =====")
        set_seed(18)  # reseed per scenario so results do not depend on the scenario order
        cross_val_scenario(folder_name, scenario)
        fit_scenario(folder_name, scenario)
        evaluate_scenario(folder_name, scenario)

    # Plot the fraction of intact neurons sweep
    plot_fraction_intact_sweep(folder_name, frac_intact_list, color_base='tab:orange', color_hybrid='tab:purple')

    # Replot the cross-validation scores for each scenario
    for f in frac_intact_list:
        replot_cv_scores(folder_name, f"mixed_{f:.2f}_multiple_trials", figsize=(1.15, 1.0))

