import torch
import matplotlib.pyplot as plt
import numpy as np
np.seterr(invalid='ignore', divide='ignore') 
import seaborn as sns

from utils import load_data, save_data, set_seed, create_folder

from tqdm import tqdm
from models import fit_baseline_burst_model, fit_event_model, fit_burst_multiplexed_model, fit_burst_null_model, log_evidence_scores
from misc import compute_mean_sem_scores
from plotting_functions import plot_mean_scores, plot_baseline_score

from plotting_functions import plot_training_dynamics
from misc import kfold_indices
from analyze_spiking_simulation import cross_val_dataset_event_model, cross_val_dataset_burst_hybrid_model, cross_val_dataset_burst_multiplexed_model, fit_dataset, evaluate_model_performance, posterior_predictive_checks
from analyze_spiking_simulation import replot_cv_scores


# Create the train and test sets for the VBN dataset
def train_test_split_vbn(events, bursts, test_fraction=0.2, seed=18):
    """Split the events and bursts data into training and testing sets keeping samples from the same trials together. VBN dataset does not have multiple bins per trial."""

    nb_neurons, nb_trials = events.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nb_trials)
    test_size = int(round(test_fraction * nb_trials))
    test_trials, train_trials = np.sort(perm[:test_size]), np.sort(perm[test_size:])

    to_samples = lambda x, t: torch.tensor(x[:, t].reshape(nb_neurons, -1).T, dtype=torch.float32)
    data_train = {"events": to_samples(events, train_trials), "bursts": to_samples(bursts, train_trials)}
    data_test = {"events": to_samples(events, test_trials), "bursts": to_samples(bursts, test_trials)}

    return data_train, data_test

def create_train_and_test_sets_vbn(mouse_id, data, results_path='../results/data_analysis/'):

    # Select the best mouse and session based on highest average event rate across neurons
    best_mouse_id = mouse_id
    mouse_data = data[best_mouse_id]

    # Loop through each session for the selected mouse and extract event and burst rates
    # Should be one familiar and one novel session
    for session_id in mouse_data.keys():

        create_folder(results_path + f'mouse_{best_mouse_id}/session_{session_id}/')

        session_data = mouse_data[session_id]
        events = session_data['event_rates'] # Key is "rates", but it is actually the number of events
        bursts = session_data['burst_rates']

        # Train-test split
        data_train, data_test = train_test_split_vbn(events, bursts, test_fraction=0.2)

        print(data_train['events'].shape, data_train['bursts'].shape)

        # Find neurons with zero bursting activity in the training set and remove them from both training and testing sets
        zero_burst_neurons = (data_train['bursts'].sum(dim=0) == 0) | (data_test['bursts'].sum(dim=0) == 0)
        zero_burst_neurons = zero_burst_neurons.numpy()
        if zero_burst_neurons.any():
            data_train['events'] = data_train['events'][:, ~zero_burst_neurons]
            data_train['bursts'] = data_train['bursts'][:, ~zero_burst_neurons]
            data_test['events'] = data_test['events'][:, ~zero_burst_neurons]
            data_test['bursts'] = data_test['bursts'][:, ~zero_burst_neurons]

        #print(zero_burst_neurons)
        print(data_train['events'].shape, data_train['bursts'].shape)
  
        # Save the train and test data for later use
        save_data(data_train, results_path + f'mouse_{best_mouse_id}/session_{session_id}/training_data.pkl')
        save_data(data_test, results_path + f'mouse_{best_mouse_id}/session_{session_id}/testing_data.pkl')

    return


# Independent model comparison
def independent_model_comparison_vbn(data_folder, results_folder, device):
    """
    Compare the independent baseline model with the multiplexed and null models.

    Parameters:
    - data_folder: Folder containing the training and testing data.
    - results_folder: Folder containing the fitted models.
    - device: Device to run the computations on.
    """

    # Load the training data for the current session
    data_train = load_data(f"{data_folder}/training_data.pkl")
    data_test = load_data(f"{data_folder}/testing_data.pkl")

    # Load the fit results
    fit_results = load_data(f"{results_folder}/model_fitting_results.pkl")

    # Get scores of the base, hybrid, and null models
    scores_multiplexed_model = log_evidence_scores(model=fit_results['multiplexed_model_test'], data=data_test, device=device, 
                                                   nb_posterior_samples=1000, event_model=False)

    scores_null_model = log_evidence_scores(model=fit_results['null_model_test'], data=data_test, device=device, 
                                            nb_posterior_samples=1000, event_model=False)

    scores_hybrid_model = log_evidence_scores(model=fit_results['hybrid_model_test'], data=data_test, device=device, 
                                             nb_posterior_samples=1000, event_model=False)

    nb_samples, nb_neurons = data_train['events'].shape

    # Fit the baseline model
    event_model_train = fit_results['event_model_train']
    nb_latents = {"events": event_model_train.nb_latents_events, "bursts": 0}
    baseline_burst_model_train = fit_baseline_burst_model(data_train=data_train, event_model=event_model_train, nb_latents=nb_latents, device=device)

    scores_baseline_model = log_evidence_scores(model=baseline_burst_model_train, data=data_test, device=device, 
                                            nb_posterior_samples=1000, event_model=False)

    names = ['Base', "Hybrid", 'Null', 'Indep.']

    events_per_sample = data_test["events"].sum(dim=1)
    plot_baseline_score(names=names, score_1=scores_multiplexed_model, score_2=scores_hybrid_model, 
                        score_3=scores_null_model, score_4=scores_baseline_model, events_per_sample=events_per_sample, 
                        file_name=f"{results_folder}/model_comparison_indep.svg")

    return


## Wrapper functions
def cross_val_all_models_vbn(data_folder, results_folder, seed=18):
    """
    Cross validation wrapper for all models on the vbn dataset.
    """
  
    # Load the training data 
    training_data = load_data(f"{data_folder}/training_data.pkl")
    
    ################## Cross-validation ########################
    # Dimensions to test
    nb_neurons = training_data["events"].shape[1]
    cap = nb_neurons - 1
    event_dim_list = np.arange(1, min(15, cap) + 1)
    burst_dim_list = np.arange(1, min(15, cap) + 1)
    burst_dim_hybrid_list = np.arange(0, min(15, cap) + 1)

    # Generate K-fold splits for cross-validation
    kfold_params = {"nb_splits": 5, "shuffle": True}
    nb_samples_train, _ = training_data["events"].shape
    kfold_splits = kfold_indices(nb_samples=nb_samples_train, nb_splits=kfold_params["nb_splits"], shuffle=kfold_params["shuffle"], seed=seed)

    # Parameters for training
    training_params = {"nb_iter": 5_000, "batch_size": None, "lr": 1e-2, "final_ratio": 1e-4}

    # Run cross-validation for the event model on the training data
    best_event_dim = cross_val_dataset_event_model(training_data, event_dim_list, kfold_splits, training_params, 
                                  results_path=f"{results_folder}", fraction_threshold=1.0)

    # Run cross-validation for the burst multiplexed model on the training data
    best_burst_dim= cross_val_dataset_burst_multiplexed_model(training_data, best_event_dim, burst_dim_list, kfold_splits, training_params,
                                  results_path=f"{results_folder}", fraction_threshold=1.0)

    # Run cross-validation for the hybrid model on the two-compartment data
    best_hybrid_dim = cross_val_dataset_burst_hybrid_model(training_data, best_event_dim, burst_dim_hybrid_list, kfold_splits, training_params,
                                  results_path=f"{results_folder}", fraction_threshold=1.0)

    return

def fit_all_models_vbn(data_folder, results_folder):
    """
    Wrapper to fit all models on the training data using the best dimensions obtained from cross-validation.
    
    :param data_folder: The folder containing the training and testing data.
    :param results_folder: The folder where the fitted model results will be saved.
    """
    # Load the training and testing data
    data_train = load_data(f"{data_folder}/training_data.pkl")
    data_test = load_data(f"{data_folder}/testing_data.pkl")

    # Load the best dimensions from the cross-validation results
    best_event_dim = load_data(f"{results_folder}/best_event_dim.pkl")
    best_burst_base_dim = load_data(f"{results_folder}/best_burst_base_dim.pkl")
    best_burst_hybrid_dim = load_data(f"{results_folder}/best_burst_hybrid_dim.pkl")

    nb_latents_base = {"events": best_event_dim, "bursts": best_burst_base_dim}
    nb_latents_hybrid = {"events": best_event_dim, "bursts": best_burst_hybrid_dim}

    # Fit the dataset using all models
    fit_dataset(data_train=data_train, data_test=data_test, 
                nb_latents_base=nb_latents_base, nb_latents_hybrid=nb_latents_hybrid, 
                results_path=f"{results_folder}")

    return

def evaluate_all_models_vbn(data_folder, results_folder, ylim=[-7.5, 7.5], yticks=[-5, 0, 5]):
    """
    Evaluate all models for vbn data.
    
    :param data_folder: The name of the folder containing the fitting results and data.
    :param results_folder: The folder where the fitted model results are saved.
    """

    # Load the data and the fitting results
    data_train = load_data(f"{data_folder}/training_data.pkl")
    data_test = load_data(f"{data_folder}/testing_data.pkl")
    results = load_data(f"{results_folder}/model_fitting_results.pkl")


    ## Plot the training dynamics for the model
    plot_training_dynamics(fit_results=results, 
                            data_train=data_train, 
                            data_test=data_test, 
                            results_path=f"{results_folder}")


    ## Evaluate model performance vs. the null model
    evaluate_model_performance(fit_results=results, data_test=data_test, 
                                file_name=f"{results_folder}/score_difference.svg", 
                                ylim=ylim, yticks=yticks)
    
    ## Posterior predictive checks
    ppc_results_path = f"{results_folder}/posterior_predictive_checks"
    create_folder(ppc_results_path)
    posterior_predictive_checks(results, data_test, ppc_results_path)

    plt.close() # close plots


    return




if __name__ == "__main__":

    device = torch.device('cpu')
    seed = 18
    set_seed(seed) # Set a random seed for reproducibility

    results_path = 'results/data_analysis/vbn/' # For saving the analysis results
    create_folder(results_path)

    data = load_data('data/vbn/merged_data_mIDs.pkl') # Load the preprocessed data

    mouse_ids = [530862, 574082, 599294]
    y_lims = {530862: [-15, 7.5], 574082: [-10, 7.5], 599294: [-7.5, 7.5]} 
    y_ticks = {530862: [-15, -10, -5, 0, 5], 574082: [-10, -5, 0, 5], 599294: [-5, 0, 5]}

    for mouse_id in mouse_ids:
        print(f"Processing mouse {mouse_id}...")
        create_train_and_test_sets_vbn(mouse_id, data, 'data/vbn/') # Create and save the train and test sets for the best mouse and session

        # Get session ids and loop through the sessions
        session_ids = data[mouse_id].keys()
        for session_id in session_ids:

            # Cross validate all the models for the current session
            create_folder(f"{results_path}/mouse_{mouse_id}/session_{session_id}")
            cross_val_all_models_vbn(data_folder=f"data/vbn/mouse_{mouse_id}/session_{session_id}", 
                                     results_folder=f"{results_path}/mouse_{mouse_id}/session_{session_id}", 
                                     seed=seed)

            # Fit the models for the current session
            fit_all_models_vbn(data_folder=f"data/vbn/mouse_{mouse_id}/session_{session_id}", 
                               results_folder=f"{results_path}/mouse_{mouse_id}/session_{session_id}")

            # Evaluate the model fits for the current session
            evaluate_all_models_vbn(data_folder=f"data/vbn/mouse_{mouse_id}/session_{session_id}",
                                      results_folder=f"{results_path}/mouse_{mouse_id}/session_{session_id}", 
                                      ylim=y_lims[mouse_id], yticks=y_ticks[mouse_id])

            # Replot the cv scores
            replot_cv_scores(f"data_analysis/vbn/mouse_{mouse_id}", f"session_{session_id}", figsize=(1.15, 1.0),
                             results_root="results", xtick_step=5)

            # Perform independent model comparison for the current session
            independent_model_comparison_vbn(data_folder=f"data/vbn/mouse_{mouse_id}/session_{session_id}", 
                               results_folder=f"{results_path}/mouse_{mouse_id}/session_{session_id}", device=device)
