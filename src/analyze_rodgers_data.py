import torch
import matplotlib.pyplot as plt
import numpy as np
np.seterr(invalid='ignore', divide='ignore') 
import seaborn as sns

from utils import load_data, save_data, set_seed, create_folder
from plotting_functions import plot_training_dynamics, plot_baseline_score
from misc import kfold_indices
from analyze_spiking_simulation import cross_val_dataset_event_model, cross_val_dataset_burst_hybrid_model, cross_val_dataset_burst_multiplexed_model, fit_dataset, evaluate_model_performance, posterior_predictive_checks
from analyze_spiking_simulation import replot_cv_scores
from models import fit_baseline_burst_model, log_evidence_scores

def train_test_split(events, bursts, test_fraction=0.2, seed=18):
    """Split the events and bursts data into training and testing sets keeping samples from the same trials together."""

    nb_neurons, nb_trials, nb_bins = events.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nb_trials)
    test_size = int(round(test_fraction * nb_trials))
    test_trials, train_trials = np.sort(perm[:test_size]), np.sort(perm[test_size:])

    to_samples = lambda x, t: torch.tensor(x[:, t, :].reshape(nb_neurons, -1).T, dtype=torch.float32)
    data_train = {"events": to_samples(events, train_trials), "bursts": to_samples(bursts, train_trials)}
    data_test = {"events": to_samples(events, test_trials), "bursts": to_samples(bursts, test_trials)}

    return data_train, data_test

def create_train_and_test_sets(events, bursts, new_data_path, seed):

    # Train-test split
    data_train, data_test = train_test_split(events, bursts, test_fraction=0.2, seed=seed)

    print("Data shapes after train-test split:")
    print("Training:", data_train['events'].shape, "Testing:", data_test['events'].shape)

    # Find neurons with zero bursting activity in the training set and remove them from both training and testing sets
    zero_burst_neurons = (data_train['bursts'].sum(dim=0) == 0) | (data_test['bursts'].sum(dim=0) == 0)
    zero_burst_neurons = zero_burst_neurons.numpy()
    if zero_burst_neurons.any():
        data_train['events'] = data_train['events'][:, ~zero_burst_neurons]
        data_train['bursts'] = data_train['bursts'][:, ~zero_burst_neurons]
        data_test['events'] = data_test['events'][:, ~zero_burst_neurons]
        data_test['bursts'] = data_test['bursts'][:, ~zero_burst_neurons]

        print("Removed zero burst neurons:", zero_burst_neurons.sum())


    # Save the train and test data for later use
    save_data(data_train, new_data_path + f'training_data.pkl')
    save_data(data_test, new_data_path + f'testing_data.pkl')

    return

## Wrapper functions
def cross_val_all_models_rodgers(data_folder, results_folder, seed=18):
    """
    Cross validation wrapper for all models on the rodgers dataset.
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

def fit_all_models_rodgers(data_folder, results_folder):
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

def evaluate_all_models_rodgers(data_folder, results_folder):
    """
    Evaluate all models for rodgers data.
    
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
                                ylim=[-7.5, 7.5], yticks=[-5, 0, 5])
    
    ## Posterior predictive checks
    ppc_results_path = f"{results_folder}/posterior_predictive_checks"
    create_folder(ppc_results_path)
    posterior_predictive_checks(results, data_test, ppc_results_path)

    plt.close() # close plots

    return

# Independent model comparison
def independent_model_comparison_rodgers(data_folder, results_folder, device):
    """
    Compare the independent baseline model with the multiplexed and null models.

    Parameters:
    - data_folder: Folder containing the training and testing data.
    - results_folder: Folder containing the fitted models.
    - device: Device to run the computations on.
    """

    # Load the training data for the current session
    # Load the data and the fitting results
    data_train = load_data(f"{data_folder}/training_data.pkl")
    data_test = load_data(f"{data_folder}/testing_data.pkl")
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


def run_rodgers_analysis(data_folder, results_path, mouse_ids, device, seed=18):
    """
    Run the full analysis pipeline for the Rodgers dataset, including data preparation, cross-validation, model fitting, evaluation, and plotting.
    
    :param data_folder: Path to the folder containing the Rodgers dataset.
    :param results_path: Path to the folder where results will be saved.
    :param mouse_ids: List of mouse IDs to process.
    :param device: Torch device to use for computations.
    :param seed: Random seed for reproducibility.
    """

    for mouse_id in mouse_ids:
        print(f"Processing session {mouse_id}...")
        set_seed(seed)

        new_data_path = data_folder + f'{mouse_id}/'
        create_folder(new_data_path)

        new_results_path = results_path + f'{mouse_id}/'
        create_folder(new_results_path)

        # Load the data for the current mouse
        data = load_data(data_folder + f'{mouse_id}_behavior+ecephys+image.pkl')

        # Create and save the train and test sets for the current mouse and session
        create_train_and_test_sets(events=data['event_counts'], bursts=data['burst_counts'], new_data_path=new_data_path, seed=seed)

        # Perform cross-validation for all models on the current mouse's data
        cross_val_all_models_rodgers(data_folder=new_data_path, results_folder=new_results_path, seed=seed)

        # Fit all models on the current mouse's data using the best dimensions from cross-validation
        fit_all_models_rodgers(data_folder=new_data_path, results_folder=new_results_path)

        # Evaluate the fitted models for the current mouse
        evaluate_all_models_rodgers(data_folder=new_data_path, results_folder=new_results_path)

        # Replot the cross-validation curves from the saved scores
        replot_cv_scores("data_analysis/rodgers", mouse_id, figsize=(1.15, 1.0), xtick_step=5)

        independent_model_comparison_rodgers(data_folder=new_data_path, results_folder=new_results_path, device=device)


    return

if __name__ == "__main__":

    seed = 18 # Set a random seed for reproducibility
    device = torch.device('cpu')

    results_path = '../results/data_analysis/rodgers/'
    create_folder(results_path)

    mouse_ids_deep = ["sub-221CR_ses-20190514T142312", "sub-221CR_ses-20190515T160400",
                    "sub-KF132_ses-20180223T152924", "sub-KF132_ses-20180222T194100"]
    run_rodgers_analysis(data_folder='../data/rodgers_deep_choice_binned_0.25/discrimination/',
                        results_path=results_path, 
                        mouse_ids=mouse_ids_deep, 
                        device=device, seed=18)

    mouse_ids_L5B = ["sub-KM131_ses-20180118T170853", "sub-KM131_ses-20180119T170444"]
    run_rodgers_analysis(data_folder='../data/rodgers_L5B_choice_binned_0.25/discrimination/',
                        results_path=results_path, 
                        mouse_ids=mouse_ids_L5B, 
                        device=device, seed=18)