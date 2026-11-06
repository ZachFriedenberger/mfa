import numpy as np
np.seterr(invalid='ignore', divide='ignore') 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
device = torch.device('cpu')
print(f"Using device: {device}")

from utils import load_data, create_folder, save_data, set_seed

from plotting_functions import plot_subspace_error_estimate, plot_simulated_sensitivity_analysis, plot_score_difference_multiple
from plotting_functions import plot_simulated_sensitivity_analysis_base_hybrid
from models import fit_fa_model_events_and_bursts, fit_event_model, fit_burst_multiplexed_model, fit_burst_null_model, log_evidence_scores, fit_burst_hybrid_model
from data_generators import data_generation, generate_random_input_weights
from metrics import estimate_projection_distance

from itertools import product
from tqdm import tqdm
from functools import reduce
from operator import mul


# Helper functions for sensitivity analysis
def g(x):
    return x**2 / (1 + x**2)

def soma_subspace_error(K_soma, sigma_soma, nb_samples, mu_soma):

    mean_sparsity = np.mean(np.exp(-mu_soma))
    weight_term  = (1/sigma_soma**2) * np.exp(-(1/2)*sigma_soma**2) * (K_soma - g(sigma_soma))
    error = (1/nb_samples) * weight_term * mean_sparsity

    return error

def sigmoid(x):
    return 1/(1+np.exp(-x))

def phi(x):
    return sigmoid(x)*(1-sigmoid(x))

def gamma(x):
    return 1 - 6*sigmoid(x) + 6*sigmoid(x)**2

def dendrite_subspace_error(K_dendrite, sigma_soma, sigma_dendrite, nb_samples, mu_soma, mu_dendrite):

    sparsity = np.exp(-mu_soma)

    mean_term = np.mean(sparsity * (1/phi(mu_dendrite)) *(K_dendrite - g(gamma(mu_dendrite)*sigma_dendrite**2)))

    error = (1/sigma_dendrite**2) * np.exp(-(1/2)*sigma_soma**2) * mean_term / nb_samples
    
    return error

# Code for running simulations and fitting the simulated data
def fit_models_simulated_data(data, nb_latents, model_type="Base"):

    training_params = {"nb_iter": 5_000, "batch_size": None, "lr": 1e-2, "final_ratio": 1e-4}

    # Fit the event model
    event_model, _ = fit_event_model(data_train=data, nb_latents=nb_latents, training_params=training_params, device=device, print_loss=False, data_test=None)

    if model_type == "Base":
        # Fit the base burst model
        burst_model, _ = fit_burst_multiplexed_model(data_train=data, event_model=event_model, nb_latents=nb_latents, training_params=training_params, 
                                    device=device, print_loss=False, data_test=None)
    elif model_type == "Hybrid":
        # Fit the hybrid burst model
        burst_model, _ = fit_burst_hybrid_model(data_train=data, event_model=event_model, nb_latents=nb_latents, training_params=training_params, 
                                    device=device, print_loss=False, data_test=None)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    
    W_event_learned = event_model.event_decoder.W.cpu().detach().numpy()
    W_burst_learned = burst_model.burst_decoder.W.cpu().detach().numpy()

    return W_event_learned, W_burst_learned

def run_simulation(mu_event_true, mu_burst_true, nb_neurons, nb_samples, nb_dim_soma, nb_dim_dendrite, sigma_soma, sigma_dendrite, 
                   nb_repeats=100,  simulation_weight_function=generate_random_input_weights, w_burst=None, model_type="Base"):
    """
    Run the simulation for the given parameters.

    Parameters:
    - mu_event_true: True mean for event data
    - mu_burst_true: True mean for burst data
    - nb_neurons: Number of neurons
    - nb_samples: Number of samples
    - nb_dim_soma: Number of latent dimensions for soma
    - nb_dim_dendrite: Number of latent dimensions for dendrite
    - sigma_soma: Standard deviation for soma weights
    - sigma_dendrite: Standard deviation for dendrite weights
    - nb_repeats: Number of simulation repeats
    - simulation_weight_function: Function to generate simulation weights

    Returns:
    - output: Dictionary containing distances and zero event counts
    """ 

    weight_params = {'nb_neurons': nb_neurons, 'norm_distance': 0.5, 'nb_dim_soma': nb_dim_soma, 'nb_dim_dendrite': nb_dim_dendrite, 
                     'sigma_soma': sigma_soma, 'sigma_dendrite': sigma_dendrite, 'mu_soma': mu_event_true, 'mu_dendrite': mu_burst_true}

    # Init results containers
    distance_events, distance_bursts, nb_zero_events = np.zeros((3, nb_repeats))
    for i in range(nb_repeats):

        # Generate random subspaces for events and bursts
        W_event_true, W_burst_true = simulation_weight_function(weight_params)

        # Sample data according to model type
        if w_burst is None and model_type == "Base":
            events, bursts, _, _, _, _ = data_generation(W_event=W_event_true, W_burst=W_burst_true, 
                                                        mu_event=mu_event_true, mu_burst=mu_burst_true, nb_samples=nb_samples)
        elif w_burst is not None and model_type == "Hybrid":
            events, bursts, _, _, _, _ = data_generation(W_event=W_event_true, W_burst=W_burst_true, 
                                                        mu_event=mu_event_true, mu_burst=mu_burst_true, w_burst=w_burst, nb_samples=nb_samples)

        data = {'events': events.T.float(), 'bursts': bursts.T.float()}


        nb_latents = {"events": weight_params['nb_dim_soma'], "bursts": weight_params['nb_dim_dendrite']}
        W_event_learned, W_burst_learned = fit_models_simulated_data(data, nb_latents, model_type=model_type)


        # Estimate projection distance
        nb_zero_events[i] = (events == 0.0).sum().item()
        distance_events[i] = estimate_projection_distance(W_event_true.cpu().numpy(), W_event_learned)
        distance_bursts[i] = estimate_projection_distance(W_burst_true.cpu().numpy(), W_burst_learned)

    output = {
        'mean_nb_zero_events': np.mean(nb_zero_events),
        'std_nb_zero_events': np.std(nb_zero_events),
        'mean_distance_event': np.mean(distance_events),
        'std_distance_event': np.std(distance_events),
        'mean_distance_burst': np.mean(distance_bursts),
        'std_distance_burst': np.std(distance_bursts)
    }

    return output



# Theoretical sensitivity analysis for the dataset
def sensitivity_analysis_dataset(model_fitting_results, results_path):
    """
    Perform theoretical sensitivity analysis for the dataset.

    Parameters:
    -----------
    model_fitting_results : dict
        Dictionary containing the model fitting results for the event and multiplexed models.
    results_path : str
        Path to save the resulting plots.
    """

    # Extract model fitting parameters for soma and dendrite subspaces
    event_model_train = model_fitting_results['event_model_train']
    multiplexed_model_train = model_fitting_results['multiplexed_model_train']
    multiplexed_model_test = model_fitting_results['multiplexed_model_test']

    nb_samples_train = event_model_train.nb_samples
    nb_samples_test = multiplexed_model_test.nb_samples

    print(f"Number of training samples: {nb_samples_train}")
    print(f"Number of testing samples: {nb_samples_test}")

    W_soma = event_model_train.event_decoder.W.detach().cpu().numpy()
    W_dendrite = multiplexed_model_train.burst_decoder.W.detach().cpu().numpy()

    _, nb_soma_dims = W_soma.shape
    _, nb_dendrite_dims = W_dendrite.shape

    mu_soma = event_model_train.event_decoder.mu.detach().cpu().numpy()
    mu_dendrite = multiplexed_model_train.burst_decoder.mu.detach().cpu().numpy()

    sigma_soma_squared_est, sigma_dendrite_squared_est = estimate_weight_variances(W_soma, W_dendrite)

    # Plot the soma subspace error as a function of K_soma and sigma_soma
    K_soma_vec = np.arange(1, nb_soma_dims + 3)
    sigma_soma_vec_squared = np.logspace(-2, 1, 100)
    sigma_soma_vec = np.sqrt(sigma_soma_vec_squared)
    K_soma_grid, sigma_soma_grid = np.meshgrid(K_soma_vec, sigma_soma_vec)
    error_grid_soma = soma_subspace_error(K_soma_grid, sigma_soma_grid, nb_samples_train, mu_soma)

    plot_subspace_error_estimate(error_grid=error_grid_soma, K_grid=K_soma_grid, 
                             sigma_grid=sigma_soma_grid, dim_estimate=nb_soma_dims, 
                             sigma_var_estimate=sigma_soma_squared_est, figsize=(1.15, 1.0), cmap='Blues_r',
                             file_name=f'{results_path}/soma_subspace_error_theory.svg')


    # Plot the dendrite subspace error as a function of K_dendrite and sigma_dendrite
    K_dendrite_vec = np.arange(1, nb_dendrite_dims + 3)
    sigma_dendrite_vec_squared = np.logspace(-1, 1, 100)
    sigma_dendrite_vec = np.sqrt(sigma_dendrite_vec_squared)

    K_dendrite_grid, sigma_dendrite_grid = np.meshgrid(K_dendrite_vec, sigma_dendrite_vec)
    sigma_soma = np.sqrt(sigma_soma_squared_est) # From estimate above

    error_grid_dendrite = np.zeros((len(sigma_dendrite_vec), len(K_dendrite_vec)))
    for i, K_dendrite in enumerate(K_dendrite_vec):
        for j, sigma_dendrite in enumerate(sigma_dendrite_vec):
            error_grid_dendrite[j, i] = dendrite_subspace_error(K_dendrite, sigma_soma, sigma_dendrite, nb_samples_train, mu_soma, mu_dendrite)

    plot_subspace_error_estimate(error_grid=error_grid_dendrite, K_grid=K_dendrite_grid, 
                             sigma_grid=sigma_dendrite_grid, dim_estimate=nb_dendrite_dims, 
                             sigma_var_estimate=sigma_dendrite_squared_est, cmap='Reds_r', figsize=(1.15, 1.0),
                             file_name=f'{results_path}/dendrite_subspace_error_theory.svg')
    


    return

def estimate_weight_variances(W_soma, W_dendrite):
    """
    Estimate the variances of the soma and dendrite weight matrices.

    Parameters:
    -----------
    W_soma : np.ndarray
        Soma weight matrix of shape (nb_neurons, nb_soma_dims).
    W_dendrite : np.ndarray
        Dendrite weight matrix of shape (nb_neurons, nb_dendrite_dims).

    Returns:
    --------
    sigma_soma_squared_est : float
        Estimated variance of the soma weights.
    sigma_dendrite_squared_est : float
        Estimated variance of the dendrite weights.
    """

    # Estimate the soma and dendrite weight variances
    nb_neurons, nb_soma_dims = W_soma.shape
    _, nb_dendrite_dims = W_dendrite.shape

    sigma_soma_squared_est = (1/nb_neurons) * np.linalg.norm(W_soma, ord='fro')**2
    sigma_dendrite_squared_est = (1/nb_neurons) * np.linalg.norm(W_dendrite, ord='fro')**2

    return sigma_soma_squared_est, sigma_dendrite_squared_est


# Simulation-based sensitivity analysis for the dataset
def sensitivity_analysis_dataset_simulation(model_fitting_results, results_path, nb_repeats=5):

    # Extract model fitting parameters for soma/event model
    event_model_train = model_fitting_results['event_model_train']
    nb_samples_train = event_model_train.nb_samples

    W_soma = event_model_train.event_decoder.W.detach().cpu().numpy()
    nb_neurons, nb_soma_dims = W_soma.shape
    mu_soma = event_model_train.event_decoder.mu.detach().cpu().numpy()

    burst_models = ['multiplexed_model_train', 'hybrid_model_train']
    burst_model_names = ["Base", "Hybrid"]

    # Loop over burst models to extract dendrite weight matrices and their dimensions
    for burst_model, burst_model_name in zip(burst_models, burst_model_names):

        # Unpack burst model parameters
        burst_model_train = model_fitting_results[burst_model]
        W_dendrite = burst_model_train.burst_decoder.W.detach().cpu().numpy()
        _, nb_dendrite_dims = W_dendrite.shape
        mu_dendrite = burst_model_train.burst_decoder.mu.detach().cpu().numpy()

        if burst_model_name == "Hybrid":
            w_burst = burst_model_train.burst_decoder.w.detach().cpu().numpy()
        else: 
            w_burst = None

        # Estimate the weight variances for the current model
        sigma_soma_squared_est, sigma_dendrite_squared_est = estimate_weight_variances(W_soma, W_dendrite)

        # Run the simulation for the current burst model
        output = run_simulation(mu_event_true=mu_soma, mu_burst_true=mu_dendrite, nb_neurons=nb_neurons, 
                                nb_samples=nb_samples_train, nb_dim_soma=nb_soma_dims, nb_dim_dendrite=nb_dendrite_dims, 
                                sigma_soma=np.sqrt(sigma_soma_squared_est), sigma_dendrite=np.sqrt(sigma_dendrite_squared_est), nb_repeats=nb_repeats, 
                                model_type=burst_model_name, w_burst=w_burst)
    
        # Compute the point estimate from theory
        soma_error_point_estimate = soma_subspace_error(nb_soma_dims, np.sqrt(sigma_soma_squared_est), nb_samples_train, mu_soma)
        dendrite_error_point_estimate = dendrite_subspace_error(nb_dendrite_dims, np.sqrt(sigma_soma_squared_est), np.sqrt(sigma_dendrite_squared_est), 
                                                            nb_samples_train, mu_soma, mu_dendrite)
        
        simulated_data = {
            'output': output,
            'soma_error_point_estimate': soma_error_point_estimate,
            'dendrite_error_point_estimate': dendrite_error_point_estimate
        }

        print(f"Soma error point estimate: {soma_error_point_estimate}")
        print(f"Dendrite error point estimate: {dendrite_error_point_estimate}")

        save_data(simulated_data, f'{results_path}/simulated_sensitivity_analysis_data_{burst_model_name.lower()}_model.pkl')
        

        plot_simulated_sensitivity_analysis(output, soma_error_point_estimate, dendrite_error_point_estimate, 
                                            file_name=f'{results_path}/simulated_sensitivity_analysis_{burst_model_name.lower()}_model.svg', 
                                            figsize=(0.75, 1.0))

    return


## Wrapper function for sensitivity analysis
def sensitivity_analysis(data_folder, results_folder):

    # Load the model fitting results for the mouse
    model_fitting_results = load_data(f'{results_folder}/model_fitting_results.pkl')

    # Plot the theoretical subspace error estimate
    #sensitivity_analysis_dataset(model_fitting_results, f'{results_folder}')

    # Run simulations to validate the theoretical estimates
    sensitivity_analysis_dataset_simulation(model_fitting_results, f'{results_folder}', nb_repeats=5)
    
    return

def replot_sensitivity_base_hybrid(results_folder, file_name=None):
    """Replot the saved base and hybrid sensitivity simulations of one session on a single plot (no refitting)."""
    base_data = load_data(f'{results_folder}/simulated_sensitivity_analysis_data_base_model.pkl')
    hybrid_data = load_data(f'{results_folder}/simulated_sensitivity_analysis_data_hybrid_model.pkl')
    if file_name is None:
        file_name = f'{results_folder}/simulated_sensitivity_analysis_base_hybrid_model.svg'
    plot_simulated_sensitivity_analysis_base_hybrid(base_data, hybrid_data, file_name=file_name)
    return


if __name__ == "__main__":

    device = torch.device('cpu')
    set_seed(18) # Set a random seed for reproducibility

    # VBN dataset
    data = load_data('data/vbn/merged_data_mIDs.pkl') # Load the preprocessed data to get the session info
    mouse_ids  = [574082, 599294, 530862]
    for mouse_id in mouse_ids:
        for session_id in data[mouse_id].keys():
            print(f"Running sensitivity analysis for mouse {mouse_id}, session {session_id}...")

            sensitivity_analysis(data_folder='data', results_folder=f'results/data_analysis/vbn/mouse_{mouse_id}/session_{session_id}')
            replot_sensitivity_base_hybrid(f'results/data_analysis/vbn/mouse_{mouse_id}/session_{session_id}')
        
    # Rogers dataset - L5b neurons
    mouse_ids  = ["sub-KM131_ses-20180118T170853", "sub-KM131_ses-20180119T170444"] 
    for mouse_id in mouse_ids:
        print(f"Processing session {mouse_id}...")

        sensitivity_analysis(data_folder='data/rodgers_choice_binned_0.25/discrimination/', results_folder=f'results/data_analysis/rodgers/{mouse_id}')
        replot_sensitivity_base_hybrid(f'results/data_analysis/rodgers/{mouse_id}')

    # Rodgers dataset - deep neurons
    mouse_ids = ["sub-KF132_ses-20180223T152924", "sub-KF132_ses-20180222T194100",
                "sub-221CR_ses-20190514T142312", "sub-221CR_ses-20190515T160400"]
    for mouse_id in mouse_ids:
        print(f"Processing session {mouse_id}...")

        sensitivity_analysis(data_folder='data/rodgers_deep_choice_binned_0.25/discrimination/', results_folder=f'results/data_analysis/rodgers/{mouse_id}')
        replot_sensitivity_base_hybrid(f'results/data_analysis/rodgers/{mouse_id}')