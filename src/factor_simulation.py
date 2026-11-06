import torch
import numpy as np
from tqdm import tqdm

np.seterr(invalid='ignore', divide='ignore') 

from metrics import estimate_projection_distance

from data_generators import generate_simulation_weights, generate_principal_angles_grassmann, generate_gaussian_process_inputs, generate_data_with_total_spikes
from models import fit_fa_model

from utils import set_seed, create_folder, save_data, save_meta_data, load_data
from misc import align_sign_one_dimensional, update_dict, compute_performance_stats, update_dict_element
from misc import update_dict_element_one_level, update_dict_one_level, compute_performance_stats_dimensions

from plotting_functions import plot_observed_counts_distributions, plot_expected_counts_distributions, plot_latent_trajectories
from plotting_functions import plot_error_vs_angle, plot_error_heatmap, plot_error_difference_heatmap, plot_dimension_estimation_heatmap

def fit_all_fa_models_forced_latents(events, bursts, spikes):

    # Fit a FA model on the spike counts
    transformed_firing_rates = np.sqrt(spikes)
    W_est_firing_rates, z_est_firing_rates, _ = fit_fa_model(transformed_firing_rates, max_nb_latents=10, cv=5, force_nb_latents=2)

    # Fit a FA model on the event rate
    transformed_event_rates = np.sqrt(events)
    W_est_event_rates, z_est_event_rates, _ = fit_fa_model(transformed_event_rates, max_nb_latents=10, cv=5, force_nb_latents=1)

    # Fit a FA model on the burst fraction
    burst_fraction = np.nan_to_num(bursts/events, nan=0.0)
    transformed_burst_fraction = np.arcsin(np.sqrt(burst_fraction))
    W_est_burst_fraction, z_est_burst_fraction, _ = fit_fa_model(transformed_burst_fraction, max_nb_latents=10, cv=5, force_nb_latents=1)

    return W_est_event_rates, W_est_burst_fraction, W_est_firing_rates, z_est_event_rates, z_est_burst_fraction, z_est_firing_rates

def fit_all_fa_models_dimension_est(events, bursts, spikes):

    # Fit a FA model on the spike counts
    transformed_firing_rates = np.sqrt(spikes)
    _, _, best_nb_latent_firing_rates = fit_fa_model(transformed_firing_rates, max_nb_latents=5, cv=5, refit_model=False)

    # Fit a FA model on the event rate
    transformed_event_rates = np.sqrt(events)
    _, _, best_nb_latent_event_rates = fit_fa_model(transformed_event_rates, max_nb_latents=5, cv=5, refit_model=False)

    # Fit a FA model on the burst fraction
    burst_fraction = np.nan_to_num(bursts/events, nan=0.0)
    transformed_burst_fraction = np.arcsin(np.sqrt(burst_fraction))
    _, _, best_nb_latent_burst_fraction = fit_fa_model(transformed_burst_fraction, max_nb_latents=5, cv=5, refit_model=False)

    return best_nb_latent_event_rates, best_nb_latent_burst_fraction, best_nb_latent_firing_rates

def evaluate_dimension_error(nb_latent_event_rates, nb_latent_burst_fraction, nb_latent_firing_rates, nb_latent_true):
    FA_error = abs(nb_latent_firing_rates - nb_latent_true) / nb_latent_true
    MFA_error = abs(nb_latent_event_rates + nb_latent_burst_fraction - nb_latent_true) / nb_latent_true
    return FA_error, MFA_error

def evaluate_all_fa_models(W_est_event_rates, W_est_burst_fraction, W_est_firing_rates, W_event_true, W_burst_true):

    # Combine the estimated event and burst subspaces by concatenating their weight matrices
    W_est_combined_MFA = np.hstack((W_est_event_rates, W_est_burst_fraction))
    W_true_combined = np.hstack((W_event_true, W_burst_true))

    # Evaluate MFA
    error_event_MFA = estimate_projection_distance(W_true=W_event_true, W_est=W_est_event_rates)
    error_burst_MFA = estimate_projection_distance(W_true=W_burst_true, W_est=W_est_burst_fraction)
    error_combined_MFA = estimate_projection_distance(W_true=W_true_combined, W_est=W_est_combined_MFA)

    # Evaluate firing rate model
    error_combined_FA = estimate_projection_distance(W_true=W_true_combined, W_est=W_est_firing_rates)
    error_event_FA = estimate_projection_distance(W_true=W_event_true, W_est=W_est_firing_rates[:, [0]])
    error_burst_FA = estimate_projection_distance(W_true=W_burst_true, W_est=W_est_firing_rates[:, [1]])

    results = {
        'MFA': {
            'event_error': error_event_MFA,
            'burst_error': error_burst_MFA,
            'combined_error': error_combined_MFA
        },
        'FA': {'event_error': error_event_FA,
               'burst_error': error_burst_FA,
               'combined_error': error_combined_FA
          }}
    

    return results

def recovered_dimension_error(weight_params, input_params, data_params, simulation_params, path_to_save):

    # Upack parameters
    nb_neurons = weight_params['nb_neurons']


    nb_steps = input_params['nb_steps']
    input_dt = input_params['dt']

    nb_spike_per_burst = data_params['nb_spike_per_burst']
    data_dt = data_params['dt']

    nb_latent = weight_params['nb_dim_soma'] # Assuming the number of latent dimensions is the same for soma and dendrite, but this can be changed if needed

    nb_repeats = simulation_params['nb_repeats']
    norm_distance_list = simulation_params['norm_distance_list']
    sigma_dendrite_list = simulation_params['sigma_dendrite_list']


    # Create folders to save results
    create_folder(f"results/{path_to_save}")
    create_folder(f"results/{path_to_save}/plots")

    # Intitialize dictionaries to store results
    sigma_length = len(sigma_dendrite_list)
    grassmann_length = len(norm_distance_list)
    all_results_means = {'FA_dim_error': np.zeros((sigma_length, grassmann_length)), 'MFA_dim_error': np.zeros((sigma_length, grassmann_length))}
    all_results_stds = {'FA_dim_error': np.zeros((sigma_length, grassmann_length)), 'MFA_dim_error': np.zeros((sigma_length, grassmann_length))}

    # Loop over different values of sigma_dendrite and norm_distance
    for i, sigma_dendrite in enumerate(tqdm(sigma_dendrite_list, desc="Sigma Dendrite")):
        weight_params['sigma_dendrite'] = sigma_dendrite

        for j, norm_distance in enumerate(tqdm(norm_distance_list, desc="Norm Distance", leave=False)):
            weight_params['norm_distance'] = norm_distance

            repeat_results = {'FA_dim_error': [], 'MFA_dim_error': []}
            for _ in range(nb_repeats):

                mu_event = torch.ones(nb_neurons) * weight_params['mu_soma']
                mu_burst = torch.ones(nb_neurons) * weight_params['mu_dendrite']

                # Generate weights given the alignment in terms of the normalized Grassmann distance
                W_soma, W_dendrite = generate_simulation_weights(weight_params=weight_params, generate_principal_angles_function=generate_principal_angles_grassmann)

                # Generate latent factors from a Gaussian process with a squared exponential kernel
                z_soma, z_dendrite = generate_gaussian_process_inputs(nb_steps, nb_latent, input_dt, input_params)

                # Generate the observed data (events and bursts) with the total number of spikes fixed by the nb_spike_per_burst parameter
                events, bursts, spikes, _, _, _ = generate_data_with_total_spikes(W_soma, W_dendrite, mu_event, mu_burst, z_soma, z_dendrite, data_dt, nb_spike_per_burst)
    
                # Fit the FA models
                best_nb_latent_event_rates, best_nb_latent_burst_fraction, best_nb_latent_firing_rates = fit_all_fa_models_dimension_est(events, bursts, spikes)

                # Evaluate the dimension estimation error
                FA_dim_error, MFA_dim_error = evaluate_dimension_error(nb_latent_event_rates=best_nb_latent_event_rates, 
                                                               nb_latent_burst_fraction=best_nb_latent_burst_fraction, 
                                                               nb_latent_firing_rates=best_nb_latent_firing_rates, 
                                                               nb_latent_true=(weight_params['nb_dim_soma'] + weight_params['nb_dim_dendrite']))


                # Append results to a dictionary
                repeat_results = update_dict_one_level(repeat_results, {'MFA_dim_error': MFA_dim_error, 'FA_dim_error': FA_dim_error})

            # Compute means and stds of the results over repeats
            results_means, results_stds = compute_performance_stats_dimensions(repeat_results)

            # Update the all_results_means and all_results_stds dictionaries
            all_results_means = update_dict_element_one_level(all_results_means, results_means, i, j)
            all_results_stds = update_dict_element_one_level(all_results_stds, results_stds, i, j)

    save_data({'all_results_means': all_results_means, 'all_results_stds': all_results_stds}, 
              filename=f"results/{path_to_save}/estimated_dimensions_data.pkl")

    meta_data = {'weight_params': weight_params, 'input_params': input_params, 'data_params': data_params, 'simulation_params': simulation_params}
    save_meta_data(meta_data, file_name=f"results/{path_to_save}/meta_data.json")



    return

def recovered_latent_dynamics(weight_params, input_params, data_params, path_to_save):
    '''
    Docstring for recovered_latent_dynamics

    :param weight_params: Description
    :param input_params: Description
    :param data_params: Description
    :param path_to_save: Description
    '''

    # Upack parameters
    nb_neurons = weight_params['nb_neurons']
    nb_steps = input_params['nb_steps']
    input_dt = input_params['dt']
    nb_spike_per_burst = data_params['nb_spike_per_burst']
    data_dt = data_params['dt']
    nb_latent = weight_params['nb_dim_soma'] # Assuming the number of latent dimensions is the same for soma and dendrite, but this can be changed if needed


    # Create folders to save results
    create_folder(f"results/{path_to_save}")
    create_folder(f"results/{path_to_save}/plots")

    mu_event = torch.ones(nb_neurons) * weight_params['mu_soma']
    mu_burst = torch.ones(nb_neurons) * weight_params['mu_dendrite']

    # Generate weights given the alignment in terms of the normalized Grassmann distance
    W_soma, W_dendrite = generate_simulation_weights(weight_params=weight_params, generate_principal_angles_function=generate_principal_angles_grassmann)

    # Generate latent factors from a Gaussian process with a squared exponential kernel
    z_soma, z_dendrite = generate_gaussian_process_inputs(nb_steps, nb_latent, input_dt, input_params)

    # Generate the observed data (events and bursts) with the total number of spikes fixed by the nb_spike_per_burst parameter
    events, bursts, spikes, event_rate, burst_rate, spike_rate = generate_data_with_total_spikes(W_soma, W_dendrite, mu_event, mu_burst, z_soma, z_dendrite, data_dt, nb_spike_per_burst)

    # Plot the distributions of expected counts and observed counts
    plot_observed_counts_distributions(events, bursts, spikes, file_name=f"results/{path_to_save}/plots/observed_counts_distributions.svg")
    plot_expected_counts_distributions(event_rate, burst_rate, spike_rate, file_name=f"results/{path_to_save}/plots/expected_counts_distributions.svg")

    # Save the generated data and meta data
    data = {'events': events, 'bursts': bursts, 'spikes': spikes, 'event_rate': event_rate, 'burst_rate': burst_rate, 'spike_rate': spike_rate,
            'W_soma': W_soma, 'W_dendrite': W_dendrite, 'mu_soma': mu_event, 'mu_dendrite': mu_burst, 'z_soma': z_soma, 'z_dendrite': z_dendrite}
    
    meta_data = {'weight_params': weight_params, 'input_params': input_params, 'data_params': data_params}

    save_meta_data(meta_data, file_name=f"results/{path_to_save}/meta_data.json")
    save_data(data, filename=f"results/{path_to_save}/data.pkl")

    # Fit FA models on the different observed data and get the estimated latent factors
    _, _, _, z_est_event_rates, z_est_burst_fraction, z_est_firing_rates = fit_all_fa_models_forced_latents(events, bursts, spikes)

    # Align the signs of the estimated latent factors with the ground truth latent factors for better visualization
    z_est_firing_rates_1 = align_sign_one_dimensional(z_est_firing_rates[:, 0], z_soma[0, :])
    z_est_firing_rates_2 = align_sign_one_dimensional(z_est_firing_rates[:, 1], z_dendrite[0, :])
    z_est_event_rates_1 = align_sign_one_dimensional(z_est_event_rates[:, 0], z_soma[0, :])
    z_est_burst_fraction_1 = align_sign_one_dimensional(z_est_burst_fraction[:, 0], z_dendrite[0, :])

    plot_latent_trajectories(z_est_event_rates_1, z_soma[0, :], z_est_burst_fraction_1, z_dendrite[0, :], 
                             file_name=f"results/{path_to_save}/plots/latent_trajectories_MFA.svg")
    
    plot_latent_trajectories(z_est_firing_rates_1, z_soma[0, :], z_est_firing_rates_2, z_dendrite[0, :], 
                             file_name=f"results/{path_to_save}/plots/latent_trajectories_FA.svg")
    
    latent_data = {'z_est_event_rates_1': z_est_event_rates_1, 'z_est_burst_fraction_1': z_est_burst_fraction_1, 
                   'z_est_firing_rates_1': z_est_firing_rates_1, 'z_est_firing_rates_2': z_est_firing_rates_2}
    
    save_data(latent_data, filename=f"results/{path_to_save}/inferred_latent_factors.pkl")

    return

def recovered_subspace_error(weight_params, input_params, data_params, simulation_params, path_to_save):

    # Upack parameters
    nb_neurons = weight_params['nb_neurons']
    nb_steps = input_params['nb_steps']
    input_dt = input_params['dt']

    nb_spike_per_burst = data_params['nb_spike_per_burst']
    data_dt = data_params['dt']

    nb_repeats = simulation_params['nb_repeats']
    norm_distance_list = simulation_params['norm_distance_list']
    sigma_dendrite_list = simulation_params['sigma_dendrite_list']

    nb_latent = weight_params['nb_dim_soma'] # Assuming the number of latent dimensions is the same for soma and dendrite, but this can be changed if needed


    # Create folders to save results
    create_folder(f"results/{path_to_save}")
    create_folder(f"results/{path_to_save}/plots")

    # Intitialize dictionaries to store results
    sigma_length = len(sigma_dendrite_list)
    grassmann_length = len(norm_distance_list)

    all_results_means = {'MFA': {'event_error': np.zeros((sigma_length, grassmann_length)), 'burst_error': np.zeros((sigma_length, grassmann_length)), 'combined_error': np.zeros((sigma_length, grassmann_length))},
                            'FA': {'event_error': np.zeros((sigma_length, grassmann_length)), 'burst_error': np.zeros((sigma_length, grassmann_length)), 'combined_error': np.zeros((sigma_length, grassmann_length))}}
    all_results_stds = {'MFA': {'event_error': np.zeros((sigma_length, grassmann_length)), 'burst_error': np.zeros((sigma_length, grassmann_length)), 'combined_error': np.zeros((sigma_length, grassmann_length))},
                        'FA': {'event_error': np.zeros((sigma_length, grassmann_length)), 'burst_error': np.zeros((sigma_length, grassmann_length)), 'combined_error': np.zeros((sigma_length, grassmann_length))}}

    # Loop over different values of sigma_burst and norm_grassmann_distance
    for i, sigma_dendrite in enumerate(tqdm(sigma_dendrite_list, desc="Sigma Dendrite")):

        weight_params['sigma_dendrite'] = sigma_dendrite

        for j, norm_distance in enumerate(tqdm(norm_distance_list, desc="Norm Distance", leave=False)):

            weight_params['norm_distance'] = norm_distance

            repeat_results = {'MFA': {'event_error': [], 'burst_error': [], 'combined_error': []},
                        'FA': {'event_error': [], 'burst_error': [], 'combined_error': []}}

            for _ in range(nb_repeats):

                mu_event = torch.ones(nb_neurons) * weight_params['mu_soma']
                mu_burst = torch.ones(nb_neurons) * weight_params['mu_dendrite']

                # Generate weights given the alignment in terms of the normalized Grassmann distance
                W_soma, W_dendrite = generate_simulation_weights(weight_params=weight_params, generate_principal_angles_function=generate_principal_angles_grassmann)

                # Generate latent factors from a Gaussian process with a squared exponential kernel
                z_soma, z_dendrite = generate_gaussian_process_inputs(nb_steps, nb_latent, input_dt, input_params)

                # Generate the observed data (events and bursts) with the total number of spikes fixed by the nb_spike_per_burst parameter
                events, bursts, spikes, _, _, _ = generate_data_with_total_spikes(W_soma, W_dendrite, mu_event, mu_burst, z_soma, z_dendrite, data_dt, nb_spike_per_burst)
    

                # Fit the FA models
                W_est_event_rates, W_est_burst_fraction, W_est_firing_rates, _, _, _ = fit_all_fa_models_forced_latents(events, bursts, spikes)

                # Evaluate the subspace estimation error
                results = evaluate_all_fa_models(W_est_event_rates, W_est_burst_fraction, W_est_firing_rates, W_soma.cpu().numpy(), W_dendrite.cpu().numpy())

                # Append results to a dictionary
                repeat_results = update_dict(repeat_results, results)

            # Compute means and stds of the results over repeats
            results_means, results_stds = compute_performance_stats(repeat_results)

            # Update the all_results_means and all_results_stds dictionaries
            all_results_means = update_dict_element(all_results_means, results_means, i, j)
            all_results_stds = update_dict_element(all_results_stds, results_stds, i, j)

    

    save_data({'all_results_means': all_results_means, 'all_results_stds': all_results_stds}, 
              filename=f"results/{path_to_save}/subspace_estimation_error_data.pkl")
    
    meta_data = {'weight_params': weight_params, 'input_params': input_params, 'data_params': data_params, 'simulation_params': simulation_params}
    save_meta_data(meta_data, file_name=f"results/{path_to_save}/meta_data.json")

    return 

def recovered_subspace_error_plots(simulation_params, path_to_save):

    figsize = (2.0, 2.0)

    sigma_dendrite_list = simulation_params['sigma_dendrite_list']
    norm_distance_list = simulation_params['norm_distance_list']
    nb_repeats = simulation_params['nb_repeats']

    # Load the results data
    data = load_data(f"results/{path_to_save}/subspace_estimation_error_data.pkl")
    all_results_means = data['all_results_means']
    all_results_stds = data['all_results_stds']

    # Plot the error as a function of the angle between the somatic and dendritic weights for a fixed value of sigma_dendrite
    row = np.argwhere(np.isclose(sigma_dendrite_list, 0.2))[0][0] # Get the index of the row corresponding to sigma_dendrite = 0.2 for plotting
    plot_error_vs_angle(all_results_means, all_results_stds, norm_distance_list, nb_repeats, model_key='MFA', row=row,
                        file_name=f"results/{path_to_save}/plots/subspace_estimation_error_MFA.svg")
    
    plot_error_vs_angle(all_results_means, all_results_stds, norm_distance_list, nb_repeats, model_key='FA', row=row,
                        file_name=f"results/{path_to_save}/plots/subspace_estimation_error_FA.svg")
    
    # Plot the error as a function of both the angle between the somatic and dendritic weights and the burst noise level
    plot_error_heatmap(all_results_means, model_key='MFA', x_vals=norm_distance_list, y_vals=sigma_dendrite_list, 
                       figsize=figsize, file_name=f"results/{path_to_save}/plots/subspace_estimation_error_heatmap_MFA.svg")
    
    plot_error_heatmap(all_results_means, model_key='FA', x_vals=norm_distance_list, y_vals=sigma_dendrite_list, 
                       figsize=figsize, file_name=f"results/{path_to_save}/plots/subspace_estimation_error_heatmap_FA.svg")
    
    plot_error_difference_heatmap(all_results_means, figsize=figsize, 
                                  file_name=f"results/{path_to_save}/plots/subspace_estimation_error_difference_heatmap.svg")

    return

def recovered_dimension_error_plots(simulation_params, path_to_save):

    figsize = (2.0, 2.0)

    sigma_dendrite_list = simulation_params['sigma_dendrite_list']
    norm_distance_list = simulation_params['norm_distance_list']

    # Load the results data
    data = load_data(f"results/{path_to_save}/estimated_dimensions_data.pkl")
    all_results_means = data['all_results_means']

    # Plot the estimated dimensions as a function of the angle between the somatic and dendritic weights and the burst noise level
    plot_dimension_estimation_heatmap(all_results_means, x_vals=norm_distance_list, y_vals=sigma_dendrite_list, 
                                      figsize=figsize, file_name=f"results/{path_to_save}/plots/estimated_dimensions_heatmap.svg")


def simulate_recovered_latent_dynamics_examples():
    '''
    This function runs three example simulations with different angles (normalized Grassmann distances in 1-dim) between the somatic and dendritic 
    weights to show how the angle affects the ability of FA/MFA to recover the latent factors.

    Latent factors are generated from a Gaussian process with a squared exponential kernel, 
    and the observed data (events, bursts, and spikes) are generated from the latent factors using the generative model described in the paper.
    
    The results are saved in the "rate_vs_multiplexed_factor_analysis/example_simulation_X" folders, where X is the simulation number (1, 2, or 3). 
    Each folder contains the generated data, meta data, inferred latent factors, and plots of the observed counts distributions, 
    expected counts distributions, and latent trajectories.
    '''

    nb_latent = 1 # Fixed for this simulation, but can be changed for more complex simulations with more latent factors

    # Set seed for reproducibility
    set_seed(18) # My favorite number... 

    # Setup parameters for the simulation
    weight_params = {'nb_neurons': 200, 'sigma_soma': 0.2, 'sigma_dendrite': 0.2, 'mu_soma': 2.0, 'mu_dendrite': 0.0, 
                     'nb_dim_soma' :nb_latent, 'nb_dim_dendrite': nb_latent}
    input_params = {'tau_s':10.0, 'tau_d':10.0, 'I_s_amplitude':1, 'I_d_amplitude':1, 'nb_steps': 500, 'dt': 1.0}
    data_params = {'nb_spike_per_burst': 3, 'dt': 1.0}

    # Partial overlap between somatic and dendritic weights
    weight_params['norm_distance'] = 0.5
    path_to_save = "rate_vs_multiplexed_factor_analysis/example_simulation_1"
    recovered_latent_dynamics(weight_params, input_params, data_params, path_to_save)

    # Complete overlap between somatic and dendritic weights
    weight_params['norm_distance'] = 0.0
    path_to_save = "rate_vs_multiplexed_factor_analysis/example_simulation_2"
    recovered_latent_dynamics(weight_params, input_params, data_params, path_to_save)

    # Complete orthogonality between somatic and dendritic weights
    weight_params['norm_distance'] = 1.0
    path_to_save = "rate_vs_multiplexed_factor_analysis/example_simulation_3"
    recovered_latent_dynamics(weight_params, input_params, data_params, path_to_save) 

    return

def simulate_recovered_subspaces():
    '''
     This function runs a simulation varying the angle between the somatic and dendritic weights and the
     noise in the burst generation to show how these parameters affect the ability of FA/MFA to recover the latent subspaces.
     
     Recovery is measure using the subspace and dimension error as defined in the main text.
     '''

    nb_latent = 1 # Fixed for this simulation, but can be changed for more complex simulations with more latent factors

    # Set seed for reproducibility
    set_seed(18) # My favorite number... 

    # Run simulation varying the angle between the somatic and dendritic weights and the noise in the burst generation 
    # to show how these parameters affect the ability of FA/MFA to recover the latent subspaces
    weight_params = {'nb_neurons': 200, 'sigma_soma': 0.2, 'sigma_dendrite': 0.2, 'norm_distance': 0.5, 'mu_soma': 2.0, 'mu_dendrite': 0.0,
                     'nb_dim_soma': nb_latent, 'nb_dim_dendrite': nb_latent}
    input_params = {'tau_s':10.0, 'tau_d':10.0, 'I_s_amplitude':1, 'I_d_amplitude':1, 'nb_steps': 500, 'dt': 1.0}
    data_params = {'nb_spike_per_burst': 3, 'dt': 1.0}

    # With forced latents
    simulation_params = {'nb_repeats':100, 'norm_distance_list': np.linspace(0, 1, 10).tolist(), 
                         'sigma_dendrite_list': np.linspace(0.1, 0.4, 10).tolist()}
    
    path_to_save = "rate_vs_multiplexed_factor_analysis/subspace_estimation_error"
    recovered_subspace_error(weight_params, input_params, data_params, simulation_params, path_to_save)
    recovered_subspace_error_plots(simulation_params, path_to_save)

    # Estimate the number of latent dimensions
    path_to_save = "rate_vs_multiplexed_factor_analysis/estimated_dimensions"
    recovered_dimension_error(weight_params, input_params, data_params, simulation_params, path_to_save)
    recovered_dimension_error_plots(simulation_params, path_to_save)


    return




if __name__ == "__main__":

    # Fig. 1b - Example trajectories of the latent factors inferred by FA/MFA compared with the ground truth latent factors
    simulate_recovered_latent_dynamics_examples()

    # Fig. 1c-d - Subspace recovery errors as a function of the angle between the somatic and dendritic weights and the burst noise level
    simulate_recovered_subspaces()