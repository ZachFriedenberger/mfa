from asyncio import events

from pyparsing import results
import torch
import numpy as np
np.seterr(invalid='ignore', divide='ignore') 
from tqdm import trange
import matplotlib.pyplot as plt

from data_generators import generate_constant_gaussian_factors, simulate_bursting_model, generate_simulation_weights
from data_generators import generate_principal_angles_grassmann
from utils import create_folder, save_data, load_data, set_seed
from plotting_functions import plot_mean_counts_histogram, plot_total_counts_histogram, plot_example_trial, plot_burst_statistics_vs_fraction, plot_burst_length_intact_vs_silenced
from misc import compute_total_counts, burst_statistics




def simulate_example_trial(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save):
    """
    Simulate an example trial of the spiking model and save the results. Generate single trial voltage traces and raster plot.
    
    :param W_s: Soma weight matrix.
    :param W_d: Dendrite weight matrix.
    :param input_params_latents: Dictionary containing latent input parameters.
    :param input_params_spiking: Dictionary containing spiking input parameters.
    :param neuron_params: Dictionary containing neuron parameters.
    :param noise_params: Dictionary containing noise parameters.
    :param device: Device to run the simulation on (e.g., 'cpu' or 'cuda').
    :param path_to_save: Path to save the simulation results.
    :return: None
    """

    # Create folders to save results
    create_folder(f"results/{path_to_save}")

    # Unpack simulation parameters
    nb_steps_integration = input_params_latents['nb_steps']*10 # convert from 1 ms to 0.1 ms steps
    time_step_integration = torch.tensor(input_params_latents['dt']/10) # ms

    input_params_spiking['time_step_integration'] = time_step_integration
    input_params_spiking['nb_steps_integration'] = nb_steps_integration

    bin_width = input_params_latents['bin_width'] # ms
    nb_bins = input_params_latents['nb_steps'] // bin_width

    nb_neurons, nb_latent_soma = W_s.shape
    _, nb_latent_dendrite = W_d.shape

    # Generate latent factors for the example trial
    z_event, z_burst = generate_constant_gaussian_factors(nb_latent_soma, nb_latent_dendrite, 
                                                          nb_bins, bin_width/time_step_integration) # bin width in terms of integration steps 0.1 ms
        
    # Run spiking simulation and record the membrane potentials for the example simulation
    mem_s_rec, mem_d_rec, spk_rec =  simulate_bursting_model(z_event, z_burst, W_s, W_d, input_params_spiking, 
                                                             nb_neurons, time_step_integration, nb_steps_integration, 
                                                             neuron_params, noise_params, device, record_potential=True)
        
    # Save simulated data for plotting
    spiking_data = {'mem_s': mem_s_rec, 'mem_d': mem_d_rec, 'spk': spk_rec}
    input_data = {'z_event': z_event, 'z_burst': z_burst, 'W_s': W_s, 'W_d': W_d}

    save_data(spiking_data, f"results/{path_to_save}/spiking_data.pkl")
    save_data(input_data, f"results/{path_to_save}/input_data.pkl")

    # Save metadata about the simulation parameters
    metadata = {'input_params_latents': input_params_latents, 'input_params_spiking': input_params_spiking, 
                'weight_params': weight_params, 'neuron_params': neuron_params, 'noise_params': noise_params}
    
    save_data(metadata, f"results/{path_to_save}/metadata.pkl")

    return

def simulate_trials(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save, nb_trials=500):
    """
    Simulate multiple trials of the bursting neuron model and save the total count data.
    
    :param W_s: Somatic weight matrix.
    :param W_d: Dendritic weight matrix.
    :param input_params_latents: Latent input parameters.
    :param input_params_spiking: Spiking input parameters.
    :param neuron_params: Neuron parameters.
    :param noise_params: Noise parameters.
    :param device: Device to run the simulation on.
    :param path_to_save: Path to save the simulation results.
    :param nb_trials: Number of trials to simulate.
    """

    # Create folders to save results
    create_folder(f"results/{path_to_save}")
    
    # Unpack simulation parameters
    nb_steps_integration = input_params_latents['nb_steps']*10 # convert from 1 ms to 0.1 ms steps
    time_step_integration = torch.tensor(input_params_latents['dt']/10) # ms

    input_params_spiking['time_step_integration'] = time_step_integration
    input_params_spiking['nb_steps_integration'] = nb_steps_integration

    bin_width = input_params_latents['bin_width'] # ms
    nb_bins = input_params_latents['nb_steps'] // bin_width

    nb_neurons, nb_latent_soma = W_s.shape
    _, nb_latent_dendrite = W_d.shape

    spk_rec_trials = torch.zeros((nb_trials, nb_neurons, nb_steps_integration), dtype=torch.float)
    for trial in trange(nb_trials):

        # Generate latent factors for the example trial
        z_event_trial, z_burst_trial = generate_constant_gaussian_factors(nb_latent_soma, nb_latent_dendrite, nb_bins, bin_width/time_step_integration) # bin width in terms of integration steps 0.1 ms
        
        # Run spiking simulation and record the membrane potentials for the example simulation
        spk_rec =  simulate_bursting_model(z_event_trial, z_burst_trial, W_s, W_d, input_params_spiking, nb_neurons, time_step_integration, nb_steps_integration, 
                                            neuron_params, noise_params, device, record_potential=False)
        
        spk_rec_trials[trial] = spk_rec

    # Save spiking data and weights for all trials
    save_data({'spk_rec_trials': spk_rec_trials}, f"results/{path_to_save}/spk_rec_trials.pkl")
    #save_data({'W_s': W_s, 'W_d': W_d}, f"results/{path_to_save}/weights.pkl")


    # Compute total counts across all trials
    total_counts = compute_total_counts(spk_rec_trials, time_step_integration, nb_steps_integration, 
                                        bin_width, nb_trials, nb_neurons, input_params_latents['max_time'], input_params_latents['min_time'])
    
    # Save total counts data
    save_data(total_counts, f"results/{path_to_save}/total_counts.pkl")

    # Save metadata about the simulation parameters
    metadata = {'input_params_latents': input_params_latents, 'input_params_spiking': input_params_spiking, 
                'weight_params': weight_params, 'neuron_params': neuron_params, 'noise_params': noise_params}
    save_data(metadata, f"results/{path_to_save}/metadata.json")

    return

def plot_simulated_trials_counts(path_to_save):

    # Load total counts data
    total_counts = load_data(f"results/{path_to_save}/total_counts.pkl")

    file_name = f"results/{path_to_save}/mean_counts_histogram.svg"
    plot_mean_counts_histogram(total_counts, bins=10, file_name=file_name)

    file_name = f"results/{path_to_save}/total_counts_histogram.svg"
    plot_total_counts_histogram(total_counts, file_name=file_name)

    return

def simulate_multiple_trials(W_s, W_d, nb_trials, input_params_latents, input_params_spiking, neuron_params, noise_params, device, results_folder):
    """
    Simulate multiple trials for the given neuron and input parameters. Saves the spiking data and total counts for both training and testing datasets.

    :param W_s: Somatic weight matrix.
    :param W_d: Dendritic weight matrix.
    :param nb_trials: Total number of trials to simulate.
    :param input_params_latents: Latent input parameters.
    :param input_params_spiking: Spiking input parameters.
    :param neuron_params: Neuron parameters.
    :param noise_params: Noise parameters.
    :param device: Device to run the simulation on.
    :return: None
    """

    nb_test_trials = nb_trials // 5 # Number of test trials (20% of total trials)

    # Simulate two-compartment model with latents to both compartments
    path_to_save = f"{results_folder}/two_compartments_multiple_trials/training_data"
    simulate_trials(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save, nb_trials=nb_trials)
    plot_simulated_trials_counts(path_to_save)

    path_to_save = f"{results_folder}/two_compartments_multiple_trials/testing_data"
    simulate_trials(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save, nb_trials=nb_test_trials)
    plot_simulated_trials_counts(path_to_save)

    # Simulate single-compartment model with latents only to the soma - set dendritic input weights to zero and remove spike frequency adaptation
    neuron_params['g_s'] = 0
    W_d_zero = torch.zeros_like(W_d)

    path_to_save = f"{results_folder}/single_compartment_multiple_trials/training_data"
    simulate_trials(W_s, W_d_zero, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save, nb_trials=nb_trials) 
    plot_simulated_trials_counts(path_to_save)

    path_to_save = f"{results_folder}/single_compartment_multiple_trials/testing_data"
    simulate_trials(W_s, W_d_zero, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save, nb_trials=nb_test_trials) 
    plot_simulated_trials_counts(path_to_save)

    return

def simulate_single_trials(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, results_folder):
    """
    Simulate a single trial for the given neuron and input parameters.
    
    :param W_s: Somatic weight matrix.
    :param W_d: Dendritic weight matrix.
    :param input_params_latents: Latent input parameters.
    :param input_params_spiking: Spiking input parameters.
    :param neuron_params: Neuron parameters.
    :param noise_params: Noise parameters.
    :param device: Device to run the simulation on.
    :return: None
    """

    ### Full two-comapartment model with latents to both compartments
    path_to_save = f"{results_folder}/two_compartments_example_trial"
    plot_params = {'min_time': 250, 'max_time': 1250, 'fig_size':(1.15, 1.0)} # Only plot the 1 second of simualtion for clarity

    simulate_example_trial(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save)
    plot_example_trial(plot_params, path_to_save)

    ### Single-compartment model with latents only to the soma - set dendritic input weights to zero and remove spike frequency adaptation
    path_to_save = f"{results_folder}/single_compartment_example_trial"
    neuron_params['g_s'] = 0 # No dendritic input, only somatic input
    W_d_zero = torch.zeros_like(W_d)
    simulate_example_trial(W_s, W_d_zero, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save)
    plot_example_trial(plot_params, path_to_save)

    return

def simulate_multiple_mixed_trials(W_s, W_d, frac_intact, nb_trials, input_params_latents, input_params_spiking, neuron_params,
                          noise_params, device, results_folder, mix_seed=0, seed=18):
    """
    Simulate a mixed population: a random fraction of neurons is intact (dendritic coupling g_s and dendritic input W_d),
    the rest are silenced (g_s = 0, W_d rows = 0). All neurons share the same latents, somatic weights, noise and adaptation.

    :param frac_intact: Fraction of intact neurons (0 = all silenced, 1 = all intact).
    :param mix_seed: Seed for choosing which neurons are intact (separate RNG, so it does not shift the latents).
    :param seed: Global seed reset before each split, so every frac_intact sees the same latents and noise.
    :return: Boolean mask of intact neurons.
    """
    nb_neurons = W_s.shape[0]
    nb_test_trials = nb_trials // 5

    # Choose intact neurons with a dedicated RNG
    rng = np.random.default_rng(mix_seed)
    intact = np.zeros(nb_neurons, dtype=bool)
    intact[rng.choice(nb_neurons, int(round(frac_intact * nb_neurons)), replace=False)] = True
    intact_t = torch.from_numpy(intact.astype(np.float32))

    # Per-neuron dendritic coupling and dendritic input (zero rows for silenced neurons, incl. private columns)
    neuron_params['g_s'] = neuron_params['g_s'] * intact_t

    folder = f"{results_folder}/mixed_{frac_intact:.2f}_multiple_trials"
    for split, n in (("training_data", nb_trials), ("testing_data", nb_test_trials)):

        set_seed(seed + (split == "testing_data"))  # same latents/noise across fractions; train and test differ

        path_to_save = f"{folder}/{split}"
        simulate_trials(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params,
                        device, path_to_save, nb_trials=n)
        save_data({'intact_mask': intact, 'frac_intact': frac_intact, 'mix_seed': mix_seed},
                  f"results/{path_to_save}/intact_mask.pkl")
        plot_simulated_trials_counts(path_to_save)

    return intact

def simulate_mixed_single_trial(W_s, W_d, frac_intact, input_params_latents, input_params_spiking, neuron_params,
                                noise_params, device, results_folder, mix_seed=0, seed=18):
    """
    Simulate and plot an example trial of the mixed population (same intact neurons as simulate_multiple_mixed_trials).

    :param frac_intact: Fraction of intact neurons (0 = all silenced, 1 = all intact).
    :param mix_seed: Seed for choosing which neurons are intact (same RNG as simulate_multiple_mixed_trials).
    :param seed: Global seed reset before the trial, so every frac_intact sees the same latents and noise.
    :return: Boolean mask of intact neurons.
    """
    nb_neurons = W_s.shape[0]

    # Choose intact neurons with a dedicated RNG
    rng = np.random.default_rng(mix_seed)
    intact = np.zeros(nb_neurons, dtype=bool)
    intact[rng.choice(nb_neurons, int(round(frac_intact * nb_neurons)), replace=False)] = True
    intact_t = torch.from_numpy(intact.astype(np.float32))

    neuron_params['g_s'] = neuron_params['g_s'] * intact_t

    path_to_save = f"{results_folder}/mixed_{frac_intact:.2f}_example_trial"
    plot_params = {'min_time': 250, 'max_time': 1250, 'fig_size':(1.15, 1.0)} # Only plot the 1 second of simualtion for clarity

    set_seed(seed) # same latents/noise across fractions
    simulate_example_trial(W_s, W_d, input_params_latents, input_params_spiking, neuron_params, noise_params, device, path_to_save)
    save_data({'intact_mask': intact, 'frac_intact': frac_intact, 'mix_seed': mix_seed}, f"results/{path_to_save}/intact_mask.pkl")
    plot_example_trial(plot_params, path_to_save)

    return intact

def plot_mixed_burst_statistics(results_folder, frac_intact_list, color='k'):
    """
    Plot the baseline burst fraction and mean burst length of the mixed example trials against the fraction of intact neurons.

    :param results_folder: Results folder of the simulations.
    :param frac_intact_list: Fractions of intact neurons to plot (each needs a mixed_{f:.2f}_example_trial folder).
    :param color: Color of the points.
    """
    burst_fraction, burst_length = [], []
    for f in frac_intact_list:
        path_to_save = f"{results_folder}/mixed_{f:.2f}_example_trial"
        spk_rec = load_data(f"results/{path_to_save}/spiking_data.pkl")['spk']
        input_params_spiking = load_data(f"results/{path_to_save}/metadata.pkl")['input_params_spiking']

        stats = burst_statistics(spk_rec.cpu().numpy(), input_params_spiking['nb_steps_integration'], input_params_spiking['time_step_integration'].numpy())
        burst_fraction.append((stats['burst_fraction_mean'], stats['burst_fraction_sem']))
        burst_length.append((stats['burst_length_mean'], stats['burst_length_sem']))
        print(f"f={f:.2f}  F0={stats['burst_fraction_mean']:.2f}+-{stats['burst_fraction_sem']:.2f} %  "
              f"L0={stats['burst_length_mean']:.2f}+-{stats['burst_length_sem']:.2f}  ({stats['nb_bursts']} bursts)")

    plot_burst_statistics_vs_fraction(frac_intact_list, burst_fraction, burst_length, color=color,
                                      file_name=f"results/{results_folder}/burst_statistics_vs_fraction.svg")
    plt.close('all')

    return

def plot_mixed_burst_length_distributions(results_folder):
    """
    Plot the burst length distributions of the fully intact (f = 1) and fully silenced (f = 0) example trials on one plot.

    :param results_folder: Results folder of the simulations (needs mixed_1.00_example_trial and mixed_0.00_example_trial).
    """
    spk_recs = [load_data(f"results/{results_folder}/mixed_{f:.2f}_example_trial/spiking_data.pkl")['spk'].cpu().numpy() for f in (1.0, 0.0)]
    input_params_spiking = load_data(f"results/{results_folder}/mixed_1.00_example_trial/metadata.pkl")['input_params_spiking']

    plot_burst_length_intact_vs_silenced(spk_recs[0], spk_recs[1], input_params_spiking['nb_steps_integration'], input_params_spiking['time_step_integration'].numpy(),
                                         file_name=f"results/{results_folder}/burst_length_intact_vs_silenced.svg")
    plt.close('all')

    return


def reset_parameters():
    """
    Reset and return the simulation parameters.
    
    :return: Tuple containing input_params_spiking, neuron_params, noise_params, weight_params, input_params_latents
    """

    # Input parameters
    input_params_spiking = {'I_s_baseline':250, 'I_d_baseline':250}

    # Neuron parameters
    neuron_params = {'E_L': -70, 'tau_s': 16, 'g_s': 1300, 'C_s': 370, 'tau_ws': 100, 
                        'b_ws': -200, 'tau_d': 7, 'g_d': 1200, 'c_d': 2600, 'C_d': 170,
                        'a_wd': -26, 'tau_wd': 30, 'E_d': -38, 'D_d': 6, 'V_T': -50,
                        'abs_ref': 3}

    # Noise parameters
    noise_params = {'mu_s': 0, 'sigma_s': 650, 'mu_d': -270, 'sigma_d': 450, 'tau_noise': 2}

    # Weight parameters
    weight_params = {'norm_distance': 0.5, 'nb_dim_soma': 3, 'nb_dim_dendrite': 3, 'nb_neurons': 50, 'sigma_soma': 180.0, 'sigma_dendrite': 180.0, 
                     'sigma_soma_private': 120.0, 'sigma_dendrite_private': 120.0}

    # Input parameters for latent factor generation
    input_params_latents = {'nb_steps': 5250, 'dt': 1.0, 'bin_width': 250, 'min_time': 250, 'max_time': 5250} # Only simulate the first second for clarity

    return input_params_spiking, neuron_params, noise_params, weight_params, input_params_latents



if __name__ == "__main__":

    results_folder = "evidence_of_burst_coding"
    create_folder(results_folder)


    # Configure device - cpu here for memory reasons
    device = torch.device('cpu')
    print(f"Using device: {device}")

    nb_trials = 500

    #################### Generate weights and save them for later use #########################
    set_seed(18) # Set seed for reproducibility of the weights.

    # Reset parameters
    input_params_spiking, neuron_params, noise_params, weight_params, input_params_latents = reset_parameters()

    # Generate simulation weights
    W_s, W_d = generate_simulation_weights(weight_params=weight_params, generate_principal_angles_function=generate_principal_angles_grassmann)
    save_data({'W_s': W_s, 'W_d': W_d}, f"results/{results_folder}/weights.pkl")

    # Add a private noise component to the weights
    W_s = torch.hstack([W_s, torch.eye(W_s.shape[0]) * weight_params['sigma_soma_private']])
    W_d = torch.hstack([W_d, torch.eye(W_d.shape[0]) * weight_params['sigma_dendrite_private']])
    save_data({'W_s': W_s, 'W_d': W_d}, f"results/{results_folder}/weights_with_private.pkl")

    # Run multiple mixed trials for different fractions of intact neurons
    frac_intact_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for frac_intact in frac_intact_list:
        input_params_spiking, neuron_params, noise_params, weight_params, input_params_latents = reset_parameters()
        simulate_multiple_mixed_trials(W_s, W_d, frac_intact, nb_trials, input_params_latents, input_params_spiking,
                                       neuron_params, noise_params, device, results_folder)

    # Run mixed single trials for different fractions of intact neurons
    frac_intact_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for frac_intact in frac_intact_list:
        input_params_spiking, neuron_params, noise_params, weight_params, input_params_latents = reset_parameters()
        simulate_mixed_single_trial(W_s, W_d, frac_intact, input_params_latents, input_params_spiking,
                                    neuron_params, noise_params, device, results_folder)

    # Plot the burst statistics and burst length distributions for the mixed trials
    plot_mixed_burst_statistics(results_folder, frac_intact_list)
    plot_mixed_burst_length_distributions(results_folder)