import numpy as np
import torch
from neuron_models import Bursting_Model
from utils import set_seed


# Used my multile simualtions
def generate_input_weights_from_angles(generate_principal_angles_function, norm_distance, nb_dim_soma, nb_dim_dendrite, nb_units, W_s_unit=None, W_orth_unit=None):

    # Minimum dimension / rank
    r_dim = np.min([nb_dim_soma, nb_dim_dendrite])

    # Generate somatic weights and orthogonal components if needed
    if W_s_unit is None and W_orth_unit is None:

        W = np.random.normal(size=(nb_units, nb_dim_soma*2), loc=0.0, scale=1.0)
        Q, R = np.linalg.qr(W)

        W_s_unit = Q[:, :nb_dim_soma] # orthognormal weights
        W_orth_unit = Q[:, nb_dim_soma:] # orthogonal weights

    # Generate principal angles
    principal_angles = generate_principal_angles_function(r_dim, norm_distance)
    principal_angles_ordered = np.sort(principal_angles)
    S = np.diag(np.cos(principal_angles_ordered)) 

    # Generate left and right singular vectors
    M = np.random.randn(nb_dim_soma, nb_dim_dendrite)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)

    W_d_unit = W_s_unit @ U @ S @ Vt + W_orth_unit @ U @ np.sqrt(np.eye(r_dim) - S**2) @ Vt

    W_s = W_s_unit * np.sqrt(nb_units)  # Scale by sqrt of number of units. This is to keep the scale the same as the rank-1 case, since Q is orthonormal.
    W_d = W_d_unit * np.sqrt(nb_units)
    W_orth = W_orth_unit * np.sqrt(nb_units)

    return W_s, W_d, W_orth, principal_angles_ordered, U, Vt

def generate_principal_angles_projection(nb_dimensions, norm_distance, nb_tries=1000):

    angle_domain = (0, np.pi/2)

    if not (0.0 <= norm_distance <= 1.0):
        raise ValueError("norm_distance must be in [0, 1].")

    if norm_distance == 0.0:
        return np.zeros(nb_dimensions)

    S = nb_dimensions * norm_distance
    r = np.sqrt(S)          # radius in y = sin(theta) space

    # Rejection sampling to ensure angles are in [0, pi/2]
    for _ in range(nb_tries):  # Limit the number of attempts to avoid infinite loops

        random_directions = np.abs(np.random.randn(nb_dimensions))
        random_directions /= np.linalg.norm(random_directions)  # Normalize to lie on the unit sphere

        y = random_directions * r

        if np.max(y) > 1.0:
            continue

        angles = np.arcsin(y)  # Scale by r and take arcsin to get angles

        # check if all angles are in the desired range
        if np.all((angles >= angle_domain[0]) & (angles <= angle_domain[1])):
            return angles

    return "Failed to generate valid angles after many attempts."

def generate_principal_angles_grassmann(nb_dimensions, norm_grassmann_distance):

    angle_domain = (0, np.pi/2)
    max_grassmann_distance = (np.pi/2) * np.sqrt(nb_dimensions)

    # Special cases (norm = 0 or norm = 1)
    if norm_grassmann_distance == 0:
        return np.zeros(nb_dimensions)
    if norm_grassmann_distance == 1:
        return np.ones(nb_dimensions) * (np.pi/2)

    good_sample = False
    while not good_sample:

        random_angles = np.random.randn(nb_dimensions)
        normalization_factor = np.sqrt(np.sum(random_angles**2))
        normalized_angles = random_angles / normalization_factor
        angles = normalized_angles * norm_grassmann_distance * max_grassmann_distance

        # check if all angles are in the desired range
        if np.all((angles >= angle_domain[0]) & (angles <= angle_domain[1])):
            good_sample = True

    return angles




# Data generators for Fig 1 - Subspace and factor recovery simulation

def generate_one_dimensional_weights(nb_neurons, nb_latent, sigma_event, sigma_burst, norm_grassmann_distance, mu_s, mu_d):

    W_event, W_burst, _, _, _, _ = generate_grassmann_input_weights(norm_grassmann_distance, nb_latent, nb_latent, nb_neurons)
    W_event = W_event * (sigma_event / np.sqrt(nb_latent))
    W_burst = W_burst * (sigma_burst / np.sqrt(nb_latent))

    W_event = torch.Tensor(W_event)
    W_burst = torch.Tensor(W_burst)
    mu_event = torch.ones(nb_neurons)*mu_s
    mu_burst = torch.ones(nb_neurons)*mu_d

    return W_event, W_burst, mu_event, mu_burst

def generate_data_with_total_spikes(W_event, W_burst, mu_event, mu_burst, z_event, z_burst, dt, nb_spike_per_burst):
    events, bursts, z_event, z_burst, event_rate, burst_prob = data_generation(W_event, W_burst, mu_event, mu_burst, 
                                                                               z_event=z_event, z_burst=z_burst, dt=dt)
    
    spikes = events + (nb_spike_per_burst-1)*bursts
    burst_rate = burst_prob*event_rate
    spike_rate = event_rate + (nb_spike_per_burst-1)*burst_rate


    spikes = spikes.cpu().numpy().T
    events = events.cpu().numpy().T
    bursts = bursts.cpu().numpy().T

    return events, bursts, spikes, event_rate, burst_rate, spike_rate

def exponential_kerenl(x, y, tau):
    
    '''
    Exponential kernel function evlauated for two points x and y.
    Assume zero mean function. 
    
    inputs
    -------
    x: float
    tau: smoothing constant for covariance kernel
    
    retruns
    ------- 
    k: evaluated kernel
    
    '''
    
    k = np.exp((-1/2)*((x-y)/tau)**2)
    
    return k

def gaussian_process(x, tau, nb_samples=1, noise_variance=1e-6):
    """
    Sample from a Gaussian Process.
    
    Parameters:
    - kernel (callable): Kernel function that computes the covariance matrix. 
                         Should take two inputs and return a scalar.
    - X (ndarray): Input points of shape (n, d), where n is the number of points 
                   and d is the dimensionality of each point.
    - num_samples (int): Number of samples to draw from the GP.
    - noise_variance (float): Small noise variance for numerical stability.
    
    Returns:
    - samples (ndarray): GP samples of shape (num_samples, n).
    """
    
    # Compute covariance matrix
    n = x.shape[0]
    K = np.array([[exponential_kerenl(x1, x2, tau) for x2 in x] for x1 in x])
    
    # Add a small noise variance for numerical stability
    K += noise_variance * np.eye(n)
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(K)
    
    # Sample from the standard normal distribution
    z = np.random.normal(size=(n, nb_samples))
    
    # Compute samples using the covariance matrix
    samples = L @ z
    
    return samples.T

def generate_gaussian_process_inputs(nb_steps, nb_latent, dt, params):
    """
    Generate inputs for a Gaussian Process.

    Parameters:
    - nb_steps (int): Number of time steps.
    - nb_latent (int): Number of latent variables.
    - dt (float): Time step size.
    - params (dict): Dictionary containing GP parameters.

    Returns:
    - I_s_factors (torch.Tensor): GP samples for somatic inputs.
    - I_d_factors (torch.Tensor): GP samples for dendritic inputs.
    """

    time = np.arange(nb_steps)*dt
    
    I_s_factors = gaussian_process(x=time, tau=params['tau_s'], nb_samples=nb_latent)*params['I_s_amplitude']
    I_d_factors = gaussian_process(x=time, tau=params['tau_d'], nb_samples=nb_latent)*params['I_d_amplitude']

    I_s_factors = torch.Tensor(I_s_factors)
    I_d_factors = torch.Tensor(I_d_factors)
    
    return I_s_factors, I_d_factors

def generate_grassmann_input_weights(norm_grassmann_distance, nb_dim_soma, nb_dim_dendrite, nb_units, W_s_unit=None, W_orth_unit=None):
    """
    Generate input weights for somatic and dendritic subspaces based on the specified normalized Grassmann distance.
    
    :param norm_grassmann_distance: Normalized Grassmann distance between the somatic and dendritic subspaces (0 to 1).
    :param nb_dim_soma: Number of somatic input dimensions.
    :param nb_dim_dendrite: Number of dendritic input dimensions.
    :param nb_units: Number of units (neurons).
    :param W_s_unit: Optional precomputed orthonormal somatic weight matrix.
    :param W_orth_unit: Optional precomputed orthogonal complement of the somatic weight matrix.
    :return: Tuple containing somatic weight matrix, dendritic weight matrix, orthogonal component, ordered principal angles, and singular vectors U and Vt.
    """

    # Minimum dimension / rank
    r_dim = np.min([nb_dim_soma, nb_dim_dendrite])

    # Generate somatic weights and orthogonal components if needed
    if W_s_unit is None and W_orth_unit is None:

        W = np.random.normal(size=(nb_units, nb_dim_soma*2), loc=0.0, scale=1.0)
        Q, R = np.linalg.qr(W)

        W_s_unit = Q[:, :nb_dim_soma] # orthognormal weights
        W_orth_unit = Q[:, nb_dim_soma:] # orthogonal weights

    # Generate principal angles
    principal_angles = generate_principal_angles(r_dim, norm_grassmann_distance, (0, np.pi/2))
    principal_angles_ordered = np.sort(principal_angles)
    S = np.diag(np.cos(principal_angles_ordered)) 

    # Generate left and right singular vectors
    M = np.random.randn(nb_dim_soma, nb_dim_dendrite)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)

    W_d_unit = W_s_unit @ U @ S @ Vt + W_orth_unit @ U @ np.sqrt(np.eye(r_dim) - S**2) @ Vt

    W_s = W_s_unit * np.sqrt(nb_units)  # Scale by sqrt of number of units. This is to keep the scale the same as the rank-1 case, since Q is orthonormal.
    W_d = W_d_unit * np.sqrt(nb_units)
    W_orth = W_orth_unit * np.sqrt(nb_units)

    return W_s, W_d, W_orth, principal_angles_ordered, U, Vt

def generate_principal_angles(nb_dimensions, norm_grassmann_distance, angle_domain):
    """
    Generate principal angles for subspaces based on the specified normalized Grassmann distance.
    
    :param nb_dimensions: Number of dimensions of the subspaces.
    :param norm_grassmann_distance: Normalized Grassmann distance between the subspaces (0 to 1).
    :param angle_domain: Tuple specifying the allowed range of angles (min_angle, max_angle).

    :return: Array of principal angles.
    """

    max_grassmann_distance = (np.pi/2) * np.sqrt(nb_dimensions)

    # Special cases (norm = 0 or norm = 1)
    if norm_grassmann_distance == 0:
        return np.zeros(nb_dimensions)
    if norm_grassmann_distance == 1:
        return np.ones(nb_dimensions) * (np.pi/2)

    good_sample = False
    while not good_sample:

        random_angles = np.random.randn(nb_dimensions)
        normalization_factor = np.sqrt(np.sum(random_angles**2))
        normalized_angles = random_angles / normalization_factor
        angles = normalized_angles * norm_grassmann_distance * max_grassmann_distance

        # check if all angles are in the desired range
        if np.all((angles >= angle_domain[0]) & (angles <= angle_domain[1])):
            good_sample = True

    return angles


# Data generators for Fig 2 - Spiking data
def simulate_bursting_model(I_s_factors, I_d_factors, W_s, W_d, input_params, nb_units, time_step, nb_steps, neuron_params, noise_params, device, record_potential=False):
    """
    Simulate the bursting model for a given set of inputs and weights.
    
    Returns the recorded somatic and dendritic membrane potentials and spikes if record_potential is True, otherwise only returns spikes.
    
    :param I_s_factors: Somatic input factors.
    :param I_d_factors: Dendritic input factors.
    :param W_s: Somatic weight matrix.
    :param W_d: Dendritic weight matrix.
    :param input_params: Dictionary containing input parameters such as baseline currents.
    :param nb_units: Number of units in the model.
    :param time_step: Time step for the simulation.
    :param nb_steps: Number of simulation steps.
    :param neuron_params: Dictionary containing neuron parameters.
    :param noise_params: Dictionary containing noise parameters.
    :param device: Device to run the simulation on (e.g., 'cpu' or 'cuda').
    :param record_potential: Boolean indicating whether to record membrane potentials.
    """


    # Initialize arrays for somatic activity
    spk_rec = torch.empty((nb_units, nb_steps), device='cpu')
    if record_potential:
        mem_s_rec = torch.empty((nb_units, nb_steps), device='cpu')
        mem_d_rec = torch.empty((nb_units, nb_steps), device='cpu')
    
    I_s_factors = I_s_factors.to(device)
    I_d_factors = I_d_factors.to(device)
    W_s = W_s.to(device)
    W_d = W_d.to(device)
    
    I_s_baseline = torch.tensor(input_params['I_s_baseline']).to(device)
    I_d_baseline = torch.tensor(input_params['I_d_baseline']).to(device)

    with torch.no_grad():
        # Run model forward pass and move results to CPU
        BM = Bursting_Model(num_units=nb_units, time_step=time_step, 
                                      neuron_params=neuron_params, noise_params=noise_params).to(device)

        if record_potential:
            mem_s_rec, mem_d_rec, spk_rec = BM(I_s_factors=I_s_factors, I_d_factors=I_d_factors,
                        W_s=W_s, W_d=W_d, I_s_baseline=I_s_baseline, I_d_baseline=I_d_baseline, record_potential=record_potential)

            spk_rec = spk_rec.detach().cpu()

            # Free GPU memory for each batch
            del BM
            torch.cuda.empty_cache()

            return mem_s_rec, mem_d_rec, spk_rec

        else:
            spk_rec = BM(I_s_factors=I_s_factors, I_d_factors=I_d_factors,
                            W_s=W_s, W_d=W_d, I_s_baseline=I_s_baseline, I_d_baseline=I_d_baseline)

            spk_rec = spk_rec.detach().cpu()

            # Free GPU memory for each batch
            del BM
            torch.cuda.empty_cache()

            return spk_rec

def generate_constant_gaussian_factors(nb_latent_soma, nb_latent_dendrite, nb_bins, bin_width):

    """
    Generate constant Gaussian latent factors for somatic and dendritic inputs.
    
    :param nb_latent_soma: Number of somatic latent factors.
    :param nb_latent_dendrite: Number of dendritic latent factors.
    :param nb_bins: Number of time bins.
    :param bin_width: Width of each time bin.

    :return: Tuple containing somatic and dendritic latent factors as torch tensors.
    """

    z_event_tmp = np.random.randn(nb_latent_soma, nb_bins)
    z_event = np.repeat(z_event_tmp, bin_width, axis=1)

    z_burst_tmp = np.random.randn(nb_latent_dendrite, nb_bins)
    z_burst = np.repeat(z_burst_tmp, bin_width, axis=1)

    z_event = torch.tensor(z_event, dtype=torch.float)
    z_burst = torch.tensor(z_burst, dtype=torch.float)

    return z_event, z_burst

def generate_simulation_weights(weight_params, generate_principal_angles_function):

    """
    Generate simulation weights for somatic and dendritic inputs based on the specified weight parameters and principal angles function.

    :param weight_params: Dictionary containing weight parameters such as norm distance, number of dimensions, number of neurons, and variances.
    :param generate_principal_angles_function: Function to generate principal angles for the weight matrices.

    :return: Tuple containing somatic and dendritic weight matrices as torch tensors.
    """

    # Unpack weight parameters
    norm_distance = weight_params['norm_distance']
    nb_dim_soma = weight_params['nb_dim_soma']
    nb_dim_dendrite = weight_params['nb_dim_dendrite']
    nb_neurons = weight_params['nb_neurons']
    sigma_soma = weight_params['sigma_soma']
    sigma_dendrite = weight_params['sigma_dendrite']

    # Generate synaptic weights with the desired alignment properties
    W_s, W_d, _, _, _, _ = generate_input_weights_from_angles(generate_principal_angles_function=generate_principal_angles_function, 
                                                                                    norm_distance=norm_distance, nb_dim_soma=nb_dim_soma, 
                                                                                    nb_dim_dendrite=nb_dim_dendrite, nb_units=nb_neurons)

    # Scale the weights by the desired variance and the number of latent dimensions
    W_s *= sigma_soma / np.sqrt(nb_dim_soma)
    W_d *= sigma_dendrite / np.sqrt(nb_dim_dendrite)

    # Convert weights to torch tensors
    W_s = torch.tensor(W_s, dtype=torch.float)
    W_d = torch.tensor(W_d, dtype=torch.float)

    return W_s, W_d

# Data generators for Fig 4 - Sensitivity analysis 
def data_generation(W_event, W_burst, mu_event, mu_burst, z_event=None, z_burst=None, w_burst=None, dt=1.0, nb_samples=100):

    _ , nb_latents_event = W_event.shape
    _ , nb_latents_burst = W_burst.shape

    # Sample latent variables
    if z_event is None:
        z_event = torch.randn(nb_latents_event, nb_samples, device=W_event.device)
    if z_burst is None:
        z_burst = torch.randn(nb_latents_burst, nb_samples, device=W_burst.device)

    # Poisson rate for events
    log_rate = W_event @ z_event + mu_event[:, None]
    log_rate = torch.clamp(log_rate, max=10)  # Prevent overflow in exp
    event_rate = torch.exp(log_rate)

    # Binomial probability for bursts
    if w_burst is None:
        burst_prob = torch.sigmoid(W_burst @ z_burst + mu_burst[:, None])
    else:
        w_burst = torch.as_tensor(w_burst, dtype=log_rate.dtype, device=log_rate.device)
        log_centered_rate = log_rate - log_rate.mean(axis=1, keepdim=True)
        burst_prob = torch.sigmoid(w_burst[:, None]*log_centered_rate + W_burst @ z_burst + mu_burst[:, None])

    # Sample Poisson counts for events
    poisson = torch.distributions.Poisson(event_rate * dt)
    events = poisson.sample()

    # Sample Binomial counts for bursts
    binomial = torch.distributions.Binomial(events, burst_prob)
    bursts = binomial.sample()
    
    return events, bursts, z_event, z_burst, event_rate, burst_prob

def generate_random_input_weights(weight_params):

    nb_dim_soma = weight_params['nb_dim_soma']
    nb_dim_dendrite = weight_params['nb_dim_dendrite']
    nb_neurons = weight_params['nb_neurons']
    sigma_soma = weight_params['sigma_soma']
    sigma_dendrite = weight_params['sigma_dendrite']

    # Generate random synaptic weights
    W_s = np.random.randn(nb_neurons, nb_dim_soma)
    W_d = np.random.randn(nb_neurons, nb_dim_dendrite)

    # Scale the weights by the desired variance and the number of latent dimensions
    W_s *= sigma_soma / np.sqrt(nb_dim_soma)
    W_d *= sigma_dendrite / np.sqrt(nb_dim_dendrite)

    # Convert weights to torch tensors
    W_s = torch.tensor(W_s, dtype=torch.float)
    W_d = torch.tensor(W_d, dtype=torch.float)

    return W_s, W_d



# Old data generators for testing model fitting code - to be removed
def data_generation_simple(W_event, W_burst, mu_event, mu_burst, z_event=None, z_burst=None, dt=1.0, nb_samples=100):

    _ , nb_latents_event = W_event.shape


    # Sample latent variables
    if z_event is None:
        z_event = torch.randn(nb_latents_event, nb_samples, device=W_event.device)

    # Poisson rate for events
    log_rate = W_event @ z_event + mu_event[:, None]
    log_rate = torch.clamp(log_rate, max=10)  # Prevent overflow in exp
    event_rate = torch.exp(log_rate)

    print(event_rate.shape, W_burst.shape, mu_burst.shape)

    log_rate_mean = log_rate.mean(axis=1)

    # Binomial probability for bursts
    burst_prob = torch.sigmoid(W_burst[:, None] * (log_rate-log_rate_mean[:, None]) + mu_burst[:, None])

    # Sample Poisson counts for events
    poisson = torch.distributions.Poisson(event_rate * dt)
    events = poisson.sample()

    # Sample Binomial counts for bursts
    binomial = torch.distributions.Binomial(events, burst_prob)
    bursts = binomial.sample()
    
    return events, bursts, z_event, z_burst, event_rate, burst_prob

def create_datasets(nb_neurons, nb_samples, weight_params, data_type):

    nb_val_samples = int(nb_samples * 0.2)
    
    W_event, W_burst = simulation_weights(weight_params)

    mu_s = weight_params['mu_s']
    mu_d = weight_params['mu_d']

    mu_event = torch.ones(nb_neurons) * mu_s
    mu_burst = torch.ones(nb_neurons) * mu_d

    if data_type == "null_data":

        W_burst = W_burst[:, 0] # A single vector only
        events_train, bursts_train, _, _, _, _ = data_generation_simple(W_event, W_burst, mu_event, mu_burst, nb_samples=nb_samples)
        events_val, bursts_val, _, _, _, _ = data_generation_simple(W_event, W_burst, mu_event, mu_burst, nb_samples=nb_val_samples)

    elif data_type == "multiplexed_data":

        events_train, bursts_train, _, _, _, _ = data_generation(W_event, W_burst, mu_event, mu_burst, nb_samples=nb_samples)
        events_val, bursts_val, _, _, _, _ = data_generation(W_event, W_burst, mu_event, mu_burst, nb_samples=nb_val_samples)

    else: 
        raise ValueError(f"Invalid data type: {data_type}")

    # Transpose to shape (nb_samples, nb_neurons)
    events_train = events_train.T
    bursts_train = bursts_train.T
    events_val = events_val.T
    bursts_val = bursts_val.T

    data_train = {"events": events_train, "bursts": bursts_train}
    data_val = {"events": events_val, "bursts": bursts_val}

    return data_train, data_val, W_event, W_burst

