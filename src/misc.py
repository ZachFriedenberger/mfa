import numpy as np
import torch



def bootstrap_confidence_interval(log_evidence_score_difference, events_per_sample=None, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap the mean score difference over test samples, with a two-sided p-value for difference = 0.

    :param log_evidence_score_difference: Tensor of per-sample score differences (nb_samples,).
    :param events_per_sample: Optional tensor of total event counts per sample (nb_samples,). If given, the
                              statistic is the per-event difference (sum of differences / sum of events).
    :param n_bootstrap: Number of bootstrap resamples (sets the smallest reportable p-value, 1/n_bootstrap).
    :param alpha: Significance level for the confidence interval.
    :return: Estimate, lower bound, upper bound and p-value.
    """

    difference = log_evidence_score_difference.flatten()
    nb_samples = difference.shape[0]

    # Resample samples with replacement; each sample keeps its own difference and event count
    idx = torch.randint(0, nb_samples, (n_bootstrap, nb_samples), device=difference.device)
    if events_per_sample is None:
        estimate = difference.mean().item()
        bootstrap_stats = difference[idx].mean(dim=1)
    else:
        events = events_per_sample.flatten().to(difference.device)
        estimate = (difference.sum() / events.sum()).item()
        bootstrap_stats = difference[idx].sum(dim=1) / events[idx].sum(dim=1)
    bootstrap_stats = bootstrap_stats.cpu().numpy()

    # Confidence interval
    lower_bound = np.percentile(bootstrap_stats, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    # Two-sided p-value: twice the smaller tail of the bootstrap distribution beyond 0
    p_value = 2 * min((bootstrap_stats <= 0).mean(), (bootstrap_stats >= 0).mean())
    p_value = min(1.0, max(p_value, 1.0 / n_bootstrap))

    return estimate, lower_bound, upper_bound, p_value

def compute_mean_sem_scores(scores_list, dim_list):

    mean_scores_list, sem_scores_list  = [], []
    for dim in dim_list:

        mean_scores = np.mean(scores_list[dim], axis=1)
        nb_cv_splits = len(mean_scores)

        mean_scores_list.append(np.mean(mean_scores))
        sem_scores_list.append(np.std(mean_scores) / np.sqrt(nb_cv_splits))

    print(len(mean_scores_list), len(sem_scores_list))

    return mean_scores_list, sem_scores_list

def get_best_dim(dim_list, mean_scores_list, sem_scores_list):

    # Find the lower bound for the best score (max mean score - 1 SEM)
    max_score_idx = np.argmax(mean_scores_list)
    lower_bound = mean_scores_list[max_score_idx] - sem_scores_list[max_score_idx]

    print(max_score_idx, lower_bound)

    # Find smallest dimension with mean score within 1 SEM of the best score
    best_dim = None
    for dim, mean_score in zip(dim_list, mean_scores_list):
        if mean_score >= lower_bound:
            best_dim = dim
            break

    return best_dim

def get_best_dim_max_gain(dim_list, mean_scores_list, sem_scores_list,fraction_threshold=0.95):

    max_score_idx = np.argmax(mean_scores_list)
    min_score_indx = np.argmin(mean_scores_list)

    cv_gain = mean_scores_list[max_score_idx] - mean_scores_list[min_score_indx]

    # Find smallest dimension with mean score that achieves the specified fraction of the maximum CV gain
    for i, dim in enumerate(dim_list):
        fraction_max_gain = (mean_scores_list[i] - mean_scores_list[min_score_indx]) / cv_gain

        if fraction_max_gain >= fraction_threshold:
            print(f"Dimension {dim} achieves {fraction_max_gain:.2%} of the maximum CV gain.")

            best_dim = dim
            break

    best_dim_indx = np.where(dim_list == best_dim)[0][0]

    # Now find the lower bound for the best score
    lower_bound = mean_scores_list[best_dim_indx] - sem_scores_list[best_dim_indx]

    # Find smallest dimension with mean score within 1 SEM of the best score
    for dim, mean_score in zip(dim_list, mean_scores_list):
        if mean_score >= lower_bound:
            print(f"Dimension {dim} is the smallest dimension with mean score within 1 SEM of the best score.")
            best_dim = dim
            break

    return best_dim

def compute_total_counts(spk_rec_trials, time_step, nb_steps, bin_width, nb_trials, nb_neurons, max_time, min_time):

    time = np.arange(nb_steps)*time_step.numpy()
    nb_bins = (max_time - min_time) // bin_width

    total_counts = {'burst_counts':np.zeros((nb_trials, nb_neurons, nb_bins)), 
                    'event_counts':np.zeros((nb_trials, nb_neurons, nb_bins)), 
                    'spike_counts':np.zeros((nb_trials, nb_neurons, nb_bins))}

    for i in range(nb_trials):

        classified_spikes = burst_classification(spk_rec=spk_rec_trials[i].cpu().numpy(), burst_thresh=15, time=time, track_burst_lengths=False)
        counts = compute_single_trial_counts(classified_spikes, min_time, max_time, bin_width)

        for key in total_counts.keys():
            total_counts[key][i] = counts[key]

    return total_counts


def burst_classification(spk_rec, burst_thresh, time, track_burst_lengths=False): 
    
    nb_units, nb_steps = np.shape(spk_rec)
    if track_burst_lengths:
        outputs = {'burst_times': [], 'event_times': [], 'spike_times': [], 'burst_lengths': []}
    else:
        outputs = {'burst_times': [], 'event_times': [], 'spike_times': []}

    for k in range(nb_units):
        spikes = spk_rec[k, :]
        spike_times = time[spikes==1]  # get spike times
        ISIs = np.diff(spike_times)   # compute inter-spike intervals

        burst_times, singlet_times, burst_lengths = [], [], []

        if len(ISIs) > 0:
            i = 0
            current_burst = 0
            burst_length_counter = 0

            while i < len(ISIs):
                if ISIs[i] < burst_thresh and current_burst == 0: # first spike in a burst
                    burst_times.append(spike_times[i])
                    current_burst = 1
                    burst_length_counter = 1

                elif ISIs[i]<burst_thresh and current_burst==1: # remaining spikes in a burst
                    burst_length_counter +=1

                elif ISIs[i] > burst_thresh and current_burst == 0: # singlet event spike
                    singlet_times.append(spike_times[i])

                elif ISIs[i] > burst_thresh and current_burst == 1:
                    current_burst = 0
                    burst_length_counter += 1
                    burst_lengths.append(burst_length_counter)
                    burst_length_counter = 0

                i += 1

            # Handle the last spike
            if ISIs[-1] > burst_thresh:
                singlet_times.append(spike_times[-1])
        else:
            # Single spike or no spikes: assign spike_times accordingly
            if len(spike_times) == 1:
                singlet_times = [spike_times[0]]

        event_times = np.sort(np.hstack((singlet_times, burst_times)))

        outputs['burst_times'].append(burst_times)
        outputs['event_times'].append(event_times)
        outputs['spike_times'].append(spike_times)

        if track_burst_lengths:
            outputs['burst_lengths'].append(burst_lengths)

    return outputs


def burst_statistics(spk_rec, nb_steps, time_step, burst_thresh=15):
    """
    Baseline burst fraction and burst length of a simulated trial (same definitions as the plot_number_spikes_per_burst annotations).

    :param spk_rec: Spike record (nb_units, nb_steps).
    :param nb_steps: Number of integration steps.
    :param time_step: Integration time step (ms).
    :param burst_thresh: ISI threshold (ms) for a burst.
    :return: Dictionary with the mean and SEM of the burst fraction (%, across neurons) and burst length (spikes, across bursts).
    """
    time = np.arange(nb_steps)*np.asarray(time_step)
    classified_spikes = burst_classification(spk_rec=np.asarray(spk_rec), burst_thresh=burst_thresh, time=time, track_burst_lengths=True)

    # Burst fraction per neuron (nan for neurons without events)
    burst_fractions = np.array([len(b) / len(e) if len(e) > 0 else np.nan
                                for b, e in zip(classified_spikes['burst_times'], classified_spikes['event_times'])])
    nb_units_not_nan = np.sum(~np.isnan(burst_fractions))

    # Burst lengths pooled across neurons
    burst_lengths = np.array([y for x in classified_spikes['burst_lengths'] for y in x], dtype=float)
    nb_bursts = len(burst_lengths)

    return {'burst_fraction_mean': np.nanmean(burst_fractions) * 100,
            'burst_fraction_sem': np.nanstd(burst_fractions) / np.sqrt(nb_units_not_nan) * 100,
            'burst_length_mean': np.mean(burst_lengths) if nb_bursts > 0 else np.nan,
            'burst_length_sem': np.std(burst_lengths) / np.sqrt(nb_bursts) if nb_bursts > 0 else np.nan,
            'nb_bursts': nb_bursts}



def compute_single_trial_counts(classified_spikes, min_time, max_time, bin_width):

    nb_units = len(classified_spikes['spike_times'])
    
    counts_dict = {}
    counts_keys = ['burst_counts', 'event_counts', 'spike_counts']

    bins = np.arange(min_time, max_time + bin_width, bin_width)

    for k, key in enumerate(classified_spikes.keys()):
        
        counts_units = []
        for i in range(nb_units):
            spike_times = classified_spikes[key][i]
            counts, _ = np.histogram(spike_times, bins=bins)
            counts_units.append(counts)
    
        # Convert to numpy array
        counts_units = np.asarray(counts_units)

        # Store in the dictionary
        counts_dict[counts_keys[k]] = counts_units
        
    return counts_dict

def align_sign_one_dimensional(y, x, eps=1e-12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    s = float(np.dot(x, y) / (nx * ny))  # cosine similarity in [-1, 1]
    flip = -1.0 if s < 0 else 1.0

    return flip * y

def update_dict(old_dict, new_results):
    for key in new_results.keys():
        for subkey in new_results[key].keys():
            old_dict[key][subkey].append(new_results[key][subkey])
    return old_dict

def update_dict_one_level(old_dict, new_results):
    for key in new_results.keys():
        old_dict[key].append(new_results[key])
    return old_dict

def update_dict_element_one_level(old_dict, new_results, i, j):
    for key in new_results.keys():
        old_dict[key][i, j] = new_results[key]
    return old_dict

def compute_performance_stats(all_results):
    
    all_results_means = {'MFA': {'event_error': [], 'burst_error': [], 'combined_error': []},
                         'FA': {'event_error': [], 'burst_error': [], 'combined_error': []}}
    all_results_stds = {'MFA': {'event_error': [], 'burst_error': [], 'combined_error': []},
                        'FA': {'event_error': [], 'burst_error': [], 'combined_error': []}}

    for model in all_results.keys():
        for metric in all_results[model].keys():
            all_results_means[model][metric] = np.mean(all_results[model][metric])
            all_results_stds[model][metric] = np.std(all_results[model][metric])

    return all_results_means, all_results_stds

def compute_performance_stats_dimensions(repeated_results):

    all_results_means = {'FA_dim_error': [], 'MFA_dim_error': []}
    all_results_stds = {'FA_dim_error': [], 'MFA_dim_error': []}

    for metric in repeated_results.keys():
        all_results_means[metric] = np.mean(repeated_results[metric])
        all_results_stds[metric] = np.std(repeated_results[metric])

    return all_results_means, all_results_stds

def update_dict_element(old_dict, new_results, i, j):
    for key in new_results.keys():
        for subkey in new_results[key].keys():
            old_dict[key][subkey][i, j] = new_results[key][subkey]

    return old_dict

def kfold_indices(nb_samples, nb_splits=5, shuffle=True, drop_remainder=True, seed=18):
    """
    Generate train/validation indices for k-fold cross-validation.
    
    :param nb_samples: Total number of samples in the dataset.
    :param nb_splits: Number of folds for cross-validation.
    :param shuffle: Whether to shuffle the indices before splitting.
    :param drop_remainder: Whether to drop the remainder if n_samples is not divisible by n_splits.
    :return: List of (train_idx, val_idx) tuples for each fold.
    """

    indices = np.arange(nb_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    if drop_remainder:
        usable = (nb_samples // nb_splits) * nb_splits
        indices = indices[:usable]

    # Equal-size folds only (requires len(indices) divisible by n_splits)
    folds = np.split(indices, nb_splits)

    splits = []
    for k in range(nb_splits):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(nb_splits) if i != k])
        splits.append((train_idx, val_idx))

    return splits

def compute_relative_mean_sem_scores(scores_list, dim_list):
    """
    Compute the relative mean and standard error of the mean (SEM) scores for each dimension in dim_list
    compared to the first dimension (which serves as the baseline). This is useful for evaluating the improvement in scores
    relative to the baseline dimension.
    
    :param scores_list: Dictionary where keys are dimensions and values are arrays of scores for each fold.
    :param dim_list: List of dimensions to evaluate, with the first dimension being the baseline.
    """

    
    scores_first_dim = scores_list[dim_list[0]]
    mean_scores_first_dim = np.mean(scores_first_dim, axis=1)

    mean_scores_list, sem_scores_list  = [], []
    for dim in dim_list:

        # Mean over samples within fold
        mean_scores = np.mean(scores_list[dim], axis=1)
        nb_cv_splits = len(mean_scores)
        mean_scores_difference = mean_scores - mean_scores_first_dim

        # Mean/sem over folds
        mean_scores_list.append(np.mean(mean_scores_difference))
        sem_scores_list.append(np.std(mean_scores_difference, ddof=1) / np.sqrt(nb_cv_splits))

    return mean_scores_list, sem_scores_list

def compute_data_statistics(data_2d):
    """
    Compute various statistics for a 2D dataset of neural activity.
    
    :param data_2d: Dictionary where each key maps to a 2D tensor of shape (nb_neurons, nb_samples).
    :return: Dictionary containing computed statistics for each key in the input data.
    """

    statistics = {}
    for key in data_2d.keys():

        x = data_2d[key]
        nb_neurons, nb_samples = x.shape

        # Single neuron statistics
        counts = x.flatten()
        mean_counts = torch.mean(x, dim=1)  # (nb_neurons,)
        fano_factor = torch.var(x, dim=1) / mean_counts
        fano_factor = torch.nan_to_num(fano_factor, nan=0.0, posinf=0.0, neginf=0.0)

        # Spike count correlations
        spike_count_correlations = torch.corrcoef(x)  # (nb_neurons, nb_neurons)

        # Get upper triangular values with diagonal excluded
        spike_count_correlations = spike_count_correlations.numpy()
        triu_indices = np.triu_indices(nb_neurons, k=1)
        spike_count_correlations = spike_count_correlations[triu_indices]

        # Compute the eigenvalues of the covariance matrix
        covariance_matrix = torch.cov(x)
        eigenvalues = torch.linalg.eigvalsh(covariance_matrix)

        statistics[key] =(counts, mean_counts, fano_factor, spike_count_correlations, eigenvalues)

    return statistics

def compute_statistics_single_dataset(x):
    """
    Compute various statistics for a single dataset of neural activity.
    
    :param x: 2D tensor of shape (nb_neurons, nb_samples).
    :return: Tuple containing counts, mean counts, Fano factor, spike count correlations, and eigenvalues.
    """
    # Ensure input is a 2D tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    nb_neurons, nb_samples = x.shape

    # Single neuron statistics
    counts = x.flatten()
    mean_counts = torch.mean(x, dim=1)  # (nb_neurons,)
    fano_factor = torch.var(x, dim=1) / mean_counts
    fano_factor = torch.nan_to_num(fano_factor, nan=0.0, posinf=0.0, neginf=0.0)
    # Spike count correlations
    spike_count_correlations = torch.corrcoef(x)  # (nb_neurons, nb_neurons)

    # Get upper triangular values with diagonal excluded
    spike_count_correlations = spike_count_correlations.numpy()
    triu_indices = np.triu_indices(nb_neurons, k=1)
    spike_count_correlations = spike_count_correlations[triu_indices]

    # Compute the eigenvalues of the covariance matrix
    covariance_matrix = torch.cov(x)
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)

    statistics = (counts.numpy(), mean_counts.numpy(), fano_factor.numpy(), spike_count_correlations, eigenvalues.numpy())

    return statistics
    
def compute_data_statistics_multiple_datasets(data_posterior_samples):
    """
    Compute various statistics for multiple datasets of posterior samples.
    
    :param data_posterior_samples: Dictionary where each key maps to a 3D tensor of shape (nb_neurons, nb_samples, nb_posterior_samples).
    :return: Dictionary containing computed statistics for each key in the input data. Each statistic is stored as a numpy array with shape (nb_posterior_samples, ...).
    """

    nb_neurons, nb_samples, nb_posterior_samples = data_posterior_samples['events'].shape

    statistics = {}
    for key in data_posterior_samples.keys():
        statistics[key] = {"counts": [], "mean_counts": [], "fano_factors": [], "spike_count_correlations": [], "eigenvalues": []}
        for i in range(nb_posterior_samples):

            data_samples_single_dataset = data_posterior_samples[key][:, :, i]  # (nb_neurons, nb_samples)
            statistics_tmp = compute_statistics_single_dataset(data_samples_single_dataset)

            statistics[key]["counts"].append(statistics_tmp[0])
            statistics[key]["mean_counts"].append(statistics_tmp[1])
            statistics[key]["fano_factors"].append(statistics_tmp[2])
            statistics[key]["spike_count_correlations"].append(statistics_tmp[3])
            statistics[key]["eigenvalues"].append(statistics_tmp[4])

    for key in statistics.keys():
        for stat_key in statistics[key].keys():
            statistics[key][stat_key] = np.asarray(statistics[key][stat_key])

    return statistics

def convert_data_for_plot(data_posterior):
    """
    Convert 3D posterior sample data into 2D format suitable for plotting.
    
    :param data_posterior: Dictionary where each key maps to a 3D tensor of shape (nb_neurons, nb_samples, nb_posterior_samples).
    :return: Dictionary where each key maps to a 2D tensor of shape (nb_posterior_samples * nb_samples, nb_neurons).
    """

    data_2d_dict = {}
    for key in data_posterior.keys():
        samples = data_posterior[key]  # (nb_neurons, nb_samples, nb_posterior_samples) or similar

        samples_2d = samples.reshape(samples.shape[0], -1) # (nb_neurons, nb_samples * nb_posterior_samples)

        # If numpy array, convert to tensor
        if isinstance(samples_2d, np.ndarray):
            samples_2d = torch.tensor(samples_2d)

        data_2d_dict[key] = samples_2d.T

    return data_2d_dict

def mean_histograms(data, bins):
    """
    Compute the mean and standard deviation of histograms over posterior samples.
    
    :param data: 2D array of shape (nb_posterior_samples, nb_neurons) containing the data for which histograms are computed.
    :param bins: Array of bin edges to use for the histograms.
    """

    nb_posterior_samples, nb_neurons = data.shape

    histograms = []
    for i in range(nb_posterior_samples):
        hist, _ = np.histogram(data[i, :], bins=bins, density=True)
        histograms.append(hist)

    histograms = np.asarray(histograms)

    mean_histogram = np.mean(histograms, axis=0)
    std_histogram = np.std(histograms, axis=0)

    return mean_histogram, std_histogram
