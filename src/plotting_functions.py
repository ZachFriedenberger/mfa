import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import seaborn as sns
import numpy as np
from misc import burst_classification, bootstrap_confidence_interval, mean_histograms
from utils import load_data, create_folder

from matplotlib.ticker import MaxNLocator


mpl.rcParams.update({
    # --- figure geometry (avoid LaTeX rescaling fonts) ---
    'text.usetex': False, 
    'svg.fonttype': 'none',

    # --- fonts: Times-like serif (fallbacks included) ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],

    # --- font sizes ---
    "font.size": 8,           # base (legend etc.)
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,

    # --- aesthetics that usually read well in papers ---
    "axes.linewidth": 0.8,
    'lines.linewidth': 0.8, 
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})

def plot_score_difference_multiple(scores_multiplexed_model_all, scores_null_model_all, nb_repeats, file_name=None, ylim=None, yticks=None, figsize=(1.0, 1.05)):

    scores_multiplexed_model_all_tmp = np.array(scores_multiplexed_model_all)
    scores_null_model_all_tmp = np.array(scores_null_model_all)

    difference_means = np.mean(scores_multiplexed_model_all_tmp, axis=1) - np.mean(scores_null_model_all_tmp, axis=1)

    mean = np.mean(difference_means)
    std = np.std(difference_means)/np.sqrt(nb_repeats)

    plt.figure(figsize=figsize)

    plt.hlines(0, 0.5, 1.5, colors='k', linestyles='dashed')

    plt.plot(1, mean, marker='.', color='tab:orange')
    plt.errorbar(1, mean, yerr=std, fmt='none', color='tab:orange', capsize=3)

    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
    sns.despine()
    plt.xticks([])
    plt.xlabel("")
    plt.ylabel('Mean score \n difference')

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    plt.show()

    return

def plot_simulated_sensitivity_analysis(output, soma_error_point_estimate, dendrite_error_point_estimate, file_name=None, 
                                        ylim=[5e-3, 1e0], figsize=(1.15, 1.0)):

    plt.figure(figsize=figsize)

    plt.plot(0, soma_error_point_estimate, marker='*', color='gray', label='Theory')
    plt.plot(0, output['mean_distance_event'], marker='.', color='tab:blue', label='Simulation')
    plt.errorbar(0, output['mean_distance_event'], yerr=output['std_distance_event'], fmt='none', color='tab:blue', capsize=3)

    plt.plot(1, dendrite_error_point_estimate, marker='*', color='gray')
    plt.plot(1, output['mean_distance_burst'], marker='.', color='tab:red')
    plt.errorbar(1, output['mean_distance_burst'], yerr=output['std_distance_burst'], fmt='none', color='tab:red', capsize=3)
    sns.despine()

    plt.yscale('log')

    plt.ylim(ylim)
    plt.xlim([-0.5, 1.5])
    plt.xticks([0, 1], ['Soma', 'Dendrite'])
    plt.ylabel('Subspace error')

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    return

def plot_simulated_sensitivity_analysis_base_hybrid(base_data, hybrid_data, file_name=None, soma_ylim=[1e-3, 1e0], dendrite_ylim=[5e-2, 1e0],
                                                    figsize=(1.3, 1.0), color_soma='tab:blue', color_base='tab:orange', color_hybrid='tab:purple'):
    """
    Simulated (mean +/- std over repeats) and theoretical subspace errors, in two side-by-side panels with their own
    y limits: the soma (left) and the dendrite of the base and hybrid models (right). The soma is the same in both
    simulations (same event model), so it is taken from the base-model simulation.

    :param base_data: Saved output of the base-model simulation (simulated_sensitivity_analysis_data_base_model.pkl).
    :param hybrid_data: Saved output of the hybrid-model simulation (simulated_sensitivity_analysis_data_hybrid_model.pkl).
    :param soma_ylim: Y-axis limits of the soma panel.
    :param dendrite_ylim: Y-axis limits of the dendrite panel.
    """
    fig, (ax_soma, ax_dendrite) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1.6], 'wspace': 0.9})

    # axis, x position, theory point estimate, simulated mean, simulated std, color
    points = [(ax_soma, 0.0, base_data['soma_error_point_estimate'], base_data['output']['mean_distance_event'], base_data['output']['std_distance_event'], color_soma),
              (ax_dendrite, -0.2, base_data['dendrite_error_point_estimate'], base_data['output']['mean_distance_burst'], base_data['output']['std_distance_burst'], color_base),
              (ax_dendrite, 0.2, hybrid_data['dendrite_error_point_estimate'], hybrid_data['output']['mean_distance_burst'], hybrid_data['output']['std_distance_burst'], color_hybrid)]

    for ax, x, theory, mean, std, color in points:
        ax.plot(x, theory, marker='*', color='gray')
        ax.plot(x, mean, marker='.', color=color)
        ax.errorbar(x, mean, yerr=std, fmt='none', color=color, capsize=3)

    for ax, ylim, xlim, label in [(ax_soma, soma_ylim, [-0.5, 0.5], 'Soma'), (ax_dendrite, dendrite_ylim, [-0.6, 0.6], 'Dendrite')]:
        ax.set_yscale('log')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xticks([0], [label], rotation=45)
        # A major tick at every decade, with unlabelled minor ticks between
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    sns.despine(fig=fig)
    ax_soma.set_ylabel('Subspace error')

    handles = [Line2D([], [], marker='*', color='gray', ls='none', label='Theory'),
               Line2D([], [], marker='.', color=color_soma, ls='none', label='Soma'),
               Line2D([], [], marker='.', color=color_base, ls='none', label='Base'),
               Line2D([], [], marker='.', color=color_hybrid, ls='none', label='Hybrid')]
    ax_dendrite.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, handletextpad=0.1, borderaxespad=0.2)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return
def plot_subspace_error_estimate(error_grid, K_grid, sigma_grid, dim_estimate, sigma_var_estimate, cmap='Blues_r', figsize=(1.15, 1.0), file_name=None):

    fig, ax = plt.subplots(figsize=figsize)

    error_plot = np.ma.masked_less_equal(error_grid, 0)

    im = ax.pcolormesh(
        K_grid,
        sigma_grid**2,
        error_plot,
        shading='auto',
        cmap=cmap,
        norm=LogNorm(vmin=error_plot.min(), vmax=1.0)
    )

    ax.set_xlabel("Latents")
    ax.set_ylabel("Weight variance")
    ax.set_yscale("log")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Subspace error")

    levels = [1e-1, 1e0]

    cs = ax.contour(
        K_grid,
        sigma_grid**2,   # match pcolormesh
        error_grid,
        levels=levels,
        colors='black',
        linewidths=1.0
    )

    ax.clabel(
        cs,
        levels=levels,
        inline=True,
        fmt={1e-2: r"$10^{-2}$", 1e-1: r"$10^{-1}$", 1e0: r"$10^0$"},
        fontsize=6
    )

    plt.plot(
        dim_estimate,
        sigma_var_estimate,
        marker='*',
        color='gray')

    plt.xticks(K_grid[0, :])

    sns.despine()
    plt.show()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    return

def plot_baseline_score(names, score_1, score_2, score_3, score_4, events_per_sample=None, file_name=None):

    scale = 1e3 if events_per_sample is not None else 1.0

    # Point estimate and 95% CI from the same statistic (per event if events_per_sample is given, else per sample)
    score_means, ci_lower, ci_upper = [], [], []
    for score in [score_1, score_2, score_3, score_4]:
        estimate, lower_bound, upper_bound, p_value = bootstrap_confidence_interval(scale*score, events_per_sample=events_per_sample)
        score_means.append(estimate)
        ci_lower.append(lower_bound)
        ci_upper.append(upper_bound)

    lower_err = [m - l for m, l in zip(score_means, ci_lower)]
    upper_err = [u - m for m, u in zip(score_means, ci_upper)]
    asymmetric_error_bars = [lower_err, upper_err]

    plt.figure(figsize=(2.0, 1.0))
    plt.plot(names, score_means, '.', color='k')
    plt.errorbar(names, score_means, color='none', yerr=asymmetric_error_bars, ecolor='k', capsize=3)
    plt.ylabel('Score (mnats/event)' if events_per_sample is not None else 'Score (nats/sample)')

    sns.despine()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')

    return

def plot_mean_scores(best_dim, dim_list, mean_scores_list, sem_scores_list, color='tab:blue', file_name=None, xlabel=None, xticks=None, figsize=(1.5, 1.0)):
    """
    Plot the mean scores with error bars and highlight the best dimension.
    
    :param best_dim: The best latent dimension to highlight.
    :param dim_list: List of latent dimensions.
    :param mean_scores_list: List of mean scores corresponding to each latent dimension.
    :param sem_scores_list: List of standard error of the mean (SEM) corresponding to each latent dimension.
    :param color: Color for the plot.
    :param file_name: Path to save the resulting plot. If None, the plot is not saved.
    :param xlabel: Label for the x-axis. If None, defaults to 'Latent dimensions'.
    :param xticks: List of x-axis tick positions. If None, defaults to dim_list.
    """

    plt.figure(figsize=figsize)

    plt.plot(dim_list, mean_scores_list, marker='o', color=color, markersize=2, label='Mean score')
    plt.errorbar(dim_list, mean_scores_list, yerr=sem_scores_list, fmt='None', color=color, capsize=2)

    indx = np.where(dim_list == best_dim)[0][0]

    plt.plot(best_dim, mean_scores_list[indx]+2*sem_scores_list[indx],
            marker='*', color='k', label=f'Best dim: {best_dim}')

    sns.despine()
    if xticks is not None:
        plt.xticks(xticks)
    else:
        plt.xticks(dim_list)

    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Latent dimensions')

    plt.ylabel('Mean score \n (nats/sample)')

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    plt.axhline(0, color='k', lw=0.5, ls='--')

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return best_dim


def plot_mean_counts_histogram(total_counts, bins=10, file_name=None):

    plt.figure(figsize=(5.5, 1.5))

    plt.subplot(1, 2, 1)
    plt.hist(np.mean(total_counts['spike_counts'], axis=(0, 2)), bins=bins, density=True, histtype='step', color='k', label='Spikes')
    plt.hist(np.mean(total_counts['burst_counts'], axis=(0, 2)), bins=bins, density=True, histtype='step', color='tab:orange', label='Bursts')
    plt.hist(np.mean(total_counts['event_counts'], axis=(0, 2)), bins=bins, density=True, histtype='step', color='tab:blue', label='Events')

    plt.xlabel('Mean counts')
    plt.ylabel('Density')
    plt.xticks([0, 2.5, 5])
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)

    burst_fraction = np.mean(total_counts['burst_counts'], axis=(0, 2)) / np.mean(total_counts['event_counts'], axis=(0, 2))
    nb_nans = np.isnan(burst_fraction).sum()

    burst_fraction = burst_fraction[~np.isnan(burst_fraction)] # Remove NaNs corresponding to bins with zero spikes

    plt.hist(burst_fraction, bins=bins, density=True, histtype='step', color='tab:red')
    plt.xticks([0, 0.5, 1])
    plt.xlim(0, 1)
    plt.xlabel('Mean burst fraction')

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return


def plot_total_counts_histogram(total_counts, file_name=None):

    bins = np.arange(0, total_counts['spike_counts'].max()+1) 

    plt.figure(figsize=(5.5, 1.5))

    plt.subplot(1, 2, 1)
    plt.hist(total_counts['spike_counts'].flatten(),bins=bins, density=True, histtype='step', color='k', label='Spikes')
    plt.hist(total_counts['burst_counts'].flatten(),bins=bins, density=True, histtype='step', color='tab:orange', label='Bursts')
    plt.hist(total_counts['event_counts'].flatten(),bins=bins, density=True, histtype='step', color='tab:blue', label='Events')

    plt.xlabel('Counts')
    plt.ylabel('Density')
    plt.xticks([0, 6, 12])
    plt.yticks([0, 0.3, 0.6])
    plt.legend(frameon=False)

    plt.subplot(1, 2, 2)

    burst_fraction = total_counts['burst_counts'].flatten() / total_counts['event_counts'].flatten()
    nb_nans = np.isnan(burst_fraction).sum()

    nb_trials, nb_neurons, nb_bins = total_counts['event_counts'].shape

    nan_fraction = nb_nans / (nb_trials * nb_neurons * nb_bins)
    burst_fraction = burst_fraction[~np.isnan(burst_fraction)] # Remove NaNs corresponding to bins with zero spikes

    bins = np.linspace(0, 1, 16) - 0.5/15
    plt.hist(burst_fraction, bins=bins, density=True, histtype='step', color='tab:red')

    plt.xlabel('Burst fraction')
    plt.xticks([0, 0.5, 1])
    plt.xlim(0, 1)

    # text box with the percentage of bins with zero spikes
    plt.text(0.5, 0.5, 'Zero event fraction: {:.2f}'.format(nan_fraction), transform=plt.gca().transAxes, ha='center', va='center')    
    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return


def plot_example_trial(plot_params, path_to_save):

    results_dir = f"results/{path_to_save}/plots/"
    create_folder(results_dir)
    create_folder(results_dir + 'voltage_traces/')

    # Load the data for the example trial
    spiking_data = load_data(f"results/{path_to_save}/spiking_data.pkl")
    metadata = load_data(f"results/{path_to_save}/metadata.pkl")

    # Unpack data and parameters
    mem_s_rec = spiking_data['mem_s']
    mem_d_rec = spiking_data['mem_d']
    spk_rec = spiking_data['spk']

    input_params_spiking = metadata['input_params_spiking']
    nb_steps_integration = input_params_spiking['nb_steps_integration']
    time_step_integration = input_params_spiking['time_step_integration']

    weight_params = metadata['weight_params']
    nb_neurons = weight_params['nb_neurons']

    # Time vector for plotting
    min_time = plot_params['min_time']
    max_time = plot_params['max_time']
    time = np.arange(nb_steps_integration)*time_step_integration.numpy()

    fig_size = plot_params['fig_size']

    # Plot voltage traces for the first 100 neurons
    for unit_num in range(nb_neurons):
        plot_voltage_trace(mem_d=mem_d_rec, mem_s=mem_s_rec, time=time, min_time=min_time,
                            max_time=max_time, unit_num=unit_num, spk_rec=spk_rec, fig_size=fig_size, file_name=results_dir + 'voltage_traces/voltage_traces_unit_{}.svg'.format(unit_num))

        plt.close('all')

    # Plot raster plot for first 100 neurons
    plot_raster(spk_rec, nb_steps_integration, time_step_integration, min_time=min_time, max_time=max_time, fig_size=fig_size, file_name=results_dir + 'raster_plot.svg')
    plt.close('all')

    # Plot the number of spikes per burst histogram
    plot_number_spikes_per_burst(spk_rec, nb_steps_integration, time_step_integration, nb_neurons, fig_size=fig_size, file_name=results_dir + 'number_spikes_per_burst_histogram.svg')
    plt.close('all')

    return


def plot_voltage_trace(mem_d, mem_s, time, min_time, max_time, unit_num, spk_rec, spike_height=50, fig_size=(1.0, 1.0), file_name=None):

    time_mask = np.logical_and(time>=min_time, time<=max_time)
    time_plot = (time[time_mask] - min_time)/1e3 # convert to seconds

    mem_s = mem_s[unit_num, time_mask]
    mem_d = mem_d[unit_num, time_mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=fig_size)

    # Dendritic membrane potential
    mem_d = mem_d.detach().cpu().numpy()
    ax1.plot(time_plot, mem_d, color='k', label=r'$u_{d}$')
    ax1.set_xticklabels([])
    ax1.set_yticks([-75, -25])

    # Somatic membrane potential
    classified_spikes = burst_classification(spk_rec=spk_rec.cpu().numpy(), burst_thresh=15, time=time)
    burst_times = classified_spikes['burst_times'][unit_num]
    spk = spk_rec[unit_num, time_mask].cpu().numpy()

    dat = 1.0*mem_s
    dat[spk>0.0] += spike_height
    dat = dat.detach().cpu().numpy()

    for burst_time in burst_times: 
        if burst_time >= min_time and burst_time <= max_time:
            ax2.plot((burst_time-min_time)/1e3, 10, '.', ms=3, color='tab:orange')
    

    ax2.plot(time_plot, dat, color='k', label=r'$u_{s}$')
    ax2.set_ylabel(r'Potential (mV)')
    ax2.set_xlabel('Time (s)')
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels([0, 0.5, 1])

    sns.despine()
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.subplots_adjust(hspace=0.5)  # Adjust spacing

    if file_name is not None:
        plt.savefig(file_name)

    return

def plot_inputs(I_s, I_d, time, fig_params, file_name=None):
    
    figsize = fig_params['figsize']
    y_ticks_somatic = fig_params['y_ticks_somatic']
    y_lim_somatic = fig_params['y_lim_somatic']
    y_ticks_dendrite = fig_params['y_ticks_dendrite']
    y_lim_dendrite = fig_params['y_lim_dendrite']


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    
    # Plot the somatic input
    ax2.plot(time, I_s[0,:], color='tab:blue', label=r'$I_{s}$')
    ax2.set_xticks([0, 250, 500])
    ax2.set_ylabel(r'Inputs (pA)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_yticks(y_ticks_somatic)
    ax2.set_ylim(y_lim_somatic)

    # Plot the dendritic input
    ax1.plot(time, I_d[0,:], color='tab:red', label=r'$I_{d}$')
    ax1.set_xticks([0, 250, 500])
    ax1.set_xticklabels([])
    ax1.set_yticks(y_ticks_dendrite)
    ax1.set_ylim(y_lim_dendrite)
    
    sns.despine()
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.subplots_adjust(hspace=0.5)  # Adjust spacing

    if file_name is not None:
        plt.savefig(file_name)

    return

def plot_raster(spk_rec, nb_steps, time_step, min_time, max_time, fig_size=(1.0, 1.0), file_name=None):

    time = np.arange(nb_steps)*time_step.numpy()
    classified_spikes = burst_classification(spk_rec=spk_rec.cpu().numpy(), burst_thresh=15, time=time)

    # classified spikes are sorted by neuron, then trial. We need to reverse this for plotting
    nb_units = len(classified_spikes['spike_times'])

    spike_times = [] # list of lists for trial trial_num
    burst_times = []
    for i in range(nb_units):

        spike_times_centered = (np.asarray(classified_spikes['spike_times'][i]) - min_time)/1e3 # convert to seconds and center around min_time for plotting
        burst_times_centered = (np.asarray(classified_spikes['burst_times'][i]) - min_time)/1e3
        spike_times.append(list(spike_times_centered))
        burst_times.append(list(burst_times_centered))

    plt.figure(figsize=fig_size)
    ax = plt.gca()

    ax.eventplot(spike_times, color='k', linewidths=0.8)
    ax.eventplot(burst_times, color='tab:orange', linewidths=0.8)

    ax.set_ylabel('Neuron')
    ax.set_xlabel('Time (s)')
    
    ax.set_yticks([0, 25, 50])
    ax.set_ylim([0, 50])
    ax.set_xlim([0, (max_time-min_time)/1e3])
    ax.set_xticks([0, 0.5, 1]) # Only plot 1 second for clarity

    sns.despine()

    if file_name is not None:
        plt.savefig(file_name)

    return

def plot_number_spikes_per_burst(spk_rec, nb_steps, time_step, nb_units, fig_size=(1.0, 1.0), file_name=None):

    # Classify the spikes
    time = np.arange(nb_steps)*time_step.numpy()
    classified_spikes = burst_classification(spk_rec=spk_rec.cpu().numpy(), burst_thresh=15, time=time, track_burst_lengths=True)

    # Flatten the burst lengths array for the histogram
    burst_lengths = classified_spikes['burst_lengths']
    flat_burst_lengths = [y for x in burst_lengths for y in x]

    # Compute the baseline burst fraction
    burst_fractions = []
    for i in range(nb_units):
        nb_events = len(classified_spikes['event_times'][i])
        nb_bursts = len(classified_spikes['burst_times'][i])
        burst_fractions.append(nb_bursts / nb_events) if nb_events > 0 else burst_fractions.append(np.nan)


    nb_units_not_nan = np.sum(~np.isnan(burst_fractions))
    mean_burst_fraction = (np.nanmean(burst_fractions) * 100)
    sem_burst_fraction = (np.nanstd(burst_fractions) / np.sqrt(nb_units_not_nan)) * 100

    mean_burst_length = np.mean(flat_burst_lengths)
    sem_burst_length = np.std(flat_burst_lengths) / np.sqrt(len(flat_burst_lengths))

    # Plot the histogram
    plt.figure(figsize=fig_size)
    bins = np.arange(2, max(flat_burst_lengths)+2) - 0.5 # Bin edges for integer burst lengths
    plt.hist(flat_burst_lengths, bins=bins, density=True, color='tab:orange', histtype='step', fill=False)
    plt.vlines(np.mean(flat_burst_lengths), ymin=0, ymax=0.25, color='k', linestyle='--')

    plt.xlabel('Burst Length')
    plt.ylabel('Density')
    sns.despine()
    plt.xlim([1, 15])
    plt.xticks([0, 5, 10, 15])

    ax = plt.gca()
    ax.text(0.99, 0.99, "$F_{0}=$" + str(np.round(mean_burst_fraction, 3)) + "$\\pm$" + str(np.round(sem_burst_fraction, 3)), transform=ax.transAxes, ha="right", va="top")
    ax.text(0.99, 0.85, "$L_{0}=$" + str(np.round(mean_burst_length, 3)) + "$\\pm$" + str(np.round(sem_burst_length, 3)), transform=ax.transAxes, ha="right", va="top")

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return

def plot_error_difference_heatmap(all_results_means, figsize=(1.75, 2.0), file_name=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    burst_error_diff = all_results_means['FA']['burst_error'] - all_results_means['MFA']['burst_error']
    event_error_diff  = all_results_means['FA']['event_error'] - all_results_means['MFA']['event_error']

    burst_cmap = mpl.cm.get_cmap("Reds")  
    event_cmap = mpl.cm.get_cmap("Blues")

    # white at 0 by construction for diverging maps like RdBu_r
    im1 = ax1.imshow(burst_error_diff, origin='lower', aspect='auto', extent=[0, 90, 0.1, 0.5], vmin=0, vmax=1.0, cmap=burst_cmap)
    im2 = ax2.imshow(event_error_diff, origin='lower', aspect='auto', extent=[0, 90, 0.1, 0.5], vmin=0, vmax=0.2, cmap=event_cmap)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.2, pad=0.02)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.2, pad=0.02)
    cbar2.set_label('Subspace error difference')

    ax2.set_ylabel('Burst power')
    ax2.set_xlabel('Angle (deg.)')
    ax1.set_xticks([0, 45, 90]); ax1.set_xticklabels([])
    ax2.set_xticks([0, 45, 90]); ax2.set_xticklabels([0, 45, 90])

    ax2.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4])

    sns.despine()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return


def plot_dimension_estimation_heatmap(all_results_means, x_vals, y_vals, figsize=(1.75, 2.0), file_name=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    FA_dim_error = all_results_means['FA_dim_error']  # (Ny, Nx)
    MFA_dim_error = all_results_means['MFA_dim_error']  # (Ny, Nx)

  
    x_vals = np.asarray(x_vals)*90 # Angles in degrees

    x_edges = centers_to_edges(x_vals)  # len Nx+1
    y_edges = centers_to_edges(y_vals)  # len Ny+1

    im1 = ax1.pcolormesh(x_edges, y_edges, FA_dim_error, shading="flat",
                         vmin=0.0, vmax=0.5, cmap="Purples")
    im2 = ax2.pcolormesh(x_edges, y_edges, MFA_dim_error, shading="flat",
                         vmin=0.0, vmax=0.5, cmap="Purples")

    fig.colorbar(im1, ax=ax1, fraction=0.2, pad=0.02)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.2, pad=0.02)
    cbar2.set_label("Dimension error")

    ax2.set_ylabel("Burst power")
    ax2.set_xlabel("Angle (deg.)")

    # ticks can now be the actual values
    ax2.set_xticks([0, 45, 90])
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax2.set_yticks([0.1, 0.2, 0.3, 0.4])

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)


    return

def centers_to_edges(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("v must be 1D with length >= 2")
    dv = np.diff(v)
    if not np.all(dv > 0):
        raise ValueError("v must be strictly increasing")
    edges = np.empty(v.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (v[:-1] + v[1:])
    edges[0] = v[0] - 0.5 * dv[0]
    edges[-1] = v[-1] + 0.5 * dv[-1]
    return edges

def plot_error_heatmap(all_results_means, model_key, x_vals, y_vals, figsize=(1.75, 2.0), file_name=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    burst_errors = all_results_means[model_key]['burst_error']  # (Ny, Nx)
    event_errors = all_results_means[model_key]['event_error']  # (Ny, Nx)

    x_vals = np.asarray(x_vals)*90 # Angles in degrees

    x_edges = centers_to_edges(x_vals)  # len Nx+1
    y_edges = centers_to_edges(y_vals)  # len Ny+1

    im1 = ax1.pcolormesh(x_edges, y_edges, burst_errors, shading="flat",
                         vmin=0, vmax=1.0, cmap="Reds")
    im2 = ax2.pcolormesh(x_edges, y_edges, event_errors, shading="flat",
                         vmin=0, vmax=0.05, cmap="Blues")

    fig.colorbar(im1, ax=ax1, fraction=0.2, pad=0.02)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.2, pad=0.02)
    cbar2.set_label("Subspace error")

    ax2.set_ylabel("Burst power")
    ax2.set_xlabel("Angle (deg.)")

    # ticks can now be the actual values
    ax2.set_xticks([0, 45, 90])
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax2.set_yticks([0.1, 0.2, 0.3, 0.4])

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)


    return

def plot_error_vs_angle(all_results_means, all_results_stds, norm_grassmann_distances, nb_repeats, model_key, row, file_name=None):

    plt.figure(figsize=(1.5, 1.0))

    angles = np.array(norm_grassmann_distances) * 90

    burst_error_means = all_results_means[model_key]['burst_error'][row, :]
    burst_error_sems = all_results_stds[model_key]['burst_error'][row, :] / np.sqrt(nb_repeats)

    event_error_means = all_results_means[model_key]['event_error'][row, :]
    event_error_sems = all_results_stds[model_key]['event_error'][row, :] / np.sqrt(nb_repeats)

    plt.plot(angles, burst_error_means, color='tab:red', label='Dendritic')
    plt.fill_between(angles, burst_error_means - burst_error_sems,
                    burst_error_means + burst_error_sems, color='tab:red', alpha=0.3, edgecolor="none")


    plt.plot(angles, event_error_means, color='tab:blue', label='Somatic')
    plt.fill_between(angles, event_error_means - event_error_sems,
                    event_error_means + event_error_sems, color='tab:blue', alpha=0.3, edgecolor="none")

    plt.xlabel('Angle (deg.)')
    plt.ylabel('Subspace error')
    plt.legend(frameon=False, fontsize=6)
    sns.despine()

    plt.yticks([0, 0.5, 1])
    plt.xticks([0, 45, 90])
    plt.ylim([0, 1.1])
    plt.xlim([-2, 90])

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return

def plot_latent_trajectories(z_event_est, z_event_true, z_burst_est, z_burst_true, file_name=None):

    plt.figure(figsize=(2.0, 2.0))

    plt.subplot(2,1,1)
    plt.plot(z_burst_est, color='tab:red', label='Inferred')
    plt.plot(z_burst_true, color='k', label='True')

    plt.xticks([0, 250, 500], [])
    plt.yticks([-3, 0, 3])
    plt.ylim([-4, 4])
    plt.ylabel('Dendrite')
    plt.legend(frameon=False, fontsize=6)

    plt.subplot(2,1,2)

    plt.plot(z_event_est, color='tab:blue', label='Inferred')
    plt.plot(z_event_true, color='k', label='True')

    plt.xticks([0, 250, 500])
    plt.yticks([-3, 0, 3])
    plt.ylim([-4, 4])
    plt.legend(frameon=False, fontsize=6)

    plt.xlabel('Time (a.u.)')
    plt.ylabel('Soma')

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return

def plot_expected_counts_distributions(event_rate, burst_rate, spike_rate, file_name=None):

    plt.figure(figsize=(3, 1.25))

    plt.subplot(1, 2, 1)

    events_falttened = event_rate.flatten()
    events_falttened = events_falttened[events_falttened<30] # Remove outliers for better visualization
    plt.hist(events_falttened, bins=50, density=True, color='tab:blue', histtype='step', label=r'$E$')

    bursts_falttened = burst_rate.flatten()
    bursts_falttened = bursts_falttened[bursts_falttened<30] # Remove outliers for better visualization
    plt.hist(bursts_falttened, bins=50, density=True, color='tab:orange', histtype='step', label=r'$B$')

    spikes_falttened = spike_rate.flatten()
    spikes_falttened = spikes_falttened[spikes_falttened<30] # Remove outliers for better visualization
    plt.hist(spikes_falttened, bins=50, density=True, color='k', histtype='step', label=r'$A$')

    plt.xticks([0, 15, 30])
    plt.yticks([0, 0.4, 0.8])
    plt.ylabel('Density')
    plt.xlabel(r'Expected counts, $\lambda\Delta t$')
    plt.legend(frameon=False)


    plt.subplot(1, 2, 2)

    burst_fraction_flattened = burst_rate.flatten() / event_rate.flatten()
    burst_fraction_flattened = np.nan_to_num(burst_fraction_flattened, nan=0.0)
    plt.hist(burst_fraction_flattened, bins=50, density=True, color='tab:red', histtype='step', label=r'$\rho$')

    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 6, 12])
    plt.xlabel(r'Burst probability, $\rho$')

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return

def plot_observed_counts_distributions(events, bursts, spikes, file_name=None):

    plt.figure(figsize=(3, 1.25))

    plt.subplot(1, 2, 1)

    bins = np.arange(0, 51, 1)

    events_falttened = events.flatten()
    events_falttened = events_falttened[events_falttened<=50] # Remove outliers for better visualization
    plt.hist(events_falttened, bins=bins, density=True, color='tab:blue', histtype='step', label=r'$E$')

    bursts_falttened = bursts.flatten()
    bursts_falttened = bursts_falttened[bursts_falttened<=50] # Remove outliers for better visualization
    plt.hist(bursts_falttened, bins=bins, density=True, color='tab:orange', histtype='step', label=r'$B$')

    spikes_falttened = spikes.flatten()
    spikes_falttened = spikes_falttened[spikes_falttened<=50] # Remove outliers for better visualization
    plt.hist(spikes_falttened, bins=bins, density=True, color='k', histtype='step', label=r'$A$')

    plt.xticks([0, 25, 50])
    plt.yticks([0, 0.1, 0.2])
    plt.ylabel('Density')
    plt.xlabel('Counts')
    plt.legend(frameon=False)


    plt.subplot(1, 2, 2)

    burst_fraction_flattened = bursts.flatten() / events.flatten()
    burst_fraction_flattened = np.nan_to_num(burst_fraction_flattened, nan=0.0)
    plt.hist(burst_fraction_flattened, bins=50, density=True, color='tab:red', histtype='step', label=r'$F$')

    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 3, 6])
    plt.xlabel(r'Burst fraction, $F$')

    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return


# Training dynamics plotting functions
def plot_training_dynamics(fit_results, data_train, data_test, results_path):

    """
    Plot the training dynamics, including the ELBO for different models.
    
    :param fit_results: Dictionary containing the results of model fitting, including training and test statistics for different models.
    :param data_train: Dictionary containing the training data, including event and burst information.
    :param data_test: Dictionary containing the test data, including event and burst information.
    :param results_path: Path to the directory where the resulting plots will be saved.
    """

    nb_samples_train, nb_neurons = data_train['events'].shape
    nb_samples_test, _ = data_test['events'].shape

    # Plot the ELBO for the burst base model
    plot_elbo(fit_results['multiplexed_stats_train']['loss'], fit_results['multiplexed_stats_train']['kl_divergence_scaled'], 
           nb_neurons=nb_neurons, nb_samples=nb_samples_train, file_name=f"{results_path}/elbo_multiplexed_train.svg")

    plot_elbo(fit_results['multiplexed_stats_test']['loss'], fit_results['multiplexed_stats_test']['kl_divergence_scaled'],
            nb_neurons=nb_neurons, nb_samples=nb_samples_test, file_name=f"{results_path}/elbo_multiplexed_test.svg")

    # Plot the ELBO for the null model
    plot_elbo(fit_results['null_stats_train']['loss'], fit_results['null_stats_train']['kl_divergence_scaled'],
                nb_neurons=nb_neurons, nb_samples=nb_samples_train, file_name=f"{results_path}/elbo_null_train.svg")

    plot_elbo(fit_results['null_stats_test']['loss'], fit_results['null_stats_test']['kl_divergence_scaled'],
                nb_neurons=nb_neurons, nb_samples=nb_samples_test, file_name=f"{results_path}/elbo_null_test.svg")

    # Plot the ELBO for the burst hybrid model
    plot_elbo(fit_results['hybrid_stats_train']['loss'], fit_results['hybrid_stats_train']['kl_divergence_scaled'], 
           nb_neurons=nb_neurons, nb_samples=nb_samples_train, file_name=f"{results_path}/elbo_hybrid_train.svg")

    plot_elbo(fit_results['hybrid_stats_test']['loss'], fit_results['hybrid_stats_test']['kl_divergence_scaled'],
            nb_neurons=nb_neurons, nb_samples=nb_samples_test, file_name=f"{results_path}/elbo_hybrid_test.svg")
    
    return

def plot_elbo(elbos, kl_divergences, nb_samples, nb_neurons, file_name=None):
    """
    Plot the elbo and KL divergence over iterations.
    
    :param elbos: List or array of ELBO values over iterations.
    :param kl_divergences: List or array of KL divergence values over iterations.
    :param nb_samples: Number of samples used in the estimation.
    :param nb_neurons: Number of neurons in the model.
    :param file_name: Path to save the plot. If None, the plot is not saved.
    """

    # Plot the ELBO
    plt.figure(figsize=(5.5, 2))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(elbos)), np.array(elbos)/ (nb_samples*nb_neurons))
    plt.xscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    sns.despine()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(kl_divergences)), np.array(kl_divergences)/(nb_samples))
    plt.xscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('KL divergence')
    sns.despine()
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)
        
    return


def format_p_value(p_value, n_bootstrap=10000):
    """Format a bootstrap p-value for plotting."""
    return f"p<{1.0 / n_bootstrap:g}" if p_value <= 1.0 / n_bootstrap else f"p={p_value:.2g}"


def plot_score_difference(scores_multiplexed, scores_null, scores_hybrid, events_per_sample=None, file_name=None, ylim=None, yticks=None,
                          ax_size=(0.5, 1.0 * 0.77)):
    """
    Plot the score differences between the base model vs null, and the hybrid model vs null, with 95% bootstrap CIs
    and two-sided bootstrap p-values.

    :param scores_multiplexed: Array of scores for the multiplexed (base) model.
    :param scores_null: Array of scores for the null model.
    :param scores_hybrid: Array of scores for the hybrid model.
    :param events_per_sample: Optional total event counts per test sample. If given, plots mnats/event instead of nats/sample.
    :param file_name: File name to save the plot.
    :param ylim: Y-axis limits for the plot.
    :param yticks: Y-axis tick values for the plot.
    :param ax_size: Axes (width, height) in inches; height 0.77 matches the rasters and burst-length histograms.
    """

    # Per-event differences are reported in mnats/event
    scale = 1e3 if events_per_sample is not None else 1.0

    results = []
    for scores in [scores_multiplexed, scores_hybrid]:
        estimate, lower, upper, p_value = bootstrap_confidence_interval(scores - scores_null, events_per_sample=events_per_sample)
        results.append((scale * estimate, scale * lower, scale * upper, p_value))

    margins = (0.6, 0.45, 0.1, 0.15)   # left, bottom, right, top (inches); trimmed by bbox_inches='tight'
    fig = plt.figure(figsize=(ax_size[0] + margins[0] + margins[2], ax_size[1] + margins[1] + margins[3]))
    fig.add_axes([margins[0] / fig.get_figwidth(), margins[1] / fig.get_figheight(),
                  ax_size[0] / fig.get_figwidth(), ax_size[1] / fig.get_figheight()])

    plt.hlines(0, 0.5, 2.5, colors='k', linestyles='dashed')

    for x, (estimate, lower, upper, p_value) in zip([1, 2], results):
        plt.plot(x, estimate, '.', color='tab:orange')
        plt.errorbar([x], [estimate], yerr=[[estimate - lower], [upper - estimate]], fmt='none', ecolor='tab:orange', capsize=3)
        plt.text(x, upper, format_p_value(p_value), ha='center', va='bottom', fontsize=5)
    plt.xticks([1, 2], ["Base", "Hybrid"], rotation=45)

    if ylim is not None:
        plt.ylim(ylim)
    elif events_per_sample is None:
        plt.ylim([-4, 8])
    if yticks is not None:
        plt.yticks(yticks)
    elif events_per_sample is None:
        plt.yticks([-4, 0, 4, 8])

    sns.despine()
    plt.xlabel("")
    plt.ylabel('Score difference \n (mnats/event)' if events_per_sample is not None else 'Score \n difference')

    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    return results

def plot_summary_statistics(statistics_data, statistics_multiplexed, statistics_null, color_data='tab:blue', file_name=None):
    """
    Plot the summary statistics for the test data, multiplexed model, and null model.
    
    :param statistics_data: Real data statistics as a tuple (counts, mean counts, Fano factors, spike count correlations, eigenvalues).
    :param statistics_multiplexed: Multiplexed model statistics as a dictionary with keys 'counts', 'mean_counts', 'fano_factors', 'spike_count_correlations', 'eigenvalues'.
    :param statistics_null: Null model statistics as a dictionary with keys 'counts', 'mean_counts', 'fano_factors', 'spike_count_correlations', 'eigenvalues'.
    :param color_data: Color to use for the real data plots.
    :param file_name: File name to save the plot. If None, the plot is not saved.
    """


    counts_data, mean_counts_data, fano_factors_data, spike_count_correlations_data, _ = statistics_data

    plt.figure(figsize=(5.5, 1.5))

    # Count distribution
    plt.subplot(1, 4, 1)

    max_counts = np.max(counts_data.numpy())
    bins = np.arange(0, max_counts+1) - 0.5
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.hist(counts_data.numpy(), bins=bins, density=True, label='Test data', histtype='step', color=color_data, zorder=2)

    mean_histogram, std_histogram = mean_histograms(statistics_multiplexed['counts'], bins)
    plt.stairs(mean_histogram, bins, label='MFA', color='k', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='k', alpha=0.2, zorder=1)

    mean_histogram, std_histogram = mean_histograms(statistics_null['counts'], bins)
    plt.stairs(mean_histogram, bins, label='Null', color='gray', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='gray', alpha=0.2, zorder=1)

    plt.xlim([0, 10])

    sns.despine()
    plt.xlabel('Counts')
    plt.ylabel('Density')
    plt.legend(frameon=False)

    plt.subplot(1, 4, 2)

    bins = np.linspace(0, mean_counts_data.max().item() + 0.5, 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.hist(mean_counts_data.numpy(), bins=bins, density=True,  histtype='step', color=color_data, zorder=2)

    mean_histogram, std_histogram = mean_histograms(statistics_multiplexed['mean_counts'], bins)
    plt.stairs(mean_histogram, bins, label='MFA', color='k', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='k', alpha=0.2, zorder=1)

    mean_histogram, std_histogram = mean_histograms(statistics_null['mean_counts'], bins)
    plt.stairs(mean_histogram, bins, label='Null', color='gray', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram -  std_histogram, mean_histogram + std_histogram, step='mid', color='gray', alpha=0.2, zorder=1)
 
    sns.despine()
    plt.xlabel('Mean counts')

    plt.subplot(1, 4, 3)

    bins = np.linspace(0, fano_factors_data.max().item() + 0.5, 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.hist(fano_factors_data.numpy(), bins=bins, density=True,  histtype='step', color=color_data, zorder=2)

    mean_histogram, std_histogram = mean_histograms(statistics_multiplexed['fano_factors'], bins)
    plt.stairs(mean_histogram, bins, label='MFA', color='k', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='k', alpha=0.2, zorder=1)

    mean_histogram, std_histogram = mean_histograms(statistics_null['fano_factors'], bins)
    plt.stairs(mean_histogram, bins, label='Null', color='gray', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram -  std_histogram, mean_histogram + std_histogram, step='mid', color='gray', alpha=0.2, zorder=1)

    sns.despine()
    plt.xlabel('Fano factor')

    plt.subplot(1, 4, 4)

    bins = np.linspace(-1, 1, 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.hist(spike_count_correlations_data, bins=bins, density=True, histtype='step', color=color_data, zorder=2)

    mean_histogram, std_histogram = mean_histograms(statistics_multiplexed['spike_count_correlations'], bins)
    plt.stairs(mean_histogram, bins, label='MFA', color='k', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='k', alpha=0.2, zorder=1)

    mean_histogram, std_histogram = mean_histograms(statistics_null['spike_count_correlations'], bins)
    plt.stairs(mean_histogram, bins, label='Null', color='gray', linestyle='-', zorder=3)
    plt.fill_between(bin_centers, mean_histogram - std_histogram, mean_histogram + std_histogram, step='mid', color='gray', alpha=0.2, zorder=1)
        
    plt.xlabel('Pairwise \n noise correlation')
    sns.despine()
    plt.xlim([-0.75, 0.75])

    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return



# Axes box (inches) of the example-trial raster and burst length plots: fig_size (1.15, 1.0) with matplotlib's default subplot margins
EXAMPLE_TRIAL_AX_SIZE = (1.15 * 0.775, 1.0 * 0.77)

def fixed_size_axes(ax_size=EXAMPLE_TRIAL_AX_SIZE, margins=(0.6, 0.45, 0.1, 0.1)):
    """
    Create a figure whose axes box has a fixed size in inches, so plots with different tick labels have identical axes.
    Save with bbox_inches='tight' so the labels outside the box are included.

    :param ax_size: (width, height) of the axes box in inches.
    :param margins: (left, bottom, right, top) margins in inches around the axes box.
    :return: The axes.
    """
    left, bottom, right, top = margins
    fig_width, fig_height = left + ax_size[0] + right, bottom + ax_size[1] + top
    fig = plt.figure(figsize=(fig_width, fig_height))
    return fig.add_axes([left / fig_width, bottom / fig_height, ax_size[0] / fig_width, ax_size[1] / fig_height])

# Plot the score difference vs. fraction of intact neurons
def plot_score_difference_vs_fraction(fractions, results_base, results_hybrid, color_base='tab:orange', color_hybrid='tab:purple',
                                      offset=0.012, legend_loc='upper left', file_name=None, ylim=None, yticks=None, ax_size=EXAMPLE_TRIAL_AX_SIZE):
    """
    Plot the per-event score difference vs. null (95% bootstrap CIs) of the base and hybrid models against the fraction of intact neurons.

    :param fractions: Fractions of intact neurons.
    :param results_base: List of (estimate, lower, upper, p_value) tuples for base - null, one per fraction.
    :param results_hybrid: List of (estimate, lower, upper, p_value) tuples for hybrid - null, one per fraction.
    :param color_base: Color of the base model.
    :param color_hybrid: Color of the hybrid model.
    :param offset: Horizontal offset between the two models so their error bars do not overlap.
    :param legend_loc: Legend location.
    :param file_name: File name to save the plot.
    :param ylim: Y-axis limits for the plot.
    :param yticks: Y-axis tick values for the plot.
    :param ax_size: (width, height) of the axes box in inches (same default as plot_dims_vs_fraction).
    """

    order = np.argsort(fractions)
    fractions = np.asarray(fractions, dtype=float)[order]

    fixed_size_axes(ax_size)
    plt.axhline(0, color='k', linestyle='dashed', linewidth=0.75)

    for results, color, label, dx in [(results_base, color_base, 'Base', -offset), (results_hybrid, color_hybrid, 'Hybrid', offset)]:
        estimate, lower, upper, _ = (np.array(v, dtype=float) for v in zip(*[results[i] for i in order]))
        plt.plot(fractions + dx, estimate, '-', color=color, linewidth=0.75)
        plt.errorbar(fractions + dx, estimate, yerr=[estimate - lower, upper - estimate], fmt='.', color=color, capsize=2, label=label)

    plt.xlim([-0.05, 1.05])
    plt.xticks([0, 0.5, 1])
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)

    sns.despine()
    plt.legend(frameon=False, fontsize=6, handlelength=1, handletextpad=0.3, borderaxespad=0, loc=legend_loc)
    plt.xlabel('Fraction intact')
    plt.ylabel('Score difference \n (mnats/event)')

    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    return

def plot_dims_vs_fraction(fractions, dims, colors, ylabel, offset=0.012, legend_loc='lower right', file_name=None, ylim=None, ax_size=EXAMPLE_TRIAL_AX_SIZE):
    """
    Plot the number of latent dimensions selected by cross-validation against the fraction of intact neurons.

    :param fractions: Fractions of intact neurons.
    :param dims: Dictionary {label: list of selected dimensions, one per fraction}.
    :param colors: Dictionary {label: color}.
    :param ylabel: Y-axis label.
    :param offset: Horizontal offset between series so overlapping points stay visible.
    :param legend_loc: Legend location.
    :param file_name: File name to save the plot.
    :param ylim: Y-axis limits for the plot.
    :param ax_size: (width, height) of the axes box in inches (same default as plot_score_difference_vs_fraction).
    """

    order = np.argsort(fractions)
    fractions = np.asarray(fractions, dtype=float)[order]
    shifts = offset * (np.arange(len(dims)) - (len(dims) - 1) / 2)

    fixed_size_axes(ax_size)
    for (label, values), dx in zip(dims.items(), shifts):
        values = np.asarray(values, dtype=float)[order]
        plt.plot(fractions + dx, values, '.-', color=colors[label], linewidth=0.8, label=label)

    plt.xlim([-0.05, 1.05])
    plt.xticks([0, 0.5, 1])
    if ylim is not None:
        plt.ylim(ylim)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    sns.despine()
    if len(dims) > 1:
        plt.legend(frameon=False, fontsize=6, handletextpad=0.2, loc=legend_loc)
    plt.xlabel('Fraction intact')
    plt.ylabel(ylabel)

    if file_name is not None:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    return

def plot_burst_statistics_vs_fraction(fractions, burst_fraction, burst_length, color='k', gap=0.08,
                                      yticks_fraction=(5, 20, 35), yticks_length=(2, 3, 4),
                                      file_name=None, ax_size=EXAMPLE_TRIAL_AX_SIZE, margins=(0.6, 0.45, 0.1, 0.1)):
    """
    Plot the baseline burst fraction (top) and mean burst length (bottom) against the fraction of intact neurons.
    The two panels share the x axis and together fill the same axes box as plot_dims_vs_fraction.

    :param fractions: Fractions of intact neurons.
    :param burst_fraction: List of (mean, sem) burst fractions in %, one per fraction.
    :param burst_length: List of (mean, sem) burst lengths in spikes, one per fraction.
    :param color: Color of the points.
    :param yticks_fraction: Y-axis ticks of the burst fraction panel.
    :param yticks_length: Y-axis ticks of the burst length panel.
    :param gap: Vertical gap between the two panels in inches.
    :param file_name: File name to save the plot.
    :param ax_size: (width, height) of the combined axes box in inches.
    :param margins: (left, bottom, right, top) margins in inches around the axes box.
    """

    order = np.argsort(fractions)
    fractions = np.asarray(fractions, dtype=float)[order]

    left, bottom, right, top = margins
    fig_width, fig_height = left + ax_size[0] + right, bottom + ax_size[1] + top
    panel_height = (ax_size[1] - gap) / 2

    # Math labels in Times; \overline draws the bar as a line (\bar writes a combining character that shows as a circle in the svg)
    with mpl.rc_context({'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic'}):
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax_bottom = fig.add_axes([left / fig_width, bottom / fig_height, ax_size[0] / fig_width, panel_height / fig_height])
        ax_top = fig.add_axes([left / fig_width, (bottom + panel_height + gap) / fig_height, ax_size[0] / fig_width, panel_height / fig_height],
                              sharex=ax_bottom)

        for ax, values, ylabel, yticks in [(ax_top, burst_fraction, r'$\overline{F}$ (%)', yticks_fraction), (ax_bottom, burst_length, r'$\overline{L}$', yticks_length)]:
            mean, sem = (np.array(v, dtype=float) for v in zip(*[values[i] for i in order]))
            ax.plot(fractions, mean, '-', color=color, linewidth=0.75)
            ax.errorbar(fractions, mean, yerr=sem, fmt='.', color=color, capsize=2)
            ax.set_ylabel(ylabel)
            ax.set_yticks(yticks) # limits autoscale to cover both the ticks and the data
            sns.despine(ax=ax)

        ax_bottom.set_xlim([-0.05, 1.05])
        ax_bottom.set_xticks([0, 0.5, 1])
        ax_bottom.set_xlabel('Fraction intact')
        ax_top.tick_params(labelbottom=False)
        fig.align_ylabels([ax_top, ax_bottom])

        if file_name is not None:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')

    return

def plot_burst_length_intact_vs_silenced(spk_rec_intact, spk_rec_silenced, nb_steps, time_step, color_intact='tab:orange', color_silenced='grey',
                                         fig_size=(1.15, 1.0), file_name=None):
    """
    Superimpose the burst length distributions of the fully intact and fully silenced populations, with their means.

    :param spk_rec_intact: Spike record (nb_units, nb_steps) of the intact population.
    :param spk_rec_silenced: Spike record (nb_units, nb_steps) of the silenced population.
    :param nb_steps: Number of integration steps.
    :param time_step: Integration time step (ms).
    :param color_intact: Color of the intact population (solid mean line).
    :param color_silenced: Color of the silenced population (dashed mean line).
    :param fig_size: Figure size (same as the example trial plots).
    :param file_name: File name to save the plot.
    """

    time = np.arange(nb_steps)*np.asarray(time_step)

    plt.figure(figsize=fig_size)
    for spk_rec, color, linestyle, label in [(spk_rec_intact, color_intact, 'dashed', 'Intact'), (spk_rec_silenced, color_silenced, 'dashed', 'Silenced')]:
        classified_spikes = burst_classification(spk_rec=np.asarray(spk_rec), burst_thresh=15, time=time, track_burst_lengths=True)
        flat_burst_lengths = [y for x in classified_spikes['burst_lengths'] for y in x]

        bins = np.arange(2, max(flat_burst_lengths)+2) - 0.5 # Bin edges for integer burst lengths
        plt.hist(flat_burst_lengths, bins=bins, density=True, color=color, histtype='step', fill=False, label=label)
        plt.axvline(np.mean(flat_burst_lengths), color=color, linestyle=linestyle)

    plt.xlabel('Burst Length')
    plt.ylabel('Density')
    sns.despine()
    plt.xlim([1, 10])
    plt.xticks([0, 5, 10])
    plt.ylim([0, 1.0])
    plt.yticks([0, 0.5, 1.0])
    plt.legend(frameon=False, fontsize=6, handlelength=1, handletextpad=0.3, borderaxespad=0, loc='upper right')

    if file_name is not None:
        plt.savefig(file_name, dpi=300)

    return