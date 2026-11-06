from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

MFA_REPO = Path("~/Documents/GitHub/multiplexed-factor-analysis").expanduser()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(MFA_REPO / "src"))      # read-only import of the Rodgers burst definition
from rodgers_choice_binned_counts import ISI_THRESHOLD_S, classify_bursts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update({
    "text.usetex": False,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    # math in Times; \overline draws the bar as a line (\bar writes a combining character that shows as a circle in the svg)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "font.size": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})

FIG_SIZE = (1.15, 1.0)
VBN_TICK_WIDTH, VBN_WINDOW_LEN = 0.8, 1.5     # notebook values; the reference for scaling the tick width

CUSTOM_COLORS = {"bluish_green": (0, 158. / 255, 115. / 255), "pink": (0.8, 0.6, 0.7)}

VBN_AREAS = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
VBN_WINDOW = (-0.75, 0.75)
VBN_SHADING = [(-0.75, -0.5, CUSTOM_COLORS["bluish_green"], 0.3), (-0.5, 0.0, "gray", 0.1),
               (0.0, 0.25, CUSTOM_COLORS["pink"], 0.3), (0.25, 0.75, "gray", 0.1)]

RODGERS_LAYERS = {"L5b": {"5b"}, "L5": {"5a", "5b"}, "deep": {"5a", "5b", "6"}, "all": None}
RODGERS_WINDOW = {"choice": (-3.0, 3.0), "start": (0.0, 6.0)}     # same windows as the MFA fits
RODGERS_XLABEL = {"choice": "Time from choice (s)", "start": "Time from trial start (s)"}


def default_tick_width(window):
    return VBN_TICK_WIDTH * VBN_WINDOW_LEN / (window[1] - window[0])


# --------------------------------------------------------------------------- #
# Shared plots                                                                #
# --------------------------------------------------------------------------- #

def plot_burst_lengths(lengths, min_burst_size=2, fig_size=FIG_SIZE, file_name=None):
    """Histogram of burst lengths (VBN plot_number_of_spikes_per_burst_data)."""
    lengths = np.asarray(lengths)
    fig = plt.figure(figsize=fig_size)
    try:
        bins = np.arange(min_burst_size, lengths.max() + 2) - 0.5   # integer lengths
        plt.hist(lengths, bins=bins, density=True, color="tab:orange", histtype="step", fill=False)

        plt.xlabel("Burst length")
        plt.ylabel("Density")
        sns.despine()
        plt.xlim([0, 10.5])
        plt.xticks([0, 5, 10])
        plt.ylim([0, 0.8])
        plt.yticks([0, 0.4, 0.8])
        plt.vlines(np.mean(lengths), ymin=0, ymax=plt.ylim()[1], color="tab:orange", linestyle="--")   # mean, to the top of the y axis

        mean_len = np.mean(lengths)
        sem_len = np.std(lengths) / np.sqrt(lengths.size)
        ax = plt.gca()
        # label starts just right of the mean line (x in data, y in axes coordinates)
        ax.text(mean_len + 0.4, 0.85, "$\\overline{L}=$" + str(np.round(mean_len, 3)) + "$\\pm$" + str(np.round(sem_len, 3)),
                transform=ax.get_xaxis_transform(), ha="left", va="top")

        if file_name is not None:
            plt.savefig(file_name, dpi=300)
    finally:
        plt.close(fig)


def burst_fraction_text(bf):
    """'F = m ± sem %' (F overlined) over neurons, from per-neuron burst fractions (NaN = neuron without events, dropped)."""
    bf = 100 * np.asarray(bf, dtype=float)
    bf = bf[np.isfinite(bf)]
    return f"$\\overline{{F}}$ = {bf.mean():.1f} $\\pm$ {bf.std() / np.sqrt(bf.size):.1f}%"


def plot_raster(spike_rows, burst_rows, *, window, xticks, xticklabels, xlabel, shading=(), event_line=None,
                tick_width=None, burst_width_scale=2.0, title=None, fig_size=FIG_SIZE, file_name=None):
    """Single-trial raster (VBN plot_raster_from_data): one row per neuron, spikes in black, burst onsets in
    orange on top (burst_width_scale x wider). `shading` = [(start, end, color, alpha)]; `event_line` = time of
    a dashed vertical line; `title` = text above the panel (the baseline burst fraction). Neurons are numbered from 1 (row k at y = k), the y axis starts at 0."""
    tick_width = default_tick_width(window) if tick_width is None else tick_width
    lo, hi = window
    n = len(spike_rows)
    fig = plt.figure(figsize=fig_size)
    try:
        ax = fig.gca()
        rows = np.arange(1, n + 1)   # explicit offsets, neurons numbered from 1
        ax.eventplot(spike_rows, lineoffsets=rows, color="k", linewidths=tick_width)
        ax.eventplot(burst_rows, lineoffsets=rows, color="tab:orange", linewidths=tick_width * burst_width_scale)

        for start, end, color, alpha in shading:
            ax.axvspan(max(start, lo), min(end, hi), color=color, alpha=alpha, zorder=0, ec="none")
        if event_line is not None:
            ax.axvline(event_line, color="k", linestyle="--", linewidth=0.6, zorder=0)

        plt.xticks(xticks, xticklabels)
        plt.xlim(lo, hi)
        plt.ylim(0, n + 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))   # 3 whole-neuron ticks
        ax.set_ylabel("Neuron")
        ax.set_xlabel(xlabel)
        if title is not None:
            ax.set_title(title, fontsize=8)
        sns.despine()

        if file_name is not None:
            plt.savefig(file_name, dpi=300)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# VBN                                                                         #
# --------------------------------------------------------------------------- #

VBN_DATA_DIR = MFA_REPO / "data" / "vbn"
VBN_BURST_LENGTHS = MFA_REPO / "data" / "vbn" / "burst_lengths.pkl"


def run_vbn(out_dir, mice=(574078, 599294), n_example_trials=3, tick_width=None, burst_width_scale=2.0, seed=0,
            data_dir=VBN_DATA_DIR, burst_lengths_file=VBN_BURST_LENGTHS):
    """Per-session burst-length histograms and example rasters for `mice` (`n_example_trials` random hits and as
    many misses per session, as for Rodgers; None = every trial)."""
    out_dir = Path(out_dir)

    burst_length_data = pickle.load(open(burst_lengths_file, "rb"))
    for mouse_id in mice:
        (out_dir / "burst_lengths" / f"mouse_{mouse_id}").mkdir(parents=True, exist_ok=True)
        for session_id, lengths in burst_length_data[mouse_id].items():
            plot_burst_lengths(lengths, min_burst_size=2,
                               file_name=out_dir / "burst_lengths" / f"mouse_{mouse_id}" / f"burst_lengths_session_{session_id}.svg")
    print(f"VBN: burst lengths for mice {list(mice)}")

    rng = np.random.default_rng(seed)
    data = pickle.load(open(data_dir / "spike_times.pkl", "rb"))
    counts = pickle.load(open(data_dir / "merged_data_mIDs.pkl", "rb"))
    for mouse_id in mice:
        for session_id, sess in data[mouse_id].items():
            c = counts[mouse_id][session_id]
            title = burst_fraction_text(np.mean(c["burst_rates"], axis=1) / np.mean(c["event_rates"], axis=1))
            areas = [a for a in VBN_AREAS if sess[a] is not None]
            folder = out_dir / "raster_plots" / f"mouse_{mouse_id}" / f"session_{session_id}"
            folder.mkdir(parents=True, exist_ok=True)
            hits = np.asarray(sess["outcome"]) == "hit"
            if n_example_trials is None:
                trials = np.arange(hits.size)
            else:
                trials = np.sort(np.concatenate([rng.choice(np.flatnonzero(m), size=min(n_example_trials, int(m.sum())), replace=False)
                                                 for m in (hits, ~hits)]))
            for trial in trials:
                spikes = [sess[a]["spike_times"][k][trial] for a in areas for k in range(len(sess[a]["spike_times"]))]
                bursts = [sess[a]["burst_times"][k][trial] for a in areas for k in range(len(sess[a]["burst_times"]))]
                plot_raster(spikes, bursts, window=VBN_WINDOW, xticks=[-0.75, 0, 0.75], xticklabels=["-750", "0", "+750"],
                            xlabel="Time from change (ms)", shading=VBN_SHADING, tick_width=tick_width,
                            burst_width_scale=burst_width_scale, title=title,
                            file_name=folder / f"raster_image_change_{trial}_{'hit' if hits[trial] else 'miss'}.svg")
            print(f"VBN mouse {mouse_id} session {session_id}: {len(spikes)} neurons, {trials.size} rasters, {title}")


# --------------------------------------------------------------------------- #
# Rodgers                                                                     #
# --------------------------------------------------------------------------- #

def rodgers_trial_rows(session, trial, anchor_times, window, isi_threshold, min_burst_size):
    """Spike and burst-onset times relative to the anchor, one row per neuron, deepest neuron at the bottom.
    Bursts are classified on the whole spike train before cutting to the window, so a burst starting just
    before the window edge is still classified correctly."""
    t0 = float(anchor_times[trial])
    lo, hi = window
    order = np.argsort(-np.asarray(session["depths"], dtype=float))
    spike_rows, burst_rows = [], []
    for k in order:
        st = np.sort(np.asarray(session["spike_times"][k], dtype=float))
        bt = classify_bursts(st, isi_threshold, min_burst_size)[0]
        spike_rows.append(st[(st >= t0 + lo) & (st < t0 + hi)] - t0)
        burst_rows.append(bt[(bt >= t0 + lo) & (bt < t0 + hi)] - t0)
    return spike_rows, burst_rows


def keep_layers(session, layers):
    allowed = RODGERS_LAYERS[layers]
    if allowed is None:
        return session
    keep = np.flatnonzero([str(lyr) in allowed for lyr in session["layers"]])
    out = dict(session)
    for key in ("layers", "depths", "unit_ids", "spike_times"):
        v = session[key]
        out[key] = [v[i] for i in keep] if isinstance(v, list) else np.asarray(v)[keep]
    out["n_neurons"] = keep.size
    return out


def run_rodgers(out_dir, task="discrimination", layers="deep", align="choice", sessions=None, min_burst_size=2,
                isi_threshold=ISI_THRESHOLD_S, n_example_trials=3, tick_width=None, burst_width_scale=2.0, seed=0,
                data_dir=MFA_REPO / "data" / "rodgers"):
    """Per-session (+ pooled) burst-length histograms and example rasters (`n_example_trials` random hits and as
    many misses per session; None = every trial). `sessions`: list of stems such as 'sub-KF132_ses-20180223'
    (prefix match on the pickle name); None = all sessions of the task."""
    out_dir = Path(out_dir) / f"{task}_{layers}_{align}"
    (out_dir / "burst_lengths").mkdir(parents=True, exist_ok=True)
    window = RODGERS_WINDOW[align]
    lo, hi = window
    if align == "choice":
        xticks, xticklabels = [lo, 0, hi], [f"{lo:+g}", "0", f"{hi:+g}"]
    else:
        xticks, xticklabels = [lo, (lo + hi) / 2, hi], [f"{x:g}" for x in (lo, (lo + hi) / 2, hi)]
    pkls = sorted((Path(data_dir) / task).glob("*.pkl"))
    if sessions:
        pkls = [p for p in pkls if any(p.name.startswith(s) for s in sessions)]
    if not pkls:
        raise FileNotFoundError(f"no session pickles in {Path(data_dir) / task} for {sessions}")

    rng = np.random.default_rng(seed)
    pooled = []
    for pkl in pkls:
        session = keep_layers(pickle.load(open(pkl, "rb")), layers)
        if session["n_neurons"] == 0:
            print(f"{pkl.name}: no {layers} neurons, skipped"); continue
        name = pkl.stem.split("_behavior")[0]

        # burst lengths over each neuron's whole recording, pooled over the session's neurons
        lengths = np.concatenate([classify_bursts(st, isi_threshold, min_burst_size)[2] for st in session["spike_times"]])
        pooled.append(lengths)
        if lengths.size:
            plot_burst_lengths(lengths, min_burst_size, file_name=out_dir / "burst_lengths" / f"burst_lengths_{name}.svg")

        anchor = np.asarray(session["choice_times"] if align == "choice" else session["trial_start_times"], dtype=float)
        # baseline burst fraction: per neuron, bursts / events in the raster windows of all trials
        bf = []
        for st in session["spike_times"]:
            bt, et, _ = classify_bursts(st, isi_threshold, min_burst_size)
            n_b, n_e = (np.diff(np.searchsorted(x, np.stack([anchor + lo, anchor + hi]), side="left"), axis=0).sum()
                        for x in (bt, et))
            bf.append(n_b / n_e if n_e else np.nan)
        title = burst_fraction_text(bf)
        hits = np.asarray(session["hits"], dtype=bool)
        if n_example_trials is None:
            trials = np.arange(hits.size)
        else:
            trials = np.sort(np.concatenate([rng.choice(np.flatnonzero(m), size=min(n_example_trials, int(m.sum())), replace=False)
                                             for m in (hits, ~hits)]))
        folder = out_dir / "raster_plots" / name
        folder.mkdir(parents=True, exist_ok=True)
        for j in trials:
            spikes, bursts = rodgers_trial_rows(session, j, anchor, window, isi_threshold, min_burst_size)
            # grey band: response window open -> choice lick (relative to the anchor)
            rwin = float(session["response_window_open_times"][j]) - anchor[j]
            choice = float(session["choice_times"][j]) - anchor[j]
            plot_raster(spikes, bursts, window=window, xticks=xticks, xticklabels=xticklabels,
                        xlabel=RODGERS_XLABEL[align], shading=[(rwin, choice, "gray", 0.1)], event_line=choice,
                        tick_width=tick_width, burst_width_scale=burst_width_scale, title=title,
                        file_name=folder / f"raster_trial_{session['trial_ids'][j]}_{'hit' if hits[j] else 'miss'}.svg")
        print(f"Rodgers {name}: {session['n_neurons']} {layers} neurons, {lengths.size} bursts, {trials.size} rasters, {title}")

    pooled = np.concatenate(pooled)
    if pooled.size:
        plot_burst_lengths(pooled, min_burst_size, file_name=out_dir / "burst_lengths" / "burst_lengths_all_sessions.svg")
    print(f"-> {out_dir}")


if __name__ == "__main__":

    out = ROOT / "results" / "data_rasters"

    # VBN: burst lengths and example rasters (3 random hits + 3 misses per session) for these mice
    run_vbn(out / "vbn", mice=[574078, 599294], n_example_trials=3)

    # Rodgers: burst lengths and example rasters (3 random hits + 3 misses per session), L5b
    run_rodgers(out / "rodgers", task="discrimination", layers="L5b", align="choice",
                sessions=["sub-KM131_ses-20180119", "sub-KM131_ses-20180118"],
                n_example_trials=3)

    # Rodgers: burst lengths and example rasters (3 random hits + 3 misses per session), L5a + L5b
    run_rodgers(out / "rodgers", task="discrimination", layers="deep", align="choice",
                sessions=["sub-KF132_ses-20180223", "sub-KF132_ses-20180222"],
                n_example_trials=3)