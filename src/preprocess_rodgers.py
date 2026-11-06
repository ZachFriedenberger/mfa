from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

ISI_THRESHOLD_S = 0.015      # same 15 ms threshold as the rest of the project
BIN_WIDTH_S = 0.25 # bin width in seconds

# Anchors the bins can be aligned to either the choice time or the start of the trial.
ANCHORS = {
    "choice": ("choice_times", (-3.0, 3.0)),
    "trial_start": ("trial_start_times", (0.0, 6.0)),
}

# Layer groups for --layer_group. Values are the raw `layers` labels stored in
# the session pickles ('1', '2/3', '4', '5a', '5b', '6', '6b').
LAYER_GROUPS = {"L5B": {"5b"}, "L5": {"5b", "5a"}, "deep": {"5b", "5a", "6"}}

# Per-neuron session fields; these are subset together when filtering neurons.
PER_NEURON_KEYS = ("layers", "depths", "unit_ids", "spike_times")

# Session fields copied unchanged from the input pickle into the output.
PASSTHROUGH_KEYS = (
    "asset_path", "session_id", "subject_id", "task_type",
    "n_neurons", "n_trials", "layers", "depths", "unit_ids",
    "trial_ids", "trial_start_times", "trial_stop_times",
    "response_window_open_times", "choice_times",
    "outcomes", "hits", "stimulus", "choice", "rewarded_side",
    "servo_position",
)


def classify_bursts(spike_times: np.ndarray, isi_threshold: float, min_burst_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one neuron's spike train into burst times and event times.

    Returns
      burst_times   : first-spike time of each burst
      event_times   : sorted burst_times + singlet spike times
      burst_lengths : number of spikes in each burst, aligned with burst_times
                      (burst k spans the burst_lengths[k] spikes starting at
                      burst_times[k])
    """

    # Sort the spike times and convert to a numpy array of floats.
    t = np.sort(np.asarray(spike_times, dtype=float))
    if t.size == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    # Group spikes into runs of spikes. That is, spikes in potential bursts.
    starts_new_run = np.ones(t.size, dtype=bool)
    starts_new_run[1:] = np.diff(t) >= isi_threshold
    run_starts = np.flatnonzero(starts_new_run)           # index of first spike
    run_sizes = np.diff(np.append(run_starts, t.size))    # spikes per run

    # Identify which runs are bursts based on their size.
    is_burst = run_sizes >= min_burst_size
    burst_times = t[run_starts[is_burst]]
    burst_lengths = run_sizes[is_burst]

    # Singlets: all spikes belonging to runs that are too short to be bursts.
    singlet_times = np.concatenate([t[s:s + n] for s, n in zip(run_starts[~is_burst], run_sizes[~is_burst])]) if np.any(~is_burst) else np.array([])
    event_times = np.sort(np.concatenate([burst_times, singlet_times]))

    return burst_times, event_times, burst_lengths


def count_in_bins(times: np.ndarray, anchors: np.ndarray, rel_edges: np.ndarray) -> np.ndarray:
    """Count the number of `times` falling into bins defined by `anchors` and `rel_edges`.

    Returns an array of shape (n_anchors, n_bins) where each element is the count of `times`
    in the corresponding bin [anchor + rel_edges[k], anchor + rel_edges[k+1]).

    `times` must be sorted.
    """

    # searchsorted(side="left") gives the number of times < each edge, so the
    # difference between consecutive edges counts the half-open bin.
    abs_edges = anchors[:, None] + rel_edges[None, :]
    n_before = np.searchsorted(times, abs_edges, side="left")
    return np.diff(n_before, axis=1)


def filter_neurons(session: dict, layer_group: str) -> dict:
    """Return a copy of `session` keeping only neurons in `layer_group`."""

    # Determine which neurons to keep based on the specified layer group.
    allowed = LAYER_GROUPS[layer_group]
    keep = np.flatnonzero([str(lyr) in allowed for lyr in session["layers"]])

    # Create a copy of the session dictionary to modify. Keep only the neurons specified by `keep`.
    out = dict(session)
    for key in PER_NEURON_KEYS:
        values = session[key]
        out[key] = [values[i] for i in keep] if isinstance(values, list) \
            else np.asarray(values)[keep]
    out["n_neurons"] = keep.size
    return out


def process_session(session: dict, *, isi_threshold: float, min_burst_size: int, rel_edges: np.ndarray, layer_group: str | None = None, anchor: str = "choice") -> dict:
    """Process a single session, counting spikes, events, and bursts in time bins relative to an anchor time.

    session        : dictionary containing session data.
    isi_threshold  : max ISI (s) between consecutive spikes within a burst.
    min_burst_size : minimum spikes per burst.
    rel_edges      : bin edges (s) relative to the anchor.
    layer_group    : keep only neurons in this layer group (L5 = layers 5a + 5b).
                     None keeps all excitatory neurons.
    anchor         : what the bins are aligned to, a key of ANCHORS
                     ("choice" or "trial_start").

    Returns a dictionary with binned counts and metadata.
    """
    # Extract the anchor times and initialize count arrays.
    anchor_times = np.asarray(session[ANCHORS[anchor][0]], dtype=float)
    n_neurons, n_trials, n_bins = session["n_neurons"], anchor_times.size, rel_edges.size - 1

    # Flag trials whose window extends outside [trial_start, trial_stop), i.e.
    # into the previous or next trial.
    window_outside_trial = (
        (anchor_times + rel_edges[0] < np.asarray(session["trial_start_times"], dtype=float))
        | (anchor_times + rel_edges[-1] > np.asarray(session["trial_stop_times"], dtype=float))
    )

    spike_counts = np.zeros((n_neurons, n_trials, n_bins), dtype=np.int32)
    event_counts = np.zeros_like(spike_counts)
    burst_counts = np.zeros_like(spike_counts)
    event_times_all, burst_times_all, burst_lengths_all = [], [], []


    # Count spikes, events, and bursts in each bin for each neuron.
    for k, st in enumerate(session["spike_times"]):
        st = np.sort(np.asarray(st, dtype=float))
        burst_times, event_times, burst_lengths = classify_bursts(st, isi_threshold, min_burst_size)

        spike_counts[k] = count_in_bins(st, anchor_times, rel_edges)
        event_counts[k] = count_in_bins(event_times, anchor_times, rel_edges)
        burst_counts[k] = count_in_bins(burst_times, anchor_times, rel_edges)
        event_times_all.append(event_times)
        burst_times_all.append(burst_times)
        burst_lengths_all.append(burst_lengths)

    out = {key: session.get(key) for key in PASSTHROUGH_KEYS}
    out.update({
        "spike_counts": spike_counts,
        "event_counts": event_counts,
        "burst_counts": burst_counts,
        "anchor": anchor,
        "bin_edges": rel_edges,
        "bin_centers": rel_edges[:-1] + np.diff(rel_edges) / 2,
        "window_outside_trial": window_outside_trial,
        "event_times": event_times_all,
        "burst_times": burst_times_all,
        "burst_lengths": burst_lengths_all,
        "isi_threshold": isi_threshold,
        "min_burst_size": min_burst_size,
        "layer_group": layer_group,
    })
    return out


def main(input_dir: str | Path, output_dir: str | Path, min_burst_size: int,
         task: str = "discrimination", layer_group: str | None = None,
         isi_threshold: float = ISI_THRESHOLD_S, bin_width: float = BIN_WIDTH_S,
         anchor: str = "choice", window: tuple[float, float] | None = None):
    """Bin every session pickle in <input_dir>/<task>/ and write the results
    to <output_dir>/<task>/.

    input_dir      : folder written by download_rodgers_data.py
                     (contains detection/ and discrimination/).
    output_dir     : where the binned pickles are written.
    min_burst_size : minimum spikes per burst.
    task           : "discrimination" or "detection".
    layer_group    : keep only neurons in this layer group (L5 = layers 5a + 5b).
                     None keeps all excitatory neurons.
    isi_threshold  : max ISI (s) between consecutive spikes within a burst.
    bin_width      : bin width (s).
    anchor         : what the bins are aligned to: "choice" (choice lick) or
                     "trial_start".
    window         : (start, stop) in seconds relative to the anchor. None uses
                     the anchor's default: (-3, 3) for "choice", (0, 6) for
                     "trial_start".
    """

    # Validate input arguments.
    if min_burst_size < 2:
        raise ValueError(f"min_burst_size must be at least 2, got {min_burst_size}")
    if task not in ("discrimination", "detection"):
        raise ValueError(f"task must be 'discrimination' or 'detection', got {task!r}")
    if layer_group is not None and layer_group not in LAYER_GROUPS:
        raise ValueError(f"layer_group must be one of {sorted(LAYER_GROUPS)} "
                         f"or None, got {layer_group!r}")
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {sorted(ANCHORS)}, got {anchor!r}")
    if window is None:
        window = ANCHORS[anchor][1]

    # Bin edges across the window; rounding avoids float drift (e.g. 2.9999999).
    rel_edges = np.round(np.arange(window[0], window[1] + bin_width / 2, bin_width), 6)

    in_dir = Path(input_dir).expanduser() / task
    out_dir = Path(output_dir).expanduser() / task
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all session pickle files in the input directory.
    pkls = sorted(in_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No session pickles found in {in_dir}")

    # Loop through the pickel files and process each session.
    for pkl in pkls:
        with open(pkl, "rb") as f:
            session = pickle.load(f)

        # Filter neurons by the specified layer group, if any.
        if layer_group is not None:
            session = filter_neurons(session, layer_group)
            if session["n_neurons"] == 0:
                print(f"{pkl.name}: no {layer_group} neurons, skipped")
                continue

        # Process the session to compute binned event and bursts counts.
        out = process_session(session, isi_threshold=isi_threshold, min_burst_size=min_burst_size, rel_edges=rel_edges, layer_group=layer_group, anchor=anchor)

        # Save the processed session to the output directory.
        with open(out_dir / pkl.name, "wb") as f:
            pickle.dump(out, f)
        print(f"{pkl.name}: {out['n_neurons']} neurons x {out['n_trials']} trials "
              f"x {rel_edges.size - 1} bins, "
              f"{out['window_outside_trial'].sum()} trials with window outside trial")


if __name__ == "__main__":

    anchors = ANCHORS.keys()
    for anchor in anchors:
            
            main(input_dir="data/rodgers", output_dir=f"data/rodgers_L5B_{anchor}_binned_{BIN_WIDTH_S}", min_burst_size=2, 
                 task="discrimination", anchor=anchor, layer_group="L5B")

            main(input_dir="data/rodgers", output_dir=f"data/rodgers_deep_{anchor}_binned_{BIN_WIDTH_S}", min_burst_size=2, 
                 task="discrimination", anchor=anchor, layer_group="deep")