from __future__ import annotations
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import h5py
import pynwb
import remfile
from dandi.dandiapi import DandiAPIClient

DANDISET_ID = "000231"
DANDISET_VERSION = "draft"

def discover_ephys_assets(dandiset_id: str = DANDISET_ID, version: str = DANDISET_VERSION):
    """
    Discover ecephys-containing NWB assets in a DANDI dataset.

    :param dandiset_id: The ID of the DANDI dataset to query.
    :param version: The version of the DANDI dataset to query.
    :return: A list of tuples, each containing the path and S3 URL of an ecephys-containing NWB asset.
    """

    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, version)
    out = []

    print(f"Discovering ecephys assets in DANDI dataset {dandiset_id} (version {version})...")
    for asset in dandiset.get_assets():
        if "ecephys" in asset.path and asset.path.endswith(".nwb"):
            url = asset.get_content_url(follow_redirects=1, strip_query=True)
            out.append((asset.path, url))
    return out

def session_filename(asset_path: str) -> str:
    return Path(asset_path).stem + ".pkl"

def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x

def _as_str_array(series_or_array) -> np.ndarray:
    return np.array([_decode(v) for v in np.asarray(series_or_array)], dtype=object)

def open_remote_nwb(s3_url: str):
    """Open an NWB file by HTTP-streaming the underlying HDF5.

    :param s3_url: The S3 URL to access the NWB asset.
    :return: A tuple containing the NWBHDF5IO object and the read NWBFile object.
    """
    remote_file = remfile.File(s3_url)
    h5_file = h5py.File(remote_file, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)
    return io, io.read()

def process_session(asset_path: str,s3_url: str) -> dict | None:
    """Pre-process a session from a DANDI NWB asset.

    :param asset_path: The path of the NWB asset within the DANDI dataset.
    :param s3_url: The S3 URL to access the NWB asset.
    :return: A dictionary containing pre-processed session data, or None if the session is not usable.
    """
    io, nwbfile = open_remote_nwb(s3_url)
    try:

        # Extract unit information from the NWB file.
        units_df = nwbfile.units.to_dataframe()

        # Check if the necessary columns are present in the units dataframe. We want to analyze excitatory neuron spike times. 
        if "inhibitory" not in units_df.columns or "spike_times" not in units_df.columns:
            return None

        # Check that there are excitatory neurons present in the units dataframe.
        excitatory_mask = ~units_df["inhibitory"].astype(bool).to_numpy()
        units_exc = units_df.loc[excitatory_mask]
        n_neurons = len(units_exc)
        if n_neurons == 0:
            return None

        # Get the cortical layer and depth information for each excitatory neuron.
        layers = _as_str_array(units_exc["layer"]) if "layer" in units_exc.columns else \
            np.array([None] * n_neurons, dtype=object)
        depths = units_exc["depth"].astype(float).to_numpy() if "depth" in units_exc.columns else \
            np.full(n_neurons, np.nan)

        # Spike times per excitatory unit (each cell is already a numpy array)
        spike_times_list = [np.asarray(st, dtype=float) for st in units_exc["spike_times"].tolist()]


        # Extract trials information from the NWB file. Keep only random, non-opto, valid trials with a real choice (left or right).
        trials_df = nwbfile.trials.to_dataframe()
        outcome_str = trials_df["outcome"].astype(str).map(_decode)
        choice_str = trials_df["choice"].astype(str).map(_decode)
        keep = (
            (~trials_df["ignore_trial"].astype(bool).to_numpy())
            & (trials_df["stim_is_random"].astype(bool).to_numpy())
            & outcome_str.isin({"correct", "error"}).to_numpy()
            & choice_str.isin({"left", "right"}).to_numpy()
        )
        trials_kept = trials_df.loc[keep].reset_index(drop=True)
        if len(trials_kept) == 0:
            return None
        n_trials = len(trials_kept)

        # Extract the task type
        stim_all = set(_decode(s) for s in trials_df["stimulus"].astype(str).unique())
        task_type = "detection" if "nothing" in stim_all else "discrimination"

        # Lick times: every session stores them at processing['behavior']['licks']
        licks_ts = nwbfile.processing["behavior"]["licks"].time_series
        lick_times = {side: np.asarray(licks_ts[side].timestamps[:], dtype=float)
                      for side in ("left", "right")}

        # Outcome for each trial: 'correct' indicates a hit, 'error' indicates a miss.
        outcomes = _as_str_array(trials_kept["outcome"])
        hits = outcomes == "correct"

        subject_id = nwbfile.subject.subject_id if nwbfile.subject is not None else None
        session_id = getattr(nwbfile, "session_id", None)

        return {
            "asset_path": asset_path,
            "session_id": session_id,
            "subject_id": subject_id,
            "task_type": task_type,
            "n_neurons": n_neurons,
            "n_trials": n_trials,
            "layers": layers,
            "depths": depths,
            "unit_ids": units_exc.index.to_numpy(),
            "spike_times": spike_times_list,
            "trial_ids": trials_kept["trial"].astype(int).to_numpy(),
            "trial_start_times": trials_kept["start_time"].astype(float).to_numpy(),
            "trial_stop_times": trials_kept["stop_time"].astype(float).to_numpy(),
            "response_window_open_times": trials_kept["response_window_open_time"].astype(float).to_numpy(),
            "choice_times": trials_kept["choice_time"].astype(float).to_numpy(),
            "outcomes": outcomes,
            "hits": hits,
            "stimulus": _as_str_array(trials_kept["stimulus"]),
            "choice": _as_str_array(trials_kept["choice"]),
            "rewarded_side": _as_str_array(trials_kept["rewarded_side"]),
            "servo_position": _as_str_array(trials_kept["servo_position"])
                if "servo_position" in trials_kept.columns else None,
            "lick_times": lick_times,
        }
    finally:
        io.close()




def main(output_dir: str, overwrite: bool):
    """
    Main function to download and process Rodgers ecephys data from the DANDI archive.
    
    :param output_dir: Directory where processed session data and summary will be saved.
    :param overwrite: Whether to overwrite existing processed session files.
    """

    # Create output directories if they don't exist
    output_root = Path(output_dir)
    (output_root / "detection").mkdir(parents=True, exist_ok=True)
    (output_root / "discrimination").mkdir(parents=True, exist_ok=True)

    # Get the list of ecephys assets from the DANDI archive
    assets = discover_ephys_assets()
    print(f"Found {len(assets)} ecephys sessions in dandiset {DANDISET_ID}.")

    summaries = []
    for asset_path, url in tqdm(assets, desc="Sessions"):
        # Skip if any already-saved file exists for this asset, in either task subdir
        if not overwrite:
            existing = [output_root / "detection" / session_filename(asset_path), output_root / "discrimination" / session_filename(asset_path)]
            if any(p.exists() for p in existing):
                continue

        try:
            session = process_session(asset_path, url) # Process the session
        except Exception as e:
            print(f"\n[skip] {asset_path}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        if session is None:
            continue

        # Save the processed session to the appropriate output directory
        out_path = output_root / session["task_type"] / session_filename(asset_path)
        with open(out_path, "wb") as f:
            pickle.dump(session, f)

        summaries.append({
            "asset_path": asset_path,
            "subject_id": session["subject_id"],
            "task_type": session["task_type"],
            "n_neurons": session["n_neurons"],
            "n_trials": session["n_trials"],
            "n_hits": int(session["hits"].sum()),
            "n_misses": int((~session["hits"]).sum()),
        })


    # Save the summary statisticsof all processed sessions
    summary_df = pd.DataFrame(summaries)
    summary_path = output_root / "summary.csv"
    if summary_path.exists() and not overwrite:
        old = pd.read_csv(summary_path)
        summary_df = pd.concat([old, summary_df], ignore_index=True).drop_duplicates(
            subset=["asset_path"], keep="last"
        )
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summaries)} new sessions; summary at {summary_path}")


if __name__ == "__main__":

    main(output_dir="data/rodgers", overwrite=False)