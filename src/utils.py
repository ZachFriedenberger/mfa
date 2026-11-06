import numpy as np
import pickle
import os
import json
import torch


def set_seed(seed=42):
    """
    Set seeds for NumPy and PyTorch (CPU & GPU) to ensure reproducibility.
    
    :param seed: Seed value to be set for random number generators.
    :return: The seed value that was set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # For multi-GPU setups
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")

    return seed

def create_folder(folder_path):
    """
    Create a folder if it does not already exist.
    
    :param folder_path: Path to the folder to be created.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

def save_meta_data(meta_data, file_name):
    """
    Save metadata to a JSON file.
    
    :param meta_data: Dictionary containing metadata to be saved.
    :param file_name: Path to the JSON file where metadata will be saved.
    """

    with open(file_name, "w") as f:
        json.dump(meta_data, f, indent=4)
    return

def save_data(data, filename):
    """
    Save data to a file using pickle.
    
    :param data: Data to be saved.
    :param filename: Path to the file where data will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    """
    Load data from a pickle file.
    
    :param filename: Path to the pickle file to be loaded.
    :return: Data loaded from the pickle file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_count_data(path_to_save, fraction=1.0):
    """
    Load count data from saved pickle files and optionally subsample it.
    
    :param path_to_save: Relative path to the folder containing the saved count data.
    :param fraction: Fraction of the total samples to be loaded (between 0 and 1).
    :return: Dictionary containing the subsampled 'events' and 'bursts' data as PyTorch tensors.
    """

    # load counts data
    total_counts = load_data(f"../results/{path_to_save}/total_counts.pkl")
    events, bursts = total_counts['event_counts'], total_counts['burst_counts']

    # Reshape the count data
    events_samples = events.transpose(0, 2, 1).reshape(-1, events.shape[1])
    bursts_samples = bursts.transpose(0, 2, 1).reshape(-1, bursts.shape[1])
    events_samples = torch.tensor(events_samples, dtype=torch.float)
    bursts_samples = torch.tensor(bursts_samples, dtype=torch.float)

    nb_samples = events_samples.shape[0]
    nb_samples_tmp = int(nb_samples*fraction)

    events_samples = events_samples[:nb_samples_tmp, :] # Quick test
    bursts_samples = bursts_samples[:nb_samples_tmp, :]

    data = {'events': events_samples, 'bursts': bursts_samples}

    return data