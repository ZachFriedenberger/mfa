import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch

from utils import load_data, create_folder, save_data
from misc import compute_statistics_single_dataset

def reformat_data(data, dt=0.25):
    """
    Reformat the data by concatenating across areas and transposing.
    
    :param data: Dictionary containing neural data organized by mouse, session, and area.
    :param dt: Bin size for the time dimension. Used to convert rates to counts. 
    """

    area_keys = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
    rate_keys = ['event_rates', 'burst_rates', 'firing_rates']

    data_new = {}
    for mouse_id in data.keys():
            data_new[mouse_id] = {}
            for session in data[mouse_id].keys():
                    data_new[mouse_id][session] = {}
                    for rate in rate_keys:
                            data_new[mouse_id][session][rate] = []

                            for area in area_keys:
                                    if data[mouse_id][session][area] is not None: 
                                            data_new[mouse_id][session][rate].append(data[mouse_id][session][area][rate]['image']*dt)

                            data_new[mouse_id][session][rate] = np.concatenate(data_new[mouse_id][session][rate], axis=1) 
                            data_new[mouse_id][session][rate] = data_new[mouse_id][session][rate].T 

    return data_new


if __name__ == "__main__":

    create_folder('data/vbn')
    data = load_data('data/vbn/vformatted_data_v5_mIDs.pkl')
    data_new = reformat_data(data) # Merge and reformat the data
    save_data(data_new, 'data/vbn/merged_data_mIDs.pkl') # Save the merged and reformatted data
    