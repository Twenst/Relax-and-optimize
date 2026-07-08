import json
import os
import numpy as np
from torch import nn


class Constants:
    baseInstanceDataPath = "data/instances.npz"
    instancesDatasetPath = "data/dataset"
    configFilePath = "config.json"
    savedModelsPath = "saved_models"

    defaultConfig = {
        "learning_rate": 0.0001,
        "num_epochs": 10,
        "batch_size": 10,
        "n_rep": 10,
        "n_instances": 150,
        "train_size": 0.6,
        "test_size": 0.2,
        "timeout": 50e-3,
        "hidden_dims": [32],
    }


def parse_vars(vars_vect, n_facilities, n_clients):
    '''
    Parse the variable vector into facility and client variables.
    There are n_facilities * n_clients X variables and n_clients Y variables.
    '''
    x = vars_vect[0 : n_facilities * n_clients]
    y = vars_vect[n_facilities * n_clients :]
    x = np.reshape(x, (n_facilities, n_clients))
    return x, y


def compute_gaps_from_callback(obj_bound_arr):
    obj_bound_arr = np.array(obj_bound_arr)
    gaps = np.abs(obj_bound_arr[:, 0] - obj_bound_arr[:, 1]) / np.abs(
        obj_bound_arr[:, 0]
    )
    return gaps


def save_config(config):
    with open(Constants.configFilePath, "w") as f:
        json.dump(config, f, indent=4)


def load_config(configFilePath=None):
    if configFilePath is None:
        configFilePath = Constants.configFilePath
        
    if not os.path.exists(configFilePath):
        print(f"Config file not found: {configFilePath}, using default configuration.")
        config = {}
    else:
        with open(configFilePath, "r") as f:
            config = json.load(f)

    for key in Constants.defaultConfig:
        if key not in config:
            print(f"Using default value for {key}: {Constants.defaultConfig[key]}")
            config[key] = Constants.defaultConfig[key]

    config["n_instances"] = min(config["n_instances"], len(os.listdir(Constants.instancesDatasetPath)))


    return config

def time_to_date(time_int):
    import datetime
    return datetime.datetime.fromtimestamp(time_int).strftime('%Y-%m-%d_%H-%M-%S')

def load_linear_nn_from_dict(w_dict):
    layer_sizes = []
    
    # Look for weight sizes to determine layer sizes
    for i in range(len(w_dict)//2):
        layer_sizes.append(w_dict[f"{2*i}.weight"].shape[1])
    layer_sizes.append(1)
    
    model = nn.Sequential()
    for i in range(len(layer_sizes) - 1):
        model.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i != len(layer_sizes) - 2:
            model.append(nn.ReLU())
    
    return model

def get_avg_first_feasible_solution_time(times):
    avg_times = [time_list[0] for time_list in times]
    return np.mean(avg_times)

def get_centered_times(times):
    centered_times = []
    for time_list in times:
        t0 = time_list[0]
        centered_times.append([t - t0 for t in time_list])
    return centered_times

def get_avg_gap_over_time(gaps, times):
    # First, we need to align the gaps based on time. We can create a common time grid and interpolate the gaps.
    common_time_grid = np.linspace(0, max(max(times)), num=100)  # 100 points from 0 to max time
    avg_gaps = []
    
    for gap_list, time_list in zip(gaps, times):
        # Interpolate gaps at the common time grid
        interpolated_gaps = np.interp(common_time_grid, time_list, gap_list)
        avg_gaps.append(interpolated_gaps)
    
    # Now compute the average gap at each point in the common time grid
    avg_gaps = np.mean(avg_gaps, axis=0)
    
    return common_time_grid, avg_gaps

def get_time_to_reach_gap_threshold(gaps, times, threshold):
    """
    Compute avg, min, max time to get to a certain gap threshold (e.g., 5%) for both methods.
    """
    times_to_threshold = []
    for gap_list, time_list in zip(gaps, times):
        time_to_threshold = None
        for gap, time in zip(gap_list, time_list):
            if gap <= threshold:
                time_to_threshold = time
                break
        if time_to_threshold is not None:
            times_to_threshold.append(time_to_threshold)
    return times_to_threshold
