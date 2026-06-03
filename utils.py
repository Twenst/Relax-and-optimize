import json
import os
import numpy as np


class Constants:
    baseInstanceDataPath = "data/instances.npz"
    instancesDatasetPath = "data/dataset"
    configFilePath = "config.json"
    savedModelsPath = "saved_models"

    defaultConfig = {
        "learning_rate": 0.0001,
        "num_epochs": 10,
        "batch_size": 10,
        "hidden_size": 64,
        "n_rep": 10,
        "n_instances": 150,
        "train_size": 0.6,
        "test_size": 0.2,
        "timeout": 50e-3,
        "hidden_dims": [],
        "milp_mode": True,
    }


def parse_vars(vars_vect, n_facilities, n_clients):
    x = vars_vect[0 : n_facilities * n_clients]
    y = vars_vect[n_facilities * n_clients :]
    x = np.reshape(x, (n_facilities, n_clients))
    return x, y


def compute_gaps_from_callback(callback_vals):
    callback_vals = np.array(callback_vals)
    gaps = np.abs(callback_vals[:, 0] - callback_vals[:, 1]) / np.abs(
        callback_vals[:, 0]
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