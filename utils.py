import numpy as np

class Constants:
    baseInstanceDataPath = "data/instances.npz"
    instancesDatasetPath = "data/dataset"

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
