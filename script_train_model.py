import os
from time import time
import utils
import numpy as np
import torch
import torch.optim as optim
import argparse
from dataset import load_instance_and_solution
from nn_model import LinearNNModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configFilePath", nargs="?", type=str, default=None)
    args = parser.parse_args()
    
    print("Loading configuration...", end="")
    config = utils.load_config(args.configFilePath)
    print("Done.")
    print("Loading instances and solutions...", end="")
    
    data = load_instance_and_solution(utils.Constants.instancesDatasetPath, config, print_load_time=True, keep_warm_start=False)

    layer_sizes = config["hidden_dims"]
    layer_sizes.insert(0, data.train.X.shape[1])
    layer_sizes.append(1)
    
    model = LinearNNModel(layer_sizes, data.train.X.shape[1])

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    losses = []
    best_model = None
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    for epoch in range(config["num_epochs"]):
        loss, val_loss, time_took = model.epoch_pass(optimizer, data, config)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, took {time_took:.2f} seconds.")
        losses.append(loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model = model.state_dict()
            best_val_loss_epoch = epoch
    print("Training completed.")
    
    print(f"Best validation loss achieved at epoch {best_val_loss_epoch+1} with val loss {best_val_loss:.4f}.")
    
    # save model
    os.makedirs(utils.Constants.savedModelsPath, exist_ok=True)
    configId = f"epochs{config['num_epochs']}_lr{config['learning_rate']}_hidden{config['hidden_dims']}"
    model_name = f"model_{configId}_{utils.time_to_date(int(time()))}.pth"
    torch.save(best_model, f"{utils.Constants.savedModelsPath}/{model_name}")
    
    # Evaluation of the model on the test set
    
    train_dataset = data.train

    regular_ws_objs = []
    regular_ws_bounds = []
    regular_ws_gaps = []
    regular_ws_times = []
    perturbed_ws_objs = []
    perturbed_ws_bounds = []
    perturbed_ws_gaps = []
    perturbed_ws_times = []

    for idx, inst in enumerate(train_dataset.instances):
        # Regular relaxation
        _, values1, time1 = inst.get_solved_model_using_thetas(np.zeros(inst.n_facilities), shift_times=False)
        
        # Perturbed relaxation
        thetas = model(train_dataset.x_i(idx)).detach().numpy().flatten()
        _, values2, time2 = inst.get_solved_model_using_thetas(thetas, shift_times=False)
        
        gap1 = utils.compute_gaps_from_callback(values1)
        gap2 = utils.compute_gaps_from_callback(values2)

        regular_ws_objs.append([v[0] for v in values1])
        regular_ws_bounds.append([v[1] for v in values1])
        regular_ws_gaps.append(gap1)
        regular_ws_times.append(time1)

        perturbed_ws_objs.append([v[0] for v in values2])
        perturbed_ws_bounds.append([v[1] for v in values2])
        perturbed_ws_gaps.append(gap2)
        perturbed_ws_times.append(time2)

    # Get centered times
    avg_first_time_gurobi = utils.get_avg_first_feasible_solution_time(regular_ws_times)
    centered_time_gurobi = utils.get_centered_times(regular_ws_times)
    avg_first_time_perturbed = utils.get_avg_first_feasible_solution_time(perturbed_ws_times)
    centered_time_perturbed = utils.get_centered_times(perturbed_ws_times)

    # Get average gap over time
    common_time_gurobi, avg_gap_gurobi = utils.get_avg_gap_over_time(regular_ws_gaps, centered_time_gurobi)
    common_time_perturbed, avg_gap_perturbed = utils.get_avg_gap_over_time(perturbed_ws_gaps, centered_time_perturbed)

    test_results = {}

    for threshold in [0.5, 0.25, 0.1, 0.05, 0.01]:
        regular_times_to_threshold = utils.get_time_to_reach_gap_threshold(regular_ws_gaps, regular_ws_times, threshold)
        perturbed_times_to_threshold = utils.get_time_to_reach_gap_threshold(perturbed_ws_gaps, perturbed_ws_times, threshold)

        test_results[f"{threshold*100}% Gap"] = {
            "regular": regular_times_to_threshold,
            "perturbed": perturbed_times_to_threshold
        }

        print(f"Time to reach {threshold*100}% gap:")
        print(f"Regular WS:         avg={np.mean(regular_times_to_threshold):.2f}s, min={np.min(regular_times_to_threshold):.2f}s, max={np.max(regular_times_to_threshold):.2f}s")
        print(f"Perturbed WS:       avg={np.mean(perturbed_times_to_threshold):.2f}s, min={np.min(perturbed_times_to_threshold):.2f}s, max={np.max(perturbed_times_to_threshold):.2f}s")
        print(f"Relative speedup (Regular / Perturbed): {np.mean(regular_times_to_threshold)/np.mean(perturbed_times_to_threshold):.2f}x")
        
    # first solution comparison between the two methods
    proportion_better_first_solution = sum(1 for p, g in zip(perturbed_ws_objs, regular_ws_objs) if p[0] < g[0]) / len(perturbed_ws_objs)
    print(f"Perturbed model found a better first solution than Regular WS: {(proportion_better_first_solution*100):.2f}% of the time.")
    
    # Avg time to reconstruct first solution
    avg_time_to_first_solution_regular_ws = utils.get_avg_first_feasible_solution_time(regular_ws_times)
    avg_time_to_first_solution_perturbed_ws = utils.get_avg_first_feasible_solution_time(perturbed_ws_times)
    print(f"Average time to reconstruct first solution:")
    print(f"Regular WS:     {avg_time_to_first_solution_regular_ws:.4f}s")
    print(f"Perturbed WS:   {avg_time_to_first_solution_perturbed_ws:.4f}s")
    print(f"Relative speedup (Regular / Perturbed): {(avg_time_to_first_solution_regular_ws/avg_time_to_first_solution_perturbed_ws):.2f}x")

    test_results["first_solution_comparison"] = {
        "proportion_better_first_solution": proportion_better_first_solution,
        "avg_time_to_first_solution_regular_ws": avg_time_to_first_solution_regular_ws,
        "avg_time_to_first_solution_perturbed_ws": avg_time_to_first_solution_perturbed_ws
    }

    np.savez(f"{utils.Constants.savedModelsPath}/test_results_{configId}_{utils.time_to_date(int(time()))}.npz",
             test_results=test_results)
