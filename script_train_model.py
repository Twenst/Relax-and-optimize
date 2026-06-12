import os
from time import time
from cflinstance import CFLInstance
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import argparse
from dataset import Data, load_instance_and_solution


def compute_loss(thetas, instances: List[CFLInstance], y_true, config):
    fy_score = torch.dot(thetas.reshape(-1), y_true)
    idx = 0
    for instance in instances:
        # idx : idx + instance.n_facilities
        inst_thetas = thetas[idx: idx + instance.n_facilities].reshape(-1)
        
        # compute E[O.y(O)] where O is noised thetas and y is the solution of the model with those noised thetas
        esp = torch.tensor(0.0, dtype=torch.float32)
        
        for _ in range(config["n_rep"]):
            noised_thetas = inst_thetas + torch.randn_like(inst_thetas) * 0.2
            noised_thetas_arr = noised_thetas.detach().numpy()
            if config["milp_mode"]:
                thetaed_model = instance.get_solved_model_using_thetas(noised_thetas_arr, timeout=50e-3)
            else:
                thetaed_model = instance.get_solved_relaxation_using_thetas(noised_thetas_arr)

            _, y_vals = utils.parse_vars(thetaed_model.getVars(), instance.n_facilities, instance.n_clients)
            y_vals = torch.tensor([v.X for v in y_vals], dtype=torch.float32)
            esp = esp + torch.dot(noised_thetas, y_vals)
                    
        fy_score = fy_score - esp / config["n_rep"]
    
        idx += instance.n_facilities
    return fy_score/len(instances)

def epoch_pass(model, optimizer, data: Data, config):
    start_time = time()
    data.train.shuffle()
    total_loss = torch.tensor(0.0, dtype=torch.float32)
    n_batches = 0
    for batch in data.train.get_batches(config["batch_size"]):
        pred = model(batch.X)
        loss = compute_loss(pred, batch.instances, batch.y, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
        n_batches += 1
    total_loss /= n_batches    
    
    # validation loss
    with torch.no_grad():
        pred = model(data.val.X)
        val_loss = compute_loss(pred, data.val.instances, data.val.y, config)
    end_time = time()
    
    return total_loss, val_loss, end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configFilePath", nargs="?", type=str, default=None)
    args = parser.parse_args()
    
    print("Loading configuration...", end="")
    config = utils.load_config(args.configFilePath)
    print("Done.")
    print("Loading instances and solutions...", end="")
    
    data = load_instance_and_solution(utils.Constants.instancesDatasetPath, config, print_load_time=True)

    layer_sizes = config["hidden_dims"]
    layer_sizes.insert(0, data.train.X.shape[1])
    layer_sizes.append(1)
    
    model = nn.Sequential()
    for i in range(len(layer_sizes) - 1):
        model.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i != len(layer_sizes) - 2:
            model.append(nn.ReLU())

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    losses = []
    for epoch in range(config["num_epochs"]):
        loss, val_loss, time_took = epoch_pass(model, optimizer, data, config)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, took {time_took:.2f} seconds.")
        losses.append(loss.item())

    print("Training completed.")
    
    with torch.no_grad():
        pred_test = model(data.test.X)
        test_loss = compute_loss(pred_test, data.test.instances, data.test.y, config)
    print(f"Test Loss: {test_loss.item():.4f}")
    
    # save model
    os.makedirs(utils.Constants.savedModelsPath, exist_ok=True)
    configId = f"epochs{config['num_epochs']}_lr{config['learning_rate']}_hidden{config['hidden_dims']}"
    model_name = f"model_{configId}_{utils.time_to_date(int(time()))}.pth"
    torch.save(model.state_dict(), f"{utils.Constants.savedModelsPath}/{model_name}")