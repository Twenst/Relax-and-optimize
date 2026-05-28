from time import time

from cflinstance import CFLInstance
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

def generate_dataset(instances, solutions, config):
    x = []
    y = []
    
    for instance, solution in zip(instances, solutions):
        # for each facility, build an array with each of its associated features
        features = instance.compute_features()
              
        reshaped_features = np.array(list(features.values())).T
        y_solution = abs(solution["y"])
        x.append(reshaped_features)
        y.append(y_solution)
        
    n_inst = len(instances)
    split_idx = int(np.floor(n_inst*(1 - config["test_size"])))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    
    x_train = torch.tensor(np.concatenate(x_train, axis=0), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate(y_train, axis=0), dtype=torch.float32)
    x_test = torch.tensor(np.concatenate(x_test, axis=0), dtype=torch.float32)
    y_test = torch.tensor(np.concatenate(y_test, axis=0), dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test

def compute_loss(thetas, instances: List[CFLInstance], y_true, config):
    fy_score = torch.dot(thetas.reshape(-1), y_true)
    idx = 0
    for instance in instances:
        # idx jsqu'a idx + instance.n_facilities
        inst_thetas = thetas[idx: idx + instance.n_facilities].detach().numpy().reshape(-1)
        
        # compute E[O.y] where O is noised thetas and y is the solution of the model with those noised thetas
        esp = torch.tensor(0.0, dtype=torch.float32)
        
        for _ in range(config["n_rep"]):
            noised_thetas = inst_thetas + np.random.normal(0, 0.2, size=inst_thetas.shape)
            if config["milp_mode"]:
                thetaed_model = instance.get_solved_model_using_thetas(noised_thetas, timeout=50e-3)
            else:
                thetaed_model = instance.get_solved_relaxation_using_thetas(noised_thetas)

            _, y_vals = utils.parse_vars(thetaed_model.getVars(), instance.n_facilities, instance.n_clients)
            y_vals = torch.tensor([v.X for v in y_vals], dtype=torch.float32)
            esp = esp - torch.dot(y_vals, thetas[idx: idx + instance.n_facilities].reshape(-1))
                    
        # esp = np.max(esp, axis=0)
        fy_score = fy_score + esp / config["n_rep"]
    
        idx += instance.n_facilities
    return fy_score

def epoch_pass(model, optimizer, X_train, y_train, train_instances, config):
    start_time = time()
    pred = model(X_train)
    loss = compute_loss(pred, train_instances, y_train, config)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    end_time = time()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, took {end_time - start_time:.2f} seconds.")
    
    return loss


if __name__ == "__main__":
    print("Loading configuration...", end="")
    config = utils.load_config()
    print("Done.")
    print("Loading instances and solutions...", end="")
    start_time = time()
    instances_and_sols = []

    load_times = []
    for i in range(config["n_instances"]):
        start_load_time = time()
        instances_and_sols.append(CFLInstance.load_instance_and_solution(f"{utils.Constants.instancesDatasetPath}/instance_{i}.npz"))
        end_load_time = time()
        load_times.append(end_load_time - start_load_time)
        print(f"Loaded instance {i+1}/{config['n_instances']} (took {end_load_time - start_load_time:.4f} seconds)", end="\r")
        
    instances = [inst["instance"] for inst in instances_and_sols]
    solutions = [sol["solution"] for sol in instances_and_sols]
    end_time = time()
    
    print(f"Done. Time taken: {end_time - start_time:.2f} seconds, average load time: {np.mean(load_times):.4f} seconds.")

    X_train, y_train, X_test, y_test = generate_dataset(instances, solutions, config)
    train_instances = instances[:int(np.floor(len(instances)*(1 - config["test_size"])))]
    
    model = nn.Linear(X_train.shape[1], 1)

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    losses = []
    for epoch in range(config["num_epochs"]):
        loss = epoch_pass(model, optimizer, X_train, y_train, train_instances, config)
        losses.append(loss.item())

    print("Training completed.")
    print("final loss:", loss.item())