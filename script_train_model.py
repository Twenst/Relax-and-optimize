from cflinstance import CFLInstance
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def generate_dataset(instances, solutions):
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
    split_idx = int(np.floor(n_inst*0.8))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    
    x_train = torch.tensor(np.concatenate(x_train, axis=0), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate(y_train, axis=0), dtype=torch.float32)
    x_test = torch.tensor(np.concatenate(x_test, axis=0), dtype=torch.float32)
    y_test = torch.tensor(np.concatenate(y_test, axis=0), dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test

def compute_loss(thetas, instances: list[CFLInstance], y_true, n_rep = 15):
    fy_score = - torch.dot(thetas.reshape(-1), y_true)
    true_loss = torch.tensor(0.0, dtype=torch.float32)
    idx = 0
    for instance in instances:
        # idx jsqu'a idx + instance.n_facilities
        inst_thetas = thetas[idx: idx + instance.n_facilities].detach().numpy().reshape(-1)
        
        # compute E[O.y] where O is noised thetas and y is the solution of the model with those noised thetas
        esp = torch.tensor(0.0, dtype=torch.float32)
        
        for _ in range(n_rep):
            noised_thetas = inst_thetas + np.random.normal(0, 0.2, size=inst_thetas.shape)
            thetaed_model = instance.get_solved_model_using_thetas(-noised_thetas, timeout=25e-3)

            _, y_vals = utils.parse_vars(thetaed_model.getVars(), instance.n_facilities, instance.n_clients)
            y_vals = torch.tensor([v.X for v in y_vals], dtype=torch.float32)
            esp = esp + torch.dot(y_vals, torch.tensor(inst_thetas, dtype=torch.float32))
            true_loss = true_loss + ((y_vals - y_true[idx: idx + instance.n_facilities])**2).sum()
                    
        # esp = np.max(esp, axis=0)
        fy_score = fy_score + esp / n_rep
    
        idx += instance.n_facilities
    return fy_score, true_loss

if __name__ == "__main__":
    print("Loading instances and solutions...", end="")
    instances_and_sols = [CFLInstance.load_instance_and_solution(f"{utils.Constants.instancesDatasetPath}/instance_{i}.npz") for i in range(300)]
    instances = [inst["instance"] for inst in instances_and_sols]
    solutions = [sol["solution"] for sol in instances_and_sols]

    print("Done.")
    
    X_train, y_train, X_test, y_test = generate_dataset(instances, solutions)
    train_instances = instances[:int(np.floor(len(instances)*0.8))]
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    model = nn.Linear(X_train.shape[1], 1)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    print("Starting training...")
    # training loop
    losses = []
    for epoch in range(50):
        # for batch_X, batch_y in dataloader:
            pred = model(X_train)
            # print("pred", pred)
            loss, true_loss = compute_loss(pred, train_instances, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, True Loss: {true_loss.item():.4f}")
            losses.append(loss.item())

    print("Training completed.")
    print("final loss:", loss.item())