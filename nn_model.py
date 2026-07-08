import numpy as np
import utils
import torch
import torch.nn as nn
from typing import List
from time import time
from cflinstance import CFLInstance
from dataset import Data


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        
        self.model = None

    def forward(self, x):
        return self.model(x)

    def loss(self, thetas, instances: List[CFLInstance], y_true, config):
        """
        Computes the loss for the given thetas and instances.
        The loss is a Fenchel-Young loss over the y values obtained from solving with the perturbed thetas.
        """
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

                thetaed_model = instance.get_solved_relaxation_using_thetas(noised_thetas_arr)

                _, y_vals = utils.parse_vars(thetaed_model.getVars(), instance.n_facilities, instance.n_clients)
                y_vals = torch.tensor([v.X for v in y_vals], dtype=torch.float32)
                esp = esp + torch.dot(noised_thetas, y_vals)
                        
            fy_score = fy_score - esp / config["n_rep"]
        
            idx += instance.n_facilities
        return fy_score/len(instances)
    
    def validation_loss(self, thetas, instances: List[CFLInstance], y_true):
        '''Same as compute_loss but without the noise perturbation. Result is not differentiable.'''
    
        fy_score = torch.dot(thetas.reshape(-1), y_true)
        idx = 0
        for instance in instances:
            # idx : idx + instance.n_facilities
            inst_thetas = thetas[idx: idx + instance.n_facilities].reshape(-1).detach().numpy()
            
            thetaed_model = instance.get_solved_relaxation_using_thetas(inst_thetas)

            _, y_vals = utils.parse_vars(thetaed_model.getVars(), instance.n_facilities, instance.n_clients)
            y_vals = np.array([v.X for v in y_vals], dtype=np.float32)
                        
            fy_score = fy_score - np.dot(inst_thetas, y_vals)
        
            idx += instance.n_facilities
        return fy_score/len(instances)
    
    def epoch_pass(self, optimizer, data: Data, config):
        """
        Performs one epoch of training on the model using the provided optimizer and data.
        Returns the average training loss, validation loss, and time taken for the epoch.
        """
        start_time = time()
        data.train.shuffle()
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        n_batches = 0
        for batch in data.train.get_batches(config["batch_size"]):
            pred = self.forward(batch.X)
            loss = self.loss(pred, batch.instances, batch.y, config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            n_batches += 1
        total_loss /= n_batches    
        
        # validation loss
        with torch.no_grad():
            pred = self.forward(data.val.X)
            val_loss = self.validation_loss(pred, data.val.instances, data.val.y)
        end_time = time()
        
        return total_loss, val_loss, end_time - start_time

class LinearNNModel(NNModel):
    def __init__(self, hidden_layer_sizes, n_input_features):
        super(LinearNNModel, self).__init__()
        
        hidden_layer_sizes.insert(0, n_input_features)
        hidden_layer_sizes.append(1)
        
        self.model = nn.Sequential()
        for i in range(len(hidden_layer_sizes) - 1):
            self.model.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            if i != len(hidden_layer_sizes) - 2:
                self.model.append(nn.ReLU())

class GNNModel(NNModel):
    def __init__(self):
        # TODO
        pass
