from time import time
import numpy as np
import torch
from cflinstance import CFLInstance

class Dataset:
    def __init__(self, instances, x, y):
        self.instances = instances
        self._x = x
        self._y = y

    @staticmethod
    def init_from_solutions(instances, solutions):
        instances = instances
        x = []
        y = []
    
        for instance, solution in zip(instances, solutions):
            # for each facility, build an array with each of its associated features
            features = instance.compute_features()
                
            reshaped_features = np.array(list(features.values())).T
            y_solution = abs(solution["y"])
            x.append(reshaped_features)
            y.append(y_solution)
        
        x = np.array(x)
        y = np.array(y)
        
        return Dataset(instances, x, y)

    @property
    def X(self):
        return torch.tensor(np.concatenate(self._x, axis=0), dtype=torch.float32)

    @property
    def y(self):
        return torch.tensor(np.concatenate(self._y, axis=0), dtype=torch.float32)
    
    def x_i(self, i):
        return torch.tensor(self._x[i], dtype=torch.float32)
    
    def y_i(self, i):
        return torch.tensor(self._y[i], dtype=torch.float32)

    def shuffle_instances(self):
        # shuffle the dataset while keeping the same order between X, y and insts
        perm = np.random.permutation(len(self.instances))
        
        self._x = self._x[perm, :]
        self._y = self._y[perm, :]
        self.instances = [self.instances[p] for p in perm]
        
    def shuffle_in_instance(self):
        # For each instance, shuffle the order of the facilities
        for i in range(len(self.instances)):
            perm = np.random.permutation(self.instances[i].n_facilities)
            self._x[i] = self._x[i][perm]
            self._y[i] = self._y[i][perm]
            
    def shuffle(self):
        self.shuffle_instances()
        self.shuffle_in_instance()

    def get_batches(self, batch_size):
        for i in range(0, len(self.instances), batch_size):
            yield Dataset(self.instances[i:i+batch_size], self._x[i:i+batch_size], self._y[i:i+batch_size])

class Data:
    def __init__(self, instances, solutions, config):
        train_size = config["train_size"]
        test_size = config["test_size"]
        split_idx1 = int(np.floor(len(instances)*train_size))
        split_idx2 = int(np.floor(len(instances)*test_size))
        
        self.train = Dataset.init_from_solutions(instances[:split_idx1], solutions[:split_idx1])
        self.test = Dataset.init_from_solutions(instances[split_idx1:split_idx1+split_idx2], solutions[split_idx1:split_idx1+split_idx2])
        self.val = Dataset.init_from_solutions(instances[split_idx1+split_idx2:], solutions[split_idx1+split_idx2:])
                
        
def load_instance_and_solution(file_path, config, print_load_time=False, keep_warm_start=True):
    instances_and_sols = []
    
    load_times = []
    start_time = time()
    for i in range(config["n_instances"]):
        start_load_time = time()
        instances_and_sols.append(CFLInstance.load_instance_and_solution(f"{file_path}/instance_{i}.npz"))
        end_load_time = time()
        load_times.append(end_load_time - start_load_time)
        if print_load_time:
            print(f"Loaded instance {i+1}/{config['n_instances']} (took {end_load_time - start_load_time:.4f} seconds)", end="\r")
        
    instances = [inst["instance"] for inst in instances_and_sols]
    if not keep_warm_start:
        for inst in instances:
            inst.discard_warm_start()
    solutions = [sol["solution"] for sol in instances_and_sols]
    end_time = time()
    
    if print_load_time:
        print(f"Done. Time taken: {end_time - start_time:.2f} seconds, average load time: {np.mean(load_times):.4f} seconds.")

    data = Data(instances, solutions, config)
    return data