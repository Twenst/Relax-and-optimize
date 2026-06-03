import numpy as np
import torch

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
        split_idx1 = int(np.floor(len(instances)*(1 - train_size)))
        split_idx2 = int(np.floor(len(instances)*(1 - train_size - test_size)))
        
        self.train = Dataset.init_from_solutions(instances[:split_idx1], solutions[:split_idx1])
        self.test = Dataset.init_from_solutions(instances[split_idx1:split_idx1+split_idx2], solutions[split_idx1:split_idx1+split_idx2])
        self.val = Dataset.init_from_solutions(instances[split_idx1+split_idx2:], solutions[split_idx1+split_idx2:])