import sys
sys.path.append('../')

import os
from gurobipy import GRB
from cflinstance import CFLInstance
from utils import Constants

if __name__ == "__main__":
    os.makedirs(Constants.instancesDatasetPath, exist_ok=True)
    number_of_instances = 100
    for seed in range(number_of_instances):
        print(f"{seed}/{number_of_instances}")
        instance = CFLInstance(n_facilities=10, n_clients=50, seed=seed, rho=0.5)
        val, times = instance.solve()
        if instance.m.status == GRB.OPTIMAL:
            real_obj = instance.m.ObjVal
            bound = instance.m.ObjBound
            if bound is not None:
                gap = abs(real_obj - bound)
                print(f"Took {times[-1]:.2f} seconds, objective: {real_obj:.8f}, solver gap : {gap:.8f}")
            else:
                # Fall back to previously tracked values but note possible reasons for mismatch
                print(f"Took {times[-1]:.2f} seconds, objective: {real_obj:.8f}, tracked GAP value: {abs(val[-1][0] - val[-1][1]):.8f} (may differ from solver due to numerical/recording differences)")
            instance.save_instance(f"{Constants.instancesDatasetPath}/instance_{seed}.npz")
            
        else:
            print(f"Instance not solved to optimality, status code: {instance.m.status}")