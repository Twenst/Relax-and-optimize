import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from utils import Constants, parse_vars


class CFLInstance:
    def __init__(
        self,
        n_facilities,
        n_clients,
        rho=0.8,
        seed=42,
        generate_new_data=True,
    ):
        self.n_facilities = n_facilities
        self.n_clients = n_clients
        self.rho = rho
        self.seed = seed
        if generate_new_data:
            self.data = self.generate_instance()
            self.init_model()
            self.status = "unsolved"
        else:
            self.status = "uninitialized"

    def copy_instance_data(self):
        data = {
            "transport": self.data["transport"].copy(),
            "demands": self.data["demands"].copy(),
            "capacities": self.data["capacities"].copy(),
            "opening_costs": self.data["opening_costs"].copy(),
            "facilities": self.data["facilities"].copy(),
            "clients": self.data["clients"].copy(),
        }

        new_instance = CFLInstance(
            self.n_facilities,
            self.n_clients,
            self.rho,
            self.seed,
            generate_new_data=False
        )
        new_instance.data = data
        new_instance.init_model()
        return new_instance

    def generate_instance(self):
        rng = np.random.default_rng(self.seed)

        instancesData = np.load(Constants.baseInstanceDataPath)
        selected_cli_idx = rng.choice(len(instancesData["cli_pos"]), size=self.n_clients, replace=False)
        selected_fac_idx = rng.choice(len(instancesData["fac_pos"]), size=self.n_facilities, replace=False)
        fac_pos = instancesData["fac_pos"][selected_fac_idx]
        cli_pos = instancesData["cli_pos"][selected_cli_idx]

        # Euclidian costs
        diff = fac_pos[:, None, :] - cli_pos[None, :, :]
        transport = np.linalg.norm(diff, axis=2)

        # Demands and capacities
        demands = instancesData["demands"][selected_cli_idx]
        total_demand = demands.sum()

        capacities = instancesData["capacities"][selected_fac_idx]
        total_capacity = capacities.sum()

        capacities *= total_demand / (total_capacity * self.rho)

        # Opening costs
        opening = instancesData["opening_costs"][selected_fac_idx]
        opening *= total_demand / (total_capacity * self.rho)

        return {
            "transport": transport,
            "demands": demands,
            "capacities": capacities,
            "opening_costs": opening,
            "facilities": fac_pos,
            "clients": cli_pos,
        }

    def init_model(self):
        self.m = gp.Model("CFL")

        self.x = self.m.addVars(
            self.n_facilities, self.n_clients, vtype=gp.GRB.CONTINUOUS, name="x"
        )
        self.y = self.m.addVars(self.n_facilities, vtype=gp.GRB.BINARY, name="y")

        self.m.setObjective(
            gp.quicksum(
                self.data["transport"][i, j] * self.x[i, j]
                for i in range(self.n_facilities)
                for j in range(self.n_clients)
            )
            + gp.quicksum(
                self.data["opening_costs"][i] * self.y[i]
                for i in range(self.n_facilities)
            ),
            gp.GRB.MINIMIZE,
        )

        for j in range(self.n_clients):
            self.m.addConstr(
                gp.quicksum(self.x[i, j] for i in range(self.n_facilities))
                == self.data["demands"][j],
                name=f"eqx{j}",
            )
            for i in range(self.n_facilities):
                self.m.addConstr(self.x[i, j] <= self.y[i], name=f"x{i},{j}lty{i}")

        for i in range(self.n_facilities):
            self.m.addConstr(
                gp.quicksum(
                    self.data["demands"][j] * self.x[i, j]
                    for j in range(self.n_clients)
                )
                <= self.data["capacities"][i] * self.y[i],
                name=f"demand{i}",
            )

    def plot_instance(self):
        plt.scatter(
            self.data["facilities"][:, 0],
            self.data["facilities"][:, 1],
            c="red",
            label="Facilities",
        )
        for i in range(self.n_facilities):
            plt.annotate(
                str(i),
                (self.data["facilities"][i, 0], self.data["facilities"][i, 1]),
                fontsize=8,
            )
        plt.scatter(
            self.data["clients"][:, 0],
            self.data["clients"][:, 1],
            c="blue",
            label="Clients",
        )
        plt.legend()
        plt.title("CFL Instance")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        plt.show()

    def plot_solution(self):
        if self.status == "unsolved":
            print("Can't plot solution as the instance hasn't been solved yet!")
            return
        for i in range(self.n_facilities):
            for j in range(self.n_clients):
                if self.x[i, j].x > 0:
                    plt.plot(
                        [self.data["facilities"][i, 0], self.data["clients"][j, 0]],
                        [self.data["facilities"][i, 1], self.data["clients"][j, 1]],
                        "gray",
                        alpha=0.5,
                    )

        filtered_fac_idx = [i for i in range(self.n_facilities) if self.y[i].x > 0]
        plt.scatter(
            self.data["facilities"][filtered_fac_idx, 0],
            self.data["facilities"][filtered_fac_idx, 1],
            c="red",
            label="Facilities",
        )
        for i in filtered_fac_idx:
            plt.annotate(
                str(i),
                (self.data["facilities"][i, 0], self.data["facilities"][i, 1]),
                fontsize=8,
            )
        plt.scatter(
            self.data["clients"][:, 0],
            self.data["clients"][:, 1],
            c="blue",
            label="Clients",
        )

        plt.legend()
        plt.title("CFL Solution")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid()
        plt.show()

    def solve(self, timeout=60, gap=1e-4):
        """
        Solve the CFL instance. Returns the callback values and times for plotting the convergence curve.
        """
        self.m.Params.OutputFlag = 0
        self.m.Params.MIPGap = gap
        self.m.Params.TimeLimit = timeout
        self.m.Params.Cuts = 2

        callback_vals = []
        callback_times = []
        
        self.m.reset()  # Reset the model to clear any previous optimization results

        def callback(model, where):
            if where == gp.GRB.Callback.MIP:
                objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                callback_vals.append([objbst, objbnd])
                callback_times.append(model.cbGet(gp.GRB.Callback.RUNTIME))

        self.m.optimize(callback=callback)
        self.status = "solved"
        return callback_vals, callback_times

    def unnoised_objective(self):
        return sum(
            self.data["transport"][i, j] * self.x[i, j].x
            for i in range(self.n_facilities)
            for j in range(self.n_clients)
        ) + sum(
            self.data["opening_costs"][i] * self.y[i].x
            for i in range(self.n_facilities)
        )

    def get_relaxation(self, noise_level=False):
        y_noise = np.zeros(self.n_facilities)

        if noise_level:
            rng = np.random.default_rng(self.seed)
            y_noise = rng.normal(0, noise_level, self.n_facilities)

        return self.get_perturbed_relaxation(y_noise)
    
    def get_perturbed_relaxation(self, thetas):
        self.m.update()
        relaxation = self.m.relax()
        relaxation.update()
        
        relaxation.setObjective(
            gp.quicksum(
                (self.data["transport"][i, j])
                * relaxation.getVarByName(f"x[{i},{j}]")
                for i in range(self.n_facilities)
                for j in range(self.n_clients)
            )
            + gp.quicksum(
                (self.data["opening_costs"][i] + thetas[i])
                * relaxation.getVarByName(f"y[{i}]")
                for i in range(self.n_facilities)
            ),
            gp.GRB.MINIMIZE,
        )
        return relaxation

    def init_warm_start(self, x_vals, y_vals):
        for i in range(self.n_facilities):
            for j in range(self.n_clients):
                self.x[i, j].start = x_vals[i, j].x
            self.y[i].start = y_vals[i].x

    # Features
    def compute_features(self):
        k = self.n_clients // self.n_facilities
        features = {
            "capacities" : self.data["capacities"],
            "opening_costs" : self.data["opening_costs"],            
            "avg_distance_to_k_nearest_clients": self.mean_distance_to_k_nearest_clients(k=k),
            "avg_distance_to_k_nearest_facilities": self.mean_distance_to_k_nearest_facilities(k=k),
            "avg_demands_of_k_nearest_clients": self.mean_demands_of_k_nearest_clients(k=k),
            "avg_capacity_of_k_nearest_facilities": self.mean_capacity_of_k_nearest_facilities(k=k),
            "avg_distance_to_all_clients": self.mean_distance_to_k_nearest_clients(k=self.n_clients),
            "avg_distance_to_all_facilities": self.mean_distance_to_k_nearest_facilities(k=self.n_facilities-1),
            "avg_demands_of_all_clients": self.mean_demands_of_k_nearest_clients(k=self.n_clients),
            "avg_capacity_of_all_facilities": self.mean_capacity_of_k_nearest_facilities(k=self.n_facilities-1),
            "distance_to_nearest_client": self.mean_distance_to_k_nearest_clients(k=1),
            "distance_to_nearest_facility": self.mean_distance_to_k_nearest_facilities(k=1),
            "capacity_of_nearest_facility": self.mean_capacity_of_k_nearest_facilities(k=1),
            "demands_of_nearest_client": self.mean_demands_of_k_nearest_clients(k=1),
            "nearest_client_count": self.nearest_client_count(),
        }
        return features
    
    def mean_distance_to_k_nearest_clients(self, k=3):
        sorted_distances = np.sort(self.data["transport"], axis=1)
        return sorted_distances[:, :k].mean(axis=1)
    
    def mean_distance_to_k_nearest_facilities(self, k=3):
        fac_pos = self.data["facilities"]
        distances = np.linalg.norm(fac_pos[:, None, :] - fac_pos[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        
        return np.sort(distances, axis=1)[:, :k].mean(axis=1)
    
    def mean_demands_of_k_nearest_clients(self, k=3):
        sorted_indices = np.argsort(self.data["transport"], axis=1)
        return np.mean(self.data["demands"][sorted_indices[:, :k]], axis=1)
    
    def mean_capacity_of_k_nearest_facilities(self, k=3):
        fac_pos = self.data["facilities"]
        capacities = self.data["capacities"]
        
        distances = np.linalg.norm(fac_pos[:, None, :] - fac_pos[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        indexes = np.argsort(distances, axis=1)[:, :k]
        
        return capacities[indexes].mean(axis=1)

    def nearest_client_count(self):
        sorted_indices = np.argmin(self.data["transport"], axis=0)
        return np.bincount(sorted_indices, minlength=self.n_facilities)
    
    def save_instance(self, path):
        if self.status == "solved":
            np.savez(
                path,
                facilities=self.data["facilities"],
                clients=self.data["clients"],
                transport=self.data["transport"],
                opening_costs=self.data["opening_costs"],
                demands=self.data["demands"],
                capacities=self.data["capacities"],
                solved=True,
                x=np.array([[self.x[i, j].x for j in range(self.n_clients)] for i in range(self.n_facilities)]),
                y=np.array([self.y[i].x for i in range(self.n_facilities)]),
            )
        else:
            np.savez(
                path,
                facilities=self.data["facilities"],
                clients=self.data["clients"],
                transport=self.data["transport"],
                opening_costs=self.data["opening_costs"],
                demands=self.data["demands"],
                capacities=self.data["capacities"],
                solved=False,
            )

    @staticmethod
    def load_instance_and_solution(path):
        data = np.load(path)
        inst = CFLInstance(
            n_facilities=len(data["facilities"]),
            n_clients=len(data["clients"]),
            seed=0,
            generate_new_data=False
        )
        inst.data = {
            "facilities": data["facilities"],
            "clients": data["clients"],
            "transport": data["transport"],
            "opening_costs": data["opening_costs"],
            "demands": data["demands"],
            "capacities": data["capacities"],
        }
        inst.init_model()
        inst.status = "solved" if data["solved"] else "unsolved"
        if data["solved"]:
            for i in range(inst.n_facilities):
                for j in range(inst.n_clients):
                    inst.x[i, j].start = data["x"][i, j]
                inst.y[i].start = data["y"][i]
        return {"instance": inst, "solution": {"x": data["x"], "y": data["y"]} if data["solved"] else None}
    
    def get_solved_model_using_thetas(self, thetas, timeout=60):
        perturbed_instance = self.get_perturbed_relaxation(thetas)

        perturbed_instance.Params.OutputFlag = 0
        perturbed_instance.Params.TimeLimit = 60
        perturbed_instance.optimize()
        
        
        # warmed_inst = self.copy_instance_data()
        x_vals, y_vals = parse_vars(
            perturbed_instance.getVars(), self.n_facilities, self.n_clients
        )
        self.init_warm_start(x_vals=x_vals, y_vals=y_vals)
        
        self.solve(timeout=timeout, gap=1e-9)
        
        return self.m

    def get_solved_relaxation_using_thetas(self, thetas):
        perturbed_instance = self.get_perturbed_relaxation(thetas)
        perturbed_instance.Params.OutputFlag = 0
        perturbed_instance.optimize()
        return perturbed_instance