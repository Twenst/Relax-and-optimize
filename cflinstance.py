import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt


class CFLInstance:
    def __init__(
        self,
        n_facilities,
        n_clients,
        rho=0.8,
        seed=42,
        mode="uniform",
        noise_level=False,
    ):
        self.n_facilities = n_facilities
        self.n_clients = n_clients
        self.rho = rho
        self.seed = seed
        self.mode = mode
        self.data = self.generate_instance()
        self.init_model(noise_level)
        self.status = "unsolved"

    @staticmethod
    def copy_instance_data(instance, noise_level=False):
        data = {
            "transport": instance.data["transport"].copy(),
            "demands": instance.data["demands"].copy(),
            "capacities": instance.data["capacities"].copy(),
            "opening_costs": instance.data["opening_costs"].copy(),
            "facilities": instance.data["facilities"].copy(),
            "clients": instance.data["clients"].copy(),
        }

        new_instance = CFLInstance(
            instance.n_facilities,
            instance.n_clients,
            instance.rho,
            instance.seed,
            instance.mode,
            noise_level,
        )
        new_instance.data = data
        new_instance.init_model(noise_level)
        return new_instance

    def generate_instance(self):
        rng = np.random.default_rng(self.seed)

        match self.mode:
            case "uniform":
                fac_pos = rng.uniform(0, 1, (self.n_facilities, 2))
                cli_pos = rng.uniform(0, 1, (self.n_clients, 2))
            case "cluster":
                fac_pos = rng.uniform(0, 1, (self.n_facilities, 2))
                cli_pos = np.vstack(
                    [
                        fac_pos + rng.normal(0, 0.2, (self.n_facilities, 2))
                        for _ in range(self.n_clients // self.n_facilities + 1)
                    ]
                )[: self.n_clients]
            case _:
                raise ValueError(
                    "Invalid mode. Choose 'uniform', 'cluster', or 'hybrid'."
                )

        # Euclidian costs
        diff = fac_pos[:, None, :] - cli_pos[None, :, :]
        transport = np.linalg.norm(diff, axis=2)

        # Demands and capacities
        demands = rng.uniform(0, 1, self.n_clients)
        total_demand = demands.sum()

        capacities = rng.uniform(0, 1, self.n_facilities)
        total_capacity = capacities.sum()

        capacities *= total_demand / (total_capacity * self.rho)

        opening = capacities * rng.uniform(0.8, 1.2, self.n_facilities)

        return {
            "transport": transport,
            "demands": demands,
            "capacities": capacities,
            "opening_costs": opening,
            "facilities": fac_pos,
            "clients": cli_pos,
        }

    def init_model(self, noise_level=False):
        self.m = gp.Model("CFL")

        self.x = self.m.addVars(
            self.n_facilities, self.n_clients, vtype=gp.GRB.BINARY, name="x"
        )
        self.y = self.m.addVars(self.n_facilities, vtype=gp.GRB.BINARY, name="y")

        x_noise, y_noise = np.zeros((self.n_facilities, self.n_clients)), np.zeros(
            self.n_facilities
        )
        if noise_level:  # std if not falsy
            rng = np.random.default_rng(self.seed)
            x_noise = rng.normal(0, noise_level, x_noise.shape)
            y_noise = rng.normal(0, noise_level, y_noise.shape)

        self.m.setObjective(
            gp.quicksum(
                (self.data["transport"][i, j] + x_noise[i, j]) * self.x[i, j]
                for i in range(self.n_facilities)
                for j in range(self.n_clients)
            )
            + gp.quicksum(
                (self.data["opening_costs"][i] + y_noise[i]) * self.y[i]
                for i in range(self.n_facilities)
            ),
            gp.GRB.MINIMIZE,
        )

        for j in range(self.n_clients):
            self.m.addConstr(
                gp.quicksum(self.x[i, j] for i in range(self.n_facilities)) == 1,
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

    def solve(self, timeout=60):
        """
        Solve the CFL instance. Returns the callback values and times for plotting the convergence curve.
        """
        self.m.Params.OutputFlag = 0
        self.m.Params.MIPGap = 1e-4
        self.m.Params.TimeLimit = timeout
        self.m.Params.Cuts = 2

        callback_vals = []
        callback_times = []

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
