import numpy as np
import matplotlib.pyplot as plt


def plot_objective_progress(callback_times, callback_vals, skipped=0, label=None):
    callback_vals = np.array(callback_vals)
    plt.plot(
        callback_times[skipped:],
        callback_vals[skipped:, 0],
        "b+-",
        label="Best Integer Solution",
    )
    plt.plot(
        callback_times[skipped:], callback_vals[skipped:, 1], "r+-", label="Best Bound"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Objective Value")
    plt.title(
        f"{label}"
        if label
        else f"Gurobi Callback Progress (first {skipped} steps skipped)"
    )
    plt.legend()
    plt.grid()
    plt.show()
