import numpy as np
import os
from utils import read_initial_conditions, save_trajectories
import matplotlib.pyplot as plt


def compute_gravitational_force(body1, body2, G=6.67430e-11):
    """Compute the gravitational force exerted on body1 by body2."""
    r_vec = body2["position"] - body1["position"]
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)
    force_mag = G * body1["mass"] * body2["mass"] / r_mag**2
    return force_mag * r_vec / r_mag


def simulate_n_bodies(bodies, dt, steps):
    """Simulate n-body problem and return trajectories."""
    trajectories = {i: [body["position"].copy()] for i, body in enumerate(bodies)}

    for _ in range(steps):
        forces = [np.zeros(3) for _ in bodies]

        for i, body1 in enumerate(bodies):
            for j, body2 in enumerate(bodies):
                if i != j:
                    forces[i] += compute_gravitational_force(body1, body2)

        for i, body in enumerate(bodies):
            acceleration = forces[i] / body["mass"]
            body["velocity"] += acceleration * dt
            body["position"] += body["velocity"] * dt
            trajectories[i].append(body["position"].copy())

    return trajectories


# Example usage
input_filename = "initial_conditions.txt"
output_filename = "trajectories.txt"

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct full paths to the input and output files
input_filepath = os.path.join(script_dir, input_filename)
output_filepath = os.path.join(script_dir, output_filename)

dt = 3600  # Time step (seconds)
steps = 24 * 365 * 10  # Number of steps to simulate

# Read initial conditions
bodies = read_initial_conditions(input_filepath)

# Run simulation
trajectories = simulate_n_bodies(bodies, dt, steps)

# Save trajectories
save_trajectories(output_filepath, trajectories)

print(f"Trajectories saved to {output_filepath}")

# Generate 2D plot
plt.figure(figsize=(8, 8))
for i, traj in trajectories.items():
    traj = np.array(traj)  # Convert to numpy array
    plt.plot(traj[:, 0], traj[:, 1], label=f"Body {i}")

# Customize plot
plt.title("Trajectories of N-Body System")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.axis("auto")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig("trajectories_plot.png")
plt.show()
