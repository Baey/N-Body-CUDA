import numpy as np


def read_initial_conditions(filename):
    """Read initial conditions from a text file."""
    with open(filename, "r") as file:
        bodies = []
        for line in file:
            data = list(map(float, line.split()))
            bodies.append(
                {
                    "mass": data[0],
                    "radius": data[1],
                    "position": np.array(data[2:5]),
                    "velocity": np.array(data[5:]),
                }
            )
    return bodies


def save_trajectories(filename, trajectories):
    """Save trajectories to a text file."""
    with open(filename, "w") as file:
        for step in range(len(next(iter(trajectories.values())))):
            line = []
            for i in trajectories:
                position = trajectories[i][step]
                line.extend(position)
            file.write(" ".join(map(str, line)) + "\n")


def read_trajectories(filename):
    """Read trajectories from a text file."""
    trajectories = {}
    with open(filename, "r") as file:
        for line_num, line in enumerate(file):
            data = list(map(float, line.split()))
            step_positions = np.array(data).reshape(-1, 3)
            for i, position in enumerate(step_positions):
                if i not in trajectories:
                    trajectories[i] = []
                trajectories[i].append(position)
    return trajectories
