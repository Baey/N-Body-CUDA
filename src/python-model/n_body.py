import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prettytable import PrettyTable
from typing import Iterable, List, Union
from matplotlib.animation import FuncAnimation


class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

def compute_forces(bodies):
    forces = [np.zeros(3) for _ in bodies]
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                r = body2.position - body1.position
                distance = np.linalg.norm(r)
                if distance > 0:
                    force_magnitude = G * body1.mass * body2.mass / distance**2
                    force_direction = r / distance
                    forces[i] += force_magnitude * force_direction
    return forces

def update_bodies(bodies, forces):
    for body, force in zip(bodies, forces):
        acceleration = force / body.mass
        body.velocity += acceleration * dt
        body.position += body.velocity * dt

def simulate_step(bodies):
    forces = compute_forces(bodies)
    update_bodies(bodies, forces)
    return [body.position.copy() for body in bodies]

def simulate_with_animation(bodies: Iterable[Body], steps: int) -> List:
    
    positions = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    ax.set_zlim(-2e11, 2e11)
    
    # Calculate marker sizes based on mass
    max_mass = max(body.mass for body in bodies)
    marker_sizes = [max(5, 20 * (body.mass/max_mass)**(1/3)) for body in bodies]
    colors = ['yellow', 'blue', 'gray']
    scatters = [ax.plot([], [], [], 'o', markersize=size, color=colors[i%3])[0] 
                for i, size in enumerate(marker_sizes)]

    # Set labels and title
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title('N-Body Simulation')

    def init():
        for scatter in scatters:
            scatter.set_data([], [])
            scatter.set_3d_properties([])
        return scatters

    def update(frame):
        forces = compute_forces(bodies)
        update_bodies(bodies, forces)
        for scatter, body in zip(scatters, bodies):
            scatter.set_data([body.position[0]], [body.position[1]])
            scatter.set_3d_properties([body.position[2]])
        positions.append([body.position.copy() for body in bodies])
        return scatters

    ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=20)
    plt.show()
    return positions

def simulate_no_animation(bodies, steps):
    positions = []
    for _ in range(steps):
        positions.append(simulate_step(bodies))
    return positions

def plot_trajectories(positions):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(len(positions[0])):
        trajectory = np.array([pos[i] for pos in positions])
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title('N-Body Simulation Trajectories')
    plt.show()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-body simulation')
    parser.add_argument('--animate', action='store_true', help='Show animation')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to simulate')
    parser.add_argument('--config', type=Union[str, None], default=None, help='Path to config file')
    parser.add_argument('--dt', type=float, default=3600, help='Time step in seconds')
    args = parser.parse_args()

    if args.config is None:
        config_path = Path(__file__).parent.parent / 'nbody_constant.json'
    else:
        config_path = args.config
    with open(config_path) as f:
        constants = json.load(f)
    
    G = constants['G']
    dt = args.dt if args.dt else constants['dt']
    steps = args.steps if args.steps else constants['steps']

    table = PrettyTable()
    table.field_names = ["Parameter", "Value", "Unit"]
    table.align = "l"
    table.add_row(["Gravitational constant (G)", f"{G:.2e}", "m³ kg⁻¹ s⁻²"])
    table.add_row(["Time step (dt)", f"{dt}", "s"])
    table.add_row(["Simulation steps", f"{steps}", "-"])
    print("\nSimulation Parameters:")
    print(table)
    
    bodies = [
        Body(2e30, [0, 0, 0], [0, 0, 0]),
        Body(6e24, [1.5e11, 0, 0], [0, 29.78e3, 0]),
        Body(7e22, [1.5e11 + 3.84e8, 0, 0], [0, 29.78e3 + 1.022e3, 0])
    ]
    
    if args.animate:
        positions = simulate_with_animation(bodies, steps)
    else:
        positions = simulate_no_animation(bodies, steps)
    
    plot_trajectories(positions)
