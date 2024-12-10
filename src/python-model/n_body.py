import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# Constants
G = 6.67430e-11  # Gravitational constant
dt = 3600  # Time step changed to 1 hour in seconds

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

def compute_forces(bodies):
    forces = [np.zeros(2) for _ in bodies]
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

def simulate_with_animation(bodies, steps):
    positions = []
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    
    # Calculate marker sizes based on mass
    max_mass = max(body.mass for body in bodies)
    marker_sizes = [max(5, 20 * (body.mass/max_mass)**(1/3)) for body in bodies]
    colors = ['yellow', 'blue', 'gray']
    scatters = [ax.plot([], [], 'o', markersize=size, color=colors[i])[0] 
                for i, size in enumerate(marker_sizes)]

    # Add grid and equal aspect ratio
    ax.grid(True)
    ax.set_aspect('equal')

    def init():
        for scatter in scatters:
            scatter.set_data([], [])
        return scatters

    def update(frame):
        forces = compute_forces(bodies)
        update_bodies(bodies, forces)
        for scatter, body in zip(scatters, bodies):
            scatter.set_data(body.position[0], body.position[1])
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
    for i in range(len(positions[0])):
        trajectory = np.array([pos[i] for pos in positions])
        plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('N-Body Simulation')
    plt.show()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-body simulation')
    parser.add_argument('--animate', action='store_true', help='Show animation')
    args = parser.parse_args()

    # Scale factor for better visualization
    scale = 1/1e3  # This will make distances more manageable
    
    bodies = [
        Body(2e30, [0, 0], [0, 0]),  # Sun at center
        # Earth-like orbit, scaled initial conditions
        Body(6e24, [1.5e11, 0], [0, 29.78e3]),  # Earth (orbital velocity ~29.78 km/s)
        # Moon-like orbit
        Body(7e22, [1.5e11 + 3.84e8, 0], [0, 29.78e3 + 1.022e3])  # Moon
    ]
    
    steps = 10000
    if args.animate:
        positions = simulate_with_animation(bodies, steps)
    else:
        positions = simulate_no_animation(bodies, steps)
    
    plot_trajectories(positions)
