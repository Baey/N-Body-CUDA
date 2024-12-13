import json
import typer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from prettytable import PrettyTable
from matplotlib.animation import FuncAnimation

from nbody.backends import cpu, cuda
from nbody.frontends import headless
from nbody.helpers import Body

# app = typer.Typer()

# class Body:
#     def __init__(self, mass, position, velocity):
#         self.mass = mass
#         self.position = np.array(position, dtype=float)
#         self.velocity = np.array(velocity, dtype=float)

# def compute_forces(bodies):
#     forces = [np.zeros(3) for _ in bodies]
#     for i, body1 in enumerate(bodies):
#         for j, body2 in enumerate(bodies):
#             if i != j:
#                 r = body2.position - body1.position
#                 distance = np.linalg.norm(r)
#                 if distance > 0:
#                     force_magnitude = G * body1.mass * body2.mass / distance**2
#                     force_direction = r / distance
#                     forces[i] += force_magnitude * force_direction
#     return forces

# def update_bodies(bodies, forces):
#     for body, force in zip(bodies, forces):
#         acceleration = force / body.mass
#         body.velocity += acceleration * dt
#         body.position += body.velocity * dt

# def simulate_step(bodies):
#     forces = compute_forces(bodies)
#     update_bodies(bodies, forces)
#     return [body.position.copy() for body in bodies]

# def simulate_with_animation(bodies: Iterable[Body], steps: int) -> List:
    
#     positions = []
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(-2e11, 2e11)
#     ax.set_ylim(-2e11, 2e11)
#     ax.set_zlim(-2e11, 2e11)
    
#     # Calculate marker sizes based on mass
#     max_mass = max(body.mass for body in bodies)
#     marker_sizes = [max(5, 20 * (body.mass/max_mass)**(1/3)) for body in bodies]
#     colors = ['yellow', 'blue', 'gray']
#     scatters = [ax.plot([], [], [], 'o', markersize=size, color=colors[i%3])[0] 
#                 for i, size in enumerate(marker_sizes)]

#     # Set labels and title
#     ax.set_xlabel('X position (m)')
#     ax.set_ylabel('Y position (m)')
#     ax.set_zlabel('Z position (m)')
#     ax.set_title('N-Body Simulation')

#     def init():
#         for scatter in scatters:
#             scatter.set_data([], [])
#             scatter.set_3d_properties([])
#         return scatters

#     def update(frame):
#         forces = compute_forces(bodies)
#         update_bodies(bodies, forces)
#         for scatter, body in zip(scatters, bodies):
#             scatter.set_data([body.position[0]], [body.position[1]])
#             scatter.set_3d_properties([body.position[2]])
#         positions.append([body.position.copy() for body in bodies])
#         return scatters

#     ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=20)
#     plt.show()
#     return positions

# def simulate_no_animation(bodies, steps):
#     positions = []
#     for _ in range(steps):
#         positions.append(simulate_step(bodies))
#     return positions

# def plot_trajectories(positions):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     for i in range(len(positions[0])):
#         trajectory = np.array([pos[i] for pos in positions])
#         ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    
#     ax.set_xlabel('X position (m)')
#     ax.set_ylabel('Y position (m)')
#     ax.set_zlabel('Z position (m)')
#     ax.set_title('N-Body Simulation Trajectories')
#     plt.show()

# @app.command()
def run(backend_name: str, frontend_name: str, config_file: str | None=None, steps: int | None=None, dt: float | None=None):
    if config_file is None:
        logger.info(f'No config file provided, using default one')
        config_file = Path(__file__).parent / 'nbody_constant.json'
    with open(config_file) as f:
        logger.debug(f'Reading config file from {config_file}')
        config = json.load(f)
    
    if dt is not None:
        logger.debug(f"dt = {dt} passed as an argument therefore dt in config file will not be used")
        config['dt'] = dt
    else:
        try:
            steps = config['steps']
        except KeyError:
            logger.error(f"dt hasn't been found in {config_file}. Aborting")

    if steps is not None:
        logger.debug(f"steps = {steps} passed as an argument therefore steps in config file will not be used")
    else:
        try:
            steps = config['steps']
        except KeyError:
            logger.error(f"steps hasn't been found in {config_file}. Aborting")

    table = PrettyTable()
    table.field_names = ["Parameter", "Value", "Unit"]
    table.align = "l"
    table.add_row(["Gravitational constant (G)", f"{config['G']:.2e}", "m³ kg⁻¹ s⁻²"])
    table.add_row(["Time step (dt)", f"{config['dt']}", "s"])
    table.add_row(["Simulation steps", f"{steps}", "-"])
    print("\nSimulation Parameters:")
    print(table)

    table = PrettyTable()
    table.field_names = ["Backend", "Frontend"]
    table.align = "l"
    table.add_row([f"{backend_name}", f"{frontend_name}"])
    print("\nSimulation Pipeline:")
    print(table)

    logger.info('Starting simulation')

    if backend_name == 'cpu':
        backend = cpu.Backend(config=config)
    elif backend_name == 'cuda':
        raise NotImplementedError

    if frontend_name == 'headless':
        frontend = headless.Frontend(backend=backend)
    elif frontend_name == 'matplotlib':
        raise NotImplementedError
    
    bodies = [
        Body(2e30, [0, 0, 0], [0, 0, 0]),
        Body(6e24, [1.5e11, 0, 0], [0, 29.78e3, 0]),
        Body(7e22, [1.5e11 + 3.84e8, 0, 0], [0, 29.78e3 + 1.022e3, 0])
    ]
    trajectories = frontend.simulate(bodies, steps)
    frontend.plot_trajectories(trajectories)

# Example usage
if __name__ == "__main__":
    # app()
    logger.level('INFO')
    run('cpu', 'headless')
