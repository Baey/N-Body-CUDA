import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Iterable
from matplotlib.animation import FuncAnimation
from nbody.frontends import Frontend
from nbody.frontends.headless import Frontend as HeadlessFrontend
from nbody.backends import Backend
from nbody.helpers import Body


class Frontend(Frontend):

    def __init__(self, backend: Backend, live: bool = False):
        super().__init__(backend)
        self.live = live
    
    def simulate(self, bodies: Iterable[Body], steps: int):

        def init():
            for scatter in scatters:
                scatter.set_data([], [])
                scatter.set_3d_properties([])
            return scatters

        def update(frame):
            time_step = trajectories[frame, ...]
            for scatter, body in zip(scatters, time_step):
                scatter.set_data([body[0]], [body[1]])
                scatter.set_3d_properties([body[2]])
            return scatters
        
        headless_frontend = HeadlessFrontend(backend=self.backend)
        trajectories = headless_frontend.simulate(bodies, steps)
        n = len(bodies)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-2e11, 2e11)
        ax.set_ylim(-2e11, 2e11)
        ax.set_zlim(-2e11, 2e11)
        
        # Calculate marker sizes based on mass
        max_mass = max(body.mass for body in bodies)
        marker_sizes = [max(5, 20 * (body.mass/max_mass)**(1/3)) for body in bodies]
        colors = ['yellow', 'blue', 'gray', 'green', 'red', 'purple']
        scatters = [ax.plot([], [], [], 'o', markersize=size, color=colors[i%len(colors)])[0] 
                    for i, size in enumerate(marker_sizes)]

        # Set labels and title
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')
        ax.set_title(f'{n} Body Simulation')

        ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=1)
        plt.show()
        return trajectories

        # trajectories = np.zeros((steps, len(bodies), len(bodies[0].position)))
        # for i in tqdm(range(steps)):
        #     trajectories[i] = self.backend.step(bodies)
        # return trajectories