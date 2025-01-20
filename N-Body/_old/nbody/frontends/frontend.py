import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from nbody.backends import Backend

class Frontend(ABC):

    def __init__(self, backend: Backend):
        super().__init__()
        self.backend = backend
    
    @abstractmethod
    def simulate(self) -> np.ndarray:
        raise NotImplementedError
    
    def plot_trajectories(self, trajectories):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(trajectories.shape[1]):
            trajectory = trajectories[:, i, :]
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')
        ax.set_title('N-Body Simulation Trajectories')
        plt.show()
    
