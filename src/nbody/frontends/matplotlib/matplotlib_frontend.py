import numpy as np
from tqdm import tqdm
from typing import Iterable
from nbody.frontends import Frontend
from nbody.backends import Backend
from nbody.helpers import Body


class Frontend(Frontend):

    def __init__(self, backend: Backend):
        super().__init__(backend)
    
    def simulate(self, bodies: Iterable[Body], steps: int):
        trajectories = np.zeros((steps, len(bodies), len(bodies[0].position)))
        for i in tqdm(range(steps)):
            trajectories[i] = self.backend.step(bodies)
        return trajectories