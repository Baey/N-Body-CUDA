import numpy as np
from typing import Iterable
from nbody.helpers import Body
from nbody.backends import Backend

class Backend(Backend):

    def __init__(self, config: dict):
        super().__init__(device='cpu', config=config)


    def _compute_forces(self, bodies: Iterable[Body]) -> np.ndarray:
        forces = np.zeros((len(bodies), len(bodies[0].position)), dtype=float)
        for i, body1 in enumerate(bodies):
            for j, body2 in enumerate(bodies):
                if i != j:
                    r = body2.position - body1.position
                    distance = np.linalg.norm(r)
                    if distance > 0:
                        force_magnitude = self.config['G'] * body1.mass * body2.mass / distance**2
                        force_direction = r / distance
                        forces[i] += force_magnitude * force_direction
        return forces
    
    def _update_bodies(self, bodies: Iterable[Body], forces: np.ndarray) -> None:
        for body, force in zip(bodies, forces):
            acceleration = force / body.mass
            body.velocity += acceleration * self.config['dt']
            body.position += body.velocity * self.config['dt']
    
    def step(self, bodies: Iterable[Body]) -> np.ndarray:
        forces = self._compute_forces(bodies)
        self._update_bodies(bodies, forces)
        return np.array([b.position for b in bodies])