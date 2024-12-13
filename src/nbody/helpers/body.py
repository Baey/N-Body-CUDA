import numpy as np
from typing import Union, Iterable

class Body:
    def __init__(self, mass: float, position: Iterable[float], velocity: Iterable[float]):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
    
    def __repr__(self):
        return f'M:{self.mass} P:{self.position}  V: {self.velocity}'