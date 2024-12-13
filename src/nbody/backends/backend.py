import numpy as np
from abc import ABC, abstractmethod

class Backend(ABC):

    def __init__(self, device: str, config: dict):
        super().__init__()

        self.device = device
        self.config = config
    
    @abstractmethod
    def _compute_forces(self) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _update_bodies(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    