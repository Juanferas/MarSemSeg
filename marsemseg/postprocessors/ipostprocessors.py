from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class IPostProcessor(ABC):
    @abstractmethod
    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        raise NotImplementedError("Must be implemented in subclass")
