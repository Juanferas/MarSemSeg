import typing as ty
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class IPreprocessor(ABC):

    @abstractmethod
    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        raise NotImplementedError("Must be implemented in subclass")
