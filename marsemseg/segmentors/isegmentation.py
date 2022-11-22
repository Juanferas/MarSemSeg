from abc import ABC, abstractmethod
import typing as ty
from pathlib import Path
from numpy.typing import NDArray
import numpy as np


InputSignature = NDArray[np.float_]
OutputSignature = NDArray[np.float_]

class ISegmentation(ABC):
    """
    Segmentation interface.
    """

    @abstractmethod
    def initialize(self, ckpt_file: ty.Union[str, Path]) -> None:
        raise NotImplementedError("Must be implemented in subclass")

    @abstractmethod
    def segmentation(self, batch: InputSignature) -> OutputSignature:
        raise NotImplementedError("Segmentation is not implemented")
