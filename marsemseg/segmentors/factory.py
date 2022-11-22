from .isegmentation import ISegmentation
from ..preprocessors import (
    IPreprocessor,
    BGR2RGBConverter,
    Normalizer,
    Resizer,
    Pipeline,
)
from .segmentors import Segmentor, TorchScriptSegmentor
import typing as ty
from dataclasses import dataclass


@dataclass
class SegmentorFactory:
    backend: str
    checkpoint_path: str
    config: ty.Optional[str] = None
    shape: ty.Optional[ty.Tuple[int, int]] = None
    device: ty.Optional[str] = None
    half: bool = False

    def create(self) -> ISegmentation:
        if self.backend == "interface":
            attrs = ["config", "shape", "device", "half"]
            for attr in attrs:
                if getattr(self, attr) is not None:
                    continue
                raise ValueError(
                    f"Missing required argument {attr} for interface backend"
                )

            segmentor = Segmentor(
                self.config, input_size=self.shape, device=self.device, half=self.half
            )
            segmentor.initialize(self.checkpoint_path)
        else:
            attrs = ["device"]
            for attr in attrs:
                if getattr(self, attr) is not None:
                    continue
                raise ValueError(
                    f"Missing required argument {attr} for interface backend"
                )
            segmentor = TorchScriptSegmentor(self.device, half=self.half)
            segmentor.initialize(self.checkpoint_path)
        return segmentor
