from .preprocessors import Pipeline, BGR2RGBConverter, Normalizer, Resizer, Transposer
from .ipreprocessor import IPreprocessor

from dataclasses import dataclass
import typing as ty


@dataclass
class PipelineFactory:
    backend: str
    convert_to_rgb: bool = True
    mean: ty.Optional[ty.Tuple[float, float, float]] = None
    std: ty.Optional[ty.Tuple[float, float, float]] = None
    shape: ty.Optional[ty.Tuple[int, int]] = None

    def create(self) -> IPreprocessor:
        steps: ty.List[IPreprocessor] = (
            [BGR2RGBConverter()] if self.convert_to_rgb else []
        )
        if self.backend == "interface":
            return Pipeline(steps)
        else:
            for attr in ["mean", "std", "shape"]:
                if getattr(self, attr) is not None:
                    continue
                raise ValueError(
                    f"Missing required argument {attr} for backend: {self.backend}"
                )

            return Pipeline(
                steps
                + [
                    Resizer(self.shape),
                    Normalizer(self.mean, self.std),
                    # TorchScript model expects RGB image in (b, c h, w) format
                    Transposer(rearrange="b h w c -> b c h w"),
                ]
            )
