from .ipreprocessor import IPreprocessor
import numpy as np
from numpy.typing import NDArray
import typing as ty
import cv2
import logging
from einops import rearrange


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline(IPreprocessor):
    def __init__(self, pipeline: ty.List[IPreprocessor]):
        self.pipeline = pipeline

    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        for preprocessor in self.pipeline:
            batch = preprocessor(batch)
        return batch


class Normalizer(IPreprocessor):
    def __init__(
        self, mean: ty.Tuple[float, float, float], std: ty.Tuple[float, float, float]
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        for ch in range(3):
            batch[..., ch] = (batch[..., ch] - self.mean[ch]) / self.std[ch]
        return batch


class BGR2RGBConverter(IPreprocessor):
    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        return batch[..., ::-1]


class Resizer(IPreprocessor):
    def __init__(self, size: ty.Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        if len(batch.shape) < 4:
            batch = np.expand_dims(batch, axis=0)
        B, H, W, C = batch.shape
        new_h, new_w = self.size
        # Dimensions of the new image
        # (B, new_h, new_w, C)
        ratio = new_w / float(W)
        new_h = int(H * ratio)

        batch_resized: NDArray[np.float_] = np.empty(
            (B, new_h, new_w, C), dtype=np.float_
        )
        logger.debug(f"Batch shape: {batch_resized.shape}")
        for i, img in enumerate(batch):
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Reshaped from {img.shape} -> {resized.shape}")
            batch_resized[i, ...] = resized
        return batch_resized


class Transposer(IPreprocessor):
    def __init__(self, rearrange: str = "b h w c -> b c h w") -> None:
        self.rearrange = rearrange

    def __call__(self, batch: NDArray[np.float_]) -> NDArray[np.float_]:
        return rearrange(batch, self.rearrange)
