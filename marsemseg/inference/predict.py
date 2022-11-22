from PIL import Image
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import os
import os.path as osp
import logging
import cv2
import matplotlib.pyplot as plt
import json
import torch
from dataclasses import dataclass
import typing as ty
from numpy.typing import NDArray
from tqdm import tqdm

from ..segmentors import ISegmentation
from ..preprocessors import Pipeline
from ..datasets import PALETTE, CLASSES


def predict_image(
    img: NDArray[np.float_],
    segmentor: ISegmentation,
    pipeline: Pipeline,
    shape: ty.Tuple[int, int],
    show: bool = False,
) -> NDArray[np.float_]:
    processed_frame = pipeline(img)
    preds = segmentor.segmentation(processed_frame).squeeze()
    mask = Image.fromarray(preds.astype(np.uint8)).convert("P")
    mask.putpalette(np.array(PALETTE).astype(np.uint8))
    predicted = np.asarray(mask.convert("RGB"))

    frame = np.asarray(img)
    # Resize to original size
    frame = cv2.resize(img, shape[::-1])
    predicted = cv2.resize(predicted, shape[::-1])
    # Add predicted and ground truth to frame
    overlaid = cv2.addWeighted(frame, 0.4, predicted, 0.7, 0)
    # convert to RGB
    overlaid = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
    if show:
        cv2.imshow("Predicted", overlaid)
    return overlaid


def predict_video(
    video: cv2.VideoCapture,
    output: cv2.VideoWriter,
    segmentor: ISegmentation,
    pipeline: Pipeline,
    shape: ty.Tuple[int, int],
    show: bool = False,
) -> NDArray[np.float_]:
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length, desc="Predicting...")
    result_frames = np.empty((length, *shape, 3))
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        frame: NDArray[np.float_]
        if not ret:
            break
        overlaid = predict_image(frame, segmentor, pipeline, shape)
        if show:
            cv2.imshow("frame", overlaid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        output.write(overlaid)
        result_frames[i] = overlaid
        pbar.update(1)

    video.release()
    output.release()
    if show:
        cv2.destroyAllWindows()
    print("Done")
    return result_frames
