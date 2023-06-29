from typing import Any

import cv2
import torchvision.transforms as T
from omegaconf import DictConfig


def create_transform(config: DictConfig) -> Any:
    return _create_mpiifacegaze_transform(config)


def _create_mpiifacegaze_transform(config: DictConfig) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    transform = T.Compose([
        T.Lambda(lambda x: cv2.resize(x, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224,
                                                     0.229]),  # BGR
    ])
    return transform
