from typing import Optional

import numpy as np

from .eye import Eye
from .face_parts import FaceParts, FacePartsName


class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__(FacePartsName.FACE)
        # bbox has 2 vectors:
        # (x1, y1) --> upper right corner
        # (x2, y2) --> lower left corner
        self.bbox = bbox
        x1 = bbox[0, 0]
        x2 = bbox[1, 0]
        y1 = bbox[0, 1]
        y2 = bbox[1, 1]
        self.bbow_width = x2 - x1
        self.bbow_height = y2 - y1
        self.bbox_center = (x1 + self.bbow_width / 2, y1 + self.bbow_height / 2)
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])
