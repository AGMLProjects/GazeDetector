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
        self.x1 = bbox[0, 0]
        self.x2 = bbox[1, 0]
        self.y1 = bbox[0, 1]
        self.y2 = bbox[1, 1]
        self.bbox_width = self.x2 - self.x1
        self.bbox_height = self.y2 - self.y1
        self.bbox_center = (self.x1 + self.bbox_width / 2, self.y1 + self.bbox_height / 2)
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])
