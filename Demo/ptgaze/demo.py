import datetime
import pathlib
from typing import Optional
from matplotlib import pyplot as plt

import cv2
import numpy as np
from omegaconf import DictConfig

from common import Face, Visualizer
from gaze_estimator import GazeEstimator
from utils import get_3d_face_model


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera, face_model_3d.NOSE_INDEX)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.heatmap_width = self.config.demo.heatmap_width
        self.heatmap_height = self.config.demo.heatmap_height
        self.heatmap = np.zeros((self.heatmap_width, self.heatmap_height))

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break
            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        gender, age = (-1, -1)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)
            face_frame = self._get_normalized_face(image, face)
            # TODO: send face_frame to GraNet, receive gender and age
            gender, age = (1, 25)
            if not self.config.demo.use_camera:
                self._display_demographic_data(gender, age)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
            self._display_demographic_data(gender, age)
            self.window_camera_width = self.visualizer.image.shape[1]
            self.window_camera_height = self.visualizer.image.shape[0]

        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        print(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, 'f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        normalized = face.normalized_image
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _get_normalized_face(self, image, face):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[face.y1 - 20:face.y2 + 20, face.x1 - 20:face.x2 + 20]
        plt.imshow(image)
        plt.show()
        return image

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length

        self.visualizer.draw_3d_line(face.center, face.center + length * face.gaze_vector)
        # camera a 90
        # pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))

        window_width = self.visualizer.image.shape[1]
        window_height = self.visualizer.image.shape[0]
        x = window_width - face.bbox_center[0]
        y = face.bbox_center[1]
        print(f'[face center coordinates] ({x:.2f}, {y:.2f})')

        xv, yv, _ = -face.normalized_gaze_vector
        # normalization between [-1, 1]
        xv = ((xv + 0.5) * 2) - 1
        yv = ((yv + 0.5) * 2) - 1
        print(f'[gaze vector] ({xv:.2f}, {yv:.2f})')

        # working for heatmap dimension 200x200
        # x_projected = int((self.heatmap_width / 2) + xv * 100)
        # y_projected = int((self.heatmap_height / 2) - yv * 100)
        x_projected = int(x + xv * x)
        y_projected = int(y - yv * y)
        print(f'[heatmap coordinates] ({x_projected}, {y_projected})')
        self.heatmap[x_projected - 5:x_projected + 5, y_projected - 5: y_projected + 5] += 1

    def _display_demographic_data(self, gender: int, age: int) -> None:
        self.visualizer.show_demographic_data(gender, age)

    def show_heatmap(self):
        if self.config.demo.use_camera:
            plt.imshow(np.swapaxes(self.heatmap, 0, 1), cmap='hot', interpolation='nearest')
        else:
            plt.imshow(np.rot90(self.heatmap, 3, (0, 1)), cmap='hot', interpolation='nearest')
        plt.show()
