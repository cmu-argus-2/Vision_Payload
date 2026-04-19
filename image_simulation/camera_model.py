"""
This module contains a class representing a model for a single camera on the satellite and a class managing all camera
models.
"""

from functools import cache

import numpy as np

from utils.config_utils import load_config


class CameraModel:
    """
    A class representing a model for a single camera on the satellite.
    This class contains the camera_name and the transformation from the camera frame to the body frame.
    It also contains methods for various computations related to the camera.

    The camera frame has an origin located at the optical center of the camera, with the x-axis pointing right, the
    y-axis pointing down, and the z-axis pointing out of the camera lens.

    The pixel coordinates have an origin located in the top-left corner of the image, with the u axis pointing right
    (aligned with the camera frame x-axis), and the v axis pointing down (aligned with the camera frame y-axis).

    Note that this means that the u axis corresponds to the image width and the v axis corresponds to the image height.
    Thus, to access the pixel values in an image with shape CameraModel.OUTPUT_SHAPE, you would do image[v, u, :].
    Nonetheless, pixel coordinates will still be stored as (u, v) in numpy arrays since this is the standard convention.
    """

    IMAGE_HEIGHT = 2592
    IMAGE_WIDTH = 4608
    RESOLUTION = (IMAGE_HEIGHT, IMAGE_WIDTH)
    NUM_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
    NUM_CHANNELS = 3
    # Note that this is the OUTPUT_SHAPE of both the image and the ray directions, since NUM_CHANNELS == NUM_DIMENSIONS
    OUTPUT_SHAPE = RESOLUTION + (NUM_CHANNELS,)
    DTYPE = np.uint8
    HORIZONTAL_FOV = np.deg2rad(66.1)

    def __init__(self, camera_name: str, body_R_camera: np.ndarray, t_body_to_camera: np.ndarray):
        """
        Initialize the simulation camera parameters

        Parameters:
            camera_name: The name of the camera.
            body_R_camera: A numpy array of shape (3, 3) representing the rotation matrix from body to camera frame.
            t_body_to_camera: A numpy array of shape (3,) representing the translation vector from body to camera frame,
                              in the body frame.
        """
        self.camera_name = camera_name
        self.body_R_camera = body_R_camera
        self.t_body_to_camera = t_body_to_camera

        # Apply @cache at the instance level in the constructor to ensure separate caches for each instance
        # and to avoid needing to call hash(self) in the cache implementation
        self.ray_directions_body = cache(self.ray_directions_body)

    @property
    def get_camera_name(self) -> str:
        """
        Get the name of the camera.

        Returns:
            The name of the camera.
        """
        return self.camera_name

    def get_camera_position(
        self, body_position: np.ndarray, frame_R_body: np.ndarray
    ) -> np.ndarray:
        """
        Get the camera position in the frame of interest.

        Parameters:
            body_position: A numpy array of shape (3,) representing the position of the body in the frame of interest.
            frame_R_body: A numpy array of shape (3, 3) representing the rotation matrix from the body frame to the
                          frame of interest.

        Returns:
            A numpy array of shape (3,) representing the position of the camera in the frame of interest.
        """
        return body_position + frame_R_body @ self.t_body_to_camera

    def get_camera_axis(self, frame_R_body: np.ndarray) -> np.ndarray:
        """
        Get the camera's boresight axis in the frame of interest.

        Parameters:
            frame_R_body: A numpy array of shape (3, 3) representing the rotation matrix from the body frame to the
                          frame of interest.

        Returns:
            A numpy array of shape (3,) representing the camera's boresight axis in the frame of interest.
        """
        return frame_R_body @ self.body_R_camera @ np.array([0, 0, 1])

    @staticmethod
    @cache
    def ray_directions_camera():
        """
        Generate ray directions in the camera frame for each pixel.

        Returns:
            A numpy array of shape CameraModel.OUTPUT_SHAPE consisting of ray directions in the camera frame for each
            pixel.
        """
        half_width = np.tan(CameraModel.HORIZONTAL_FOV / 2)
        half_height = half_width * (CameraModel.IMAGE_HEIGHT / CameraModel.IMAGE_WIDTH)

        x = np.linspace(-half_width, half_width, CameraModel.IMAGE_WIDTH)
        y = np.linspace(-half_height, half_height, CameraModel.IMAGE_HEIGHT)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)  # Assume unit depth

        # Stack and normalize ray directions
        ray_directions_cf = np.stack([xx, yy, zz], axis=-1)
        ray_directions_cf /= np.linalg.norm(ray_directions_cf, axis=-1, keepdims=True)

        return ray_directions_cf

    def ray_directions_body(self) -> np.ndarray:
        """
        Get the ray directions in the body frame for each pixel.

        Note that this method is dynamically wrapped with a cache in the constructor.

        :return: A numpy array of shape CameraModel.OUTPUT_SHAPE consisting of ray directions in the body frame for each
                 pixel.
        """
        ray_directions_body = CameraModel.ray_directions_camera() @ self.body_R_camera.T
        return ray_directions_body

    def pixel_to_bearing_unit_vector(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Converts pixel coordinates to bearing unit vectors in the body frame.
        Note that the output will be incorrect for any pixel coordinates outside the image bounds.

        Parameters:
            pixel_coords: An array of shape (N, 2) with pixel coordinates.

        Returns:
            A numpy array of shape (N, 3) with bearing unit vectors in the body frame.
        """
        # since it'll be cached anyway, we can just look up the desired values
        ray_directions_body = self.ray_directions_body()
        u, v = pixel_coords.T
        u = np.clip(u, 0, CameraModel.IMAGE_WIDTH - 1)
        v = np.clip(v, 0, CameraModel.IMAGE_HEIGHT - 1)

        # Get integer and fractional parts
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = np.minimum(u0 + 1, CameraModel.IMAGE_WIDTH - 1)
        v1 = np.minimum(v0 + 1, CameraModel.IMAGE_HEIGHT - 1)

        # Calculate interpolation weights
        wu = u - u0
        wv = v - v0

        # Perform bilinear interpolation
        vectors = (
            (1 - wu)[:, None] * (1 - wv)[:, None] * ray_directions_body[v0, u0]
            + wu[:, None] * (1 - wv)[:, None] * ray_directions_body[v0, u1]
            + (1 - wu)[:, None] * wv[:, None] * ray_directions_body[v1, u0]
            + wu[:, None] * wv[:, None] * ray_directions_body[v1, u1]
        )

        # Normalize the interpolated vectors to ensure they are unit vectors
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


class CameraModelManager:
    """
    A class managing all camera models.
    """

    CAMERA_NAMES = ["x+", "y+", "x-", "y-"]

    def __init__(self):
        self.camera_models = CameraModelManager.initialize_cameras()

    def __getitem__(self, camera_name: str) -> CameraModel:
        """
        Get the CameraModel object for the specified camera.

        Parameters:
            camera_name: The name of the camera.

        Returns:
            The CameraModel object for the specified camera.
        """
        self.validate_camera_name(camera_name)
        return self.camera_models[camera_name]

    def get_body_Rs_camera(self, camera_names: np.ndarray) -> np.ndarray:
        """
        Get the rotation matrices from camera to body frame for the specified cameras.

        Parameters:
            camera_names: A numpy array of shape (N,) containing the names of the cameras. Repeats are fine.

        Returns:
            A numpy array of shape (N, 3, 3) containing the rotation matrices from camera to body frame
            for the specified cameras.
        """
        return np.stack([self[camera_name].body_R_camera for camera_name in camera_names], axis=0)

    @staticmethod
    def initialize_cameras() -> dict[str, CameraModel]:
        """
        Initialize camera models for all cameras.

        Returns:
            dict: A dictionary mapping camera names to CameraModel objects.
        """
        camera_models = {}
        for camera_info in load_config()["satellite"]["cameras"]:
            camera_models[camera_info["name"]] = CameraModel(
                camera_info["name"],
                np.array(camera_info["body_R_camera"]),
                np.array(camera_info["t_body_to_camera"]),
            )
        return camera_models

    @staticmethod
    def validate_camera_name(camera_name: str) -> None:
        """
        Validate that the camera name is one of the valid camera names.

        :param camera_name: The camera name to validate.
        :raises ValueError: If the camera name is invalid.
        """
        if camera_name not in CameraModelManager.CAMERA_NAMES:
            raise ValueError(f"Invalid camera name: {camera_name}")
