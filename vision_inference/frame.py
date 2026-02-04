"""
This module contains the Frame class, which is used to store a frame from one of the cameras and associated metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np


@dataclass
class Frame:
    """
    A class to store a frame from one of the cameras and associated metadata.

    :param image: The image as a numpy array. The image is expected to be in RGB format.
    :param camera_name: The name of the camera that captured the frame.
    :param timestamp: The timestamp of the frame.
    :param frame_id: The unique ID of the frame, generated from the timestamp.
    """

    image: np.ndarray
    camera_name: str
    timestamp: datetime
    frame_id: str = field(init=False)

    def __post_init__(self):
        # convert hash to hex string
        self.frame_id = f"{hash(self.timestamp):x}"

    @property
    def debug_str(self) -> str:
        """
        :return: A debug string providing metadata about this Frame.
        """
        return f"[Camera {self.camera_name} frame {self.frame_id}]"

    def resize(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Resize the image contained in this Frame to the specified width and height.
        The resized image is returned but this Frame object is not modified.

        :param width: The width to resize the image to.
        :param height: The height to resize the image to.
        :return: The resized image as a numpy array.
        """
        return cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA)
