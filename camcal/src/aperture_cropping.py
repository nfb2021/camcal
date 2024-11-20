from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from camcal.src.cam_cal import CamAngleOffset
from camcal.src.image import ImageReader


@dataclass
class ImageOrienter:
    """Orients images based on camera calibration data, applying transformations such as rotation and masking.

    Parameters:
    -----------
    path : Path
        Path to the image file.
    output_dir : Path
        Path to the output directory.
    camera_name : str
        Name of the camera.
    offset : float
        Mean offset in degrees for the camera.
    """

    path: Path
    output_dir: Optional[Path] = None
    camera_name: str = Field(..., description="Name of the camera (IR or VIS)")
    offset: float = Field(...,
                          description="Mean offset in degrees for the camera")

    def __post_init__(self):
        """
        Initialize the ImageOrienter, loading the image and preparing the output directory.
        """
        # Load the image
        self.image: Optional[np.ndarray] = cv2.imread(str(self.path))
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.path}")

        # Determine the center of the image
        self.center: tuple[int, int] = (self.image.shape[1] // 2,
                                        self.image.shape[0] // 2)

        # Prepare the output directory, if applicable
        if self.output_dir:
            self.prepare_output_dir()

    def prepare_output_dir(self) -> None:
        """
        Create the output directory if it does not already exist.
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def create_circular_mask(self) -> np.ndarray:
        """
        Create a circular mask centered on the image.

        Returns:
        --------
            np.ndarray
                Circular mask.
        """
        mask = np.zeros((self.image.shape[0], self.image.shape[1]),
                        dtype=np.uint8)
        return cv2.circle(mask, self.center, self.image.shape[0] // 2, 255, -1)

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate the image based on the offset.

        Parameters:
        ------------------
            image : np.ndarray
                Image to rotate.

        Returns:
        --------
            np.ndarray
                Rotated image.
        """
        rotation_matrix = cv2.getRotationMatrix2D(self.center, -self.offset,
                                                  1.0)
        return cv2.warpAffine(image, rotation_matrix,
                              (image.shape[1], image.shape[0]))

    def crop_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Crop the image to a square centered around the middle.

        Parameters:
        ------------------
            image : np.ndarray
                Image to crop.

        Returns:
        --------
            np.ndarray
                Cropped square image.
        """
        side_length = self.image.shape[0]
        top_left = (self.center[0] - side_length // 2,
                    self.center[1] - side_length // 2)
        bottom_right = (self.center[0] + side_length // 2,
                        self.center[1] + side_length // 2)
        return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    def process_image(self) -> np.ndarray:
        """
        Apply a series of transformations (masking, rotation, cropping) to the image and return as RGBA PNG.

        Returns:
        --------
            np.ndarray
                Processed RGBA image.
        """
        img_copy = self.image.copy()
        circle_mask = self.create_circular_mask()
        masked_image = cv2.bitwise_and(img_copy, img_copy, mask=circle_mask)
        rotated_image = self.rotate_image(masked_image)
        cropped_image = self.crop_to_square(rotated_image)

        # Create an RGBA image
        rgba_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :,
                   3] = self.crop_to_square(circle_mask)  # Set alpha channel

        return rgba_image
