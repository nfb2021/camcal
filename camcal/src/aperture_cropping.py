from pathlib import Path
from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import Field
import cv2
import numpy as np

from camcal.src.image import ImageReader
from camcal.src.cam_cal import CamAngleOffset


@dataclass
class ImageOrienter(ImageReader):
    """Orients images based on camera calibration data, applying transformations such as rotation and masking.

    Parameters:
    -----------
    path : Path
        Path to the image file.
    output_dir : Path
        Path to the output directory.
    camera_name : str
        Name of the camera.
    offset_file : Path
        Path to the camera calibration data file.
    """

    path: Path
    output_dir: Path = Field(default='./02_processed')
    camera_name: str = Field(default="VIS")
    offset_file: Path = Field(default=Path(
        '/home/nbader/shares/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/03_camera_calib/camera_calib.parquet'
    ))

    def __post_init__(self):
        """
        Initialize the ImageOrienter, loading the image and calibration data.
        """
        self.image: Optional[np.ndarray] = cv2.imread(str(self.path))
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.path}")

        self.camera_calib_data: CamAngleOffset = CamAngleOffset.from_parquet(
            self.offset_file, self.camera_name)

        self.center: tuple[int, int] = (self.image.shape[1] // 2,
                                        self.image.shape[0] // 2)
        self.prepare_output_dir()

    def prepare_output_dir(self) -> None:
        """
        Create the output directory if it does not already exist.
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def mark_center(self) -> np.ndarray:
        """
        Mark the center of the image with a red dot.

        Returns:
        --------
            np.ndarray
                Image with center marked.
        """
        marked_image = self.image.copy()
        return cv2.circle(marked_image, self.center, 5, (0, 0, 255), -1)

    def create_circular_mask(self) -> np.ndarray:
        """
        Create a circular mask centered on the image.

        Returns:
        --------
            np.ndarray
                Circular mask.
        """
        mask = np.zeros_like(self.image, dtype=np.uint8)
        return cv2.circle(mask, self.center, self.image.shape[0] // 2,
                          (255, 255, 255), -1)

    def apply_circular_mask(self, marked_image: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        Apply a circular mask to the image.

        Parameters:
        ------------------
            marked_image : np.ndarray
                Image with center marked.
            mask : np.ndarray
                Circular mask.

        Returns:
        --------
            np.ndarray
                Image with circular mask applied.
        """
        return cv2.bitwise_and(marked_image, marked_image, mask=mask[:, :, 0])

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate the image based on camera calibration data.

        Parameters:
        ------------------
            image : np.ndarray
                Image to rotate.

        Returns:
        --------
            np.ndarray
                Rotated image.
        """
        rotation_matrix = cv2.getRotationMatrix2D(self.center,
                                                  -self.camera_calib_data.mean,
                                                  1.0)
        return cv2.warpAffine(image, rotation_matrix,
                              (image.shape[1], image.shape[0]))

    def add_alpha_channel(self, image: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        """
        Add an alpha channel to the image based on the mask.

        Parameters:
        ------------------
            image : np.ndarray
                Input image.
            mask : np.ndarray
                Mask to define alpha channel.

        Returns:
        --------
            np.ndarray
                Image with alpha channel added.
        """
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :, 3] = mask[:, :, 0]
        return rgba_image

    def apply_square_mask(self, rgba_image: np.ndarray) -> np.ndarray:
        """
        Apply a square mask to the image.

        Parameters:
        ------------------
            rgba_image : np.ndarray
                Image with alpha channel.

        Returns:
        --------
            np.ndarray
                Image with square mask applied.
        """
        square_mask = np.zeros_like(rgba_image, dtype=np.uint8)
        side_length = self.image.shape[0]
        top_left = (self.center[0] - side_length // 2,
                    self.center[1] - side_length // 2)
        bottom_right = (self.center[0] + side_length // 2,
                        self.center[1] + side_length // 2)
        cv2.rectangle(square_mask, top_left, bottom_right, (255, 255, 255), -1)
        return cv2.bitwise_and(rgba_image,
                               rgba_image,
                               mask=square_mask[:, :, 0])

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

    def save_image(self, cropped_image: np.ndarray, output_name: str) -> Path:
        """
        Save the processed image to the output directory.

        Parameters:
        ------------------
            cropped_image : np.ndarray
                Image to save.
            output_name : str
                Name of the output file.

        Returns:
        --------
            Path
                Path to the saved image file.
        """
        output_path = self.output_dir / f'{output_name}_processed.png'
        cv2.imwrite(
            str(output_path),
            cropped_image,
            # [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )
        return Path(output_path)

    def process_image(self) -> None:
        """
        Apply a series of transformations (masking, rotation, cropping) to the image and save the result.
        """
        img_copy = self.image.copy()
        circle_mask = self.create_circular_mask()
        circular_image = self.apply_circular_mask(img_copy, circle_mask)
        rotated_image = self.rotate_image(circular_image)
        rgba_image = self.add_alpha_channel(rotated_image, circle_mask)
        square_image = self.apply_square_mask(rgba_image)
        cropped_image = self.crop_to_square(square_image)
        self.save_image(cropped_image, self.path.stem)
