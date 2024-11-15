import math
import os
from pathlib import Path
from typing import Union, Optional
from pydantic.dataclasses import dataclass
from pydantic import Field
import cv2
import numpy as np
import polars as pl
from dotenv import load_dotenv

from camcal.src.image import ImageReader
from camcal.src.aperture_cropping import ImageOrienter


@dataclass
class CCS2SCS:
    """Converts image pixel coordinates from 3D Cartesian Coordinatre System (CCS) to 3D Spherical Coordinate System (SCS).

    In the CCS, the X-axis is defined as parallel to the image's width, pointing to the right. The Y-axis is defined\\
    as parallel to the image's height, pointing downwards. The Z-axis is defined as perpendicular to the image, pointing outwards.\\
    The origin of the CCS is in the center of the image. As the image is a 2D plane, all Z values are zero.

    The SCS has its origin at the center of the image, too, and follows the conventions of physics, i.e. being a right-handed\\
    coordinate system. The axes have the same orientation as in the CCS. The new coordinates are the radial distance\\
    from the origin rho, the azimuth angle phi, and the polar angle theta.


    Parameters:
    -----------
        aperture_cutoff : float
            Cutoff angle for the polar angle theta in degrees. This angle is given as the aperture of the vertical\\
                dimension of the camera used to capture the image.

        img_centered: np.ndarray
            Image to convert coordinates for.

    """


# class ImageProcessor(ImageReader):
#     """Processes images, including grayscale conversion and line detection.

#     Parameters:
#     -----------
#         path : Path
#             Path to the image file.

#     """

#     def __init__(self, path: Path) -> None:

#         super().__init__(path)
#         self.image: np.ndarray = self.read_image()

#     @property
#     def grayscale(self) -> np.ndarray:
#         """
#         Convert the image to grayscale.

#         Returns:
#         --------
#             np.ndarray
#                 Grayscale version of the image.
#         """
#         gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#         return gray_image

#     def detect_lines(self) -> Optional[np.ndarray]:
#         """
#         Detect lines in the grayscale image using the Hough Line Transform.

#         Returns:
#         --------
#             Optional[np.ndarray]
#                 Array of lines if detected, None otherwise.
#         """
#         gray = self.grayscale
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

#         if lines is None:
#             raise ValueError("No lines detected")
#         return lines

#     @classmethod
#     def from_file(cls, path: Union[str, Path]) -> "ImageProcessor":
#         """
#         Create an `ImageProcessor` instance from a file path.

#         Parameters:
#         ------------------
#             path : Union[str, Path]
#                 Path to the image file.

#         Returns:
#         --------
#             `ImageProcessor`
#                 An instance of `ImageProcessor`.
#         """
#         return cls(path=Path(path))

#     @classmethod
#     def preview(cls, img: np.ndarray, scale: float = 0.7) -> None:
#         """
#         Display a preview of the image, scaled to the specified size.

#         Parameters:
#         ------------------
#             img : np.ndarray
#                 Image to display.
#             scale : float, optional
#                 Scale factor (default is 0.7).
#         """
#         resized_img_shape = get_resized_img_shape(img, scale)
#         resized_img = cv2.resize(
#             img, (resized_img_shape['width'], resized_img_shape['height']))
#         cv2.imshow(
#             f"Preview (resized from {img.shape[1], img.shape[0]} to {resized_img_shape['height'], resized_img_shape['width']})",
#             resized_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
