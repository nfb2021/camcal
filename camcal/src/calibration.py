#%%
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.dataclasses import dataclass


@dataclass
class ImageReader:
    path: Path

    @field_validator('path')
    def check_file(cls, value):
        if not value.exists():
            raise ValueError(f"File does not exist: {value}")

        if value.suffix.lower() != '.jpg':
            raise ValueError(
                f"Invalid file extension: {value.suffix}. Expected .jpg")

        return value

    def read_image(self):
        self.image = cv2.imread(str(self.path))
        return self.image


class ImagePocessor(ImageReader):

    def __init__(self, path: Path):
        super().__init__(path)
        self.image = self.read_image()

    @property
    def grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def detect_strip_angle(self):
        # Step 1: Read the image

        # Step 2: Convert to grayscale
        gray = self.grayscale

        # Step 3: Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Step 4: Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            raise ValueError("No lines detected")
        return lines

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        return cls(path=Path(path))

    @classmethod
    def preview(cls, img):
        cv2.imshow('Detected Line', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


#%%
if __name__ == "__main__":
    pth = "/home/nbader/shares/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/03_camera_calib/.phoenix/VIS/2024-10-24/11:48:44_VIS.jpg"

    image_reader = ImagePocessor(path=pth)
    img = image_reader.read_image()
    lines = image_reader.detect_strip_angle()
    image_reader.preview(img)
