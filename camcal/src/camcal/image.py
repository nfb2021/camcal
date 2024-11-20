#%%
import os
from importlib.metadata import PathDistribution
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

__all__ = ['ImageReader']


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


#%%
if __name__ == "__main__":
    load_dotenv()
    ROOT_DIR: str = os.getenv('ROOT_DIR')

    pth = Path(ROOT_DIR) / Path("VIS/2024-10-24/11:48:44_VIS.jpg")

    image_reader = ImageReader(path=pth)
    img = image_reader.read_image()
