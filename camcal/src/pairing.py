from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import exifread
from natsort import natsorted
from pydantic import Field
from pydantic.dataclasses import dataclass

from camcal.src.camera import CameraLocationInfo


@dataclass
class ImagePair:
    ir: Path
    vis: Path


@dataclass
class PairFactory:
    """
    Factory for producing paired image data, filtered based on specified criteria.

    Parameters:
    ------------
    ir_pths : Path
        Path to the directory containing infrared (IR) images.
    vis_pths : Path
        Path to the directory containing visible (VIS) images.
    camera_loc : CameraLocationInfo
        Camera location information, including time zone and coordinates.
    clip_to_daylight : bool, optional
        Whether to filter pairs to daylight hours, default is True.
    """
    ir_pths: Path
    vis_pths: Path
    camera_loc: CameraLocationInfo
    clip_to_daylight: bool = Field(default=True)

    def __post_init__(self):
        if not self.ir_pths.exists():
            raise FileNotFoundError(f"Path {self.ir_pths} does not exist.")
        if not self.vis_pths.exists():
            raise FileNotFoundError(f"Path {self.vis_pths} does not exist.")

        self.ir_imgs = natsorted(list(self.ir_pths.glob('*.jpg')))
        self.vis_imgs = natsorted(list(self.vis_pths.glob('*.jpg')))

    def _extract_datetime(self, pth: Path) -> datetime:
        """
        Extract the datetime from a JPG image file, assigning the correct timezone.
        """
        with open(pth, "rb") as image_file:
            exif_tags = exifread.process_file(image_file)

        # Try to extract EXIF DateTime
        dt = exif_tags.get('EXIF DateTime')

        if dt:
            date_obj = datetime.strptime(str(dt), '%Y:%m:%d %H:%M:%S')
            date_obj = date_obj.replace(second=0)  # Remove seconds
        else:
            dt = int(pth.stem.split('_')[0])
            date_obj = datetime.fromtimestamp(dt).replace(second=0)

        # Localize to the timezone specified in CameraLocationInfo
        return self.camera_loc.timezone_obj.localize(date_obj)

    def _extract_datetimes_parallel(
            self, images: List[Path]) -> Dict[datetime, Path]:
        """
        Extract datetimes from image files in parallel.

        Parameters:
        ------------
        images : List[Path]
            List of image file paths.

        Returns:
        ---------
        Dict[datetime, Path]:
            Dictionary mapping extracted datetimes to their corresponding image paths.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._extract_datetime, images))
        return dict(zip(results, images))

    def _filter_to_daylight(
            self, pairs: Dict[datetime,
                              ImagePair]) -> Dict[datetime, ImagePair]:
        """
        Filter image pairs to retain only those captured during daylight hours.

        Parameters:
        ------------
        pairs : Dict[datetime, ImagePair]
            Dictionary of image pairs keyed by datetime.

        Returns:
        ---------
        Dict[datetime, ImagePair]:
            Filtered dictionary containing only daylight image pairs.
        """
        daylight_pairs: Dict[datetime, ImagePair] = {}
        for timestamp, pair in pairs.items():
            sun_times = self.camera_loc.get_sun_times(timestamp)
            if sun_times['sunrise'] <= timestamp <= sun_times['sunset']:
                daylight_pairs[timestamp] = pair
        return daylight_pairs

    def create_pairs(self) -> Dict[datetime, ImagePair]:
        """
        Generate and optionally filter image pairs based on initialization parameters.

        Returns:
        ---------
        Dict[datetime, ImagePair]:
            A dictionary where keys are timestamps and values are matched image pairs.
        """
        # Extract datetimes for IR and VIS images
        ir_dict = self._extract_datetimes_parallel(self.ir_imgs)
        vis_dict = self._extract_datetimes_parallel(self.vis_imgs)

        # Match IR and VIS images by common timestamps
        common_timestamps = set(ir_dict.keys()) & set(vis_dict.keys())
        pairs = {
            timestamp: ImagePair(ir=ir_dict[timestamp],
                                 vis=vis_dict[timestamp])
            for timestamp in sorted(common_timestamps)
        }

        # Optionally filter pairs to daylight hours
        if self.clip_to_daylight:
            return self._filter_to_daylight(pairs)

        return pairs
