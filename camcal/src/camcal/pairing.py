from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import exifread
import polars as pl
from natsort import natsorted
from pydantic import Field
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from camcal.camera import CameraLocationInfo

__all__ = ["PairFactory"]


@dataclass
class ImagePair:
    ir: Optional[Path] = None
    vis: Optional[Path] = None


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

    def _extract_datetimes_parallel(self, images: List[Path],
                                    desc: str) -> Dict[datetime, Path]:
        """
        Extract datetimes from image files in parallel with a progress bar.

        Parameters:
        ------------
        images : List[Path]
            List of image file paths.
        desc : str
            Description for the progress bar.

        Returns:
        ---------
        Dict[datetime, Path]:
            Dictionary mapping extracted datetimes to their corresponding image paths.
        """
        with ProcessPoolExecutor() as executor:
            # Wrap executor.map with tqdm for progress tracking
            results = list(
                tqdm(
                    executor.map(self._extract_datetime, images),
                    total=len(images),
                    desc=desc,
                ))
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

        print(f'Clipping to daylight hours ({sun_times['sunrise']} to {sun_times['sunset']}) active. Retained {len(daylight_pairs)} pairs of originally {len(pairs)}.')
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
        ir_dict = self._extract_datetimes_parallel(
            self.ir_imgs, desc='Extracting IR datetimes')
        vis_dict = self._extract_datetimes_parallel(
            self.vis_imgs, desc='Extracting VIS datetimes')

        # Get all unique timestamps from both dictionaries
        all_timestamps = set(ir_dict.keys()).union(set(vis_dict.keys()))

        # Create pairs, allowing for missing images
        pairs = {
            timestamp:
            ImagePair(
                ir=ir_dict.get(
                    timestamp),  # Use image from IR if available, else None
                vis=vis_dict.get(
                    timestamp)  # Use image from VIS if available, else None
            )
            for timestamp in sorted(all_timestamps)
        }

        # Optionally filter pairs to daylight hours
        if self.clip_to_daylight:
            return self._filter_to_daylight(pairs)

        return pairs

    @staticmethod
    def print_pairs(pairs: Dict[datetime, ImagePair]) -> pl.DataFrame:
        """
        Print a summary of the image pairs.

        Parameters:
        ------------
        pairs : Dict[datetime, ImagePair]
            Dictionary of image pairs keyed by datetime.
        """
        print(f"Found {len(pairs)} image pairs.")
        times, irs, viss = [], [], []
        for k, v in pairs.items():
            time = k.strftime('%Y-%m-%d %H:%M:%S')
            if not v.ir:
                ir = None
            else:
                ir = str(v.ir.name)

            if not v.vis:
                vis = None
            else:
                vis = str(v.vis.name)

            times.append(time)
            irs.append(ir)
            viss.append(vis)

        # print(_dict)
        df = pl.DataFrame({'Timestamp': times, 'IR': irs, 'VIS': viss})

        print(df)
        return df
