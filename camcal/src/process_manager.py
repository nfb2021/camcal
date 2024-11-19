from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator

import xarray as xr
from pydantic import Field
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from camcal.src.aperture_cropping import ImageOrienter
from camcal.src.cam_cal import CamAngleOffset
from camcal.src.pairing import ImagePair
from camcal.src.processing import Png2NetCDF


@dataclass
class ImageProcessingManager:
    image_pairs: Dict[datetime, ImagePair]
    offset_file: Path
    clip_to_daylight: bool = Field(default=True)

    def __post_init__(self):
        if not self.offset_file.exists():
            raise FileNotFoundError(
                f"Offset file {self.offset_file} does not exist.")

        # Load camera calibration data for both cameras
        self.camera_offsets = {
            "IR": CamAngleOffset.from_parquet(self.offset_file, "IR"),
            "VIS": CamAngleOffset.from_parquet(self.offset_file, "VIS"),
        }

    # def _load_offsets(self) -> Dict[str, Dict[str, float]]:
    #     """
    #     Load calibration offsets for each camera from the Parquet file.

    #     Returns:
    #     ---------
    #     Dict[str, Dict[str, float]]:
    #         A dictionary containing the calibration data for each camera.
    #     """
    #     import polars as pl
    #     offsets_df = pl.read_parquet(self.offset_file)
    #     return {
    #         row["camera"]: {
    #             "mean_deg": row["mean [deg]"],
    #             "median_deg": row["median [deg]"],
    #             "std_deg": row["std [deg]"]
    #         }
    #         for row in offsets_df.to_dicts()
    #     }

    def _crop_image(self, image_path: Path, timestamp: datetime,
                    camera_name: str) -> xr.Dataset:
        """
        Crop and process a single image.

        Parameters:
        ------------
        image_path : Path
            Path to the image file.
        timestamp : datetime
            Timestamp of the image.
        camera_name : str
            Camera name ("IR" or "VIS").

        Returns:
        --------
        xr.Dataset
            Processed image dataset.
        """
        offset = self.camera_offsets[camera_name].mean
        image_orienter = ImageOrienter(
            path=image_path,
            output_dir=None,  # Not saving to disk
            camera_name=camera_name,
            offset=offset,  # Pass the offset directly
        )
        cropped_img = image_orienter.process_image()  # Returns np.ndarray
        png2netcdf = Png2NetCDF(img_arr=cropped_img)
        ds = png2netcdf.get_dataset()
        ds = ds.assign_coords({
            "Epoch": timestamp,
            "Camera": camera_name
        })  # Add metadata
        return ds

    # def _crop_images_generator_parallel(self) -> Generator[xr.Dataset, None, None]:
    #     """
    #     Process images in parallel for both cameras, yielding cropped datasets one by one.

    #     Yields:
    #     -------
    #     xr.Dataset
    #         The cropped image as an `xarray.Dataset`.
    #     """
    #     with ProcessPoolExecutor() as executor:
    #         futures = {}
    #         for timestamp, pair in self.image_pairs.items():
    #             for camera_name in ["IR", "VIS"]:
    #                 image_path = str(pair.ir if camera_name ==
    #                                  "IR" else pair.vis)
    #                 futures[executor.submit(self._crop_image, image_path,
    #                                         timestamp,
    #                                         camera_name)] = (timestamp,
    #                                                          camera_name)

    #         for future in tqdm(as_completed(futures),
    #                            total=len(futures),
    #                            desc="Cropping images"):
    #             yield future.result()

    def _crop_images_generator(self) -> Generator[xr.Dataset, None, None]:
        """
        Yield cropped datasets one by one for all images.

        Yields:
        -------
        xr.Dataset
            Processed image dataset.
        """
        for timestamp, pair in self.image_pairs.items():
            for camera_name in ["IR", "VIS"]:
                image_path = pair.ir if camera_name == "IR" else pair.vis
                yield self._crop_image(image_path, timestamp, camera_name)

    def combine_to_final_dataset(self) -> xr.Dataset:
        """
        Combine all processed images into a single xarray.Dataset.

        Returns:
        --------
        xr.Dataset
            Final combined dataset.
        """
        datasets = self._crop_images_generator()
        combined_ds = None

        for ds in datasets:
            if combined_ds is None:
                combined_ds = ds.expand_dims(["Epoch", "Camera"])
            else:
                combined_ds = xr.concat(
                    [combined_ds,
                     ds.expand_dims(["Epoch", "Camera"])],
                    dim="Epoch")

        return combined_ds
