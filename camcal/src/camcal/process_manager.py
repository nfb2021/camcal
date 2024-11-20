from datetime import datetime
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import xarray as xr
from datatree import DataTree
from pydantic import Field
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from camcal.aperture_cropping import ImageOrienter
from camcal.cam_cal import CamAngleOffset
from camcal.pairing import ImagePair
from camcal.processing import CCS2SCS, Png2NetCDF
from camcal.tools import strip_timezone

__all__ = ['ImageProcessingManager']


class MissingImagePlaceholder(xr.Dataset):
    """
    Placeholder dataset for missing images.
    """
    __slots__ = ()

    @property
    def attrs(self):
        return {
            'title': 'Missing Image',
            'description': 'I am a placeholder for missing images.',
            'is_missing': True,
        }


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
        png2netcdf = Png2NetCDF(img_arr=cropped_img, pth_orig_img=image_path)
        ds = png2netcdf.get_dataset()
        ds = ds.assign_coords({"Epoch": timestamp})
        ds.attrs.update({
            'camera': camera_name,
        })
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

    # def _crop_images_generator(self) -> Generator[xr.Dataset, None, None]:
    #     """
    #     Yield cropped datasets one by one for all images.

    #     Yields:
    #     -------
    #     xr.Dataset
    #         Processed image dataset.
    #     """
    #     for timestamp, pair in self.image_pairs.items():
    #         for camera_name in ["IR", "VIS"]:
    #             image_path = pair.ir if camera_name == "IR" else pair.vis
    #             yield self._crop_image(image_path, timestamp, camera_name)

    def _crop_images_generator(self) -> Generator[xr.Dataset, None, None]:
        """
        Yield cropped datasets one by one for all images, handling missing images gracefully.

        Yields:
        -------
        xr.Dataset
            Processed image dataset or placeholder for missing images.
        """

        for timestamp, pair in self.image_pairs.items():

            timestamp = np.datetime64(strip_timezone(timestamp))
            for camera_name in ["IR", "VIS"]:
                image_path = pair.ir if camera_name == "IR" else pair.vis
                if image_path is None or not image_path.exists():
                    # Create a placeholder dataset with consistent dimensions
                    yield MissingImagePlaceholder().assign_coords(
                        Epoch=timestamp, Camera=camera_name)
                else:
                    yield self._crop_image(image_path, timestamp, camera_name)

    # def combine_to_final_dataset(self) -> xr.Dataset:
    #     """
    #     Combine all processed images into a single xarray.Dataset.

    #     Returns:
    #     --------
    #     xr.Dataset
    #         Final combined dataset.
    #     """
    #     datasets = self._crop_images_generator()
    #     combined_ds = None

    #     for ds in datasets:
    #         if combined_ds is None:
    #             combined_ds = ds.expand_dims(["Epoch", "Camera"])
    #         else:
    #             combined_ds = xr.concat(
    #                 [combined_ds,
    #                  ds.expand_dims(["Epoch", "Camera"])],
    #                 dim="Epoch")

    #     return combined_ds

    def create_datatree(self) -> DataTree:
        """
        Create a DataTree from all processed images, organizing them by camera type.

        Returns:
        --------
        DataTree
            A hierarchical DataTree containing datasets for each camera.
        """
        datasets = self._crop_images_generator()
        tree = DataTree()

        for ds in tqdm(datasets, desc="Creating DataTree"):
            if isinstance(ds, MissingImagePlaceholder):
                # Skip placeholders
                continue

            # Extract camera type from dataset attributes
            camera = ds.attrs.get("camera")  # Get scalar value for camera name
            ds["Epoch"] = ds["Epoch"].astype("datetime64[ns]")

            if camera not in tree.children:
                # Initialize a new DataTree node for this camera
                tree[camera] = DataTree(data=ds.expand_dims("Epoch"))
            else:
                # Concatenate new dataset along the Epoch dimension for the existing camera
                existing_ds = tree[camera].ds
                updated_ds = xr.concat(
                    [existing_ds, ds.expand_dims("Epoch")],
                    dim="Epoch",
                    fill_value=np.
                    nan  # Ensure no conflicts during concatenation
                )
                tree[camera] = DataTree(data=updated_ds)

        # Ensure shared dimensions like x and y are consistent across datasets
        for camera in tree.children:
            camera_ds = tree[camera].ds
            if "x" in camera_ds.dims and "y" in camera_ds.dims:
                camera_ds = camera_ds.transpose("Epoch", "y", "x", ...)
                tree[camera] = DataTree(data=camera_ds)

        return tree

    def convert_tree_to_spherical(
        self,
        tree: DataTree,
        theta_cutoff_dict: Dict[str, float] = {
            'IR': 67.0,
            'VIS': 67.0
        }
    ) -> DataTree:
        """
        Convert Cartesian coordinates to spherical coordinates for each camera node in the DataTree.

        Parameters:
        ------------
        tree : DataTree
            The original DataTree containing datasets with Cartesian coordinates.
        theta_cutoff_dict : Dict[str, float]
            Dictionary mapping camera names to theta cutoff angles. Default is 67.0 degrees.

        Returns:
        ---------
        DataTree
            Updated DataTree with spherical coordinates added to each node.
        """
        for camera, node in tree.children.items():
            if node.ds is not None:
                print(f"Converting {camera} to spherical coordinates...")
                converter = CCS2SCS(img_centered_ds=node.ds,
                                    theta_cutoff=theta_cutoff_dict[camera])
                spherical_ds = converter.convert_to_scs()
                tree[camera] = DataTree(data=spherical_ds)

        return tree
