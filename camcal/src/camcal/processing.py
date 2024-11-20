from pathlib import Path
from typing import Optional, Union

import cv2
import exifread
import numpy as np
import xarray as xr
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from camcal.tools import exif_str_to_snake_case

__all__ = ["Png2NetCDF", "CCS2SCS"]


@dataclass(config={"arbitrary_types_allowed": True})
class Png2NetCDF:
    """Converts a PNG image to a NetCDF file, with each pixel's RGBA values as variables."""

    img_arr: np.ndarray = Field(...)
    pth_orig_img: Optional[Path] = Field(None)

    @field_validator("img_arr")
    def validate_img_arr(v):
        # Ensure proper 4-channel RGBA image
        if v.ndim != 3:
            raise ValueError("Image array must have three dimensions.")
        if v.shape[2] != 4:
            raise ValueError("Image array must have four channels (RGBA).")
        return v

    @classmethod
    def from_png(
            cls,
            path: Union[str, Path],
            pth_orig_img: Optional[Union[str, Path]] = None) -> "Png2NetCDF":
        """
        Create a `Png2NetCDF` instance from a PNG file.

        Parameters:
        ------------------
            path : Union[str, Path]
                Path to the PNG file.

        Returns:
        --------
            `Png2NetCDF`
                An instance of `Png2NetCDF`.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        img_arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if pth_orig_img:
            return cls(img_arr=img_arr, pth_orig_img=pth_orig_img)

        return cls(img_arr=img_arr)

    def get_dataset(self) -> xr.Dataset:
        """
        Convert the image to an `xarray.Dataset`.

        Returns:
        --------
            xr.Dataset
                NetCDF dataset with the image data.
        """
        ds = self.__ndarr2netcdf(self.img_arr)
        # ds = self.__shift_origin_to_center(ds)
        ds = self.__add_metadata(ds)

        return ds

    def __ndarr2netcdf(self, img_arr: np.ndarray) -> xr.Dataset:
        """
        Convert a 4-channel RGBA image array to a NetCDF dataset, with origin in the center.

        Parameters:
        ------------------
            img_arr : np.ndarray
                Image array to convert. Must be in the range [0, 255].

        Returns:
        --------
            xr.Dataset
                NetCDF dataset, where each R, G, B, and mask are data variables.
        """
        # Determine the scaling factor based on the input range
        red, green, blue, alpha = img_arr[..., 0], img_arr[..., 1], img_arr[
            ..., 2], img_arr[..., 3]

        # Create DataArrays for RGB channels and the mask
        red_da = xr.DataArray(red, dims=['x', 'y'])
        green_da = xr.DataArray(green, dims=['x', 'y'])
        blue_da = xr.DataArray(blue, dims=['x', 'y'])

        # Create the dataset
        img_ds = xr.Dataset(
            {
                'red': red_da,
                'green': green_da,
                'blue': blue_da,
                # 'mask': mask_da
            },
            coords={
                'x': (['x'],
                      np.arange(-img_arr.shape[0] // 2,
                                img_arr.shape[0] // 2,
                                dtype='int16')),
                'y': (['y'],
                      np.arange(-img_arr.shape[1] // 2,
                                img_arr.shape[1] // 2,
                                dtype='int16')),
            },
        )

        return img_ds

    # def __ndarr2netcdf_org(self, img_arr: np.ndarray) -> xr.Dataset:
    #     """
    #     Convert a 4-channel RGBA image array to a NetCDF dataset.

    #     Parameters:
    #     ------------------
    #         img_arr : np.ndarray
    #             Image array to convert.

    #     Returns:
    #     --------
    #         xr.Dataset
    #             NetCDF dataset, where each R, G, B, and alpha are data variables.
    #     """
    #     # Extract RGBA channels
    #     red, green, blue, alpha = img_arr[..., 0], img_arr[..., 1], img_arr[
    #         ..., 2], img_arr[..., 3]

    #     # Create the dataset
    #     img_ds = xr.Dataset(
    #         {
    #             "red": (["y", "x"], red.astype(np.uint8)),
    #             "green": (["y", "x"], green.astype(np.uint8)),
    #             "blue": (["y", "x"], blue.astype(np.uint8)),
    #             "mask":
    #             (["y", "x"],
    #              (alpha == 255).astype(bool)),  # Mask where alpha == 255
    #         },
    #         coords={
    #             "x": np.arange(img_arr.shape[1]).astype(np.int16),
    #             "y": np.arange(img_arr.shape[0]).astype(np.int16),
    #         },
    #     )

    #     # Apply the mask to RGB channels (set to NaN where mask is False)
    #     img_ds["red"] = img_ds["red"].where(img_ds["mask"], np.nan)
    #     img_ds["green"] = img_ds["green"].where(img_ds["mask"], np.nan)
    #     img_ds["blue"] = img_ds["blue"].where(img_ds["mask"], np.nan)

    #     return img_ds

    # def __shift_origin_to_center(self, img_ds: xr.Dataset) -> xr.Dataset:
    #     """
    #     Shift the origin of the image to the center.

    #     Parameters:
    #     ------------------
    #         img_ds : xr.Dataset
    #             Image dataset to shift.

    #     Returns:
    #     --------
    #         xr.Dataset
    #             Image dataset with origin shifted to the center.
    #     """
    #     img_ds["x"] = img_ds["x"] - img_ds["x"].max() // 2
    #     img_ds["y"] = img_ds["y"] - img_ds["y"].max() // 2

    #     return img_ds

    def __add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add metadata to the image dataset.

        Parameters:
        ------------------
            ds : xr.Dataset
                Image dataset to add metadata to.

        Returns:
        --------
            xr.Dataset
                Image dataset with metadata added.
        """
        ds.coords["x"].attrs = {
            "long_name": "x-coordinate in image",
            "short_name": "x",
            "units": "pixels",
            "axis": "X",
        }

        ds.coords["y"].attrs = {
            "long_name": "y-coordinate in image",
            "short_name": "y",
            "units": "pixels",
            "axis": "Y",
        }

        ds.data_vars["red"].attrs = {
            "long_name": "Red channel",
            "short_name": "R",
            "valid_range": [0, 255],
            "units": "Intensity",
        }

        ds.data_vars["green"].attrs = {
            "long_name": "Green channel",
            "short_name": "G",
            "valid_range": [0, 255],
            "units": "Intensity",
        }

        ds.data_vars["blue"].attrs = {
            "long_name": "Blue channel",
            "short_name": "B",
            "valid_range": [0, 255],
            "units": "Intensity",
        }

        if self.pth_orig_img:
            ds.attrs.update(self.__get_img_metadata(self.pth_orig_img))

        return ds

    def __get_img_metadata(self, jpeg_pth: Path) -> dict:
        """
        Get metadata from the image.

        Returns:
        --------
            dict
                Image metadata.
        """
        if not isinstance(jpeg_pth, Path):
            jpeg_pth = Path(jpeg_pth)

        if not jpeg_pth.exists():
            raise FileNotFoundError(f"File not found: {jpeg_pth}")

        with open(jpeg_pth, "rb") as image_file:
            # Extract EXIF tags
            exif_tags = exifread.process_file(image_file)

        to_keep = [
            # "EXIF DateTime",
            "EXIF Make",
            "EXIF Model",
            "EXIF Software",
            "EXIF ExposureTime",
            "EXIF ISOSpeedRatings",
            "EXIF SubjectDistance",
        ]

        _img_metadata = {
            exif_str_to_snake_case(k): str(exif_tags[k])
            for k in to_keep if k in exif_tags
        }

        return _img_metadata


# @dataclass(config={"arbitrary_types_allowed": True})
# class Png2NetCDF:
#     """Converts a PNG image to a NetCDF file, with each pixel's RGBA values as a variable."""

#     img_arr: np.ndarray = Field(...)
#     pth_orig_img: Optional[Path] = Field(None)

#     @field_validator("img_arr")
#     def validate_img_arr(v):
#         # Additional validator to ensure proper 4-channel RGBA
#         if v.ndim != 3:
#             raise ValueError("Image array must have three dimensions.")
#         if v.shape[2] != 4:
#             raise ValueError("Image array must have four channels (RGBA).")
#         return v

#     @classmethod
#     def from_png(
#             cls,
#             path: Union[str, Path],
#             pth_orig_img: Optional[Union[str, Path]] = None) -> "Png2NetCDF":
#         """
#         Create a `Png2NetCDF` instance from a PNG file.

#         Parameters:
#         ------------------
#             path : Union[str, Path]
#                 Path to the PNG file.

#         Returns:
#         --------
#             `Png2NetCDF`
#                 An instance of `Png2NetCDF`.
#         """
#         if isinstance(path, str):
#             path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"File not found: {path}")

#         img_arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

#         if pth_orig_img:
#             return cls(img_arr=img_arr, pth_orig_img=pth_orig_img)

#         return cls(img_arr=img_arr)

#     def get_dataset(self) -> xr.Dataset:
#         """
#         Convert the image to an `xarray.Dataset`.

#         Returns:
#         --------
#             xr.Dataset
#                 NetCDF dataset with the image data.
#         """

#         ds = self.__ndarr2netcdf(self.img_arr)
#         ds = self.__shift_origin_to_center(ds)
#         ds = self.__add_metadata(ds)

#         return ds

#     def __ndarr2netcdf(self, img_arr: np.ndarray) -> xr.Dataset:
#         """
#         Convert a 4-channel RGBA image array to a NetCDF dataset.

#         Parameters:
#         ------------------
#             img_arr : np.ndarray
#                 Image array to convert.

#         Returns:
#         --------
#             xr.Dataset
#                 NetCDF dataset, where each R, G & B channel is a data variable.
#         """
#         img_data = img_arr
#         img_data = img_data.transpose(2, 0, 1)
#         img_ds = xr.Dataset({
#             "red": (["y", "x"], img_data[0, :, :], {
#                 "dtype": "uint8"
#             }),
#             "green": (["y", "x"], img_data[1, :, :], {
#                 "dtype": "uint8"
#             }),
#             "blue": (["y", "x"], img_data[2, :, :], {
#                 "dtype": "uint8"
#             }),
#             # "alpha": (["y", "x"], img_data[3, :, :], {
#             #    "dtype": "uint8"),
#         })

#         img_ds.coords['x'] = img_ds.coords['x'].astype(np.int16)
#         img_ds.coords['y'] = img_ds.coords['y'].astype(np.int16)

#         self.ds = img_ds

#         return img_ds

#     def __shift_origin_to_center(self, img_ds: xr.Dataset) -> xr.Dataset:
#         """
#         Shift the origin of the image to the center.

#         Parameters:
#         ------------------
#             img_ds : xr.Dataset
#                 Image dataset to shift.

#         Returns:
#         --------
#             xr.Dataset
#                 Image dataset with origin shifted to the center.
#         """
#         img_ds["x"] = img_ds["x"] - img_ds["x"].max() // 2
#         img_ds["y"] = img_ds["y"] - img_ds["y"].max() // 2

#         img_ds.coords['x'] = img_ds.coords['x'].astype(np.int16)
#         img_ds.coords['y'] = img_ds.coords['y'].astype(np.int16)

#         self.ds = img_ds
#         return img_ds

#     def __add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
#         """
#         Add metadata to the image dataset.

#         Parameters:
#         ------------------
#             ds : xr.Dataset
#                 Image dataset to add metadata to.

#         Returns:
#         --------
#             xr.Dataset
#                 Image dataset with metadata added.
#         """

#         ds.coords["x"].attrs = {
#             "long_name": "x-coordinate in image",
#             "short_name": "x",
#             "units": "pixels",
#             "axis": "X"
#         }

#         ds.coords["y"].attrs = {
#             "long_name": "y-coordinate in image",
#             "short_name": "y",
#             "units": "pixels",
#             "axis": "Y"
#         }

#         ds.data_vars["red"].attrs = {
#             "long_name": "Red channel",
#             "short_name": "R",
#             "valid_range": [0, 255],
#             "units": "Intensity"
#         }

#         ds.data_vars["green"].attrs = {
#             "long_name": "Green channel",
#             "short_name": "G",
#             "valid_range": [0, 255],
#             "units": "Intensity"
#         }

#         ds.data_vars["blue"].attrs = {
#             "long_name": "Blue channel",
#             "short_name": "B",
#             "valid_range": [0, 255],
#             "units": "Intensity"
#         }

#         if self.pth_orig_img:
#             ds.attrs = self.__get_img_metadata(self.pth_orig_img)

#         return ds

#     def __get_img_metadata(self, jpeg_pth: Path) -> dict:
#         """
#         Get metadata from the image.

#         Returns:
#         --------
#             dict
#                 Image metadata.
#         """
#         if not isinstance(jpeg_pth, Path):
#             jpeg_pth = Path(jpeg_pth)

#         if not jpeg_pth.exists():
#             raise FileNotFoundError(f"File not found: {jpeg_pth}")

#         with open(jpeg_pth, "rb") as image_file:
#             # Extract EXIF tags
#             exif_tags = exifread.process_file(image_file)

#         to_keep = [
#             'EXIF DateTime',
#             'EXIF Make',
#             'EXIF Model',
#             'EXIF Software',
#             'EXIF ExposureTime',
#             'EXIF ISOSpeedRatings',
#             'EXIF SubjectDistance',
#         ]

#         _org_filename = {
#             "CameraType": str(self.pth_orig_img.stem.split("_")[-1]),
#             "OriginalImage": str(self.pth_orig_img.name),
#         }

#         def sanitize_exif_key(k: str) -> str:
#             return k.split("EXIF")[-1].strip() if "EXIF" in k else k

#         _img_metadata = {
#             sanitize_exif_key(k): v
#             for k, v in exif_tags.items() if k in to_keep
#         }

#         def str2datetime(s: str) -> datetime:
#             return datetime.strptime(s, "%Y:%m:%d %H:%M:%S")

#         try:
#             _img_metadata['DateTime'] = str2datetime(
#                 _img_metadata.get('DateTime').values)
#         except:
#             _img_metadata['DateTime'] = 'Unknown'

#         return {**_org_filename, **_img_metadata}


@dataclass(config={"arbitrary_types_allowed": True})
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
        img_centered: np.ndarray
            Image to convert coordinates for.

        theta_cutoff : float
            Cutoff angle for the polar angle theta in degrees. This angle is given as the aperture of the vertical\\
                dimension of the camera used to capture the image.


    """

    img_centered_ds: xr.Dataset = Field(...)
    theta_cutoff: float = Field(
        ge=0,
        le=180,
        description="Cutoff angle for the polar angle theta in degrees.")

    def __post_init__(self):
        self.img_centered_ds = self.validate_img_centered_ds(
            self.img_centered_ds)

    def validate_img_centered_ds(cls, v):
        if not isinstance(v, xr.Dataset):
            raise ValueError("Input must be an xarray.Dataset.")

        # Ensure x and y are centered
        def is_centered(data, tolerance=1):
            min_val = float(data.min().item())
            max_val = float(data.max().item())
            return abs(abs(min_val) - abs(max_val)) <= tolerance

        if "x" not in v or "y" not in v:
            raise ValueError("Dataset must contain 'x' and 'y' dimensions.")

        if not is_centered(v["x"]):
            raise ValueError("Dimension 'x' is not centered.")

        if not is_centered(v["y"]):
            raise ValueError("Dimension 'y' is not centered.")

        return v

    def convert_to_scs(self) -> xr.Dataset:
        """
        Convert the image pixel coordinates from CCS to SCS.

        Returns:
        --------
            xr.Dataset
                Dataset with the converted coordinates.
        """
        img_ds = self.img_centered_ds
        img_ds = self.__convert_to_spherical(img_ds)
        img_ds = self.__add_metadata(img_ds)

        self.img_centered_ds = img_ds
        return img_ds

    def __convert_to_spherical(self, img_ds: xr.Dataset) -> xr.Dataset:
        """
        Convert the image pixel coordinates from Cartesian Coordinate System (CCS)
        to Spherical Coordinate System (SCS).

        Parameters:
        ------------------
            img_ds : xr.Dataset
                Dataset with the image pixel coordinates.

        Returns:
        --------
            xr.Dataset
                Dataset augmented with spherical coordinates: `rho`, `phi`, and `theta`.
        """

    def __convert_to_spherical(self,
                               img_ds: xr.Dataset,
                               theta_cutoff: float = 67) -> xr.Dataset:
        """
        Convert the image pixel coordinates from Cartesian Coordinate System (CCS)
        to Spherical Coordinate System (SCS).

        Parameters:
        ------------------
            img_ds : xr.Dataset
                Dataset with the image pixel coordinates.
            theta_cutoff : float, optional
                Maximum polar angle (zenith) in degrees, default is 67.

        Returns:
        --------
            xr.Dataset
                Dataset augmented with spherical coordinates: `rho`, `phi`, `theta`.
        """
        # Cartesian coordinates (x, y)
        x = img_ds['x'].values.astype(
            np.float64)  # Ensure float64 for large numbers
        y = img_ds['y'].values.astype(np.float64)

        # Create Cartesian grid
        X, Y = np.meshgrid(x, y, indexing='ij')  # Full Cartesian grid

        # Calculate radial distance (rho)
        rho = np.sqrt(X**2 + Y**2)

        # Determine the circular ROI radius
        max_radius_cartesian = np.min([
            np.abs(np.max(x)),
            np.abs(np.min(x)),
            np.abs(np.max(y)),
            np.abs(np.min(y))
        ])
        theta_cutoff_rad = np.radians(
            theta_cutoff)  # Convert cutoff angle to radians
        rho_max = max_radius_cartesian * np.tan(
            theta_cutoff_rad)  # Effective maximum rho

        # Apply circular mask
        valid_mask = rho <= rho_max
        rho = np.where(valid_mask, rho, np.nan)  # Mask invalid regions

        # Azimuth angle (phi) in degrees
        phi = np.degrees(np.arctan2(Y, X))  # Range: [-180°, 180°]
        # phi = (phi + 360) % 360  # Normalize to range [0°, 360°]

        # Polar angle (theta) based on radial distance
        theta = np.degrees(
            np.arctan(rho / max_radius_cartesian * np.tan(theta_cutoff_rad)))
        theta = np.clip(theta, 0,
                        theta_cutoff)  # Ensure theta stays within valid bounds

        # Add spherical coordinates to the dataset
        img_ds = img_ds.assign_coords({
            'rho': (['x', 'y'], rho.astype(np.float32)),  # Radial distance
            'phi': (['x', 'y'], phi.astype(np.float32)),  # Azimuth angle
            'theta': (['x', 'y'], theta.astype(np.float32))  # Polar angle
        })

        return img_ds

    def __add_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add metadata to the image dataset.

        Parameters:
        ------------------
            ds : xr.Dataset
                Image dataset to add metadata to.

        Returns:
        --------
            xr.Dataset
                Image dataset with metadata added.
        """
        # Add metadata to spherical coordinate variables
        spherical_metadata = {
            "rho": {
                "long_name": "Radial distance",
                "short_name": "rho",
                "valid_min": 0,
                "units": "pixels",
                "description":
                "Distance from the origin in the Cartesian grid."
            },
            "phi": {
                "long_name":
                "Azimuth angle",
                "short_name":
                "phi",
                "valid_range": [-180, 180],
                "units":
                "degrees",
                "description":
                "Angle in the x-y plane measured counterclockwise from the positive x-axis."
            },
            "theta": {
                "long_name":
                "Polar angle",
                "short_name":
                "theta",
                "valid_range": [0, self.theta_cutoff],
                "units":
                "degrees",
                "description":
                "Angle measured from the z-axis to the point in spherical coordinates."
            }
        }

        for coord, attrs in spherical_metadata.items():
            if coord in ds.coords:
                ds.coords[coord].attrs.update(attrs)

        # Add global dataset metadata
        ds.attrs.update({
            "theta_cutoff":
            f"{self.theta_cutoff} degrees (based on camera specifications)",
            "description":
            "Dataset containing spherical coordinates derived from Cartesian image data."
        })

        return ds


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

if __name__ == '__main__':

    # Valid array
    valid_img = np.zeros((100, 100, 4), dtype=np.uint8)
    png_converter = Png2NetCDF(img_arr=valid_img)  # Passes
