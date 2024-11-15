from pathlib import Path
from pydantic.dataclasses import dataclass
import polars as pl


@dataclass
class CamAngleOffset:
    """Stores camera calibration data such as mean, median, and standard deviation of angle offsets.

    Parameters:
    -----------
        cam_name : str
            Name of the camera.
        mean : float
            Mean angle offset.
        median : float
            Median angle offset.
        std : float
            Standard deviation of angle offsets.
        n : int
            Number of samples.
    """

    cam_name: str
    mean: float
    median: float
    std: float
    n: int

    @classmethod
    def from_parquet(cls, fpath: Path, camera_name: str) -> "CamAngleOffset":
        """
        Load camera calibration data from a parquet file.

        Parameters:
        ------------------
            fpath : Path
                Path to the parquet file.
            camera_name : str
                Name of the camera to retrieve data for.

        Returns:
        --------
            CamAngleOffset
                An instance of CamAngleOffset with loaded data.
        """
        data = pl.read_parquet(fpath)
        if camera_name not in data['camera'].to_list():
            raise ValueError(
                f"Camera name {camera_name} not found in parquet file. Available are: {data['camera'].unique()}"
            )

        cam_data = data.filter(pl.col("camera") == camera_name)
        return cls(cam_name=cam_data['camera'].item(),
                   mean=cam_data['mean [deg]'].item(),
                   median=cam_data['median [deg]'].item(),
                   std=cam_data['std [deg]'].item(),
                   n=cam_data['N'].item())
