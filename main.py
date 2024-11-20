import multiprocessing
from datetime import datetime
from pathlib import Path

from camcal.src.camera import CameraLocationInfo
from camcal.src.pairing import PairFactory
from camcal.src.process_manager import ImageProcessingManager

if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Define paths for IR and VIS images
    ir_pth = Path(
        '/home/nbader/shares/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/02_IR/24284'
    )
    vis_pth = Path(
        '/home/nbader/shares/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/01_VIS/24284'
    )

    ir_imgs = list(ir_pth.glob('*.jpg'))
    vis_imgs = list(vis_pth.glob('*.jpg'))

    print(f"Found {len(ir_imgs)} IR images and {len(vis_imgs)} VIS images.")

    # Camera location information
    camera_loc = CameraLocationInfo(
        name="Rosalia",
        region="Austria",
        tz="Europe/Vienna",
        lat=47.707433,
        lon=16.299553,
    )

    # Camera calibration file
    camera_calib_file = Path(
        '/home/nbader/shares/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/03_camera_calib/camera_calib.parquet'
    )

    # Initialize PairFactory and create image pairs
    pf = PairFactory(ir_pths=ir_pth, vis_pths=vis_pth, camera_loc=camera_loc)
    pairs = pf.create_pairs()

    print(f"Created {len(pairs)} image pairs.")
    print(pairs)

    # # Initialize the ImageProcessingManager
    # manager = ImageProcessingManager(
    #     image_pairs=pairs,
    #     offset_file=camera_calib_file,
    # )

    # # Combine all cropped images into a single dataset
    # final_dataset = manager.combine_to_final_dataset()
    # print("Final dataset created.")
    # print(final_dataset)
    # final_dataset.to_netcdf(
    #     "/shares/nbader/climers/Projects/GNSS_Vegetation_Study/07_data/01_Rosalia/02_canopy/02_hem_img/final_dataset.nc"
    # )
