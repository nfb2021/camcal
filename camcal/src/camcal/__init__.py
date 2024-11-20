import importlib
import pkgutil
from typing import List

from camcal.aperture_cropping import ImageOrienter
from camcal.cam_cal import CamAngleOffset
from camcal.camera import CameraLocationInfo
from camcal.cmaps import FancyRGBCmap, RealRGBCmap
from camcal.image import ImageReader
from camcal.pairing import PairFactory
from camcal.process_manager import ImageProcessingManager
from camcal.processing import CCS2SCS, Png2NetCDF

__all__: List[str] = [
    "ImageOrienter", "CameraLocationInfo", "PairFactory",
    "ImageProcessingManager", "CamAngleOffset", "FancyRGBCmap", "RealRGBCmap",
    "ImageReader", "Png2NetCDF", "CCS2SCS"
]

# # Iterate through all modules in the package
# for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
#     module = importlib.import_module(f"{__name__}.{module_name}")

#     # If the module has an `__all__`, extend and expose symbols
#     if hasattr(module, "__all__"):
#         __all__.extend(module.__all__)
#         for name in module.__all__:
#             try:
#                 value = getattr(module, name)
#                 if callable(value) or isinstance(value, (type, object)):
#                     globals()[name] = value
#             except AttributeError as e:
#                 print(f"Error accessing {name} in {module_name}: {e}")
#     else:
#         # Expose all public attributes (non-underscore-prefixed)
#         for attr in dir(module):
#             if not attr.startswith("_"):
#                 try:
#                     value = getattr(module, attr)
#                     # Ensure it's not importing something unintended (e.g., modules)
#                     if callable(value) or isinstance(value, (type, object)):
#                         globals()[attr] = value
#                         __all__.append(attr)
#                 except AttributeError as e:
#                     print(f"Error accessing {attr} in {module_name}: {e}")
