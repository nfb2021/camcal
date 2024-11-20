import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import numpy as np
from screeninfo import get_monitors


@dataclass
class Monitor:
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.monitors = get_monitors()
        for monitor in self.monitors:
            if monitor.is_primary:
                self.width = monitor.width
                self.height = monitor.height
                self.resolution = (self.width, self.height)


def get_resized_img_shape(img: np.ndarray,
                          scale: float = 0.7) -> Dict[str, float]:
    screen_res = Monitor().resolution

    scale_width = scale * screen_res[0] / img.shape[1]
    scale_height = scale * screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    return {'width': window_width, 'height': window_height}


def exif_str_to_snake_case(input_string: str) -> str:
    # Remove the "EXIF" prefix and trim whitespace
    cleaned_string = input_string.replace("EXIF", "").strip()
    # Convert to snake case
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', cleaned_string).lower()
    return 'exif_' + snake_case


def strip_timezone(dt: datetime) -> datetime:
    """Remove timezone information from a datetime object."""
    return dt.replace(tzinfo=None)


def to_nanosecond_precision(timestamp: Any) -> np.datetime64:
    """
    Convert a timestamp to nanosecond precision if it's not already.
    """
    return timestamp.astype("datetime64[ns]")


if __name__ == '__main__':
    monitor = Monitor()
    print(monitor.resolution)
