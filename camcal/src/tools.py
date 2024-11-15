from dataclasses import dataclass, field
from typing import Dict

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


if __name__ == '__main__':
    monitor = Monitor()
    print(monitor.resolution)
