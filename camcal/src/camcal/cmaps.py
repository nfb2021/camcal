from typing import Dict

import matplotlib.colors as mcolors
from pydantic.dataclasses import dataclass

__all__ = ["RealRGBCmap", "FancyRGBCmap"]


@dataclass(config={'arbitrary_types_allowed': True})
class RGBCmaps:
    """
    Container for realistic RGB colormaps with true color tones.
    """
    red: mcolors.LinearSegmentedColormap
    green: mcolors.LinearSegmentedColormap
    blue: mcolors.LinearSegmentedColormap

    def __post_init__(self, **kwargs: Dict[str,
                                           mcolors.LinearSegmentedColormap]):
        cmaps = [self.red, self.green, self.blue]
        # Ensure only the value 0 is black, and 1 starts the color gradient
        for cmap in cmaps:
            cmap.set_under(color='black')  # Only for values < 0
            cmap.set_bad(color='black')  # For NaN or invalid values

        self.__dict__.update(kwargs)

    def __getitem__(self, key: str):
        return self.__dict__[key]


# Realistic RGB colormaps with true color tones
real_red_cmap = mcolors.LinearSegmentedColormap.from_list(
    "real_red",
    ["#000000", "#8B0000", "#FF0000", "#FF7F7F"
     ]  # black, dark red, red, light red
)
real_green_cmap = mcolors.LinearSegmentedColormap.from_list(
    "real_green",
    ["#000000", "#006400", "#00FF00", "#7FFF7F"
     ]  # black, dark green, green, light green
)
real_blue_cmap = mcolors.LinearSegmentedColormap.from_list(
    "real_blue",
    ["#000000", "#00008B", "#0000FF", "#7F7FFF"
     ]  # black, dark blue, blue, light blue
)

import matplotlib.colors as mcolors

# Define pastel-inspired colormaps with modern tones
fancy_red_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fancy_red", ["#000000", "#5B0000", "#C04040", "#FF9999", "#FFCCCC"],
    N=256)

fancy_green_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fancy_green", ["#000000", "#004D00", "#4CAF50", "#99FF99", "#D9FFD9"],
    N=256)

fancy_blue_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fancy_blue",
    ["#000000", "#000033", "#00008B", "#1E90FF", "#87CEEB", "#B0E0E6"],
    N=256)

RealRGBCmap = RGBCmaps(red=real_red_cmap,
                       green=real_green_cmap,
                       blue=real_blue_cmap)

FancyRGBCmap = RGBCmaps(red=fancy_red_cmap,
                        green=fancy_green_cmap,
                        blue=fancy_blue_cmap)
