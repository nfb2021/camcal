from datetime import datetime
from typing import Dict

from astral import LocationInfo
from astral.sun import sun
from pydantic.dataclasses import dataclass
from pytz import timezone


@dataclass
class CameraLocationInfo:
    """
    Container for camera location information.

    Parameters:
    ------------
    name : str
        Name of the location.
    region : str
        Region of the location.
    tz : str
        Timezone of the location (e.g., 'Europe/Vienna').
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    """
    name: str
    region: str
    tz: str
    lat: float
    lon: float

    def __post_init__(self):
        self.loc = LocationInfo(self.name, self.region, self.tz, self.lat,
                                self.lon)
        self.timezone_obj = timezone(self.tz)

    def get_sun_times(self, date: datetime) -> Dict[str, datetime]:
        """
        Get the sunrise and sunset times for the given date.

        Parameters:
        ------------
        date : datetime
            Date for which to compute the sunrise and sunset times.

        Returns:
        ---------
        Dict[str, datetime]:
            A dictionary with 'sunrise' and 'sunset' keys containing datetime objects.
        """
        sun_times = sun(self.loc.observer, date=date.date())
        return {
            "sunrise": sun_times["sunrise"].astimezone(self.timezone_obj),
            "sunset": sun_times["sunset"].astimezone(self.timezone_obj),
        }
