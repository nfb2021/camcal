from dataclasses import dataclass, field

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


if __name__ == '__main__':
    monitor = Monitor()
    print(monitor.resolution)
