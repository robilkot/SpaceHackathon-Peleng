from typing import Callable
from .Models.CoordsMessages import *


class BBoxTracker:
    def __init__(self, on_tracked: Callable[[ObjDetectedMessage], None]) -> None:
        self.on_tracked = on_tracked

    def __track(self) -> None:
        msg = ObjDetectedMessage()  # todo result
        self.on_tracked(msg)

    def start(self):
        # todo process frames here in loop, pass filenames?
        pass
