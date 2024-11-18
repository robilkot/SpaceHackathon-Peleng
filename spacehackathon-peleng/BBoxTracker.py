from typing import Callable
from .Models.ObjDetectedMessage import *
from .BBoxTracking import *

PATH_RGB = 'data/videoset1/Seq1_camera1.mov'
PATH_IR = 'data/videoset1/Seq1_camera1T.mov'


class BBoxTracker:
    def __init__(self, on_tracked: Callable[[ObjDetectedMessage], None]) -> None:
        self.on_tracked = on_tracked

    def __track(self) -> None:
        msg = ObjDetectedMessage()  # todo result
        self.on_tracked(msg)

    def start(self):
        # todo process frames here in loop, pass filenames? передать пути
        return process_videos(PATH_RGB, PATH_IR)

