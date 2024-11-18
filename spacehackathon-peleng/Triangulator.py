from typing import Callable

import numpy as np
from openpyxl.drawing.geometry import Camera
from .Models.CoordsMessages import *


class Triangulator:
    def __init__(self,
                 cums: dict[int, Camera],
                 resolution: tuple[int, int],
                 on_triangulated: Callable[[CoordinatesTrackedMessage], None]) -> None:
        self.cums = cums
        self.resolution = np.asarray(resolution)
        self.on_triangulated = on_triangulated
    
    def transform(self, mes: ObjDetectedMessage):
        cum_point = np.asarray(mes.x + mes.w / 2, mes.y + mes.h / 2)
        rel_point = cum_point / self.resolution - 0.5

        msg = CoordinatesTrackedMessage()  # todo: this is result msg
        self.on_triangulated(msg)
