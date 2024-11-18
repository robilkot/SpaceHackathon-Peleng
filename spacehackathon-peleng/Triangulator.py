from typing import Callable

import numpy as np
from openpyxl.drawing.geometry import Camera
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.ObjDetectedMessage import ObjDetectedMessage


class Triangulator:
    def __init__(self,
                 cums: dict[int, Camera],
                 resolution: tuple[int, int],
                 on_triangulated: Callable[[CoordinatesTriangulatedMessage], None]) -> None:
        self.cums = cums
        self.resolution = np.asarray(resolution)
        self.on_triangulated = on_triangulated
    
    def transform(self, mes: ObjDetectedMessage):

        msg = CoordinatesTriangulatedMessage()  # todo: this is result msg
        self.on_triangulated(msg)
