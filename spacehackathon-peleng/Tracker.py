from dataclasses import dataclass

import numpy as np

from .CoordsMessages import *


@dataclass
class Camera:
    focal_length: float
    x: float
    y: float
    z: float
    a: float # азимут
    width: float
    height: float


class Tracker:
    def __init__(
        self, 
        cums: dict[int, Camera],
        resolution: tuple[int, int]
) -> None:
        self.cums = cums
        self.resolution = np.asarray(resolution)
    
    def __call__(self, mes: ObjDetectedMessage) -> CoordinatesTrackedMessage:
        cum_point = np.asarray(mes.x + mes.w / 2, mes.y + mes.h / 2)
        rel_point = cum_point / self.resolution - 0.5
          
