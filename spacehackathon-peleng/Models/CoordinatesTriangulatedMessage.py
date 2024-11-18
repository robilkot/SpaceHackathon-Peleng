from dataclasses import dataclass


@dataclass
class CoordinatesTriangulatedMessage:
    x: float
    y: float
    z: float
    t: float
    dl_max: float