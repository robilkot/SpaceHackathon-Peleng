from dataclasses import dataclass


@dataclass
class Camera:
    focal_length: float
    x: float
    y: float
    z: float
    a: float  # азимут
    width: float
    height: float
