from dataclasses import dataclass


@dataclass
class Camera:
    focal_length: float
    x: float
    y: float
    z: float
    a: float
    matrix_w: float
    matrix_h: float
    res_w: int
    res_h: int
