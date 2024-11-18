from dataclasses import dataclass


@dataclass
class ObjectState:
    x: float
    y: float
    z: float
    t: float
    dl_max: float  # todo
    vel: tuple[float, float, float]
    acc: tuple[float, float, float]
