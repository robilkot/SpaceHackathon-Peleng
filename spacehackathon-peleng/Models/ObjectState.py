from dataclasses import dataclass


@dataclass
class ObjectState:
    x: float
    y: float
    z: float
    t: float
    dl_max: float | None  # todo
    vel: tuple[float, float, float] | None
    acc: tuple[float, float, float] | None
