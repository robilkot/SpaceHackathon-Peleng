from dataclasses import dataclass


@dataclass
class ObjectState:
    x: float | None
    y: float | None
    z: float | None
    t: float
    dl_max: float | None  # todo
    vel: tuple[float, float, float] | None
    acc: tuple[float, float, float] | None
