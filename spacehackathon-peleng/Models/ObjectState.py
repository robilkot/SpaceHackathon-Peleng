from dataclasses import dataclass


@dataclass
class ObjectState:
    x: float | None
    y: float | None
    z: float | None
    t: float
    dl_max: float | None
    vel: list[float] | None
    acc: list[float] | None
