from typing import Callable

from Models.ObjectState import *
from Constants import *


def predict_location(timestamp: float, info: dict[float, ObjectState]) -> tuple[float | None, float | None]:
    pr1 = info.get(timestamp - TIMESTEP * 2, None)

    if pr1 is None or pr1.vel is None:
        return None, None

    return pr1.x + pr1.vel[0], pr1.y + pr1.vel[1]


def complete_object_state(cur: ObjectState, info: dict[float, ObjectState]):
    pr1 = info.get(cur.t - TIMESTEP * 2, None)
    # print(f"cur: {cur}")

    if pr1 is None or pr1.x is None or cur.x is None:
        return

    cur.vel = [(cur.x - pr1.x) / TIMESTEP,
               (cur.y - pr1.y) / TIMESTEP,
               (cur.z - pr1.z) / TIMESTEP]

    if cur.vel is not None and pr1.vel is not None:
        cur.acc = [(cur.vel[0] - pr1.vel[0]) / TIMESTEP,
                   (cur.vel[1] - pr1.vel[1]) / TIMESTEP,
                   (cur.vel[2] - pr1.vel[2]) / TIMESTEP]

    if cur.acc is not None and pr1.acc is not None:
        cur.jrk = [(cur.acc[0] - pr1.acc[0]) / TIMESTEP,
                   (cur.acc[1] - pr1.acc[1]) / TIMESTEP,
                   (cur.acc[2] - pr1.acc[2]) / TIMESTEP]


def raise_if_completed(s: ObjectState, callback: Callable[[ObjectState], None]):
    # todo: dl_max???
    if not [x for x in (s.x, s.y, s.z, s.vel, s.acc) if x is None]:
        # print(f"info collected for {s.t}")
        callback(s)
