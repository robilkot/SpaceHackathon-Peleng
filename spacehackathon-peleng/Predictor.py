from typing import Callable

from Models.ObjectState import *
from Constants import *


def complete_object_state(current: ObjectState, info: dict[float, ObjectState]):
    prev = info.get(current.t - TIMESTEP, None)
    prev2 = info.get(current.t - TIMESTEP * 2, None)

    if prev is not None and prev2 is not None:
        prev.vel = [(prev.x - prev2.x) / TIMESTEP,
                    (prev.y - prev2.y) / TIMESTEP,
                    (prev.z - prev2.z) / TIMESTEP]

        if prev.vel is not None and prev2.vel is not None:
            prev.acc = [(prev.vel[0] - prev2.vel[0]) / TIMESTEP,
                        (prev.vel[1] - prev2.vel[1]) / TIMESTEP,
                        (prev.vel[2] - prev2.vel[2]) / TIMESTEP]

        if prev.acc is not None and prev2.acc is not None:
            prev.jerk = [(prev.acc[0] - prev2.acc[0]) / TIMESTEP,
                         (prev.acc[1] - prev2.acc[1]) / TIMESTEP,
                         (prev.acc[2] - prev2.acc[2]) / TIMESTEP]

    if prev is not None:
        if prev.acc is not None and prev.jerk is not None:
            # Update current acceleration with given jerk
            current.acc[0] = prev.acc[0] + prev.jerk[0]
            current.acc[1] = prev.acc[1] + prev.jerk[1]
            current.acc[2] = prev.acc[2] + prev.jerk[2]

        if prev.vel is not None and prev.acc is not None:
            # Update current velocity with given acceleration
            current.vel[0] = prev.vel[0] + prev.acc[0]
            current.vel[1] = prev.vel[1] + prev.acc[1]
            current.vel[2] = prev.vel[2] + prev.acc[2]

        if prev.vel is not None:
            # Update current coords with given velocity
            current.x = prev.x + current.vel[0]
            current.y = prev.y + current.vel[1]
            current.z = prev.z + current.vel[2]


def raise_if_completed(s: ObjectState, callback: Callable[[ObjectState], None]):
    # todo: dl_max???
    if not [x for x in (s.x, s.y, s.z, s.vel, s.acc) if x is None]:
        # print(f"info collected for {s.t}")
        callback(s)
