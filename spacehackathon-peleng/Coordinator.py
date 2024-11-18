from typing import Callable
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.ObjectState import ObjectState


class Coordinator:
    TIMESTEP: float = 0.5

    def __init__(self, on_info_collected: Callable[[ObjectState], None]) -> None:
        self.on_info_collected = on_info_collected
        self.info: dict[float, ObjectState] = {}

    def accept(self, msg: CoordinatesTriangulatedMessage) -> None:
        # todo make sure x y z can be nullable if object disappeared from view
        state = ObjectState(msg.x, msg.y, msg.z, msg.t, None, None, None)

        self.info[msg.t] = state

        self.__complete_object_state(state)
        self.__raise_if_completed(state)

    def __complete_object_state(self, current: ObjectState):
        prev = self.info[current.t - self.TIMESTEP]
        prev2 = self.info[current.t - self.TIMESTEP * 2]

        if prev is not None and prev2 is not None:
            velocity = [(prev.x - prev2.x) / self.TIMESTEP,
                        (prev.y - prev2.y) / self.TIMESTEP,
                        (prev.z - prev2.z) / self.TIMESTEP]

            prev.vel = velocity

        if prev.vel is not None and prev2.vel is not None:
            acc = [(prev.vel[0] - prev2.vel[0]) / self.TIMESTEP,
                   (prev.vel[1] - prev2.vel[1]) / self.TIMESTEP,
                   (prev.vel[2] - prev2.vel[2]) / self.TIMESTEP]

            prev.acc = acc

        # todo extrapolate x y z for current given acc and vel if not present yet

    def __raise_if_completed(self, s: ObjectState):
        # todo: dl_max???
        if None not in (s.x, s.y, s.z, s.vel, s.acc):
            self.on_info_collected(s)
