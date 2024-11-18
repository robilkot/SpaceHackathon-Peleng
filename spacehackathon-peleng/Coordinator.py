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

        # Not optimal
        for obj in self.info.values():
            self.__complete_object_state(obj)

        for obj in self.info.values():
            self.__raise_if_completed(obj)

    def __complete_object_state(self, current: ObjectState):
        prev = self.info[current.t - self.TIMESTEP]
        prev2 = self.info[current.t - self.TIMESTEP * 2]

        # todo 3rd derivative for case when sphere stalled

        if prev is not None and prev2 is not None:
            prev.vel = [(prev.x - prev2.x) / self.TIMESTEP,
                        (prev.y - prev2.y) / self.TIMESTEP,
                        (prev.z - prev2.z) / self.TIMESTEP]

        if prev.vel is not None and prev2.vel is not None:
            prev.acc = [(prev.vel[0] - prev2.vel[0]) / self.TIMESTEP,
                        (prev.vel[1] - prev2.vel[1]) / self.TIMESTEP,
                        (prev.vel[2] - prev2.vel[2]) / self.TIMESTEP]

        # Update current velocity with given acceleration
        current.vel[0] = prev.vel[0] + prev.acc[0]
        current.vel[1] = prev.vel[1] + prev.acc[1]
        current.vel[2] = prev.vel[2] + prev.acc[2]

        # Update current coords with given velocity
        current.x = prev.x + current.vel[0]
        current.y = prev.y + current.vel[1]
        current.z = prev.z + current.vel[2]

    def __raise_if_completed(self, s: ObjectState):
        # todo: dl_max???
        if not [x for x in (s.x, s.y, s.z, s.vel, s.acc) if x is None]:
            self.on_info_collected(s)
