import math
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
        if self.info[current.t - self.TIMESTEP] is not None:
            prev = self.info[current.t - self.TIMESTEP]

            velocity = math.sqrt(
                (current.x - prev.x) ** 2 + (current.y - prev.y) ** 2 + (current.z - prev.z) ** 2
            ) / self.TIMESTEP

            current.vel = velocity

        # todo calculate acc

    def __raise_if_completed(self, state: ObjectState):
        if state.dl_max is not None and state.acc is not None and state.vel is not None:
            self.on_info_collected(state)
