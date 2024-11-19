from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Predictor import *


class Coordinator:
    def __init__(self, on_info_collected: Callable[[ObjectState], None]) -> None:
        self.on_info_collected = on_info_collected
        self.info: dict[float, ObjectState] = {}

    def accept(self, msg: CoordinatesTriangulatedMessage) -> None:
        state = ObjectState(msg.x, msg.y, msg.z, msg.t, None, None, None, None)
        self.info[msg.t] = state

        # Not optimal
        for obj in self.info.values():
            complete_object_state(obj, self.info)

        for obj in self.info.values():
            raise_if_completed(obj, self.on_info_collected)
