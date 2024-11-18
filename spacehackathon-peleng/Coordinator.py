from typing import Callable
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.ObjectState import ObjectState


class Coordinator:
    def __init__(self, on_info_collected: Callable[[ObjectState], None]) -> None:
        self.on_info_collected = on_info_collected

        self.info: dict[float, ObjectState] = {}

    def accept(self, msg: CoordinatesTriangulatedMessage) -> None:
        if self.info[msg.t] is None:
            self.info[msg.t] = [msg]
        else:
            self.info[msg.t].append(msg)

        # Проверить,

        result_coords = CoordinatesTriangulatedMessage()  # todo
        self.on_info_collected(result_coords)  # Когда вся инфа для назождения координат собрана, вызваем ивент
