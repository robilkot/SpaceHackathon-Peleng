from typing import Callable
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage


class Coordinator:
    # Внутри должен быть словарь время-инфа об объекте
    # и по мере получения всей нужной инфы об объекте запись из словаря удаляется
    # и записывается в файл
    def __init__(self, on_info_collected: Callable[[CoordinatesTriangulatedMessage], None]) -> None:
        self.on_info_collected = on_info_collected

    def accept(self, msg: CoordinatesTriangulatedMessage) -> None:
        # todo сделать по комментарию выше

        result_coords = CoordinatesTriangulatedMessage()  # todo
        self.on_info_collected(result_coords)  # Когда вся инфа для назождения координат собрана, вызваем ивент
