from typing import Callable
from .Models.ObjDetectedMessage import *
from .BBoxTracking import *

PATH_RGB = 'data/videoset1/Seq1_camera1.mov'
PATH_IR = 'data/videoset1/Seq1_camera1T.mov'


class BBoxTracker:
    def __init__(self, on_tracked: Callable[[ObjDetectedMessage], None]) -> None:
        self.on_tracked = on_tracked

    def __track(self) -> None:
        msg = ObjDetectedMessage()  # todo result
        self.on_tracked(msg)

    def start(self):
        '''
        
        Создать по одному потоку для каждой камеры
        В каждом потоке:
            находим разность текущего и прошлого кадров РГБ
            находим разность текущего и прошлого кадров ИК
            находим ббоксы на РГБ
            находим ббоксы на ИК
            опицонально: отсеиваем слишком большие ббоксы на РГБ
            опицонально: отсеиваем слишком большие ббоксы на ИК
            находим пару ббоксов из РГБ и ИК, которые пересекаются
            если пара найдена, создаем ObjDetectedMessage на основе ббокса на ИК !
            вызываем self.on_tracked(msg) где мсг - новое сообщение

        '''

        # return process_videos(PATH_RGB, PATH_IR)

