from dataclasses import dataclass


'''
Каждый кадр леша отправляет ObjDetectedMessage для каждой камеры где обнаружен объект
Егор принимает мессаджи, трианглуирует координаты, формирует сообщение CoordinatesTrackedMessage
и отправляет Тимуру для записи в итоговый док

Если на какой-то камере объект пропал, то леша НЕ отправляет сообщение для этой камеры на этом кадре
угадывание траектории пропавшего объекта сдлеаем позже на стороне леши 
'''

@dataclass
class ObjDetectedMessage:
    cam_id: int
    x: int  # coordinates in image
    y: int
    w: int
    h: int
    t: float  # time in seconds at which the detection happened
