from dataclasses import dataclass


@dataclass
class DetectionMessage:
    cam_id: int
    t: float  # time in seconds at which the detection happened


@dataclass
class ObjDetectedMessage(DetectionMessage):
    x: int  # coordinates in image
    y: int
    w: int
    h: int


@dataclass
class ObjNotDetectedMessage(DetectionMessage):
    pass
