from typing import Callable, Iterable


import numpy as np
import numpy.linalg as npla
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.DetectionMessage import DetectionMessage, ObjNotDetectedMessage
from Models.Camera import Camera


def midpoint_triangulate(camera_point: dict[Camera, np.ndarray]):
    n = len(camera_point)                                         # No. of cameras

    I = np.eye(3)                                        # 3x3 identity matrix
    A = np.zeros((3,n))
    B = np.zeros((3,n))
    sigma2 = np.zeros((3,1))

    for i, camera in enumerate(camera_point):
        a = -np.transpose(camera.R).dot(camera.T)        # ith camera position
        A[:,i,None] = a

        b = npla.inv(camera.P).dot(camera_point[camera])              # Directional vector
        b = b / b[3]
        b = b[:3,None] - a
        b = b / npla.norm(b)
        B[:,i,None] = b

        sigma2 = sigma2 + b.dot(b.T.dot(a))

    C = (n * I) - B.dot(B.T)
    Cinv = npla.inv(C)
    sigma1 = np.sum(A, axis=1)[:,None]
    m1 = I + B.dot(np.transpose(B).dot(Cinv))
    m2 = Cinv.dot(sigma2)

    midpoint = (1/n) * m1.dot(sigma1) - m2        
    return midpoint


class Scene:
    def __init__(
        self,
        cam_ids: list[int],
    ) -> None:
        self._cam_dict: dict[int, np.ndarray] = dict.fromkeys(cam_ids, None)
    
    def is_full(self):
        for i, cam in self._cam_dict.items():
            print(i, cam is None)
            if cam is None:
                return False
        return True
    
    def __iter__(self) -> Iterable[tuple[int, np.ndarray]]:
        return iter(self._cam_dict.values())
    
    def __getitem__(sefl, id):
        return sefl._cam_dict[id]
    
    def __setitem__(self, id, position):
        self._cam_dict[id] = position
        
        
from dataclasses import dataclass


@dataclass
class MatCamera:
    P: np.matrix
    R: np.matrix
    T: np.matrix
    width: float
    height: float
    resolution: tuple[int, int]


class Triangulator:
    def __init__(self,
                 cams: dict[int, Camera],
                 on_triangulated: Callable[[CoordinatesTriangulatedMessage], None]
    ) -> None:
        self.cams = dict.fromkeys(cams.keys())
        
        for id, cam in cams.items():
            R = np.asarray(
                [
                    [np.cos(cam.a), -np.sin(cam.a), 0],
                    [np.sin(cam.a), np.cos(cam.a), 0],
                    [0, 0, 1],
                ]
            )
            T = np.asarray([cam.x, cam.y, cam.z])
            P = np.hstack((R, T.reshape((-1, 1))))
            self.cams[id] = MatCamera(P, R, T, cam.matrix_w, cam.matrix_h, (cam.res_w, cam.res_h))
        
        self.on_triangulated = on_triangulated
        self.scenes = {}
    
    def transform(self, mes: DetectionMessage | ObjNotDetectedMessage):
        if isinstance(mes, ObjNotDetectedMessage):
            print("Triangulator: ObjNotDetectedMessage instance has received")
            return CoordinatesTriangulatedMessage(
                None,
                None,
                None,
                mes.t,
                None
            )
        scene = self.scenes.get(mes.t, Scene(self.cams.keys()))

        cam = self.cams[mes.cam_id]
        scene[mes.cam_id] = np.asarray(
            (
                (mes.x - mes.w / 2) / cam.resolution[0] * cam.width,
                -(mes.y - mes.h / 2) / cam.resolution[1] * cam.height
            )
        )
        self.scenes[mes.t] = scene

        if scene.is_full():
            points = np.asarray([x for x in self.scenes.values()])
            
            midpoint = midpoint_triangulate(points, self.cams.values())
            
            msg = CoordinatesTriangulatedMessage(
                *midpoint,
                mes.t,
                0 # TODO
            )
            print("on_triangulated")
            self.on_triangulated(msg)
            
