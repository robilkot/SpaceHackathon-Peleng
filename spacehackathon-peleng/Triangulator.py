from typing import Callable, Iterable


import numpy as np
import numpy.linalg as npla
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.ObjDetectedMessage import ObjDetectedMessage
from Models.Camera import Camera




def midpoint_triangulate(points: np.ndarray, cameras: list[Camera]):
    """
    Args:
        x:   Set of 2D points in homogeneous coords, (3 x n) matrix
        cam: Collection of n objects, each containing member variables
                 cam.P - 3x4 camera matrix
                 cam.R - 3x3 rotation matrix
                 cam.T - 3x1 translation matrix
    Returns:
        midpoint: 3D point in homogeneous coords, (4 x 1) matrix
    """

    n = len(cameras)                                         # No. of cameras

    I = np.eye(3)                                        # 3x3 identity matrix
    A = np.zeros((3,n))
    B = np.zeros((3,n))
    sigma2 = np.zeros((3,1))

    for i in range(n):
        a = -np.transpose(cameras[i].R).dot(cameras[i].T)        # ith camera position
        A[:,i,None] = a

        b = npla.pinv(cameras[i].P).dot(points[:,i])              # Directional vector
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
        return all(self._cam_dict.values)
    
    def __iter__(self) -> Iterable[tuple[int, np.ndarray]]:
        return iter(self._cam_dict.values)
    
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


class Triangulator:
    def __init__(self,
                 cams: dict[int, Camera],
                 on_triangulated: Callable[[CoordinatesTriangulatedMessage], None]
    ) -> None:
        self.cams = dict.fromkeys(cams.keys)
        
        for id, cam in cams.items:
            R = np.asarray(
                [
                    [np.cos(cam.a), -np.sin(cam.a), 0],
                    [np.sin(cam.a), np.cos(cam.a), 0],
                    [0, 0, 1],
                ]
            )
            T = np.asarray([cam.x, cam.y, cam.z])
            P = np.vstack((R, T.reshape((-1, 1))))
            self.cams[id] = MatCamera(P, R, T)
        
        self.on_triangulated = on_triangulated
        self.scenes = {}
    
    def transform(self, mes: ObjDetectedMessage):
        scene = self.scenes[mes.cam_id] if mes.cam_id in self.scenes else Scene(self.cams.keys) 
        scene[mes.cam_id] = np.asarray((mes.w + mes.x / 2, mes.h + mes.y / 2))
        if mes.cam_id in self.scenes:
            self.scenes[mes.cam_id] = scene
        
        if scene.is_full():
            points = np.asarray([x for x in self.scenes.values])
            
            midpoint = midpoint_triangulate(points, self.cams.values)
            
            msg = CoordinatesTriangulatedMessage(
                *midpoint,
                mes.t,
                0 # TODO
            )  
            self.on_triangulated(msg)
            
