from typing import Callable, Iterable
from dataclasses import dataclass


import numpy as np
import numpy.linalg as npla
from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.DetectionMessage import DetectionMessage, ObjNotDetectedMessage
from Models.Camera import Camera


@dataclass()
class MatCamera:
    P: np.matrix
    R: np.matrix
    T: np.matrix
    width: float
    height: float
    resolution: tuple[int, int]


# def midpoint_triangulate(cameras: dict[int, MatCamera], points: dict[int, np.ndarray]):
#     n = len(cameras)                                         # No. of cameras
#
#     I = np.eye(3)                                        # 3x3 identity matrix
#     A = np.zeros((3,n))
#     B = np.zeros((3,n))
#     sigma2 = np.zeros((3,1))
#
#     for i, camera in cameras.items():
#         a = -np.transpose(camera.R).dot(camera.T)        # ith camera position
#         A[:,i - 1,None] = a.reshape((-1, 1))
#
#         b = npla.pinv(camera.P).dot(points[i])          # Directional vector
#         b = b / b[3]
#         b = b[:3,None] - a
#         b = b / npla.norm(b)
#         B[:,i - 1,None] = b
#
#         sigma2 = sigma2 + b.dot(b.T.dot(a))
#
#     C = (n * I) - B.dot(B.T)
#     Cinv = npla.inv(C)
#     sigma1 = np.sum(A, axis=1)[:,None]
#     m1 = I + B.dot(np.transpose(B).dot(Cinv))
#     m2 = Cinv.dot(sigma2)
#
#     midpoint = (1/n) * m1.dot(sigma1) - m2
#     return midpoint


def midpoint_triangulate(x, cam):
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

    n = len(cam)                                         # No. of cameras

    I = np.eye(3)                                        # 3x3 identity matrix
    A = np.zeros((3,n))
    B = np.zeros((3,n))
    sigma2 = np.zeros((3,1))

    for i in range(n):
        a = -np.transpose(cam[i].R).dot(cam[i].T).reshape((-1, 1))        # ith camera position
        A[:,i,None] = a

        b = npla.pinv(cam[i].P).dot(x[:,i])              # Directional vector
        print(b)
        b = b / b[3]
        print(b)
        b = b[:3,None] - a
        print(b)
        b = b / npla.norm(b)
        print(b)
        B[:,i,None] = b

        sigma2 = sigma2 + b.dot(b.T.dot(a))

    C = (n * I) - B.dot(B.T)
    Cinv = npla.inv(C)
    sigma1 = np.sum(A, axis=1)[:,None]
    m1 = I + B.dot(np.transpose(B).dot(Cinv))
    m2 = Cinv.dot(sigma2)

    midpoint = (1/n) * m1.dot(sigma1) - m2
    return np.vstack((midpoint, 1))


class Scene:
    def __init__(
        self,
        cam_ids: list[int],
    ) -> None:
        self.cam_dict: dict[int, np.ndarray] = dict.fromkeys(cam_ids, None)
    
    def is_full(self):
        for i, cam in self.cam_dict.items():
            print(i, cam is None)
            if cam is None:
                return False
        return True
    
    def __iter__(self) -> Iterable[tuple[int, np.ndarray]]:
        return iter(self.cam_dict.items())
    
    def __getitem__(self, id):
        return self.cam_dict[id]
    
    def __setitem__(self, id, position):
        self.cam_dict[id] = position


class Triangulator:
    def __init__(self,
                 cams: dict[int, Camera],
                 on_triangulated: Callable[[CoordinatesTriangulatedMessage], None]
    ) -> None:
        self.cams: dict[int, MatCamera] = dict.fromkeys(cams.keys())
        
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
    
    def transform(self, mes: DetectionMessage):
        scene = self.scenes.get(mes.t, Scene(self.cams.keys()))
        cam = self.cams[mes.cam_id]
        if isinstance(mes, ObjNotDetectedMessage):
            scene[mes.cam_id] = 0
            print("Triangulator: ObjNotDetectedMessage instance has received")
        else:
            scene[mes.cam_id] = np.asarray(
                (
                    (mes.x - mes.w / 2) / cam.resolution[0] * cam.width,
                    -(mes.y - mes.h / 2) / cam.resolution[1] * cam.height
                )
            )
        self.scenes[mes.t] = scene

        print(f"Triangulator: DetectionMessage instance has received. {mes.cam_id = }: ", scene[mes.cam_id], hash(scene))
        print(f"Scene {mes.t} {scene.is_full()}")
        print('\n'.join(f"{i}: {x is not None}" for i, x in scene))

        if scene.is_full():
            for _, point in scene:
                if point is 0:
                    print("Triangulator: CoordinatesTriangulatedMessage with none has sent")
                    msg = CoordinatesTriangulatedMessage(
                        None,
                        None,
                        None,
                        mes.t,
                        None
                    )
                    self.on_triangulated(msg)
                    return

            ids = [x for x in self.cams]
            points = np.asarray([[*scene[x], 1] for x in ids])
            cams = np.asarray([self.cams[x] for x in ids])
            midpoint = midpoint_triangulate(points, cams)
            msg = CoordinatesTriangulatedMessage(
                float(midpoint[0]),
                float(midpoint[1]),
                float(midpoint[2]),
                mes.t,
                0 # TODO
            )
            print("on_triangulated ", midpoint)
            self.on_triangulated(msg)
            
