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


def midpoint_triangulate(A, B):
    n = 3                                         # No. of cameras

    I = np.eye(3)
    sigma2 = np.zeros((3,1))

    for i in range(n):
        a = A[:,i,None]
        b = B[:,i,None]

        sigma2 = sigma2 + b.dot(b.transpose().dot(a))

    C = (n * I) - B.dot(B.transpose())
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

        self.A = np.asarray([[x.x, x.y, x.z] for x in cams.values()])
        self.cams: dict[int, MatCamera] = dict.fromkeys(cams.keys())
        self._cams = cams
        for id, cam in cams.items():
            R = np.asarray(
                [
                    [np.cos(cam.a), -np.sin(cam.a), 0],
                    [np.sin(cam.a), np.cos(cam.a), 0],
                    [0, 0, 1],
                ]
            ).transpose()
            T = np.asarray([cam.x, cam.y, cam.z]).reshape((-1, 1))
            T = -R.dot(T)
            # P = np.hstack((R, T.reshape((-1, 1))))
            K = np.asarray(
                [
                    [cam.focal_length, 0, cam.matrix_w / 2  ],
                    [0, cam.focal_length, cam.matrix_h / 2],
                    [0, 0, 1]
                ]
            )
            P = K.dot(np.hstack((R, T.reshape((-1, 1)))))
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
            print(mes.x + mes.w / 2, cam.resolution[0], cam.width)
            scene[mes.cam_id] = np.asarray(
                (
                    (mes.x + mes.w / 2) / cam.resolution[0] - 0.5,
                    0.5 - (mes.y + mes.h / 2) / cam.resolution[1]
                )
            )
        self.scenes[mes.t] = scene

        print(f"Triangulator: DetectionMessage instance has received. {mes.cam_id = }: ", scene[mes.cam_id])
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

            B = []
            for id, cam in self._cams.items():
                print(scene[id])
                print(self._cams[id].focal_length)
                b = np.asarray(
                    [
                        self._cams[id].focal_length,
                        scene[id][0],
                        scene[id][1],
                    ]
                )
                b = b / np.linalg.norm(b)
                b = -b.dot(self.cams[id].R)

                B.append(b)

            B = np.asarray(B)
            print(self.A, B)
            midpoint = midpoint_triangulate(self.A, B)

            msg = CoordinatesTriangulatedMessage(
                float(midpoint[0]),
                float(midpoint[1]),
                float(midpoint[2]),
                mes.t,
                0 # TODO
            )
            print("on_triangulated ", midpoint)
            self.on_triangulated(msg)
            
