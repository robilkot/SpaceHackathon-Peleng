from typing import Callable, Iterable
from dataclasses import dataclass


import numpy as np
import cv2
from Constants import *

from Models.CoordinatesTriangulatedMessage import CoordinatesTriangulatedMessage
from Models.DetectionMessage import DetectionMessage, ObjNotDetectedMessage, ObjDetectedMessage
from Models.Camera import Camera


@dataclass()
class MatCamera:
    P: np.matrix
    T: np.matrix


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

            # https://en.wikipedia.org/wiki/Camera_resectioning
            fx = cam.focal_length / cam.matrix_w * cam.res_w
            fy = cam.focal_length / cam.matrix_h * cam.res_h
            cx = cam.res_w / 2
            cy = cam.res_h / 2

            K = np.asarray([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])

            theta = np.radians(cam.azimut)  # у тебя было не в радианах, но это не помогло
            R = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
            ])

            T = np.array([cam.x, cam.y, cam.z])

            RT = np.hstack((R, T.reshape(-1, 1)))
            P = K @ RT
            self.cams[id] = MatCamera(P, T)
            # print(f"cam{id}", K, R, T, P, sep='\n')

        self.on_triangulated = on_triangulated
        self.scenes = {}
    
    def transform(self, mes: DetectionMessage):
        scene = self.scenes.get(mes.t, Scene(self.cams.keys()))
        cam = self.cams[mes.cam_id]
        if isinstance(mes, ObjNotDetectedMessage):
            scene[mes.cam_id] = 0
            print("Triangulator: ObjNotDetectedMessage instance has received")
        elif isinstance(mes, ObjDetectedMessage):
            scene[mes.cam_id] = np.array((
                    (mes.x + mes.w / 2),
                    (mes.y + mes.h / 2),
            ))
        else:
            print("The fuck???")

        self.scenes[mes.t] = scene

        print(f"Triangulator: DetectionMessage instance has received. {mes.cam_id = }: ", scene[mes.cam_id])
        print(f"Scene {mes.t} {scene.is_full()}")
        print('\n'.join(f"{i}: {x is not None}" for i, x in scene))

        # todo ДЕТЕКТИТЬ ПО ДВУМ КАМЕРАМ ЕСЛИ ТРЕТЬЕЙ НЕТ это надо сделать сейчас и отлаживать конкретно на этом примере окда?
        # Глянь тут на камерах 1 и 3 где-то справа сверху шар норм трекается
        if scene.is_full():
            counter = 0
            for _, point in scene:
                if point is 0:
                    counter += 1

            if counter >= 2:
                print("Triangulator: CoordinatesTriangulatedMessage with none was sent")
                msg = CoordinatesTriangulatedMessage(
                    None,
                    None,
                    None,
                    mes.t,
                    None
                )
                self.on_triangulated(msg)
                return

            if counter == 1:
                point_4d = cv2.triangulatePoints(self.cams[1].P, self.cams[3].P, scene[1], scene[3])

            print("Triangulator: start computing")
            print("Triangulator: ", self.cams[1].P, self.cams[3].P, scene[1], scene[3], sep='\n')

            point_4d = cv2.triangulatePoints(self.cams[1].P, self.cams[3].P, scene[1], scene[3])

            point_3d = point_4d[:3] / point_4d[3]
            print(f"t: {mes.t} {point_3d = }")

            msg = CoordinatesTriangulatedMessage(
                float(point_3d[0]),
                float(point_3d[1]),
                float(point_3d[2]),
                mes.t,
                0 # TODO
            )
            self.on_triangulated(msg)
            
