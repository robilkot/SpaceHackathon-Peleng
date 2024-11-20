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
            fx = cam.focal_length / cam.matrix_w * cam.res_w  # взято с гпт В общем случае нет потому что есть неквадратные пиксели Надеюсь у нас не так)
            fy = cam.focal_length / cam.matrix_h * cam.res_h  # чисто теоритически cam.matrix_h / cam.res_h = cam.matrix_w  cam.res_w
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
            self.cams[id] = MatCamera(P, T) # А T нам вообще нужно? а хуй его знает, мб забыл удалить
            # print(f"cam{id}", K, R, T, P, sep='\n')

        self.on_triangulated = on_triangulated
        self.scenes = {}
    # блять я ща возьму вручную координаты шара с видео и будем ебашить
    def transform(self, mes: DetectionMessage):
        scene = self.scenes.get(mes.t, Scene(self.cams.keys()))
        if isinstance(mes, ObjNotDetectedMessage):
            scene[mes.cam_id] = 0
            print("Triangulator: ObjNotDetectedMessage instance has received")
        elif isinstance(mes, ObjDetectedMessage):
            coords = np.array((
                (mes.x + mes.w / 2),
                (mes.y + mes.h / 2),
            ))
            scene[mes.cam_id] = coords
        else:
            print("The fuck???")

        self.scenes[mes.t] = scene

        print(f"Triangulator: DetectionMessage instance has received. {mes.cam_id = }: ", scene[mes.cam_id])
        print(f"Scene {mes.t} {scene.is_full()}")
        print('\n'.join(f"{i}: {x is not None}" for i, x in scene))

        # todo ДЕТЕКТИТЬ ПО ДВУМ КАМЕРАМ ЕСЛИ ТРЕТЬЕЙ НЕТ это надо сделать сейчас и отлаживать конкретно на этом примере окда? check
        # Глянь тут на камерах 1 и 3 где-то справа сверху шар норм трекается
        # 
        full = scene.is_full()
        if full:
            # true_cams = {i: cam for i, cam in self.cams.items() if scene[i] is not 0}
            # print(f"Triangulator: {' '.join([*true_cams.keys()])}")
            #
            # if len(true_cams) < 2:
            #     print("Triangulator: CoordinatesTriangulatedMessage with none was sent")
            #     msg = CoordinatesTriangulatedMessage(
            #         None,
            #         None,
            #         None,
            #         mes.t,
            #         None
            #     )
            #     self.on_triangulated(msg)
            #     return

            # print("Triangulator: start computing")
            # point_4d: np.ndarray
            # if len(true_cams) == 2:
            #     i, j = true_cams.keys()
            #     print("Triangulator: ", *[cam.P for cam in true_cams.values()], *[scene[i] for i in true_cams], sep='\n')
            #     point_4d = cv2.triangulatePoints(self.cams[i].P, self.cams[j].P, scene[i], scene[j])
            # else:
            #     print("Triangulator: ", self.cams[1].P, self.cams[3].P, scene[1], scene[3], sep='\n')
            #     point_4d = cv2.triangulatePoints(self.cams[1].P, self.cams[3].P, scene[1], scene[3])


            if scene[1] is 0 or scene[3] is 0:
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

            print("Triangulator: ", self.cams[1].P, self.cams[3].P, scene[1], scene[3], sep='\n')
            point_4d = cv2.triangulatePoints(self.cams[1].P, self.cams[3].P, scene[1], scene[3]) # в опенсв все от левого верзнего угла

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
            

def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)


if __name__ == "__main__":
    # t = 6.0
    # scene1 = (78, 71)
    # scene3 = (844, 91)
    #
    # focal = 35 / 1000
    # matrix_w = 23.760 / 1000
    # matrix_h = 13.365 / 1000
    # cam1_ = Camera(focal, 270, 850, 33, -73, matrix_w, matrix_h, 1920, 1072)
    # cam3 = Camera(focal, 1030, 550, 45, -150,  matrix_w, matrix_h, 1920, 1072)

    scene1 = (168, 30)
    scene3 = (362, 273)

    focal = 50
    matrix_w = 36
    matrix_h = 24
    w = 960
    h = 540

    x1 = 1
    y1 = -1
    z1 = 0.5  # Тут оси блендевоске, z верх
    x2 = 1
    y2 = 1
    z2 = 0.7
    cam1_ = Camera(focal, x1, z1, y1, 45, matrix_w, matrix_h, w, h)
    cam3 = Camera(focal, x2, z2, y2, 135,  matrix_w, matrix_h, w, h)

    Ps = {}
    RTs = {}
    Ks = {}
    for id, cam1 in enumerate([cam1_, cam3]):
        fx = cam1.focal_length / cam1.matrix_w * cam1.res_w  # взято с гпт В общем случае нет потому что есть неквадратные пиксели Надеюсь у нас не так)
        fy = cam1.focal_length / cam1.matrix_h * cam1.res_h  # чисто теоритически cam.matrix_h / cam.res_h = cam.matrix_w  cam.res_w
        cx = cam1.res_w / 2
        cy = cam1.res_h / 2

        K = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        Ks[id] = K
        theta = np.radians(cam1.azimut)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]).T

        R = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ]
        ).T

        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), -np.cos(theta)]
            ]
        ).T

        T = np.array([cam1.x, cam1.y, cam1.z]) #  тут у верх а не z - я это учел

        RT = np.hstack((R, np.dot(-R, T.reshape(-1, 1))))
        P1 = K @ RT

        RTs[id] = RT
        Ps[id] = P1 # Это матрица должна переводить 3д в 2д/ ДАВАЙ RT мб попробуем, она же за перемщеение и поворот вроде

    P1 = Ps[0]
    P3 = Ps[1]

    fx = cam1.focal_length / cam1.matrix_w * cam1.res_w  # взято с гпт В общем случае нет потому что есть неквадратные пиксели Надеюсь у нас не так)
    fy = cam1.focal_length / cam1.matrix_h * cam1.res_h  # чисто теоритически cam.matrix_h / cam.res_h = cam.matrix_w  cam.res_w
    cx = cam1.res_w / 2
    cy = cam1.res_h / 2

    scene1 = (np.asarray(scene1, dtype=float) - [cx, cy]) / [fx, fy] #/ [w, h]
    scene3 = (np.asarray(scene3, dtype=float) - [cx, cy]) / [fx, fy] #/ [w, h]

    # points1 is a (N, 1, 2) float32 from cornerSubPix
    # points1u = cv2.UMat()
    # points2u = cv2.UMat()
    # cv2.undistortPoints(scene1, Ks[0], None, points1u, None, P1)
    # cv2.undistortPoints(scene3, Ks[1], None, points2u, None, P3)

    points4d = cv2.triangulatePoints(P1, P3, scene1, scene3)
    points3d = (points4d[:3, :] / points4d[3, :]).T

    # p = triangulate_points(P1, P3, scene1, scene3)
    # p /= p[3]
    # print('Projected point from openCV:', p)

    print(f'{scene1 = }, {scene3 = }')

    point_4d = cv2.triangulatePoints(P1, P3, scene1, scene3)
    # 0, 0.7, -0.2 это ответ для данных текущих

    point_3d = point_4d[:3] / point_4d[3]
    print(f"{point_3d = }")
    print(P1.dot([0, 0.7, -0.2, 1]) / P1.dot([0, 0.7, -0.2, 1])[2], scene1) # tut sceilit nado

    # проверяй, надо 289.99	-175.00	171.41

