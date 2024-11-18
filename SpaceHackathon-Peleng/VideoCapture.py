from __future__ import annotations
from typing import Callable

import cv2
import threading


class Capture:
    def __init__(self,
                 left_url: str = 'http://192.168.1.81:8080/video',
                 right_url: str = 'http://192.168.1.82:8080/video',
                 on_left_frame: Callable[[Capture], None] = None,
                 on_right_frame: Callable[[Capture], None] = None):

        cap_left = cv2.VideoCapture(left_url)  # First device
        # cap_right = cv2.VideoCapture(right_url)  # Second device

        tl = threading.Thread(target=self.__capture_func, args=(cap_left, 'left', lambda _: on_left_frame(self)))
        # tr = threading.Thread(target=self.__capture_func, args=(cap_right, 'right', lambda _: on_right_frame(self)))

        tl.start()
        # tr.start()

    def __capture_func(self, cap: cv2.VideoCapture, name: str, callback: Callable[[], None]):
        while True:
            ret, frame = cap.read()
            cv2.imshow(name, frame)
            callback()

            if cv2.waitKey(1) == 27:
                exit(0)


if __name__ == '__main__':
    capture = Capture()
