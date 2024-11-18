import os
from typing import Callable

import cv2
import numpy as np


def get_region(
    frame: cv2.typing.MatLike, 
    x, 
    y
):
    radius = 20
    x1 = x - radius
    y1 = y - radius
    x2 = x + radius
    y2 = y + radius
    x1 = 0 if x1 < 0 else frame.shape[1] if x1 > frame.shape[1] else x1
    y1 = 0 if y1 < 0 else frame.shape[0] if y1 > frame.shape[0] else y1
    x2 = 0 if x2 < 0 else frame.shape[1] if x2 > frame.shape[1] else x2
    y2 = 0 if y2 < 0 else frame.shape[0] if y2 > frame.shape[0] else y2
    return frame[y1:y2,x1:x2]


def get_gray_region(
    frame: cv2.typing.MatLike, 
    point: tuple[int, int]
):
    region = get_region(frame, *point)
    return cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)


def get_error(
    region1: cv2.typing.MatLike, 
    region2: cv2.typing.MatLike
):
    d = region1 - region2
    err = (d ** 2).sum((0, 1))
    return err
    

def track_point(
    filename: str | bytes | os.PathLike,
    startpoint: np.ndarray, 
    on_frame: Callable[[cv2.typing.MatLike, np.ndarray], None]
):
    cap = cv2.VideoCapture(filename)

    point = startpoint
    
    if not cap.isOpened():
        raise IOError("Error opening video stream or file")
    
    region = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            if region is None:   
                region = get_gray_region(frame, point)
            else:
                dx = 0
                dy = 0
                new_region = get_gray_region(frame, point)
                error = get_error(new_region, region)
                for i in range(-30, 10, 1):
                    for j in range(-30, 10, 1):
                        new_region = get_gray_region(frame, point - (i, j))
                        new_error = get_error(new_region, region)
                        if new_error < error:
                            error = new_error
                            dx, dy = i, j
                point += (dx, dy)
                region = new_region
            if cv2.waitKey(25) and 0xFF == ord('q'):
                break
            
            on_frame(frame, point)
            
        else:
            break
        
    cap.release()
    
    
    
def on_frame(frame: cv2.typing.MatLike, point: np.ndarray):
    frame = cv2.ellipse(frame, point, (5, 5), 0, 0, 360, 255, -1)
    cv2.imshow('Frame', frame)
    
    
if __name__ == '__main__':
    track_point('video/left.mp4', np.asarray((1150, 400)), on_frame)
        
    cv2.destroyAllWindows()
    