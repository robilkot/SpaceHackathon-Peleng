import threading

import cv2
from typing import Callable
from Models.DetectionMessage import *
from Constants import *

PATH_RGB = 'data/videoset{}/Seq{}_camera{}.mov'
PATH_IR = 'data/videoset{}/Seq{}_camera{}T.mov'

class BBoxTracker:
    def __init__(self, on_tracked: Callable[[DetectionMessage], None]) -> None:
        self.on_tracked = on_tracked

    TIMESTEP: float = 0.5

    def __read_downscaled(self, cap):
        ret, frame = cap.read()

        # return frame  # when NOT downscaling
        if not ret:
            return

        # Get the current frame size
        height, width, _ = frame.shape
        scale_percent = 25
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def __get_contours(self, f1_rgb, f2_rgb):
        rgb_diff = cv2.absdiff(f1_rgb, f2_rgb)
        rgb_gray = cv2.cvtColor(rgb_diff, cv2.COLOR_BGR2GRAY)
        _, rgb_thresh = cv2.threshold(rgb_gray, 45, 255, cv2.THRESH_BINARY)
        rgb_dilated = cv2.dilate(rgb_thresh, None, iterations=6)
        rgb_contours, _ = cv2.findContours(rgb_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return rgb_contours

    def __check_box_intersection(self, rgb_box: tuple[int, int, int, int], ir_box: tuple[int, int, int, int]) -> bool:
        x_rgb, y_rgb, w_rgb, h_rgb = rgb_box
        x_ir, y_ir, w_ir, h_ir = ir_box
        if x_rgb + w_rgb < x_ir or x_ir + w_ir < x_rgb or y_rgb + h_rgb < y_ir or y_ir + h_ir < y_rgb:
            return False
        return True

    def __find_intersecting_boxes(self, rgb_boxes: list[tuple[int, int, int, int]],
                                  ir_boxes: list[tuple[int, int, int, int]]) -> \
            list[tuple[int, int, int, int]]:
        intersecting_ir_boxes = []
        for rgb_box in rgb_boxes:
            for ir_box in ir_boxes:
                if self.__check_box_intersection(rgb_box, ir_box):
                    intersecting_ir_boxes.append(ir_box)
        return intersecting_ir_boxes

    def __process_video(self, rgb_path: str, ir_path: str, cam_id: int):
        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_ir = cv2.VideoCapture(ir_path)

        f1_rgb = self.__read_downscaled(cap_rgb)
        f1_ir = self.__read_downscaled(cap_ir)

        rgb_bboxes = []
        ir_bboxes = []

        last_frame_ticks = cv2.getTickCount()
        tracking_frame_number = 1  # Increments each TIMESTEP
        while cap_rgb.isOpened() and cap_ir.isOpened():

            f2_rgb = self.__read_downscaled(cap_rgb)
            f2_ir = self.__read_downscaled(cap_ir)

            if f2_rgb is None or f2_ir is None:
                break

            rgb_contours = self.__get_contours(f1_rgb, f2_rgb)
            ir_contours = self.__get_contours(f1_ir, f2_ir)

            f1_rgb = f2_rgb
            f1_ir = f2_ir

            display1 = f2_rgb.copy()
            display2 = f2_ir.copy()

            current_frame_ticks = cv2.getTickCount()
            passed = (current_frame_ticks - last_frame_ticks) / cv2.getTickFrequency()

            for contour in rgb_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                rgb_bboxes.append((x, y, w, h))
                cv2.rectangle(display1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for contour in ir_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                ir_bboxes.append((x, y, w, h))
                cv2.rectangle(display2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Time to send event
            if passed > self.TIMESTEP:
                last_frame_ticks = current_frame_ticks

                timestamp = (tracking_frame_number - 1) * self.TIMESTEP
                tracking_frame_number += 1
                intersecting_boxes = self.__find_intersecting_boxes(rgb_bboxes, ir_bboxes)

                try:
                    bbox = next(x for x in intersecting_boxes if x in ir_bboxes)
                    msg = ObjDetectedMessage(cam_id, timestamp, bbox[0], bbox[1], bbox[2], bbox[3])
                except StopIteration:
                    msg = ObjNotDetectedMessage(cam_id, timestamp)

                try:
                    print(f'cam_id:{cam_id} t:{timestamp} detect {"FAIL" if msg is ObjNotDetectedMessage else ""}')
                    self.on_tracked(msg)
                except:
                    print("EGOR, YOUR STUFF CRASHED")

            cv2.imshow(f"RGB{cam_id}", display1)
            cv2.imshow(f"IR{cam_id}", display2)

            # cv2.waitKey(0)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap_rgb.release()
        cv2.destroyAllWindows()

    def start(self):
        for i in range(CAMERAS_COUNT):
            path_rgb = PATH_RGB.format(EXAMPLE, EXAMPLE, i+1)
            path_ir = PATH_IR.format(EXAMPLE, EXAMPLE, i+1)
            t = threading.Thread(target=self.__process_video, args=(path_rgb, path_ir, i + 1))
            t.start()

        # self.__process_video(PATH_RGB, PATH_IR, 1)


if __name__ == '__main__':
    tracker = BBoxTracker(lambda msg: print(msg.t))
    tracker.start()
