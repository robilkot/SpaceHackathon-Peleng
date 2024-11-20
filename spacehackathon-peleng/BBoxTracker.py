import math
import threading

import cv2
from Models.DetectionMessage import *
from Coordinator import *


class BBoxTracker:
    def __init__(self, on_tracked: Callable[[DetectionMessage], None]) -> None:
        self.on_tracked = on_tracked
        self.__exiting: bool = False

    def __read_downscaled(self, cap):
        ret, frame = cap.read()

        # return frame  # when NOT downscaling
        if not ret:
            return

        # Get the current frame size
        height, width, _ = frame.shape
        scale_percent = 100
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def __get_contours(self, f1_rgb, f2_rgb):
        rgb_diff = cv2.absdiff(f1_rgb, f2_rgb)
        rgb_gray = cv2.cvtColor(rgb_diff, cv2.COLOR_BGR2GRAY)
        _, rgb_thresh = cv2.threshold(rgb_gray, 45, 255, cv2.THRESH_BINARY)
        rgb_dilated = cv2.dilate(rgb_thresh, None, iterations=7)
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
        def exit2(error: Exception | None = None):
            if DEBUG_TRACKER:
                print(f'cam_id:{cam_id} is exiting {"(CRASH)" if error else ""}')
                if error is not None:
                    print(error)
            self.__exiting = True
            cap_rgb.release()
            cap_ir.release()
            if SHOW_TRACKER:
                cv2.destroyWindow(f"RGB{cam_id}")
                cv2.destroyWindow(f"IR{cam_id}")

        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_ir = cv2.VideoCapture(ir_path)

        f1_rgb = self.__read_downscaled(cap_rgb)
        f1_ir = self.__read_downscaled(cap_ir)

        info: dict[float, ObjectState] = {}

        lost_track: bool = False
        last_frame_ticks = cv2.getTickCount()
        tracking_frame_number = 1  # Increments each TIMESTEP
        track = []
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        cap_ir.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        circle_positions = []
        tracking_frame_number = START_FRAME / FRAME_RATE
        frame = START_FRAME
        while cap_rgb.isOpened() and cap_ir.isOpened():
            rgb_bboxes = []
            ir_bboxes = []

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
            frame += 1
            if frame % FRAME_RATE == 0:
                last_frame_ticks = current_frame_ticks

                timestamp = (tracking_frame_number - 1) * TIMESTEP
                tracking_frame_number += 1

                try:
                    if len(rgb_bboxes) == 1 and len(ir_bboxes) == 0:
                        bbox = rgb_bboxes[0]
                    elif len(rgb_bboxes) == 0 and len(ir_bboxes) == 1:
                        bbox = ir_bboxes[0]
                    else:
                        bbox = next(x for x in self.__find_intersecting_boxes(rgb_bboxes, ir_bboxes))

                    # clear predictions, we now have real data
                    if lost_track:
                        info = {}
                        lost_track = False

                    # get current centers
                    _x = (2 * bbox[0] + bbox[2]) / 2
                    _y = (2 * bbox[1] + bbox[3]) / 2

                    x_pred, y_pred = predict_location(timestamp, info)

                    # print(f"actual: {_x, _y} predict: {x_pred, y_pred}")

                    if x_pred is not None and y_pred is not None:
                        thresh = 100
                        dist = math.sqrt((y_pred - _y) ** 2 + (x_pred - _x) ** 2)
                        # print(f"dist {dist}")
                        if dist < thresh:
                            if DEBUG_TRACKER:
                                print(f"cam_id:{cam_id} real")
                            x_res, y_res = _x, _y
                        else:
                            if DEBUG_TRACKER:
                                print(f"cam_id:{cam_id} prediction")
                            x_res, y_res = x_pred, y_pred
                    else:
                        if DEBUG_TRACKER:
                            print(f"cam_id:{cam_id} real")
                        x_res, y_res = _x, _y

                    state = ObjectState(x_res, y_res, 0, timestamp, None, None, None, None)
                    info[timestamp] = state

                    info = {key: value for key, value in zip(info.keys(), info.values()) if value.t > timestamp - LEFTED_FRAMES * TIMESTEP}
                    # print(f"purged, {len(info)} left")

                    if DEBUG_TRACKER:
                        print(f'cam_id:{cam_id} t:{timestamp} ok')
                    try:
                        self.on_tracked(ObjDetectedMessage(cam_id, timestamp, bbox[0], bbox[1], bbox[2], bbox[3]))
                    except Exception as e:
                        exit2(e)

                except StopIteration:
                    lost_track = True

                    if DEBUG_TRACKER:
                        print(f"cam_id:{cam_id} prediction")
                    x_pred, y_pred = predict_location(timestamp, info)
                    info[timestamp] = ObjectState(x_pred, y_pred, 0, timestamp, None, None, None, None)

                    if DEBUG_TRACKER:
                        print(f'cam_id:{cam_id} t:{timestamp} FAIL')

                    msg = ObjNotDetectedMessage(cam_id, timestamp)
                    try:
                        self.on_tracked(msg)
                    except Exception as e:
                        exit2(e)
                finally:
                    # Debug
                    last_center = info[timestamp]
                    x = last_center.x if last_center.x is not None else 0
                    y = last_center.y if last_center.y is not None else 0
                    track.append((x, y))

                    for o in info.values():
                        complete_object_state(o, info)

            for t in track:
                try:
                    if frame % 5 == 0:
                        circle_positions.append((int(t[0]), int(t[1])))

                    for (x, y) in circle_positions:
                        cv2.circle(display1, (x, y), radius=5, color=(0, 0, 255), thickness=2)
                        cv2.circle(display2, (x, y), radius=5, color=(0, 0, 255), thickness=2)
                except:
                    # Хз что случилось но бывает
                    pass

            if SHOW_TRACKER:
                cv2.imshow(f"RGB{cam_id}", display1)
                cv2.imshow(f"IR{cam_id}", display2)

            # cv2.waitKey(0)  # For stepping through each frame
            if self.__exiting or cv2.waitKey(25) & 0xFF == ord('q'):
                exit2()

    def start(self):
        for i in range(CAMERAS_COUNT):
            path_rgb = PATH_RGB.format(EXAMPLE, EXAMPLE, i+1)
            path_ir = PATH_IR.format(EXAMPLE, EXAMPLE, i+1)
            t = threading.Thread(target=self.__process_video, args=(path_rgb, path_ir, i + 1))
            t.start()

        while True:
            if input("q to exit") == 'q':
                self.__exiting = True
                break


if __name__ == '__main__':
    tracker = BBoxTracker(lambda msg: print(msg.t))
    tracker.start()

