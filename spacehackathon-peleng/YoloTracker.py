import math
import threading

import cv2
from ultralytics import YOLO
from Models.DetectionMessage import *
from Coordinator import *


class YoloTracker:
    def __init__(self, on_tracked: Callable[[DetectionMessage], None]) -> None:
        self.on_tracked = on_tracked
        self.__exiting: bool = False
        self.yolo = YOLO('yolo11n.pt')

    def __read_downscaled(self, cap):
        ret, frame = cap.read()

        # return frame  # when NOT downscaling
        if not ret:
            return

        # Get the current frame size
        height, width, _ = frame.shape
        scale_percent = 50
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def getColours(self, cls_num):
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
                 (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)

    def __process_video(self, rgb_path: str, ir_path: str, cam_id: int):
        def exit2(error: Exception | None = None):
            if DEBUG_TRACKER:
                print(f'cam_id:{cam_id} is exiting {"(CRASH)" if error else ""}')
                if error is not None:
                    print(error)
            self.__exiting = True
            cap_rgb.release()
            cv2.destroyWindow(f"RGB{cam_id}")
            cv2.destroyWindow(f"IR{cam_id}")

        cap_rgb = cv2.VideoCapture(rgb_path)
        cap_ir = cv2.VideoCapture(ir_path)

        info: dict[float, ObjectState] = {}

        lost_track: bool = False
        last_frame_ticks = cv2.getTickCount()
        tracking_frame_number = 1  # Increments each TIMESTEP
        track = []
        while cap_rgb.isOpened() and cap_ir.isOpened():
            rgb_bboxes = []
            ir_bboxes = []

            f2_rgb = self.__read_downscaled(cap_rgb)
            f2_ir = self.__read_downscaled(cap_ir)

            display1 = f2_rgb.copy()
            display2 = f2_ir.copy()

            if f2_rgb is None or f2_ir is None:
                break

            results_rgb = self.yolo.track(f2_rgb, stream=True)

            for result in results_rgb:
                # get the classes names
                classes_names = result.names

                # iterate over each box
                for box in result.boxes:
                    # check if confidence is greater than 40 percent
                    if box.conf[0] > 0.4:
                        # get coordinates
                        [x1, y1, x2, y2] = box.xyxy[0]
                        # convert to int
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # get the class
                        cls = int(box.cls[0])

                        # get the class name
                        class_name = classes_names[cls]

                        # get the respective colour
                        colour = self.getColours(cls)

                        # draw the rectangle
                        cv2.rectangle(f2_rgb, (x1, y1), (x2, y2), colour, 2)

                        # put the class name and confidence on the image
                        cv2.putText(f2_rgb, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

            current_frame_ticks = cv2.getTickCount()
            passed = (current_frame_ticks - last_frame_ticks) / cv2.getTickFrequency()

            # Time to send event
            if passed > TIMESTEP:
                last_frame_ticks = current_frame_ticks

                timestamp = (tracking_frame_number - 1) * TIMESTEP
                tracking_frame_number += 1

                if DEBUG_TRACKER:
                    print(f'cam_id:{cam_id} t:{timestamp} ok')
                try:
                    pass
                    # self.on_tracked(ObjDetectedMessage(cam_id, timestamp, bbox[0], bbox[1], bbox[2], bbox[3]))
                except Exception as e:
                    exit2(e)

            cv2.imshow(f"RGB{cam_id}", display1)
            # cv2.imshow(f"IR{cam_id}", display2)

            # cv2.waitKey(0)  # For stepping through each frame
            if self.__exiting or cv2.waitKey(25) & 0xFF == ord('q'):
                exit2()


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
