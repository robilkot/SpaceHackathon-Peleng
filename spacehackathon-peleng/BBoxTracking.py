import threading
import cv2

PATH_RGB = 'data/videoset1/Seq1_camera1.mov'
PATH_IR = 'data/videoset1/Seq1_camera1T.mov'
TIMESTEP: float = 0.5


def read_downscaled(cap):
    ret, frame = cap.read()
    if not ret:
        return

    # Get the current frame size
    height, width, _ = frame.shape
    scale_percent = 50
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def get_contours(f1_rgb, f2_rgb):
    rgb_diff = cv2.absdiff(f1_rgb, f2_rgb)
    rgb_gray = cv2.cvtColor(rgb_diff, cv2.COLOR_BGR2GRAY)
    _, rgb_thresh = cv2.threshold(rgb_gray, 45, 255, cv2.THRESH_BINARY)
    rgb_dilated = cv2.dilate(rgb_thresh, None, iterations=6)
    rgb_contours, _ = cv2.findContours(rgb_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return rgb_contours


def process_video(rgb_path: str, ir_path: str):
    cap_rgb = cv2.VideoCapture(rgb_path)
    cap_ir = cv2.VideoCapture(ir_path)

    f1_rgb = read_downscaled(cap_rgb)
    f1_ir = read_downscaled(cap_ir)
    # ret, f1_rgb = cap_rgb.read()
    # ret, f1_ir = cap_rgb.read()

    last_frame_ticks = cv2.getTickCount()
    bounding_boxes = []
    while cap_rgb.isOpened() and cap_ir.isOpened():

        # ret, f2_rgb = cap_rgb.read()
        # ret, f2_ir = cap_ir.read()
        f2_rgb = read_downscaled(cap_rgb)
        f2_ir = read_downscaled(cap_ir)

        if f2_rgb is None or f2_ir is None:
            break

        rgb_contours = get_contours(f1_rgb, f2_rgb)
        ir_contours = get_contours(f1_ir, f2_ir)

        f1_rgb = f2_rgb
        f1_ir = f2_ir

        display1 = f2_rgb.copy()
        display2 = f2_ir.copy()

        current_frame_ticks = cv2.getTickCount()
        passed = (current_frame_ticks - last_frame_ticks) / cv2.getTickFrequency()

        if passed > TIMESTEP:
            last_frame_ticks = current_frame_ticks
            for contour in rgb_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                cv2.rectangle(display1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            for contour in ir_contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))
                cv2.rectangle(display2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("RGB", display1)
        cv2.imshow("IR", display2)

        # cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap_rgb.release()
    cv2.destroyAllWindows()
    return bounding_boxes

#shitcoding of intersection
def check_box_intersection(rgb_box: tuple[int, int, int, int], ir_box: tuple[int, int, int, int]) -> bool:
    x_rgb, y_rgb, w_rgb, h_rgb = rgb_box
    x_ir, y_ir, w_ir, h_ir = ir_box
    if x_rgb + w_rgb < x_ir or x_ir + w_ir < x_rgb or y_rgb + h_rgb < y_ir or y_ir + h_ir < y_rgb:
        return False
    return True

def find_intersecting_boxes(rgb_boxes: list[tuple[int, int, int, int]], ir_boxes: list[tuple[int, int, int, int]]) -> \
list[tuple[int, int, int, int]]:
    intersecting_ir_boxes = []
    for rgb_box in rgb_boxes:
        for ir_box in ir_boxes:
            if check_box_intersection(rgb_box, ir_box):
                intersecting_ir_boxes.append(ir_box)
    return intersecting_ir_boxes


if __name__ == "__main__":
    rgb_boxes = process_video(PATH_RGB, PATH_IR)
    # ir_boxes = process_video(PATH_IR)
    # intersecting_boxes = find_intersecting_boxes(rgb_boxes, ir_boxes)

    # for box_pair in intersecting_boxes:
    #     print(f"Intersecting Boxes: {box_pair}")
