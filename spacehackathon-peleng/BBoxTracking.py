import threading
import cv2

PATH_RGB = 'data/videoset1/Seq1_camera1.mov'
PATH_IR = 'data/videoset1/Seq1_camera1T.mov'

def process_video(path: str):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    bounding_boxes = []
    while cap.isOpened():
        timer = cv2.getTickCount()
        ret, frame2 = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=6)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame1, f"FPS: {int(fps)}", (75, 50), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.imshow("test", frame1)

        frame1 = frame2

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
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
    rgb_boxes = process_video(PATH_RGB)
    ir_boxes = process_video(PATH_IR)
    intersecting_boxes = find_intersecting_boxes(rgb_boxes, ir_boxes)

    for box_pair in intersecting_boxes:
        print(f"Intersecting Boxes: {box_pair}")