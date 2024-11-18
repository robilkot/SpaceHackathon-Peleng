import cv2
import threaing
#todo: ебануть параллельно
def process_video_thread(path: str, bounding_boxes: list, window_name: str):
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        timer = cv2.getTickCount()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=6)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame1, f"FPS: {int(fps)}", (75, 50), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_videos(path_rgb: str, path_ir: str) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    bounding_boxes_rgb = []
    bounding_boxes_ir = []

    thread_rgb = threading.Thread(target=process_video_thread, args=(path_rgb, bounding_boxes_rgb, "RGB Video"))
    thread_ir = threading.Thread(target=process_video_thread, args=(path_ir, bounding_boxes_ir, "IR Video"))

    thread_rgb.start()
    thread_ir.start()

    thread_rgb.join()
    thread_ir.join()

    return bounding_boxes_rgb, bounding_boxes_ir
