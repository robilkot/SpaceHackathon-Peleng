import cv2
def process_video(path: str) -> list[tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(path)
    bounding_boxes = []
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        timer = cv2.getTickCount()
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=6)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255,0), 2)
            print(f"Bounding Box for RGB: x={x}, y={y}, w={w}, h={h}")
            bounding_boxes.append((x, y, w, h))

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame1, f"FPS: {int(fps)}", (75, 50), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        cv2.imshow("Test", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return bounding_boxes


if __name__ == "__main__":
    bbox_rgb = process_video('E:\\SpaceHackathon-Peleng\\data\\videoset1\\Seq1_camera1.mov')
    bbox_ir = process_video('E:\\SpaceHackathon-Peleng\\data\\videoset1\\Seq1_camera1T.mov')
