import cv2

camera = True
if camera:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('video/left.mp4')

tracker = cv2.legacy.TrackerMedianFlow.create()

success, img = cap.read()
if not success:
    print("log error tracking next frame")
    cap.release()
    exit()

bbox = cv2.selectROI('Tracking', img, False)
tracker.init(img, bbox)


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    if not success:
        print("log error tracking next frame")
        break

    success, bbox = tracker.update(img)

    if success:
        print(f"Bounding Box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, f"FPS: {int(fps)}", (75, 50), cv2.FONT_ITALIC, 0.7, (0, 0, 255), 2)


    cv2.imshow("Tracking", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
