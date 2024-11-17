import cv2
import threading

capture_left = cv2.VideoCapture('http://192.168.1.81:8080/video')  # First device
# capture_right = cv2.VideoCapture('http://192.168.1.82:8080/video')  # Second device


def capture_func(cap: cv2.VideoCapture, name: str):
    while True:
        ret, frame = cap.read()
        cv2.imshow(name, frame)

        if cv2.waitKey(1) == 27:
            exit(0)


thread_left = threading.Thread(target=capture_func, args=(capture_left, 'left'))
# thread_right = threading.Thread(target=capture_func, args=(capture_right, 'right'))

thread_left.start()
# thread_right.start()

print("Started capturing")

