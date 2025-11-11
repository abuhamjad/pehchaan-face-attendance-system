# save as find_cam.py and run it
import cv2
for i in range(4):
    cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, _ = cam.read()
    print(f"Index {i}: opened={cam.isOpened()}, frame_ok={ok}")
    cam.release()
