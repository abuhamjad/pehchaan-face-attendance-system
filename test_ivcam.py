import cv2

# Try changing the index number based on output from find_camera_index.py
CAM_INDEX = 1  # usually 1 or 2 for iVCam

cam = cv2.VideoCapture(CAM_INDEX)

if not cam.isOpened():
    print("[ERROR] Could not open iVCam. Try a different index.")
    exit()

print("[INFO] iVCam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Frame not received.")
        break

    cv2.imshow("iVCam Feed (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
