import time

import cv2
from deepface import DeepFace

filename = "./videos/video_1.mp4"
cam = cv2.VideoCapture(filename)
org_fps = cam.get(cv2.CAP_PROP_FPS)
print(org_fps)
print(int(cam.get(cv2.CAP_PROP_FRAME_COUNT)))
frame_rate = 60
prev = 0

while cam.isOpened():
    time_elapsed = time.time() - prev

    ret, frame = cam.read()
    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        if ret is True:
            # cv2.imshow(filename, frame)
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print(result)
        else:
            break

cam.release()
cv2.destroyAllWindows()
